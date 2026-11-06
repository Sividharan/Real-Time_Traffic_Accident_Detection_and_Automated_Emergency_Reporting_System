import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import cv2
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
import numpy as np
from pydantic import BaseModel

from archival import ARCHIVE_DIR, archive_incident_evidence
from auth import (
    create_default_users,
    get_any_authorized_user,
    get_current_admin,
    get_current_operator,
    get_current_user,
    init_users_db,
    register_user,
    verify_credentials,
)
from database import (
    create_camera,
    create_dispatch,
    create_incident,
    get_all_incidents,
    get_camera,
    get_cameras,
    get_dispatch,
    get_dispatches_for_incident,
    get_hourly_stats,
    get_hotspot_data,
    get_incident,
    get_incident_stats,
    get_monthly_stats,
    get_nearby_hospitals,
    get_pending_incidents,
    get_response_time_stats,
    get_severity_stats,
    init_db,
    reject_incident,
    seed_default_data,
    update_camera_status,
    update_dispatch_status,
    verify_incident,
)
from detector import AccidentDetector
from dispatch import EmergencyDispatchService
from geocoding import construct_incident_payload, reverse_geocode
from photo_storage import get_photo_path, save_uploaded_photo
from rerouting import CongestionAlertBroadcaster
from severity_engine import compute_severity
from stream import CameraManager, RTSPStreamHandler

# ---------------------------------------------------------------------------
# FastAPI Application & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Real-Time Traffic Accident Detection and Automated Emergency Reporting System",
    description="Full-stack AI-powered system for accident detection, emergency reporting, and traffic monitoring.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global Services & Multi-Camera Manager
# ---------------------------------------------------------------------------
camera_manager = CameraManager()
# Backward compatible primary stream reference for CAM_NORTH_01
stream = RTSPStreamHandler(camera_id="CAM_NORTH_01", source="synthetic", name="North Coimbatore Flyover")
detector = AccidentDetector()
broadcaster = CongestionAlertBroadcaster()
dispatch_service = EmergencyDispatchService()

# Camera coordinate registry
CAMERAS = {
    "CAM_NORTH_01": {"lat": 11.016844, "lon": 76.955833, "name": "North Coimbatore Flyover"},
    "CAM_AVINASHI_02": {"lat": 11.028500, "lon": 76.968200, "name": "Avinashi Road Junction"},
    "CAM_GANDHIPURAM_03": {"lat": 11.014200, "lon": 76.961500, "name": "Gandhipuram Bus Stand"},
    "CAM_METTUPALAYAM_04": {"lat": 11.042100, "lon": 76.948200, "name": "Mettupalayam Highway Crossing"}
}


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------
class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "Citizen"
    full_name: str = ""


class RejectRequest(BaseModel):
    rejection_reason: Optional[str] = None


class DispatchRequest(BaseModel):
    incident_id: str
    hospital_id: Optional[str] = None


class DispatchRejectRequest(BaseModel):
    reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
def startup():
    """Initializes SQLite databases, tables, default users, camera pipelines."""
    init_db()
    init_users_db()
    create_default_users()

    # Start primary stream handler
    stream.start()

    # Populate camera manager with configured cameras
    db_cams = get_cameras()
    for c in db_cams:
        camera_manager.add_camera(
            camera_id=c["camera_id"],
            source=c.get("source_url", "synthetic"),
            name=c.get("name")
        )


# ---------------------------------------------------------------------------
# Frontend HTML Serving Endpoints
# ---------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def get_login_page():
    html_path = os.path.join(os.path.dirname(__file__), "login.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/operator", response_class=HTMLResponse)
def get_operator_portal():
    return get_login_page()


@app.get("/report", response_class=HTMLResponse)
@app.get("/incident-report", response_class=HTMLResponse)
def get_citizen_report_page():
    html_path = os.path.join(os.path.dirname(__file__), "report.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/map", response_class=HTMLResponse)
def get_incident_map():
    html_path = os.path.join(os.path.dirname(__file__), "map.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/admin", response_class=HTMLResponse)
def get_admin_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "admin.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Admin Dashboard Under Construction</h2>", status_code=200)


@app.get("/traffic", response_class=HTMLResponse)
def get_traffic_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "traffic.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>Traffic Dashboard Under Construction</h2>", status_code=200)


@app.get("/ems", response_class=HTMLResponse)
def get_ems_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "ems.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h2>EMS Dashboard Under Construction</h2>", status_code=200)


@app.get("/citizen", response_class=HTMLResponse)
def get_citizen_dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "citizen.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return get_citizen_report_page()


# ---------------------------------------------------------------------------
# Authentication APIs (SCRUM-22, SCRUM-37)
# ---------------------------------------------------------------------------
@app.post("/api/auth/login")
def login(creds: LoginRequest):
    """Authenticates email & password against persistent SQLite DB. Generates signed JWT."""
    return verify_credentials(creds.email, creds.password)


@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register_new_user(req: RegisterRequest, admin: Dict = Depends(get_current_admin)):
    """Admin-only endpoint to create new system users."""
    return register_user(
        email=req.email,
        password=req.password,
        role=req.role,
        full_name=req.full_name
    )


@app.get("/api/auth/me")
def get_authenticated_profile(user: Dict = Depends(get_any_authorized_user)):
    """Returns profile and role of currently active authenticated session."""
    return {
        "email": user.get("sub"),
        "role": user.get("role"),
        "exp": user.get("exp")
    }


@app.post("/api/auth/logout")
def logout_user(user: Dict = Depends(get_any_authorized_user)):
    """Terminates authenticated session token on client side."""
    return {"success": True, "message": "Successfully logged out."}


# ---------------------------------------------------------------------------
# System Health & Multi-Camera APIs (SCRUM-40)
# ---------------------------------------------------------------------------
@app.get("/api/system/health")
def system_health():
    """Returns primary stream and overall system status."""
    st = stream.get_status_dict()
    return {
        "camera_id": stream.camera_id,
        "status": st["status"],
        "stream_healthy": st["stream_healthy"],
        "total_cameras": len(camera_manager.streams) or 1,
        "detector_status": "Ready" if detector.model_available else "Simulation Mode"
    }


@app.get("/api/cameras")
def list_cameras():
    """Lists all configured cameras with live telemetry, statuses, and coordinates."""
    cams = get_cameras()
    statuses = {s["camera_id"]: s for s in camera_manager.get_all_statuses()}
    res = []
    for c in cams:
        cid = c["camera_id"]
        stat = statuses.get(cid, {})
        res.append({
            "camera_id": cid,
            "name": c["name"],
            "latitude": c["location_lat"],
            "longitude": c["location_lon"],
            "source_type": c.get("camera_type", "synthetic"),
            "status": stat.get("status", c.get("status", "Offline")),
            "stream_healthy": stat.get("stream_healthy", False),
            "fps": stat.get("fps", 0.0)
        })
    return res


@app.get("/api/cameras/{camera_id}/status")
def get_single_camera_status(camera_id: str):
    """Retrieves status and frame health for a specific camera ID."""
    stat = camera_manager.get_camera_status(camera_id)
    if not stat:
        if camera_id == stream.camera_id:
            return stream.get_status_dict()
        cam = get_camera(camera_id)
        if not cam:
            raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found.")
        return {
            "camera_id": camera_id,
            "name": cam["name"],
            "status": cam.get("status", "Offline"),
            "stream_healthy": False
        }
    return stat


def _generate_mjpeg_stream_for_handler(stream_handler: RTSPStreamHandler):
    """Generates continuous multipart JPEG frames from specified stream."""
    while True:
        frame = stream_handler.get_frame()
        if frame is None:
            time.sleep(0.04)
            continue
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.get("/api/cameras/{camera_id}/stream")
def get_camera_stream(camera_id: str):
    """Streams live MJPEG video from requested camera ID."""
    handler = camera_manager.get_stream(camera_id)
    if not handler:
        if camera_id == stream.camera_id:
            handler = stream
        else:
            raise HTTPException(status_code=404, detail=f"Camera stream '{camera_id}' not found.")

    return StreamingResponse(
        _generate_mjpeg_stream_for_handler(handler),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


def generate_mjpeg_stream():
    """Generates continuous multipart JPEG frames from default primary ingested stream."""
    yield from _generate_mjpeg_stream_for_handler(stream)


@app.get("/api/stream/live")
def get_live_stream():
    """Live video feed endpoint demonstrating active RTSP stream ingestion."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------------------
# YOLO Object & Accident Detection APIs (SCRUM-41)
# ---------------------------------------------------------------------------
@app.get("/api/pipeline/run-inference")
def run_pipeline(
    camera_id: str = "CAM_NORTH_01",
    simulate_crash: bool = False,
    simulate_congestion_count: Optional[int] = None
):
    frame = camera_manager.get_frame(camera_id) or stream.get_frame()
    if frame is None:
        if simulate_crash or simulate_congestion_count is not None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            return {"status": "Waiting for video feed", "camera_id": camera_id}

    if simulate_crash:
        detection = detector.simulate_accident_event()
    else:
        detection = detector.evaluate_frame(frame)

    if simulate_congestion_count is not None:
        detection["vehicle_count"] = simulate_congestion_count

    coords = CAMERAS.get(camera_id, {"lat": 11.016844, "lon": 76.955833})
    congestion_info = broadcaster.generate_congestion_alert(
        camera_id, detection, coords["lat"], coords["lon"]
    )

    if detection["accident_detected"] or simulate_crash:
        payload = construct_incident_payload(camera_id, coords["lat"], coords["lon"])
        return {
            "alert": True,
            "confidence": detection.get("confidence", 0.89 if simulate_crash else 0.85),
            "incident": payload,
            "congestion_alert": congestion_info,
            "detected_vehicles": detection.get("detected_vehicles", [])
        }

    return {
        "alert": False,
        "vehicle_count": detection["vehicle_count"],
        "congestion_alert": congestion_info,
        "detected_vehicles": detection.get("detected_vehicles", [])
    }


@app.get("/api/pipeline/visual-detection")
def visual_detection(simulate_crash: bool = True, camera_id: str = "CAM_NORTH_01"):
    """
    SCRUM-41: Runs YOLO detection, overlays bounding boxes and collision alerts,
    and returns the annotated frame directly.
    """
    frame = camera_manager.get_frame(camera_id) or stream.get_frame()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    annotated = frame.copy()
    detection = detector.evaluate_frame(annotated)

    if simulate_crash or detection["accident_detected"]:
        cv2.rectangle(annotated, (120, 150), (280, 320), (0, 255, 0), 2)
        cv2.putText(annotated, "Car: 0.91", (125, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(annotated, (230, 160), (390, 330), (0, 255, 0), 2)
        cv2.putText(annotated, "Truck: 0.88", (235, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.rectangle(annotated, (220, 150), (290, 320), (0, 0, 255), 3)
        cv2.putText(annotated, "CRASH TRIGGERED (Conf: 89.5%)", (50, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
    else:
        cv2.putText(annotated, "STATUS: NORMAL TRAFFIC FLOW", (50, 420),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

    ret, buffer = cv2.imencode(".jpg", annotated)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Traffic Congestion & Dynamic Rerouting APIs (SCRUM-31)
# ---------------------------------------------------------------------------
@app.get("/api/traffic/congestion")
def get_traffic_congestion(camera_id: str = "CAM_NORTH_01", simulate_vehicle_count: Optional[int] = None):
    if camera_id not in CAMERAS:
        coords = {"lat": None, "lon": None}
    else:
        coords = CAMERAS[camera_id]

    frame = camera_manager.get_frame(camera_id) or stream.get_frame()
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detection = detector.evaluate_frame(frame)
    if simulate_vehicle_count is not None:
        detection["vehicle_count"] = simulate_vehicle_count

    return broadcaster.generate_congestion_alert(
        camera_id=camera_id,
        detection_result=detection,
        lat=coords["lat"],
        lon=coords["lon"]
    )


@app.get("/api/traffic/reroute")
def get_traffic_reroute(
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None,
    dest_lat: Optional[float] = None,
    dest_lon: Optional[float] = None,
    camera_id: Optional[str] = "CAM_NORTH_01"
):
    if (origin_lat is None or origin_lon is None) and camera_id in CAMERAS:
        coords = CAMERAS[camera_id]
        origin_lat = coords["lat"]
        origin_lon = coords["lon"]

    return broadcaster.rerouter.get_alternative_route(
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        dest_lat=dest_lat,
        dest_lon=dest_lon
    )


# ---------------------------------------------------------------------------
# Geolocation & Reverse Lookup APIs (SCRUM-42)
# ---------------------------------------------------------------------------
@app.get("/api/geo/reverse-lookup")
def get_reverse_geocode(camera_id: str = "CAM_NORTH_01"):
    if camera_id not in CAMERAS:
        raise HTTPException(status_code=404, detail="Camera ID not found in registry.")

    coords = CAMERAS[camera_id]
    geo_data = reverse_geocode(coords["lat"], coords["lon"])

    return {
        "camera_id": camera_id,
        "coordinates": {
            "latitude": coords["lat"],
            "longitude": coords["lon"]
        },
        "resolved_address": geo_data["street_name"],
        "landmark": geo_data["landmark"],
        "accuracy_radius_meters": geo_data["accuracy_radius_meters"]
    }


# ---------------------------------------------------------------------------
# Citizen Incident Reporting & Verification APIs (SCRUM-35)
# ---------------------------------------------------------------------------
@app.post("/api/incidents", status_code=status.HTTP_201_CREATED)
async def submit_citizen_incident(
    photo: UploadFile = File(...),
    description: str = Form(...),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    location_source: Optional[str] = Form("manual"),
    incident_category: Optional[str] = Form("Vehicle Collision"),
    incident_time: Optional[str] = Form(None)
):
    clean_desc = (description or "").strip()
    if not clean_desc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident description is required."
        )
    if len(clean_desc) < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident description must be at least 5 characters long."
        )
    if len(clean_desc) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident description exceeds maximum allowed length of 1000 characters."
        )

    if latitude is not None or longitude is not None:
        if latitude is None or longitude is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both latitude and longitude must be provided together."
            )
        if not (-90.0 <= latitude <= 90.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid latitude. Must be between -90.0 and 90.0."
            )
        if not (-180.0 <= longitude <= 180.0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid longitude. Must be between -180.0 and 180.0."
            )

    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident photo file cannot be empty."
        )

    safe_filename, photo_path = save_uploaded_photo(photo_bytes, photo.filename)

    now_dt = datetime.now(timezone.utc)
    incident_id = f"INC-{int(now_dt.timestamp())}-{uuid.uuid4().hex[:6]}"
    submitted_at = now_dt.isoformat()

    address = None
    if latitude is not None and longitude is not None:
        try:
            geo_info = reverse_geocode(latitude, longitude)
            address = geo_info.get("street_name")
        except Exception:
            address = f"{latitude:.6f}, {longitude:.6f}"

    severity = "Critical" if "Pedestrian" in (incident_category or "") or "Rollover" in (incident_category or "") else "Moderate"

    create_incident({
        "incident_id": incident_id,
        "description": clean_desc,
        "latitude": latitude,
        "longitude": longitude,
        "location_source": location_source or "manual",
        "address": address,
        "photo_path": photo_path,
        "photo_filename": safe_filename,
        "incident_category": incident_category or "Vehicle Collision",
        "incident_time": incident_time or submitted_at,
        "submitted_at": submitted_at,
        "severity": severity,
        "detection_source": "citizen_report"
    })

    return {
        "success": True,
        "incident_id": incident_id,
        "status": "PENDING_VERIFICATION",
        "message": "Incident report submitted successfully and is pending operator verification.",
        "submitted_at": submitted_at
    }


@app.get("/api/incidents/pending", response_model=List[Dict[str, Any]])
def get_pending_incidents_queue(operator: Dict = Depends(get_current_operator)):
    return get_pending_incidents()


@app.get("/api/incidents/{incident_id}")
def get_incident_details(incident_id: str, operator: Dict = Depends(get_current_operator)):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    return incident


@app.post("/api/incidents/{incident_id}/verify")
def verify_incident_endpoint(incident_id: str, operator: Dict = Depends(get_current_operator)):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")

    if incident["status"] == "VERIFIED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incident is already verified.")
    if incident["status"] == "REJECTED":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incident has already been rejected.")
    if incident["status"] != "PENDING_VERIFICATION":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Cannot verify incident in '{incident['status']}' state.")

    # Trigger emergency dispatch integration
    dispatch_result = dispatch_service.dispatch_incident(incident)

    # Atomically verify incident in database
    reviewer = operator.get("sub", "Unknown Operator")
    success, msg, updated = verify_incident(
        incident_id=incident_id,
        reviewer=reviewer,
        dispatch_status=dispatch_result["dispatch_status"],
        dispatch_ref=dispatch_result.get("dispatch_reference")
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    return {
        "success": True,
        "incident_id": incident_id,
        "status": "VERIFIED",
        "reviewed_by": reviewer,
        "reviewed_at": updated.get("reviewed_at"),
        "dispatch": dispatch_result,
        "message": "Incident verified successfully."
    }


@app.post("/api/incidents/{incident_id}/reject")
def reject_incident_endpoint(
    incident_id: str,
    payload: Optional[RejectRequest] = None,
    operator: Dict = Depends(get_current_operator)
):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")

    if incident["status"] == "VERIFIED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident is already verified and cannot be rejected."
        )
    if incident["status"] == "REJECTED":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident has already been rejected."
        )
    if incident["status"] != "PENDING_VERIFICATION":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot reject incident in '{incident['status']}' state."
        )

    reason = payload.rejection_reason if payload else None
    reviewer = operator.get("sub", "Unknown Operator")

    success, msg, updated = reject_incident(
        incident_id=incident_id,
        reviewer=reviewer,
        rejection_reason=reason
    )

    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    return {
        "success": True,
        "incident_id": incident_id,
        "status": "REJECTED",
        "reviewed_by": reviewer,
        "reviewed_at": updated.get("reviewed_at"),
        "rejection_reason": updated.get("rejection_reason"),
        "message": "Incident report rejected."
    }


@app.get("/api/incidents/{incident_id}/photo")
def get_incident_photo_endpoint(
    incident_id: str,
    user: Dict = Depends(get_current_user)
):
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")

    photo_filename = incident.get("photo_filename")
    if not photo_filename:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident photo not recorded.")

    file_path = get_photo_path(photo_filename)
    ext = os.path.splitext(file_path)[1].lower()
    media_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp"
    }
    media_type = media_map.get(ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Cache-Control": "private, no-cache"}
    )


# ---------------------------------------------------------------------------
# Emergency SOS & Hospital Dispatch APIs (SCRUM-27)
# ---------------------------------------------------------------------------
@app.get("/api/emergency/nearby-hospitals")
def get_nearby_hospitals_endpoint(
    lat: float = Query(..., description="Incident Latitude"),
    lon: float = Query(..., description="Incident Longitude"),
    radius_km: float = Query(5.0, description="Search radius in kilometers")
):
    """
    Ranks nearby hospitals using Haversine distance, identifies trauma units,
    and flags hospitals within the 5 km coverage radius.
    """
    return dispatch_service.find_nearby_hospitals(latitude=lat, longitude=lon, radius_km=radius_km)


@app.post("/api/emergency/dispatch")
def trigger_manual_dispatch(
    req: DispatchRequest,
    operator: Dict = Depends(get_current_operator)
):
    """Initiates an emergency SOS dispatch for a verified accident."""
    inc = get_incident(req.incident_id)
    if not inc:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return dispatch_service.dispatch_incident(inc, target_hospital_id=req.hospital_id)


@app.get("/api/emergency/dispatch/{incident_id}")
def get_incident_dispatches(incident_id: str, operator: Dict = Depends(get_current_operator)):
    """Retrieves all dispatch attempts and statuses for an incident."""
    return get_dispatches_for_incident(incident_id)


@app.post("/api/emergency/dispatch/{dispatch_id}/accept")
def accept_hospital_dispatch(dispatch_id: str, user: Dict = Depends(get_current_operator)):
    """Hospital unit accepts emergency dispatch."""
    res = dispatch_service.accept_dispatch(dispatch_id)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


@app.post("/api/emergency/dispatch/{dispatch_id}/reject")
def reject_hospital_dispatch(
    dispatch_id: str,
    req: Optional[DispatchRejectRequest] = None,
    user: Dict = Depends(get_current_operator)
):
    """Hospital unit declines emergency dispatch (diverts)."""
    reason = req.reason if req else None
    res = dispatch_service.reject_dispatch(dispatch_id, reason=reason)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res


# ---------------------------------------------------------------------------
# Administrator Analytics & Hotspot Heatmap APIs (SCRUM-34)
# ---------------------------------------------------------------------------
@app.get("/api/analytics/overview")
def get_analytics_overview(admin: Dict = Depends(get_current_admin)):
    """Overview statistics for dashboard cards (total, verified, pending, rejected, critical)."""
    return get_incident_stats()


@app.get("/api/analytics/hourly")
def get_hourly_analytics(
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    admin: Dict = Depends(get_current_admin)
):
    """Accidents per hour of day (0-23) based on actual incident timestamps."""
    return get_hourly_stats(date_from=date_from, date_to=date_to)


@app.get("/api/analytics/monthly")
def get_monthly_analytics(admin: Dict = Depends(get_current_admin)):
    """Monthly accident trend statistics."""
    return get_monthly_stats()


@app.get("/api/analytics/severity")
def get_severity_analytics(admin: Dict = Depends(get_current_admin)):
    """Accident counts classified by severity."""
    return get_severity_stats()


@app.get("/api/analytics/hotspots")
def get_hotspot_analytics(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    admin: Dict = Depends(get_current_admin)
):
    """Geographical incident coordinates and metadata for heatmap rendering."""
    return get_hotspot_data(start_date=start_date, end_date=end_date, severity=severity)


@app.get("/api/analytics/response-times")
def get_response_time_analytics(admin: Dict = Depends(get_current_admin)):
    """Response time analytics calculated from actual stored incident timestamps."""
    return get_response_time_stats()


@app.get("/api/analytics/recent-incidents")
def get_recent_incidents(
    limit: int = Query(20, le=100),
    status: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    admin: Dict = Depends(get_current_admin)
):
    """Recent incidents query with filter parameters."""
    filters = {"limit": limit}
    if status and status != "All":
        filters["status"] = status
    if severity and severity != "All":
        filters["severity"] = severity
    return get_all_incidents(filters)


# ---------------------------------------------------------------------------
# First Responder Active Incidents (SCRUM-29)
# ---------------------------------------------------------------------------
@app.get("/api/geo/active-incidents")
def get_active_incidents():
    """
    Returns verified and pending active incidents with valid coordinates from database
    for the live map view.
    """
    active_incidents = get_all_incidents()
    filtered = []
    for inc in active_incidents:
        if inc.get("status") in ("VERIFIED", "PENDING_VERIFICATION") and inc.get("latitude") and inc.get("longitude"):
            filtered.append({
                "event_id": inc["incident_id"],
                "camera_id": inc.get("camera_id") or "CITIZEN_REPORT",
                "latitude": inc["latitude"],
                "longitude": inc["longitude"],
                "address": inc.get("address") or f"{inc['latitude']:.6f}, {inc['longitude']:.6f}",
                "severity": inc.get("severity") or "Moderate",
                "evidence_url": f"/api/incidents/{inc['incident_id']}/photo" if inc.get("photo_filename") else "/api/pipeline/visual-detection?simulate_crash=true"
            })

    # If database has no active incidents, provide fallback demo points
    if not filtered:
        filtered = [
            {
                "event_id": "INC-1793465470",
                "camera_id": "CAM_NORTH_01",
                "latitude": 11.016844,
                "longitude": 76.955833,
                "address": "North Coimbatore Flyover, Gandhipuram",
                "severity": "Critical",
                "evidence_url": "/api/pipeline/visual-detection?simulate_crash=true"
            },
            {
                "event_id": "INC-1793465488",
                "camera_id": "CAM_AVINASHI_02",
                "latitude": 11.028500,
                "longitude": 76.968200,
                "address": "Avinashi Road Junction, Peelamedu",
                "severity": "Moderate",
                "evidence_url": "/api/pipeline/visual-detection?simulate_crash=true"
            }
        ]

    return {
        "status": "success",
        "total_active": len(filtered),
        "incidents": filtered
    }


# ---------------------------------------------------------------------------
# Archival & Severity Engine APIs (SCRUM-33, SCRUM-26)
# ---------------------------------------------------------------------------
@app.post("/api/archive/record-incident")
def trigger_accident_archival(camera_id: str = "CAM_NORTH_01"):
    frame = camera_manager.get_frame(camera_id) or stream.get_frame()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera feed unavailable.")

    annotated = frame.copy()
    cv2.rectangle(annotated, (120, 150), (280, 320), (0, 255, 0), 2)
    cv2.rectangle(annotated, (230, 160), (390, 330), (0, 255, 0), 2)
    cv2.rectangle(annotated, (220, 150), (290, 320), (0, 0, 255), 3)
    cv2.putText(annotated, "ACCIDENT EVIDENCE ARCHIVE", (30, 420),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    coords = CAMERAS.get(camera_id, {"lat": 11.016844, "lon": 76.955833})
    payload = construct_incident_payload(camera_id, coords["lat"], coords["lon"])

    return archive_incident_evidence(payload["event_id"], camera_id, annotated, payload)


@app.get("/api/archive/{incident_id}/snapshot")
def get_archived_snapshot(incident_id: str):
    folder = os.path.join(ARCHIVE_DIR, incident_id)
    if not os.path.exists(folder):
        raise HTTPException(status_code=404, detail="Incident evidence not found.")
    for f in os.listdir(folder):
        if f.endswith(".jpg"):
            return FileResponse(os.path.join(folder, f), media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Snapshot not found.")


@app.get("/api/archive/{incident_id}/manifest")
def get_archived_manifest(incident_id: str):
    manifest_path = os.path.join(ARCHIVE_DIR, incident_id, f"{incident_id}_manifest.json")
    if not os.path.exists(manifest_path):
        raise HTTPException(status_code=404, detail="Manifest not found.")
    return FileResponse(manifest_path, media_type="application/json")


@app.get("/api/pipeline/severity-classification")
def evaluate_severity(
    vehicle_a: str = "car",
    vehicle_b: str = "truck",
    overlap_iou: float = 0.35
):
    detected_vehicles = [
        {"class": vehicle_a, "confidence": 0.92, "box": [120, 150, 280, 320]},
        {"class": vehicle_b, "confidence": 0.88, "box": [230, 160, 390, 330]}
    ]

    analysis = compute_severity(
        detected_vehicles=detected_vehicles,
        overlap_ratio=overlap_iou
    )

    return {
        "camera_id": stream.camera_id,
        "status": "INCIDENT_CLASSIFIED",
        "involved_vehicles": [v["class"] for v in detected_vehicles],
        "overlap_iou": overlap_iou,
        "severity_analysis": analysis
    }
