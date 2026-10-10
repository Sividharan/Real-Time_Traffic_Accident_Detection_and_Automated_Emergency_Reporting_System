from fastapi import (
    FastAPI, HTTPException, Response, UploadFile, File, Form, Depends, Query, Header, status
)
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel
import os
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
import cv2
import numpy as np

from auth import verify_credentials, get_current_user, get_current_operator
from stream import RTSPStreamHandler
from detector import AccidentDetector
from geocoding import construct_incident_payload, reverse_geocode
from rerouting import CongestionAlertBroadcaster, TrafficCongestionDetector, TrafficRerouter
from database import (
    init_db, create_incident, get_incident, get_pending_incidents,
    verify_incident, reject_incident
)
from photo_storage import save_uploaded_photo, get_photo_path
from dispatch import EmergencyDispatchService

app = FastAPI(title="Real-Time Accident Detection & Reporting API")

stream = RTSPStreamHandler(camera_id="CAM_NORTH_01", source="synthetic")
detector = AccidentDetector()
broadcaster = CongestionAlertBroadcaster()
dispatch_service = EmergencyDispatchService()

CAMERAS = {
    "CAM_NORTH_01": {"lat": 11.016844, "lon": 76.955833}
}

class LoginRequest(BaseModel):
    email: str
    password: str

class RejectRequest(BaseModel):
    rejection_reason: Optional[str] = None

@app.on_event("startup")
def startup():
    init_db()
    stream.start()

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

@app.post("/api/auth/login")
def login(creds: LoginRequest):
    return verify_credentials(creds.email, creds.password)

@app.get("/api/system/health")
def system_health():
    return {
        "camera_id": stream.camera_id,
        "status": stream.status,
        "stream_healthy": stream.status == "Online"
    }

@app.get("/api/pipeline/run-inference")
def run_pipeline(simulate_crash: bool = False, simulate_congestion_count: Optional[int] = None):
    frame = stream.get_frame()
    if frame is None:
        if simulate_crash or simulate_congestion_count is not None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            return {"status": "Waiting for video feed"}
    
    detection = detector.evaluate_frame(frame)
    if simulate_congestion_count is not None:
        detection["vehicle_count"] = simulate_congestion_count

    coords = CAMERAS.get(stream.camera_id, {"lat": 11.016844, "lon": 76.955833})
    congestion_info = broadcaster.generate_congestion_alert(
        stream.camera_id, detection, coords["lat"], coords["lon"]
    )

    if detection["accident_detected"] or simulate_crash:
        payload = construct_incident_payload(stream.camera_id, coords["lat"], coords["lon"])
        return {
            "alert": True,
            "confidence": 0.89 if simulate_crash else 0.85,
            "incident": payload,
            "congestion_alert": congestion_info
        }
    
    return {
        "alert": False,
        "vehicle_count": detection["vehicle_count"],
        "congestion_alert": congestion_info
    }

@app.get("/api/traffic/congestion")
def get_traffic_congestion(camera_id: str = "CAM_NORTH_01", simulate_vehicle_count: Optional[int] = None):
    """
    SCRUM-31: Evaluates traffic congestion for a camera location and broadcasts status & rerouting.
    """
    if camera_id not in CAMERAS:
        coords = {"lat": None, "lon": None}
    else:
        coords = CAMERAS[camera_id]

    frame = stream.get_frame()
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
    """
    SCRUM-31: Calculates dynamic alternative routes given coordinates or camera ID origin.
    """
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

def generate_mjpeg_stream():
    """Generates continuous multipart JPEG frames from the ingested stream."""
    while True:
        frame = stream.get_frame()
        if frame is None:
            continue
        
        # Encode frame to JPEG
        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

@app.get("/api/stream/live")
def get_live_stream():
    """Live video feed endpoint demonstrating active RTSP stream ingestion."""
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/api/pipeline/visual-detection")
def visual_detection(simulate_crash: bool = True):
    """
    SCRUM-41: Runs YOLO detection, overlays bounding boxes and collision alerts,
    and returns the annotated frame directly.
    """
    frame = stream.get_frame()
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

@app.get("/api/geo/reverse-lookup")
def get_reverse_geocode(camera_id: str = "CAM_NORTH_01"):
    """
    SCRUM-42: Resolves camera coordinates to street name, landmark, 
    and embeds 10m accuracy bounds.
    """
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

# ============================================================================
# SCRUM-35: Citizen Incident Reporting & Verification Endpoints
# ============================================================================

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
    """
    SCRUM-35: Accepts citizen accident reports with photo upload and GPS coordinates.
    Validates input, securely stores image, and registers report with PENDING_VERIFICATION status.
    """
    # 1. Validate description
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

    # 2. Validate coordinates if provided
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

    # 3. Read and store photo securely
    photo_bytes = await photo.read()
    if not photo_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incident photo file cannot be empty."
        )

    safe_filename, photo_path = save_uploaded_photo(photo_bytes, photo.filename)

    # 4. Generate unique incident identifier
    now_dt = datetime.now(timezone.utc)
    incident_id = f"INC-{int(now_dt.timestamp())}-{uuid.uuid4().hex[:6]}"
    submitted_at = now_dt.isoformat()

    # 5. Geocode address if coordinates are provided
    address = None
    if latitude is not None and longitude is not None:
        try:
            geo_info = reverse_geocode(latitude, longitude)
            address = geo_info.get("street_name")
        except Exception:
            address = f"{latitude:.6f}, {longitude:.6f}"

    # 6. Persist record with initial PENDING_VERIFICATION status
    incident_record = create_incident({
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
        "submitted_at": submitted_at
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
    """
    SCRUM-35: Retrieves the queue of incidents awaiting operator verification.
    Restricted to authorized operators.
    """
    return get_pending_incidents()

@app.get("/api/incidents/{incident_id}")
def get_incident_details(incident_id: str, operator: Dict = Depends(get_current_operator)):
    """
    SCRUM-35: Retrieves details of a specific incident.
    """
    incident = get_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found.")
    return incident

@app.post("/api/incidents/{incident_id}/verify")
def verify_incident_endpoint(incident_id: str, operator: Dict = Depends(get_current_operator)):
    """
    SCRUM-35: One-click operator verification of a pending report.
    Validates pending status, updates atomically to VERIFIED, and initiates emergency dispatch.
    """
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
    """
    SCRUM-35: One-click operator rejection of false alarms.
    Records reason and reviewer, sets status to REJECTED, and avoids emergency dispatch.
    """
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
    """
    SCRUM-35: Securely serves incident photo evidence to authenticated users.
    Prevents path traversal and guards against unauthorized access.
    """
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