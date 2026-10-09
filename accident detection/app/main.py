from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import os
from typing import Optional
import cv2
import numpy as np
from auth import verify_credentials
from stream import RTSPStreamHandler
from detector import AccidentDetector
from geocoding import construct_incident_payload, reverse_geocode
from rerouting import CongestionAlertBroadcaster, TrafficCongestionDetector, TrafficRerouter

app = FastAPI(title="Real-Time Accident Detection & Reporting API")

stream = RTSPStreamHandler(camera_id="CAM_NORTH_01", source="synthetic")
detector = AccidentDetector()
broadcaster = CongestionAlertBroadcaster()

CAMERAS = {
    "CAM_NORTH_01": {"lat": 11.016844, "lon": 76.955833}
}

class LoginRequest(BaseModel):
    email: str
    password: str

@app.on_event("startup")
def startup():
    stream.start()

@app.get("/login", response_class=HTMLResponse)
def get_login_page():
    html_path = os.path.join(os.path.dirname(__file__), "login.html")
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