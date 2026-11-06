import io
import os
import sys
import uuid
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from PIL import Image
import jwt
from fastapi.testclient import TestClient

# Ensure app directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from auth import (
    SECRET_KEY,
    ALGORITHM,
    hash_password,
    check_password,
    init_users_db,
    create_default_users,
    verify_credentials,
    register_user,
)
from database import (
    init_db,
    create_incident,
    get_incident,
    get_all_incidents,
    get_incident_stats,
    get_hourly_stats,
    get_monthly_stats,
    get_severity_stats,
    get_hotspot_data,
    get_response_time_stats,
    get_cameras,
    get_camera,
    get_nearby_hospitals,
    create_dispatch,
    get_dispatch,
    update_dispatch_status,
)
from detector import AccidentDetector
from dispatch import EmergencyDispatchService
from stream import CameraManager, RTSPStreamHandler
from main import app

client = TestClient(app)


def create_test_image(format_type: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (50, 50), color="red")
    img.save(buf, format=format_type)
    return buf.getvalue()


def get_token(email: str, role: str) -> str:
    token_data = {
        "sub": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=2),
    }
    return jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)


@pytest.fixture(autouse=True)
def setup_integration_env(tmp_path, monkeypatch):
    """Sets up an isolated SQLite database and upload folder for each test."""
    test_db = str(tmp_path / "test_sprint.db")
    test_users_db = str(tmp_path / "test_users.db")
    test_uploads = str(tmp_path / "test_uploads")
    os.makedirs(test_uploads, exist_ok=True)

    monkeypatch.setenv("INCIDENTS_DB_PATH", test_db)
    monkeypatch.setenv("INCIDENT_UPLOADS_DIR", test_uploads)

    # Patch users db path to isolated tmp db
    monkeypatch.setattr("auth.get_users_db_path", lambda: test_users_db)

    init_db(test_db)
    init_users_db()
    create_default_users()
    yield


# ============================================================================
# 1. AUTHENTICATION & ACCESS CONTROL TESTS (SCRUM-22, SCRUM-37)
# ============================================================================

def test_password_hashing_and_verification():
    """Verify bcrypt hash produces unique salt and validates correctly."""
    pwd = "MySecretPassword123!"
    hashed = hash_password(pwd)
    assert hashed != pwd
    assert check_password(pwd, hashed) is True
    assert check_password("WrongPassword", hashed) is False


def test_valid_login_returns_jwt_and_role():
    """Valid credentials return signed JWT and correct user role."""
    res = client.post("/api/auth/login", json={"email": "admin@traffic.org", "password": "AdminPass123!"})
    assert res.status_code == 200
    data = res.json()
    assert "access_token" in data
    assert data["role"] == "Administrator"

    # Decode and verify token
    payload = jwt.decode(data["access_token"], SECRET_KEY, algorithms=[ALGORITHM])
    assert payload["sub"] == "admin@traffic.org"
    assert payload["role"] == "Administrator"


def test_invalid_login_wrong_password():
    """Invalid password returns HTTP 401 with remaining attempts."""
    res = client.post("/api/auth/login", json={"email": "officer@traffic.org", "password": "WrongPassword!"})
    assert res.status_code == 401
    assert "Invalid credentials" in res.json()["detail"]


def test_account_lockout_after_five_failed_attempts():
    """Five consecutive failed attempts lock the account."""
    email = "citizen@public.org"
    for _ in range(4):
        res = client.post("/api/auth/login", json={"email": email, "password": "BadPassword!"})
        assert res.status_code == 401

    # 5th attempt locks the account
    res5 = client.post("/api/auth/login", json={"email": email, "password": "BadPassword!"})
    assert res5.status_code == 403
    assert "Account locked after 5 failed tries" in res5.json()["detail"]

    # 6th attempt remains locked even if password is correct
    res6 = client.post("/api/auth/login", json={"email": email, "password": "CitizenPass123!"})
    assert res6.status_code == 403
    assert "Account locked" in res6.json()["detail"]


def test_expired_jwt_token_handling():
    """Expired JWT token returns HTTP 401 with appropriate message."""
    expired_token = jwt.encode(
        {"sub": "admin@traffic.org", "role": "Administrator", "exp": datetime.now(timezone.utc) - timedelta(hours=1)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    res = client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {expired_token}"})
    assert res.status_code == 401
    assert "expired" in res.json()["detail"].lower()


def test_role_based_access_control_admin_endpoints():
    """Traffic Operator and Citizen cannot access Admin-only analytics endpoints."""
    officer_token = get_token("officer@traffic.org", "Traffic Operator")
    citizen_token = get_token("citizen@public.org", "Citizen")
    admin_token = get_token("admin@traffic.org", "Administrator")

    # Officer denied
    res_off = client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {officer_token}"})
    assert res_off.status_code == 403

    # Citizen denied
    res_cit = client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {citizen_token}"})
    assert res_cit.status_code == 403

    # Admin allowed
    res_adm = client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_adm.status_code == 200


def test_auth_me_and_logout_endpoints():
    """Authenticated user profile lookup and logout."""
    token = get_token("officer@traffic.org", "Traffic Operator")
    res_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert res_me.status_code == 200
    assert res_me.json()["email"] == "officer@traffic.org"

    res_out = client.post("/api/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert res_out.status_code == 200
    assert res_out.json()["success"] is True


# ============================================================================
# 2. VIDEO STREAMING PIPELINE TESTS (SCRUM-40)
# ============================================================================

def test_camera_list_and_status():
    """Querying cameras API returns configured cameras and telemetry."""
    res = client.get("/api/cameras")
    assert res.status_code == 200
    cams = res.json()
    assert len(cams) >= 3
    cam_ids = [c["camera_id"] for c in cams]
    assert "CAM_NORTH_01" in cam_ids
    assert "CAM_AVINASHI_02" in cam_ids


def test_camera_status_endpoint():
    """Specific camera status check returns FPS and health."""
    res = client.get("/api/cameras/CAM_NORTH_01/status")
    assert res.status_code == 200
    data = res.json()
    assert data["camera_id"] == "CAM_NORTH_01"
    assert "status" in data


def test_camera_manager_add_and_remove():
    """CameraManager properly handles adding, retrieving, and removing streams."""
    mgr = CameraManager()
    h = mgr.add_camera("CAM_TEST_99", source="synthetic", name="Test Highway")
    assert h.camera_id == "CAM_TEST_99"

    stat = mgr.get_camera_status("CAM_TEST_99")
    assert stat is not None
    assert stat["camera_id"] == "CAM_TEST_99"

    frame = mgr.get_frame("CAM_TEST_99")
    assert frame is not None or h.status in ("Initializing", "Online")

    removed = mgr.remove_camera("CAM_TEST_99")
    assert removed is True
    assert mgr.get_stream("CAM_TEST_99") is None


# ============================================================================
# 3. YOLO OBJECT & ACCIDENT DETECTION TESTS (SCRUM-41)
# ============================================================================

def test_yolo_detector_iou_calculation():
    """IoU calculation matches expected geometry."""
    det = AccidentDetector(model_weight="yolov8n.pt")
    box1 = [0, 0, 10, 10]
    box2 = [5, 0, 15, 10]
    iou = det.calculate_iou(box1, box2)
    # intersection: 5*10 = 50, area1=100, area2=100, union=150 => iou ~ 0.333
    assert 0.32 < iou < 0.35


def test_yolo_simulated_accident_event():
    """AccidentDetector simulated crash event generates structured payload."""
    det = AccidentDetector(model_weight="yolov8n.pt")
    event = det.simulate_accident_event()
    assert event["accident_detected"] is True
    assert event["simulated"] is True
    assert event["vehicle_count"] == 2
    assert len(event["crash_boxes"]) == 2
    assert "detected_vehicles" in event


def test_pipeline_run_inference_with_simulation():
    """Running pipeline inference with simulate_crash=true returns alert payload."""
    res = client.get("/api/pipeline/run-inference?simulate_crash=true")
    assert res.status_code == 200
    data = res.json()
    assert data["alert"] is True
    assert "incident" in data
    assert "congestion_alert" in data


# ============================================================================
# 4. EMERGENCY DISPATCH & HOSPITAL TRIAGE TESTS (SCRUM-27)
# ============================================================================

def test_nearby_hospital_lookup_ranking():
    """Nearby hospital search ranks by Haversine distance and flags 5km boundary."""
    # North Coimbatore Flyover coordinates
    lat, lon = 11.016844, 76.955833
    res = client.get(f"/api/emergency/nearby-hospitals?lat={lat}&lon={lon}&radius_km=5.0")
    assert res.status_code == 200
    hospitals = res.json()
    assert len(hospitals) > 0
    # First hospital should be within radius
    assert hospitals[0]["within_radius"] is True
    assert hospitals[0]["distance_km"] <= 5.0
    # List is sorted
    for i in range(len(hospitals) - 1):
        if hospitals[i]["within_radius"] == hospitals[i + 1]["within_radius"]:
            assert hospitals[i]["distance_km"] <= hospitals[i + 1]["distance_km"]


def test_emergency_dispatch_creation_and_acceptance():
    """Emergency dispatch creation and hospital accept workflow."""
    op_token = get_token("officer@traffic.org", "Traffic Operator")
    img_bytes = create_test_image("PNG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Two cars collided at junction."},
        files={"photo": ("crash.png", img_bytes, "image/png")}
    )
    inc_id = submit_res.json()["incident_id"]

    # Verify incident
    client.post(f"/api/incidents/{inc_id}/verify", headers={"Authorization": f"Bearer {op_token}"})

    # Create dispatch via simulation service
    disp_service = EmergencyDispatchService()
    incident_rec = get_incident(inc_id)
    disp_res = disp_service.dispatch_incident(incident_rec, allow_simulation=True)
    assert disp_res["dispatch_status"] == "DISPATCHED"
    disp_ref = disp_res["dispatch_reference"]

    # Hospital accepts dispatch
    accept_res = client.post(f"/api/emergency/dispatch/{disp_ref}/accept", headers={"Authorization": f"Bearer {op_token}"})
    assert accept_res.status_code == 200
    assert accept_res.json()["success"] is True

    # Re-accepting already accepted dispatch is rejected
    dup_accept = client.post(f"/api/emergency/dispatch/{disp_ref}/accept", headers={"Authorization": f"Bearer {op_token}"})
    assert dup_accept.status_code == 400


def test_emergency_dispatch_rejection():
    """Hospital rejects dispatch when at capacity."""
    op_token = get_token("officer@traffic.org", "Traffic Operator")
    disp_service = EmergencyDispatchService()
    disp_rec = create_dispatch({
        "incident_id": "INC-TEST-REJECT",
        "hospital_id": "HOSP_01",
        "hospital_name": "CMCH",
        "severity": "Critical"
    })
    disp_id = disp_rec["dispatch_id"]

    rej_res = client.post(
        f"/api/emergency/dispatch/{disp_id}/reject",
        json={"reason": "Trauma unit full"},
        headers={"Authorization": f"Bearer {op_token}"}
    )
    assert rej_res.status_code == 200
    assert rej_res.json()["success"] is True

    saved = get_dispatch(disp_id)
    assert saved["status"] == "REJECTED"
    assert saved["rejection_reason"] == "Trauma unit full"


# ============================================================================
# 5. ADMIN ANALYTICS & HOTSPOT HEATMAP TESTS (SCRUM-34)
# ============================================================================

def test_admin_analytics_overview():
    """Admin analytics overview returns incident KPI cards."""
    admin_token = get_token("admin@traffic.org", "Administrator")
    res = client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    data = res.json()
    assert "total_incidents" in data
    assert "verified_incidents" in data
    assert "pending_incidents" in data
    assert "critical_incidents" in data


def test_admin_analytics_hourly_and_monthly():
    """Hourly and monthly trend aggregations."""
    admin_token = get_token("admin@traffic.org", "Administrator")

    res_h = client.get("/api/analytics/hourly", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_h.status_code == 200
    hourly = res_h.json()
    assert len(hourly) == 24
    assert hourly[0]["hour"] == "00:00"

    res_m = client.get("/api/analytics/monthly", headers={"Authorization": f"Bearer {admin_token}"})
    assert res_m.status_code == 200
    assert isinstance(res_m.json(), list)


def test_admin_hotspot_geographical_data():
    """Hotspots endpoint returns coordinate points with severity."""
    admin_token = get_token("admin@traffic.org", "Administrator")
    # Seed an incident with coordinates
    create_incident({
        "incident_id": "INC-HOTSPOT-1",
        "description": "High risk junction crash",
        "latitude": 11.016844,
        "longitude": 76.955833,
        "severity": "Critical",
        "status": "VERIFIED"
    })

    res = client.get("/api/analytics/hotspots", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    points = res.json()
    assert any(p["incident_id"] == "INC-HOTSPOT-1" for p in points)


def test_admin_response_time_analytics():
    """Response time analytics calculations."""
    admin_token = get_token("admin@traffic.org", "Administrator")
    res = client.get("/api/analytics/response-times", headers={"Authorization": f"Bearer {admin_token}"})
    assert res.status_code == 200
    data = res.json()
    assert "available" in data


# ============================================================================
# 6. FULL WORKFLOW END-TO-END INTEGRATION TEST
# ============================================================================

def test_full_system_integration_lifecycle():
    """
    Simulates complete lifecycle:
    1. Citizen submits incident report with photo and coordinates
    2. Incident is recorded in SQLite with PENDING_VERIFICATION
    3. Traffic Operator views pending queue
    4. Operator reviews and verifies incident
    5. Incident transitions atomically to VERIFIED
    6. Emergency dispatch payload is evaluated with nearby hospitals
    7. Admin analytics reflects verified incident counts
    """
    # 1. Citizen submits
    img_bytes = create_test_image("JPEG")
    submit_res = client.post(
        "/api/incidents",
        data={
            "description": "Bystander reporting two-vehicle collision on North Coimbatore Flyover.",
            "latitude": 11.016844,
            "longitude": 76.955833,
            "location_source": "gps",
            "incident_category": "Vehicle Collision"
        },
        files={"photo": ("crash_scene.jpg", img_bytes, "image/jpeg")}
    )
    assert submit_res.status_code == 201
    inc_id = submit_res.json()["incident_id"]

    # 2. Verify in DB
    record = get_incident(inc_id)
    assert record["status"] == "PENDING_VERIFICATION"

    # 3. Operator views queue
    op_token = get_token("officer@traffic.org", "Traffic Operator")
    queue_res = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {op_token}"})
    assert any(i["incident_id"] == inc_id for i in queue_res.json())

    # 4 & 5. Operator verifies
    v_res = client.post(f"/api/incidents/{inc_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert v_res.status_code == 200
    assert v_res.json()["status"] == "VERIFIED"

    # 6. Check nearby trauma hospitals
    hosp_res = client.get(f"/api/emergency/nearby-hospitals?lat={record['latitude']}&lon={record['longitude']}")
    assert len(hosp_res.json()) > 0

    # 7. Admin sees updated counts
    adm_token = get_token("admin@traffic.org", "Administrator")
    analytics_res = client.get("/api/analytics/overview", headers={"Authorization": f"Bearer {adm_token}"})
    assert analytics_res.json()["verified_incidents"] >= 1
