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

from auth import SECRET_KEY, ALGORITHM
from main import app
from database import init_db, get_incident, get_pending_incidents
from dispatch import EmergencyDispatchService

client = TestClient(app)

def create_valid_test_image(format_type: str = "PNG", size=(60, 60), color="blue") -> bytes:
    """Generates valid image bytes in memory."""
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format=format_type)
    return buf.getvalue()

def generate_jwt_token(email: str, role: str) -> str:
    """Helper to generate signed JWT tokens for various test roles."""
    token_data = {
        "sub": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=2),
    }
    return jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Configures isolated test database and uploads directories for each test."""
    test_db = str(tmp_path / "test_incidents.db")
    test_uploads = str(tmp_path / "test_uploads")
    os.makedirs(test_uploads, exist_ok=True)

    monkeypatch.setenv("INCIDENTS_DB_PATH", test_db)
    monkeypatch.setenv("INCIDENT_UPLOADS_DIR", test_uploads)

    init_db(test_db)
    yield

# ============================================================================
# A. Citizen Report Submission Tests
# ============================================================================

def test_valid_citizen_report_submission():
    """Valid report is persisted with PENDING_VERIFICATION and returns unique incident ID."""
    img_bytes = create_valid_test_image("JPEG")
    files = {"photo": ("crash.jpg", img_bytes, "image/jpeg")}
    data = {
        "description": "Two sedans collided at the north intersection. Moderate frontend damage.",
        "latitude": 11.016844,
        "longitude": 76.955833,
        "location_source": "gps",
        "incident_category": "Vehicle Collision"
    }

    res = client.post("/api/incidents", data=data, files=files)
    assert res.status_code == 201
    body = res.json()
    assert body["success"] is True
    assert body["incident_id"].startswith("INC-")
    assert body["status"] == "PENDING_VERIFICATION"
    assert "submitted_at" in body

    # Verify record in persistent storage
    saved = get_incident(body["incident_id"])
    assert saved is not None
    assert saved["status"] == "PENDING_VERIFICATION"
    assert saved["description"] == data["description"]
    assert saved["latitude"] == 11.016844
    assert saved["longitude"] == 76.955833
    assert saved["location_source"] == "gps"
    assert saved["photo_filename"].endswith(".jpg")

def test_unique_incident_id_generation():
    """Consecutive incident submissions receive unique identifiers."""
    img_bytes = create_valid_test_image("PNG")
    files1 = {"photo": ("img1.png", img_bytes, "image/png")}
    files2 = {"photo": ("img2.png", img_bytes, "image/png")}

    res1 = client.post("/api/incidents", data={"description": "Accident report #1 at crossing."}, files=files1)
    res2 = client.post("/api/incidents", data={"description": "Accident report #2 at highway."}, files=files2)

    assert res1.status_code == 201
    assert res2.status_code == 201
    assert res1.json()["incident_id"] != res2.json()["incident_id"]

# ============================================================================
# B. Input Validation Tests
# ============================================================================

def test_missing_or_short_description():
    """Missing or whitespace-only description is rejected with HTTP 400."""
    img_bytes = create_valid_test_image("PNG")
    files = {"photo": ("test.png", img_bytes, "image/png")}

    # Missing / whitespace
    res = client.post("/api/incidents", data={"description": "   "}, files=files)
    assert res.status_code == 400
    assert "Incident description is required" in res.json()["detail"]

    # Less than minimum length
    res_short = client.post("/api/incidents", data={"description": "Car"}, files=files)
    assert res_short.status_code == 400
    assert "at least 5 characters" in res_short.json()["detail"]

def test_description_exceeding_max_length():
    """Description exceeding 1000 characters is rejected."""
    img_bytes = create_valid_test_image("PNG")
    files = {"photo": ("test.png", img_bytes, "image/png")}
    long_desc = "A" * 1001

    res = client.post("/api/incidents", data={"description": long_desc}, files=files)
    assert res.status_code == 400
    assert "exceeds maximum allowed length" in res.json()["detail"]

def test_invalid_coordinates():
    """Invalid latitude/longitude boundaries or unpaired coordinates are rejected."""
    img_bytes = create_valid_test_image("PNG")
    files = {"photo": ("test.png", img_bytes, "image/png")}

    # Lat out of range
    res = client.post("/api/incidents", data={"description": "Accident on hill road.", "latitude": 95.0, "longitude": 75.0}, files=files)
    assert res.status_code == 400
    assert "Invalid latitude" in res.json()["detail"]

    # Lon out of range
    res = client.post("/api/incidents", data={"description": "Accident on hill road.", "latitude": 10.0, "longitude": 190.0}, files=files)
    assert res.status_code == 400
    assert "Invalid longitude" in res.json()["detail"]

    # Unpaired coordinates
    res = client.post("/api/incidents", data={"description": "Accident on hill road.", "latitude": 10.0}, files=files)
    assert res.status_code == 400
    assert "Both latitude and longitude must be provided together" in res.json()["detail"]

def test_unsupported_image_extension():
    """Executable or script files are strictly rejected."""
    bad_content = b"echo 'malicious payload'"
    files = {"photo": ("payload.sh", bad_content, "text/x-shellscript")}
    data = {"description": "Accident report with script upload attempt."}

    res = client.post("/api/incidents", data=data, files=files)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]

def test_corrupted_or_fake_image_content():
    """Files with image extensions but non-image content are rejected."""
    fake_jpg = b"Not a real image file, just plain text disguised."
    files = {"photo": ("fake.jpg", fake_jpg, "image/jpeg")}
    data = {"description": "Accident report with text file renamed to jpg."}

    res = client.post("/api/incidents", data=data, files=files)
    assert res.status_code == 400
    assert "does not match a valid image signature" in res.json()["detail"] or "invalid or corrupted" in res.json()["detail"]

def test_oversized_image_upload(monkeypatch):
    """Images exceeding the configurable size threshold are rejected with HTTP 413."""
    # Set max size to 50KB for testing
    monkeypatch.setenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024))
    buf = io.BytesIO()
    # Random pixel data prevents PNG compression, generating ~120KB payload
    img = Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3))
    img.save(buf, format="PNG")
    large_img = buf.getvalue()

    files = {"photo": ("large.png", large_img, "image/png")}
    data = {"description": "Accident report with oversized image."}

    res = client.post("/api/incidents", data=data, files=files)
    assert res.status_code == 413
    assert "exceeds maximum allowed file size" in res.json()["detail"]

# ============================================================================
# C. Operator Authorization Tests
# ============================================================================

def test_unauthenticated_user_cannot_access_pending_queue():
    """Unauthenticated requests to operator queue receive HTTP 401."""
    res = client.get("/api/incidents/pending")
    assert res.status_code == 401

def test_citizen_cannot_access_or_verify_incidents():
    """Citizen role cannot access operator queue or verify an incident."""
    citizen_token = generate_jwt_token("citizen@public.org", "Citizen")

    # Queue access denied
    res_queue = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {citizen_token}"})
    assert res_queue.status_code == 403
    assert "Operator privileges are required" in res_queue.json()["detail"]

    # Verification attempt denied
    res_verify = client.post("/api/incidents/INC-99999/verify", headers={"Authorization": f"Bearer {citizen_token}"})
    assert res_verify.status_code == 403

def test_authorized_operator_can_access_queue():
    """Authorized Traffic Operator and EMS Responder can access the pending queue."""
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")
    ems_token = generate_jwt_token("ems@citygov.org", "EMS Responder")

    res_op = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {op_token}"})
    assert res_op.status_code == 200
    assert isinstance(res_op.json(), list)

    res_ems = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {ems_token}"})
    assert res_ems.status_code == 200
    assert isinstance(res_ems.json(), list)

# ============================================================================
# D. One-Click Verification Tests
# ============================================================================

def test_one_click_verification_flow():
    """Operator can verify a pending incident, recording reviewer and removing from queue."""
    img_bytes = create_valid_test_image("PNG")
    files = {"photo": ("crash.png", img_bytes, "image/png")}
    submit_res = client.post("/api/incidents", data={"description": "Bus collided with guardrail on bypass."}, files=files)
    incident_id = submit_res.json()["incident_id"]

    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    # Check incident is present in pending queue
    pending_list = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {op_token}"}).json()
    assert any(inc["incident_id"] == incident_id for inc in pending_list)

    # Operator verifies the incident
    verify_res = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert verify_res.status_code == 200
    v_data = verify_res.json()
    assert v_data["success"] is True
    assert v_data["status"] == "VERIFIED"
    assert v_data["reviewed_by"] == "officer@traffic.org"
    assert v_data["reviewed_at"] is not None

    # Incident should no longer appear in pending queue
    new_pending = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {op_token}"}).json()
    assert not any(inc["incident_id"] == incident_id for inc in new_pending)

def test_repeated_verification_is_rejected():
    """Attempting to verify an already verified incident is rejected."""
    img_bytes = create_valid_test_image("PNG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Motorcycle skidded off ramp."},
        files={"photo": ("bike.png", img_bytes, "image/png")}
    )
    incident_id = submit_res.json()["incident_id"]
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    # First verification
    res1 = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert res1.status_code == 200

    # Repeated verification
    res2 = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert res2.status_code == 400
    assert "already verified" in res2.json()["detail"]

# ============================================================================
# E. One-Click Rejection Tests
# ============================================================================

def test_one_click_rejection_flow():
    """Operator can reject false alarms with a reason; report is removed and dispatch not triggered."""
    img_bytes = create_valid_test_image("PNG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Apparent smoke near bridge."},
        files={"photo": ("smoke.png", img_bytes, "image/png")}
    )
    incident_id = submit_res.json()["incident_id"]
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    reject_res = client.post(
        f"/api/incidents/{incident_id}/reject",
        json={"rejection_reason": "False alarm: Controlled agricultural burn."},
        headers={"Authorization": f"Bearer {op_token}"}
    )
    assert reject_res.status_code == 200
    r_data = reject_res.json()
    assert r_data["success"] is True
    assert r_data["status"] == "REJECTED"
    assert r_data["reviewed_by"] == "officer@traffic.org"
    assert "False alarm" in r_data["rejection_reason"]

    # Verify not in pending queue
    pending = client.get("/api/incidents/pending", headers={"Authorization": f"Bearer {op_token}"}).json()
    assert not any(inc["incident_id"] == incident_id for inc in pending)

    # Cannot verify a rejected incident
    verify_attempt = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert verify_attempt.status_code == 400
    assert "already been rejected" in verify_attempt.json()["detail"]

    # Repeated rejection is rejected
    repeat_reject = client.post(f"/api/incidents/{incident_id}/reject", headers={"Authorization": f"Bearer {op_token}"})
    assert repeat_reject.status_code == 400
    assert "already been rejected" in repeat_reject.json()["detail"]

# ============================================================================
# F. Emergency Dispatch Integration Tests
# ============================================================================

def test_dispatch_unconfigured_reports_accurate_status():
    """When dispatch API is not configured, verification reports DISPATCH_NOT_CONFIGURED without faking contact."""
    img_bytes = create_valid_test_image("PNG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Car rollover on expressway."},
        files={"photo": ("car.png", img_bytes, "image/png")}
    )
    incident_id = submit_res.json()["incident_id"]
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    with patch.dict(os.environ, {"EMERGENCY_DISPATCH_API_URL": ""}):
        from main import dispatch_service
        dispatch_service.api_url = None

        res = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
        assert res.status_code == 200
        disp_info = res.json()["dispatch"]
        assert disp_info["dispatch_status"] == "DISPATCH_NOT_CONFIGURED"
        assert "not configured" in disp_info["message"]
        assert disp_info["dispatch_reference"] is None

@patch("requests.post")
def test_successful_dispatch_integration(mock_post):
    """When dispatch service succeeds, reference ID is recorded and stored."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.content = b'{"dispatch_id": "DISP-889900", "message": "Responders en route."}'
    mock_response.json.return_value = {"dispatch_id": "DISP-889900", "message": "Responders en route."}
    mock_post.return_value = mock_response

    img_bytes = create_valid_test_image("PNG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Overturned oil tanker with minor leak."},
        files={"photo": ("tanker.png", img_bytes, "image/png")}
    )
    incident_id = submit_res.json()["incident_id"]
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    from main import dispatch_service
    dispatch_service.api_url = "http://mock-dispatch-service.internal/api/dispatch"

    res = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert res.status_code == 200
    disp_info = res.json()["dispatch"]
    assert disp_info["dispatch_status"] == "DISPATCHED"
    assert disp_info["dispatch_reference"] == "DISP-889900"

    # Reset dispatch url
    dispatch_service.api_url = None

@patch("requests.post")
def test_dispatch_failure_handled_gracefully(mock_post):
    """When dispatch service returns HTTP 500 or times out, failure is recorded without crashing."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response

    img_bytes = create_valid_test_image("PNG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Rear-end collision on highway."},
        files={"photo": ("crash.png", img_bytes, "image/png")}
    )
    incident_id = submit_res.json()["incident_id"]
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    from main import dispatch_service
    dispatch_service.api_url = "http://mock-dispatch-service.internal/api/dispatch"

    res = client.post(f"/api/incidents/{incident_id}/verify", headers={"Authorization": f"Bearer {op_token}"})
    assert res.status_code == 200
    disp_info = res.json()["dispatch"]
    assert disp_info["dispatch_status"] == "DISPATCH_FAILED"
    assert "500" in disp_info["message"]

    dispatch_service.api_url = None

# ============================================================================
# G. Photo Retrieval & Security Tests
# ============================================================================

def test_secure_photo_access():
    """Authenticated operator can view photo; unauthenticated cannot; missing file returns 404."""
    img_bytes = create_valid_test_image("JPEG")
    submit_res = client.post(
        "/api/incidents",
        data={"description": "Two car crash on highway exit."},
        files={"photo": ("photo.jpg", img_bytes, "image/jpeg")}
    )
    incident_id = submit_res.json()["incident_id"]
    op_token = generate_jwt_token("officer@traffic.org", "Traffic Operator")

    # Unauthenticated access fails
    res_unauth = client.get(f"/api/incidents/{incident_id}/photo")
    assert res_unauth.status_code == 401

    # Authenticated access succeeds via header
    res_auth = client.get(f"/api/incidents/{incident_id}/photo", headers={"Authorization": f"Bearer {op_token}"})
    assert res_auth.status_code == 200
    assert res_auth.headers["content-type"] == "image/jpeg"

    # Authenticated access succeeds via query token
    res_query = client.get(f"/api/incidents/{incident_id}/photo?token={op_token}")
    assert res_query.status_code == 200

    # Non-existent incident photo
    res_404 = client.get("/api/incidents/INC-NONEXISTENT/photo", headers={"Authorization": f"Bearer {op_token}"})
    assert res_404.status_code == 404

# ============================================================================
# H. Web Page Endpoints
# ============================================================================

def test_web_page_routes():
    """Verify that citizen reporting and login portal HTML pages load successfully."""
    res_report = client.get("/report")
    assert res_report.status_code == 200
    assert "Manual Incident Reporting" in res_report.text

    res_login = client.get("/login")
    assert res_login.status_code == 200
    assert "Dispatch Sentinel" in res_login.text

    res_operator = client.get("/operator")
    assert res_operator.status_code == 200
    assert "Command Center" in res_operator.text
