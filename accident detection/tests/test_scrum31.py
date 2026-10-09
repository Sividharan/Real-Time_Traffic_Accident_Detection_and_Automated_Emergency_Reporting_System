import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Ensure app directory is on Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))

from rerouting import TrafficCongestionDetector, TrafficRerouter, CongestionAlertBroadcaster
from main import app

client = TestClient(app)

def test_normal_traffic_no_congestion_alert():
    """1. Normal traffic should not trigger a congestion alert."""
    detector = TrafficCongestionDetector(threshold=5)
    result = detector.evaluate_congestion({"vehicle_count": 3})
    assert result["is_congested"] is False
    assert result["congestion_status"] == "NORMAL"
    assert result["vehicle_count"] == 3

def test_exceeding_threshold_triggers_congestion_alert():
    """2. Traffic exceeding the configured threshold should trigger a congestion alert."""
    detector = TrafficCongestionDetector(threshold=5)
    result = detector.evaluate_congestion({"vehicle_count": 7})
    assert result["is_congested"] is True
    assert result["congestion_status"] == "MODERATE_CONGESTION"
    assert result["vehicle_count"] == 7

def test_missing_vehicle_detection_data():
    """3. Missing vehicle detection data should be handled safely."""
    detector = TrafficCongestionDetector(threshold=5)
    
    res1 = detector.evaluate_congestion(None)
    assert res1["is_congested"] is False
    assert res1["congestion_status"] == "DATA_UNAVAILABLE"
    
    res2 = detector.evaluate_congestion({})
    assert res2["is_congested"] is False
    assert res2["congestion_status"] == "DATA_UNAVAILABLE"
    
    res3 = detector.evaluate_congestion("invalid")
    assert res3["is_congested"] is False
    assert res3["congestion_status"] == "DATA_UNAVAILABLE"

def test_missing_location_information():
    """4. Missing location information should not crash the app."""
    broadcaster = CongestionAlertBroadcaster()
    alert = broadcaster.generate_congestion_alert(
        camera_id="CAM_UNKNOWN",
        detection_result={"vehicle_count": 8},
        lat=None,
        lon=None
    )
    assert alert["congestion_alert"] is True
    assert alert["location"]["address"] == "Location information unavailable"
    assert alert["location"]["latitude"] is None

@patch("requests.get")
def test_valid_alternative_route_from_routing_api(mock_get):
    """5. A valid alternative route should be displayed when the routing service returns one."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "routes": [
            {
                "name": "Express Bypass Highway",
                "distance": 12500,
                "duration": 900,
                "time_saved_minutes": 8.5
            }
        ]
    }
    mock_get.return_value = mock_response

    rerouter = TrafficRerouter(api_url="http://mock-routing-api.internal/route")
    route_info = rerouter.get_alternative_route(11.016844, 76.955833, 11.050000, 76.980000)
    
    assert route_info["status"] == "SUCCESS"
    assert route_info["suggested_route"] == "Express Bypass Highway"
    assert route_info["duration_minutes"] == 15.0
    assert route_info["distance_km"] == 12.5
    assert route_info["estimated_time_saved_minutes"] == 8.5

@patch("requests.get")
def test_routing_api_failure_handled_gracefully(mock_get):
    """6. Routing API failures should be handled gracefully."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    rerouter = TrafficRerouter(api_url="http://mock-routing-api.internal/route")
    route_info = rerouter.get_alternative_route(11.016844, 76.955833)
    
    assert route_info["status"] == "API_ERROR"
    assert "500" in route_info["message"]
    assert "suggested_route" in route_info

    mock_get.side_effect = Exception("Connection timed out")
    route_info_exc = rerouter.get_alternative_route(11.016844, 76.955833)
    assert route_info_exc["status"] == "API_CONNECTION_FAILED"
    assert "Connection timed out" in route_info_exc["message"]

def test_missing_api_configuration():
    """7. Missing API configuration should produce a clear message."""
    rerouter = TrafficRerouter(api_url=None)
    route_info = rerouter.get_alternative_route(11.016844, 76.955833)
    
    assert route_info["status"] == "FALLBACK_NO_API_CONFIGURED"
    assert "Routing API is not configured" in route_info["message"]
    assert "suggested_route" in route_info

def test_existing_application_functionality():
    """8. Existing application functionality should remain intact."""
    health_res = client.get("/api/system/health")
    assert health_res.status_code == 200
    assert "camera_id" in health_res.json()

    login_res = client.post("/api/auth/login", json={"email": "officer@traffic.org", "password": "SecurePass123!"})
    assert login_res.status_code == 200
    assert "access_token" in login_res.json()

    geo_res = client.get("/api/geo/reverse-lookup?camera_id=CAM_NORTH_01")
    assert geo_res.status_code == 200
    assert "resolved_address" in geo_res.json()

    inf_res = client.get("/api/pipeline/run-inference?simulate_crash=true")
    assert inf_res.status_code == 200
    assert inf_res.json()["alert"] is True
    assert "congestion_alert" in inf_res.json()

    traffic_res = client.get("/api/traffic/congestion?simulate_vehicle_count=10")
    assert traffic_res.status_code == 200
    assert traffic_res.json()["congestion_alert"] is True

    reroute_res = client.get("/api/traffic/reroute?camera_id=CAM_NORTH_01")
    assert reroute_res.status_code == 200
    assert "suggested_route" in reroute_res.json()
