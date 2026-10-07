from datetime import datetime, timezone
import requests

def reverse_geocode(lat: float, lon: float) -> dict:
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json"}
    headers = {"User-Agent": "TrafficAccidentDetection/1.0"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=3).json()
        address = res.get("address", {})
        road = address.get("road") or address.get("street") or "Unknown Road"
        suburb = address.get("suburb") or address.get("neighbourhood") or "City Center"
        landmark = res.get("name") or address.get("amenity") or "Intersection"
        formatted_address = f"{road}, near {landmark}, {suburb}"
    except Exception:
        formatted_address = f"Coordinates: {lat:.6f}, {lon:.6f}"

    return {
        "street_name": formatted_address,
        "landmark": landmark if 'landmark' in locals() else "Unknown",
        "accuracy_radius_meters": 5.0
    }

def construct_incident_payload(camera_id: str, lat: float, lon: float, severity: str = "High"):
    location_info = reverse_geocode(lat, lon)
    return {
        "event_id": f"INC-{int(datetime.now(timezone.utc).timestamp())}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "camera_id": camera_id,
        "severity": severity,
        "location": {
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "address": location_info["street_name"],
            "accuracy_meters": location_info["accuracy_radius_meters"]
        },
        "dispatch_status": "PENDING_VERIFICATION"
    }