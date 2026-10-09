import os
import requests
from typing import Dict, Any, Optional
from geocoding import reverse_geocode

class TrafficCongestionDetector:
    def __init__(self, threshold: Optional[int] = None):
        if threshold is not None:
            self.threshold = threshold
        else:
            env_thresh = os.getenv("CONGESTION_VEHICLE_THRESHOLD", "5")
            try:
                self.threshold = int(env_thresh)
            except ValueError:
                self.threshold = 5

    def evaluate_congestion(self, detection_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates whether current vehicle detection data indicates traffic congestion.
        Handles missing or invalid detection data safely.
        """
        if not detection_result or not isinstance(detection_result, dict):
            return {
                "is_congested": False,
                "vehicle_count": 0,
                "threshold": self.threshold,
                "congestion_status": "DATA_UNAVAILABLE",
                "message": "Detection data unavailable or invalid."
            }

        vehicle_count = detection_result.get("vehicle_count")
        if vehicle_count is None or not isinstance(vehicle_count, (int, float)) or vehicle_count < 0:
            return {
                "is_congested": False,
                "vehicle_count": 0,
                "threshold": self.threshold,
                "congestion_status": "DATA_UNAVAILABLE",
                "message": "Vehicle count unavailable or invalid."
            }

        vehicle_count = int(vehicle_count)
        is_congested = vehicle_count >= self.threshold

        if vehicle_count < self.threshold:
            status = "NORMAL"
        elif vehicle_count < self.threshold * 2:
            status = "MODERATE_CONGESTION"
        else:
            status = "HEAVY_CONGESTION"

        return {
            "is_congested": is_congested,
            "vehicle_count": vehicle_count,
            "threshold": self.threshold,
            "congestion_status": status,
            "message": f"Traffic flow evaluation complete: {status}."
        }


class TrafficRerouter:
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.getenv("ROUTING_API_URL")
        self.api_key = api_key or os.getenv("ROUTING_API_KEY")

    def get_alternative_route(
        self,
        origin_lat: Optional[float],
        origin_lon: Optional[float],
        dest_lat: Optional[float] = None,
        dest_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Retrieves dynamic alternative routing details.
        If no API is configured or connection fails, provides clear fallback message.
        """
        if origin_lat is None or origin_lon is None:
            return {
                "status": "LOCATION_UNAVAILABLE",
                "message": "Origin coordinates unavailable for routing.",
                "suggested_route": "Standard traffic advisory fallback route",
                "estimated_time_saved_minutes": None,
                "distance_km": None,
                "duration_minutes": None
            }

        if not self.api_url:
            return {
                "status": "FALLBACK_NO_API_CONFIGURED",
                "message": "Routing API is not configured. Set ROUTING_API_URL in environment variables to enable dynamic routing.",
                "suggested_route": "Bypass via Outer Ring Road / Northern Expressway",
                "estimated_time_saved_minutes": None,
                "distance_km": None,
                "duration_minutes": None
            }

        dest_lat = dest_lat if dest_lat is not None else origin_lat + 0.05
        dest_lon = dest_lon if dest_lon is not None else origin_lon + 0.05

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        try:
            params = {
                "origin": f"{origin_lat},{origin_lon}",
                "destination": f"{dest_lat},{dest_lon}"
            }
            res = requests.get(self.api_url, params=params, headers=headers, timeout=5)
            if res.status_code == 200:
                data = res.json()
                if "routes" in data and len(data["routes"]) > 0:
                    best_route = data["routes"][0]
                    duration_mins = round(best_route.get("duration", 0) / 60.0, 1) if best_route.get("duration") is not None else None
                    distance_km = round(best_route.get("distance", 0) / 1000.0, 1) if best_route.get("distance") is not None else None
                    route_name = best_route.get("name") or "Dynamic Bypass Route"
                    return {
                        "status": "SUCCESS",
                        "message": "Alternative route calculated via Routing API.",
                        "suggested_route": route_name,
                        "estimated_time_saved_minutes": best_route.get("time_saved_minutes"),
                        "distance_km": distance_km,
                        "duration_minutes": duration_mins
                    }
                elif "suggested_route" in data:
                    return {
                        "status": "SUCCESS",
                        "message": data.get("message", "Alternative route retrieved."),
                        "suggested_route": data.get("suggested_route"),
                        "estimated_time_saved_minutes": data.get("estimated_time_saved_minutes"),
                        "distance_km": data.get("distance_km"),
                        "duration_minutes": data.get("duration_minutes")
                    }
                else:
                    return {
                        "status": "API_UNEXPECTED_FORMAT",
                        "message": "Routing API returned success but payload format was unexpected.",
                        "suggested_route": "Advisory Route: Follow primary detour signs",
                        "estimated_time_saved_minutes": None,
                        "distance_km": None,
                        "duration_minutes": None
                    }
            else:
                return {
                    "status": "API_ERROR",
                    "message": f"Routing API request failed with HTTP status code {res.status_code}.",
                    "suggested_route": "Advisory Route: Follow primary detour signs",
                    "estimated_time_saved_minutes": None,
                    "distance_km": None,
                    "duration_minutes": None
                }
        except Exception as e:
            return {
                "status": "API_CONNECTION_FAILED",
                "message": f"Failed to connect to Routing API: {str(e)}",
                "suggested_route": "Advisory Route: Follow primary detour signs",
                "estimated_time_saved_minutes": None,
                "distance_km": None,
                "duration_minutes": None
            }


class CongestionAlertBroadcaster:
    def __init__(self, detector: Optional[TrafficCongestionDetector] = None, rerouter: Optional[TrafficRerouter] = None):
        self.detector = detector or TrafficCongestionDetector()
        self.rerouter = rerouter or TrafficRerouter()

    def generate_congestion_alert(
        self,
        camera_id: str,
        detection_result: Optional[Dict[str, Any]],
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        dest_lat: Optional[float] = None,
        dest_lon: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generates structured congestion alert with location metadata and dynamic rerouting fallback.
        """
        eval_res = self.detector.evaluate_congestion(detection_result)
        
        location_data = None
        if lat is not None and lon is not None:
            try:
                geo_info = reverse_geocode(lat, lon)
                location_data = {
                    "camera_id": camera_id,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "address": geo_info.get("street_name", f"{lat:.6f}, {lon:.6f}"),
                    "landmark": geo_info.get("landmark", "Unknown")
                }
            except Exception:
                location_data = {
                    "camera_id": camera_id,
                    "latitude": round(float(lat), 6),
                    "longitude": round(float(lon), 6),
                    "address": f"Coordinates: {lat:.6f}, {lon:.6f}",
                    "landmark": "Unknown"
                }
        else:
            location_data = {
                "camera_id": camera_id,
                "latitude": None,
                "longitude": None,
                "address": "Location information unavailable",
                "landmark": "Unknown"
            }

        route_data = self.rerouter.get_alternative_route(lat, lon, dest_lat, dest_lon)

        return {
            "camera_id": camera_id,
            "congestion_alert": eval_res["is_congested"],
            "congestion_status": eval_res["congestion_status"],
            "vehicle_count": eval_res["vehicle_count"],
            "threshold": eval_res["threshold"],
            "message": eval_res["message"],
            "location": location_data,
            "alternative_route": route_data
        }
