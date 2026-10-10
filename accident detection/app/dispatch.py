import os
from typing import Any, Dict, Optional
import requests

class EmergencyDispatchService:
    """
    Emergency services dispatch integration client.
    Handles dispatching verified traffic incidents to external emergency services.
    Reports accurate status when unconfigured, offline, or failing, without inventing responses.
    """

    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.getenv("EMERGENCY_DISPATCH_API_URL")
        self.api_key = api_key or os.getenv("EMERGENCY_DISPATCH_API_KEY")

    def dispatch_incident(self, incident: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dispatches a verified incident to the emergency dispatch system.
        Returns a structured outcome dictionary.
        """
        incident_id = incident.get("incident_id")

        # 1. Unconfigured integration check
        if not self.api_url:
            return {
                "dispatch_status": "DISPATCH_NOT_CONFIGURED",
                "message": "Emergency dispatch API endpoint is not configured. External responders were not contacted.",
                "dispatch_reference": None
            }

        # 2. Idempotency guard: avoid duplicate dispatch calls
        if incident.get("dispatch_status") == "DISPATCHED" and incident.get("dispatch_reference"):
            return {
                "dispatch_status": "DISPATCHED",
                "message": "Emergency services were already dispatched for this incident.",
                "dispatch_reference": incident.get("dispatch_reference")
            }

        payload = {
            "incident_id": incident_id,
            "timestamp": incident.get("submitted_at"),
            "category": incident.get("incident_category", "Vehicle Collision"),
            "description": incident.get("description"),
            "location": {
                "latitude": incident.get("latitude"),
                "longitude": incident.get("longitude"),
                "source": incident.get("location_source", "manual"),
                "address": incident.get("address")
            },
            "severity": incident.get("severity", "High")
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TrafficAccidentDispatch/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        try:
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
            if response.status_code in (200, 201):
                res_data = response.json() if response.content else {}
                ref = (
                    res_data.get("dispatch_id")
                    or res_data.get("dispatch_reference")
                    or res_data.get("reference_id")
                    or f"DISP-{incident_id}"
                )
                msg = res_data.get("message") or "Emergency responders dispatched successfully."
                return {
                    "dispatch_status": "DISPATCHED",
                    "message": msg,
                    "dispatch_reference": str(ref)
                }
            else:
                return {
                    "dispatch_status": "DISPATCH_FAILED",
                    "message": f"Dispatch service responded with HTTP status code {response.status_code}.",
                    "dispatch_reference": None
                }
        except requests.Timeout:
            return {
                "dispatch_status": "DISPATCH_FAILED",
                "message": "Emergency dispatch service connection timed out after 5 seconds.",
                "dispatch_reference": None
            }
        except Exception as exc:
            return {
                "dispatch_status": "DISPATCH_FAILED",
                "message": f"Emergency dispatch connection failed: {str(exc)}",
                "dispatch_reference": None
            }
