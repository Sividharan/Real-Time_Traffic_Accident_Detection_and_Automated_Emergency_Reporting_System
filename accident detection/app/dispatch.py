import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests

from database import (
    create_dispatch,
    get_dispatch,
    get_dispatches_for_incident,
    get_nearby_hospitals,
    update_dispatch_status,
)


class EmergencyDispatchService:
    """
    Emergency SOS Dispatch and First Responder Notification Integration Client.
    Calculates nearby trauma centers, manages dispatches to eligible hospitals,
    prevents duplicate dispatches, handles hospital accept/reject workflows,
    and supports simulated webhook mode for development safety.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        default_radius_km: float = 5.0
    ):
        self.api_url = api_url or os.getenv("EMERGENCY_DISPATCH_API_URL")
        self.api_key = api_key or os.getenv("EMERGENCY_DISPATCH_API_KEY")
        self.default_radius_km = default_radius_km

    def find_nearby_hospitals(
        self,
        latitude: float,
        longitude: float,
        radius_km: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries database for hospitals, calculates distance using Haversine formula,
        and flags trauma capability.
        """
        r = radius_km if radius_km is not None else self.default_radius_km
        return get_nearby_hospitals(latitude, longitude, radius_km=r)

    def dispatch_incident(
        self,
        incident: Dict[str, Any],
        target_hospital_id: Optional[str] = None,
        allow_simulation: bool = False
    ) -> Dict[str, Any]:
        """
        Dispatches a verified accident incident to the nearest eligible hospital or webhook.
        Ensures idempotency, avoids duplicate dispatches, and reports accurate status when unconfigured.
        """
        incident_id = incident.get("incident_id")

        # 1. Unconfigured integration check (strictly reports DISPATCH_NOT_CONFIGURED unless explicitly simulating)
        if not self.api_url and not allow_simulation:
            return {
                "dispatch_status": "DISPATCH_NOT_CONFIGURED",
                "message": "Emergency dispatch API endpoint is not configured. External responders were not contacted.",
                "dispatch_reference": None
            }

        # 2. Idempotency guard: prevent duplicate dispatches
        if incident.get("dispatch_status") == "DISPATCHED" and incident.get("dispatch_reference"):
            return {
                "dispatch_status": "DISPATCHED",
                "message": "Emergency services were already dispatched for this incident.",
                "dispatch_reference": incident.get("dispatch_reference")
            }

        existing_dispatches = get_dispatches_for_incident(incident_id)
        for d in existing_dispatches:
            if d.get("status") in ("DISPATCHED", "ACCEPTED"):
                return {
                    "dispatch_status": d["status"],
                    "message": "Emergency services were already dispatched for this incident.",
                    "dispatch_reference": d["dispatch_id"],
                    "hospital_name": d.get("hospital_name")
                }

        # 3. Select eligible hospital within radius
        lat = incident.get("latitude")
        lon = incident.get("longitude")
        chosen_hospital = None

        if lat is not None and lon is not None:
            nearby = self.find_nearby_hospitals(lat, lon, radius_km=self.default_radius_km)
            if target_hospital_id:
                for h in nearby:
                    if h["hospital_id"] == target_hospital_id:
                        chosen_hospital = h
                        break
            if not chosen_hospital and nearby:
                severity = incident.get("severity", "Moderate")
                if severity == "Critical":
                    trauma_hospitals = [h for h in nearby if h.get("within_radius") and h.get("has_trauma_center")]
                    if trauma_hospitals:
                        chosen_hospital = trauma_hospitals[0]
                if not chosen_hospital:
                    within_r = [h for h in nearby if h.get("within_radius")]
                    if within_r:
                        chosen_hospital = within_r[0]
                    else:
                        chosen_hospital = nearby[0]

        dispatch_id = f"DISP-{uuid.uuid4().hex[:8].upper()}"
        now_iso = datetime.now(timezone.utc).isoformat()
        hospital_name = chosen_hospital.get("name") if chosen_hospital else "Central EMS Dispatch"
        hospital_id = chosen_hospital.get("hospital_id") if chosen_hospital else "HOSP_GEN"

        # 4. If unconfigured but allow_simulation requested:
        if not self.api_url and allow_simulation:
            create_dispatch({
                "dispatch_id": dispatch_id,
                "incident_id": incident_id,
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "severity": incident.get("severity", "Moderate"),
                "status": "DISPATCHED",
                "dispatched_at": now_iso
            })
            return {
                "dispatch_status": "DISPATCHED",
                "message": f"Simulated SOS dispatch sent to {hospital_name}.",
                "dispatch_reference": dispatch_id,
                "hospital_id": hospital_id,
                "hospital_name": hospital_name,
                "simulated": True
            }

        # 5. External webhook call
        payload = {
            "incident_id": incident_id,
            "timestamp": incident.get("submitted_at") or now_iso,
            "category": incident.get("incident_category", "Vehicle Collision"),
            "description": incident.get("description"),
            "location": {
                "latitude": lat,
                "longitude": lon,
                "source": incident.get("location_source", "manual"),
                "address": incident.get("address")
            },
            "severity": incident.get("severity", "High"),
            "hospital": {
                "id": hospital_id,
                "name": hospital_name
            }
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TrafficAccidentDispatch/1.0"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            headers["X-API-Key"] = self.api_key

        try:
            res = requests.post(self.api_url, json=payload, headers=headers, timeout=5.0)
            if res.status_code in (200, 201):
                res_data = res.json() if res.content else {}
                ref = (
                    res_data.get("dispatch_id")
                    or res_data.get("dispatch_reference")
                    or res_data.get("reference_id")
                    or f"DISP-{incident_id}"
                )
                msg = res_data.get("message") or "Emergency responders dispatched successfully."
                create_dispatch({
                    "dispatch_id": str(ref),
                    "incident_id": incident_id,
                    "hospital_id": hospital_id,
                    "hospital_name": hospital_name,
                    "severity": incident.get("severity", "Moderate"),
                    "status": "DISPATCHED",
                    "dispatched_at": now_iso
                })
                return {
                    "dispatch_status": "DISPATCHED",
                    "message": msg,
                    "dispatch_reference": str(ref)
                }
            else:
                return {
                    "dispatch_status": "DISPATCH_FAILED",
                    "message": f"Dispatch service responded with HTTP status code {res.status_code}.",
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

    def accept_dispatch(self, dispatch_id: str) -> Dict[str, Any]:
        """Hospital unit accepts the emergency dispatch."""
        disp = get_dispatch(dispatch_id)
        if not disp:
            return {"success": False, "message": f"Dispatch record '{dispatch_id}' not found."}
        if disp.get("status") == "ACCEPTED":
            return {"success": False, "message": "Dispatch has already been accepted."}

        now_iso = datetime.now(timezone.utc).isoformat()
        ok, msg, updated = update_dispatch_status(dispatch_id, "ACCEPTED", accepted_at=now_iso)
        return {
            "success": ok,
            "message": "Dispatch accepted by emergency unit.",
            "dispatch": updated
        }

    def reject_dispatch(self, dispatch_id: str, reason: Optional[str] = None) -> Dict[str, Any]:
        """Hospital unit rejects dispatch (e.g. at capacity, diverted)."""
        disp = get_dispatch(dispatch_id)
        if not disp:
            return {"success": False, "message": f"Dispatch record '{dispatch_id}' not found."}

        now_iso = datetime.now(timezone.utc).isoformat()
        rejection_reason = reason or "Hospital trauma unit at full capacity - diverted"
        ok, msg, updated = update_dispatch_status(
            dispatch_id, "REJECTED", rejected_at=now_iso, rejection_reason=rejection_reason
        )
        return {
            "success": ok,
            "message": "Dispatch declined by responder.",
            "dispatch": updated
        }
