import os
import json
from datetime import datetime, timezone
import cv2

ARCHIVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "archive"))
os.makedirs(ARCHIVE_DIR, exist_ok=True)

def archive_incident_evidence(incident_id: str, camera_id: str, frame, metadata: dict) -> dict:
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    incident_folder = os.path.join(ARCHIVE_DIR, incident_id)
    os.makedirs(incident_folder, exist_ok=True)

    # 1. Save visual snapshot
    snapshot_filename = f"{incident_id}_{timestamp_str}_snapshot.jpg"
    snapshot_path = os.path.join(incident_folder, snapshot_filename)
    cv2.imwrite(snapshot_path, frame)

    # 2. Save manifest metadata
    manifest_data = {
        "incident_id": incident_id,
        "camera_id": camera_id,
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_file": snapshot_filename,
        "incident_details": metadata
    }
    manifest_path = os.path.join(incident_folder, f"{incident_id}_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=4)

    return {
        "incident_id": incident_id,
        "status": "ARCHIVED",
        "snapshot_url": f"/api/archive/{incident_id}/snapshot",
        "manifest_url": f"/api/archive/{incident_id}/manifest"
    }
