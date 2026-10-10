import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "incidents.db")

def get_db_path(custom_path: Optional[str] = None) -> str:
    """Returns the active SQLite database path from parameter, environment, or default."""
    if custom_path:
        return custom_path
    env_path = os.getenv("INCIDENTS_DB_PATH")
    if env_path:
        return env_path
    return DEFAULT_DB_PATH

def get_db_connection(custom_path: Optional[str] = None) -> sqlite3.Connection:
    """Creates a configured SQLite connection with row factories and timeout."""
    path = get_db_path(custom_path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(custom_path: Optional[str] = None) -> None:
    """Initializes the incidents table schema and indexes if they do not exist."""
    conn = get_db_connection(custom_path)
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incidents (
                    incident_id TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    location_source TEXT DEFAULT 'manual',
                    address TEXT,
                    photo_path TEXT NOT NULL,
                    photo_filename TEXT NOT NULL,
                    incident_category TEXT DEFAULT 'Other',
                    incident_time TEXT,
                    submitted_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING_VERIFICATION',
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    rejection_reason TEXT,
                    dispatch_status TEXT NOT NULL DEFAULT 'PENDING_VERIFICATION',
                    dispatch_reference TEXT,
                    created_at TEXT NOT NULL
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_submitted ON incidents(submitted_at);
            """)
    finally:
        conn.close()

def create_incident(data: Dict[str, Any], custom_path: Optional[str] = None) -> Dict[str, Any]:
    """Persists a new incident record atomically into the database."""
    init_db(custom_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "incident_id": data["incident_id"],
        "description": data["description"].strip(),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "location_source": data.get("location_source", "manual"),
        "address": data.get("address"),
        "photo_path": data["photo_path"],
        "photo_filename": data["photo_filename"],
        "incident_category": data.get("incident_category", "Vehicle Collision"),
        "incident_time": data.get("incident_time") or now_iso,
        "submitted_at": data.get("submitted_at") or now_iso,
        "status": "PENDING_VERIFICATION",
        "reviewed_by": None,
        "reviewed_at": None,
        "rejection_reason": None,
        "dispatch_status": "PENDING_VERIFICATION",
        "dispatch_reference": None,
        "created_at": now_iso
    }

    conn = get_db_connection(custom_path)
    try:
        with conn:
            conn.execute("""
                INSERT INTO incidents (
                    incident_id, description, latitude, longitude, location_source,
                    address, photo_path, photo_filename, incident_category, incident_time,
                    submitted_at, status, reviewed_by, reviewed_at, rejection_reason,
                    dispatch_status, dispatch_reference, created_at
                ) VALUES (
                    :incident_id, :description, :latitude, :longitude, :location_source,
                    :address, :photo_path, :photo_filename, :incident_category, :incident_time,
                    :submitted_at, :status, :reviewed_by, :reviewed_at, :rejection_reason,
                    :dispatch_status, :dispatch_reference, :created_at
                )
            """, record)
    finally:
        conn.close()

    return record

def get_incident(incident_id: str, custom_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Retrieves an incident by its unique ID."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_pending_incidents(custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves all pending verification incidents sorted newest first."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM incidents 
            WHERE status = 'PENDING_VERIFICATION' 
            ORDER BY submitted_at DESC
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

def verify_incident(
    incident_id: str,
    reviewer: str,
    dispatch_status: str = "PENDING_VERIFICATION",
    dispatch_ref: Optional[str] = None,
    custom_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Atomically updates a pending incident to VERIFIED status with reviewer information.
    Guards against race conditions, duplicate verifications, and invalid transitions.
    """
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return False, "Incident not found.", None

            current_status = row["status"]
            if current_status == "VERIFIED":
                return False, "Incident is already verified.", dict(row)
            if current_status == "REJECTED":
                return False, "Incident has already been rejected.", dict(row)
            if current_status != "PENDING_VERIFICATION":
                return False, f"Invalid status transition from {current_status}.", dict(row)

            now_iso = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
                UPDATE incidents
                SET status = 'VERIFIED',
                    reviewed_by = ?,
                    reviewed_at = ?,
                    dispatch_status = ?,
                    dispatch_reference = ?
                WHERE incident_id = ? AND status = 'PENDING_VERIFICATION'
            """, (reviewer, now_iso, dispatch_status, dispatch_ref, incident_id))

            if cursor.rowcount == 0:
                return False, "Concurrent review conflict: incident was already processed.", None

            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            updated_row = dict(cursor.fetchone())
            return True, "Incident verified successfully.", updated_row
    finally:
        conn.close()

def reject_incident(
    incident_id: str,
    reviewer: str,
    rejection_reason: Optional[str] = None,
    custom_path: Optional[str] = None
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Atomically updates a pending incident to REJECTED status with reason and reviewer.
    Guards against race conditions, duplicate rejections, and invalid transitions.
    """
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            row = cursor.fetchone()
            if not row:
                return False, "Incident not found.", None

            current_status = row["status"]
            if current_status == "VERIFIED":
                return False, "Incident is already verified and cannot be rejected.", dict(row)
            if current_status == "REJECTED":
                return False, "Incident has already been rejected.", dict(row)
            if current_status != "PENDING_VERIFICATION":
                return False, f"Invalid status transition from {current_status}.", dict(row)

            now_iso = datetime.now(timezone.utc).isoformat()
            reason = (rejection_reason or "").strip() or "Rejected by operator as false alarm"
            cursor.execute("""
                UPDATE incidents
                SET status = 'REJECTED',
                    reviewed_by = ?,
                    reviewed_at = ?,
                    rejection_reason = ?,
                    dispatch_status = 'NOT_DISPATCHED'
                WHERE incident_id = ? AND status = 'PENDING_VERIFICATION'
            """, (reviewer, now_iso, reason, incident_id))

            if cursor.rowcount == 0:
                return False, "Concurrent review conflict: incident was already processed.", None

            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            updated_row = dict(cursor.fetchone())
            return True, "Incident report rejected.", updated_row
    finally:
        conn.close()
