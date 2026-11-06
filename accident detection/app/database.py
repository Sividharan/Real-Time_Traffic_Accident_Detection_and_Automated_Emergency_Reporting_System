import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

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
    """Initializes database schema, creates all required tables and indexes."""
    conn = get_db_connection(custom_path)
    try:
        with conn:
            # 1. Incidents table
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

            # Add optional extension columns to incidents table if they do not exist
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(incidents);")
            columns = [col[1] for col in cursor.fetchall()]

            extension_cols = {
                "severity": "TEXT DEFAULT 'Moderate'",
                "camera_id": "TEXT",
                "detection_source": "TEXT DEFAULT 'manual'",
                "response_time_seconds": "REAL"
            }
            for col_name, col_type in extension_cols.items():
                if col_name not in columns:
                    try:
                        conn.execute(f"ALTER TABLE incidents ADD COLUMN {col_name} {col_type};")
                    except Exception:
                        pass

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_incidents_submitted ON incidents(submitted_at);
            """)

            # 2. Cameras table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    location_lat REAL,
                    location_lon REAL,
                    source_url TEXT DEFAULT 'synthetic',
                    status TEXT DEFAULT 'Offline',
                    camera_type TEXT DEFAULT 'synthetic',
                    created_at TEXT NOT NULL
                );
            """)

            # 3. Hospitals table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hospitals (
                    hospital_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    address TEXT,
                    has_trauma_center INTEGER DEFAULT 0,
                    phone TEXT,
                    capacity TEXT DEFAULT 'Available',
                    created_at TEXT NOT NULL
                );
            """)

            # 4. Emergency Dispatches table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS emergency_dispatches (
                    dispatch_id TEXT PRIMARY KEY,
                    incident_id TEXT NOT NULL,
                    hospital_id TEXT,
                    hospital_name TEXT,
                    severity TEXT,
                    status TEXT DEFAULT 'PENDING',
                    dispatched_at TEXT,
                    accepted_at TEXT,
                    rejected_at TEXT,
                    rejection_reason TEXT,
                    response_time_seconds REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
                );
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_dispatches_incident ON emergency_dispatches(incident_id);
            """)

            # 5. Incident Events table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS incident_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    description TEXT,
                    camera_id TEXT,
                    severity TEXT,
                    latitude REAL,
                    longitude REAL,
                    detection_confidence REAL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (incident_id) REFERENCES incidents(incident_id)
                );
            """)

        # Seed defaults
        seed_default_data(custom_path)
    finally:
        conn.close()


def seed_default_data(custom_path: Optional[str] = None) -> None:
    """Seeds default cameras and hospitals if the tables are empty."""
    conn = get_db_connection(custom_path)
    try:
        now_iso = datetime.now(timezone.utc).isoformat()
        with conn:
            # Seed cameras
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) as count FROM cameras")
            if cur.fetchone()["count"] == 0:
                default_cameras = [
                    ("CAM_NORTH_01", "North Coimbatore Flyover", 11.016844, 76.955833, "synthetic", "Online", "synthetic", now_iso),
                    ("CAM_AVINASHI_02", "Avinashi Road Junction", 11.028500, 76.968200, "synthetic", "Online", "synthetic", now_iso),
                    ("CAM_GANDHIPURAM_03", "Gandhipuram Bus Stand", 11.014200, 76.961500, "synthetic", "Online", "synthetic", now_iso),
                    ("CAM_METTUPALAYAM_04", "Mettupalayam Highway Crossing", 11.042100, 76.948200, "synthetic", "Online", "synthetic", now_iso)
                ]
                conn.executemany("""
                    INSERT INTO cameras (camera_id, name, location_lat, location_lon, source_url, status, camera_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, default_cameras)

            # Seed hospitals
            cur.execute("SELECT COUNT(*) as count FROM hospitals")
            if cur.fetchone()["count"] == 0:
                default_hospitals = [
                    ("HOSP_01", "Coimbatore Medical College Hospital", 11.016800, 76.955800, "Trichy Rd, Coimbatore", 1, "+91 422 230 1393", "Available", now_iso),
                    ("HOSP_02", "PSG Hospitals", 11.024300, 76.939600, "Peelamedu, Coimbatore", 1, "+91 422 257 0170", "Available", now_iso),
                    ("HOSP_03", "KG Hospital", 11.002800, 76.961400, "Arts College Rd, Coimbatore", 1, "+91 422 221 2121", "Available", now_iso),
                    ("HOSP_04", "Sri Ramakrishna Hospital", 11.008900, 76.947600, "Siddhapudur, Coimbatore", 1, "+91 422 450 0000", "Available", now_iso),
                    ("HOSP_05", "Ganga Hospital", 11.021700, 76.981300, "Swarnambika Layout, Coimbatore", 1, "+91 422 248 5000", "Available", now_iso),
                    ("HOSP_06", "Royal Care Super Speciality Hospital", 11.065200, 77.034500, "Neelambur, Coimbatore", 1, "+91 422 222 7000", "Available", now_iso)
                ]
                conn.executemany("""
                    INSERT INTO hospitals (hospital_id, name, latitude, longitude, address, has_trauma_center, phone, capacity, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, default_hospitals)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Incidents Operations
# ---------------------------------------------------------------------------

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
        "photo_path": data.get("photo_path", ""),
        "photo_filename": data.get("photo_filename", ""),
        "incident_category": data.get("incident_category", "Vehicle Collision"),
        "incident_time": data.get("incident_time") or now_iso,
        "submitted_at": data.get("submitted_at") or now_iso,
        "status": data.get("status", "PENDING_VERIFICATION"),
        "reviewed_by": None,
        "reviewed_at": None,
        "rejection_reason": None,
        "dispatch_status": data.get("dispatch_status", "PENDING_VERIFICATION"),
        "dispatch_reference": None,
        "severity": data.get("severity", "Moderate"),
        "camera_id": data.get("camera_id"),
        "detection_source": data.get("detection_source", "manual"),
        "response_time_seconds": None,
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
                    dispatch_status, dispatch_reference, severity, camera_id,
                    detection_source, response_time_seconds, created_at
                ) VALUES (
                    :incident_id, :description, :latitude, :longitude, :location_source,
                    :address, :photo_path, :photo_filename, :incident_category, :incident_time,
                    :submitted_at, :status, :reviewed_by, :reviewed_at, :rejection_reason,
                    :dispatch_status, :dispatch_reference, :severity, :camera_id,
                    :detection_source, :response_time_seconds, :created_at
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
    Calculates actual response time between submission and verification.
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

            now_dt = datetime.now(timezone.utc)
            now_iso = now_dt.isoformat()

            # Calculate actual response time in seconds
            response_time_seconds = None
            submitted_str = row["submitted_at"]
            if submitted_str:
                try:
                    sub_dt = datetime.fromisoformat(submitted_str.replace("Z", "+00:00"))
                    diff = (now_dt - sub_dt).total_seconds()
                    if diff >= 0:
                        response_time_seconds = round(diff, 2)
                except Exception:
                    pass

            cursor.execute("""
                UPDATE incidents
                SET status = 'VERIFIED',
                    reviewed_by = ?,
                    reviewed_at = ?,
                    dispatch_status = ?,
                    dispatch_reference = ?,
                    response_time_seconds = ?
                WHERE incident_id = ? AND status = 'PENDING_VERIFICATION'
            """, (reviewer, now_iso, dispatch_status, dispatch_ref, response_time_seconds, incident_id))

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

            now_dt = datetime.now(timezone.utc)
            now_iso = now_dt.isoformat()
            reason = (rejection_reason or "").strip() or "Rejected by operator as false alarm"

            response_time_seconds = None
            submitted_str = row["submitted_at"]
            if submitted_str:
                try:
                    sub_dt = datetime.fromisoformat(submitted_str.replace("Z", "+00:00"))
                    diff = (now_dt - sub_dt).total_seconds()
                    if diff >= 0:
                        response_time_seconds = round(diff, 2)
                except Exception:
                    pass

            cursor.execute("""
                UPDATE incidents
                SET status = 'REJECTED',
                    reviewed_by = ?,
                    reviewed_at = ?,
                    rejection_reason = ?,
                    dispatch_status = 'NOT_DISPATCHED',
                    response_time_seconds = ?
                WHERE incident_id = ? AND status = 'PENDING_VERIFICATION'
            """, (reviewer, now_iso, reason, response_time_seconds, incident_id))

            if cursor.rowcount == 0:
                return False, "Concurrent review conflict: incident was already processed.", None

            cursor.execute("SELECT * FROM incidents WHERE incident_id = ?", (incident_id,))
            updated_row = dict(cursor.fetchone())
            return True, "Incident report rejected.", updated_row
    finally:
        conn.close()


def get_all_incidents(
    filters: Optional[Dict[str, Any]] = None,
    custom_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves all incidents with optional filtering for analytics and admin views."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        query = "SELECT * FROM incidents WHERE 1=1"
        params: List[Any] = []

        if filters:
            if filters.get("status"):
                query += " AND status = ?"
                params.append(filters["status"])
            if filters.get("severity"):
                query += " AND severity = ?"
                params.append(filters["severity"])
            if filters.get("camera_id"):
                query += " AND camera_id = ?"
                params.append(filters["camera_id"])
            if filters.get("start_date"):
                query += " AND submitted_at >= ?"
                params.append(filters["start_date"])
            if filters.get("end_date"):
                query += " AND submitted_at <= ?"
                params.append(filters["end_date"])

        query += " ORDER BY submitted_at DESC"
        if filters and filters.get("limit"):
            query += " LIMIT ?"
            params.append(int(filters["limit"]))

        cursor = conn.cursor()
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Cameras Operations
# ---------------------------------------------------------------------------

def create_camera(data: Dict[str, Any], custom_path: Optional[str] = None) -> Dict[str, Any]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    record = {
        "camera_id": data["camera_id"],
        "name": data["name"],
        "location_lat": data.get("location_lat"),
        "location_lon": data.get("location_lon"),
        "source_url": data.get("source_url", "synthetic"),
        "status": data.get("status", "Online"),
        "camera_type": data.get("camera_type", "synthetic"),
        "created_at": now_iso
    }
    try:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO cameras 
                (camera_id, name, location_lat, location_lon, source_url, status, camera_type, created_at)
                VALUES (:camera_id, :name, :location_lat, :location_lon, :source_url, :status, :camera_type, :created_at)
            """, record)
    finally:
        conn.close()
    return record


def get_cameras(custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM cameras ORDER BY camera_id ASC")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_camera(camera_id: str, custom_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM cameras WHERE camera_id = ?", (camera_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_camera_status(camera_id: str, status: str, custom_path: Optional[str] = None) -> bool:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("UPDATE cameras SET status = ? WHERE camera_id = ?", (status, camera_id))
            return cur.rowcount > 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Hospitals & Haversine Distance
# ---------------------------------------------------------------------------

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates distance between two coordinates in kilometers using Haversine formula."""
    R = 6371.0  # Earth's radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def get_hospitals(custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM hospitals ORDER BY name ASC")
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def get_nearby_hospitals(
    lat: float, lon: float, radius_km: float = 5.0, custom_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Ranks nearby hospitals by Haversine distance, identifies trauma facilities,
    and returns eligible hospitals within radius_km.
    """
    all_hospitals = get_hospitals(custom_path)
    results = []
    for h in all_hospitals:
        dist = haversine_distance(lat, lon, h["latitude"], h["longitude"])
        h_copy = dict(h)
        h_copy["distance_km"] = dist
        h_copy["within_radius"] = dist <= radius_km
        results.append(h_copy)

    # Sort primarily by within_radius (True first), then distance ascending
    results.sort(key=lambda x: (not x["within_radius"], x["distance_km"]))
    return results


# ---------------------------------------------------------------------------
# Emergency Dispatches Operations
# ---------------------------------------------------------------------------

def create_dispatch(data: Dict[str, Any], custom_path: Optional[str] = None) -> Dict[str, Any]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    dispatch_id = data.get("dispatch_id") or f"DISP-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "dispatch_id": dispatch_id,
        "incident_id": data["incident_id"],
        "hospital_id": data.get("hospital_id"),
        "hospital_name": data.get("hospital_name"),
        "severity": data.get("severity", "Moderate"),
        "status": data.get("status", "DISPATCHED"),
        "dispatched_at": data.get("dispatched_at") or now_iso,
        "accepted_at": None,
        "rejected_at": None,
        "rejection_reason": None,
        "response_time_seconds": None,
        "created_at": now_iso
    }
    try:
        with conn:
            conn.execute("""
                INSERT INTO emergency_dispatches (
                    dispatch_id, incident_id, hospital_id, hospital_name, severity,
                    status, dispatched_at, accepted_at, rejected_at, rejection_reason,
                    response_time_seconds, created_at
                ) VALUES (
                    :dispatch_id, :incident_id, :hospital_id, :hospital_name, :severity,
                    :status, :dispatched_at, :accepted_at, :rejected_at, :rejection_reason,
                    :response_time_seconds, :created_at
                )
            """, record)
    finally:
        conn.close()
    return record


def get_dispatch(dispatch_id: str, custom_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM emergency_dispatches WHERE dispatch_id = ?", (dispatch_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_dispatches_for_incident(incident_id: str, custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM emergency_dispatches WHERE incident_id = ? ORDER BY created_at DESC", (incident_id,))
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def update_dispatch_status(
    dispatch_id: str, status: str, custom_path: Optional[str] = None, **kwargs
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Updates emergency dispatch status (ACCEPTED, REJECTED) and logs response time."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        with conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM emergency_dispatches WHERE dispatch_id = ?", (dispatch_id,))
            row = cur.fetchone()
            if not row:
                return False, "Dispatch not found.", None

            disp = dict(row)
            now_dt = datetime.now(timezone.utc)
            now_iso = now_dt.isoformat()

            accepted_at = kwargs.get("accepted_at")
            rejected_at = kwargs.get("rejected_at")
            rejection_reason = kwargs.get("rejection_reason")

            resp_seconds = None
            if status == "ACCEPTED" and disp.get("dispatched_at"):
                try:
                    d_dt = datetime.fromisoformat(disp["dispatched_at"].replace("Z", "+00:00"))
                    resp_seconds = round((now_dt - d_dt).total_seconds(), 2)
                except Exception:
                    pass

            cur.execute("""
                UPDATE emergency_dispatches
                SET status = ?,
                    accepted_at = COALESCE(?, accepted_at),
                    rejected_at = COALESCE(?, rejected_at),
                    rejection_reason = COALESCE(?, rejection_reason),
                    response_time_seconds = COALESCE(?, response_time_seconds)
                WHERE dispatch_id = ?
            """, (status, accepted_at, rejected_at, rejection_reason, resp_seconds, dispatch_id))

            cur.execute("SELECT * FROM emergency_dispatches WHERE dispatch_id = ?", (dispatch_id,))
            return True, "Dispatch status updated.", dict(cur.fetchone())
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Analytics Helper Functions (Actual SQLite aggregations)
# ---------------------------------------------------------------------------

def get_incident_stats(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """Calculates overall metrics from actual stored incident records."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as total FROM incidents")
        total = cur.fetchone()["total"]

        cur.execute("SELECT COUNT(*) as cnt FROM incidents WHERE status = 'VERIFIED'")
        verified = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM incidents WHERE status = 'PENDING_VERIFICATION'")
        pending = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM incidents WHERE status = 'REJECTED'")
        rejected = cur.fetchone()["cnt"]

        cur.execute("SELECT COUNT(*) as cnt FROM incidents WHERE severity = 'Critical'")
        critical = cur.fetchone()["cnt"]

        # Calculate actual average response time from incidents that have response_time_seconds
        cur.execute("SELECT AVG(response_time_seconds) as avg_resp FROM incidents WHERE response_time_seconds IS NOT NULL")
        avg_resp = cur.fetchone()["avg_resp"]

        return {
            "total_incidents": total,
            "verified_incidents": verified,
            "pending_incidents": pending,
            "rejected_incidents": rejected,
            "critical_incidents": critical,
            "average_response_time_seconds": round(avg_resp, 1) if avg_resp is not None else None
        }
    finally:
        conn.close()


def get_hourly_stats(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    custom_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Groups incident frequencies by hour of day (0-23)."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        query = """
            SELECT strftime('%H', submitted_at) as hour, COUNT(*) as count 
            FROM incidents 
            WHERE submitted_at IS NOT NULL
        """
        params: List[Any] = []
        if date_from:
            query += " AND submitted_at >= ?"
            params.append(date_from)
        if date_to:
            query += " AND submitted_at <= ?"
            params.append(date_to)

        query += " GROUP BY hour ORDER BY hour ASC"
        cur.execute(query, params)
        rows = cur.fetchall()

        # Format map for all 24 hours
        hour_map = {f"{h:02d}": 0 for h in range(24)}
        for r in rows:
            if r["hour"] in hour_map:
                hour_map[r["hour"]] = r["count"]

        return [{"hour": f"{h}:00", "count": count} for h, count in sorted(hour_map.items())]
    finally:
        conn.close()


def get_monthly_stats(custom_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Groups incident frequencies by month."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT strftime('%Y-%m', submitted_at) as month, COUNT(*) as count 
            FROM incidents 
            WHERE submitted_at IS NOT NULL
            GROUP BY month 
            ORDER BY month ASC
        """)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_severity_stats(custom_path: Optional[str] = None) -> Dict[str, int]:
    """Calculates counts per severity level."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT COALESCE(severity, 'Moderate') as sev, COUNT(*) as count 
            FROM incidents 
            GROUP BY sev
        """)
        res = {"Critical": 0, "Moderate": 0, "Minor": 0}
        for r in cur.fetchall():
            res[r["sev"]] = r["count"]
        return res
    finally:
        conn.close()


def get_hotspot_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    severity: Optional[str] = None,
    custom_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Returns geographical coordinates, accident frequency, and severity for heatmap rendering."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        query = """
            SELECT 
                incident_id,
                latitude,
                longitude,
                address,
                severity,
                status,
                submitted_at,
                camera_id
            FROM incidents
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        """
        params: List[Any] = []
        if start_date:
            query += " AND submitted_at >= ?"
            params.append(start_date)
        if end_date:
            query += " AND submitted_at <= ?"
            params.append(end_date)
        if severity and severity != "All":
            query += " AND severity = ?"
            params.append(severity)

        query += " ORDER BY submitted_at DESC"
        cur.execute(query, params)
        return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def get_response_time_stats(custom_path: Optional[str] = None) -> Dict[str, Any]:
    """Retrieves response time analytics from actual stored timestamps."""
    init_db(custom_path)
    conn = get_db_connection(custom_path)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT 
                COUNT(response_time_seconds) as samples,
                AVG(response_time_seconds) as avg_time,
                MIN(response_time_seconds) as min_time,
                MAX(response_time_seconds) as max_time
            FROM incidents
            WHERE response_time_seconds IS NOT NULL
        """)
        row = cur.fetchone()
        if not row or row["samples"] == 0:
            return {
                "available": False,
                "message": "No response time data recorded yet.",
                "sample_count": 0,
                "average_seconds": None,
                "min_seconds": None,
                "max_seconds": None
            }
        return {
            "available": True,
            "sample_count": row["samples"],
            "average_seconds": round(row["avg_time"], 1),
            "min_seconds": round(row["min_time"], 1),
            "max_seconds": round(row["max_time"], 1)
        }
    finally:
        conn.close()
