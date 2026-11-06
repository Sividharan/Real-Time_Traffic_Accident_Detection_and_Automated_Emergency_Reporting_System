import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import bcrypt
from fastapi import Depends, Header, HTTPException, Query, status
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "traffic-system-super-secret-key")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
TOKEN_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "8"))


def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def check_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


# ---------------------------------------------------------------------------
# Persistent SQLite user storage
# ---------------------------------------------------------------------------

def get_users_db_path() -> str:
    """Returns the path to the users SQLite database (same directory as app code)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.db")


def _get_users_conn() -> sqlite3.Connection:
    path = get_users_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_users_db() -> None:
    """Creates the users table and unique index if they do not exist."""
    conn = _get_users_conn()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'Citizen',
                    full_name TEXT DEFAULT '',
                    failed_attempts INTEGER DEFAULT 0,
                    is_locked INTEGER DEFAULT 0,
                    locked_at TEXT,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                )
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)
            """)
    finally:
        conn.close()


def create_default_users() -> None:
    """Inserts default development / demo users if they do not already exist."""
    defaults = [
        ("admin@traffic.org", "AdminPass123!", "Administrator", "System Admin"),
        ("officer@traffic.org", "SecurePass123!", "Traffic Operator", "Traffic Officer"),
        ("ems@citygov.org", "ResponderPass123!", "EMS Responder", "EMS Responder"),
        ("citizen@public.org", "CitizenPass123!", "Citizen", "Public Citizen"),
    ]
    now_iso = datetime.now(timezone.utc).isoformat()
    conn = _get_users_conn()
    try:
        with conn:
            for email, password, role, full_name in defaults:
                existing = conn.execute(
                    "SELECT id FROM users WHERE email = ?", (email,)
                ).fetchone()
                if not existing:
                    pw_hash = hash_password(password)
                    conn.execute(
                        """INSERT INTO users
                           (email, password_hash, role, full_name, failed_attempts,
                            is_locked, created_at)
                           VALUES (?, ?, ?, ?, 0, 0, ?)""",
                        (email, pw_hash, role, full_name, now_iso),
                    )
    finally:
        conn.close()


def verify_credentials(email: str, password: str) -> Dict:
    """
    Authenticates a user against the persistent SQLite store.
    Handles account lockout after 5 consecutive failed attempts.
    Returns a JWT access token on success.
    """
    # Ensure tables / defaults exist (idempotent)
    init_users_db()
    create_default_users()

    conn = _get_users_conn()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()

        if not row:
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        user = dict(row)

        if user["is_locked"]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account locked due to 5 consecutive failed login attempts.",
            )

        if not check_password(password, user["password_hash"]):
            new_attempts = user["failed_attempts"] + 1
            if new_attempts >= 5:
                conn.execute(
                    """UPDATE users
                       SET failed_attempts = ?, is_locked = 1, locked_at = ?
                       WHERE id = ?""",
                    (new_attempts, datetime.now(timezone.utc).isoformat(), user["id"]),
                )
                conn.commit()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account locked after 5 failed tries.",
                )
            conn.execute(
                "UPDATE users SET failed_attempts = ? WHERE id = ?",
                (new_attempts, user["id"]),
            )
            conn.commit()
            remaining = 5 - new_attempts
            raise HTTPException(
                status_code=401,
                detail=f"Invalid credentials. {remaining} attempt(s) remaining.",
            )

        # Success – reset failures and record login time
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE users SET failed_attempts = 0, last_login = ? WHERE id = ?",
            (now_iso, user["id"]),
        )
        conn.commit()

        token_data = {
            "sub": email,
            "role": user["role"],
            "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS),
        }
        token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
        return {"access_token": token, "token_type": "bearer", "role": user["role"]}
    finally:
        conn.close()


def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None),
) -> Dict:
    """
    FastAPI dependency to extract and validate the JWT session token from
    either the Authorization header (Bearer <token>) or query parameter.
    """
    raw_token = None
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            raw_token = parts[1]
        elif len(parts) == 1:
            raw_token = parts[0]
    elif token:
        raw_token = token

    if not raw_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided.",
        )

    try:
        payload = jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please log in again.",
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


def get_current_operator(user: Dict = Depends(get_current_user)) -> Dict:
    """
    FastAPI dependency that enforces operator role-based access control (RBAC).
    Restricts access to Traffic Operators, EMS Responders, and Administrators.
    """
    role = user.get("role")
    if role not in ("Traffic Operator", "EMS Responder", "Admin", "Administrator"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Operator privileges are required to perform this action.",
        )
    return user


def get_current_admin(user: Dict = Depends(get_current_user)) -> Dict:
    """FastAPI dependency that restricts access to Administrator users only."""
    role = user.get("role")
    if role != "Administrator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Administrator privileges are required.",
        )
    return user


def get_any_authorized_user(user: Dict = Depends(get_current_user)) -> Dict:
    """FastAPI dependency that allows any authenticated user regardless of role."""
    return user


def register_user(
    email: str, password: str, role: str = "Citizen", full_name: str = ""
) -> Dict:
    """Registers a new user in the persistent store (for admin use)."""
    init_users_db()
    conn = _get_users_conn()
    try:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email,)
        ).fetchone()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with email '{email}' already exists.",
            )
        pw_hash = hash_password(password)
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT INTO users
               (email, password_hash, role, full_name, failed_attempts,
                is_locked, created_at)
               VALUES (?, ?, ?, ?, 0, 0, ?)""",
            (email, pw_hash, role, full_name, now_iso),
        )
        conn.commit()
        return {
            "success": True,
            "email": email,
            "role": role,
            "full_name": full_name,
            "created_at": now_iso,
        }
    finally:
        conn.close()
