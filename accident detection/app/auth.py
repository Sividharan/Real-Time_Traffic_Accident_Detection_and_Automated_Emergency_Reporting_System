import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import bcrypt
from fastapi import Depends, Header, HTTPException, Query, status
import jwt

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "traffic-system-super-secret-key")
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def check_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

# In-memory user database & lockout tracker
USER_DB = {
    "officer@traffic.org": {
        "password": hash_password("SecurePass123!"),
        "role": "Traffic Operator",
        "failed_attempts": 0,
        "is_locked": False,
    },
    "ems@citygov.org": {
        "password": hash_password("ResponderPass123!"),
        "role": "EMS Responder",
        "failed_attempts": 0,
        "is_locked": False,
    },
    "citizen@public.org": {
        "password": hash_password("CitizenPass123!"),
        "role": "Citizen",
        "failed_attempts": 0,
        "is_locked": False,
    },
}

def verify_credentials(email: str, password: str) -> Dict:
    user = USER_DB.get(email)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    
    if user["is_locked"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account locked due to 5 consecutive failed login attempts."
        )

    if not check_password(password, user["password"]):
        user["failed_attempts"] += 1
        if user["failed_attempts"] >= 5:
            user["is_locked"] = True
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account locked after 5 failed tries."
            )
        remaining = 5 - user["failed_attempts"]
        raise HTTPException(
            status_code=401, 
            detail=f"Invalid credentials. {remaining} attempt(s) remaining."
        )

    user["failed_attempts"] = 0
    token_data = {
        "sub": email,
        "role": user["role"],
        "exp": datetime.now(timezone.utc) + timedelta(hours=8),
    }
    token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

def get_current_user(
    authorization: Optional[str] = Header(None),
    token: Optional[str] = Query(None)
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
            detail="Authentication credentials were not provided."
        )

    try:
        payload = jwt.decode(raw_token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token has expired. Please log in again."
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token."
        )

def get_current_operator(user: Dict = Depends(get_current_user)) -> Dict:
    """
    FastAPI dependency that enforces operator role-based access control (RBAC).
    Restricts access to Traffic Operators, EMS Responders, and Administrators.
    """
    role = user.get("role")
    if role not in ("Traffic Operator", "EMS Responder", "Admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: Operator privileges are required to perform this action."
        )
    return user
