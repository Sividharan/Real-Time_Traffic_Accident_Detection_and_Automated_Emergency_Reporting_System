from datetime import datetime, timedelta, timezone
from typing import Dict
import bcrypt
from fastapi import HTTPException, status
import jwt

SECRET_KEY = "traffic-system-super-secret-key"
ALGORITHM = "HS256"

def hash_password(password: str) -> str:
    pwd_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

def check_password(plain_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))

# In-memory mock database & lockout tracker
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