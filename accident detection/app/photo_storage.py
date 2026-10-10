import io
import os
import uuid
from typing import Tuple
from PIL import Image
from fastapi import HTTPException, UploadFile, status

DEFAULT_UPLOADS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uploads"
)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB

def get_uploads_dir() -> str:
    """Returns the uploads directory path, configurable via environment variable."""
    directory = os.getenv("INCIDENT_UPLOADS_DIR", DEFAULT_UPLOADS_DIR)
    os.makedirs(directory, exist_ok=True)
    return directory

def get_max_upload_size() -> int:
    """Returns the maximum allowed upload size in bytes."""
    val = os.getenv("MAX_UPLOAD_SIZE_BYTES")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return DEFAULT_MAX_SIZE_BYTES

def validate_image_bytes(content: bytes, original_filename: str) -> str:
    """
    Validates file extension, magic headers, and image integrity via PIL.
    Returns normalized lower-case extension if valid, or raises HTTPException.
    """
    # 1. Check file size
    max_size = get_max_upload_size()
    if len(content) > max_size:
        max_mb = max_size / (1024 * 1024)
        status_code = getattr(status, "HTTP_413_CONTENT_TOO_LARGE", 413)
        raise HTTPException(
            status_code=status_code,
            detail=f"Uploaded photo exceeds maximum allowed file size of {max_mb:.1f} MB."
        )

    if len(content) < 16:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or too small to be a valid image."
        )

    # 2. Check filename extension
    ext = os.path.splitext(original_filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: JPG, JPEG, PNG, WEBP."
        )

    # 3. Magic bytes inspection
    is_jpeg = content.startswith(b"\xff\xd8\xff")
    is_png = content.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = content.startswith(b"RIFF") and len(content) >= 12 and content[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content does not match a valid image signature."
        )

    # 4. Strict PIL validation to guard against corrupted or crafted files
    try:
        img_buffer = io.BytesIO(content)
        with Image.open(img_buffer) as img:
            img.verify()
            format_name = (img.format or "").upper()
            if format_name not in {"JPEG", "PNG", "WEBP"}:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Image format '{format_name}' is not permitted."
                )
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file contains invalid or corrupted image data."
        )

    return ext

def save_uploaded_photo(content: bytes, original_filename: str) -> Tuple[str, str]:
    """
    Validates and stores the image securely under a random UUID-based filename.
    Returns (safe_filename, relative_storage_path).
    """
    ext = validate_image_bytes(content, original_filename)
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    uploads_dir = get_uploads_dir()
    destination_path = os.path.abspath(os.path.join(uploads_dir, safe_filename))

    # Path traversal safeguard
    if not destination_path.startswith(os.path.abspath(uploads_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Illegal filesystem path detected."
        )

    with open(destination_path, "wb") as f:
        f.write(content)

    return safe_filename, safe_filename

def get_photo_path(safe_filename: str) -> str:
    """
    Resolves safe filename to absolute path and verifies it exists within uploads directory.
    """
    # Prevent traversal in filename
    basename = os.path.basename(safe_filename)
    if basename != safe_filename or ".." in safe_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image reference."
        )

    uploads_dir = get_uploads_dir()
    file_path = os.path.abspath(os.path.join(uploads_dir, safe_filename))

    if not file_path.startswith(os.path.abspath(uploads_dir)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image path."
        )

    if not os.path.isfile(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incident photo file could not be found on storage."
        )

    return file_path
