"""Image utility functions."""

import base64
import mimetypes
from pathlib import Path


def image_bytes_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes as base64 text."""

    return base64.b64encode(image_bytes).decode("utf-8")


def guess_mime_type(filename: str) -> str:
    """Guess an image MIME type from a filename."""

    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type and mime_type.startswith("image/"):
        return mime_type
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"

