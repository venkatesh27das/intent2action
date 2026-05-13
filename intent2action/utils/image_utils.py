"""Image utility functions."""

import base64
import mimetypes
from io import BytesIO
from pathlib import Path

from PIL import Image


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


def resize_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    max_dimension: int | None,
) -> bytes:
    """Resize an image if either dimension exceeds max_dimension."""

    if not max_dimension or max_dimension <= 0:
        return image_bytes

    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        largest_dimension = max(width, height)
        if largest_dimension <= max_dimension:
            return image_bytes

        scale = max_dimension / largest_dimension
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.LANCZOS,
        )
        buffer = BytesIO()
        resized.save(buffer, format=_image_format(mime_type))
        return buffer.getvalue()


def _image_format(mime_type: str) -> str:
    if mime_type == "image/jpeg":
        return "JPEG"
    if mime_type == "image/webp":
        return "WEBP"
    return "PNG"
