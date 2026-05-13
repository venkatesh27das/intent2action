"""Image parsing."""

from typing import Any

from intent2action.utils.image_utils import (
    guess_mime_type,
    image_bytes_to_base64,
    resize_image_bytes,
)


class ImageParser:
    """Prepare image input for multimodal model inference."""

    def parse(
        self,
        image_bytes: bytes,
        filename: str,
        context: dict[str, Any] | None = None,
        max_dimension: int | None = 1280,
    ) -> dict[str, Any]:
        """Return normalized image payload."""

        if not image_bytes:
            raise ValueError("Image upload is empty.")
        mime_type = guess_mime_type(filename)
        optimized_image_bytes = resize_image_bytes(image_bytes, mime_type, max_dimension)
        return {
            "image_base64": image_bytes_to_base64(optimized_image_bytes),
            "filename": filename,
            "mime_type": mime_type,
            "context": context or {},
            "original_image_bytes": len(image_bytes),
            "optimized_image_bytes": len(optimized_image_bytes),
        }
