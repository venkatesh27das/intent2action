"""Image parsing."""

from typing import Any

from intent2action.utils.image_utils import guess_mime_type, image_bytes_to_base64


class ImageParser:
    """Prepare image input for multimodal model inference."""

    def parse(
        self,
        image_bytes: bytes,
        filename: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return normalized image payload."""

        if not image_bytes:
            raise ValueError("Image upload is empty.")
        return {
            "image_base64": image_bytes_to_base64(image_bytes),
            "filename": filename,
            "mime_type": guess_mime_type(filename),
            "context": context or {},
        }

