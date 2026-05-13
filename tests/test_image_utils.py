"""Image utility tests."""

from io import BytesIO

from PIL import Image

from intent2action.core.image_parser import ImageParser
from intent2action.utils.image_utils import resize_image_bytes


def _png(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_resize_image_bytes_downscales_large_image() -> None:
    image_bytes = _png(200, 100)

    resized = resize_image_bytes(image_bytes, "image/png", max_dimension=100)

    with Image.open(BytesIO(resized)) as image:
        assert image.size == (100, 50)


def test_image_parser_reports_optimized_size() -> None:
    parsed = ImageParser().parse(
        _png(200, 100),
        "chart.png",
        max_dimension=100,
    )

    assert parsed["mime_type"] == "image/png"
    assert parsed["original_image_bytes"] >= parsed["optimized_image_bytes"]
    assert parsed["image_base64"]
