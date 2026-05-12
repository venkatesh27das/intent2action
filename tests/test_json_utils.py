"""JSON utility tests."""

from intent2action.utils.json_utils import extract_json_text, loads_model_json


def test_extract_json_from_markdown() -> None:
    raw = '```json\n{"hello": "world"}\n```'
    assert extract_json_text(raw) == '{"hello": "world"}'


def test_loads_model_json_repairs_trailing_commas() -> None:
    parsed = loads_model_json('{"items": [1, 2,],}')
    assert parsed == {"items": [1, 2]}

