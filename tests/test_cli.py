"""CLI tests."""

import json

from intent2action import cli
from intent2action.schemas import ActionInferenceResponse


class FakePipeline:
    """Fake pipeline for CLI tests."""

    def infer_from_text(self, content: str, context: dict | None = None) -> ActionInferenceResponse:
        return _response("text")

    def infer_from_image(
        self,
        image_bytes: bytes,
        filename: str,
        context: dict | None = None,
    ) -> ActionInferenceResponse:
        return _response("image")


def _response(input_type: str) -> ActionInferenceResponse:
    return ActionInferenceResponse(
        input_summary="summary",
        input_type=input_type,
        extracted_entities=[],
        detected_intents=[],
        possible_actions=[],
        clarifying_questions=[],
        raw_model_output=None,
        warnings=[],
    )


def test_cli_version(capsys) -> None:
    assert cli.main(["version"]) == 0

    assert capsys.readouterr().out.strip() == "1.0.0"


def test_cli_config_hides_api_key(capsys, monkeypatch) -> None:
    monkeypatch.setenv("INTENT2ACTION_API_KEY", "secret")
    cli.get_settings.cache_clear()

    assert cli.main(["config"]) == 0

    output = capsys.readouterr().out
    assert "secret" not in output
    assert "api_key" not in output

    cli.get_settings.cache_clear()


def test_cli_infer_text_outputs_response_json(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli, "ActionInferencePipeline", lambda: FakePipeline())

    assert cli.main(["infer", "--text", "hello"]) == 0

    body = json.loads(capsys.readouterr().out)
    assert body["input_type"] == "text"
    assert body["schema_version"] == "1.0"
