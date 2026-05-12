"""Pipeline tests with mocked LM Studio responses."""

import json

from intent2action.core.pipeline import ActionInferencePipeline


class FakeLMStudioClient:
    """Fake client for tests."""

    def generate_text(self, messages: list[dict]) -> str:
        return json.dumps(
            {
                "input_summary": "Client reports a blank sales dashboard since yesterday.",
                "input_type": "text",
                "extracted_entities": [
                    {
                        "name": "dashboard",
                        "value": "sales dashboard",
                        "entity_type": "system",
                        "confidence": 0.9,
                    }
                ],
                "detected_intents": [
                    {
                        "intent": "issue_investigation",
                        "confidence": 0.9,
                        "rationale": "A blank dashboard needs investigation.",
                    }
                ],
                "possible_actions": [
                    {
                        "action_name": "Investigate dashboard refresh",
                        "action_type": "investigate_issue",
                        "description": "Review refresh status.",
                        "rationale": "Blank dashboards can be caused by refresh failures.",
                        "required_inputs": ["dashboard_url", "timeframe"],
                        "available_inputs": ["timeframe"],
                        "missing_inputs": [],
                        "suggested_tools": ["BI platform"],
                        "confidence": 0.9,
                        "risk_level": "low",
                        "execution_mode": "human_approval_required",
                        "ranking_score": 0,
                    }
                ],
                "clarifying_questions": ["What is the dashboard link?"],
                "raw_model_output": None,
                "warnings": [],
            }
        )

    def generate_multimodal(self, prompt: str, image_base64: str, mime_type: str) -> str:
        return self.generate_text([])


def test_text_pipeline_with_mocked_model() -> None:
    pipeline = ActionInferencePipeline(llm_client=FakeLMStudioClient())

    response = pipeline.infer_from_text(
        "Client is asking why the sales dashboard is blank since yesterday.",
        {"domain": "data_analytics"},
    )

    assert response.input_type == "text"
    assert response.possible_actions[0].missing_inputs == ["dashboard_url"]
    assert response.possible_actions[0].ranking_score > 0

