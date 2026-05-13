"""End-to-end action inference pipeline."""

from pathlib import Path
from typing import Any

from intent2action.app.config import Settings, get_settings
from intent2action.core.action_generator import ActionGenerator
from intent2action.core.action_ranker import ActionRanker
from intent2action.core.entity_extractor import EntityExtractor
from intent2action.core.image_parser import ImageParser
from intent2action.core.input_classifier import InputClassifier
from intent2action.core.intent_detector import IntentDetector
from intent2action.core.missing_input_detector import MissingInputDetector
from intent2action.core.output_validator import OutputValidator
from intent2action.core.risk_scorer import RiskScorer
from intent2action.core.text_parser import TextParser
from intent2action.providers.factory import get_model_client
from intent2action.providers.openai_compatible_client import OpenAICompatibleClient
from intent2action.schemas.response import ActionInferenceResponse

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
EXECUTION_CLAIM_PHRASES = (
    "has been sent",
    "was sent",
    "sent the",
    "ticket created",
    "created the ticket",
    "email sent",
    "updated the record",
    "deleted",
    "approved payment",
    "triggered deployment",
    "executed",
)


def load_prompt(name: str) -> str:
    """Load a prompt template from package prompts."""

    return (PROMPT_DIR / name).read_text(encoding="utf-8")


class ActionInferencePipeline:
    """Infer possible actions from text or image input without executing anything."""

    def __init__(
        self,
        llm_client: OpenAICompatibleClient | Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_client = llm_client or get_model_client(self.settings)
        self.input_classifier = InputClassifier()
        self.text_parser = TextParser()
        self.image_parser = ImageParser()
        self.entity_extractor = EntityExtractor()
        self.intent_detector = IntentDetector()
        self.action_generator = ActionGenerator()
        self.missing_input_detector = MissingInputDetector()
        self.risk_scorer = RiskScorer(enable_override=self.settings.enable_risk_override)
        self.action_ranker = ActionRanker()
        self.output_validator = OutputValidator(
            llm_client=self.llm_client,
            enable_repair=self.settings.enable_json_repair,
        )

    def infer_from_text(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> ActionInferenceResponse:
        """Infer action candidates from text."""

        input_type = self.input_classifier.classify("text")
        parsed = self.text_parser.parse(content, context)
        prompt = self._build_text_prompt(parsed)
        raw = self.llm_client.generate_text(
            [
                {"role": "system", "content": load_prompt("system_prompt.md")},
                {"role": "user", "content": prompt},
            ]
        )
        response = self.output_validator.validate(raw)
        return self._post_process_response(response, input_type, raw)

    def infer_from_image(
        self,
        image_bytes: bytes,
        filename: str,
        context: dict[str, Any] | None = None,
    ) -> ActionInferenceResponse:
        """Infer action candidates from an image."""

        input_type = self.input_classifier.classify("image")
        parsed = self.image_parser.parse(image_bytes, filename, context)
        prompt = self._build_image_prompt(parsed)
        raw = self.llm_client.generate_multimodal(
            prompt=prompt,
            image_base64=parsed["image_base64"],
            mime_type=parsed["mime_type"],
        )
        response = self.output_validator.validate(raw)
        return self._post_process_response(response, input_type, raw)

    def _post_process_response(
        self,
        response: ActionInferenceResponse,
        input_type: str,
        raw_output: str,
    ) -> ActionInferenceResponse:
        entities = self.entity_extractor.extract(
            [entity.model_dump() for entity in response.extracted_entities]
        )
        intents = self.intent_detector.detect(
            [intent.model_dump() for intent in response.detected_intents]
        )
        actions = self.action_generator.generate(
            [action.model_dump(mode="json") for action in response.possible_actions]
        )

        processed = []
        seen: set[str] = set()
        for action in actions:
            key = action.action_name.strip().lower()
            if key in seen or action.confidence < self.settings.min_confidence:
                continue
            seen.add(key)
            action = self.missing_input_detector.detect(action)
            action = self.risk_scorer.score(action)
            processed.append(action)

        ranked = self.action_ranker.rank(processed)[: self.settings.max_actions]
        warnings = list(response.warnings)
        if any(_contains_execution_claim(action) for action in ranked):
            warnings.append(
                "Model output was checked for execution claims; actions are inference only."
            )

        return response.model_copy(
            update={
                "input_type": input_type,
                "extracted_entities": entities,
                "detected_intents": intents,
                "possible_actions": ranked,
                "raw_model_output": raw_output,
                "warnings": warnings,
            }
        )

    @staticmethod
    def _build_text_prompt(parsed: dict[str, Any]) -> str:
        return (
            f"{load_prompt('action_generation.md')}\n\n"
            f"Input type: text\n"
            f"Content:\n{parsed['content']}\n\n"
            f"Context:\n{parsed['context']}"
        )

    @staticmethod
    def _build_image_prompt(parsed: dict[str, Any]) -> str:
        return (
            f"{load_prompt('action_generation.md')}\n\n"
            "Input type: image\n"
            f"Filename: {parsed['filename']}\n"
            f"Context:\n{parsed['context']}\n\n"
            "Inspect the image and infer structured action candidates."
        )


def _contains_execution_claim(action: Any) -> bool:
    text = " ".join(
        [
            getattr(action, "action_name", ""),
            getattr(action, "description", ""),
            getattr(action, "rationale", ""),
        ]
    ).lower()
    return any(phrase in text for phrase in EXECUTION_CLAIM_PHRASES)
