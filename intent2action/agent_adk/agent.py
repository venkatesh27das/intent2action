"""Google ADK agent and A2A server entrypoint for intent2action."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import uvicorn

from intent2action.core.pipeline import ActionInferencePipeline

try:
    from google.adk.agents import Agent
except ImportError:  # pragma: no cover - optional ADK extra.
    Agent = None  # type: ignore[assignment]


AGENT_INSTRUCTION = """
You are intent2action, an action inference agent.

Your job is to convert text or images into structured, ranked action candidates.
You must use the provided tools for inference. You must never execute actions,
send messages, create tickets, update records, approve payments, trigger
deployments, or call operational tools.

Return the structured tool result as the final answer. If a user asks you to
perform an action, infer the possible action and clearly preserve the
inference-only boundary.
""".strip()


def infer_actions_from_text(content: str, context_json: str = "{}") -> dict[str, Any]:
    """Infer possible actions from text without executing them."""

    context = _parse_context(context_json)
    response = ActionInferencePipeline().infer_from_text(content, context)
    return response.model_dump(mode="json")


def infer_actions_from_image_base64(
    image_base64: str,
    filename: str = "image.png",
    context_json: str = "{}",
) -> dict[str, Any]:
    """Infer possible actions from a base64-encoded image without executing them."""

    context = _parse_context(context_json)
    image_bytes = base64.b64decode(image_base64)
    response = ActionInferencePipeline().infer_from_image(
        image_bytes=image_bytes,
        filename=filename,
        context=context,
    )
    return response.model_dump(mode="json")


def create_agent() -> object:
    """Create the ADK root agent."""

    if Agent is None:
        raise RuntimeError(
            "Google ADK is not installed. Install with `pip install -e '.[agent]'`."
        )

    return Agent(
        name="intent2action_agent",
        model="gemini-2.0-flash",
        description=(
            "Infers structured action candidates from text or images without executing them."
        ),
        instruction=AGENT_INSTRUCTION,
        tools=[infer_actions_from_text, infer_actions_from_image_base64],
    )


root_agent = create_agent() if Agent is not None else None


def create_a2a_app(port: int = 8010) -> object:
    """Create an A2A ASGI app from the ADK root agent."""

    if root_agent is None:
        raise RuntimeError(
            "Google ADK is not installed. Install with `pip install -e '.[agent]'`."
        )

    try:
        from google.adk.a2a.utils.agent_to_a2a import to_a2a
    except ImportError as exc:  # pragma: no cover - optional ADK extra.
        raise RuntimeError(
            "Google ADK A2A support is not installed. Install with "
            "`pip install -e '.[agent]'`."
        ) from exc

    agent_card = Path(__file__).with_name("agent_card.json")
    return to_a2a(root_agent, port=port, agent_card=str(agent_card))


def main(argv: list[str] | None = None) -> int:
    """Run the ADK A2A agent server."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    app = create_a2a_app(port=args.port)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


def _parse_context(context_json: str) -> dict[str, Any] | None:
    if not context_json.strip():
        return None
    parsed = json.loads(context_json)
    if not isinstance(parsed, dict):
        raise ValueError("context_json must decode to a JSON object")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
