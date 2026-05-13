"""Command line interface for intent2action."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from intent2action import __version__
from intent2action.app.config import get_settings
from intent2action.core.pipeline import ActionInferencePipeline
from intent2action.providers.openai_compatible_client import OpenAICompatibleClientError


def main(argv: list[str] | None = None) -> int:
    """Run the intent2action CLI."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "version":
        print(__version__)
        return 0
    if args.command == "config":
        return _print_config()
    if args.command == "infer":
        return _infer_text(args)
    if args.command == "infer-image":
        return _infer_image(args)

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="intent2action",
        description="Infer structured action candidates without executing actions.",
    )
    subparsers = parser.add_subparsers(dest="command")

    infer = subparsers.add_parser("infer", help="Infer actions from text")
    text_source = infer.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text", help="Text content to analyze")
    text_source.add_argument("--file", type=Path, help="Path to a UTF-8 text file")
    infer.add_argument("--context", default=None, help="Optional JSON object with context")

    infer_image = subparsers.add_parser("infer-image", help="Infer actions from an image")
    infer_image.add_argument("image", type=Path, help="Path to a PNG, JPEG, or WEBP image")
    infer_image.add_argument("--context", default=None, help="Optional JSON object with context")

    subparsers.add_parser("config", help="Print effective non-secret model configuration")
    subparsers.add_parser("version", help="Print package version")
    return parser


def _infer_text(args: argparse.Namespace) -> int:
    content = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    context = _parse_context(args.context)
    try:
        response = ActionInferencePipeline().infer_from_text(content, context)
    except OpenAICompatibleClientError as exc:
        _print_error(exc)
        return 1
    print(response.model_dump_json(indent=2))
    return 0


def _infer_image(args: argparse.Namespace) -> int:
    context = _parse_context(args.context)
    try:
        response = ActionInferencePipeline().infer_from_image(
            image_bytes=args.image.read_bytes(),
            filename=args.image.name,
            context=context,
        )
    except OpenAICompatibleClientError as exc:
        _print_error(exc)
        return 1
    print(response.model_dump_json(indent=2))
    return 0


def _print_config() -> int:
    settings = get_settings()
    config = {
        "model_provider": {
            "type": settings.model_provider_type,
            "base_url": settings.model_base_url,
            "model": settings.model_name,
            "timeout_seconds": settings.model_timeout_seconds,
            "max_retries": settings.model_max_retries,
            "supports_vision": settings.model_supports_vision,
        },
        "inference": {
            "max_actions": settings.max_actions,
            "min_confidence": settings.min_confidence,
            "enable_json_repair": settings.enable_json_repair,
            "enable_risk_override": settings.enable_risk_override,
        },
    }
    print(json.dumps(config, indent=2))
    return 0


def _parse_context(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise SystemExit("--context must be a JSON object")
    return parsed


def _print_error(exc: OpenAICompatibleClientError) -> None:
    payload = {
        "error": exc.code,
        "message": str(exc),
    }
    print(json.dumps(payload, indent=2), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
