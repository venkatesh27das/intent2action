"""Benchmark live intent2action inference against an OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw

from intent2action.app.config import Settings
from intent2action.core.pipeline import ActionInferencePipeline


@dataclass(frozen=True)
class BenchmarkCase:
    """A single text benchmark case."""

    case_id: str
    input_text: str
    context: dict[str, Any]
    min_actions: int
    expected_intent_keywords: list[list[str]]
    expected_action_keywords: list[list[str]]
    input_type: str = "text"
    image_text: str | None = None


def load_cases(path: Path) -> list[BenchmarkCase]:
    """Load benchmark cases from YAML."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = []
    for item in data.get("cases", []):
        cases.append(
            BenchmarkCase(
                case_id=item["id"],
                input_text=item.get("input", item.get("image_text", "")),
                context=item.get("context", {}),
                min_actions=item.get("min_actions", 1),
                expected_intent_keywords=item.get("expected_intent_keywords", []),
                expected_action_keywords=item.get("expected_action_keywords", []),
                input_type=item.get("input_type", "text"),
                image_text=item.get("image_text"),
            )
        )
    return cases


def load_all_cases(text_path: Path, image_path: Path | None) -> list[BenchmarkCase]:
    """Load text cases plus optional synthetic image cases."""

    cases = load_cases(text_path)
    if image_path and image_path.exists():
        image_cases = []
        for case in load_cases(image_path):
            image_cases.append(
                BenchmarkCase(
                    case_id=case.case_id,
                    input_text=case.input_text,
                    context=case.context,
                    min_actions=case.min_actions,
                    expected_intent_keywords=case.expected_intent_keywords,
                    expected_action_keywords=case.expected_action_keywords,
                    input_type="image",
                    image_text=case.image_text or case.input_text,
                )
            )
        cases.extend(image_cases)
    return cases


def normalize(value: str) -> str:
    """Normalize text for loose benchmark matching."""

    return " ".join(value.lower().replace("_", " ").split())


def any_keyword_match(text: str, alternatives: list[str]) -> bool:
    """Return true if any alternative appears in text."""

    normalized_text = normalize(text)
    return any(normalize(keyword) in normalized_text for keyword in alternatives)


def coverage_score(text: str, expected_groups: list[list[str]]) -> tuple[float, list[list[str]]]:
    """Measure expected keyword-group coverage."""

    if not expected_groups:
        return 1.0, []
    misses = [group for group in expected_groups if not any_keyword_match(text, group)]
    return (len(expected_groups) - len(misses)) / len(expected_groups), misses


def has_execution_claim(response_text: str) -> bool:
    """Detect wording that could imply execution rather than inference."""

    unsafe_phrases = [
        "has been sent",
        "was sent",
        "ticket created",
        "created the ticket",
        "email sent",
        "updated the record",
        "deleted",
        "approved payment",
        "triggered deployment",
    ]
    return any(phrase in normalize(response_text) for phrase in unsafe_phrases)


def percentile(values: list[float], pct: float) -> float:
    """Return a percentile for a non-empty list."""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def evaluate_case(
    pipeline: ActionInferencePipeline,
    case: BenchmarkCase,
    run_index: int,
) -> dict[str, Any]:
    """Run and score a single case."""

    started = time.perf_counter()
    try:
        if case.input_type == "image":
            response = pipeline.infer_from_image(
                image_bytes=render_text_image(case.image_text or case.input_text),
                filename=f"{case.case_id}.png",
                context=case.context,
            )
        else:
            response = pipeline.infer_from_text(case.input_text, case.context)
        latency_seconds = time.perf_counter() - started
        response_text = response.model_dump_json()
        actions_text = " ".join(
            " ".join(
                [
                    action.action_name,
                    action.action_type,
                    action.description,
                    action.rationale,
                    " ".join(action.missing_inputs),
                    " ".join(action.suggested_tools),
                ]
            )
            for action in response.possible_actions
        )
        intents_text = " ".join(
            f"{intent.intent} {intent.rationale}" for intent in response.detected_intents
        )
        action_coverage, missed_actions = coverage_score(
            actions_text,
            case.expected_action_keywords,
        )
        intent_coverage, missed_intents = coverage_score(
            intents_text,
            case.expected_intent_keywords,
        )
        min_action_score = 1.0 if len(response.possible_actions) >= case.min_actions else 0.0
        sorted_score = 1.0 if _is_ranked(response.possible_actions) else 0.0
        missing_input_score = (
            1.0
            if any(action.missing_inputs for action in response.possible_actions)
            or all(not action.required_inputs for action in response.possible_actions)
            else 0.0
        )
        safety_score = 0.0 if has_execution_claim(response_text) else 1.0
        benchmark_score = round(
            action_coverage * 0.35
            + intent_coverage * 0.15
            + min_action_score * 0.15
            + sorted_score * 0.10
            + missing_input_score * 0.10
            + safety_score * 0.15,
            4,
        )
        return {
            "case_id": case.case_id,
            "input_type": case.input_type,
            "run": run_index,
            "ok": True,
            "latency_seconds": round(latency_seconds, 4),
            "actions_count": len(response.possible_actions),
            "intents_count": len(response.detected_intents),
            "entities_count": len(response.extracted_entities),
            "action_coverage": round(action_coverage, 4),
            "intent_coverage": round(intent_coverage, 4),
            "benchmark_score": benchmark_score,
            "missed_action_keyword_groups": missed_actions,
            "missed_intent_keyword_groups": missed_intents,
            "top_action": response.possible_actions[0].action_name
            if response.possible_actions
            else None,
        }
    except Exception as exc:
        return {
            "case_id": case.case_id,
            "input_type": case.input_type,
            "run": run_index,
            "ok": False,
            "latency_seconds": round(time.perf_counter() - started, 4),
            "error": str(exc),
        }


def _is_ranked(actions: list[Any]) -> bool:
    scores = [action.ranking_score for action in actions]
    return scores == sorted(scores, reverse=True)


def render_text_image(text: str) -> bytes:
    """Render a simple benchmark PNG with text."""

    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    draw.text((32, 96), text, fill="black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize benchmark results."""

    latencies = [item["latency_seconds"] for item in results if item["ok"]]
    scores = [item["benchmark_score"] for item in results if item["ok"]]
    return {
        "total_runs": len(results),
        "successful_runs": sum(1 for item in results if item["ok"]),
        "schema_success_rate": round(
            sum(1 for item in results if item["ok"]) / len(results),
            4,
        )
        if results
        else 0.0,
        "latency_seconds": {
            "mean": round(statistics.mean(latencies), 4) if latencies else None,
            "p50": round(percentile(latencies, 0.50), 4) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 4) if latencies else None,
            "min": round(min(latencies), 4) if latencies else None,
            "max": round(max(latencies), 4) if latencies else None,
        },
        "benchmark_score": {
            "mean": round(statistics.mean(scores), 4) if scores else None,
            "min": round(min(scores), 4) if scores else None,
            "max": round(max(scores), 4) if scores else None,
        },
    }


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default="benchmarks/text_cases.yaml")
    parser.add_argument("--image-cases", default="benchmarks/image_cases.yaml")
    parser.add_argument("--include-images", action="store_true")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--base-url",
        default=os.getenv(
            "INTENT2ACTION_BASE_URL",
            os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("INTENT2ACTION_MODEL", os.getenv("LMSTUDIO_MODEL", "local-model")),
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    settings = Settings(
        model_base_url=args.base_url,
        model_name=args.model,
        model_timeout_seconds=args.timeout,
    )
    pipeline = ActionInferencePipeline(settings=settings)
    image_cases = Path(args.image_cases) if args.include_images else None
    cases = load_all_cases(Path(args.cases), image_cases)

    results: list[dict[str, Any]] = []
    for run_index in range(1, args.runs + 1):
        for case in cases:
            result = evaluate_case(pipeline, case, run_index)
            results.append(result)
            status = "ok" if result["ok"] else "failed"
            print(
                f"{case.case_id} run={run_index} status={status} "
                f"latency={result['latency_seconds']}s "
                f"score={result.get('benchmark_score')}"
            )

    report = {
        "model": args.model,
        "base_url": args.base_url,
        "cases_file": args.cases,
        "image_cases_file": args.image_cases if args.include_images else None,
        "summary": summarize(results),
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(item["ok"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
