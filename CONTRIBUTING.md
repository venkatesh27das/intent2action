# Contributing

Thanks for helping improve intent2action.

## Development Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Quality Checks

Run these before opening a pull request:

```bash
ruff check .
pytest
```

## Project Principles

- Keep the default path local-first.
- Do not add paid cloud LLM APIs to the core MVP path.
- The system must infer possible actions only. It must not execute actions.
- Add tests for schema, ranking, risk, missing-input, provider, and API behavior when changing those areas.
- Keep modules small and typed.

## Benchmarking

Use the benchmark harness for live model regressions:

```bash
python scripts/benchmark.py --model your-lm-studio-model --runs 1
```

Benchmark scores are heuristic regression signals. Do not present them as ground-truth accuracy without a larger labeled evaluation set.

