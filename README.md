# intent2action

intent2action is an open-source, local-first Python framework that converts text and image inputs into structured, ranked action candidates.

It is built for developers who want a clean action inference layer in front of downstream agents, automation systems, or human review workflows. It calls text or multimodal models through any OpenAI-compatible chat completions endpoint and returns validated JSON.

This project does not execute actions. It only infers possible actions.

## What It Does

- Accepts text or image input.
- Extracts entities and detects likely intents.
- Generates ranked action candidates.
- Identifies available and missing inputs.
- Scores risk and recommends an execution mode.
- Returns strict Pydantic v2 JSON for downstream systems.

## What It Does Not Do

- It does not send emails.
- It does not create tickets.
- It does not update records.
- It does not trigger deployments, approvals, payments, or tools.
- It does not execute any action proposed by the model.

## Why It Exists

Many automation systems jump too quickly from input to execution. intent2action creates a local-first reasoning boundary: it tells you what could be done, what information is missing, how risky each action is, and whether a human should review it.

## Architecture

```text
Input
  -> Input Classifier
  -> Text/Image Parser
  -> Entity Extractor
  -> Intent Detector
  -> Action Candidate Generator
  -> Risk Scorer
  -> Missing Input Detector
  -> Action Ranker
  -> JSON Response Validator
```

The LLM/VLM handles understanding and candidate generation. Deterministic Python handles schema validation, missing-input detection, risk overrides, ranking, deduplication, and JSON repair.

## Installation

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Copy the example environment file if you want local overrides:

```bash
cp .env.example .env
```

You can also use the CLI after installation:

```bash
intent2action version
intent2action config
intent2action infer --text "Client is asking why the dashboard is blank."
intent2action infer-image screenshot.png --context '{"domain":"data_analytics"}'
```

## Model Provider Configuration

intent2action is local-first by default and supports any OpenAI-compatible endpoint that exposes `/v1/chat/completions`, including LM Studio, Ollama's OpenAI-compatible endpoint, vLLM, llama.cpp servers, compatible TGI deployments, OpenAI, and self-hosted servers.

Generic configuration:

```bash
INTENT2ACTION_BASE_URL=http://localhost:1234/v1
INTENT2ACTION_API_KEY=not-needed
INTENT2ACTION_MODEL=local-model
INTENT2ACTION_TIMEOUT_SECONDS=120
INTENT2ACTION_MAX_RETRIES=2
INTENT2ACTION_SUPPORTS_VISION=true
```

Environment priority is:

1. `INTENT2ACTION_*` variables
2. legacy `LMSTUDIO_*` variables
3. `configs/default.yaml`

LM Studio example:

```bash
INTENT2ACTION_BASE_URL=http://localhost:1234/v1
INTENT2ACTION_API_KEY=not-needed
INTENT2ACTION_MODEL=gemma-4-local
```

Ollama OpenAI-compatible endpoint example:

```bash
INTENT2ACTION_BASE_URL=http://localhost:11434/v1
INTENT2ACTION_API_KEY=not-needed
INTENT2ACTION_MODEL=llava:latest
```

vLLM example:

```bash
INTENT2ACTION_BASE_URL=http://localhost:8000/v1
INTENT2ACTION_API_KEY=not-needed
INTENT2ACTION_MODEL=Qwen2.5-VL-7B-Instruct
```

OpenAI example:

```bash
INTENT2ACTION_BASE_URL=https://api.openai.com/v1
INTENT2ACTION_API_KEY=<your_api_key>
INTENT2ACTION_MODEL=gpt-4o-mini
```

Legacy LM Studio variables are still supported:

```bash
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
```

Vision support depends on the model and endpoint. Text-only models can still use `/infer-actions`. `/infer-actions/image` requires a multimodal model and an endpoint that supports the OpenAI-compatible image message format.

For Docker on macOS or Windows, a host model server usually needs to be reachable from containers at an address such as `http://host.docker.internal:1234/v1`.

Example provider YAML files are available in `configs/examples/` for LM Studio, Ollama, vLLM, OpenAI, and text-only local models.

Remote endpoints can receive the text, images, and context you submit. Keep the default local endpoint for private data unless you have reviewed the remote provider's retention and access controls.

## Run The API

```bash
uvicorn intent2action.app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Text inference:

```bash
curl -X POST http://localhost:8000/infer-actions \
  -H "Content-Type: application/json" \
  -d '{
    "input_type": "text",
    "content": "Client is asking why the sales dashboard is blank since yesterday.",
    "context": {
      "domain": "data_analytics",
      "user_role": "business_analyst"
    }
  }'
```

Image inference:

```bash
curl -X POST http://localhost:8000/infer-actions/image \
  -F "image=@screenshot.png" \
  -F 'context={"domain":"data_analytics","user_role":"business_analyst"}'
```

## Run The Streamlit UI

```bash
streamlit run intent2action/ui/streamlit_app.py
```

## Example Response

```json
{
  "input_summary": "A client reports that the sales dashboard has been blank since yesterday.",
  "input_type": "text",
  "extracted_entities": [],
  "detected_intents": [],
  "possible_actions": [
    {
      "action_name": "Investigate dashboard refresh",
      "action_type": "investigate_issue",
      "description": "Review dashboard refresh status and recent failures.",
      "rationale": "A blank dashboard often indicates a refresh or query failure.",
      "required_inputs": ["dashboard_link", "timeframe"],
      "available_inputs": ["timeframe"],
      "missing_inputs": ["dashboard_link"],
      "suggested_tools": ["BI platform", "dashboard logs"],
      "confidence": 0.88,
      "risk_level": "low",
      "execution_mode": "human_approval_required",
      "ranking_score": 0.59
    }
  ],
  "clarifying_questions": ["What is the dashboard link?"],
  "raw_model_output": null,
  "warnings": []
}
```

## Project Structure

```text
intent2action/
  app/          FastAPI app and config
  cli.py        Command line interface
  core/         Pipeline steps and deterministic scoring
  providers/    OpenAI-compatible model clients
  schemas/      Pydantic v2 request and response models
  registry/     Action ontology, intent taxonomy, risk rules
  prompts/      System and task prompts
  ui/           Streamlit app
  utils/        JSON, image, and logging helpers
examples/       Sample inputs and expected output shape
tests/          Unit tests with mocked model responses
```

## Docker

```bash
docker compose up --build
```

API: `http://localhost:8000`

UI: `http://localhost:8501`

A model server is assumed to run separately on the host machine unless you configure a remote endpoint. Set `INTENT2ACTION_BASE_URL` to a host-accessible address if needed.

## Public API Contract

The v1-stable surface is:

- `POST /infer-actions` for text inference.
- `POST /infer-actions/image` for image inference.
- `GET /health` for service and non-secret provider metadata.
- `GET /health/model` for best-effort provider reachability.
- `ActionInferenceRequest`, `ActionInferenceResponse`, `ActionCandidate`, `DetectedIntent`, and `ExtractedEntity` schemas.
- `OpenAICompatibleClient`, `OpenAICompatibleClientError`, and `get_model_client`.
- `INTENT2ACTION_*` configuration variables. Legacy `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL` remain supported for compatibility.

Response objects include `schema_version` and `package_version` metadata. New optional response fields may be added in a future minor release, but existing v1 fields and meanings should remain compatible.

## Tests And Linting

```bash
pytest
ruff check .
```

Tests do not require a real model server. The provider and pipeline tests use mocked responses.

CI runs linting and tests on Python 3.11 and 3.12. Tagged releases build distribution artifacts through the release workflow.

## Benchmarking

Run the live benchmark against an OpenAI-compatible endpoint:

```bash
python scripts/benchmark.py --model google/gemma-4-e4b --runs 1 --output reports/benchmark_report.json
```

Include the synthetic image benchmark case:

```bash
python scripts/benchmark.py \
  --model google/gemma-4-e4b \
  --include-images \
  --runs 1 \
  --output reports/benchmark_report.json
```

The benchmark reports schema success rate, latency mean/p50/p95/min/max, action coverage, intent coverage, action count, top action, and a weighted benchmark score. The score is a lightweight regression metric based on keyword coverage and safety checks; it is not a substitute for a larger human-labeled evaluation set.

Latest local benchmark snapshot against LM Studio on 2026-05-13 using `google/gemma-4-e4b`:

| Metric | Value |
| --- | ---: |
| Text cases | 4 |
| Schema success rate | 100% |
| Mean latency | 22.95s |
| p50 latency | 20.25s |
| p95 latency | 29.85s |
| Min / max latency | 19.83s / 31.49s |
| Mean benchmark score | 0.8969 |

Per-case text scores from that run:

| Case | Score | Latency |
| --- | ---: | ---: |
| dashboard_issue | 1.0000 | 19.83s |
| meeting_notes | 1.0000 | 19.90s |
| requirement_breakdown | 0.7500 | 31.49s |
| customer_email | 0.8375 | 20.60s |

Accuracy note: the benchmark score is a heuristic regression signal. It checks schema validity, expected intent/action keyword coverage, ranking order, missing-input behavior, and execution-safety wording. A stronger accuracy claim requires a larger human-labeled evaluation set.

## Release Readiness

`1.0.0` includes:

- Local-first OpenAI-compatible provider.
- Text and image FastAPI inference endpoints.
- CLI for text, image, config, and version commands.
- Streamlit UI.
- Strict schemas and deterministic post-processing.
- Unit tests that do not require a real model server.
- CI for Python 3.11 and 3.12.
- Release artifact build workflow.
- Docker and docker-compose support.
- Benchmark harness for live local model evaluation.

## Roadmap

- Add richer action ontology metadata.
- Add optional local embedding-based deduplication.
- Add provider adapters for non-OpenAI-compatible runtimes.
- Add larger human-labeled structured evaluation fixtures.

## Contributing

Contributions are welcome. Keep the project local-first, avoid paid external APIs in the default path, and preserve the core rule: infer actions only, never execute them.

## License

Apache-2.0. See [LICENSE](LICENSE).
