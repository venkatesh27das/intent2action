# intent2action

Turn messy text or images into structured, ranked action candidates.

intent2action is a Python framework for the step before automation: understanding what could be done, what information is missing, how risky each action is, and whether a human should review it. It works with any OpenAI-compatible `/v1/chat/completions` endpoint and returns validated JSON for apps, APIs, agents, dashboards, or review queues.

intent2action never executes actions. It does not send emails, create tickets, update records, trigger deployments, approve payments, or call tools.

## At A Glance

| Capability | What you get |
| --- | --- |
| Text and image inputs | Customer emails, meeting notes, requirements, support messages, UI images, dashboards, forms. |
| Structured JSON | Pydantic-validated entities, intents, action candidates, missing inputs, risk levels, and warnings. |
| Any compatible model endpoint | LM Studio, Ollama, vLLM, llama.cpp servers, OpenAI, Azure-style compatible gateways, or your own `/v1/chat/completions` server. |
| Safe by design | The package infers possible actions only. Execution belongs to your app, policy layer, or human workflow. |

## Real-World Use Cases

| Use case | Example input | Useful output |
| --- | --- | --- |
| Customer support triage | "The invoice total is wrong and the client is upset." | Investigate billing discrepancy, collect invoice ID, draft customer follow-up, flag medium risk. |
| Data and BI operations | "Sales dashboard has been blank since yesterday." | Check dashboard refresh, inspect source query, ask for dashboard URL, rank investigation actions. |
| Product and engineering intake | "Users need SSO with Okta before enterprise rollout." | Break down implementation tasks, identify missing tenant details, classify requirement intent. |
| Meeting follow-up | Notes from a planning call. | Extract owners, decisions, follow-up actions, missing deadlines, and clarifying questions. |
| Image understanding | Image of an error modal, form, dashboard, document, or workflow state. | Infer likely remediation steps from visual context and request missing details. |
| Human approval queues | Any unstructured request before automation. | Produce action candidates with `execution_mode`, `risk_level`, and required inputs for review. |

## Tiny Example

Input:

```text
Client is asking why the sales dashboard is blank since yesterday.
```

Output shape:

```json
{
  "input_type": "text",
  "detected_intents": [
    {"intent": "issue_investigation", "confidence": 0.9}
  ],
  "possible_actions": [
    {
      "action_name": "Investigate dashboard refresh",
      "missing_inputs": ["dashboard_link"],
      "risk_level": "low",
      "execution_mode": "human_approval_required"
    }
  ]
}
```

## Image Example

The repository includes this example sales chart image:

![Sales history chart example](benchmarks/example%20slaes%20plot.png)

Run image inference against it:

```bash
intent2action infer-image "benchmarks/example slaes plot.png" \
  --context '{"domain":"sales_analytics","user_role":"finance_analyst"}'
```

Example output from `google/gemma-4-e4b` through LM Studio:

```json
{
  "input_summary": "A sales history chart showing Total invoiced, Total cashed in (€), and Total cashed in (%) from March 2022 to March 2023. The graph highlights key metrics for February 2023.",
  "input_type": "image",
  "extracted_entities": [
    {
      "name": "Chart Title",
      "value": "Sales history",
      "entity_type": "chart_title",
      "confidence": 1.0
    },
    {
      "name": "Metric: Total invoiced (Feb 2023)",
      "value": "12,014.70 k€",
      "entity_type": "financial_metric",
      "confidence": 0.95
    },
    {
      "name": "Metric: Total cashed in (%) (Feb 2023)",
      "value": "21.26%",
      "entity_type": "percentage_metric",
      "confidence": 0.95
    }
  ],
  "detected_intents": [
    {
      "intent": "Analyze sales trends and performance",
      "confidence": 0.98
    },
    {
      "intent": "Generate a summary report of key findings",
      "confidence": 0.95
    }
  ],
  "possible_actions": [
    {
      "action_name": "Extract raw data table from chart",
      "risk_level": "low",
      "execution_mode": "auto_possible"
    },
    {
      "action_name": "Generate a presentation slide with key insights",
      "risk_level": "low",
      "execution_mode": "draft_only"
    },
    {
      "action_name": "Draft a performance summary email",
      "risk_level": "medium",
      "execution_mode": "human_approval_required"
    }
  ],
  "clarifying_questions": [
    "Are you looking to extract the raw data from this chart for further analysis?",
    "Do you need me to summarize any specific trend or period shown in the graph?"
  ],
  "warnings": []
}
```

The exact wording depends on the configured model, but the response follows the same `ActionInferenceResponse` JSON schema and still does not execute any action.

## Quick Start

Install:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Configure a model endpoint, for example LM Studio on `http://localhost:1234/v1`:

```bash
INTENT2ACTION_BASE_URL=http://localhost:1234/v1
INTENT2ACTION_API_KEY=not-needed
INTENT2ACTION_MODEL=local-model
INTENT2ACTION_SUPPORTS_VISION=true
```

Run text inference:

```bash
intent2action infer --text "Client is asking why the sales dashboard is blank since yesterday."
```

Run image inference:

```bash
intent2action infer-image image.png --context '{"domain":"data_analytics"}'
```

Check non-secret configuration:

```bash
intent2action config
```

## Choose Your Model Setup

| Setup | Base URL | API key | Example model |
| --- | --- | --- | --- |
| LM Studio | `http://localhost:1234/v1` | `not-needed` | `gemma-4-local` |
| Ollama OpenAI-compatible endpoint | `http://localhost:11434/v1` | `not-needed` | `llava:latest` |
| vLLM | `http://localhost:8000/v1` | `not-needed` | `Qwen2.5-VL-7B-Instruct` |
| OpenAI | `https://api.openai.com/v1` | your API key | `gpt-4o-mini` |
| Self-hosted compatible server | your `/v1` URL | depends on server | your model ID |

Full environment example:

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

Legacy LM Studio variables still work:

```bash
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=local-model
```

Example provider YAML files are available in `configs/examples/`.

Remote endpoints can receive the text, images, and context you submit. Use an endpoint whose retention and access controls match your data.

## Common Commands

CLI:

```bash
intent2action version
intent2action config
intent2action infer --text "Break this customer email into possible follow-up actions."
intent2action infer --file examples/text_inputs/customer_email.txt
intent2action infer-image image.png --context '{"user_role":"analyst"}'
```

FastAPI:

```bash
uvicorn intent2action.app.main:app --reload
curl http://localhost:8000/health
```

Streamlit UI:

```bash
streamlit run intent2action/ui/streamlit_app.py
```

Docker:

```bash
docker compose up --build
```

API: `http://localhost:8000`

UI: `http://localhost:8501`

For Docker on macOS or Windows, a host model server usually needs a host-accessible base URL such as `http://host.docker.internal:1234/v1`.

## API Examples

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
  -F "image=@image.png" \
  -F 'context={"domain":"data_analytics","user_role":"business_analyst"}'
```

Model health:

```bash
curl http://localhost:8000/health/model
```

`/health/model` is best-effort. Some OpenAI-compatible servers do not implement `/models`, so a missing models endpoint does not necessarily mean chat completions will fail.

## Example Response

```json
{
  "input_summary": "A client reports that the sales dashboard has been blank since yesterday.",
  "input_type": "text",
  "schema_version": "1.0",
  "package_version": "1.0.0",
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

## Response Fields

| Field | Meaning |
| --- | --- |
| `input_summary` | Short model-generated summary of the submitted text or image. |
| `input_type` | `text` or `image`. |
| `schema_version` | Response schema version for downstream compatibility. |
| `package_version` | intent2action package version that produced the response. |
| `extracted_entities` | Structured entities found in the input. |
| `detected_intents` | Likely intents with confidence and rationale. |
| `possible_actions` | Ranked actions that could be taken. These are not executed. |
| `required_inputs` | Inputs needed before an action can be safely considered. |
| `available_inputs` | Inputs already present in the original content or context. |
| `missing_inputs` | Required inputs not found in the original content or context. |
| `risk_level` | Deterministic low, medium, or high risk classification. |
| `execution_mode` | Recommended handling mode such as `draft_only` or `human_approval_required`. |
| `clarifying_questions` | Questions to ask before proceeding. |
| `warnings` | Safety or validation notes from post-processing. |

## Vision Support

Text-only models can use:

- CLI: `intent2action infer`
- API: `POST /infer-actions`

Image inference requires:

- A multimodal model.
- An endpoint that accepts OpenAI-compatible image message payloads.
- `INTENT2ACTION_SUPPORTS_VISION=true`.

If vision is disabled, the Streamlit UI and provider client return:

```text
The configured model provider does not have vision support enabled.
```

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Endpoint not reachable | Confirm `INTENT2ACTION_BASE_URL` points to the server's `/v1` base URL and the server is running. |
| `curl` works but Python fails to connect | Check whether your terminal, container, notebook, or sandbox blocks Python network access to localhost. Run `python -c "import httpx; print(httpx.get('http://localhost:1234/v1/models').status_code)"` from the same environment that runs intent2action. |
| Model not found | Set `INTENT2ACTION_MODEL` to the exact model ID exposed by your endpoint. |
| API key rejected | Check `INTENT2ACTION_API_KEY`; use `not-needed` only for endpoints that do not require auth. |
| Image inference fails | Confirm the model is multimodal and supports OpenAI-compatible `image_url` message content. |
| Invalid JSON from model | Try a stronger instruction-following model or enable JSON repair in config. |
| Docker cannot reach model server | Use a host-accessible URL such as `http://host.docker.internal:1234/v1` on macOS or Windows. |
| Remote provider sees private data | Choose an endpoint whose data policy matches your privacy requirements. |

## Public API Contract

The v1-stable surface is:

- `POST /infer-actions` for text inference.
- `POST /infer-actions/image` for image inference.
- `GET /health` for service and non-secret provider metadata.
- `GET /health/model` for best-effort provider reachability.
- `ActionInferenceRequest`, `ActionInferenceResponse`, `ActionCandidate`, `DetectedIntent`, and `ExtractedEntity` schemas.
- `OpenAICompatibleClient`, `OpenAICompatibleClientError`, and `get_model_client`.
- `INTENT2ACTION_*` configuration variables. Legacy `LMSTUDIO_BASE_URL` and `LMSTUDIO_MODEL` remain supported for compatibility.

New optional response fields may be added in a future minor release, but existing v1 fields and meanings should remain compatible.

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

The model handles understanding and candidate generation. Deterministic Python handles schema validation, missing-input detection, risk overrides, ranking, deduplication, JSON repair, and execution-safety checks.

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

## Tests And Linting

```bash
pytest
ruff check .
```

Tests do not require a real model server. Provider and pipeline tests use mocked responses.

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

Latest benchmark snapshot against LM Studio on 2026-05-13 using `google/gemma-4-e4b`:

| Metric | Value |
| --- | ---: |
| Text cases | 4 |
| Schema success rate | 100% |
| Mean latency | 22.95s |
| p50 latency | 20.25s |
| p95 latency | 29.85s |
| Min / max latency | 19.83s / 31.49s |
| Mean benchmark score | 0.8969 |

Accuracy note: the benchmark score is a heuristic regression signal. A stronger accuracy claim requires a larger human-labeled evaluation set.

## Release Readiness

`1.0.0` includes:

- OpenAI-compatible provider configuration.
- Text and image FastAPI inference endpoints.
- CLI for text, image, config, and version commands.
- Streamlit UI.
- Strict schemas and deterministic post-processing.
- Unit tests that do not require a real model server.
- CI for Python 3.11 and 3.12.
- Release artifact build workflow.
- Docker and docker-compose support.
- Benchmark harness for live model evaluation.

## Roadmap

- Add richer action ontology metadata.
- Add optional local embedding-based deduplication.
- Add provider adapters for non-OpenAI-compatible runtimes.
- Add larger human-labeled structured evaluation fixtures.

## Contributing

Contributions are welcome. Keep provider configuration generic, avoid hardcoding one hosted service as the only path, and preserve the core rule: infer actions only, never execute them.

## License

Apache-2.0. See [LICENSE](LICENSE).
