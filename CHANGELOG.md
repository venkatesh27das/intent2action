# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0 - 2026-05-13

Initial MVP release.

- Added FastAPI endpoints for health, text inference, and image inference.
- Added local LM Studio OpenAI-compatible provider client.
- Added strict Pydantic v2 schemas for action inference requests and responses.
- Added modular inference pipeline with deterministic validation, missing-input detection, risk override, ranking, deduplication, and JSON extraction/repair.
- Added Streamlit UI for local text and image inference.
- Added prompts, action ontology, intent taxonomy, and risk rules.
- Added benchmark harness for live LM Studio evaluation.
- Added Docker and docker-compose support.
- Added unit tests and GitHub Actions CI.

