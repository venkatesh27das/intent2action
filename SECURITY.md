# Security Policy

## Model Provider Usage

intent2action is local-first by default and can run against any OpenAI-compatible chat completions endpoint, including local runtimes and remote APIs.

Users are responsible for the model endpoint they configure and for any data they submit to that endpoint. Remote providers may receive text, images, context, prompts, and generated outputs. Do not send sensitive data to a remote endpoint unless you have reviewed its retention and access controls.

API keys must not be logged, shown in the UI, or returned by health endpoints.

## Action Safety Boundary

intent2action must not execute actions. It does not send emails, create tickets, update records, approve payments, delete data, trigger deployments, or call operational tools. It only returns structured action candidates for downstream human or system review.

## Reporting Security Issues

Please open a private security advisory or contact the maintainers before publishing details of a vulnerability. Include reproduction steps, expected impact, and affected versions when possible.
