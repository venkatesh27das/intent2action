# Security Policy

## Local-First Model Usage

intent2action is designed to run against local models through LM Studio's OpenAI-compatible API. The MVP does not integrate cloud LLM providers.

Users are responsible for the model they load into LM Studio and for any data they submit to that local model.

## Action Safety Boundary

intent2action must not execute actions. It does not send emails, create tickets, update records, approve payments, delete data, trigger deployments, or call operational tools. It only returns structured action candidates for downstream human or system review.

## Reporting Security Issues

Please open a private security advisory or contact the maintainers before publishing details of a vulnerability. Include reproduction steps, expected impact, and affected versions when possible.

