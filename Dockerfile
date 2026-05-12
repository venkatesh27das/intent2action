FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md LICENSE ./
COPY configs ./configs
COPY intent2action ./intent2action

RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "intent2action.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

