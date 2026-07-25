# syntax=docker/dockerfile:1.7
# Multi-stage build: wheels are compiled once in the builder and copied into a
# slim runtime, keeping the final image free of compilers and build headers.

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY veritas ./veritas

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Model SDKs are installed here rather than as base dependencies so the image
# works with either provider out of the box.
RUN pip install --upgrade pip \
 && pip install ".[openai,anthropic]"


FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VERITAS_DB_PATH=/data/veritas.db \
    VERITAS_LOG_JSON=true

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 veritas \
 && mkdir -p /data \
 && chown veritas:veritas /data

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=veritas:veritas veritas ./veritas
COPY --chown=veritas:veritas pyproject.toml ./

USER veritas
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "veritas.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
