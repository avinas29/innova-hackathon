# syntax=docker/dockerfile:1.7
#
# Single-image build: Next.js UI + FastAPI API, one process, one port.
#
# The UI is a static export (every page is "use client"), so there is no Node
# runtime in the final image — FastAPI serves the built files directly. That
# gives one Render service, one domain, and no CORS, and keeps the runtime
# image to a Python base rather than Python + Node.
#
# Build:  docker build -t veritas .
# Run:    docker run -p 8000:8000 --env-file .env veritas

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — build the UI to static files
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS frontend

WORKDIR /build
ENV NEXT_TELEMETRY_DISABLED=1

COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci

COPY frontend/ ./

# Forced empty, and deliberately NOT an ARG.
#
# This image always serves the UI and the API from one origin, so relative
# requests are always correct — there is no deployment of this image where a
# fixed API host would be right.
#
# It must not be an ARG because Render (and other PaaS) automatically pass the
# service's environment variables into the Docker build. A dev-only
# NEXT_PUBLIC_API_URL=http://localhost:8000 sitting in the dashboard would then
# be baked into the bundle, and every visitor's browser would be told to call
# *their own machine* — which fails as an opaque "Load failed" plus a CORS
# error that reads like a server fault. Hardcoding empty here removes that
# whole failure mode.
ENV NEXT_PUBLIC_API_URL=""

RUN npm run build && test -f out/index.html

# Guard: never ship a bundle that tells browsers to call localhost. Cheap to
# check, and it turns a silent production breakage into a failed build.
RUN if grep -rq "localhost:8000" out/_next/static/chunks/; then \
      echo "BUILD REJECTED: bundle contains a localhost API URL"; \
      exit 1; \
    fi; \
    echo "bundle verified: no hardcoded API host"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — build the Python environment
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS backend

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
WORKDIR /build

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

COPY backend/pyproject.toml ./
COPY backend/veritas ./veritas

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# All three model SDKs, so one image works with whichever key is supplied.
RUN pip install --upgrade pip && pip install ".[openai,anthropic,gemini]"


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VERITAS_STATIC_DIR=/app/static \
    VERITAS_DB_PATH=/data/veritas.db \
    VERITAS_LOG_JSON=true \
    PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 veritas \
 && mkdir -p /data /app/static \
 && chown -R veritas:veritas /data /app

WORKDIR /app
COPY --from=backend /opt/venv /opt/venv
COPY --chown=veritas:veritas backend/veritas ./veritas
COPY --chown=veritas:veritas backend/pyproject.toml ./
COPY --from=frontend --chown=veritas:veritas /build/out ./static

USER veritas

# No VOLUME declaration. Docker materialises a VOLUME as a fresh anonymous
# mount at container start, which shadows the build-time `chown` and lands
# root-owned — so the unprivileged user cannot create the database and startup
# dies with "unable to open database file". docker-compose binds a named volume
# to /data explicitly, which is the correct place to decide persistence, and
# platforms like Render need no volume at all.
EXPOSE 8000

# Render (and most PaaS) inject $PORT and expect the process to bind it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "exec uvicorn veritas.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
