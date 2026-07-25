"""FastAPI application factory."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from veritas.api.routes import router
from veritas.config import env_summary, get_settings
from veritas.logging import configure_logging, get_logger
from veritas.storage.db import get_db

log = get_logger(__name__)

DESCRIPTION = """
Autonomous multi-agent research and fact-verification system.

Every claim in a generated report is decomposed, retrieved against independent
sources, adversarially reviewed, and assigned a **calibrated** confidence score.
`NEI` (not enough information) is a first-class verdict rather than a failure.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    get_db()  # create schema up front so the first request is not the migration
    log.info("VERITAS API starting", **env_summary())
    yield
    log.info("VERITAS API shutting down")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="VERITAS",
        description=DESCRIPTION,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing_middleware(request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Response-Time-ms"] = f"{elapsed_ms:.1f}"
        if elapsed_ms > 2000:
            log.warning(
                "slow request", path=request.url.path, ms=round(elapsed_ms, 1)
            )
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.error(
            "unhandled exception",
            path=request.url.path,
            error=str(exc)[:300],
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "error": str(exc)[:200]},
        )

    app.include_router(router)

    @app.get("/", tags=["system"])
    async def root() -> dict:
        return {
            "name": "VERITAS",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()
