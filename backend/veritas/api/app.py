"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

    # Wake the search backend in the background.
    #
    # Both services sleep on free hosting. This one is usually free: the API
    # itself only starts because someone loaded the page, and they will spend
    # far longer reading it than SearXNG needs to boot. Deliberately not
    # awaited — the platform health-checks this process, and blocking startup
    # on a third-party wake would fail the deploy.
    # `offline` is the hard network kill-switch the test suite sets. Spawning a
    # 90-second polling task here regardless left one lingering per API test,
    # which hung teardown until pytest was SIGKILLed.
    warm_task: asyncio.Task | None = None
    if settings.searxng_url and not settings.offline:
        from veritas.tools.search import warm_searxng

        warm_task = asyncio.create_task(warm_searxng(settings.searxng_url))

    yield

    if warm_task is not None and not warm_task.done():
        warm_task.cancel()
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

    # API routes are registered BEFORE the static mount. FastAPI matches in
    # registration order, so a catch-all mounted first would swallow /api/*.
    app.include_router(router)

    _mount_frontend(app)

    return app


def _frontend_dir() -> Path | None:
    """Locate the exported Next.js build, if one was bundled.

    Checked in order: an explicit override, the image layout (``/app/static``),
    then the local monorepo path so `veritas serve` also serves the UI after a
    `npm run build`.
    """
    candidates = [
        os.environ.get("VERITAS_STATIC_DIR", ""),
        "/app/static",
        str(Path(__file__).resolve().parents[3] / "frontend" / "out"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "index.html").is_file():
            return Path(candidate)
    return None


def _mount_frontend(app: FastAPI) -> None:
    """Serve the exported UI from the same origin as the API.

    One process on one port means one Render service, one domain, and no CORS.
    When no export is present — the usual local setup, with `next dev` on its
    own port — the API still runs and ``/`` returns service metadata instead.
    """
    static_dir = _frontend_dir()

    if static_dir is None:
        @app.get("/", tags=["system"])
        async def root() -> dict:
            return {
                "name": "VERITAS",
                "version": "1.0.0",
                "docs": "/docs",
                "health": "/health",
                "ui": "not bundled — run `npm run build` in frontend/, or use the dev server",
            }

        log.info("no exported frontend found — serving API only")
        return

    # Hashed build assets are immutable, so they can be cached hard.
    assets = static_dir / "_next"
    if assets.is_dir():
        app.mount("/_next", StaticFiles(directory=assets), name="next-assets")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    async def spa(path: str) -> FileResponse:
        """Serve a static file, falling back to index.html for app routes.

        Registered last, so it only sees paths no API route claimed. Unknown
        /api/* paths are excluded explicitly — returning the HTML shell for a
        mistyped endpoint would turn a clear 404 into a confusing parse error
        in the client.
        """
        if path.startswith(("api/", "docs", "redoc", "openapi.json", "health")):
            raise HTTPException(status_code=404, detail=f"no such endpoint: /{path}")

        target = (static_dir / path).resolve()
        # Containment check: block traversal outside the export directory.
        if target.is_file() and target.is_relative_to(static_dir.resolve()):
            return FileResponse(target)

        nested = target / "index.html"
        if nested.is_file() and nested.is_relative_to(static_dir.resolve()):
            return FileResponse(nested)

        return FileResponse(static_dir / "index.html")

    log.info("serving bundled frontend", path=str(static_dir))


app = create_app()
