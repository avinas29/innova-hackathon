"""HTTP API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from veritas.api.manager import get_manager
from veritas.config import env_summary, get_settings
from veritas.logging import get_logger
from veritas.schemas import RunRequest, VerifyRequest, VerifyResponse
from veritas.storage.db import get_db

log = get_logger(__name__)
router = APIRouter()

# `Source.content` holds up to 60,000 characters of extracted page text per
# source — working data the verification pipeline needs and the browser never
# reads. Serialising it made a finished run's payload ~538 KB, of which 82% was
# that dead weight. On a small instance the response was slow enough that the
# browser gave up with a bare "Load failed" after an otherwise successful run.
_REPORT_EXCLUDE = {"sources": {"__all__": {"content"}}}


def public_report(report) -> dict:
    """Serialise a report for the wire, minus server-side-only bulk."""
    return report.model_dump(mode="json", exclude=_REPORT_EXCLUDE)


class RunAccepted(BaseModel):
    run_id: str
    status: str
    topic: str
    stream_url: str


class RunSummary(BaseModel):
    run_id: str
    topic: str
    status: str
    created_at: str
    finished_at: str | None = None
    tokens: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    config: dict
    active_runs: int


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Liveness plus the resolved provider configuration.

    Exposes which model backend is actually in use, so a demo never silently
    reports offline-heuristic output as model output.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        config=env_summary(),
        active_runs=len(get_manager().list_active()),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/api/runs", response_model=RunAccepted, status_code=202, tags=["runs"])
async def create_run(request: RunRequest) -> RunAccepted:
    """Start a research run. Returns immediately; progress streams over SSE."""
    settings = get_settings()
    if request.max_questions:
        settings.max_research_questions = request.max_questions
    if request.max_claims:
        settings.max_claims = request.max_claims

    handle = await get_manager().start(request.topic)
    return RunAccepted(
        run_id=handle.run_id,
        status=handle.status.value,
        topic=handle.topic,
        stream_url=f"/api/runs/{handle.run_id}/stream",
    )


@router.get("/api/runs", response_model=list[RunSummary], tags=["runs"])
async def list_runs(limit: int = Query(default=25, ge=1, le=200)) -> list[RunSummary]:
    rows = await asyncio.to_thread(get_db().list_runs, limit)
    return [
        RunSummary(
            run_id=str(r["id"]),
            topic=str(r["topic"]),
            status=str(r["status"]),
            created_at=str(r["created_at"]),
            finished_at=str(r["finished_at"]) if r["finished_at"] else None,
            tokens=int(r["prompt_tokens"] or 0) + int(r["completion_tokens"] or 0),
        )
        for r in rows
    ]


@router.get("/api/runs/{run_id}", tags=["runs"])
async def get_run(run_id: str) -> dict:
    """Full run state. Returns the report once the run has finished."""
    manager = get_manager()
    handle = manager.get(run_id) or await manager.load_from_db(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")

    return {
        "run_id": handle.run_id,
        "topic": handle.topic,
        "status": handle.status.value,
        "error": handle.error,
        "finished": handle.finished,
        "event_count": len(handle.events),
        "report": public_report(handle.report) if handle.report else None,
    }


@router.get("/api/runs/{run_id}/stream", tags=["runs"])
async def stream_run(run_id: str):
    """SSE progress stream.

    Buffered history replays first, so a client connecting mid-run — or
    reconnecting after a drop — still sees the whole story.
    """
    manager = get_manager()
    handle = manager.get(run_id) or await manager.load_from_db(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    return EventSourceResponse(manager.stream(run_id))


@router.get("/api/runs/{run_id}/report", response_class=PlainTextResponse, tags=["runs"])
async def get_report(run_id: str) -> str:
    """The final report as markdown."""
    manager = get_manager()
    handle = manager.get(run_id) or await manager.load_from_db(run_id)
    if handle is None:
        raise HTTPException(status_code=404, detail=f"run {run_id} not found")
    if handle.report is None:
        raise HTTPException(status_code=409, detail=f"run {run_id} has not produced a report yet")
    return handle.report.body_markdown or "_No report body was generated._"


@router.get("/api/runs/{run_id}/graph", tags=["runs"])
async def get_graph(run_id: str) -> dict:
    """Evidence graph: claims, sources, clusters and contradictions as nodes/edges.

    Consumed directly by the frontend's force-directed view.
    """
    manager = get_manager()
    handle = manager.get(run_id) or await manager.load_from_db(run_id)
    if handle is None or handle.report is None:
        raise HTTPException(status_code=404, detail=f"no graph available for run {run_id}")

    report = handle.report
    nodes: list[dict] = []
    edges: list[dict] = []

    for claim in report.claims:
        nodes.append(
            {
                "id": claim.id,
                "type": "claim",
                "label": claim.verify_text[:140],
                "verdict": claim.verdict.value,
                "confidence": round(claim.confidence, 3),
                "retracted": claim.retracted,
                "category": claim.category.value,
            }
        )

    for source in report.sources:
        nodes.append(
            {
                "id": source.id,
                "type": "source",
                "label": source.title[:120] or source.domain,
                "domain": source.domain,
                "url": source.url,
                "tier": source.credibility_tier.value,
                "credibility": round(source.credibility_score, 3),
            }
        )

    by_url = {s.url: s.id for s in report.sources}
    for item in report.evidence:
        target = item.source_id or by_url.get(item.url, "")
        if not target:
            continue
        edges.append(
            {
                "source": item.claim_id,
                "target": target,
                "type": "evidence",
                "stance": item.stance.value,
                "weight": round(item.entailment_score, 3),
                "derivative": item.is_derivative,
                "cluster": item.cluster_id,
            }
        )

    for conflict in report.contradictions:
        edges.append(
            {
                "source": conflict.evidence_a,
                "target": conflict.evidence_b,
                "type": "contradiction",
                "weight": round(conflict.score, 3),
                "explanation": conflict.explanation[:200],
            }
        )

    return {
        "run_id": run_id,
        "nodes": nodes,
        "edges": edges,
        "clusters": [c.model_dump(mode="json") for c in report.clusters],
        "metrics": report.metrics.model_dump(mode="json"),
    }


@router.delete("/api/runs/{run_id}", status_code=204, tags=["runs"])
async def delete_run(run_id: str) -> None:
    manager = get_manager()
    await manager.cancel(run_id)
    await asyncio.to_thread(get_db().delete_run, run_id)


# ─────────────────────────────────────────────────────────────────────────────
# Single-claim fast path
# ─────────────────────────────────────────────────────────────────────────────


@router.post("/api/verify", response_model=VerifyResponse, tags=["verify"])
async def verify_claim_endpoint(request: VerifyRequest) -> VerifyResponse:
    """Verify one claim without a full research run.

    Runs the same subgraph the batch path uses, so results are comparable.
    """
    from veritas.graph.nodes import verify_single_claim
    from veritas.state import build_context

    context = await build_context(get_settings())
    try:
        text = f"{request.claim} ({request.context})" if request.context else request.claim
        result = await verify_single_claim(context, text)
    except Exception as exc:
        log.error("single-claim verification failed", error=str(exc)[:300])
        raise HTTPException(status_code=500, detail=f"verification failed: {exc}") from exc
    finally:
        await context.aclose()

    claims = result.get("verified_claims", [])
    if not claims:
        raise HTTPException(status_code=500, detail="verification produced no result")

    return VerifyResponse(
        claim=claims[0],
        evidence=result.get("evidence", []),
        clusters=result.get("clusters", []),
        sources=[],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────


class EvalRequest(BaseModel):
    dataset: str = "builtin"
    limit: int = 30
    include_baseline: bool = True


@router.post("/api/eval", tags=["eval"])
async def run_eval(request: EvalRequest) -> dict:
    """Benchmark the pipeline against a single-LLM baseline.

    This is the endpoint that substantiates the project's central claim, so it
    is part of the product surface rather than a side script.
    """
    from veritas.eval.run import run_evaluation

    try:
        return await run_evaluation(
            dataset=request.dataset,
            limit=request.limit,
            include_baseline=request.include_baseline,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        log.error("evaluation failed", error=str(exc)[:300], exc_info=True)
        raise HTTPException(status_code=500, detail=f"evaluation failed: {exc}") from exc


@router.get("/api/confidence/explain", tags=["verify"])
async def explain_confidence(
    entail_max: float = 0.8,
    agreement: float = 0.6,
    independence: float = 0.5,
    source_quality: float = 0.7,
    consistency: float = 1.0,
    sufficiency: float = 0.6,
    stated_conf: float = 0.7,
    n_independent: int = 2,
) -> dict:
    """Interactive confidence explainer — feed features, see the breakdown.

    Powers the UI's "why this score?" panel and makes the model inspectable
    rather than a black box.
    """
    from veritas.schemas import ConfidenceFeatures
    from veritas.state import _load_confidence_model
    from veritas.verify.confidence import independence_ceiling

    features = ConfidenceFeatures(
        entail_max=entail_max,
        agreement=agreement,
        independence=independence,
        source_quality=source_quality,
        consistency=consistency,
        sufficiency=sufficiency,
        stated_conf=stated_conf,
    )
    model = _load_confidence_model()
    raw = model.raw_score(features)
    calibrated = model.calibrated_score(features)
    ceiling = independence_ceiling(n_independent)

    return {
        "features": features.model_dump(),
        "raw_score": round(raw, 4),
        "calibrated_score": round(calibrated, 4),
        "independence_ceiling": ceiling,
        "final_score": round(min(calibrated, ceiling), 4),
        "capped": calibrated > ceiling,
        "contributions": model.explain(features),
        "bias": model.bias,
    }


@router.get("/api/search/health", tags=["system"])
async def search_health(
    query: str = Query(default="eiffel tower paris", description="Probe query"),
) -> dict:
    """Live-probe every configured search provider.

    ``/health`` lists providers that are *configured*; this reports which ones
    actually answer. The difference matters: a self-hosted SearXNG with a wrong
    URL, or DuckDuckGo blocking a datacenter IP, both look perfectly healthy in
    configuration and return nothing at run time. Without this, a deployment
    that silently retrieves no evidence is indistinguishable from one that
    works — every claim just comes back "not established".
    """
    import time as _time

    from veritas.tools.search import SearchClient

    settings = get_settings()
    client = SearchClient(settings)
    results: list[dict] = []

    try:
        for provider in client._providers:  # noqa: SLF001 - diagnostic surface
            started = _time.perf_counter()
            entry: dict = {"provider": provider.name}
            try:
                found = await provider.search(query, limit=3)
                entry.update(
                    ok=bool(found),
                    results=len(found),
                    sample_domains=[r.domain for r in found[:3]],
                )
                if not found:
                    entry["note"] = "reachable but returned no results"
            except Exception as exc:
                entry.update(ok=False, results=0, error=f"{type(exc).__name__}: {exc}"[:200])
            entry["ms"] = round((_time.perf_counter() - started) * 1000)
            results.append(entry)
    finally:
        await client.aclose()

    working = [r["provider"] for r in results if r.get("ok")]
    return {
        "query": query,
        "configured": client.provider_names,
        "working": working,
        "any_working": bool(working),
        "providers": results,
        "note": (
            "Retrieval is healthy."
            if working
            else "NO search provider is returning results — every claim will "
            "come back NEI. Check SEARXNG_URL, or add TAVILY_API_KEY."
        ),
    }


@router.get("/api/cache/stats", tags=["system"])
async def cache_stats() -> dict:
    row = await asyncio.to_thread(
        get_db().query_one, "SELECT COUNT(*) AS n FROM cache", ()
    )
    return {"entries": int(row["n"]) if row else 0, "enabled": get_settings().cache_enabled}


@router.delete("/api/cache", tags=["system"])
async def clear_cache() -> dict:
    cleared = await asyncio.to_thread(get_db().cache_clear)
    return {"cleared": cleared}
