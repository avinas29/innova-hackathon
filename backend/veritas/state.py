"""Graph state and the per-run service container.

Every key that a fan-out writes to carries an explicit ``operator.add`` reducer.
This is not optional: LangGraph's default is last-write-wins, so parallel
branches writing to an unreduced key silently discard each other's results. That
failure is invisible — the run completes, just with a fraction of the evidence —
which makes it exactly the kind of bug that survives to the demo. There is a
test asserting every fan-out key has a reducer.
"""

from __future__ import annotations

import asyncio
import operator
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from veritas.config import Settings, get_settings
from veritas.llm.client import LLMClient
from veritas.logging import get_logger
from veritas.schemas import (
    Claim,
    Contradiction,
    Evidence,
    EvidenceCluster,
    ResearchPlan,
    RunEvent,
    Source,
)

log = get_logger(__name__)


class GraphState(TypedDict, total=False):
    """State threaded through the research graph."""

    run_id: str
    topic: str

    plan: ResearchPlan | None

    # Fan-out targets — all reduced.
    sources: Annotated[list[Source], operator.add]
    evidence: Annotated[list[Evidence], operator.add]
    clusters: Annotated[list[EvidenceCluster], operator.add]
    contradictions: Annotated[list[Contradiction], operator.add]
    findings: Annotated[list[str], operator.add]
    warnings: Annotated[list[str], operator.add]
    verified_claims: Annotated[list[Claim], operator.add]

    # Single-writer keys.
    claims: list[Claim]
    draft: str
    executive_summary: str
    report_markdown: str
    reflection_loops: int
    started_at: float
    status: str


def initial_state(run_id: str, topic: str) -> GraphState:
    return GraphState(
        run_id=run_id,
        topic=topic,
        plan=None,
        sources=[],
        evidence=[],
        clusters=[],
        contradictions=[],
        findings=[],
        warnings=[],
        verified_claims=[],
        claims=[],
        draft="",
        executive_summary="",
        report_markdown="",
        reflection_loops=0,
        started_at=time.time(),
        status="RUNNING",
    )


class ClaimTask(TypedDict):
    """Payload sent to one parallel claim-verification branch."""

    run_id: str
    claim: Claim
    topic: str


class ResearchTask(TypedDict):
    """Payload sent to one parallel research branch."""

    run_id: str
    question_id: str
    question: str
    kind: str
    topic: str


@dataclass
class RunContext:
    """Per-run service container.

    Passed through LangGraph's ``configurable`` channel rather than captured in
    closures, so a node is a plain function of (state, context) and can be
    unit-tested with a hand-built context.
    """

    settings: Settings = field(default_factory=get_settings)
    llm: LLMClient = field(default=None)  # type: ignore[assignment]
    search: Any = None
    fetcher: Any = None
    academic: Any = None
    vector_store: Any = None
    entailment: Any = None
    confidence_model: Any = None
    contradiction_detector: Any = None
    event_sink: Any = None
    _budget_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def emit(
        self,
        run_id: str,
        node: str,
        message: str,
        level: str = "info",
        **payload: Any,
    ) -> None:
        """Publish a progress event to the SSE stream and the events table."""
        event = RunEvent(
            run_id=run_id, node=node, level=level, message=message, payload=payload
        )
        if self.event_sink is not None:
            try:
                self.event_sink(event)
            except Exception as exc:  # a broken sink must never fail a run
                log.debug("event sink failed", error=str(exc)[:160])

        getattr(log.bind(run=run_id, node=node), level, log.info)(message, **payload)

    def elapsed(self, started_at: float) -> float:
        return time.time() - started_at

    def over_budget(self, started_at: float) -> tuple[bool, str]:
        """Check both budget ceilings. Returns ``(exceeded, reason)``."""
        if self.llm is not None and self.llm.usage.total >= self.settings.max_tokens_per_run:
            return True, (
                f"token budget exhausted: {self.llm.usage.total} / "
                f"{self.settings.max_tokens_per_run}"
            )
        elapsed = self.elapsed(started_at)
        if elapsed >= self.settings.max_wall_seconds:
            return True, f"time budget exhausted: {elapsed:.0f}s / {self.settings.max_wall_seconds}s"
        return False, ""

    async def aclose(self) -> None:
        for service in (self.llm, self.search, self.fetcher, self.academic):
            if service is not None and hasattr(service, "aclose"):
                try:
                    await service.aclose()
                except Exception as exc:
                    log.debug("cleanup failed", service=type(service).__name__, error=str(exc)[:120])


async def build_context(
    settings: Settings | None = None, event_sink: Any = None
) -> RunContext:
    """Wire up every service a run needs."""
    from veritas.evidence.store import VectorStore, build_embedder
    from veritas.tools.academic import AcademicClient
    from veritas.tools.fetch import ContentFetcher
    from veritas.tools.search import SearchClient
    from veritas.verify.contradiction import ContradictionDetector
    from veritas.verify.entailment import build_entailment_backend

    settings = settings or get_settings()
    llm = LLMClient(settings)

    context = RunContext(
        settings=settings,
        llm=llm,
        search=SearchClient(settings),
        fetcher=ContentFetcher(settings),
        academic=AcademicClient(),
        vector_store=VectorStore(build_embedder(settings)),
        entailment=build_entailment_backend(llm, settings),
        confidence_model=_load_confidence_model(),
        contradiction_detector=ContradictionDetector(llm),
        event_sink=event_sink,
    )
    log.info(
        "run context ready",
        provider=llm.provider_name,
        entailment=context.entailment.name,
        search=",".join(context.search.provider_names) or "none",
    )
    return context


def _load_confidence_model():
    """Load a fitted calibration bundle if one exists, else fall back to priors."""
    from pathlib import Path

    from veritas.verify.calibration import CalibrationBundle, is_sane
    from veritas.verify.confidence import ConfidenceModel

    for candidate in (
        Path("calibration.json"),
        Path(__file__).resolve().parents[2] / "calibration.json",
    ):
        if not candidate.exists():
            continue
        try:
            bundle = CalibrationBundle.load(candidate)
            model = bundle.to_model()
        except Exception as exc:
            log.warning("calibration bundle unreadable", error=str(exc)[:160])
            continue

        # A bundle fitted on degenerate data produces uniformly wrong scores
        # with no error anywhere. Verify before trusting it.
        ok, reason = is_sane(model)
        if not ok:
            log.error(
                "rejecting calibration bundle — falling back to prior weights",
                path=str(candidate),
                reason=reason,
                hint="re-run `veritas eval` with a real provider, then `veritas calibrate`",
            )
            continue

        log.info("loaded fitted calibration", path=str(candidate), **bundle.metadata)
        return model

    log.info("no calibration bundle found — using prior weights (run `veritas calibrate`)")
    return ConfidenceModel()
