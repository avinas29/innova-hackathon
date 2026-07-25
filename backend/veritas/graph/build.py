"""Graph assembly and the run orchestrator.

Topology
--------
::

    START → plan ─(Send × N)→ research_worker ─┐
                                               ├→ draft → extract_claims
                                               ┘
      ─(Send × M)→ verify_claim ─┐
                                 ├→ contradictions → [reflect ⟲] → report → END
                                 ┘

Two dynamic fan-outs (``Send``), both reducing into ``operator.add`` keys. The
reflection loop is bounded by ``max_reflection_loops``; an unbounded repair loop
is one of the classic ways an agent run burns its entire budget on a single
stubborn claim.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from veritas.config import Settings, get_settings
from veritas.graph.nodes import (
    contradiction_node,
    dispatch_research,
    dispatch_verification,
    draft_node,
    extract_claims_node,
    plan_node,
    reflect_node,
    report_node,
    research_worker,
    should_reflect,
    verify_claim,
)
from veritas.logging import get_logger
from veritas.schemas import (
    Claim,
    ResearchReport,
    RunMetrics,
    RunStatus,
    TokenUsage,
    Verdict,
)
from veritas.state import GraphState, RunContext, build_context, initial_state

log = get_logger(__name__)


def build_graph(checkpointer: Any | None = None):
    """Compile the research graph."""
    graph = StateGraph(GraphState)

    graph.add_node("plan", plan_node)
    graph.add_node("research_worker", research_worker)
    graph.add_node("draft", draft_node)
    graph.add_node("extract_claims", extract_claims_node)
    graph.add_node("verify_claim", verify_claim)
    graph.add_node("contradictions", contradiction_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("report", report_node)

    graph.add_edge(START, "plan")

    # Map: research questions → parallel workers. Reduce: back into `draft`.
    # Each conditional edge also lists its bypass target: a fan-out that yields
    # zero branches must still have a reachable next node, or the graph halts
    # silently with a partial run.
    graph.add_conditional_edges("plan", dispatch_research, ["research_worker", "draft"])
    graph.add_edge("research_worker", "draft")

    graph.add_edge("draft", "extract_claims")

    # Map: check-worthy claims → parallel verification branches.
    graph.add_conditional_edges(
        "extract_claims", dispatch_verification, ["verify_claim", "contradictions"]
    )
    graph.add_edge("verify_claim", "contradictions")

    # Bounded repair loop.
    graph.add_conditional_edges(
        "contradictions", should_reflect, {"reflect": "reflect", "report": "report"}
    )
    graph.add_conditional_edges(
        "reflect", should_reflect, {"reflect": "reflect", "report": "report"}
    )

    graph.add_edge("report", END)

    return graph.compile(checkpointer=checkpointer or MemorySaver())


class ResearchRunner:
    """Executes a research run end to end and assembles the report."""

    def __init__(self, context: RunContext, settings: Settings | None = None) -> None:
        self.context = context
        self.settings = settings or get_settings()
        self.graph = build_graph()

    async def run(self, topic: str, run_id: str | None = None) -> ResearchReport:
        run_id = run_id or f"run_{uuid.uuid4().hex[:12]}"
        started = time.time()
        state = initial_state(run_id, topic)

        config = {
            "configurable": {"ctx": self.context, "thread_id": run_id},
            "recursion_limit": 60,
        }

        self.context.emit(run_id, "runner", f"Run started: {topic}")

        try:
            final: GraphState = await self.graph.ainvoke(state, config=config)
            status = RunStatus(final.get("status", "COMPLETED"))
        except Exception as exc:
            log.error("run failed", run=run_id, error=str(exc)[:400], exc_info=True)
            self.context.emit(run_id, "runner", f"Run failed: {exc}", level="error")
            return ResearchReport(
                run_id=run_id,
                topic=topic,
                status=RunStatus.FAILED,
                warnings=[f"Run failed: {exc}"],
                metrics=RunMetrics(
                    duration_seconds=time.time() - started,
                    tokens=self.context.llm.usage,
                ),
            )

        report = self._assemble(run_id, topic, final, status, started)
        self.context.emit(
            run_id,
            "runner",
            f"Run complete in {report.metrics.duration_seconds:.1f}s",
            supported=report.metrics.supported,
            refuted=report.metrics.refuted,
            nei=report.metrics.nei,
            tokens=report.metrics.tokens.total,
        )
        return report

    def _assemble(
        self,
        run_id: str,
        topic: str,
        final: GraphState,
        status: RunStatus,
        started: float,
    ) -> ResearchReport:
        verified: list[Claim] = final.get("verified_claims", [])
        all_claims: list[Claim] = final.get("claims", [])
        evidence = final.get("evidence", [])
        clusters = final.get("clusters", [])
        sources = final.get("sources", [])
        contradictions = final.get("contradictions", [])

        # Claims filtered out by the check-worthiness gate never went through
        # verification; include them so the UI can show what was skipped and why.
        verified_ids = {c.id for c in verified}
        skipped = [c for c in all_claims if c.id not in verified_ids]

        supported = sum(1 for c in verified if c.verdict is Verdict.SUPPORTED and not c.retracted)
        refuted = sum(1 for c in verified if c.verdict is Verdict.REFUTED)
        nei = sum(1 for c in verified if c.verdict is Verdict.NEI)
        retracted = sum(1 for c in verified if c.retracted)

        confidences = [c.confidence for c in verified]
        usage: TokenUsage = self.context.llm.usage

        metrics = RunMetrics(
            total_claims=len(all_claims),
            checkworthy_claims=len(verified),
            supported=supported,
            refuted=refuted,
            nei=nei,
            retracted=retracted,
            mean_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            unique_sources=len({s.url for s in sources}),
            unique_domains=len({s.domain for s in sources if s.domain}),
            evidence_items=len(evidence),
            independent_clusters=len(clusters),
            contradictions=len(contradictions),
            duration_seconds=time.time() - started,
            tokens=usage,
        )

        return ResearchReport(
            run_id=run_id,
            topic=topic,
            status=status,
            executive_summary=final.get("executive_summary", ""),
            body_markdown=final.get("report_markdown", ""),
            plan=final.get("plan"),
            claims=[*verified, *skipped],
            sources=sources,
            evidence=evidence,
            clusters=clusters,
            contradictions=contradictions,
            metrics=metrics,
            warnings=final.get("warnings", []),
        )


async def run_research(
    topic: str,
    run_id: str | None = None,
    settings: Settings | None = None,
    event_sink: Any = None,
) -> ResearchReport:
    """Convenience entrypoint: build a context, run, tear down."""
    settings = settings or get_settings()
    context = await build_context(settings, event_sink)
    try:
        runner = ResearchRunner(context, settings)
        return await runner.run(topic, run_id)
    finally:
        await context.aclose()
