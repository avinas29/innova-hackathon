"""Graph nodes.

Each node is an async function of ``(state, config)`` returning a partial state
update. Services come from the :class:`RunContext` in ``config["configurable"]``,
never from module globals, so every node is unit-testable in isolation.

Node failure policy: a node that fails degrades its own contribution and records
a warning. It never raises into the graph. A crashed verification branch yields
an NEI claim with the error attached — losing one claim is recoverable, losing
the run is not.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Send
from pydantic import BaseModel, Field

from veritas.evidence.credibility import domain_of, score_url
from veritas.evidence.dedup import cluster_evidence, partition_for_asymmetry
from veritas.evidence.store import chunk_text
from veritas.llm.client import system, user
from veritas.logging import get_logger
from veritas.prompts import (
    ADJUDICATOR_SYSTEM,
    ADJUDICATOR_USER,
    ADVOCATE_SYSTEM,
    PLANNER_SYSTEM,
    PLANNER_USER,
    QUERY_GEN_SYSTEM,
    QUERY_GEN_USER,
    REFLECTION_SYSTEM,
    REFLECTION_USER,
    REPORT_SYSTEM,
    REPORT_USER,
    REVIEWER_USER,
    SCEPTIC_SYSTEM,
    SYNTHESIS_SYSTEM,
    SYNTHESIS_USER,
)
from veritas.schemas import (
    Claim,
    ClaimCategory,
    Evidence,
    ResearchPlan,
    ResearchQuestion,
    Source,
    SourceKind,
    Stance,
    Verdict,
)
from veritas.state import ClaimTask, GraphState, ResearchTask, RunContext
from veritas.verify.checkworthy import (
    CheckworthinessClassifier,
    heuristic_category,
    prioritise,
)
from veritas.verify.claims import ClaimExtractor
from veritas.verify.confidence import (
    apply_verdict_floor,
    consistency_from_samples,
    extract_features,
)

log = get_logger(__name__)


def ctx_of(config: RunnableConfig) -> RunContext:
    context = (config or {}).get("configurable", {}).get("ctx")
    if context is None:
        raise RuntimeError("RunContext missing from graph config")
    return context


# ─────────────────────────────────────────────────────────────────────────────
# Structured output models
# ─────────────────────────────────────────────────────────────────────────────


class PlannedQuestion(BaseModel):
    question: str
    rationale: str = ""
    kind: str = "WEB"
    priority: int = Field(default=3, ge=1, le=5)


class PlanResult(BaseModel):
    scope_notes: str = ""
    questions: list[PlannedQuestion] = Field(default_factory=list)


class QueryResult(BaseModel):
    queries: list[str] = Field(default_factory=list)


class SynthesisResult(BaseModel):
    executive_summary: str = ""
    body_markdown: str = ""


class ReviewerResult(BaseModel):
    assessment: str = ""


class AdjudicationResult(BaseModel):
    verdict: str = "NEI"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""
    minority_report: str = ""


class ReflectionResult(BaseModel):
    retract: bool = False
    revision: str = ""
    explanation: str = ""


class ReportResult(BaseModel):
    body_markdown: str = ""
    executive_summary: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# 1. Planner
# ─────────────────────────────────────────────────────────────────────────────


async def plan_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    context = ctx_of(config)
    run_id, topic = state["run_id"], state["topic"]
    context.emit(run_id, "planner", f"Planning research for: {topic}")

    limit = context.settings.max_research_questions
    try:
        result = await context.llm.structured(
            [system(PLANNER_SYSTEM), user(PLANNER_USER.format(topic=topic, max_questions=limit))],
            PlanResult,
            role="strong",
            task="plan",
            max_tokens=2048,
        )
        questions = [
            ResearchQuestion(
                question=q.question,
                rationale=q.rationale,
                kind=_parse_kind(q.kind),
                priority=q.priority,
            )
            for q in result.questions
            if q.question.strip()
        ][:limit]
        scope_notes = result.scope_notes
    except Exception as exc:
        log.warning("planner failed — using fallback decomposition", error=str(exc)[:200])
        questions, scope_notes = _fallback_questions(topic, limit), "fallback plan"

    if not questions:
        questions = _fallback_questions(topic, limit)

    plan = ResearchPlan(topic=topic, questions=questions, scope_notes=scope_notes)
    context.emit(
        run_id,
        "planner",
        f"Planned {len(questions)} research questions",
        questions=[q.question for q in questions],
    )
    return {"plan": plan}


def _parse_kind(raw: str) -> SourceKind:
    try:
        return SourceKind((raw or "WEB").strip().upper())
    except ValueError:
        return SourceKind.WEB


def _fallback_questions(topic: str, limit: int) -> list[ResearchQuestion]:
    templates = [
        (f"What is the current factual state of {topic}?", SourceKind.WEB, 5),
        (f"What quantitative evidence exists about {topic}?", SourceKind.WEB, 4),
        (f"What does peer-reviewed research say about {topic}?", SourceKind.ACADEMIC, 4),
        (f"What criticisms or contrary evidence exist about {topic}?", SourceKind.WEB, 4),
        (f"What changed most recently regarding {topic}?", SourceKind.WEB, 3),
        (f"Who are the primary authorities or data sources on {topic}?", SourceKind.WEB, 3),
    ]
    return [
        ResearchQuestion(question=q, kind=k, priority=p, rationale="fallback decomposition")
        for q, k, p in templates[:limit]
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Research fan-out
# ─────────────────────────────────────────────────────────────────────────────


def dispatch_research(state: GraphState) -> list[Send]:
    """Map phase: one parallel branch per research question.

    Anthropic's orchestrator-worker measurements attribute most of the
    multi-agent gain to exactly this — parallel subagents with disjoint
    objectives, each with its own context window.
    """
    plan = state.get("plan")
    if plan is None or not plan.questions:
        # An empty Send list would strand the graph with no reachable node, so
        # route straight to the next stage instead.
        return ["draft"]
    return [
        Send(
            "research_worker",
            ResearchTask(
                run_id=state["run_id"],
                question_id=q.id,
                question=q.question,
                kind=q.kind.value,
                topic=state["topic"],
            ),
        )
        for q in plan.questions
    ]


async def research_worker(task: ResearchTask, config: RunnableConfig) -> dict[str, Any]:
    """One research branch: search, fetch, chunk, index."""
    context = ctx_of(config)
    run_id, question = task["run_id"], task["question"]
    kind = task.get("kind", "WEB")

    context.emit(run_id, "researcher", f"Researching: {question}", kind=kind)

    try:
        results = await _gather_results(context, question, kind)
    except Exception as exc:
        log.warning("research branch failed", question=question[:70], error=str(exc)[:200])
        return {"warnings": [f"Research failed for '{question[:60]}': {exc}"]}

    if not results:
        return {"warnings": [f"No results for '{question[:60]}'"]}

    urls = [r.url for r in results]
    pages = await context.fetcher.fetch_many(urls)

    sources: list[Source] = []
    findings: list[str] = []
    index_ids: list[str] = []
    index_texts: list[str] = []
    index_meta: list[dict] = []

    for result in results:
        page = pages.get(result.url)
        tier, credibility = score_url(result.url)
        content = page.text if page and page.ok else ""
        degraded = not content

        source = Source(
            url=result.url,
            domain=domain_of(result.url) or result.domain,
            title=(page.title if page and page.title else result.title)[:400],
            snippet=result.snippet[:2000],
            content=content,
            kind=_parse_kind(kind),
            credibility_tier=tier,
            credibility_score=credibility,
            published_at=result.published_at,
            fetch_ok=bool(page and page.ok),
            degraded=degraded,
        )
        sources.append(source)

        for position, chunk in enumerate(chunk_text(source.best_text)[:12]):
            index_ids.append(f"{source.id}#{position}")
            index_texts.append(chunk)
            index_meta.append(
                {
                    "source_id": source.id,
                    "url": source.url,
                    "domain": source.domain,
                    "title": source.title,
                    "credibility": credibility,
                }
            )

        if source.best_text:
            findings.append(
                f"[{source.domain}] {source.title}\n{source.best_text[:1200]}"
            )

    if index_ids:
        try:
            await context.vector_store.add(index_ids, index_texts, index_meta)
        except Exception as exc:
            log.warning("vector indexing failed", error=str(exc)[:200])

    context.emit(
        run_id,
        "researcher",
        f"Collected {len(sources)} sources for: {question[:60]}",
        sources=len(sources),
        indexed=len(index_ids),
    )
    return {"sources": sources, "findings": findings}


async def _gather_results(context: RunContext, question: str, kind: str):
    """Route a question to the right retrieval mix for its kind."""
    if kind == "ACADEMIC":
        scholarly, web = await asyncio.gather(
            context.academic.search_scholarly(question, limit=6),
            context.search.search(question, limit=4),
            return_exceptions=True,
        )
        merged = []
        for batch in (scholarly, web):
            if not isinstance(batch, BaseException):
                merged.extend(batch)
        return merged

    queries = [question, f"{question} evidence data"]
    web = await context.search.search_many(queries, limit=6)

    # Wikipedia gives a cheap, high-recall entity anchor for general questions.
    try:
        reference = await context.academic.wikipedia(question, limit=2)
        web.extend(reference)
    except Exception:
        pass

    # Keyless rescue. Web search can fail wholesale in deployment — no key
    # configured, or DuckDuckGo blocking the datacenter IP range that most
    # hosting providers sit in. Without this the run produces no evidence at
    # all and every claim degrades to NEI, which looks like a broken system
    # rather than an unconfigured one. arXiv and Semantic Scholar need no key
    # and answer from any IP.
    if not web:
        log.warning(
            "no web results — falling back to keyless scholarly sources",
            question=question[:70],
        )
        try:
            web = await context.academic.search_scholarly(question, limit=6)
        except Exception as exc:
            log.warning("scholarly fallback failed", error=str(exc)[:160])

    return web[:12]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Draft synthesis
# ─────────────────────────────────────────────────────────────────────────────


async def draft_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    context = ctx_of(config)
    run_id = state["run_id"]
    findings = state.get("findings", [])

    if not findings:
        context.emit(run_id, "synthesiser", "No findings to synthesise", level="warning")
        return {
            "draft": "",
            "executive_summary": "",
            "warnings": ["No sources were retrieved; nothing to verify."],
        }

    context.emit(run_id, "synthesiser", f"Synthesising draft from {len(findings)} findings")

    corpus = "\n\n---\n\n".join(findings)[:40000]
    try:
        result = await context.llm.structured(
            [
                system(SYNTHESIS_SYSTEM),
                user(SYNTHESIS_USER.format(topic=state["topic"], findings=corpus)),
            ],
            SynthesisResult,
            role="strong",
            task="synthesis",
            max_tokens=4096,
        )
        draft, summary = result.body_markdown, result.executive_summary
    except Exception as exc:
        log.warning("synthesis failed", error=str(exc)[:200])
        return {"draft": "", "warnings": [f"Synthesis failed: {exc}"]}

    context.emit(run_id, "synthesiser", f"Draft written ({len(draft)} chars)")
    return {"draft": draft, "executive_summary": summary}


# ─────────────────────────────────────────────────────────────────────────────
# 4. Claim extraction + check-worthiness
# ─────────────────────────────────────────────────────────────────────────────


async def extract_claims_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    context = ctx_of(config)
    run_id, draft = state["run_id"], state.get("draft", "")

    if not draft.strip():
        return {"claims": []}

    context.emit(run_id, "claim_extractor", "Extracting atomic claims")

    extractor = ClaimExtractor(context.llm, context.settings)
    claims = await extractor.extract(draft)

    if not claims:
        return {"claims": [], "warnings": ["No claims could be extracted from the draft."]}

    classifier = CheckworthinessClassifier(context.llm)
    claims = await classifier.classify(claims)

    worthy = [c for c in claims if c.category is ClaimCategory.CHECK_WORTHY]
    context.emit(
        run_id,
        "claim_extractor",
        f"Extracted {len(claims)} claims, {len(worthy)} check-worthy",
        total=len(claims),
        checkworthy=len(worthy),
        skipped=len(claims) - len(worthy),
    )
    return {"claims": claims}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Verification fan-out
# ─────────────────────────────────────────────────────────────────────────────


def dispatch_verification(state: GraphState) -> list[Send]:
    """Map phase: one parallel branch per check-worthy claim.

    Note this phase typically dominates wall-clock: each claim costs roughly
    eight model calls, and on a token-capped tier those are paced apart.
    """
    claims = state.get("claims", [])
    if not claims:
        return ["contradictions"]

    from veritas.config import get_settings

    limit = get_settings().max_claims
    selected = prioritise(claims, limit)
    if not selected:
        return ["contradictions"]
    return [
        Send(
            "verify_claim",
            ClaimTask(run_id=state["run_id"], claim=claim, topic=state["topic"]),
        )
        for claim in selected
    ]


async def verify_claim(task: ClaimTask, config: RunnableConfig) -> dict[str, Any]:
    """The per-claim verification subgraph.

    query generation → retrieval → entailment → independence clustering →
    asymmetric adversarial review → adjudication → calibrated confidence
    """
    context = ctx_of(config)
    claim: Claim = task["claim"]
    run_id = task["run_id"]

    try:
        return await _verify_claim_inner(context, claim, run_id)
    except Exception as exc:
        log.error("claim verification crashed", claim=claim.id, error=str(exc)[:300])
        claim.verdict = Verdict.NEI
        claim.confidence = 0.0
        claim.raw_confidence = 0.0
        claim.error = str(exc)[:300]
        claim.rationale = f"Verification failed: {exc}"[:400]
        return {
            "verified_claims": [claim],
            "warnings": [f"Verification failed for claim {claim.id}: {exc}"],
        }


async def verify_single_claim(
    context: RunContext, claim_text: str, run_id: str = "adhoc"
) -> dict[str, Any]:
    """Verify one claim without a full research run — the API's fast path.

    Runs the identical subgraph the batch path uses, so a single-claim result is
    directly comparable to one produced inside a report.
    """
    claim = Claim(text=claim_text, decontextualised=claim_text, source_sentence=claim_text)
    claim.category, claim.checkworthy_score = heuristic_category(claim_text)
    return await _verify_claim_inner(context, claim, run_id)


async def _verify_claim_inner(
    context: RunContext, claim: Claim, run_id: str
) -> dict[str, Any]:
    settings = context.settings
    claim_text = claim.verify_text

    # Announce the start, not just the finish.
    #
    # Verification is by far the longest phase: each claim costs ~8 model calls
    # carrying full evidence text, and on a token-capped free tier the limiter
    # deliberately sleeps between them. Emitting only on completion leaves the
    # UI silent for minutes, which is indistinguishable from a hang — the most
    # common reason someone kills a run that was working fine.
    context.emit(
        run_id,
        "verifier",
        f"Verifying: {claim_text[:70]}",
        claim_id=claim.id,
        phase="start",
    )

    # ── 5a. Query generation (RARR-style, includes an adversarial query) ─────
    queries = await _generate_queries(context, claim_text)

    # ── 5b. Retrieval: existing corpus first, then fresh web search ──────────
    candidates: list[tuple[str, dict]] = []

    try:
        for hit in await context.vector_store.search(claim_text, k=6):
            if hit.score > 0.25:
                candidates.append((hit.text, hit.metadata))
    except Exception as exc:
        log.debug("vector search failed", error=str(exc)[:160])

    try:
        results = await context.search.search_many(queries, limit=4)

        # Keyless fallback. Web search can fail wholesale — no key configured, a
        # provider outage, or DuckDuckGo serving an anti-bot challenge. Without
        # this, every claim silently degrades to NEI and the run looks broken
        # rather than under-resourced. Wikipedia needs no key and is reliable.
        if not results:
            log.debug("web search empty — falling back to reference sources")
            results = await context.academic.wikipedia(claim_text, limit=3)
            if not results:
                results = await context.academic.search_scholarly(claim_text, limit=3)

        pages = await context.fetcher.fetch_many([r.url for r in results[:8]])
        for result in results[:8]:
            page = pages.get(result.url)
            text = page.text if page and page.ok else result.snippet
            if not text:
                continue
            tier, credibility = score_url(result.url)
            best = _most_relevant_chunk(text, claim_text)
            candidates.append(
                (
                    best,
                    {
                        "source_id": "",
                        "url": result.url,
                        "domain": domain_of(result.url),
                        "title": result.title,
                        "credibility": credibility,
                    },
                )
            )
    except Exception as exc:
        log.warning("claim retrieval failed", claim=claim.id, error=str(exc)[:200])

    candidates = _dedupe_candidates(candidates)[: settings.max_evidence_per_claim]

    if not candidates:
        claim.verdict = Verdict.NEI
        claim.rationale = "No evidence could be retrieved for this claim."
        claim.features = extract_features([], [], Verdict.NEI, 0.0, 1.0, retrieval_attempted=True)
        context.confidence_model.score_claim(claim)
        apply_verdict_floor(claim)
        return {"verified_claims": [claim]}

    # ── 5c. Entailment scoring ──────────────────────────────────────────────
    judgements = await context.entailment.score_batch(
        claim_text,
        [(text, meta.get("domain", "")) for text, meta in candidates],
        concurrency=settings.verify_concurrency,
    )

    evidence: list[Evidence] = []
    for (text, meta), judgement in zip(candidates, judgements, strict=True):
        evidence.append(
            Evidence(
                claim_id=claim.id,
                source_id=meta.get("source_id", "") or "",
                url=meta.get("url", ""),
                domain=meta.get("domain", ""),
                snippet=text[:2500],
                stance=judgement.parsed_stance(),
                entailment_score=judgement.score,
                relevance=judgement.relevance,
                reasoning=judgement.reasoning[:500],
                credibility_score=float(meta.get("credibility", 0.45)),
            )
        )

    # ── 5d. Independence clustering ─────────────────────────────────────────
    clusters, report = await cluster_evidence(
        evidence, claim.id, context.vector_store, settings.dedup_threshold
    )

    # ── 5e. Asymmetric adversarial review ───────────────────────────────────
    side_a, side_b = partition_for_asymmetry(clusters)
    by_id = {e.id: e for e in evidence}

    advocate, sceptic = await asyncio.gather(
        _review(context, ADVOCATE_SYSTEM, claim_text, side_a, by_id),
        _review(context, SCEPTIC_SYSTEM, claim_text, side_b, by_id),
    )

    # ── 5f. Adjudication, with adaptive self-consistency sampling ───────────
    verdict, stated, rationale, minority, consistency = await _adjudicate(
        context, claim_text, clusters, evidence, advocate, sceptic
    )

    # ── 5g. Calibrated confidence ───────────────────────────────────────────
    claim.verdict = verdict
    claim.rationale = rationale
    claim.minority_report = minority
    claim.advocate_argument = advocate
    claim.sceptic_argument = sceptic
    claim.evidence_ids = [e.id for e in evidence]
    claim.cluster_ids = [c.id for c in clusters]
    claim.citations = sorted({e.url for e in evidence if e.url and e.stance is not Stance.NEUTRAL})
    claim.features = extract_features(
        clusters, evidence, verdict, stated, consistency, retrieval_attempted=True
    )
    # Only clusters that actually corroborate the verdict count toward the
    # independence ceiling — neutral and opposing clusters are not corroboration.
    aligned_stance = Stance.SUPPORTS if verdict is Verdict.SUPPORTED else Stance.REFUTES
    n_independent = sum(1 for c in clusters if c.stance is aligned_stance)
    context.confidence_model.score_claim(claim, n_independent=n_independent)
    apply_verdict_floor(claim)

    context.emit(
        run_id,
        "verifier",
        f"{verdict.value}: {claim_text[:70]}",
        claim_id=claim.id,
        verdict=verdict.value,
        confidence=round(claim.confidence, 3),
        evidence=len(evidence),
        clusters=len(clusters),
        independence=round(report.independence_ratio, 2),
    )

    return {"verified_claims": [claim], "evidence": evidence, "clusters": clusters}


async def _generate_queries(context: RunContext, claim_text: str) -> list[str]:
    try:
        result = await context.llm.structured(
            [system(QUERY_GEN_SYSTEM), user(QUERY_GEN_USER.format(claim=claim_text))],
            QueryResult,
            role="fast",
            task="query_generation",
            max_tokens=400,
        )
        queries = [q.strip() for q in result.queries if q.strip()][:3]
    except Exception as exc:
        log.debug("query generation failed", error=str(exc)[:160])
        queries = []

    if not queries:
        from veritas.llm.client import tokenize

        base = " ".join(tokenize(claim_text)[:10])
        queries = [base, f"{base} evidence", f"{base} debunked OR incorrect"]
    return queries


def _most_relevant_chunk(text: str, claim: str) -> str:
    """Pick the chunk of a page with the highest lexical overlap with the claim."""
    from veritas.llm.client import containment

    chunks = chunk_text(text, target_chars=900)
    if not chunks:
        return text[:900]
    return max(chunks, key=lambda c: containment(claim, c))


def _dedupe_candidates(candidates: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    seen: set[str] = set()
    out: list[tuple[str, dict]] = []
    for text, meta in candidates:
        key = (meta.get("url", "") or "") + text[:120]
        if key in seen:
            continue
        seen.add(key)
        out.append((text, meta))
    return out


async def _review(
    context: RunContext,
    system_prompt: str,
    claim_text: str,
    clusters: list,
    by_id: dict[str, Evidence],
) -> str:
    """One side of the adversarial pair, seeing only its own evidence half."""
    if not clusters:
        return "No evidence was allocated to this reviewer."

    lines: list[str] = []
    for cluster in clusters:
        item = by_id.get(cluster.representative_id)
        if item is None:
            continue
        lines.append(
            f"[{item.id}] ({item.domain}, stance={item.stance.value}, "
            f"entailment={item.entailment_score:.2f}, cluster_size={cluster.size})\n"
            f"{item.snippet[:900]}"
        )

    if not lines:
        return "No evidence was allocated to this reviewer."

    try:
        result = await context.llm.structured(
            [
                system(system_prompt),
                user(REVIEWER_USER.format(claim=claim_text, evidence="\n\n".join(lines))),
            ],
            ReviewerResult,
            role="fast",
            task="review",
            max_tokens=400,
        )
        return result.assessment.strip() or "No assessment produced."
    except Exception as exc:
        log.debug("reviewer failed", error=str(exc)[:160])
        return f"Reviewer unavailable ({type(exc).__name__})."


async def _adjudicate(
    context: RunContext,
    claim_text: str,
    clusters: list,
    evidence: list[Evidence],
    advocate: str,
    sceptic: str,
) -> tuple[Verdict, float, str, str, float]:
    """Final verdict, with adaptive self-consistency sampling.

    Extra samples cost tokens, so we spend them only where they carry
    information: when the evidence itself is split, or the first pass lands in
    the uncertain band. Confident, unconflicted claims are decided in one call.
    """
    summary = _evidence_summary(clusters, evidence)
    messages = [
        system(ADJUDICATOR_SYSTEM),
        user(
            ADJUDICATOR_USER.format(
                claim=claim_text,
                evidence_summary=summary,
                advocate=advocate,
                sceptic=sceptic,
                n_clusters=len(clusters),
                n_evidence=len(evidence),
            )
        ),
    ]

    first = await context.llm.structured(
        messages, AdjudicationResult, role="strong", task="adjudication", max_tokens=800
    )
    verdict = _parse_verdict(first.verdict)

    has_support = any(c.stance is Stance.SUPPORTS for c in clusters)
    has_refute = any(c.stance is Stance.REFUTES for c in clusters)
    borderline = 0.35 <= first.confidence <= 0.75
    conflicted = has_support and has_refute

    samples = context.settings.consistency_samples
    if samples <= 1 or not (borderline or conflicted):
        return (
            verdict,
            first.confidence,
            first.rationale,
            first.minority_report,
            1.0,
        )

    extra = await asyncio.gather(
        *(
            context.llm.structured(
                messages,
                AdjudicationResult,
                role="strong",
                temperature=0.7,
                task="adjudication",
                max_tokens=800,
                use_cache=False,
            )
            for _ in range(samples - 1)
        ),
        return_exceptions=True,
    )

    verdicts = [verdict]
    for result in extra:
        if isinstance(result, BaseException):
            continue
        verdicts.append(_parse_verdict(result.verdict))

    consistency = consistency_from_samples(verdicts)
    log.debug(
        "consistency sampling",
        samples=len(verdicts),
        consistency=round(consistency, 2),
        verdicts=[v.value for v in verdicts],
    )
    return verdict, first.confidence, first.rationale, first.minority_report, consistency


def _evidence_summary(clusters: list, evidence: list[Evidence]) -> str:
    by_id = {e.id: e for e in evidence}
    lines: list[str] = []
    for cluster in sorted(clusters, key=lambda c: -c.entailment_score):
        item = by_id.get(cluster.representative_id)
        if item is None:
            continue
        lines.append(
            f"- [{cluster.stance.value}] {item.domain} "
            f"(entailment {cluster.entailment_score:.2f}, credibility "
            f"{cluster.credibility_score:.2f}, {cluster.size} correlated item(s))\n"
            f"  \"{item.snippet[:400]}\""
        )
    return "\n".join(lines) or "No evidence."


def _parse_verdict(raw: str) -> Verdict:
    normalised = (raw or "").strip().upper()
    if normalised.startswith("SUPPORT"):
        return Verdict.SUPPORTED
    if normalised.startswith("REFUT"):
        return Verdict.REFUTED
    return Verdict.NEI


# ─────────────────────────────────────────────────────────────────────────────
# 6. Global contradiction pass
# ─────────────────────────────────────────────────────────────────────────────


async def contradiction_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    context = ctx_of(config)
    run_id = state["run_id"]
    evidence = state.get("evidence", [])

    if len(evidence) < 2:
        return {}

    context.emit(run_id, "contradiction", f"Scanning {len(evidence)} evidence items for conflicts")

    by_claim: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_claim.setdefault(item.claim_id, []).append(item)

    batches = await asyncio.gather(
        *(
            context.contradiction_detector.detect(items, claim_id)
            for claim_id, items in by_claim.items()
            if len(items) >= 2
        ),
        return_exceptions=True,
    )

    found = []
    for batch in batches:
        if isinstance(batch, BaseException):
            log.warning("contradiction batch failed", error=str(batch)[:160])
            continue
        found.extend(batch)

    context.emit(
        run_id,
        "contradiction",
        f"Found {len(found)} source-vs-source contradictions",
        count=len(found),
    )
    return {"contradictions": found}


# ─────────────────────────────────────────────────────────────────────────────
# 7. Reflection / repair (RARR)
# ─────────────────────────────────────────────────────────────────────────────


async def reflect_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    """Revise or retract claims the evidence did not support."""
    context = ctx_of(config)
    run_id = state["run_id"]
    claims = state.get("verified_claims", [])
    evidence = state.get("evidence", [])

    failing = [c for c in claims if c.verdict is not Verdict.SUPPORTED and not c.retracted]
    if not failing:
        return {"reflection_loops": state.get("reflection_loops", 0) + 1}

    context.emit(run_id, "reflection", f"Repairing {len(failing)} unsupported claims")

    by_claim: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_claim.setdefault(item.claim_id, []).append(item)

    results = await asyncio.gather(
        *(_repair(context, claim, by_claim.get(claim.id, [])) for claim in failing),
        return_exceptions=True,
    )

    revised = retracted = 0
    for claim, result in zip(failing, results, strict=True):
        if isinstance(result, BaseException):
            continue
        if result.retract:
            claim.retracted = True
            claim.rationale = (
                f"{claim.rationale} [retracted: {result.explanation}]"
            ).strip()[:600]
            retracted += 1
        elif result.revision.strip():
            claim.revision = result.revision.strip()
            revised += 1

    context.emit(
        run_id,
        "reflection",
        f"Revised {revised}, retracted {retracted}",
        revised=revised,
        retracted=retracted,
    )
    return {"reflection_loops": state.get("reflection_loops", 0) + 1}


async def _repair(context: RunContext, claim: Claim, evidence: list[Evidence]) -> ReflectionResult:
    snippets = "\n\n".join(
        f"[{e.stance.value}] {e.domain}: {e.snippet[:500]}" for e in evidence[:6]
    ) or "No evidence retrieved."

    return await context.llm.structured(
        [
            system(REFLECTION_SYSTEM),
            user(
                REFLECTION_USER.format(
                    claim=claim.verify_text,
                    verdict=claim.verdict.value,
                    rationale=claim.rationale[:400],
                    evidence=snippets,
                )
            ),
        ],
        ReflectionResult,
        role="fast",
        task="reflection",
        max_tokens=600,
    )


def should_reflect(state: GraphState) -> str:
    """Loop guard: reflect at most ``max_reflection_loops`` times."""
    from veritas.config import get_settings

    loops = state.get("reflection_loops", 0)
    if loops >= get_settings().max_reflection_loops:
        return "report"

    claims = state.get("verified_claims", [])
    unsupported = [c for c in claims if c.verdict is not Verdict.SUPPORTED and not c.retracted]
    return "reflect" if unsupported else "report"


# ─────────────────────────────────────────────────────────────────────────────
# 8. Report generation
# ─────────────────────────────────────────────────────────────────────────────


async def report_node(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    context = ctx_of(config)
    run_id = state["run_id"]
    claims = state.get("verified_claims", [])
    sources = state.get("sources", [])
    contradictions = state.get("contradictions", [])

    context.emit(run_id, "report", "Generating final citation-backed report")

    supported = [c for c in claims if c.verdict is Verdict.SUPPORTED and not c.retracted]
    uncertain = [c for c in claims if c.verdict is Verdict.NEI and not c.retracted]
    corrections = [c for c in claims if c.retracted or c.revision]

    cited_urls = sorted({url for c in supported for url in c.citations})
    url_index = {url: i + 1 for i, url in enumerate(cited_urls)}
    by_url = {s.url: s for s in sources}

    source_lines = [
        f"[{i}] {by_url.get(url).title if by_url.get(url) else url} — {url}"
        for url, i in url_index.items()
    ]

    try:
        result = await context.llm.structured(
            [
                system(REPORT_SYSTEM),
                user(
                    REPORT_USER.format(
                        topic=state["topic"],
                        supported=_format_claims(supported, url_index) or "None.",
                        uncertain=_format_claims(uncertain, url_index) or "None.",
                        corrections=_format_corrections(corrections) or "None.",
                        contradictions=_format_contradictions(contradictions) or "None.",
                        sources="\n".join(source_lines) or "None.",
                    )
                ),
            ],
            ReportResult,
            role="strong",
            task="report",
            max_tokens=6000,
        )
        markdown = result.body_markdown
        summary = result.executive_summary or state.get("executive_summary", "")
    except Exception as exc:
        log.warning("report generation failed — emitting deterministic report", error=str(exc)[:200])
        markdown = _deterministic_report(
            state["topic"], supported, uncertain, corrections, contradictions, source_lines
        )
        summary = state.get("executive_summary", "")

    if source_lines and "## Sources" not in markdown:
        markdown += "\n\n## Sources\n\n" + "\n".join(source_lines)

    context.emit(
        run_id,
        "report",
        f"Report complete: {len(supported)} supported, {len(uncertain)} uncertain",
        supported=len(supported),
        uncertain=len(uncertain),
        retracted=sum(1 for c in claims if c.retracted),
    )
    return {"report_markdown": markdown, "executive_summary": summary, "status": "COMPLETED"}


def _format_claims(claims: list[Claim], url_index: dict[str, int]) -> str:
    lines = []
    for claim in claims:
        markers = "".join(f"[{url_index[u]}]" for u in claim.citations if u in url_index)
        text = claim.revision or claim.verify_text
        lines.append(f"- {text} {markers} (confidence {claim.confidence:.2f})")
    return "\n".join(lines)


def _format_corrections(claims: list[Claim]) -> str:
    lines = []
    for claim in claims:
        if claim.retracted:
            lines.append(f"- RETRACTED: \"{claim.verify_text}\" — {claim.rationale[:200]}")
        elif claim.revision:
            lines.append(f"- REVISED: \"{claim.verify_text}\" → \"{claim.revision}\"")
    return "\n".join(lines)


def _format_contradictions(contradictions: list) -> str:
    return "\n".join(
        f"- {c.domain_a} vs {c.domain_b} (severity {c.score:.2f}): {c.explanation[:200]}"
        for c in contradictions[:15]
    )


def _deterministic_report(
    topic: str,
    supported: list[Claim],
    uncertain: list[Claim],
    corrections: list[Claim],
    contradictions: list,
    source_lines: list[str],
) -> str:
    """Fallback report assembled without a model.

    Guarantees the run still produces a usable, honest artefact if the final
    generation call fails — the verification results are the valuable part and
    they already exist by this point.
    """
    parts = [f"# Verified Research Report: {topic}", "", "## Verified findings", ""]
    if supported:
        parts.extend(
            f"- {c.revision or c.verify_text} (confidence {c.confidence:.2f})" for c in supported
        )
    else:
        parts.append("_No claims reached the support threshold._")

    if uncertain:
        parts += ["", "## Uncertain and unverified", ""]
        parts += [f"- {c.verify_text} — {c.rationale[:200]}" for c in uncertain]

    if corrections:
        parts += ["", "## Corrections", ""]
        parts += [_correction_line(c) for c in corrections]

    if contradictions:
        parts += ["", "## Conflicting sources", ""]
        parts += [
            f"- {c.domain_a} vs {c.domain_b}: {c.explanation[:200]}" for c in contradictions[:15]
        ]

    if source_lines:
        parts += ["", "## Sources", "", *source_lines]

    return "\n".join(parts)


def _correction_line(claim: Claim) -> str:
    if claim.retracted:
        return f"- RETRACTED: \"{claim.verify_text}\""
    return f"- REVISED: \"{claim.verify_text}\" → \"{claim.revision}\""


# ─────────────────────────────────────────────────────────────────────────────
# Budget guard
# ─────────────────────────────────────────────────────────────────────────────


async def budget_check(state: GraphState, config: RunnableConfig) -> dict[str, Any]:
    context = ctx_of(config)
    exceeded, reason = context.over_budget(state.get("started_at", time.time()))
    if exceeded:
        context.emit(state["run_id"], "budget", reason, level="warning")
        return {"status": "BUDGET_EXCEEDED", "warnings": [reason]}
    return {}
