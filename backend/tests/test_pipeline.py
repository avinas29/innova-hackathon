"""LLM layer, claim processing, state contract, and the end-to-end graph."""

from __future__ import annotations

import operator
from typing import get_type_hints

import pytest

from veritas.llm.client import (
    LLMClient,
    OfflineProvider,
    containment,
    extract_json,
    jaccard,
    split_sentences,
    tokenize,
    user,
)
from veritas.schemas import Claim, ClaimCategory, RunStatus, TokenUsage, Verdict
from veritas.state import GraphState
from veritas.verify.checkworthy import CheckworthinessClassifier, heuristic_category, prioritise
from veritas.verify.claims import ClaimExtractor, _sentence_fallback


class TestTextUtilities:
    def test_tokenize_drops_stopwords(self):
        tokens = tokenize("The quick brown fox is in the garden")
        assert "the" not in tokens and "is" not in tokens
        assert "quick" in tokens and "fox" in tokens

    def test_sentence_split_respects_abbreviations(self):
        sentences = split_sentences("Dr. Smith arrived. She spoke to Mr. Jones later.")
        assert len(sentences) == 2

    def test_sentence_split_respects_decimals(self):
        sentences = split_sentences("The rate was 3.5 percent. That was low.")
        assert len(sentences) == 2
        assert "3.5" in sentences[0]

    def test_similarity_bounds(self):
        assert jaccard("a b c", "a b c") == pytest.approx(1.0)
        assert jaccard("alpha beta", "gamma delta") == 0.0
        assert jaccard("", "anything") == 0.0

    def test_containment_is_directional(self):
        assert containment("sea level", "global sea level rose sharply") == pytest.approx(1.0)
        assert containment("global sea level rose sharply", "sea level") < 1.0


class TestJsonExtraction:
    def test_plain_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_block(self):
        assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}

    def test_prose_wrapped(self):
        assert extract_json('Sure! Here you go: {"a": 3} Hope that helps.') == {"a": 3}

    def test_array(self):
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_rejects_garbage(self):
        with pytest.raises(ValueError):
            extract_json("no json at all here")
        with pytest.raises(ValueError):
            extract_json("")


class TestOfflineProvider:
    async def test_is_deterministic(self, settings):
        client = LLMClient(settings, provider=OfflineProvider())
        first = await client.chat([user("<task>entailment</task>")], use_cache=False)
        second = await client.chat([user("<task>entailment</task>")], use_cache=False)
        assert first.text == second.text
        assert first.provider == "offline"

    async def test_reports_offline_provenance(self, settings):
        """Offline output must never masquerade as model output."""
        client = LLMClient(settings, provider=OfflineProvider())
        result = await client.chat([user("hello")], use_cache=False)
        assert result.provider == "offline"
        assert result.model == "offline-heuristic"

    async def test_entailment_heuristic_detects_support(self, settings):
        from veritas.schemas import Stance
        from veritas.verify.entailment import EntailmentJudgement, LLMEntailment

        backend = LLMEntailment(LLMClient(settings, provider=OfflineProvider()))
        judgement = await backend.score(
            "Sea level rose 21 centimetres since 1900",
            "Measurements show sea level rose 21 centimetres since 1900 worldwide.",
        )
        assert isinstance(judgement, EntailmentJudgement)
        assert judgement.parsed_stance() is Stance.SUPPORTS

    async def test_entailment_downgrades_irrelevant_evidence(self, settings):
        """Topical non-overlap must not produce a confident stance."""
        from veritas.schemas import Stance
        from veritas.verify.entailment import LLMEntailment

        backend = LLMEntailment(LLMClient(settings, provider=OfflineProvider()))
        judgement = await backend.score(
            "Sea level rose 21 centimetres since 1900",
            "The Python programming language was created by Guido van Rossum.",
        )
        assert judgement.parsed_stance() is Stance.NEUTRAL

    async def test_budget_enforcement(self, settings, monkeypatch):
        from veritas.llm.client import LLMError

        client = LLMClient(settings, provider=OfflineProvider())
        monkeypatch.setattr(client.settings, "max_tokens_per_run", 1)
        client._usage = TokenUsage(prompt_tokens=100, completion_tokens=100, calls=1)

        with pytest.raises(LLMError, match="budget"):
            await client.chat([user("anything")])

    async def test_usage_accumulates(self, settings):
        client = LLMClient(settings, provider=OfflineProvider())
        await client.chat([user("one")], use_cache=False)
        await client.chat([user("two")], use_cache=False)
        assert client.usage.calls == 2
        assert client.usage.total > 0


class TestClaimExtraction:
    async def test_extracts_from_draft(self, offline_llm, settings):
        draft = (
            "Global sea level rose by 21 centimetres between 1900 and 2018.\n\n"
            "The Amazon rainforest covers approximately 5.5 million square kilometres.\n\n"
            "Renewable energy accounted for 30 percent of global electricity in 2023."
        )
        claims = await ClaimExtractor(offline_llm, settings).extract(draft)
        assert len(claims) >= 2
        assert all(c.verify_text for c in claims)

    async def test_empty_draft(self, offline_llm, settings):
        assert await ClaimExtractor(offline_llm, settings).extract("") == []

    async def test_deduplicates_repeated_claims(self, offline_llm, settings):
        draft = "\n\n".join(["The measured value increased by 40 percent in 2024."] * 4)
        claims = await ClaimExtractor(offline_llm, settings).extract(draft)
        assert len(claims) == 1

    def test_fallback_keeps_only_checkable_sentences(self):
        draft = (
            "# A heading\n"
            "Yes.\n"
            "Global temperatures rose by 1.1 degrees Celsius since pre-industrial times.\n"
            "Is that true?\n"
        )
        claims = _sentence_fallback(draft, limit=10)
        texts = [c.text for c in claims]
        assert any("1.1 degrees" in t for t in texts)
        assert not any(t.endswith("?") for t in texts)

    def test_fallback_never_returns_nothing_for_real_prose(self):
        """An empty claim list silently turns the whole run into a no-op."""
        claims = _sentence_fallback(
            "The European Union adopted the regulation in 2023 after long negotiation.",
            limit=5,
        )
        assert claims


class TestCheckWorthiness:
    def test_opinions_are_non_factual(self):
        category, _ = heuristic_category("Python is the best programming language available.")
        assert category is ClaimCategory.NON_FACTUAL

    def test_predictions_are_non_factual(self):
        """No present evidence settles a claim about the future."""
        category, _ = heuristic_category("Electric vehicles will dominate the market by 2040.")
        assert category is ClaimCategory.NON_FACTUAL

    def test_statistics_are_check_worthy(self):
        category, score = heuristic_category("Renewables supplied 30% of electricity in 2023.")
        assert category is ClaimCategory.CHECK_WORTHY
        assert score > 0.5

    async def test_classifier_labels_every_claim(self, offline_llm):
        claims = [
            Claim(text="Renewables supplied 30% of electricity in 2023.", decontextualised="x1"),
            Claim(text="This framework is clearly the best choice.", decontextualised="x2"),
        ]
        classified = await CheckworthinessClassifier(offline_llm).classify(claims)
        assert all(c.category is not None for c in classified)

    def test_prioritise_respects_the_limit(self):
        claims = [
            Claim(
                text=f"claim {i}",
                decontextualised=f"claim {i}",
                category=ClaimCategory.CHECK_WORTHY,
                checkworthy_score=i / 10,
            )
            for i in range(10)
        ]
        top = prioritise(claims, 3)
        assert len(top) == 3
        assert top[0].checkworthy_score >= top[-1].checkworthy_score

    def test_prioritise_excludes_non_checkworthy(self):
        claims = [
            Claim(text="a", decontextualised="a", category=ClaimCategory.NON_FACTUAL),
            Claim(text="b", decontextualised="b", category=ClaimCategory.CHECK_WORTHY),
        ]
        assert len(prioritise(claims, 10)) == 1


class TestStateContract:
    def test_every_fanout_key_has_an_add_reducer(self):
        """Regression guard for silent parallel-write data loss.

        LangGraph's default is last-write-wins. A fan-out key without a reducer
        loses every branch's result but the last — and the run still completes,
        so nothing looks broken. Every key written by a parallel branch must be
        Annotated with operator.add.
        """
        fanout_keys = {
            "sources",
            "evidence",
            "clusters",
            "contradictions",
            "findings",
            "warnings",
            "verified_claims",
        }
        hints = get_type_hints(GraphState, include_extras=True)

        for key in fanout_keys:
            annotation = hints[key]
            metadata = getattr(annotation, "__metadata__", ())
            assert operator.add in metadata, (
                f"GraphState['{key}'] is written by a parallel branch but has no "
                f"operator.add reducer — parallel writes would be silently dropped"
            )

    def test_single_writer_keys_are_unreduced(self):
        hints = get_type_hints(GraphState, include_extras=True)
        for key in ("draft", "report_markdown", "reflection_loops"):
            assert not getattr(hints[key], "__metadata__", ())


class TestEndToEnd:
    async def test_full_graph_runs_offline(self, monkeypatch):
        """The entire pipeline must complete with no keys and no network."""
        from veritas.graph.build import run_research

        events: list = []
        report = await run_research(
            "renewable energy adoption", event_sink=events.append
        )

        assert report.status in {RunStatus.COMPLETED, RunStatus.BUDGET_EXCEEDED}
        assert report.run_id
        assert events, "the run should emit progress events"
        assert {"planner", "runner"} <= {e.node for e in events}

    async def test_graph_survives_zero_search_results(self):
        """No sources must degrade gracefully, not crash the run."""
        from veritas.graph.build import run_research

        report = await run_research("a topic with no retrievable sources whatsoever")
        assert report.status is not RunStatus.FAILED

    async def test_single_claim_verification(self):
        from veritas.graph.nodes import verify_single_claim
        from veritas.state import build_context

        context = await build_context()
        try:
            result = await verify_single_claim(context, "Water boils at 100 degrees Celsius.")
        finally:
            await context.aclose()

        claim = result["verified_claims"][0]
        assert claim.verdict in {Verdict.SUPPORTED, Verdict.REFUTED, Verdict.NEI}
        assert 0.0 <= claim.confidence <= 1.0

    async def test_claim_failure_degrades_to_nei(self, monkeypatch):
        """A crash inside one branch must lose that claim, not the run."""
        from veritas.graph import nodes
        from veritas.state import build_context

        async def boom(*args, **kwargs):
            raise RuntimeError("simulated verification failure")

        monkeypatch.setattr(nodes, "_verify_claim_inner", boom)

        context = await build_context()
        try:
            result = await nodes.verify_claim(
                {
                    "run_id": "r",
                    "topic": "t",
                    "claim": Claim(text="something", decontextualised="something"),
                },
                {"configurable": {"ctx": context}},
            )
        finally:
            await context.aclose()

        claim = result["verified_claims"][0]
        assert claim.verdict is Verdict.NEI
        assert claim.error
        assert result["warnings"]


class TestDatabase:
    def test_run_lifecycle(self):
        from veritas.storage.db import get_db

        db = get_db()
        db.create_run("run_x", "a topic", {"provider": "offline"})
        assert db.get_run("run_x")["status"] == "PENDING"

        db.set_run_status("run_x", "COMPLETED")
        row = db.get_run("run_x")
        assert row["status"] == "COMPLETED"
        assert row["finished_at"]

        db.delete_run("run_x")
        assert db.get_run("run_x") is None

    def test_events_are_ordered_and_replayable(self):
        from veritas.storage.db import get_db

        db = get_db()
        db.create_run("run_y", "topic", {})
        for i in range(5):
            db.append_event("run_y", "node", f"message {i}", payload={"i": i})

        events = db.events_since("run_y", 0)
        assert len(events) == 5
        assert [e["message"] for e in events] == [f"message {i}" for i in range(5)]

        later = db.events_since("run_y", events[2]["id"])
        assert len(later) == 2

    def test_cache_ttl_expiry(self):
        from veritas.storage.db import get_db

        db = get_db()
        db.cache_set("k", "v")
        assert db.cache_get("k", ttl_seconds=3600) == "v"
        assert db.cache_get("k", ttl_seconds=-1) is None


class TestHermeticity:
    """The suite must never make a live network call.

    Enforced by an explicit flag rather than by leaving search unconfigured:
    a keyless reference fallback was added later and silently reintroduced
    live Wikipedia traffic into every test, which hung the suite.
    """

    def test_offline_flag_is_set_for_tests(self, settings):
        assert settings.offline is True

    async def test_search_makes_no_call_when_offline(self, settings):
        from veritas.tools.search import SearchClient

        client = SearchClient(settings)
        try:
            assert await client.search("anything at all") == []
        finally:
            await client.aclose()

    async def test_academic_clients_make_no_call_when_offline(self):
        from veritas.tools.academic import AcademicClient

        client = AcademicClient(offline=True)
        try:
            assert await client.wikipedia("anything") == []
            assert await client.arxiv("anything") == []
            assert await client.semantic_scholar("anything") == []
            assert await client.search_scholarly("anything") == []
        finally:
            await client.aclose()

    async def test_fetcher_makes_no_call_when_offline(self, settings):
        from veritas.tools.fetch import ContentFetcher

        fetcher = ContentFetcher(settings)
        try:
            page = await fetcher.fetch("https://example.com/anything")
            assert not page.ok
            assert "offline" in page.error
        finally:
            await fetcher.aclose()
