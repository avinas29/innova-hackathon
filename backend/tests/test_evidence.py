"""Evidence independence, credibility and chunking."""

from __future__ import annotations

import pytest

from veritas.evidence.credibility import (
    classify_domain,
    credibility_score,
    diversity_bonus,
    domain_of,
)
from veritas.evidence.dedup import (
    UnionFind,
    cluster_evidence,
    effective_independent_count,
    hamming,
    longest_common_token_run,
    partition_for_asymmetry,
    simhash,
)
from veritas.evidence.store import HashingEmbedder, VectorStore, chunk_text
from veritas.schemas import CredibilityTier, Evidence, EvidenceCluster, Stance


class TestCredibility:
    def test_domain_extraction(self):
        assert domain_of("https://www.nature.com/articles/x") == "nature.com"
        assert domain_of("http://sub.example.co.uk/path?q=1") == "sub.example.co.uk"
        assert domain_of("not a url") == ""

    def test_tiers(self):
        assert classify_domain("nature.com") is CredibilityTier.PRIMARY
        assert classify_domain("reuters.com") is CredibilityTier.HIGH
        assert classify_domain("wikipedia.org") is CredibilityTier.MEDIUM
        assert classify_domain("medium.com") is CredibilityTier.LOW
        assert classify_domain("infowars.com") is CredibilityTier.UNRELIABLE

    def test_subdomain_inherits_parent_tier(self):
        assert classify_domain("en.wikipedia.org") is CredibilityTier.MEDIUM
        assert classify_domain("blog.arxiv.org") is CredibilityTier.PRIMARY

    def test_structural_rules(self):
        assert classify_domain("cs.stanford.edu") is CredibilityTier.PRIMARY
        assert classify_domain("data.gov") is CredibilityTier.PRIMARY
        assert classify_domain("random-site.xyz") is CredibilityTier.UNKNOWN

    def test_score_ordering(self):
        assert credibility_score("nature.com") > credibility_score("reuters.com")
        assert credibility_score("reuters.com") > credibility_score("medium.com")
        assert credibility_score("medium.com") > credibility_score("infowars.com")

    def test_diversity_rewards_distinct_credible_domains(self):
        assert diversity_bonus(["nature.com", "reuters.com"]) > diversity_bonus(["nature.com"])
        assert diversity_bonus([]) == 0.0


class TestSimHash:
    def test_identical_text_has_zero_distance(self):
        text = "The quick brown fox jumps over the lazy dog repeatedly today"
        assert hamming(simhash(text), simhash(text)) == 0

    def test_near_duplicate_is_close(self):
        a = "Global sea level rose by 21 centimetres between 1900 and 2018 per the report"
        b = "Global sea level rose by 21 centimetres between 1900 and 2018 per this report"
        assert hamming(simhash(a), simhash(b)) <= 6

    def test_unrelated_text_is_far(self):
        a = "Global sea level rose by 21 centimetres between 1900 and 2018"
        b = "The Python programming language was created by Guido van Rossum in 1991"
        assert hamming(simhash(a), simhash(b)) > 6

    def test_verbatim_run_detection(self):
        """Runs are counted in content tokens — `tokenize` strips stopwords."""
        a = "the committee concluded that emissions must fall by half before the decade ends"
        b = "Reporting today: the committee concluded that emissions must fall by half " \
            "before the decade ends, sources say"
        assert longest_common_token_run(a, b) >= 8
        assert longest_common_token_run("totally different", "nothing alike here") < 3


class TestUnionFind:
    def test_grouping(self):
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(3, 4)
        groups = {frozenset(g) for g in uf.groups().values()}
        assert groups == {frozenset({0, 1}), frozenset({2}), frozenset({3, 4})}


class TestClustering:
    async def test_syndicated_copies_collapse(self, sample_evidence):
        """Two identical snippets from different domains must become ONE cluster.

        This is the anti-echo-chamber guarantee: without it, a wire story
        reprinted N times reads as N independent confirmations.
        """
        clusters, report = await cluster_evidence(sample_evidence, "clm_1")

        assert report.raw_count == 3
        assert report.cluster_count < 3, "identical snippets should have merged"
        assert report.derivative_count >= 1

    async def test_representative_is_most_credible(self, sample_evidence):
        clusters, _ = await cluster_evidence(sample_evidence, "clm_1")
        merged = [c for c in clusters if c.size > 1]
        assert merged, "expected at least one merged cluster"
        by_id = {e.id: e for e in sample_evidence}
        for cluster in merged:
            representative = by_id[cluster.representative_id]
            members = [by_id[m] for m in cluster.member_ids]
            assert representative.credibility_score == max(m.credibility_score for m in members)

    async def test_same_domain_never_independent(self):
        evidence = [
            Evidence(
                claim_id="c",
                source_id=f"s{i}",
                url=f"https://example.com/page{i}",
                domain="example.com",
                snippet=f"An entirely unrelated statement number {i} about widgets and gears.",
                stance=Stance.SUPPORTS,
                entailment_score=0.8,
            )
            for i in range(3)
        ]
        clusters, report = await cluster_evidence(evidence, "c")
        assert report.cluster_count == 1, "same-domain evidence must be one cluster"

    async def test_empty_input(self):
        clusters, report = await cluster_evidence([], "c")
        assert clusters == []
        assert report.independence_ratio == 0.0

    async def test_distinct_domains_and_content_stay_separate(self):
        evidence = [
            Evidence(
                claim_id="c",
                source_id="s1",
                url="https://nature.com/a",
                domain="nature.com",
                snippet="Ocean acidity increased measurably across the sampled decades.",
                stance=Stance.SUPPORTS,
                entailment_score=0.9,
            ),
            Evidence(
                claim_id="c",
                source_id="s2",
                url="https://reuters.com/b",
                domain="reuters.com",
                snippet="Unemployment figures for the manufacturing sector fell last quarter.",
                stance=Stance.SUPPORTS,
                entailment_score=0.8,
            ),
        ]
        clusters, report = await cluster_evidence(evidence, "c")
        assert report.cluster_count == 2


class TestAsymmetricPartition:
    def test_partition_is_disjoint_and_complete(self):
        clusters = [
            EvidenceCluster(
                claim_id="c",
                member_ids=[f"e{i}"],
                representative_id=f"e{i}",
                entailment_score=0.9 - i * 0.1,
                credibility_score=0.8,
            )
            for i in range(6)
        ]
        side_a, side_b = partition_for_asymmetry(clusters)

        ids_a = {c.id for c in side_a}
        ids_b = {c.id for c in side_b}
        assert not (ids_a & ids_b), "reviewers must not share evidence"
        assert ids_a | ids_b == {c.id for c in clusters}

    def test_partition_balances_strength(self):
        """Neither reviewer may be handed only weak evidence."""
        clusters = [
            EvidenceCluster(
                claim_id="c",
                member_ids=[f"e{i}"],
                representative_id=f"e{i}",
                entailment_score=1.0 - i * 0.1,
                credibility_score=1.0,
            )
            for i in range(8)
        ]
        side_a, side_b = partition_for_asymmetry(clusters)
        mean_a = sum(c.entailment_score for c in side_a) / len(side_a)
        mean_b = sum(c.entailment_score for c in side_b) / len(side_b)
        assert abs(mean_a - mean_b) < 0.15

    def test_single_cluster_goes_to_one_side(self):
        clusters = [EvidenceCluster(claim_id="c", member_ids=["e"], representative_id="e")]
        side_a, side_b = partition_for_asymmetry(clusters)
        assert len(side_a) == 1 and side_b == []


class TestEffectiveCount:
    def test_singletons_count_fully(self):
        clusters = [
            EvidenceCluster(claim_id="c", member_ids=["a"], representative_id="a"),
            EvidenceCluster(claim_id="c", member_ids=["b"], representative_id="b"),
        ]
        assert effective_independent_count(clusters) == pytest.approx(2.0)

    def test_duplicates_add_sublinearly(self):
        """Ten copies must not count as ten sources."""
        big = [
            EvidenceCluster(
                claim_id="c", member_ids=[f"e{i}" for i in range(10)], representative_id="e0"
            )
        ]
        assert effective_independent_count(big) < 2.0

    def test_empty(self):
        assert effective_independent_count([]) == 0.0


class TestChunking:
    def test_short_text_is_single_chunk(self):
        assert chunk_text("A short sentence.") == ["A short sentence."]

    def test_respects_paragraphs(self):
        text = "\n\n".join([f"Paragraph number {i} with some content here." for i in range(40)])
        chunks = chunk_text(text, target_chars=300)
        assert len(chunks) > 1
        assert all(len(c) < 700 for c in chunks)

    def test_handles_oversized_paragraph(self):
        chunks = chunk_text("x" * 5000, target_chars=500)
        assert len(chunks) > 1

    def test_empty(self):
        assert chunk_text("") == []


class TestVectorStore:
    async def test_add_and_search(self):
        store = VectorStore(HashingEmbedder())
        await store.add(
            ["a", "b", "c"],
            [
                "sea level rise measured in centimetres over the century",
                "python programming language guido van rossum",
                "ocean levels increased centimetres across decades",
            ],
        )
        assert len(store) == 3
        hits = await store.search("sea level rise centimetres", k=2)
        assert hits[0].id in {"a", "c"}

    async def test_empty_search(self):
        assert await VectorStore(HashingEmbedder()).search("anything") == []

    async def test_similarity_matrix_is_symmetric(self):
        store = VectorStore(HashingEmbedder())
        matrix = await store.similarity_matrix(["alpha beta gamma", "alpha beta delta"])
        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == pytest.approx(matrix[1, 0], abs=1e-5)
        assert matrix[0, 0] == pytest.approx(1.0, abs=1e-5)

    async def test_mismatched_lengths_rejected(self):
        store = VectorStore(HashingEmbedder())
        with pytest.raises(ValueError):
            await store.add(["a", "b"], ["only one"])
