"""Evidence independence clustering — the anti-echo-chamber layer.

The failure this fixes
----------------------
Ten outlets republishing one wire story look like ten confirmations. Counting
them as ten is *communal reinforcement*: repetition drives belief instead of
evidence doing it. Any confidence score computed over raw evidence counts is
therefore systematically inflated, and inflated exactly where the claim is most
viral — the worst possible place.

We collapse correlated evidence into clusters and let each cluster cast one
weighted vote. Four independent signals feed the merge decision:

1. **SimHash** over token shingles — catches verbatim and lightly-edited copies
   in O(n²) cheap integer comparisons.
2. **Embedding cosine** — catches paraphrase and translation.
3. **Same-domain** — two snippets from one site are never independent.
4. **Verbatim span containment** — a long shared literal span means one text
   quotes the other; the quoter is marked derivative.

Union-find merges whatever any signal joins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from veritas.evidence.store import VectorStore
from veritas.llm.client import tokenize
from veritas.logging import get_logger
from veritas.schemas import Evidence, EvidenceCluster, Stance

log = get_logger(__name__)

_SIMHASH_BITS = 64
_SIMHASH_MAX_DISTANCE = 6      # ≤6 differing bits ≈ near-identical text
_SHINGLE_SIZE = 4

# Shared literal run implying quotation, counted in CONTENT tokens — `tokenize`
# strips stopwords, so 8 content tokens is roughly 13-15 words of running prose.
# Two independent journalists do not produce that by coincidence.
_MIN_VERBATIM_TOKENS = 8


# ─────────────────────────────────────────────────────────────────────────────
# SimHash
# ─────────────────────────────────────────────────────────────────────────────


def shingles(text: str, size: int = _SHINGLE_SIZE) -> list[str]:
    """Overlapping token n-grams. Word order matters for copy detection."""
    tokens = tokenize(text)
    if len(tokens) < size:
        return [" ".join(tokens)] if tokens else []
    return [" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1)]


def simhash(text: str, bits: int = _SIMHASH_BITS) -> int:
    """Charikar SimHash fingerprint of a text's shingle set."""
    import hashlib

    accumulator = [0] * bits
    grams = shingles(text)
    if not grams:
        return 0

    for gram in grams:
        digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        for bit in range(bits):
            accumulator[bit] += 1 if (value >> bit) & 1 else -1

    fingerprint = 0
    for bit in range(bits):
        if accumulator[bit] > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def longest_common_token_run(a: str, b: str) -> int:
    """Length of the longest shared consecutive token run.

    Classic DP longest-common-substring over token sequences. A long run is
    strong evidence of copy-paste rather than independent reporting.
    """
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0

    previous = [0] * (len(tb) + 1)
    best = 0
    for i in range(1, len(ta) + 1):
        current = [0] * (len(tb) + 1)
        ai = ta[i - 1]
        for j in range(1, len(tb) + 1):
            if ai == tb[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best:
                    best = current[j]
        previous = current
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Union-find
# ─────────────────────────────────────────────────────────────────────────────


class UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class ClusteringReport:
    """Diagnostics — surfaced in the UI to make the collapse visible."""

    raw_count: int = 0
    cluster_count: int = 0
    derivative_count: int = 0
    largest_cluster: int = 0

    @property
    def independence_ratio(self) -> float:
        """1.0 means every item was independent; lower means heavy duplication."""
        return self.cluster_count / self.raw_count if self.raw_count else 0.0


async def cluster_evidence(
    evidence: list[Evidence],
    claim_id: str,
    store: VectorStore | None = None,
    cosine_threshold: float = 0.86,
) -> tuple[list[EvidenceCluster], ClusteringReport]:
    """Collapse correlated evidence into independent clusters.

    Returns the clusters and a report. Mutates each ``Evidence`` in place to set
    ``cluster_id`` and ``is_derivative``.
    """
    if not evidence:
        return [], ClusteringReport()

    n = len(evidence)
    uf = UnionFind(n)
    texts = [e.snippet for e in evidence]

    # 1. SimHash near-duplicates
    fingerprints = [simhash(t) for t in texts]
    for i in range(n):
        for j in range(i + 1, n):
            if hamming(fingerprints[i], fingerprints[j]) <= _SIMHASH_MAX_DISTANCE:
                uf.union(i, j)

    # 2. Same-domain evidence is not independent
    by_domain: dict[str, list[int]] = {}
    for idx, item in enumerate(evidence):
        if item.domain:
            by_domain.setdefault(item.domain, []).append(idx)
    for indices in by_domain.values():
        for k in range(1, len(indices)):
            uf.union(indices[0], indices[k])

    # 3. Semantic near-duplicates
    if store is not None and n > 1:
        try:
            similarity = await store.similarity_matrix(texts)
            for i in range(n):
                for j in range(i + 1, n):
                    if float(similarity[i, j]) >= cosine_threshold:
                        uf.union(i, j)
        except Exception as exc:
            log.warning("semantic dedup skipped", error=str(exc)[:160])

    # 4. Verbatim quotation → mark the shorter text derivative
    derivative = [False] * n
    for i in range(n):
        for j in range(i + 1, n):
            if uf.find(i) == uf.find(j):
                continue
            if longest_common_token_run(texts[i], texts[j]) >= _MIN_VERBATIM_TOKENS:
                uf.union(i, j)
                shorter = i if len(texts[i]) < len(texts[j]) else j
                derivative[shorter] = True

    # Materialise clusters
    clusters: list[EvidenceCluster] = []
    largest = 0
    for members in uf.groups().values():
        members.sort(key=lambda idx: (-evidence[idx].credibility_score, -len(texts[idx])))
        representative = evidence[members[0]]
        largest = max(largest, len(members))

        cluster = EvidenceCluster(
            claim_id=claim_id,
            member_ids=[evidence[i].id for i in members],
            representative_id=representative.id,
            stance=_cluster_stance([evidence[i] for i in members]),
            entailment_score=max(evidence[i].entailment_score for i in members),
            credibility_score=max(evidence[i].credibility_score for i in members),
            domains=sorted({evidence[i].domain for i in members if evidence[i].domain}),
        )

        for position, idx in enumerate(members):
            evidence[idx].cluster_id = cluster.id
            # Everything after the representative is a redundant copy.
            evidence[idx].is_derivative = derivative[idx] or position > 0

        clusters.append(cluster)

    report = ClusteringReport(
        raw_count=n,
        cluster_count=len(clusters),
        derivative_count=sum(1 for e in evidence if e.is_derivative),
        largest_cluster=largest,
    )
    log.debug(
        "evidence clustered",
        claim=claim_id,
        raw=report.raw_count,
        clusters=report.cluster_count,
        independence=round(report.independence_ratio, 2),
    )
    return clusters, report


def _cluster_stance(members: list[Evidence]) -> Stance:
    """Stance of a cluster: strongest non-neutral signal wins, ties → neutral."""
    support = sum(e.entailment_score for e in members if e.stance is Stance.SUPPORTS)
    refute = sum(e.entailment_score for e in members if e.stance is Stance.REFUTES)
    if support > refute and support > 0:
        return Stance.SUPPORTS
    if refute > support and refute > 0:
        return Stance.REFUTES
    return Stance.NEUTRAL


def partition_for_asymmetry(
    clusters: list[EvidenceCluster], seed: int = 0
) -> tuple[list[EvidenceCluster], list[EvidenceCluster]]:
    """Split clusters into two *disjoint* evidence sets for adversarial review.

    This is the mechanism that rescues multi-agent debate from the martingale
    result. Debating agents given identical context provably fail to improve
    expected correctness — the disagreement carries no new information. Give
    them disjoint evidence and their disagreement becomes genuine information
    aggregation instead of personality clash.

    Interleaving by descending strength keeps both halves comparably strong, so
    neither reviewer is handed a straw position.
    """
    if len(clusters) < 2:
        return list(clusters), []

    ordered = sorted(
        clusters,
        key=lambda c: (-c.entailment_score * c.credibility_score, c.id),
    )
    rotation = seed % 2
    side_a = [c for i, c in enumerate(ordered) if i % 2 == rotation]
    side_b = [c for i, c in enumerate(ordered) if i % 2 != rotation]
    return side_a, side_b


def effective_independent_count(clusters: list[EvidenceCluster]) -> float:
    """Independent-source count, discounted for correlation within clusters.

    A cluster of one counts fully. Extra members inside a cluster add a small
    sublinear increment — repetition is weak corroboration, not none.
    """
    if not clusters:
        return 0.0
    total = 0.0
    for cluster in clusters:
        total += 1.0 + 0.15 * np.log1p(max(0, cluster.size - 1))
    return float(total)
