"""Source-versus-source contradiction detection.

Distinct from claim-versus-evidence conflict, and considerably more informative:
a reader learns much more from "Reuters and the WHO report different figures for
this" than from "our draft disagreed with a source". It is also the only place
the system detects that the *evidence base itself* is unreliable.

Cost control: pairwise NLI is O(n²) in LLM calls, which is unaffordable across a
whole run. We prefilter with cheap deterministic signals — numeric conflict on
shared units, polarity mismatch, sufficient topical overlap — and only escalate
surviving pairs to a model. In practice this cuts calls by well over an order of
magnitude while keeping the pairs that matter.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from veritas.llm.client import LLMClient, jaccard, system, user
from veritas.logging import get_logger
from veritas.prompts import CONTRADICTION_SYSTEM, CONTRADICTION_USER
from veritas.schemas import Contradiction, Evidence, Stance

log = get_logger(__name__)

_MIN_OVERLAP = 0.22           # below this the two snippets aren't about the same thing
# Hard ceiling on escalations. Contradiction detection is valuable but
# secondary: it must never consume more budget than the verdicts themselves.
# At 12 per claim it could out-spend the entire verification stage.
_MAX_PAIRS_PER_CLAIM = 3

_NUMBER_RE = re.compile(r"(\d[\d,]*\.?\d*)\s*(%|percent|million|billion|trillion|thousand|k|m|bn)?")
_NEGATION_RE = re.compile(
    r"\b(not|no|never|none|denies|denied|false|incorrect|refutes|contrary|failed to|"
    r"did not|does not|cannot|disputed|debunked)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


class ContradictionJudgement(BaseModel):
    contradictory: bool = False
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""


@dataclass(slots=True)
class CandidatePair:
    a: Evidence
    b: Evidence
    overlap: float
    reason: str


def extract_numbers(text: str) -> list[tuple[float, str]]:
    """Numeric values with their unit suffix, normalised to a common scale."""
    out: list[tuple[float, str]] = []
    for raw, unit in _NUMBER_RE.findall(text):
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        unit = (unit or "").lower()
        multiplier = {
            "thousand": 1e3, "k": 1e3,
            "million": 1e6, "m": 1e6,
            "billion": 1e9, "bn": 1e9,
            "trillion": 1e12,
        }.get(unit)
        if multiplier:
            out.append((value * multiplier, "count"))
        elif unit in {"%", "percent"}:
            out.append((value, "percent"))
        else:
            out.append((value, ""))
    return out


def numeric_conflict(text_a: str, text_b: str) -> bool:
    """True when both texts state same-unit figures that differ materially.

    Requires a shared year mention: the most common false positive is two
    sources reporting genuinely different periods, which is not a conflict.
    """
    years_a, years_b = set(_YEAR_RE.findall(text_a)), set(_YEAR_RE.findall(text_b))
    if years_a and years_b and not (years_a & years_b):
        return False

    nums_a = [n for n in extract_numbers(text_a) if n[1] == "percent"]
    nums_b = [n for n in extract_numbers(text_b) if n[1] == "percent"]
    if not nums_a or not nums_b:
        return False

    for value_a, _ in nums_a:
        for value_b, _ in nums_b:
            larger = max(abs(value_a), abs(value_b), 1e-9)
            if abs(value_a - value_b) / larger > 0.25:
                return True
    return False


def polarity_conflict(text_a: str, text_b: str) -> bool:
    return bool(_NEGATION_RE.search(text_a)) != bool(_NEGATION_RE.search(text_b))


def find_candidates(evidence: list[Evidence]) -> list[CandidatePair]:
    """Cheap prefilter for pairs worth an LLM call."""
    candidates: list[CandidatePair] = []

    for i in range(len(evidence)):
        for j in range(i + 1, len(evidence)):
            a, b = evidence[i], evidence[j]

            # Same cluster == same underlying source; not an independent conflict.
            if a.cluster_id and a.cluster_id == b.cluster_id:
                continue
            if a.domain and a.domain == b.domain:
                continue

            overlap = jaccard(a.snippet, b.snippet)
            if overlap < _MIN_OVERLAP:
                continue

            reasons = []
            if a.stance is Stance.SUPPORTS and b.stance is Stance.REFUTES:
                reasons.append("opposing stance")
            elif a.stance is Stance.REFUTES and b.stance is Stance.SUPPORTS:
                reasons.append("opposing stance")
            if numeric_conflict(a.snippet, b.snippet):
                reasons.append("numeric conflict")
            if polarity_conflict(a.snippet, b.snippet):
                reasons.append("polarity mismatch")

            if reasons:
                candidates.append(CandidatePair(a, b, overlap, ", ".join(reasons)))

    candidates.sort(key=lambda p: -p.overlap)
    return candidates[:_MAX_PAIRS_PER_CLAIM]


class ContradictionDetector:
    def __init__(self, llm: LLMClient, concurrency: int = 6) -> None:
        self.llm = llm
        self._sem = asyncio.Semaphore(concurrency)

    async def detect(self, evidence: list[Evidence], claim_id: str = "") -> list[Contradiction]:
        """Find genuine conflicts among a set of evidence items."""
        if len(evidence) < 2:
            return []

        candidates = find_candidates(evidence)
        if not candidates:
            return []

        log.debug("contradiction candidates", claim=claim_id, pairs=len(candidates))
        results = await asyncio.gather(
            *(self._judge(pair, claim_id) for pair in candidates), return_exceptions=True
        )

        found: list[Contradiction] = []
        for result in results:
            if isinstance(result, BaseException):
                log.warning("contradiction check failed", error=str(result)[:160])
                continue
            if result is not None:
                found.append(result)

        if found:
            log.info("contradictions found", claim=claim_id, count=len(found))
        return found

    async def _judge(self, pair: CandidatePair, claim_id: str) -> Contradiction | None:
        async with self._sem:
            judgement = await self.llm.structured(
                [
                    system(CONTRADICTION_SYSTEM),
                    user(
                        CONTRADICTION_USER.format(
                            evidence_a=pair.a.snippet[:2500],
                            domain_a=pair.a.domain or "unknown",
                            evidence_b=pair.b.snippet[:2500],
                            domain_b=pair.b.domain or "unknown",
                        )
                    ),
                ],
                ContradictionJudgement,
                role="fast",
                task="contradiction",
                max_tokens=400,
            )

        if not judgement.contradictory or judgement.score < 0.3:
            return None

        return Contradiction(
            claim_id=claim_id,
            evidence_a=pair.a.id,
            evidence_b=pair.b.id,
            score=judgement.score,
            explanation=judgement.explanation or pair.reason,
            domain_a=pair.a.domain,
            domain_b=pair.b.domain,
        )
