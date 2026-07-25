"""Check-worthiness gating.

VeriScore's central observation: verify only claims that are *verifiable*.
Running a retrieval-and-entailment pipeline over an opinion produces a
confident-looking verdict about something evidence cannot settle — which is
exactly the class of output this system exists to eliminate.

Three-way classification follows ClaimBuster: non-factual / factual-unimportant
/ check-worthy. Only the third class enters the verification fan-out, which also
cuts token spend substantially on a typical report.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from veritas.llm.client import LLMClient, system, user
from veritas.logging import get_logger
from veritas.prompts import CHECKWORTHY_SYSTEM, CHECKWORTHY_USER
from veritas.schemas import Claim, ClaimCategory
from veritas.verify.claims import format_claims_for_prompt

log = get_logger(__name__)

# Lexical priors used by the deterministic fallback.
_SUBJECTIVE = re.compile(
    r"\b(best|worst|greatest|beautiful|ugly|amazing|terrible|excellent|poor|"
    r"should|ought|must|need to|arguably|seemingly|likely|probably|perhaps|"
    r"believe|think|feel|hope|prefer|recommend|ideal|perfect|obvious)\b",
    re.IGNORECASE,
)
_FUTURE = re.compile(
    r"\b(will|shall|going to|expected to|forecast|projected|predict|by 20[3-9]\d)\b",
    re.IGNORECASE,
)
_CHECKABLE = re.compile(
    r"(\d[\d,.]*\s*(%|percent|million|billion|trillion|thousand|km|kg|mw|gw|tb|gb)?"
    r"|\b(19|20)\d{2}\b|\b(according to|published|reported|study|survey|report|census)\b)",
    re.IGNORECASE,
)


class Assessment(BaseModel):
    id: str
    category: str
    score: float = Field(default=0.5, ge=0.0, le=1.0)


class CheckworthyResult(BaseModel):
    assessments: list[Assessment] = Field(default_factory=list)


class CheckworthinessClassifier:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def classify(self, claims: list[Claim], batch_size: int = 20) -> list[Claim]:
        """Annotate claims in place with category and check-worthiness score."""
        if not claims:
            return claims

        by_id = {c.id: c for c in claims}
        assessed: set[str] = set()

        for start in range(0, len(claims), batch_size):
            batch = claims[start : start + batch_size]
            try:
                result = await self.llm.structured(
                    [
                        system(CHECKWORTHY_SYSTEM),
                        user(CHECKWORTHY_USER.format(claims=format_claims_for_prompt(batch))),
                    ],
                    CheckworthyResult,
                    role="fast",
                    task="checkworthiness",
                    max_tokens=2048,
                )
            except Exception as exc:
                log.warning("check-worthiness batch failed", error=str(exc)[:200])
                continue

            for assessment in result.assessments:
                claim = by_id.get(assessment.id)
                if claim is None:
                    continue
                claim.category = _parse_category(assessment.category)
                claim.checkworthy_score = _clamp(assessment.score)
                assessed.add(claim.id)

        # Anything the model skipped or mislabelled falls back to the heuristic,
        # so no claim is ever left unclassified.
        for claim in claims:
            if claim.id not in assessed:
                claim.category, claim.checkworthy_score = heuristic_category(claim.verify_text)

        counts = {c.value: 0 for c in ClaimCategory}
        for claim in claims:
            counts[claim.category.value] += 1
        log.info("check-worthiness classified", **counts)
        return claims


def heuristic_category(text: str) -> tuple[ClaimCategory, float]:
    """Deterministic three-way classification from lexical signals."""
    if _SUBJECTIVE.search(text):
        return ClaimCategory.NON_FACTUAL, 0.15
    if _FUTURE.search(text):
        return ClaimCategory.NON_FACTUAL, 0.2
    if _CHECKABLE.search(text):
        return ClaimCategory.CHECK_WORTHY, 0.85

    words = len(text.split())
    if words >= 8:
        return ClaimCategory.CHECK_WORTHY, 0.6
    return ClaimCategory.FACTUAL_UNIMPORTANT, 0.35


def _parse_category(raw: str) -> ClaimCategory:
    normalised = (raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return ClaimCategory(normalised)
    except ValueError:
        if "NON" in normalised or "OPINION" in normalised:
            return ClaimCategory.NON_FACTUAL
        if "UNIMPORTANT" in normalised or "TRIVIAL" in normalised:
            return ClaimCategory.FACTUAL_UNIMPORTANT
        return ClaimCategory.CHECK_WORTHY


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def prioritise(claims: list[Claim], limit: int) -> list[Claim]:
    """Highest check-worthiness first, for when the budget cannot cover them all."""
    worthy = [c for c in claims if c.category is ClaimCategory.CHECK_WORTHY]
    worthy.sort(key=lambda c: -c.checkworthy_score)
    return worthy[:limit]
