"""Claim extraction: atomic, decontextualised, verifiable units.

This is the highest-leverage step in the pipeline. Everything downstream —
retrieval, entailment, confidence, the UI — operates on claims, so an extraction
error is unrecoverable. Two properties matter equally:

* **Atomicity.** "X shipped in 2024 and removed the GIL" cannot receive one
  verdict; half of it may be true.
* **Decontextualisation.** "It grew 40% that year" is unverifiable in isolation
  and, worse, will be silently verified against the wrong proposition once it
  reaches a retriever. DnDScore shows decontextualisation carries as much weight
  as decomposition; we treat it as a first-class output, not a nicety.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from veritas.config import Settings, get_settings
from veritas.llm.client import LLMClient, split_sentences, system, user
from veritas.logging import get_logger
from veritas.prompts import CLAIM_EXTRACTION_SYSTEM, CLAIM_EXTRACTION_USER
from veritas.schemas import Claim, ClaimCategory

log = get_logger(__name__)


class ExtractedClaim(BaseModel):
    text: str
    decontextualised: str = ""
    source_sentence: str = ""


class ClaimExtractionResult(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list)


class ClaimExtractor:
    def __init__(self, llm: LLMClient, settings: Settings | None = None) -> None:
        self.llm = llm
        self.settings = settings or get_settings()

    async def extract(self, draft: str, max_claims: int | None = None) -> list[Claim]:
        """Extract claims from a draft report."""
        limit = max_claims or self.settings.max_claims
        draft = draft.strip()
        if not draft:
            return []

        try:
            result = await self.llm.structured(
                [
                    system(CLAIM_EXTRACTION_SYSTEM),
                    user(CLAIM_EXTRACTION_USER.format(draft=draft[:24000], max_claims=limit)),
                ],
                ClaimExtractionResult,
                role="fast",
                task="claim_extraction",
                max_tokens=4096,
            )
            extracted = result.claims
        except Exception as exc:
            log.warning("LLM claim extraction failed — using sentence fallback", error=str(exc)[:200])
            extracted = _sentence_fallback(draft, limit)

        if not extracted:
            extracted = _sentence_fallback(draft, limit)

        claims: list[Claim] = []
        seen: set[str] = set()
        for item in extracted[:limit]:
            text = _normalise(item.text)
            if not text or len(text) < 15:
                continue
            key = _dedup_key(text)
            if key in seen:
                continue
            seen.add(key)

            decontext = _normalise(item.decontextualised) or text
            claims.append(
                Claim(
                    text=text,
                    decontextualised=decontext,
                    source_sentence=_normalise(item.source_sentence) or text,
                )
            )

        log.info("claims extracted", count=len(claims), draft_chars=len(draft))
        return claims


def _sentence_fallback(draft: str, limit: int) -> list[ExtractedClaim]:
    """Deterministic backstop when structured extraction fails.

    Keeps declarative sentences carrying a checkable signal — a number, a proper
    noun, or a copula. Lower recall than the model, but it never returns nothing,
    and an empty claim list would silently turn the whole run into a no-op.
    """
    out: list[ExtractedClaim] = []
    for line in draft.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", ">", "|", "```")):
            continue
        line = re.sub(r"^[-*+]\s+", "", line)
        for sentence in split_sentences(line):
            if len(sentence) < 25 or sentence.endswith("?"):
                continue
            has_signal = bool(
                re.search(r"\d", sentence)
                or re.search(r"\b[A-Z][a-zA-Z]{2,}", sentence)
                or re.search(r"\b(is|are|was|were|has|have|had)\b", sentence)
            )
            if has_signal:
                out.append(
                    ExtractedClaim(
                        text=sentence, decontextualised=sentence, source_sentence=sentence
                    )
                )
            if len(out) >= limit:
                return out
    return out


_CITATION_RE = re.compile(r"\s*\[\d+\]")
_WS_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    text = _CITATION_RE.sub("", text or "")
    text = _WS_RE.sub(" ", text).strip()
    return text.strip("*_ \t")


def _dedup_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())[:120]


def format_claims_for_prompt(claims: list[Claim]) -> str:
    """``id | text`` lines, the format the check-worthiness prompt expects."""
    return "\n".join(f"{c.id} | {c.verify_text}" for c in claims)


def supported_claims(claims: list[Claim]) -> list[Claim]:
    from veritas.schemas import Verdict

    return [c for c in claims if c.verdict is Verdict.SUPPORTED and not c.retracted]


def checkworthy_claims(claims: list[Claim]) -> list[Claim]:
    return [c for c in claims if c.category is ClaimCategory.CHECK_WORTHY]
