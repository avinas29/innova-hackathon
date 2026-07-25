"""Single-LLM baseline — the control condition.

The problem statement asks the system to outperform a single LLM. That is an
empirical claim, and an empirical claim needs a control. This module is that
control: one model, one prompt, its own knowledge, its own verbalised
confidence. No retrieval, no decomposition, no calibration.

The comparison is deliberately fair. The baseline uses the *strong* model, not a
weak one, and gets a clean, competent prompt. Beating a strawman would prove
nothing. Where the pipeline should win is not raw accuracy on easy facts — a
frontier model already knows those — but on **calibration** and on **NEI
claims**, where a single model's documented behaviour is to answer confidently
rather than abstain.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from veritas.eval.dataset import LabelledClaim, normalise_label
from veritas.llm.client import LLMClient, system, user
from veritas.logging import get_logger
from veritas.prompts import BASELINE_SYSTEM, BASELINE_USER

log = get_logger(__name__)


class BaselineJudgement(BaseModel):
    verdict: str = "NEI"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = ""


class BaselinePrediction(BaseModel):
    claim: str
    gold: str
    predicted: str
    confidence: float
    rationale: str = ""


class SingleLLMBaseline:
    """One model, one call, no evidence."""

    name = "Single LLM (no retrieval)"

    def __init__(self, llm: LLMClient, concurrency: int = 6) -> None:
        self.llm = llm
        self._sem = asyncio.Semaphore(concurrency)

    async def predict_one(self, item: LabelledClaim) -> BaselinePrediction:
        async with self._sem:
            try:
                judgement = await self.llm.structured(
                    [
                        system(BASELINE_SYSTEM),
                        user(BASELINE_USER.format(claim=item.claim)),
                    ],
                    BaselineJudgement,
                    role="strong",
                    task="baseline",
                    max_tokens=300,
                )
            except Exception as exc:
                log.warning("baseline prediction failed", error=str(exc)[:200])
                return BaselinePrediction(
                    claim=item.claim,
                    gold=item.label,
                    predicted="NEI",
                    confidence=0.0,
                    rationale=f"error: {exc}"[:200],
                )

        return BaselinePrediction(
            claim=item.claim,
            gold=item.label,
            predicted=normalise_label(judgement.verdict),
            confidence=float(judgement.confidence),
            rationale=judgement.rationale[:300],
        )

    async def predict(self, items: list[LabelledClaim]) -> list[BaselinePrediction]:
        log.info("running single-LLM baseline", n=len(items))
        results = await asyncio.gather(
            *(self.predict_one(item) for item in items), return_exceptions=True
        )

        out: list[BaselinePrediction] = []
        for item, result in zip(items, results, strict=True):
            if isinstance(result, BaseException):
                out.append(
                    BaselinePrediction(
                        claim=item.claim,
                        gold=item.label,
                        predicted="NEI",
                        confidence=0.0,
                        rationale=str(result)[:200],
                    )
                )
            else:
                out.append(result)
        return out
