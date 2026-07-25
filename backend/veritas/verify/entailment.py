"""Entailment scoring — does this evidence ground this claim?

Two interchangeable backends:

* ``llm``   — an LLM asked to perform NLI under a strict rubric. Default,
              zero extra dependencies, works with any provider.
* ``local`` — a cross-encoder (MiniCheck / HHEM family) run locally. MiniCheck
              reports GPT-4-level grounding accuracy at 770M parameters and
              ~400× lower cost, which is the right trade at high claim volume.

Why the LLM backend is the default despite that: ``Bespoke-MiniCheck-7B`` is
licensed for non-commercial use only, and the small variants pull in ~2GB of
torch wheels. A hackathon build should not fail at `pip install`, and a
submission should not ship an unresolved licence question. The local backend is
one env var away for anyone who wants it.

The critical rubric point, encoded in the prompt: this measures **grounding in
the supplied evidence**, not truth. A model that answers from its own knowledge
here destroys the entire measurement — it is precisely the single-LLM behaviour
we claim to beat.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from veritas.config import Settings, get_settings
from veritas.llm.client import LLMClient, containment, system, user
from veritas.logging import get_logger
from veritas.prompts import ENTAILMENT_SYSTEM, ENTAILMENT_USER
from veritas.schemas import Stance

log = get_logger(__name__)


class EntailmentJudgement(BaseModel):
    stance: str = "NEUTRAL"
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""

    def parsed_stance(self) -> Stance:
        normalised = (self.stance or "").strip().upper()
        if normalised.startswith("SUPPORT") or normalised == "ENTAILMENT":
            return Stance.SUPPORTS
        if normalised.startswith("REFUT") or normalised in {"CONTRADICTION", "CONTRADICT"}:
            return Stance.REFUTES
        return Stance.NEUTRAL


class EntailmentBackend(ABC):
    name = "base"

    @abstractmethod
    async def score(self, claim: str, evidence: str, domain: str = "") -> EntailmentJudgement: ...

    async def score_batch(
        self, claim: str, evidences: list[tuple[str, str]], concurrency: int = 8
    ) -> list[EntailmentJudgement]:
        sem = asyncio.Semaphore(concurrency)

        async def one(text: str, domain: str) -> EntailmentJudgement:
            async with sem:
                try:
                    return await self.score(claim, text, domain)
                except Exception as exc:
                    log.warning("entailment scoring failed", error=str(exc)[:160])
                    return EntailmentJudgement(
                        stance="NEUTRAL", score=0.0, reasoning=f"scoring error: {exc}"[:200]
                    )

        return list(await asyncio.gather(*(one(t, d) for t, d in evidences)))


class LLMEntailment(EntailmentBackend):
    name = "llm"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def score(self, claim: str, evidence: str, domain: str = "") -> EntailmentJudgement:
        judgement = await self.llm.structured(
            [
                system(ENTAILMENT_SYSTEM),
                user(
                    ENTAILMENT_USER.format(
                        claim=claim, evidence=evidence[:6000], domain=domain or "unknown"
                    )
                ),
            ],
            EntailmentJudgement,
            role="fast",
            task="entailment",
            max_tokens=512,
        )
        # Guard against the common failure of a confident label on evidence that
        # barely mentions the claim's subject matter.
        overlap = containment(claim, evidence)
        if overlap < 0.15 and judgement.parsed_stance() is not Stance.NEUTRAL:
            log.debug("downgrading entailment on near-zero lexical overlap", overlap=overlap)
            return EntailmentJudgement(
                stance="NEUTRAL",
                score=0.0,
                relevance=overlap,
                reasoning=f"Downgraded: negligible overlap with claim ({overlap:.2f}). "
                + judgement.reasoning[:150],
            )
        return judgement


class LocalCrossEncoderEntailment(EntailmentBackend):
    """Local cross-encoder backend (MiniCheck / HHEM family).

    Loaded lazily so importing this module never costs a model load. Emits a
    binary supported/unsupported probability, which we map onto our three-way
    stance using a negation cue for the REFUTES split — the cross-encoders in
    this family score *grounding*, not direction.
    """

    name = "local"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._lock = asyncio.Lock()

    async def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is not None:
                return
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - optional extra
                raise RuntimeError(
                    "local entailment backend requires: pip install 'veritas[local-nli]'"
                ) from exc

            log.info("loading local entailment model", model=self.model_name)
            self._tokenizer = await asyncio.to_thread(AutoTokenizer.from_pretrained, self.model_name)
            self._model = await asyncio.to_thread(
                AutoModelForSeq2SeqLM.from_pretrained, self.model_name
            )
            self._model.eval()

    async def score(self, claim: str, evidence: str, domain: str = "") -> EntailmentJudgement:
        await self._ensure_loaded()
        probability = await asyncio.to_thread(self._infer, claim, evidence)

        import re

        negated = bool(
            re.search(
                r"\b(not|no|never|denies|denied|false|incorrect|refutes|contrary)\b",
                evidence.lower(),
            )
        )
        overlap = containment(claim, evidence)

        if probability >= 0.5:
            stance, score = Stance.SUPPORTS, probability
        elif negated and overlap >= 0.4:
            stance, score = Stance.REFUTES, 1.0 - probability
        else:
            stance, score = Stance.NEUTRAL, 0.0

        return EntailmentJudgement(
            stance=stance.value,
            score=round(float(score), 4),
            relevance=round(overlap, 4),
            reasoning=f"{self.model_name}: grounding probability {probability:.3f}",
        )

    def _infer(self, claim: str, evidence: str) -> float:
        import torch

        prompt = (
            f"predict: premise: {evidence[:4000]} hypothesis: {claim}"
        )
        inputs = self._tokenizer(  # type: ignore[union-attr]
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        with torch.no_grad():
            outputs = self._model(  # type: ignore[misc]
                **inputs,
                decoder_input_ids=torch.zeros((1, 1), dtype=torch.long),
            )
        logits = outputs.logits[0, 0]
        # Flan-T5 MiniCheck emits "0"/"1"; compare those two token logits.
        zero_id = self._tokenizer.convert_tokens_to_ids("▁0")  # type: ignore[union-attr]
        one_id = self._tokenizer.convert_tokens_to_ids("▁1")  # type: ignore[union-attr]
        pair = torch.tensor([logits[zero_id], logits[one_id]])
        return float(torch.softmax(pair, dim=0)[1].item())


def build_entailment_backend(
    llm: LLMClient, settings: Settings | None = None
) -> EntailmentBackend:
    settings = settings or get_settings()
    if settings.entailment_backend == "local":
        try:
            return LocalCrossEncoderEntailment(settings.local_nli_model)
        except Exception as exc:
            log.warning("local NLI unavailable — falling back to LLM backend", error=str(exc)[:200])
    return LLMEntailment(llm)
