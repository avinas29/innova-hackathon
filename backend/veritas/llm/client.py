"""Provider-agnostic LLM access.

Three backends implement one interface:

* :class:`OpenAIProvider`     — OpenAI chat completions + JSON mode
* :class:`AnthropicProvider`  — Anthropic messages + forced tool use for JSON
* :class:`OfflineProvider`    — deterministic heuristics, no network

``OfflineProvider`` exists so the complete graph, API and test suite run with no
API keys at all. Its outputs come from real (if simple) lexical algorithms —
sentence segmentation, token-overlap entailment, TF-weighted extraction. They
are **not** model outputs and are never presented as such: every result carries
``provider="offline"`` and the API surfaces that in ``/health``.

Structured generation is validated against the caller's Pydantic model, with a
single repair round-trip that feeds the parse error back to the model. That
one retry catches the overwhelming majority of malformed-JSON failures.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from veritas.config import Settings, get_settings
from veritas.logging import get_logger
from veritas.schemas import TokenUsage

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

Role = str  # "system" | "user" | "assistant"


class LLMError(RuntimeError):
    """Unrecoverable model failure after retries."""


class TransientLLMError(LLMError):
    """Retryable failure — rate limit, timeout, 5xx.

    Carries ``retry_after`` when the provider told us how long to wait. Guessing
    with exponential backoff is strictly worse than obeying a server that has
    stated the exact number.
    """

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# Google returns both a prose "Please retry in 49.25s" and a structured
# retryDelay field. Match either.
_RETRY_AFTER_RE = re.compile(
    r"(?:retry[_ ]?(?:delay|after|in)\D{0,12}|['\"]retryDelay['\"]\s*:\s*['\"]?)"
    r"(\d+(?:\.\d+)?)\s*s",
    re.IGNORECASE,
)


# Providers name the exhausted quota. A per-minute cap clears in a minute and is
# worth waiting for; a per-DAY cap will not clear today, and retrying it five
# times at ~60s each turns one dead call into five minutes of dead air.
_DAILY_QUOTA_MARKERS = (
    "perdayperproject",
    "requestsperday",
    "generaterequestsperday",
    "per day",
    "daily limit",
    "quota exceeded for metric",
)


def is_daily_quota_exhausted(text: str) -> bool:
    """True when the provider says the *daily* allowance is gone."""
    lowered = text.lower().replace("_", "").replace("-", "")
    if "perminuteperproject" in lowered or "requestsperminute" in lowered:
        return False  # a per-minute cap: worth waiting out
    return any(marker.replace("_", "").replace("-", "") in lowered
               for marker in _DAILY_QUOTA_MARKERS)


class DailyQuotaExhausted(LLMError):
    """The provider's daily allowance is gone; retrying cannot help today."""


def parse_retry_after(text: str) -> float | None:
    """Seconds the provider asked us to wait, if it said."""
    match = _RETRY_AFTER_RE.search(text)
    if not match:
        return None
    try:
        # Cap at two minutes: a longer delay means the daily quota is gone and
        # blocking the run on it helps nobody.
        return min(120.0, float(match.group(1)))
    except ValueError:
        return None


@dataclass(slots=True)
class Message:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMResult:
    text: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    provider: str = ""
    model: str = ""
    cached: bool = False


def system(content: str) -> Message:
    return Message("system", content)


def user(content: str) -> Message:
    return Message("user", content)


def assistant(content: str) -> Message:
    return Message("assistant", content)


def _classify(exc: Exception) -> LLMError:
    """Map provider SDK exceptions onto our retryable / fatal split."""
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    transient_markers = (
        "ratelimit",
        "timeout",
        "connection",
        "apiconnection",
        "internalserver",
        "serviceunavailable",
        "overloaded",
        "429",
        "500",
        "502",
        "503",
        "529",
    )
    if is_daily_quota_exhausted(text):
        return DailyQuotaExhausted(
            "daily quota exhausted for this model — retrying will not help today. "
            "Switch provider (VERITAS_LLM_PROVIDER=groq) or wait for the reset. "
            f"Provider said: {str(exc)[:200]}"
        )
    if any(m in name or m in text for m in transient_markers):
        return TransientLLMError(str(exc), retry_after=parse_retry_after(str(exc)))
    return LLMError(str(exc))


def _retry_wait(retry_state) -> float:
    """Obey the provider's stated retry delay, else exponential backoff.

    A 429 that says "retry in 49s" cannot be satisfied by a 20s-capped backoff:
    every attempt lands inside the same closed window and the call fails after
    exhausting its retries for no reason.
    """
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after:
        return float(retry_after) + 0.5
    return min(30.0, 2.0 ** retry_state.attempt_number) + random.uniform(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Base provider
# ─────────────────────────────────────────────────────────────────────────────


class BaseProvider(ABC):
    name: str = "base"

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult: ...

    async def aclose(self) -> None:  # pragma: no cover - most providers need nothing
        return None


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI
# ─────────────────────────────────────────────────────────────────────────────


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError(
                "openai package not installed. Install with: pip install 'veritas[openai]'"
            ) from exc
        self._client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=120.0, max_retries=0
        )

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=stop_after_attempt(5),
        wait=_retry_wait,
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [m.as_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise _classify(exc) from exc

        choice = resp.choices[0]
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
            calls=1,
        )
        return LLMResult(
            text=choice.message.content or "",
            usage=usage,
            provider=self.name,
            model=model,
        )

    async def aclose(self) -> None:
        await self._client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Gemini (via Google's OpenAI-compatible endpoint)
# ─────────────────────────────────────────────────────────────────────────────


class GeminiProvider(OpenAIProvider):
    """Gemini through Google's OpenAI-compatible shim.

    Google exposes ``/chat/completions`` and ``/embeddings`` at a compatibility
    base URL, so the entire OpenAI code path is reused rather than duplicated
    against a second SDK. Two behaviours differ enough to handle explicitly:

    1. **JSON mode coverage varies by model.** Where ``response_format`` is
       rejected, we retry once without it — the prompt already demands bare JSON
       and ``extract_json`` tolerates prose, so the structured path still works.
    2. **Free-tier quotas are small** (10-15 RPM). Pacing is handled upstream by
       the client's rate limiter; here we simply surface 429s as retryable.
    """

    name = "gemini"

    # Gemini 3.x models spend output budget on internal reasoning before
    # emitting anything. Measured: at max_tokens=64 the response was a truncated
    # fragment ("Here", "```json") rather than usable output; at 800 the same
    # prompt returned clean, complete JSON. Several of our calls legitimately
    # request only 300-512 tokens, so a floor is applied rather than raising
    # every call site — the models stop when done and bill actual usage, so the
    # headroom costs nothing when unused.
    MIN_OUTPUT_TOKENS = 1536

    def __init__(self, api_key: str, base_url: str, model_hint: str = "") -> None:
        super().__init__(api_key, base_url=base_url)
        self.model_hint = model_hint
        self._json_mode_supported = True

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        max_tokens = max(max_tokens, self.MIN_OUTPUT_TOKENS)
        use_json = json_mode and self._json_mode_supported
        try:
            result = await super().complete(
                messages, model, temperature, max_tokens, use_json
            )
        except LLMError as exc:
            if use_json and _is_response_format_error(exc):
                # Latch it off so we pay this probe once, not per call.
                log.warning(
                    "model rejected response_format — falling back to prompt-only JSON",
                    model=model,
                )
                self._json_mode_supported = False
                result = await super().complete(
                    messages, model, temperature, max_tokens, False
                )
            else:
                raise

        result.provider = self.name
        return result


class GroqProvider(OpenAIProvider):
    """Groq through its OpenAI-compatible endpoint.

    Measured on a live free key, against Gemini's free tier:

    ===========================  ========  ========  =========  ======
    model                        latency   req/day   tokens/min  JSON
    ===========================  ========  ========  =========  ======
    llama-3.1-8b-instant           0.3s     14,400     6,000     yes
    llama-3.3-70b-versatile        0.4s      1,000    12,000     yes
    openai/gpt-oss-120b            0.8s      1,000     8,000     yes
    (gemini-3.5-flash)             1.6s        250     n/a       yes
    ===========================  ========  ========  =========  ======

    Two orders of magnitude more requests per day, and several times faster.
    The trade-off is that throughput is capped by *tokens* per minute rather
    than requests, which the rate limiter handles explicitly.
    """

    name = "groq"


_RESPONSE_FORMAT_MARKERS = (
    "response_format",
    "response mime type",
    "json_object",
    "unknown name",
    "invalid argument",
)


def _is_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _RESPONSE_FORMAT_MARKERS)


# ─────────────────────────────────────────────────────────────────────────────
# Anthropic
# ─────────────────────────────────────────────────────────────────────────────


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise LLMError(
                "anthropic package not installed. Install with: pip install 'veritas[anthropic]'"
            ) from exc
        self._client = AsyncAnthropic(api_key=api_key, timeout=120.0, max_retries=0)

    @retry(
        retry=retry_if_exception_type(TransientLLMError),
        stop=stop_after_attempt(5),
        wait=_retry_wait,
        reraise=True,
    )
    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        # Anthropic takes the system prompt as a top-level parameter rather than
        # a message, so it has to be lifted out of the list.
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [m.as_dict() for m in messages if m.role != "system"]
        if not convo:
            convo = [{"role": "user", "content": "Proceed."}]

        if json_mode:
            # Nudge toward bare JSON; the extractor below tolerates prose anyway.
            system_text += (
                "\n\nRespond with a single valid JSON object and nothing else. "
                "No markdown fences, no commentary."
            )

        try:
            resp = await self._client.messages.create(
                model=model,
                system=system_text or "You are a precise research assistant.",
                messages=convo,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise _classify(exc) from exc

        text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "input_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "output_tokens", 0) or 0,
            calls=1,
        )
        return LLMResult(text=text, usage=usage, provider=self.name, model=model)

    async def aclose(self) -> None:
        await self._client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Offline deterministic provider
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset(
    """a an and are as at be been but by for from had has have he her his if in into is it its
    of on or that the their there they this to was were which who will with would you your can
    could should may might must not no nor so than then these those we our us i""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens with stopwords removed."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Sentence segmentation that respects common abbreviations and decimals."""
    protected = re.sub(
        r"\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Co|St|vs|etc|e\.g|i\.e|Fig|No|Vol|pp)\.",
        lambda m: m.group(0).replace(".", "\x00"),
        text,
    )
    # Not a raw string: \x00 must be a literal NUL here, and `re` would reject
    # "\x" as an unknown escape inside a raw replacement template.
    protected = re.sub(r"(\d)\.(\d)", "\\1\x00\\2", protected)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\[])", protected)
    return [p.replace("\x00", ".").strip() for p in parts if p.replace("\x00", ".").strip()]


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity — the offline entailment proxy."""
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def containment(needle: str, haystack: str) -> float:
    """Share of the needle's tokens present in the haystack."""
    sa, sb = set(tokenize(needle)), set(tokenize(haystack))
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


class OfflineProvider(BaseProvider):
    """Deterministic, network-free stand-in used by tests, CI and demos.

    Every response is a function of the prompt, so runs are reproducible. The
    ``task`` hint attached to each request selects a purpose-built heuristic;
    unknown tasks fall back to schema-shaped minimal output.
    """

    name = "offline"

    async def complete(
        self,
        messages: list[Message],
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResult:
        prompt = "\n".join(m.content for m in messages)
        task = _extract_task(prompt)
        text = _offline_dispatch(task, prompt)
        approx = max(1, len(prompt) // 4)
        return LLMResult(
            text=text,
            usage=TokenUsage(prompt_tokens=approx, completion_tokens=64, calls=1),
            provider=self.name,
            model="offline-heuristic",
        )


_TASK_RE = re.compile(r"<task>([a-z_]+)</task>")


def _extract_task(prompt: str) -> str:
    m = _TASK_RE.search(prompt)
    return m.group(1) if m else "generic"


def _payload_block(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", prompt, re.DOTALL)
    return m.group(1).strip() if m else ""


def _offline_dispatch(task: str, prompt: str) -> str:
    handler = {
        "plan": _offline_plan,
        "claim_extraction": _offline_claims,
        "checkworthiness": _offline_checkworthy,
        "query_generation": _offline_queries,
        "entailment": _offline_entailment,
        "adjudication": _offline_adjudication,
        "contradiction": _offline_contradiction,
        "synthesis": _offline_synthesis,
        "summary": _offline_summary,
    }.get(task)
    if handler is not None:
        return handler(prompt)
    return json.dumps({"result": "", "note": "offline provider: no handler for this task"})


def _offline_plan(prompt: str) -> str:
    topic = _payload_block(prompt, "topic") or "the topic"
    angles = [
        ("What is the current factual state of {t}?", "WEB", 5),
        ("What quantitative evidence or measurements exist for {t}?", "WEB", 4),
        ("What does peer-reviewed research say about {t}?", "ACADEMIC", 4),
        ("What are the main criticisms or counter-arguments about {t}?", "WEB", 3),
        ("What has changed most recently regarding {t}?", "WEB", 3),
    ]
    return json.dumps(
        {
            "scope_notes": f"Offline plan covering current state, evidence, and critique of {topic}.",
            "questions": [
                {
                    "question": q.format(t=topic),
                    "rationale": "Standard research angle.",
                    "kind": kind,
                    "priority": pri,
                }
                for q, kind, pri in angles
            ],
        }
    )


def _offline_claims(prompt: str) -> str:
    draft = _payload_block(prompt, "draft")
    claims = []
    for sent in split_sentences(draft):
        if len(sent) < 25:
            continue
        # A sentence is claim-like if it carries a number, a date, a proper noun,
        # or a declarative copula.
        has_signal = bool(
            re.search(r"\d", sent)
            or re.search(r"\b[A-Z][a-z]{2,}", sent)
            or re.search(r"\b(is|are|was|were|has|have|reached|reduces|increases)\b", sent)
        )
        if has_signal:
            claims.append({"text": sent, "decontextualised": sent, "source_sentence": sent})
        if len(claims) >= 25:
            break
    return json.dumps({"claims": claims})


def _offline_checkworthy(prompt: str) -> str:
    claims_raw = _payload_block(prompt, "claims")
    out = []
    for line in claims_raw.splitlines():
        line = line.strip()
        if not line:
            continue
        cid, _, text = line.partition("|")
        cid, text = cid.strip(), text.strip()
        subjective = bool(
            re.search(
                r"\b(best|worst|beautiful|should|ought|arguably|amazing|terrible|believe)\b",
                text.lower(),
            )
        )
        numeric = bool(re.search(r"\d", text))
        if subjective:
            cat, score = "NON_FACTUAL", 0.1
        elif numeric or len(tokenize(text)) > 6:
            cat, score = "CHECK_WORTHY", 0.85
        else:
            cat, score = "FACTUAL_UNIMPORTANT", 0.4
        out.append({"id": cid, "category": cat, "score": score})
    return json.dumps({"assessments": out})


def _offline_queries(prompt: str) -> str:
    claim = _payload_block(prompt, "claim")
    toks = tokenize(claim)[:8]
    base = " ".join(toks)
    return json.dumps(
        {
            "queries": [
                base,
                f"{base} evidence study",
                f"{base} criticism OR refuted OR incorrect",
            ]
        }
    )


def _offline_entailment(prompt: str) -> str:
    claim = _payload_block(prompt, "claim")
    evidence = _payload_block(prompt, "evidence")
    cov = containment(claim, evidence)
    negated = bool(
        re.search(r"\b(not|no|never|denies|denied|false|incorrect|refutes)\b", evidence.lower())
    )
    if cov >= 0.55 and not negated:
        stance, score = "SUPPORTS", min(0.95, 0.5 + cov / 2)
    elif cov >= 0.55 and negated:
        stance, score = "REFUTES", min(0.9, 0.4 + cov / 2)
    elif cov >= 0.25:
        stance, score = "NEUTRAL", cov
    else:
        stance, score = "NEUTRAL", cov
    return json.dumps(
        {
            "stance": stance,
            "score": round(score, 3),
            "relevance": round(cov, 3),
            "reasoning": f"Offline lexical overlap {cov:.2f}; negation={negated}.",
        }
    )


def _offline_adjudication(prompt: str) -> str:
    support = len(re.findall(r"\bSUPPORTS\b", prompt))
    refute = len(re.findall(r"\bREFUTES\b", prompt))
    if support > refute and support > 0:
        verdict, conf = "SUPPORTED", 0.7
    elif refute > support and refute > 0:
        verdict, conf = "REFUTED", 0.7
    else:
        verdict, conf = "NEI", 0.3
    return json.dumps(
        {
            "verdict": verdict,
            "confidence": conf,
            "rationale": f"Offline tally: {support} supporting vs {refute} refuting clusters.",
            "minority_report": "" if support == 0 or refute == 0 else "Evidence is split.",
        }
    )


def _offline_contradiction(prompt: str) -> str:
    a = _payload_block(prompt, "evidence_a")
    b = _payload_block(prompt, "evidence_b")
    overlap = jaccard(a, b)
    neg_a = bool(re.search(r"\b(not|no|never|denies|false)\b", a.lower()))
    neg_b = bool(re.search(r"\b(not|no|never|denies|false)\b", b.lower()))
    conflict = overlap > 0.3 and (neg_a != neg_b)
    return json.dumps(
        {
            "contradictory": conflict,
            "score": round(overlap if conflict else 0.0, 3),
            "explanation": (
                "Offline: high topical overlap with opposing polarity."
                if conflict
                else "Offline: no polarity conflict detected."
            ),
        }
    )


def _offline_synthesis(prompt: str) -> str:
    topic = _payload_block(prompt, "topic") or "the topic"
    findings = _payload_block(prompt, "findings")
    sents = split_sentences(findings)[:12]
    body = "\n\n".join(f"{s}" for s in sents) if sents else "No evidence was retrieved."
    return json.dumps(
        {
            "executive_summary": f"Offline synthesis of retrieved material on {topic}.",
            "body_markdown": f"## Findings on {topic}\n\n{body}",
        }
    )


def _offline_summary(prompt: str) -> str:
    content = _payload_block(prompt, "content")
    sents = split_sentences(content)
    # TF-weighted extractive selection: score sentences by mean term frequency.
    freq: dict[str, int] = {}
    for tok in tokenize(content):
        freq[tok] = freq.get(tok, 0) + 1
    scored = []
    for s in sents:
        toks = tokenize(s)
        if not toks:
            continue
        scored.append((sum(freq.get(t, 0) for t in toks) / math.sqrt(len(toks)), s))
    scored.sort(key=lambda x: -x[0])
    top = [s for _, s in scored[:5]]
    ordered = [s for s in sents if s in top]
    return json.dumps({"summary": " ".join(ordered)})


# ─────────────────────────────────────────────────────────────────────────────
# Client facade
# ─────────────────────────────────────────────────────────────────────────────


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Recover a JSON value from model output.

    Tries, in order: the whole string, a fenced code block, then the widest
    balanced ``{...}`` or ``[...]`` span. Raises ``ValueError`` if none parse.
    """
    text = text.strip()
    if not text:
        raise ValueError("empty response")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fence = _JSON_FENCE.search(text)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no JSON object found in response: {text[:200]!r}")


class LLMClient:
    """The single entry point every agent uses to talk to a model."""

    def __init__(self, settings: Settings | None = None, provider: BaseProvider | None = None):
        from veritas.llm.ratelimit import ModelRateLimiters

        self.settings = settings or get_settings()

        # One provider per role. They may be the same object (the common case)
        # or two different backends when the roles are split across providers
        # to draw on two free-tier quotas at once.
        if provider is not None:
            self._by_role: dict[str, BaseProvider] = {"fast": provider, "strong": provider}
        else:
            fast = self._build_provider(self.settings.provider_for("fast"))
            strong = (
                fast
                if self.settings.provider_for("strong") == self.settings.provider_for("fast")
                else self._build_provider(self.settings.provider_for("strong"))
            )
            self._by_role = {"fast": fast, "strong": strong}

        if self.settings.split_providers:
            log.info(
                "roles split across providers",
                fast=f"{self._by_role['fast'].name}:{self.settings.model_for('fast')}",
                strong=f"{self._by_role['strong'].name}:{self.settings.model_for('strong')}",
            )

        self._limiters = ModelRateLimiters(self.settings.effective_rate_limits())
        if self._limiters.enabled:
            log.info(
                "client-side rate limiting active",
                providers=",".join(sorted({p.name for p in self._by_role.values()})),
                limits=self.settings.rate_limit_summary(),
            )

        self._usage = TokenUsage()
        self._usage_lock = asyncio.Lock()
        self._cache_hits = 0

    def _build_provider(self, kind: str | None = None) -> BaseProvider:
        kind = kind or self.settings.resolved_provider
        if kind == "openai":
            return OpenAIProvider(self.settings.openai_api_key)
        if kind == "anthropic":
            return AnthropicProvider(self.settings.anthropic_api_key)
        if kind == "gemini":
            from veritas.config import GEMINI_BASE_URL

            return GeminiProvider(self.settings.gemini_api_key, GEMINI_BASE_URL)
        if kind == "groq":
            from veritas.config import GROQ_BASE_URL

            return GroqProvider(self.settings.groq_api_key, base_url=GROQ_BASE_URL)
        log.warning("no model API key configured — using deterministic offline provider")
        return OfflineProvider()

    @property
    def rate_limit_stats(self) -> dict[str, dict[str, float]]:
        return self._limiters.stats()

    # ── accounting ───────────────────────────────────────────────────────────
    @property
    def usage(self) -> TokenUsage:
        return self._usage

    @property
    def provider_name(self) -> str:
        """Name of the provider serving the fast role (the default surface)."""
        return self._by_role["fast"].name

    @property
    def provider_names(self) -> dict[str, str]:
        return {role: p.name for role, p in self._by_role.items()}

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    async def _record(self, usage: TokenUsage) -> None:
        async with self._usage_lock:
            self._usage = self._usage.merge(usage)

    def budget_exhausted(self) -> bool:
        return self._usage.total >= self.settings.max_tokens_per_run

    # ── core calls ───────────────────────────────────────────────────────────
    async def chat(
        self,
        messages: list[Message],
        *,
        role: str = "fast",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        task: str = "generic",
        use_cache: bool = True,
    ) -> LLMResult:
        """Single completion, with caching, budget checks and retries."""
        if self.budget_exhausted():
            raise LLMError(
                f"token budget exhausted ({self._usage.total} >= "
                f"{self.settings.max_tokens_per_run})"
            )

        model = self.settings.model_for("strong" if role == "strong" else "fast")
        tagged = _tag_task(messages, task)

        cache_key = None
        if use_cache and self.settings.cache_enabled and temperature == 0.0:
            cache_key = _cache_key(
                self._by_role["strong" if role == "strong" else "fast"].name,
                model,
                tagged,
                json_mode,
            )
            cached = await asyncio.to_thread(_cache_lookup, cache_key, self.settings)
            if cached is not None:
                self._cache_hits += 1
                return LLMResult(
                    text=cached,
                    provider=self._by_role["strong" if role == "strong" else "fast"].name,
                    model=model,
                    cached=True,
                )

        # Some providers apply the per-minute token cap to a SINGLE request, so
        # an oversized prompt is rejected outright and no amount of pacing or
        # retrying helps. Trim it to fit before spending the call.
        budget = self._limiters.token_budget(model)
        tagged, was_trimmed = fit_to_token_budget(tagged, budget, max_tokens)
        if was_trimmed:
            log.warning(
                "prompt trimmed to fit the model's token budget",
                task=task,
                model=model,
                budget=budget,
            )

        # Pace only real network calls: cache hits returned above, so a warm
        # rerun costs no quota and no waiting.
        #
        # Groq caps tokens per minute, so the reservation must cover both the
        # prompt we are about to send and the output we asked for.
        from veritas.llm.ratelimit import estimate_tokens

        estimated = estimate_tokens("".join(m.content for m in tagged)) + max_tokens
        await self._limiters.acquire(model, estimated_tokens=estimated)

        backend = self._by_role["strong" if role == "strong" else "fast"]
        result = await backend.complete(
            tagged, model, temperature, max_tokens, json_mode
        )
        await self._record(result.usage)
        # Replace the estimate with the real cost so the window stays accurate.
        await self._limiters.settle(model, estimated, result.usage.total)

        if cache_key is not None and result.text:
            await asyncio.to_thread(_cache_store, cache_key, result.text, self.settings)

        log.debug(
            "llm call",
            task=task,
            model=model,
            tokens=result.usage.total,
            chars=len(result.text),
        )
        return result

    async def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        role: str = "fast",
        temperature: float = 0.0,
        max_tokens: int = 2048,
        task: str = "generic",
        use_cache: bool = True,
    ) -> T:
        """Completion parsed into ``schema``, with one repair attempt on failure."""
        instruction = system(
            "Return ONLY a JSON object conforming to this schema:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}"
        )
        convo = [*messages, instruction]

        result = await self.chat(
            convo,
            role=role,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            task=task,
            use_cache=use_cache,
        )

        try:
            return schema.model_validate(extract_json(result.text))
        except (ValueError, ValidationError) as exc:
            # Bind outside the handler: Python unbinds `except ... as e` on exit.
            parse_error = str(exc)
            log.debug("structured parse failed, repairing", task=task, error=parse_error[:200])

        repair = [
            *convo,
            assistant(result.text[:4000]),
            user(
                "That response could not be parsed. The error was:\n"
                f"{parse_error}\n\n"
                "Reply with corrected JSON only — no prose, no code fences."
            ),
        ]
        repaired = await self.chat(
            repair,
            role=role,
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            task=task,
            use_cache=False,
        )
        try:
            return schema.model_validate(extract_json(repaired.text))
        except (ValueError, ValidationError) as exc:
            raise LLMError(
                f"structured output failed for task={task} after repair: {exc}"
            ) from exc

    async def aclose(self) -> None:
        for backend in {id(p): p for p in self._by_role.values()}.values():
            await backend.aclose()


def fit_to_token_budget(
    messages: list[Message], budget_tokens: int, reserved_output: int
) -> tuple[list[Message], bool]:
    """Shrink a prompt so the whole request fits inside a per-minute cap.

    Some providers enforce tokens-per-minute as a *per-request* ceiling too, so
    a single call larger than the budget is rejected with HTTP 413 forever —
    retries and pacing cannot help. Observed on Groq::

        Request too large for openai/gpt-oss-120b ... TPM: Limit 8000,
        Requested 8658

    Synthesis and report generation are the calls at risk: both concatenate the
    entire evidence corpus, so their size scales with how much research
    succeeded. Trimming the largest message keeps the call alive with slightly
    less context, which is strictly better than losing the step entirely.

    Returns the (possibly trimmed) messages and whether trimming occurred.
    """
    if budget_tokens <= 0:
        return messages, False

    # Leave headroom: the estimate is approximate and the provider counts the
    # requested output against the same budget.
    allowance = budget_tokens - reserved_output - _TOKEN_BUDGET_HEADROOM
    if allowance <= 0:
        allowance = max(256, budget_tokens // 2)

    total = sum(_estimate(m.content) for m in messages)
    if total <= allowance:
        return messages, False

    # Trim only the largest message — usually the evidence corpus. System
    # prompts carry the rules and must survive intact.
    largest = max(range(len(messages)), key=lambda i: len(messages[i].content))
    others = total - _estimate(messages[largest].content)
    keep_tokens = max(256, allowance - others)
    keep_chars = keep_tokens * 4

    trimmed = list(messages)
    original = trimmed[largest].content
    if len(original) > keep_chars:
        trimmed[largest] = Message(
            trimmed[largest].role,
            original[:keep_chars] + "\n\n[…truncated to fit the model's token budget]",
        )
    return trimmed, True


_TOKEN_BUDGET_HEADROOM = 512


def _estimate(text: str) -> int:
    return max(1, len(text) // 4)


def _tag_task(messages: list[Message], task: str) -> list[Message]:
    """Embed the task label so the offline provider (and logs) can see it."""
    tag = f"<task>{task}</task>"
    if not messages:
        return [user(tag)]
    head, *rest = messages
    return [Message(head.role, f"{tag}\n{head.content}"), *rest]


def _cache_key(provider: str, model: str, messages: list[Message], json_mode: bool) -> str:
    blob = json.dumps(
        {
            "p": provider,
            "m": model,
            "j": json_mode,
            "msgs": [m.as_dict() for m in messages],
        },
        sort_keys=True,
    )
    return "llm:" + hashlib.sha256(blob.encode()).hexdigest()


def _cache_lookup(key: str, settings: Settings) -> str | None:
    from veritas.storage.db import get_db

    try:
        return get_db().cache_get(key, settings.cache_ttl_seconds)
    except Exception as exc:  # cache must never break a run
        log.debug("cache read failed", error=str(exc))
        return None


def _cache_store(key: str, value: str, settings: Settings) -> None:
    from veritas.storage.db import get_db

    try:
        get_db().cache_set(key, value)
    except Exception as exc:
        log.debug("cache write failed", error=str(exc))
