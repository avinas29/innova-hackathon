"""Client-side rate limiting.

Why this exists
---------------
VERITAS is deliberately parallel: research questions fan out at once, and every
check-worthy claim is verified concurrently. A single modest run issues on the
order of 30-60 model calls in a few seconds.

Gemini's free tier allows **10-15 requests per minute**. Without throttling, a
run fires its whole burst immediately, collects a wall of HTTP 429s, and either
fails or degrades into retry-storm latency that is far *slower* than simply
pacing the requests in the first place.

So the limiter is not a nicety — on a free tier it is the difference between a
run that completes and one that does not.

Two mechanisms:

* :class:`RateLimiter` — sliding-window requests-per-minute gate, applied to
  every outbound model call.
* :class:`DailyQuota`  — best-effort requests-per-day counter that fails loudly
  with a clear message instead of letting a run die confusingly at 80% done.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque

from veritas.logging import get_logger

log = get_logger(__name__)


class RateLimitExceeded(RuntimeError):
    """Daily request quota exhausted."""


class RateLimiter:
    """Async sliding-window rate limiter.

    A sliding window rather than a fixed one: fixed windows permit a double
    burst across a boundary (N at 0:59, N more at 1:01), which is exactly the
    pattern that trips provider-side limits.

    ``rpm <= 0`` disables limiting entirely, which is the default for paid
    providers where client-side pacing only adds latency.
    """

    def __init__(self, rpm: int, name: str = "default", window_seconds: float = 60.0) -> None:
        self.rpm = rpm
        self.name = name
        # Injectable so tests can exercise pacing in milliseconds. The
        # alternative — monkey-patching asyncio.sleep — reaches the real module
        # and breaks pytest-asyncio's loop management; it SIGKILLed the suite.
        self.window_seconds = window_seconds
        self._window: deque[float] = deque()
        self._lock = asyncio.Lock()
        self._waits = 0
        self._wait_seconds = 0.0

    @property
    def enabled(self) -> bool:
        return self.rpm > 0

    @property
    def stats(self) -> dict[str, float]:
        return {
            "rpm": self.rpm,
            "waits": self._waits,
            "wait_seconds": round(self._wait_seconds, 2),
        }

    async def acquire(self) -> None:
        """Block until another request is permitted inside the window."""
        if not self.enabled:
            return

        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - self.window_seconds
                while self._window and self._window[0] <= cutoff:
                    self._window.popleft()

                if len(self._window) < self.rpm:
                    self._window.append(now)
                    return

                # Wait until the oldest request in the window ages out.
                sleep_for = self._window[0] - cutoff + 0.05

            self._waits += 1
            self._wait_seconds += sleep_for
            log.debug(
                "rate limit reached — pacing",
                limiter=self.name,
                rpm=self.rpm,
                sleep=round(sleep_for, 2),
            )
            await asyncio.sleep(sleep_for)


class TokenRateLimiter:
    """Sliding-window limiter on tokens per minute.

    Groq's free tier is generous on requests (1,000-14,400 per *day*) but tight
    on throughput: 6,000-12,000 tokens per minute. A request-only limiter sails
    straight past that, because our calls are large — an adjudication carries
    the full evidence set, so a handful of them exhausts a minute's budget while
    barely denting the request count.

    Token cost is not known until the response arrives, so a call reserves an
    estimate up front and reconciles against real usage afterwards.
    """

    def __init__(self, tpm: int, name: str = "default", window_seconds: float = 60.0) -> None:
        self.tpm = tpm
        self.name = name
        self.window_seconds = window_seconds
        self._window: deque[tuple[float, int]] = deque()
        self._lock = asyncio.Lock()
        self._waits = 0
        self._wait_seconds = 0.0

    @property
    def enabled(self) -> bool:
        return self.tpm > 0

    def _prune(self, now: float) -> int:
        cutoff = now - self.window_seconds
        while self._window and self._window[0][0] <= cutoff:
            self._window.popleft()
        return sum(tokens for _, tokens in self._window)

    async def acquire(self, estimated_tokens: int) -> None:
        """Block until ``estimated_tokens`` fit inside the current minute."""
        if not self.enabled:
            return

        # A single call larger than the whole budget would never fit; let it
        # through rather than deadlock, and let the provider decide.
        estimated_tokens = max(1, min(estimated_tokens, self.tpm))

        while True:
            async with self._lock:
                now = time.monotonic()
                used = self._prune(now)
                if used + estimated_tokens <= self.tpm:
                    self._window.append((now, estimated_tokens))
                    return
                oldest = self._window[0][0]
                sleep_for = max(0.01, oldest + self.window_seconds - now + 0.01)

            self._waits += 1
            self._wait_seconds += sleep_for
            # Visible at INFO: on a token-capped tier this is often the only
            # thing happening for tens of seconds, and silence reads as a hang.
            emit = log.info if sleep_for >= 5 else log.debug
            emit(
                "token budget reached — pacing (this is expected on a free tier)",
                limiter=self.name,
                tpm=self.tpm,
                used=used,
                sleep_seconds=round(sleep_for, 1),
            )
            await asyncio.sleep(sleep_for)

    async def settle(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Replace the reservation with the true cost once it is known."""
        if not self.enabled or not self._window:
            return
        estimated_tokens = max(1, min(estimated_tokens, self.tpm))
        async with self._lock:
            for index in range(len(self._window) - 1, -1, -1):
                stamp, tokens = self._window[index]
                if tokens == estimated_tokens:
                    self._window[index] = (stamp, max(1, actual_tokens))
                    return

    @property
    def stats(self) -> dict[str, float]:
        used = self._prune(time.monotonic())
        return {
            "tpm": self.tpm,
            "tokens_used_this_minute": used,
            "token_waits": self._waits,
            "token_wait_seconds": round(self._wait_seconds, 2),
        }


def estimate_tokens(text: str) -> int:
    """Rough token count for pre-flight reservation.

    ~4 characters per token is the standard English approximation. Only used to
    reserve budget; the reservation is reconciled with real usage afterwards, so
    a modest error self-corrects within one window.
    """
    return max(1, len(text) // 4)


class DailyQuota:
    """Best-effort requests-per-day counter.

    In-process only — it resets when the process does, and cannot see usage from
    other machines. Its job is not enforcement but *diagnosis*: without it, a
    free-tier user hitting the daily cap sees a pile of opaque 429s two thirds of
    the way through a run.
    """

    def __init__(self, limit: int, name: str = "default") -> None:
        self.limit = limit
        self.name = name
        self._count = 0
        self._reset_at = time.time() + 86_400
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self._count) if self.enabled else -1

    async def consume(self) -> None:
        if not self.enabled:
            return

        async with self._lock:
            now = time.time()
            if now >= self._reset_at:
                self._count = 0
                self._reset_at = now + 86_400

            if self._count >= self.limit:
                raise RateLimitExceeded(
                    f"daily request quota exhausted for {self.name}: "
                    f"{self._count}/{self.limit} requests used. "
                    "Free-tier quotas reset every 24 hours — either wait, switch "
                    "provider, or lower VERITAS_MAX_CLAIMS to spend fewer calls."
                )

            self._count += 1
            if self._count == int(self.limit * 0.8):
                log.warning(
                    "80% of daily request quota used",
                    limiter=self.name,
                    used=self._count,
                    limit=self.limit,
                )


# Published free-tier limits, keyed by model id prefix. Sourced from Google's
# rate-limit documentation; Google has revised these without notice before, so
# they are overridable via VERITAS_RPM_LIMIT / VERITAS_DAILY_LIMIT rather than
# being treated as fixed truth.
GEMINI_FREE_TIER: dict[str, tuple[int, int]] = {
    # 2.5 family — retired for keys issued after its deprecation, kept for
    # older keys that can still reach it.
    "gemini-2.5-pro": (5, 100),
    "gemini-2.5-flash-lite": (15, 1000),
    "gemini-2.5-flash": (10, 250),
    # 3.x family — current.
    #
    # The non-lite Flash models are 5 RPM, NOT 10. Confirmed from a live 429:
    #   "limit: 5, model: gemini-3.5-flash"
    #   quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier
    # An earlier guess of 10 here caused the limiter to send at twice the
    # permitted rate, so every run collected 429s partway through.
    "gemini-3.1-flash-lite": (15, 1000),
    "gemini-3.5-flash-lite": (15, 1000),
    "gemini-3.5-flash": (5, 250),
    "gemini-3.6-flash": (5, 250),
    "gemini-3.1-pro": (5, 100),
    "gemini-3": (5, 250),
}


# Groq free tier, read directly from this account's `x-ratelimit-*` response
# headers rather than from docs. Values are (rpm, requests_per_day, tpm).
#
# The reset headers disambiguate the windows: `x-ratelimit-reset-requests: 6s`
# against a 14,400 limit is a *daily* bucket (86400/14400 = 6s per refill),
# while `x-ratelimit-reset-tokens: 570ms` after spending 57 tokens against a
# 6,000 limit is a *per-minute* bucket.
#
# RPM is left at 0 (unlimited) because Groq does not impose a per-minute request
# cap on this tier — throughput is governed entirely by tokens.
GROQ_FREE_TIER: dict[str, tuple[int, int, int]] = {
    "llama-3.1-8b-instant": (0, 14_400, 6_000),
    "llama-3.3-70b-versatile": (0, 1_000, 12_000),
    "openai/gpt-oss-20b": (0, 1_000, 8_000),
    "openai/gpt-oss-120b": (0, 1_000, 8_000),
    "openai/gpt-oss-safeguard-20b": (0, 1_000, 8_000),
    "qwen/qwen3.6-27b": (0, 1_000, 8_000),
    "groq/compound-mini": (0, 1_000, 8_000),
    "groq/compound": (0, 1_000, 8_000),
    "allam-2-7b": (0, 1_000, 6_000),
}


def free_tier_limits(model: str) -> tuple[int, int, int]:
    """Best-known ``(rpm, requests_per_day, tpm)`` for a model's free tier.

    Longest prefix wins so ``gemini-2.5-flash-lite`` is not shadowed by
    ``gemini-2.5-flash``.
    """
    for prefix in sorted(GROQ_FREE_TIER, key=len, reverse=True):
        if model.startswith(prefix):
            return GROQ_FREE_TIER[prefix]

    for prefix in sorted(GEMINI_FREE_TIER, key=len, reverse=True):
        if model.startswith(prefix):
            rpm, rpd = GEMINI_FREE_TIER[prefix]
            return (rpm, rpd, 0)  # Gemini's TPM ceiling is not the binding limit

    # Unknown models get the strictest observed free-tier rate. Guessing high
    # produces 429 storms; guessing low only costs a little latency.
    return (5, 100, 0)


class ModelRateLimiters:
    """Per-model limiters across all three axes: RPM, requests/day, and TPM.

    Quotas are **per model**, not per project. Pacing everything against the
    strictest model in play wastes most of the allowance: the fast model carries
    the bulk of the traffic and usually has a far larger budget than the strong
    one, so a shared limiter would throttle ~80% of calls for no reason.

    Which axis binds depends on the provider, so all three are enforced:

    * **Gemini** — requests per minute (5-15). Tiny; RPM binds.
    * **Groq** — requests per *day* (1,000-14,400) and tokens per minute
      (6,000-12,000). Requests are effectively free; TPM binds.
    """

    def __init__(self, limits: dict[str, tuple[int, int, int]]) -> None:
        self._limiters = {
            model: RateLimiter(rpm, name=model) for model, (rpm, _, _) in limits.items()
        }
        self._quotas = {
            model: DailyQuota(rpd, name=model) for model, (_, rpd, _) in limits.items()
        }
        self._tokens = {
            model: TokenRateLimiter(tpm, name=model) for model, (_, _, tpm) in limits.items()
        }

    @property
    def enabled(self) -> bool:
        return any(
            limiter.enabled
            for group in (self._limiters, self._tokens)
            for limiter in group.values()
        ) or any(quota.enabled for quota in self._quotas.values())

    async def acquire(self, model: str, estimated_tokens: int = 0) -> None:
        """Reserve budget on every axis for one call to ``model``.

        A model with no configured limit passes straight through — an unknown
        model should not be silently throttled to some other model's budget.
        """
        quota = self._quotas.get(model)
        if quota is not None:
            await quota.consume()

        limiter = self._limiters.get(model)
        if limiter is not None:
            await limiter.acquire()

        tokens = self._tokens.get(model)
        if tokens is not None and estimated_tokens > 0:
            await tokens.acquire(estimated_tokens)

    async def settle(self, model: str, estimated_tokens: int, actual_tokens: int) -> None:
        """Reconcile a token reservation against real usage."""
        tokens = self._tokens.get(model)
        if tokens is not None and estimated_tokens > 0 and actual_tokens > 0:
            await tokens.settle(estimated_tokens, actual_tokens)

    def token_budget(self, model: str) -> int:
        """Per-minute token ceiling for a model, or 0 when uncapped.

        Callers use this to size a prompt *before* sending it: on providers
        that apply the cap per request, an oversized call is a permanent 413.
        """
        limiter = self._tokens.get(model)
        return limiter.tpm if limiter is not None else 0

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            model: {
                **limiter.stats,
                **self._tokens[model].stats,
                "daily_remaining": self._quotas[model].remaining,
            }
            for model, limiter in self._limiters.items()
        }
