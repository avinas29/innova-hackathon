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

    def __init__(self, rpm: int, name: str = "default") -> None:
        self.rpm = rpm
        self.name = name
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
                cutoff = now - 60.0
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


def free_tier_limits(model: str) -> tuple[int, int]:
    """Best-known ``(rpm, rpd)`` for a Gemini model on the free tier.

    Longest prefix wins so ``gemini-2.5-flash-lite`` is not shadowed by
    ``gemini-2.5-flash``.
    """
    for prefix in sorted(GEMINI_FREE_TIER, key=len, reverse=True):
        if model.startswith(prefix):
            return GEMINI_FREE_TIER[prefix]
    # Unknown models get the strictest observed free-tier rate. Guessing high
    # produces 429 storms; guessing low only costs a little latency.
    return (5, 100)


class ModelRateLimiters:
    """Per-model limiters.

    Gemini's quotas are **per model**, not per project. Pacing everything
    against the strictest model in play wastes most of the allowance: the fast
    model carries the bulk of the traffic and typically has 4× the daily quota
    of the strong one, so a shared limiter would throttle ~90% of calls to the
    strong model's budget for no reason.

    Tracking each model separately gives roughly four times the usable free-tier
    throughput for the same keys.
    """

    def __init__(self, limits: dict[str, tuple[int, int]]) -> None:
        self._limiters = {
            model: RateLimiter(rpm, name=model) for model, (rpm, _) in limits.items()
        }
        self._quotas = {
            model: DailyQuota(rpd, name=model) for model, (_, rpd) in limits.items()
        }

    @property
    def enabled(self) -> bool:
        return any(limiter.enabled for limiter in self._limiters.values())

    async def acquire(self, model: str) -> None:
        """Consume daily quota and pace, for one specific model.

        A model with no configured limit passes straight through — an unknown
        model should not be silently throttled to some other model's budget.
        """
        quota = self._quotas.get(model)
        if quota is not None:
            await quota.consume()

        limiter = self._limiters.get(model)
        if limiter is not None:
            await limiter.acquire()

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            model: {
                **limiter.stats,
                "daily_remaining": self._quotas[model].remaining,
            }
            for model, limiter in self._limiters.items()
        }
