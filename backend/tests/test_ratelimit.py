"""Rate limiting, quota accounting, and Gemini provider selection.

These guard the free-tier path. VERITAS fans out aggressively by design, and a
free Gemini key allows 10-15 requests per minute — without pacing, a run fires
its whole burst at once and collapses into 429s.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from veritas.config import GEMINI_BASE_URL, Settings, reset_settings_cache
from veritas.llm.ratelimit import (
    DailyQuota,
    RateLimiter,
    RateLimitExceeded,
    free_tier_limits,
)


class TestRateLimiter:
    async def test_disabled_when_rpm_is_zero(self):
        limiter = RateLimiter(0)
        assert not limiter.enabled

        started = time.monotonic()
        for _ in range(50):
            await limiter.acquire()
        assert time.monotonic() - started < 0.2

    async def test_allows_a_full_window_without_waiting(self):
        limiter = RateLimiter(10)
        started = time.monotonic()
        for _ in range(10):
            await limiter.acquire()
        assert time.monotonic() - started < 0.2
        assert limiter.stats["waits"] == 0

    async def test_paces_once_the_window_is_full(self, monkeypatch):
        """The 11th call within a minute must wait, not fail."""
        limiter = RateLimiter(3)
        slept: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            slept.append(seconds)
            # Age the window so the loop makes progress without real waiting.
            limiter._window.clear()

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        for _ in range(4):
            await limiter.acquire()

        assert slept, "the fourth call should have paced"
        assert limiter.stats["waits"] == 1

    async def test_is_safe_under_concurrency(self, monkeypatch):
        """Concurrent callers must not collectively exceed the window."""
        limiter = RateLimiter(5)

        async def fake_sleep(seconds: float) -> None:
            limiter._window.clear()

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        await asyncio.gather(*(limiter.acquire() for _ in range(5)))
        assert len(limiter._window) == 5


class TestDailyQuota:
    async def test_disabled_when_limit_is_zero(self):
        quota = DailyQuota(0)
        for _ in range(100):
            await quota.consume()
        assert quota.remaining == -1

    async def test_counts_down(self):
        quota = DailyQuota(3)
        await quota.consume()
        assert quota.remaining == 2

    async def test_raises_a_clear_error_at_the_cap(self):
        """A quota wall must be diagnosable, not a pile of opaque 429s."""
        quota = DailyQuota(2, name="gemini")
        await quota.consume()
        await quota.consume()

        with pytest.raises(RateLimitExceeded) as exc:
            await quota.consume()

        message = str(exc.value)
        assert "gemini" in message
        assert "2/2" in message
        assert "reset" in message.lower()


class TestFreeTierLimits:
    def test_known_models(self):
        assert free_tier_limits("gemini-2.5-flash-lite") == (15, 1000, 0)
        assert free_tier_limits("gemini-2.5-pro") == (5, 100, 0)

    def test_longest_prefix_wins(self):
        """flash-lite must not be shadowed by the shorter flash prefix."""
        assert free_tier_limits("gemini-2.5-flash-lite") != free_tier_limits(
            "gemini-2.5-flash"
        )

    def test_unknown_model_gets_a_conservative_default(self):
        rpm, rpd, _ = free_tier_limits("gemini-9.9-experimental")
        assert rpm > 0 and rpd > 0


class TestGeminiConfiguration:
    def _settings(self, **overrides) -> Settings:
        base = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            # Must be blanked too, or a real key in .env resolves to groq and
            # these Gemini assertions silently test the wrong provider.
            "GROQ_API_KEY": "",
            "VERITAS_LLM_PROVIDER": "auto",
        }
        return Settings(**{**base, **overrides})  # type: ignore[arg-type]

    def test_auto_selects_gemini_when_it_is_the_only_key(self):
        assert self._settings(GEMINI_API_KEY="k").resolved_provider == "gemini"

    def test_openai_still_wins_when_both_are_present(self):
        settings = self._settings(OPENAI_API_KEY="a", GEMINI_API_KEY="b")
        assert settings.resolved_provider == "openai"

    def test_gemini_model_roles(self):
        settings = self._settings(GEMINI_API_KEY="k")
        assert "flash-lite" in settings.model_for("fast")
        assert settings.model_for("strong") != settings.model_for("fast")

    def test_gemini_gets_automatic_rate_limits(self):
        """A free Gemini key must be paced without the user configuring anything."""
        limits = self._settings(GEMINI_API_KEY="k").effective_rate_limits()
        assert limits
        assert all(rpm > 0 and rpd > 0 for rpm, rpd, _ in limits.values())

    def test_limits_are_tracked_per_model(self):
        """Gemini quotas are per model — pooling them wastes the larger budget.

        Flash-Lite carries 4x the daily quota of Pro and handles most of the
        traffic; throttling it to Pro's budget would cut usable throughput by
        roughly 75% for no reason.
        """
        settings = self._settings(
            GEMINI_API_KEY="k",
            VERITAS_MODEL_FAST_GEMINI="gemini-2.5-flash-lite",  # 15 rpm / 1000 rpd
            VERITAS_MODEL_STRONG_GEMINI="gemini-2.5-pro",  # 5 rpm / 100 rpd
        )
        limits = settings.effective_rate_limits()

        assert limits["gemini-2.5-flash-lite"] == (15, 1000, 0)
        assert limits["gemini-2.5-pro"] == (5, 100, 0)

    def test_explicit_limits_override_every_model(self):
        settings = self._settings(GEMINI_API_KEY="k", VERITAS_RPM_LIMIT="60")
        limits = settings.effective_rate_limits()
        assert limits and all(rpm == 60 for rpm, _, _ in limits.values())

    def test_other_providers_are_unpaced_by_default(self):
        assert self._settings(OPENAI_API_KEY="k").effective_rate_limits() == {}

    def test_embedding_model_follows_the_provider(self):
        settings = self._settings(GEMINI_API_KEY="k")
        assert settings.embedding_model_for_provider == "gemini-embedding-2"

    def test_base_url_is_the_compatibility_endpoint(self):
        assert GEMINI_BASE_URL.endswith("/v1beta/openai/")


class TestFreeProfile:
    def test_free_profile_shrinks_the_workload(self):
        """A partial report that finishes beats a thorough one that 429s."""
        settings = Settings(GEMINI_API_KEY="k", VERITAS_PROFILE="free")  # type: ignore[arg-type]
        before = settings.max_claims
        settings.apply_profile()

        assert settings.max_claims <= 8 <= max(before, 8)
        assert settings.consistency_samples == 1
        assert settings.verify_concurrency <= 2
        assert settings.max_research_questions <= 3

    def test_default_profile_changes_nothing(self):
        settings = Settings(VERITAS_PROFILE="default", VERITAS_MAX_CLAIMS="40")  # type: ignore[arg-type]
        settings.apply_profile()
        assert settings.max_claims == 40

    def test_profile_never_raises_limits(self):
        """Applying the free profile must only ever reduce work."""
        settings = Settings(VERITAS_PROFILE="free", VERITAS_MAX_CLAIMS="2")  # type: ignore[arg-type]
        settings.apply_profile()
        assert settings.max_claims == 2


class TestModelRateLimiters:
    async def test_unknown_model_passes_through(self):
        """An unlisted model must not inherit another model's budget."""
        from veritas.llm.ratelimit import ModelRateLimiters

        limiters = ModelRateLimiters({"known": (1, 1, 0)})
        for _ in range(20):
            await limiters.acquire("some-other-model")

    async def test_models_have_independent_quotas(self):
        from veritas.llm.ratelimit import ModelRateLimiters

        limiters = ModelRateLimiters({"a": (100, 1, 0), "b": (100, 5, 0)})
        await limiters.acquire("a")

        # "a" is exhausted, but "b" must be untouched.
        with pytest.raises(RateLimitExceeded):
            await limiters.acquire("a")
        await limiters.acquire("b")

        assert limiters.stats()["b"]["daily_remaining"] == 4

    async def test_disabled_when_no_limits_configured(self):
        from veritas.llm.ratelimit import ModelRateLimiters

        assert not ModelRateLimiters({}).enabled


class TestClientWiring:
    async def test_offline_client_is_unpaced(self, settings):
        from veritas.llm.client import LLMClient, OfflineProvider

        client = LLMClient(settings, provider=OfflineProvider())
        assert client.rate_limit_stats == {}

    async def test_cache_hits_do_not_consume_quota(self, monkeypatch):
        """A warm rerun must cost neither quota nor pacing delay."""
        from veritas.config import get_settings
        from veritas.llm.client import LLMClient, OfflineProvider, user
        from veritas.llm.ratelimit import ModelRateLimiters

        monkeypatch.setenv("VERITAS_CACHE_ENABLED", "true")
        reset_settings_cache()

        settings = get_settings()
        client = LLMClient(settings, provider=OfflineProvider())
        # One request allowed for the whole run: a second network call would raise.
        client._limiters = ModelRateLimiters({settings.model_for("fast"): (100, 1, 0)})

        await client.chat([user("same prompt")], use_cache=True)
        await client.chat([user("same prompt")], use_cache=True)

        assert client.cache_hits == 1
        reset_settings_cache()


class TestRetryAfterParsing:
    """Obeying the provider's stated delay beats guessing.

    A 429 saying "retry in 49s" cannot be satisfied by a 20s-capped backoff:
    every attempt lands inside the same closed window, so the call burns all
    its retries and fails for no reason.
    """

    def test_parses_googles_prose_form(self):
        from veritas.llm.client import parse_retry_after

        assert parse_retry_after(
            "Quota exceeded. Please retry in 49.253009491s"
        ) == pytest.approx(49.25, abs=0.01)

    def test_parses_structured_retry_delay(self):
        from veritas.llm.client import parse_retry_after

        assert parse_retry_after("'retryDelay': '49s'") == 49.0

    def test_returns_none_when_absent(self):
        from veritas.llm.client import parse_retry_after

        assert parse_retry_after("Internal server error") is None

    def test_caps_absurd_delays(self):
        from veritas.llm.client import parse_retry_after

        assert parse_retry_after("retry in 9999s") == 120.0

    def test_transient_error_carries_the_delay(self):
        from veritas.llm.client import _classify

        err = _classify(RuntimeError("429 quota exceeded, please retry in 30s"))
        assert getattr(err, "retry_after", None) == 30.0


class TestObservedQuotas:
    def test_non_lite_flash_is_five_rpm(self):
        """Confirmed from a live 429: 'limit: 5, model: gemini-3.5-flash'.

        An earlier guess of 10 made the limiter send at double the permitted
        rate, so runs collected 429s partway through.
        """
        assert free_tier_limits("gemini-3.5-flash")[0] == 5
        assert free_tier_limits("gemini-3.6-flash")[0] == 5

    def test_lite_models_keep_the_larger_quota(self):
        assert free_tier_limits("gemini-3.1-flash-lite") == (15, 1000, 0)

    def test_unknown_models_get_the_strictest_rate(self):
        assert free_tier_limits("gemini-99-unknown") == (5, 100, 0)


class TestSearxng:
    def test_leads_the_default_chain(self):
        """No key, no quota, no rate limit — it should be tried first.

        Checks the field default rather than a constructed Settings: the test
        environment blanks VERITAS_SEARCH_ORDER to keep the suite offline.
        """
        from veritas.config import Settings

        default_order = Settings.model_fields["search_order"].default
        assert default_order.split(",")[0] == "searxng"

    def test_unavailable_without_a_url(self):
        from veritas.tools.search import SearxngProvider

        assert not SearxngProvider(None, "").available  # type: ignore[arg-type]
        assert SearxngProvider(None, "http://localhost:8080").available  # type: ignore[arg-type]

    def test_trailing_slash_is_normalised(self):
        from veritas.tools.search import SearxngProvider

        p = SearxngProvider(None, "http://localhost:8080/")  # type: ignore[arg-type]
        assert p.base_url == "http://localhost:8080"

    def test_appears_in_health_summary_when_configured(self):
        """An active provider must be visible in /health.

        SearXNG is configured by URL, not key. The credential lookup used by
        env_summary originally only knew about key-based providers, so an
        active SearXNG was silently omitted from the very place you check when
        retrieval misbehaves.
        """
        from veritas.config import Settings, _provider_key

        s = Settings(SEARXNG_URL="http://localhost:8080")  # type: ignore[call-arg]
        assert _provider_key(s, "searxng") == "http://localhost:8080"
        assert _provider_key(s, "unknown-provider") == ""


class TestTokenRateLimiter:
    """Groq caps tokens per minute, not requests.

    A request-only limiter sails past that cap: our calls are large (an
    adjudication carries the whole evidence set), so a handful exhausts a
    minute's token budget while barely denting the request count.
    """

    async def test_disabled_when_tpm_is_zero(self):
        from veritas.llm.ratelimit import TokenRateLimiter

        limiter = TokenRateLimiter(0)
        assert not limiter.enabled
        for _ in range(50):
            await limiter.acquire(10_000)

    async def test_allows_usage_within_budget(self):
        from veritas.llm.ratelimit import TokenRateLimiter

        limiter = TokenRateLimiter(6000)
        started = time.monotonic()
        for _ in range(5):
            await limiter.acquire(1000)
        assert time.monotonic() - started < 0.2
        assert limiter.stats["token_waits"] == 0

    async def test_paces_once_the_budget_is_spent(self, monkeypatch):
        from veritas.llm.ratelimit import TokenRateLimiter

        limiter = TokenRateLimiter(1000)

        async def fake_sleep(seconds: float) -> None:
            limiter._window.clear()

        monkeypatch.setattr(asyncio, "sleep", fake_sleep)

        await limiter.acquire(900)
        await limiter.acquire(900)  # would exceed 1000 in the same minute
        assert limiter.stats["token_waits"] == 1

    async def test_oversized_call_does_not_deadlock(self):
        """A single call larger than the whole budget must not block forever."""
        from veritas.llm.ratelimit import TokenRateLimiter

        limiter = TokenRateLimiter(1000)
        await asyncio.wait_for(limiter.acquire(50_000), timeout=2.0)

    async def test_settle_replaces_the_estimate_with_real_usage(self):
        from veritas.llm.ratelimit import TokenRateLimiter

        limiter = TokenRateLimiter(10_000)
        await limiter.acquire(5000)
        assert limiter.stats["tokens_used_this_minute"] == 5000

        await limiter.settle(5000, 800)
        assert limiter.stats["tokens_used_this_minute"] == 800

    def test_token_estimate_is_roughly_chars_over_four(self):
        from veritas.llm.ratelimit import estimate_tokens

        assert estimate_tokens("x" * 400) == 100
        assert estimate_tokens("") == 1


class TestGroqConfiguration:
    def _settings(self, **overrides) -> Settings:
        base = {
            "OPENAI_API_KEY": "",
            "ANTHROPIC_API_KEY": "",
            "GEMINI_API_KEY": "",
            "GROQ_API_KEY": "",
            "VERITAS_LLM_PROVIDER": "auto",
        }
        return Settings(**{**base, **overrides})  # type: ignore[arg-type]

    def test_auto_selects_groq(self):
        assert self._settings(GROQ_API_KEY="k").resolved_provider == "groq"

    def test_groq_is_preferred_over_gemini(self):
        """~100x the daily requests and several times faster, measured."""
        settings = self._settings(GROQ_API_KEY="g", GEMINI_API_KEY="m")
        assert settings.resolved_provider == "groq"

    def test_openai_still_wins_over_groq(self):
        settings = self._settings(OPENAI_API_KEY="o", GROQ_API_KEY="g")
        assert settings.resolved_provider == "openai"

    def test_model_roles(self):
        settings = self._settings(GROQ_API_KEY="k")
        # The high-volume role gets the largest TOKEN budget, since tokens —
        # not requests — are what Groq actually caps.
        assert settings.model_for("fast") == "llama-3.3-70b-versatile"
        assert settings.model_for("strong") == "openai/gpt-oss-120b"

    def test_fast_role_has_the_larger_token_budget(self):
        settings = self._settings(GROQ_API_KEY="k")
        fast_tpm = free_tier_limits(settings.model_for("fast"))[2]
        strong_tpm = free_tier_limits(settings.model_for("strong"))[2]
        assert fast_tpm >= strong_tpm, (
            "the fast role carries ~80% of tokens and must not be given the "
            "smaller per-minute budget"
        )

    def test_observed_quotas_from_live_headers(self):
        """Values read from this account's x-ratelimit-* response headers."""
        assert free_tier_limits("llama-3.1-8b-instant") == (0, 14_400, 6_000)
        assert free_tier_limits("llama-3.3-70b-versatile") == (0, 1_000, 12_000)
        assert free_tier_limits("openai/gpt-oss-120b") == (0, 1_000, 8_000)

    def test_groq_is_paced_on_tokens_not_requests(self):
        limits = self._settings(GROQ_API_KEY="k").effective_rate_limits()
        for rpm, rpd, tpm in limits.values():
            assert rpm == 0, "Groq imposes no per-minute request cap on this tier"
            assert rpd > 0 and tpm > 0

    def test_base_url_is_the_openai_compatible_endpoint(self):
        from veritas.config import GROQ_BASE_URL

        assert GROQ_BASE_URL == "https://api.groq.com/openai/v1"

    def test_summary_reports_the_token_axis(self):
        summary = self._settings(GROQ_API_KEY="k").rate_limit_summary()
        assert "tok/min" in summary and "/day" in summary
