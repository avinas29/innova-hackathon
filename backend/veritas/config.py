"""Central configuration.

Every tunable in the system resolves here so that a run is fully described by
(topic, Settings). Nothing reads ``os.environ`` directly outside this module.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["openai", "anthropic", "gemini", "auto", "fake"]
EntailmentBackend = Literal["llm", "local"]
Profile = Literal["default", "free"]

# Gemini speaks the OpenAI wire format at this path, so it reuses the OpenAI
# SDK rather than needing a separate client.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment or a ``.env`` file."""

    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Model providers ──────────────────────────────────────────────────────
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    llm_provider: Provider = Field(default="auto", alias="VERITAS_LLM_PROVIDER")

    model_fast: str = Field(default="gpt-4.1-mini", alias="VERITAS_MODEL_FAST")
    model_strong: str = Field(default="gpt-4.1", alias="VERITAS_MODEL_STRONG")
    model_fast_anthropic: str = Field(
        default="claude-haiku-4-5-20251001", alias="VERITAS_MODEL_FAST_ANTHROPIC"
    )
    model_strong_anthropic: str = Field(
        default="claude-sonnet-5", alias="VERITAS_MODEL_STRONG_ANTHROPIC"
    )
    # Chosen from measured latency on a live free-tier key, not from the docs:
    #   gemini-3.5-flash-lite  11.7s then 66.0s  — unusable at our call volume
    #   gemini-3.1-flash-lite   0.8s /  0.9s     — fast and consistent
    #   gemini-3.5-flash        1.6s /  2.2s     — good, clean JSON
    # The 2.5 family returns 404 for keys issued after its retirement.
    # Flash-Lite carries the larger free quota and absorbs the high-volume
    # extraction and entailment calls; Flash handles adjudication and synthesis,
    # where quality actually moves the verdict.
    model_fast_gemini: str = Field(
        default="gemini-3.1-flash-lite", alias="VERITAS_MODEL_FAST_GEMINI"
    )
    model_strong_gemini: str = Field(
        default="gemini-3.5-flash", alias="VERITAS_MODEL_STRONG_GEMINI"
    )
    embedding_model: str = Field(default="text-embedding-3-small", alias="VERITAS_EMBEDDING_MODEL")
    embedding_model_gemini: str = Field(
        default="gemini-embedding-2", alias="VERITAS_EMBEDDING_MODEL_GEMINI"
    )

    # ── Rate limiting ────────────────────────────────────────────────────────
    # 0 means "no client-side limit". Auto-populated from the model's published
    # free-tier quota when the provider is Gemini and these are left at 0.
    rpm_limit: int = Field(default=0, alias="VERITAS_RPM_LIMIT")
    daily_limit: int = Field(default=0, alias="VERITAS_DAILY_LIMIT")
    profile: Profile = Field(default="default", alias="VERITAS_PROFILE")

    # ── Search providers ─────────────────────────────────────────────────────
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    exa_api_key: str = Field(default="", alias="EXA_API_KEY")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    # Self-hosted SearXNG base URL, e.g. http://localhost:8080. No key, no
    # quota, and it aggregates multiple engines — the best default when
    # available, so it leads the chain.
    searxng_url: str = Field(default="", alias="SEARXNG_URL")
    search_order: str = Field(
        default="searxng,tavily,exa,brave,duckduckgo", alias="VERITAS_SEARCH_ORDER"
    )
    # Hard network kill-switch. Every outbound retrieval call short-circuits to
    # empty. The test suite sets this so hermeticity is an enforced property
    # rather than an accident of leaving search unconfigured — a keyless
    # fallback added later would otherwise silently reintroduce live traffic.
    offline: bool = Field(default=False, alias="VERITAS_OFFLINE")

    # ── Budgets ──────────────────────────────────────────────────────────────
    max_tokens_per_run: int = Field(default=400_000, alias="VERITAS_MAX_TOKENS_PER_RUN")
    max_wall_seconds: int = Field(default=900, alias="VERITAS_MAX_WALL_SECONDS")
    max_research_questions: int = Field(default=6, alias="VERITAS_MAX_RESEARCH_QUESTIONS")
    max_claims: int = Field(default=40, alias="VERITAS_MAX_CLAIMS")
    max_evidence_per_claim: int = Field(default=8, alias="VERITAS_MAX_EVIDENCE_PER_CLAIM")
    max_reflection_loops: int = Field(default=2, alias="VERITAS_MAX_REFLECTION_LOOPS")
    verify_concurrency: int = Field(default=8, alias="VERITAS_VERIFY_CONCURRENCY")

    # ── Verification ─────────────────────────────────────────────────────────
    entailment_backend: EntailmentBackend = Field(default="llm", alias="VERITAS_ENTAILMENT_BACKEND")
    local_nli_model: str = Field(
        default="lytang/MiniCheck-Flan-T5-Large", alias="VERITAS_LOCAL_NLI_MODEL"
    )
    dedup_threshold: float = Field(default=0.86, alias="VERITAS_DEDUP_THRESHOLD")
    consistency_samples: int = Field(default=3, alias="VERITAS_CONSISTENCY_SAMPLES")

    # ── Storage ──────────────────────────────────────────────────────────────
    db_path: str = Field(default="./veritas.db", alias="VERITAS_DB_PATH")
    cache_enabled: bool = Field(default=True, alias="VERITAS_CACHE_ENABLED")
    cache_ttl_seconds: int = Field(default=604_800, alias="VERITAS_CACHE_TTL_SECONDS")

    # ── Observability ────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", alias="VERITAS_LOG_LEVEL")
    log_json: bool = Field(default=False, alias="VERITAS_LOG_JSON")
    langfuse_public_key: str = Field(default="", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field(default="", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_HOST")

    # ── API ──────────────────────────────────────────────────────────────────
    cors_origins: str = Field(default="http://localhost:3000", alias="VERITAS_CORS_ORIGINS")

    @field_validator("dedup_threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError("dedup_threshold must lie strictly between 0 and 1")
        return v

    @field_validator("consistency_samples")
    @classmethod
    def _check_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("consistency_samples must be >= 1")
        return v

    # ── Derived helpers ──────────────────────────────────────────────────────
    @property
    def search_providers(self) -> list[str]:
        """Search backends in priority order, lowercased and de-blanked."""
        return [p.strip().lower() for p in self.search_order.split(",") if p.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def resolved_provider(self) -> Provider:
        """Concrete provider after resolving ``auto``.

        ``auto`` prefers OpenAI, then Anthropic, then Gemini, and finally the
        deterministic offline provider so the graph is always runnable.
        """
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.anthropic_api_key:
            return "anthropic"
        if self.gemini_api_key:
            return "gemini"
        return "fake"

    def model_for(self, role: Literal["fast", "strong"]) -> str:
        """Model id for a role under the resolved provider."""
        provider = self.resolved_provider
        if provider == "anthropic":
            return self.model_strong_anthropic if role == "strong" else self.model_fast_anthropic
        if provider == "gemini":
            return self.model_strong_gemini if role == "strong" else self.model_fast_gemini
        return self.model_strong if role == "strong" else self.model_fast

    @property
    def embedding_model_for_provider(self) -> str:
        if self.resolved_provider == "gemini":
            return self.embedding_model_gemini
        return self.embedding_model

    def effective_rate_limits(self) -> dict[str, tuple[int, int]]:
        """Per-model ``{model: (rpm, requests_per_day)}`` to enforce client-side.

        Gemini's quotas are per model, so limits are tracked per model rather
        than pooled — pooling against the strictest model would throttle the
        high-volume fast model to the strong model's much smaller budget.

        Explicit ``VERITAS_RPM_LIMIT`` / ``VERITAS_DAILY_LIMIT`` settings apply
        to every model and always win.
        """
        models = {self.model_for("fast"), self.model_for("strong")}

        if self.rpm_limit or self.daily_limit:
            return dict.fromkeys(models, (self.rpm_limit, self.daily_limit))

        if self.resolved_provider == "gemini":
            from veritas.llm.ratelimit import free_tier_limits

            return {m: free_tier_limits(m) for m in models}

        return {}

    def rate_limit_summary(self) -> str:
        """One-line human summary of the active limits."""
        limits = self.effective_rate_limits()
        if not limits:
            return "unlimited"
        return ", ".join(
            f"{model} {rpm}/min {rpd}/day" for model, (rpm, rpd) in sorted(limits.items())
        )

    def apply_profile(self) -> None:
        """Shrink the workload to fit a constrained quota.

        The ``free`` profile targets a run that completes inside a free-tier
        daily allowance. It trades breadth for feasibility: fewer research
        questions, fewer verified claims, less evidence per claim, and no
        self-consistency resampling (which multiplies adjudication calls).
        A partial report that finishes beats a thorough one that 429s.
        """
        if self.profile != "free":
            return
        self.max_research_questions = min(self.max_research_questions, 3)
        self.max_claims = min(self.max_claims, 8)
        self.max_evidence_per_claim = min(self.max_evidence_per_claim, 4)
        self.consistency_samples = 1
        self.verify_concurrency = min(self.verify_concurrency, 2)
        self.max_reflection_loops = min(self.max_reflection_loops, 1)

    @property
    def db_file(self) -> Path:
        path = Path(self.db_path)
        if not path.is_absolute():
            path = _REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    settings = Settings()  # type: ignore[call-arg]
    settings.apply_profile()
    return settings


def reset_settings_cache() -> None:
    """Drop the cached settings — used by tests that mutate the environment."""
    get_settings.cache_clear()


def env_summary() -> dict[str, object]:
    """Non-secret snapshot of the active configuration, safe to log or expose."""
    s = get_settings()
    return {
        "llm_provider": s.resolved_provider,
        "model_fast": s.model_for("fast"),
        "model_strong": s.model_for("strong"),
        "entailment_backend": s.entailment_backend,
        "search_providers": [
            p for p in s.search_providers if p == "duckduckgo" or _provider_key(s, p)
        ],
        "openai_key_present": bool(s.openai_api_key),
        "anthropic_key_present": bool(s.anthropic_api_key),
        "gemini_key_present": bool(s.gemini_api_key),
        "profile": s.profile,
        "rate_limits": s.rate_limit_summary(),
        "cache_enabled": s.cache_enabled,
        "db_path": str(s.db_file),
        "python_env": os.environ.get("VERITAS_ENV", "local"),
    }


def _provider_key(s: Settings, provider: str) -> str:
    """Credential (or base URL) that makes a search provider usable.

    SearXNG is configured by URL rather than key; omitting it here made an
    active provider invisible in /health and the UI header, which is exactly
    where you look to find out why retrieval is behaving oddly.
    """
    return {
        "searxng": s.searxng_url,
        "tavily": s.tavily_api_key,
        "exa": s.exa_api_key,
        "brave": s.brave_api_key,
    }.get(provider, "")
