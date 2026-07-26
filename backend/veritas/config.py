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

Provider = Literal["openai", "anthropic", "gemini", "groq", "auto", "fake"]
EntailmentBackend = Literal["llm", "local"]
Profile = Literal["default", "free"]

# Gemini speaks the OpenAI wire format at this path, so it reuses the OpenAI
# SDK rather than needing a separate client.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# Groq is OpenAI wire-compatible too, so it reuses the same client.
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

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
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    llm_provider: Provider = Field(default="auto", alias="VERITAS_LLM_PROVIDER")
    # Per-role provider override — the two roles can run on DIFFERENT providers.
    #
    # Quota, not capability, is the binding constraint on free tiers. Splitting
    # by role draws on two independent allowances at once, roughly doubling how
    # many runs a day are possible, and lets each provider do what it is best
    # at: Groq for the high-volume fast role (sub-second, large token budget),
    # a stronger model for adjudication where quality moves the verdict.
    #
    # Split by ROLE rather than round-robin: role assignment is deterministic,
    # so identical prompts always hit the same provider and stay cacheable, and
    # a run stays reproducible. Random splitting would defeat both.
    provider_fast: Provider | None = Field(default=None, alias="VERITAS_PROVIDER_FAST")
    provider_strong: Provider | None = Field(default=None, alias="VERITAS_PROVIDER_STRONG")

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
    # Chosen by measuring a live free key. Groq's binding constraint is TOKENS
    # per minute, not requests, so the high-volume role must get the model with
    # the LARGEST token budget — not the one with the most requests.
    #
    #   model                     latency  req/day   tok/min
    #   llama-3.3-70b-versatile     0.37s    1,000    12,000   <- fast role
    #   openai/gpt-oss-120b         0.83s    1,000     8,000   <- strong role
    #   llama-3.1-8b-instant        0.32s   14,400     6,000
    #
    # 8b-instant's 14,400 requests/day looks generous but is irrelevant: at
    # ~60 calls per run the request cap is never reached, while its 6,000 TPM
    # throttles the run to roughly twice the wall-clock of 70b-versatile.
    # 70b is also the better model, at effectively the same latency.
    model_fast_groq: str = Field(
        default="llama-3.3-70b-versatile", alias="VERITAS_MODEL_FAST_GROQ"
    )
    model_strong_groq: str = Field(
        default="openai/gpt-oss-120b", alias="VERITAS_MODEL_STRONG_GROQ"
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
    # "basic" costs 1 credit per search, "advanced" costs 2. On the free
    # tier (1,000 credits/month) that doubles the number of usable runs.
    tavily_search_depth: str = Field(default="basic", alias="VERITAS_TAVILY_DEPTH")
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

    @field_validator(
        "openai_api_key",
        "anthropic_api_key",
        "gemini_api_key",
        "groq_api_key",
        "tavily_api_key",
        "exa_api_key",
        "brave_api_key",
        "searxng_url",
        mode="before",
    )
    @classmethod
    def _strip_credential(cls, v: object) -> object:
        """Trim whitespace from credentials.

        Pasting into a hosting dashboard routinely appends a trailing space or
        newline. Providers reject the result as an invalid key, and the error
        says nothing about whitespace — it took a live 401 to find. Stripping
        here removes the entire failure class.
        """
        return v.strip() if isinstance(v, str) else v

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
        # Groq before Gemini: ~100x the daily requests and several times faster.
        if self.groq_api_key:
            return "groq"
        if self.gemini_api_key:
            return "gemini"
        return "fake"

    def provider_for(self, role: Literal["fast", "strong"]) -> Provider:
        """Provider serving one role, honouring a per-role override."""
        explicit = self.provider_fast if role == "fast" else self.provider_strong
        if explicit and explicit != "auto":
            return explicit
        return self.resolved_provider

    @property
    def split_providers(self) -> bool:
        """True when the two roles run on different providers."""
        return self.provider_for("fast") != self.provider_for("strong")

    def model_for(self, role: Literal["fast", "strong"]) -> str:
        """Model id for a role, under whichever provider serves that role."""
        provider = self.provider_for(role)
        if provider == "anthropic":
            return self.model_strong_anthropic if role == "strong" else self.model_fast_anthropic
        if provider == "gemini":
            return self.model_strong_gemini if role == "strong" else self.model_fast_gemini
        if provider == "groq":
            return self.model_strong_groq if role == "strong" else self.model_fast_groq
        return self.model_strong if role == "strong" else self.model_fast

    @property
    def embedding_model_for_provider(self) -> str:
        if self.resolved_provider == "gemini":
            return self.embedding_model_gemini
        return self.embedding_model

    def effective_rate_limits(self) -> dict[str, tuple[int, int, int]]:
        """Per-model ``{model: (rpm, requests_per_day, tpm)}`` enforced client-side.

        Quotas are per model, so limits are tracked per model rather than
        pooled — pooling against the strictest model would throttle the
        high-volume fast model to the strong model's much smaller budget.

        Which axis actually binds differs by provider: Gemini caps requests per
        minute (5-15), while Groq is generous on requests but caps tokens per
        minute. Both are returned and both are enforced.

        Explicit ``VERITAS_RPM_LIMIT`` / ``VERITAS_DAILY_LIMIT`` settings apply
        to every model and always win.
        """
        models = {self.model_for("fast"), self.model_for("strong")}

        if self.rpm_limit or self.daily_limit:
            return dict.fromkeys(models, (self.rpm_limit, self.daily_limit, 0))

        from veritas.llm.ratelimit import free_tier_limits

        limits: dict[str, tuple[int, int, int]] = {}
        for role in ("fast", "strong"):
            if self.provider_for(role) in {"gemini", "groq"}:
                limits[self.model_for(role)] = free_tier_limits(self.model_for(role))
        return limits

    def rate_limit_summary(self) -> str:
        """One-line human summary of the active limits."""
        limits = self.effective_rate_limits()
        if not limits:
            return "unlimited"
        parts = []
        for model, (rpm, rpd, tpm) in sorted(limits.items()):
            axes = []
            if rpm:
                axes.append(f"{rpm}/min")
            if rpd:
                axes.append(f"{rpd}/day")
            if tpm:
                axes.append(f"{tpm} tok/min")
            parts.append(f"{model} {' '.join(axes) or 'unlimited'}")
        return ", ".join(parts)

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
        # Call count scales as roughly 4 + claims x 4 once entailment is
        # batched. A free daily allowance can be as low as 20 requests on one
        # model, so these caps are what keep a run inside it.
        self.max_research_questions = min(self.max_research_questions, 3)
        self.max_claims = min(self.max_claims, 5)
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
        "groq_key_present": bool(s.groq_api_key),
        "profile": s.profile,
        # Not a secret, and the only way to diagnose a DNS failure when every
        # dashboard value is masked.
        "searxng_url": s.searxng_url or "(unset)",
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
