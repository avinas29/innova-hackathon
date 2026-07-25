"""Domain model.

These types are the contract between the graph, the storage layer, the HTTP API
and the frontend. They are intentionally free of any framework imports.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class Verdict(StrEnum):
    """Three-way verdict, mirroring FEVER's label set.

    ``NEI`` (not enough info) is a first-class outcome, not a failure. The
    literature is consistent that models almost never abstain on their own, so
    abstention has to be an explicit, reachable state in the design.
    """

    SUPPORTED = "SUPPORTED"
    REFUTED = "REFUTED"
    NEI = "NEI"


class Stance(StrEnum):
    """How one piece of evidence relates to one claim."""

    SUPPORTS = "SUPPORTS"
    REFUTES = "REFUTES"
    NEUTRAL = "NEUTRAL"


class ClaimCategory(StrEnum):
    """ClaimBuster-style check-worthiness classes.

    Only ``CHECK_WORTHY`` claims enter the verification fan-out. Trying to
    "verify" an opinion is the classic way these systems produce nonsense.
    """

    NON_FACTUAL = "NON_FACTUAL"
    FACTUAL_UNIMPORTANT = "FACTUAL_UNIMPORTANT"
    CHECK_WORTHY = "CHECK_WORTHY"


class CredibilityTier(StrEnum):
    """Coarse source-quality prior, applied before the LLM ranks anything."""

    PRIMARY = "PRIMARY"          # peer-reviewed, official statistics, standards bodies
    HIGH = "HIGH"                # major reference works, established outlets of record
    MEDIUM = "MEDIUM"            # mainstream press, reputable trade publications
    LOW = "LOW"                  # blogs, forums, marketing pages
    UNKNOWN = "UNKNOWN"          # unrated domain
    UNRELIABLE = "UNRELIABLE"    # known content farms / aggregators


CREDIBILITY_WEIGHT: dict[CredibilityTier, float] = {
    CredibilityTier.PRIMARY: 1.00,
    CredibilityTier.HIGH: 0.85,
    CredibilityTier.MEDIUM: 0.65,
    CredibilityTier.LOW: 0.35,
    CredibilityTier.UNKNOWN: 0.45,
    CredibilityTier.UNRELIABLE: 0.10,
}


class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"


class SourceKind(StrEnum):
    WEB = "WEB"
    ACADEMIC = "ACADEMIC"
    CODE = "CODE"
    REFERENCE = "REFERENCE"


# ─────────────────────────────────────────────────────────────────────────────
# Research planning
# ─────────────────────────────────────────────────────────────────────────────


class ResearchQuestion(BaseModel):
    """One decomposed sub-question, assigned to a specialised researcher."""

    id: str = Field(default_factory=lambda: new_id("q"))
    question: str
    rationale: str = ""
    kind: SourceKind = SourceKind.WEB
    priority: int = Field(default=1, ge=1, le=5)


class ResearchPlan(BaseModel):
    topic: str
    questions: list[ResearchQuestion] = Field(default_factory=list)
    scope_notes: str = ""
    created_at: datetime = Field(default_factory=_now)


# ─────────────────────────────────────────────────────────────────────────────
# Sources and evidence
# ─────────────────────────────────────────────────────────────────────────────


class Source(BaseModel):
    """A retrievable document. One URL == one Source within a run."""

    id: str = Field(default_factory=lambda: new_id("src"))
    url: str
    domain: str = ""
    title: str = ""
    snippet: str = ""
    content: str = ""
    kind: SourceKind = SourceKind.WEB
    credibility_tier: CredibilityTier = CredibilityTier.UNKNOWN
    credibility_score: float = Field(default=0.45, ge=0.0, le=1.0)
    published_at: str | None = None
    retrieved_at: datetime = Field(default_factory=_now)
    fetch_ok: bool = True
    degraded: bool = False  # search snippet only — full text unavailable

    @property
    def best_text(self) -> str:
        """Full extracted text when we have it, else the search snippet."""
        return self.content if len(self.content) > len(self.snippet) else self.snippet


class Evidence(BaseModel):
    """One snippet from one source, evaluated against one claim."""

    id: str = Field(default_factory=lambda: new_id("ev"))
    claim_id: str
    source_id: str
    url: str = ""
    domain: str = ""
    snippet: str
    stance: Stance = Stance.NEUTRAL
    entailment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    cluster_id: str | None = None
    is_derivative: bool = False  # a syndicated copy of another cluster member
    credibility_score: float = Field(default=0.45, ge=0.0, le=1.0)

    @property
    def signed_strength(self) -> float:
        """Entailment magnitude signed by stance; 0 for neutral evidence."""
        if self.stance is Stance.SUPPORTS:
            return self.entailment_score
        if self.stance is Stance.REFUTES:
            return -self.entailment_score
        return 0.0


class EvidenceCluster(BaseModel):
    """A set of near-duplicate evidence items treated as ONE independent source.

    Five outlets reprinting one wire story is one piece of evidence, not five.
    Collapsing them is what keeps confidence from being systematically inflated.
    """

    id: str = Field(default_factory=lambda: new_id("cl"))
    claim_id: str
    member_ids: list[str] = Field(default_factory=list)
    representative_id: str = ""
    stance: Stance = Stance.NEUTRAL
    entailment_score: float = Field(default=0.0, ge=0.0, le=1.0)
    credibility_score: float = Field(default=0.45, ge=0.0, le=1.0)
    domains: list[str] = Field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.member_ids)


# ─────────────────────────────────────────────────────────────────────────────
# Claims, confidence, contradictions
# ─────────────────────────────────────────────────────────────────────────────


class ConfidenceFeatures(BaseModel):
    """The seven orthogonal signals behind a confidence score.

    Deliberately *not* a single LLM opinion. Verbalised confidence is included
    as one capped feature because it carries some signal, but it is measurably
    miscalibrated (median ECE ~0.2 in the literature) and must not dominate.
    """

    entail_max: float = Field(default=0.0, ge=0.0, le=1.0)
    agreement: float = Field(default=0.0, ge=-1.0, le=1.0)
    independence: float = Field(default=0.0, ge=0.0)
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    consistency: float = Field(default=0.0, ge=0.0, le=1.0)
    sufficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    stated_conf: float = Field(default=0.5, ge=0.0, le=1.0)

    def as_vector(self) -> list[float]:
        """Ordered feature vector. Order is part of the calibrator's contract."""
        return [
            self.entail_max,
            self.agreement,
            self.independence,
            self.source_quality,
            self.consistency,
            self.sufficiency,
            self.stated_conf,
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "entail_max",
            "agreement",
            "independence",
            "source_quality",
            "consistency",
            "sufficiency",
            "stated_conf",
        ]


class Claim(BaseModel):
    """An atomic, decontextualised, independently checkable assertion."""

    id: str = Field(default_factory=lambda: new_id("clm"))
    text: str                       # as written in the draft
    decontextualised: str = ""      # self-contained rewrite; what we actually verify
    source_sentence: str = ""       # provenance back into the draft
    category: ClaimCategory = ClaimCategory.CHECK_WORTHY
    checkworthy_score: float = Field(default=1.0, ge=0.0, le=1.0)

    verdict: Verdict = Verdict.NEI
    raw_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)  # calibrated
    features: ConfidenceFeatures = Field(default_factory=ConfidenceFeatures)

    evidence_ids: list[str] = Field(default_factory=list)
    cluster_ids: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)

    rationale: str = ""
    advocate_argument: str = ""
    sceptic_argument: str = ""
    minority_report: str = ""   # preserved dissent, never averaged away
    retracted: bool = False
    revision: str = ""
    error: str = ""

    @property
    def verify_text(self) -> str:
        return self.decontextualised or self.text

    @field_validator("decontextualised")
    @classmethod
    def _default_decontext(cls, v: str) -> str:
        return v.strip()


class Contradiction(BaseModel):
    """Two evidence items that disagree with each other.

    Distinct from claim-vs-evidence conflict: this is the sources themselves
    being mutually inconsistent, which is a far stronger signal to surface.
    """

    id: str = Field(default_factory=lambda: new_id("con"))
    evidence_a: str
    evidence_b: str
    claim_id: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation: str = ""
    domain_a: str = ""
    domain_b: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Run-level results
# ─────────────────────────────────────────────────────────────────────────────


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def merge(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            calls=self.calls + other.calls,
        )


class RunMetrics(BaseModel):
    """Headline numbers, computed once at the end of a run."""

    total_claims: int = 0
    checkworthy_claims: int = 0
    supported: int = 0
    refuted: int = 0
    nei: int = 0
    retracted: int = 0
    mean_confidence: float = 0.0
    unique_sources: int = 0
    unique_domains: int = 0
    evidence_items: int = 0
    independent_clusters: int = 0
    contradictions: int = 0
    duration_seconds: float = 0.0
    tokens: TokenUsage = Field(default_factory=TokenUsage)

    @property
    def support_rate(self) -> float:
        """Share of verified claims the evidence actually backs."""
        denom = self.supported + self.refuted + self.nei
        return self.supported / denom if denom else 0.0


class RunEvent(BaseModel):
    """One streamed progress event. Persisted so a run can be replayed."""

    ts: datetime = Field(default_factory=_now)
    run_id: str = ""
    node: str = ""
    level: Literal["debug", "info", "warning", "error"] = "info"
    message: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class ResearchReport(BaseModel):
    """The final artefact returned to the caller."""

    run_id: str
    topic: str
    status: RunStatus = RunStatus.COMPLETED
    executive_summary: str = ""
    body_markdown: str = ""
    plan: ResearchPlan | None = None
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    clusters: list[EvidenceCluster] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    metrics: RunMetrics = Field(default_factory=RunMetrics)
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class RunRequest(BaseModel):
    """Inbound API payload for a new run."""

    topic: str = Field(min_length=3, max_length=1000)
    max_questions: int | None = Field(default=None, ge=1, le=12)
    max_claims: int | None = Field(default=None, ge=1, le=100)
    include_academic: bool = True
    include_code: bool = False

    @field_validator("topic")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("topic must not be blank")
        return v


class VerifyRequest(BaseModel):
    """Fast path: verify one claim without a full research run."""

    claim: str = Field(min_length=3, max_length=2000)
    context: str = ""


class VerifyResponse(BaseModel):
    claim: Claim
    evidence: list[Evidence] = Field(default_factory=list)
    clusters: list[EvidenceCluster] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
