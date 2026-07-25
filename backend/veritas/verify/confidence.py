"""Confidence scoring.

The problem this solves
-----------------------
Asking a model "how confident are you?" produces a number with a median
expected calibration error around 0.2, and models that essentially never abstain
even when abstention is the optimal policy. A system whose headline output is a
per-claim confidence score cannot be built on that.

So confidence here is a *measurement*, not an opinion: seven orthogonal features
combined by a linear model, squashed to a probability, then mapped through an
isotonic calibrator fitted on labelled data. Verbalised model confidence appears
as exactly one feature, with a deliberately small weight.

Feature design
--------------
Each feature answers a question the others cannot:

===============  ==========================================================
entail_max       Is there any strong grounding at all?
agreement        Which way does the evidence lean, net of contradiction?
independence     How many *genuinely separate* sources agree? (post-clustering)
source_quality   How good are the sources that agree?
consistency      Is the verdict stable across resampled adjudications?
sufficiency      Did retrieval find anything on-topic? (separates false from unknown)
stated_conf      What does the model itself think? (capped)
===============  ==========================================================

``sufficiency`` is what lets the system say "I don't know" instead of "false".
Conflating those two is one of the most damaging errors a fact-checker can make.
"""

from __future__ import annotations

import math

from veritas.evidence.credibility import diversity_bonus
from veritas.evidence.dedup import effective_independent_count
from veritas.logging import get_logger
from veritas.schemas import (
    Claim,
    ConfidenceFeatures,
    Evidence,
    EvidenceCluster,
    Stance,
    Verdict,
)

log = get_logger(__name__)

# Prior weights, ordered as ConfidenceFeatures.feature_names().
#
# These are *priors*, not fitted values: they encode the design reasoning so the
# system is sane before any labelled data exists. `veritas calibrate` replaces
# them with weights fitted by logistic regression on a labelled dev set, and the
# eval harness reports the improvement. Note stated_conf is weighted lowest by
# an order of magnitude relative to the evidence features — that is intentional.
DEFAULT_WEIGHTS: dict[str, float] = {
    "entail_max": 2.6,
    "agreement": 1.2,
    "independence": 1.6,
    "source_quality": 1.2,
    "consistency": 0.8,
    "sufficiency": 1.4,
    "stated_conf": 0.3,
}
DEFAULT_BIAS: float = -4.6

# Confidence ceiling as a function of independent corroborating clusters.
#
# This is a hard structural constraint, applied *after* the linear model, and it
# is the single most important guard in the file. `agreement` saturates at 1.0
# whenever no opposing evidence was retrieved — which is the normal case for an
# obscure claim nobody has contradicted. Without a ceiling, one emphatic source
# produces 0.99 confidence, which is precisely the overconfident behaviour this
# system exists to beat.
#
# No amount of entailment strength from a single source licenses near-certainty:
# the source can simply be wrong, and we would have no way to know.
INDEPENDENCE_CEILING: dict[int, float] = {
    0: 0.30,
    1: 0.70,
    2: 0.85,
    3: 0.93,
}
_CEILING_MAX = 0.97

# Independence saturates: going from 1 to 3 independent sources is a large
# epistemic jump; 8 to 10 is nearly none.
_INDEPENDENCE_SCALE = 4.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def extract_features(
    clusters: list[EvidenceCluster],
    evidence: list[Evidence],
    verdict: Verdict,
    stated_confidence: float = 0.5,
    consistency: float = 1.0,
    retrieval_attempted: bool = True,
) -> ConfidenceFeatures:
    """Compute the seven-feature vector for one claim.

    Features are always expressed *in favour of the stated verdict*: a REFUTED
    claim with strong refuting evidence scores high, exactly as a SUPPORTED
    claim with strong supporting evidence does. The score answers "how sure are
    we of this verdict", not "how likely is the claim true".
    """
    if not clusters:
        return ConfidenceFeatures(
            entail_max=0.0,
            agreement=0.0,
            independence=0.0,
            source_quality=0.0,
            consistency=consistency,
            sufficiency=0.0 if retrieval_attempted else 0.0,
            stated_conf=_clamp(stated_confidence),
        )

    # Which stance corroborates the verdict we are scoring.
    if verdict is Verdict.SUPPORTED:
        aligned, opposed = Stance.SUPPORTS, Stance.REFUTES
    elif verdict is Verdict.REFUTED:
        aligned, opposed = Stance.REFUTES, Stance.SUPPORTS
    else:
        aligned, opposed = Stance.SUPPORTS, Stance.REFUTES

    aligned_clusters = [c for c in clusters if c.stance is aligned]
    opposed_clusters = [c for c in clusters if c.stance is opposed]

    entail_max = max((c.entailment_score for c in aligned_clusters), default=0.0)

    # Credibility-weighted net direction across independent clusters.
    aligned_mass = sum(c.entailment_score * c.credibility_score for c in aligned_clusters)
    opposed_mass = sum(c.entailment_score * c.credibility_score for c in opposed_clusters)
    total_mass = aligned_mass + opposed_mass
    agreement = (aligned_mass - opposed_mass) / total_mass if total_mass > 0 else 0.0

    effective = effective_independent_count(aligned_clusters)
    independence = min(1.0, math.log1p(effective) / math.log1p(_INDEPENDENCE_SCALE))

    if aligned_clusters:
        weighted = sum(c.credibility_score * c.entailment_score for c in aligned_clusters)
        denominator = sum(c.entailment_score for c in aligned_clusters) or 1.0
        base_quality = weighted / denominator
        domains = [d for c in aligned_clusters for d in c.domains]
        source_quality = _clamp(0.75 * base_quality + 0.25 * diversity_bonus(domains))
    else:
        source_quality = 0.0

    sufficiency = _retrieval_sufficiency(clusters, evidence)

    return ConfidenceFeatures(
        entail_max=_clamp(entail_max),
        agreement=max(-1.0, min(1.0, agreement)),
        independence=_clamp(independence),
        source_quality=source_quality,
        consistency=_clamp(consistency),
        sufficiency=sufficiency,
        stated_conf=_clamp(stated_confidence),
    )


def _retrieval_sufficiency(clusters: list[EvidenceCluster], evidence: list[Evidence]) -> float:
    """Did retrieval actually find on-topic material? (CRAG-style gate.)

    Low sufficiency with no aligned evidence means "we don't know". High
    sufficiency with no aligned evidence means "we looked properly and it isn't
    there" — a much stronger statement, and the difference between an honest NEI
    and a defensible REFUTED.
    """
    if not evidence:
        return 0.0

    relevant = [e for e in evidence if e.relevance >= 0.3]
    coverage = len(relevant) / len(evidence)
    decisive = sum(1 for c in clusters if c.stance is not Stance.NEUTRAL)
    decisiveness = min(1.0, decisive / 3.0)
    volume = min(1.0, len(evidence) / 5.0)

    return _clamp(0.4 * coverage + 0.35 * decisiveness + 0.25 * volume)


def consistency_from_samples(verdicts: list[Verdict]) -> float:
    """Agreement across k resampled adjudications.

    A cheap stand-in for semantic entropy: if the same evidence yields a
    different verdict on resampling, the verdict is unstable regardless of how
    confident any single pass sounded. Returns the modal share in [0, 1].
    """
    if not verdicts:
        return 1.0
    counts: dict[Verdict, int] = {}
    for verdict in verdicts:
        counts[verdict] = counts.get(verdict, 0) + 1
    return max(counts.values()) / len(verdicts)


class ConfidenceModel:
    """Linear scorer over the feature vector, plus optional calibration."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        bias: float = DEFAULT_BIAS,
        calibrator: object | None = None,
    ) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.bias = bias
        self.calibrator = calibrator

    def raw_score(self, features: ConfidenceFeatures) -> float:
        """Uncalibrated probability from the linear model."""
        total = self.bias
        for name, value in zip(
            ConfidenceFeatures.feature_names(), features.as_vector(), strict=True
        ):
            total += self.weights.get(name, 0.0) * value
        return sigmoid(total)

    def calibrated_score(self, features: ConfidenceFeatures) -> float:
        raw = self.raw_score(features)
        if self.calibrator is None:
            return raw
        return float(self.calibrator.transform([raw])[0])  # type: ignore[attr-defined]

    def score_claim(self, claim: Claim, n_independent: int | None = None) -> Claim:
        """Attach raw and calibrated confidence to a claim, in place.

        ``n_independent`` is the count of independent evidence clusters
        supporting the verdict; when given it enforces the independence ceiling.
        """
        claim.raw_confidence = self.raw_score(claim.features)
        confidence = self.calibrated_score(claim.features)

        if n_independent is None:
            n_independent = len(claim.cluster_ids)
        ceiling = independence_ceiling(n_independent)
        if confidence > ceiling:
            confidence = ceiling
            claim.rationale = (
                f"{claim.rationale} [confidence capped at {ceiling:.2f}: only "
                f"{n_independent} independent source(s)]"
            ).strip()

        claim.confidence = confidence
        return claim

    def explain(self, features: ConfidenceFeatures) -> list[dict[str, float | str]]:
        """Per-feature contribution breakdown, for the UI's score explainer."""
        contributions: list[dict[str, float | str]] = []
        for name, value in zip(
            ConfidenceFeatures.feature_names(), features.as_vector(), strict=True
        ):
            weight = self.weights.get(name, 0.0)
            contributions.append(
                {
                    "feature": name,
                    "value": round(value, 4),
                    "weight": round(weight, 4),
                    "contribution": round(weight * value, 4),
                }
            )
        contributions.sort(key=lambda c: -abs(float(c["contribution"])))
        return contributions

    def to_dict(self) -> dict:
        payload: dict = {"weights": self.weights, "bias": self.bias}
        if self.calibrator is not None and hasattr(self.calibrator, "to_dict"):
            payload["calibrator"] = self.calibrator.to_dict()  # type: ignore[attr-defined]
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> ConfidenceModel:
        from veritas.verify.calibration import IsotonicCalibrator

        calibrator = None
        if data.get("calibrator"):
            calibrator = IsotonicCalibrator.from_dict(data["calibrator"])
        return cls(
            weights=data.get("weights", DEFAULT_WEIGHTS),
            bias=float(data.get("bias", DEFAULT_BIAS)),
            calibrator=calibrator,
        )


def independence_ceiling(n_independent: int) -> float:
    """Maximum confidence licensed by ``n`` independent corroborating sources."""
    if n_independent in INDEPENDENCE_CEILING:
        return INDEPENDENCE_CEILING[n_independent]
    return _CEILING_MAX


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def apply_verdict_floor(claim: Claim) -> Claim:
    """Enforce the invariant that an NEI verdict cannot carry high confidence.

    NEI means "we could not establish this". A high-confidence NEI is a
    contradiction in terms and would read as a strong signal in the UI, so the
    score is capped and the reason recorded.
    """
    if claim.verdict is Verdict.NEI and claim.confidence > 0.5:
        claim.confidence = 0.5
        claim.rationale = (claim.rationale + " [confidence capped: NEI verdict]").strip()
    return claim
