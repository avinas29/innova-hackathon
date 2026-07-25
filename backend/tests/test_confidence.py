"""Confidence, calibration and evaluation metrics.

These are the tests that matter most: they encode the invariants that make the
published confidence numbers trustworthy.
"""

from __future__ import annotations

import pytest

from veritas.eval.metrics import (
    auroc,
    brier_score,
    evaluate,
    expected_calibration_error,
    macro_f1,
    selective_risk_curve,
)
from veritas.schemas import (
    Claim,
    ConfidenceFeatures,
    Evidence,
    EvidenceCluster,
    Stance,
    Verdict,
)
from veritas.verify.calibration import (
    CalibrationBundle,
    IsotonicCalibrator,
    fit_logistic,
    temperature_scale,
    train_calibration,
)
from veritas.verify.confidence import (
    ConfidenceModel,
    apply_verdict_floor,
    consistency_from_samples,
    extract_features,
    independence_ceiling,
    sigmoid,
)


def _cluster(stance=Stance.SUPPORTS, entail=0.9, cred=0.85, size=1, domain="nature.com"):
    return EvidenceCluster(
        claim_id="c",
        member_ids=[f"e{i}" for i in range(size)],
        representative_id="e0",
        stance=stance,
        entailment_score=entail,
        credibility_score=cred,
        domains=[domain],
    )


def _evidence(relevance=0.8, stance=Stance.SUPPORTS):
    return Evidence(
        claim_id="c", source_id="s", snippet="text", stance=stance, relevance=relevance
    )


class TestFeatureExtraction:
    def test_no_evidence_yields_zero_features(self):
        features = extract_features([], [], Verdict.NEI)
        assert features.entail_max == 0.0
        assert features.independence == 0.0
        assert features.sufficiency == 0.0

    def test_features_align_with_the_verdict(self):
        """A REFUTED verdict backed by refuting evidence must score high.

        The score answers "how sure are we of this verdict", not "how likely is
        the claim true" — so refutation evidence corroborates a REFUTED verdict.
        """
        clusters = [_cluster(stance=Stance.REFUTES, entail=0.95)]
        evidence = [_evidence(stance=Stance.REFUTES)]

        refuted = extract_features(clusters, evidence, Verdict.REFUTED)
        supported = extract_features(clusters, evidence, Verdict.SUPPORTED)

        assert refuted.agreement > 0
        assert supported.agreement < 0

    def test_independence_saturates(self):
        few = extract_features([_cluster() for _ in range(2)], [_evidence()], Verdict.SUPPORTED)
        many = extract_features([_cluster() for _ in range(12)], [_evidence()], Verdict.SUPPORTED)
        assert many.independence > few.independence
        assert many.independence <= 1.0

    def test_conflicting_evidence_lowers_agreement(self):
        clusters = [
            _cluster(stance=Stance.SUPPORTS, entail=0.9),
            _cluster(stance=Stance.REFUTES, entail=0.9),
        ]
        features = extract_features(clusters, [_evidence()], Verdict.SUPPORTED)
        assert features.agreement == pytest.approx(0.0, abs=0.05)

    def test_sufficiency_separates_unknown_from_false(self):
        """Sufficiency must distinguish 'we found nothing' from 'we looked hard'."""
        thin = extract_features([_cluster()], [_evidence(relevance=0.0)], Verdict.SUPPORTED)
        thorough = extract_features(
            [_cluster(), _cluster(), _cluster()],
            [_evidence(relevance=0.9) for _ in range(6)],
            Verdict.SUPPORTED,
        )
        assert thorough.sufficiency > thin.sufficiency


class TestIndependenceCeiling:
    def test_single_source_cannot_produce_near_certainty(self):
        """The core guard. One source, however emphatic, is not near-certainty."""
        assert independence_ceiling(1) <= 0.75
        assert independence_ceiling(0) < independence_ceiling(1)
        assert independence_ceiling(3) < independence_ceiling(10)
        assert independence_ceiling(50) <= 1.0

    def test_ceiling_is_applied_to_the_claim(self):
        model = ConfidenceModel()
        claim = Claim(text="x", decontextualised="x")
        claim.features = ConfidenceFeatures(
            entail_max=1.0,
            agreement=1.0,
            independence=1.0,
            source_quality=1.0,
            consistency=1.0,
            sufficiency=1.0,
            stated_conf=1.0,
        )
        model.score_claim(claim, n_independent=1)

        assert claim.raw_confidence > 0.9, "raw score should be high"
        assert claim.confidence <= independence_ceiling(1)
        assert "capped" in claim.rationale

    def test_more_independent_sources_raise_the_cap(self):
        model = ConfidenceModel()
        features = ConfidenceFeatures(
            entail_max=0.95,
            agreement=1.0,
            independence=0.9,
            source_quality=0.9,
            consistency=1.0,
            sufficiency=0.9,
            stated_conf=0.9,
        )
        scores = []
        for n in (1, 2, 4):
            claim = Claim(text="x", decontextualised="x")
            claim.features = features
            model.score_claim(claim, n_independent=n)
            scores.append(claim.confidence)
        assert scores[0] < scores[1] < scores[2]


class TestConfidenceModel:
    def test_sigmoid_is_stable_at_extremes(self):
        assert 0.0 <= sigmoid(-800) <= 1e-6
        assert sigmoid(800) == pytest.approx(1.0)
        assert sigmoid(0) == pytest.approx(0.5)

    def test_stated_confidence_cannot_dominate(self):
        """Verbalised model confidence must not carry the score on its own.

        This is the whole point of the feature ensemble: an LLM claiming 1.0
        confidence with no evidence behind it must stay near the floor.
        """
        model = ConfidenceModel()
        only_stated = ConfidenceFeatures(stated_conf=1.0)
        assert model.raw_score(only_stated) < 0.25

    def test_explain_ranks_by_absolute_contribution(self):
        model = ConfidenceModel()
        contributions = model.explain(
            ConfidenceFeatures(entail_max=1.0, agreement=1.0, stated_conf=1.0)
        )
        magnitudes = [abs(float(c["contribution"])) for c in contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_round_trips_through_dict(self):
        model = ConfidenceModel()
        restored = ConfidenceModel.from_dict(model.to_dict())
        features = ConfidenceFeatures(entail_max=0.7, agreement=0.5)
        assert restored.raw_score(features) == pytest.approx(model.raw_score(features))

    def test_nei_confidence_is_floored(self):
        claim = Claim(text="x", decontextualised="x", verdict=Verdict.NEI, confidence=0.95)
        apply_verdict_floor(claim)
        assert claim.confidence <= 0.5


class TestConsistency:
    def test_unanimous(self):
        assert consistency_from_samples([Verdict.SUPPORTED] * 3) == 1.0

    def test_split(self):
        value = consistency_from_samples(
            [Verdict.SUPPORTED, Verdict.REFUTED, Verdict.SUPPORTED]
        )
        assert value == pytest.approx(2 / 3)

    def test_empty_defaults_to_certain(self):
        assert consistency_from_samples([]) == 1.0


class TestIsotonic:
    def test_monotone_fit(self):
        scores = [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 0.9, 0.95]
        labels = [0, 0, 0, 1, 0, 1, 1, 1]
        calibrator = IsotonicCalibrator().fit(scores, labels)
        out = calibrator.transform(sorted(scores))
        assert all(out[i] <= out[i + 1] + 1e-9 for i in range(len(out) - 1))

    def test_improves_calibration_of_overconfident_scores(self):
        """Systematically overconfident scores should get pulled down."""
        scores = [0.95] * 10 + [0.9] * 10
        labels = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        before = expected_calibration_error(scores, labels)
        calibrated = IsotonicCalibrator().fit(scores, labels).transform(scores)
        after = expected_calibration_error(calibrated, labels)

        assert after < before

    def test_unfitted_is_identity(self):
        assert IsotonicCalibrator().transform([0.3, 0.7]) == [0.3, 0.7]

    def test_rejects_tiny_input(self):
        with pytest.raises(ValueError):
            IsotonicCalibrator().fit([0.5], [1])

    def test_serialises(self):
        calibrator = IsotonicCalibrator().fit([0.1, 0.5, 0.9], [0, 1, 1])
        restored = IsotonicCalibrator.from_dict(calibrator.to_dict())
        assert restored.transform([0.5]) == pytest.approx(calibrator.transform([0.5]))


class TestLogisticFit:
    def test_learns_a_separable_signal(self):
        features, labels = [], []
        for i in range(60):
            positive = i % 2 == 0
            features.append(
                ConfidenceFeatures(
                    entail_max=0.9 if positive else 0.1,
                    agreement=1.0 if positive else -1.0,
                    independence=0.8 if positive else 0.1,
                    source_quality=0.9 if positive else 0.2,
                    consistency=1.0,
                    sufficiency=0.9 if positive else 0.1,
                    stated_conf=0.5,
                )
            )
            labels.append(1 if positive else 0)

        weights, bias = fit_logistic(features, labels)
        model = ConfidenceModel(weights=weights, bias=bias)
        assert model.raw_score(features[0]) > 0.7
        assert model.raw_score(features[1]) < 0.3

    def test_rejects_tiny_dataset(self):
        with pytest.raises(ValueError):
            fit_logistic([ConfidenceFeatures()] * 3, [1, 0, 1])

    def test_bundle_round_trip(self, tmp_path):
        bundle = CalibrationBundle(
            weights=dict.fromkeys(ConfidenceFeatures.feature_names(), 1.0),
            bias=-2.0,
            calibrator=IsotonicCalibrator().fit([0.2, 0.5, 0.8], [0, 1, 1]),
            metadata={"ece_after": 0.05},
        )
        path = bundle.save(tmp_path / "cal.json")
        restored = CalibrationBundle.load(path)
        assert restored.bias == -2.0
        assert restored.metadata["ece_after"] == 0.05
        assert restored.to_model().calibrator is not None

    def test_refuses_degenerate_labels(self):
        """All-one-outcome data yields a curve that flattens every score to a constant.

        This actually happened: an eval run where the pipeline abstained on
        everything produced a bundle that silently mapped the whole feature
        space to ~0, and every confidence score in the system went to zero with
        no error anywhere.
        """
        features = [ConfidenceFeatures(entail_max=i / 40) for i in range(40)]

        with pytest.raises(ValueError, match="same outcome"):
            train_calibration(features, [0] * 40)
        with pytest.raises(ValueError, match="same outcome"):
            train_calibration(features, [1] * 40)

    def test_sanity_check_accepts_the_prior_model(self):
        from veritas.verify.calibration import is_sane

        ok, reason = is_sane(ConfidenceModel())
        assert ok, reason

    def test_sanity_check_rejects_a_collapsed_model(self):
        from veritas.verify.calibration import is_sane

        collapsed = ConfidenceModel(
            weights=dict.fromkeys(ConfidenceFeatures.feature_names(), 0.0),
            bias=-5.0,
        )
        ok, reason = is_sane(collapsed)
        assert not ok
        assert "collapsed" in reason or "monotone" in reason

    def test_sanity_check_rejects_an_inverted_model(self):
        from veritas.verify.calibration import is_sane

        inverted = ConfidenceModel(
            weights=dict.fromkeys(ConfidenceFeatures.feature_names(), -2.0),
            bias=2.0,
        )
        assert not is_sane(inverted)[0]

    def test_train_calibration_reports_both_stages(self):
        features, labels = [], []
        for i in range(80):
            positive = i % 3 != 0
            features.append(
                ConfidenceFeatures(
                    entail_max=0.85 if positive else 0.2,
                    agreement=0.9 if positive else -0.5,
                    independence=0.7 if positive else 0.2,
                    source_quality=0.8 if positive else 0.3,
                    consistency=1.0,
                    sufficiency=0.8 if positive else 0.2,
                    stated_conf=0.6,
                )
            )
            labels.append(1 if positive else 0)

        _, report = train_calibration(features, labels)
        assert report["n_train"] + report["n_holdout"] == 80
        assert "ece_before" in report and "ece_after" in report

    def test_temperature_scaling(self):
        sharpened = temperature_scale([0.8], 0.5)[0]
        softened = temperature_scale([0.8], 2.0)[0]
        assert sharpened > 0.8 > softened
        with pytest.raises(ValueError):
            temperature_scale([0.5], 0.0)


class TestMetrics:
    def test_perfect_calibration_scores_zero_ece(self):
        # 100 items at confidence 1.0, all correct; 100 at 0.0, all wrong.
        confidences = [1.0] * 100 + [0.0] * 100
        correct = [1] * 100 + [0] * 100
        assert expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-9)

    def test_overconfidence_is_penalised(self):
        confidences = [0.95] * 100
        correct = [1] * 50 + [0] * 50
        assert expected_calibration_error(confidences, correct) == pytest.approx(0.45, abs=0.01)

    def test_brier(self):
        assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
        assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)

    def test_auroc_perfect_and_random(self):
        assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)
        assert auroc([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0]) == pytest.approx(0.5)

    def test_auroc_degenerate_labels(self):
        assert auroc([0.9, 0.8], [1, 1]) == 0.5

    def test_selective_risk_falls_with_coverage(self):
        confidences = [0.95, 0.9, 0.85, 0.4, 0.3, 0.2]
        correct = [1, 1, 1, 0, 0, 0]
        curve = selective_risk_curve(confidences, correct, steps=10)

        full = next(p for p in curve if p.threshold == 0.0)
        strict = [p for p in curve if p.threshold >= 0.8 and p.n > 0][0]
        assert strict.risk < full.risk
        assert strict.coverage < full.coverage

    def test_macro_f1(self):
        f1, per_label = macro_f1(
            ["SUPPORTED", "REFUTED", "NEI"],
            ["SUPPORTED", "REFUTED", "NEI"],
            ["SUPPORTED", "REFUTED", "NEI"],
        )
        assert f1 == pytest.approx(1.0)
        assert per_label["NEI"]["support"] == 1.0

    def test_evaluate_bundle(self):
        result = evaluate(
            "test",
            predictions=["SUPPORTED", "NEI", "REFUTED"],
            labels=["SUPPORTED", "NEI", "SUPPORTED"],
            confidences=[0.9, 0.4, 0.8],
        )
        assert result.n == 3
        assert result.accuracy == pytest.approx(2 / 3)
        assert result.abstention_rate == pytest.approx(1 / 3)
        assert "accuracy" in result.to_dict()

    def test_empty_inputs_are_safe(self):
        assert expected_calibration_error([], []) == 0.0
        assert brier_score([], []) == 0.0
        assert selective_risk_curve([], []) == []
        assert evaluate("empty", [], [], []).n == 0
