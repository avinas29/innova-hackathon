"""Probability calibration — isotonic regression and logistic fitting.

Implemented directly rather than pulled from scikit-learn: the two algorithms
are about eighty lines together, and this keeps the dependency footprint small
enough that the whole backend installs in seconds on a hackathon machine.

*Isotonic regression* (pool-adjacent-violators) is the right choice over Platt
scaling here because it is non-parametric — it can correct an arbitrary
monotone distortion, and our raw score comes from hand-set priors whose
distortion is certainly not a neat sigmoid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from veritas.logging import get_logger
from veritas.schemas import ConfidenceFeatures

log = get_logger(__name__)

# Below this, a non-parametric fit is dominated by whichever few labels happen
# to land in the holdout.
MIN_HOLDOUT_FOR_ISOTONIC = 15


def is_sane(model) -> tuple[bool, str]:
    """Sanity-check a fitted confidence model before trusting it.

    A calibration bundle is loaded from disk automatically, so a bad one poisons
    every score in the system silently — there is no error, just uniformly wrong
    numbers. Two probes catch the realistic failure modes:

    * **Monotonicity** — maximal evidence must outscore no evidence. A bundle
      fitted on degenerate labels inverts or flattens this.
    * **Range** — maximal evidence must clear 0.5. A collapsed isotonic curve
      maps the whole feature space to a constant near zero.
    """
    from veritas.schemas import ConfidenceFeatures

    strong = ConfidenceFeatures(
        entail_max=1.0,
        agreement=1.0,
        independence=1.0,
        source_quality=1.0,
        consistency=1.0,
        sufficiency=1.0,
        stated_conf=1.0,
    )
    weak = ConfidenceFeatures(
        entail_max=0.0,
        agreement=-1.0,
        independence=0.0,
        source_quality=0.0,
        consistency=0.0,
        sufficiency=0.0,
        stated_conf=0.0,
    )

    high = model.calibrated_score(strong)
    low = model.calibrated_score(weak)

    if high <= low:
        return False, f"not monotone: strong evidence scores {high:.3f} vs weak {low:.3f}"
    if high < 0.5:
        return False, f"collapsed range: strongest possible evidence scores only {high:.3f}"
    return True, ""


class IsotonicCalibrator:
    """Monotone piecewise-constant map from raw score to calibrated probability.

    Fitted with the pool-adjacent-violators algorithm, which finds the
    least-squares-optimal non-decreasing fit in O(n) after sorting.
    """

    def __init__(self, x: list[float] | None = None, y: list[float] | None = None) -> None:
        self.x: list[float] = x or []
        self.y: list[float] = y or []

    @property
    def fitted(self) -> bool:
        return len(self.x) > 0

    def fit(self, scores: list[float], labels: list[int]) -> IsotonicCalibrator:
        """Fit on raw scores and binary correctness labels."""
        if len(scores) != len(labels):
            raise ValueError("scores and labels must be the same length")
        if len(scores) < 3:
            raise ValueError("need at least 3 points to calibrate")

        order = np.argsort(np.asarray(scores, dtype=float))
        xs = np.asarray(scores, dtype=float)[order]
        ys = np.asarray(labels, dtype=float)[order]

        # PAVA: repeatedly merge adjacent blocks that violate monotonicity.
        values: list[float] = []
        weights: list[float] = []
        for value in ys:
            values.append(float(value))
            weights.append(1.0)
            while len(values) > 1 and values[-2] > values[-1]:
                merged_weight = weights[-2] + weights[-1]
                merged_value = (
                    values[-2] * weights[-2] + values[-1] * weights[-1]
                ) / merged_weight
                values.pop()
                weights.pop()
                values[-1] = merged_value
                weights[-1] = merged_weight

        # Expand pooled blocks back to per-sample fitted values.
        fitted: list[float] = []
        for value, weight in zip(values, weights, strict=True):
            fitted.extend([value] * int(round(weight)))

        self.x = [float(v) for v in xs]
        self.y = [min(1.0, max(0.0, v)) for v in fitted[: len(xs)]]
        log.info("isotonic calibrator fitted", points=len(self.x))
        return self

    def transform(self, scores: list[float]) -> list[float]:
        """Map raw scores through the fitted curve, interpolating between knots."""
        if not self.fitted:
            return list(scores)
        xs = np.asarray(self.x)
        ys = np.asarray(self.y)
        return [float(np.interp(s, xs, ys, left=ys[0], right=ys[-1])) for s in scores]

    def to_dict(self) -> dict:
        return {"type": "isotonic", "x": self.x, "y": self.y}

    @classmethod
    def from_dict(cls, data: dict) -> IsotonicCalibrator:
        return cls(x=list(data.get("x", [])), y=list(data.get("y", [])))


def fit_logistic(
    features: list[ConfidenceFeatures],
    labels: list[int],
    *,
    learning_rate: float = 0.35,
    epochs: int = 900,
    l2: float = 0.012,
    seed: int = 7,
) -> tuple[dict[str, float], float]:
    """Fit feature weights and bias by regularised gradient descent.

    L2 regularisation matters more than usual here: dev sets for this task are
    small (tens to low hundreds of claims), and an unregularised fit will happily
    place enormous weight on ``stated_conf`` because it correlates with
    correctness in-sample while being miscalibrated out-of-sample.
    """
    if len(features) != len(labels):
        raise ValueError("features and labels must be the same length")
    if len(features) < 10:
        raise ValueError("need at least 10 labelled examples to fit weights")

    names = ConfidenceFeatures.feature_names()
    X = np.asarray([f.as_vector() for f in features], dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n, d = X.shape

    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 0.01, size=d)
    b = 0.0

    for epoch in range(epochs):
        logits = X @ w + b
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        error = preds - y

        grad_w = (X.T @ error) / n + l2 * w
        grad_b = float(np.mean(error))

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        if epoch % 300 == 0:
            loss = _log_loss(y, preds) + 0.5 * l2 * float(np.sum(w**2))
            log.debug("logistic fit", epoch=epoch, loss=round(loss, 5))

    weights = {name: float(value) for name, value in zip(names, w, strict=True)}
    log.info(
        "logistic weights fitted",
        n=n,
        **{k: round(v, 3) for k, v in weights.items()},
    )
    return weights, float(b)


def _log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class CalibrationBundle:
    """Serialisable weights + bias + isotonic curve, persisted as one JSON file."""

    def __init__(
        self,
        weights: dict[str, float],
        bias: float,
        calibrator: IsotonicCalibrator | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.weights = weights
        self.bias = bias
        self.calibrator = calibrator
        self.metadata = metadata or {}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights,
            "bias": self.bias,
            "calibrator": self.calibrator.to_dict() if self.calibrator else None,
            "metadata": self.metadata,
        }
        path.write_text(json.dumps(payload, indent=2))
        log.info("calibration bundle saved", path=str(path))
        return path

    @classmethod
    def load(cls, path: str | Path) -> CalibrationBundle:
        data = json.loads(Path(path).read_text())
        calibrator = (
            IsotonicCalibrator.from_dict(data["calibrator"]) if data.get("calibrator") else None
        )
        return cls(
            weights=data["weights"],
            bias=float(data["bias"]),
            calibrator=calibrator,
            metadata=data.get("metadata", {}),
        )

    def to_model(self):
        from veritas.verify.confidence import ConfidenceModel

        return ConfidenceModel(
            weights=self.weights, bias=self.bias, calibrator=self.calibrator
        )


def train_calibration(
    features: list[ConfidenceFeatures],
    labels: list[int],
    *,
    holdout_fraction: float = 0.3,
    seed: int = 7,
) -> tuple[CalibrationBundle, dict[str, float]]:
    """Fit weights on a training split, then the isotonic curve on a holdout.

    Fitting both stages on the same data would let the calibrator absorb the
    logistic model's in-sample overconfidence and report a flatteringly low ECE
    that does not survive contact with new claims.
    """
    from veritas.eval.metrics import brier_score, expected_calibration_error
    from veritas.verify.confidence import ConfidenceModel

    if len(features) < 20:
        raise ValueError("need at least 20 labelled examples to train a calibration bundle")

    # Label variance is the real precondition, not just row count. Fitting on
    # all-correct or all-incorrect labels yields an isotonic curve that maps the
    # entire feature space to a constant — which then silently flattens every
    # confidence score in the system to that constant. Refuse loudly instead.
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        raise ValueError(
            f"cannot calibrate: all {len(labels)} examples have the same outcome "
            f"({'all correct' if positives else 'all incorrect'}). This usually means "
            "the pipeline abstained on everything — check that a real model provider "
            "and a search provider are configured, then re-run `veritas eval`."
        )

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(features))
    split = int(len(features) * (1 - holdout_fraction))
    train_idx, holdout_idx = indices[:split], indices[split:]

    train_features = [features[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    holdout_features = [features[i] for i in holdout_idx]
    holdout_labels = [labels[i] for i in holdout_idx]

    weights, bias = fit_logistic(train_features, train_labels, seed=seed)
    model = ConfidenceModel(weights=weights, bias=bias)

    holdout_raw = [model.raw_score(f) for f in holdout_features]
    calibrator = IsotonicCalibrator()

    # A handful of holdout points cannot support a non-parametric fit; the curve
    # would be dominated by whichever few labels happened to land there.
    if len(holdout_raw) < MIN_HOLDOUT_FOR_ISOTONIC or len(set(holdout_labels)) < 2:
        log.warning(
            "isotonic calibration skipped — holdout too small or single-class",
            n_holdout=len(holdout_raw),
            classes=len(set(holdout_labels)),
        )
    else:
        try:
            calibrator.fit(holdout_raw, holdout_labels)
        except ValueError as exc:
            log.warning("isotonic fit skipped", error=str(exc))
            calibrator = IsotonicCalibrator()

    calibrated = calibrator.transform(holdout_raw) if calibrator.fitted else holdout_raw
    report = {
        "n_total": float(len(features)),
        "n_train": float(len(train_features)),
        "n_holdout": float(len(holdout_features)),
        "ece_before": expected_calibration_error(holdout_raw, holdout_labels),
        "ece_after": expected_calibration_error(calibrated, holdout_labels),
        "brier_before": brier_score(holdout_raw, holdout_labels),
        "brier_after": brier_score(calibrated, holdout_labels),
    }

    bundle = CalibrationBundle(
        weights=weights, bias=bias, calibrator=calibrator, metadata=report
    )
    log.info(
        "calibration trained",
        ece_before=round(report["ece_before"], 4),
        ece_after=round(report["ece_after"], 4),
    )
    return bundle, report


def temperature_scale(scores: list[float], temperature: float) -> list[float]:
    """Sharpen (T<1) or soften (T>1) probabilities in logit space."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    out: list[float] = []
    for score in scores:
        clipped = min(max(score, 1e-6), 1 - 1e-6)
        logit = math.log(clipped / (1 - clipped)) / temperature
        out.append(1.0 / (1.0 + math.exp(-logit)))
    return out
