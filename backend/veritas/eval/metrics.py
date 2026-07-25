"""Evaluation metrics.

These are the numbers that decide whether the system's central claim — that it
beats a single LLM on factual correctness *and* transparency — is true. Accuracy
alone would not settle it: a system that is right slightly more often but wildly
overconfident is worse in practice than a slightly less accurate one that knows
when to abstain. Hence calibration and selective-risk metrics carry equal weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def accuracy(predictions: list[str], labels: list[str]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for p, y in zip(predictions, labels, strict=True) if p == y) / len(predictions)


def expected_calibration_error(
    confidences: list[float], correct: list[int], n_bins: int = 10
) -> float:
    """ECE with equal-width bins.

    The headline calibration number: the average gap between stated confidence
    and observed accuracy. Perfect calibration is 0. Published LLM verbalised
    confidence sits around 0.2, which is the bar to beat.
    """
    if not confidences:
        return 0.0

    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    n = len(conf)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0

    for i in range(n_bins):
        lower, upper = edges[i], edges[i + 1]
        mask = (conf > lower) & (conf <= upper) if i > 0 else (conf >= lower) & (conf <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / n) * abs(float(hit[mask].mean()) - float(conf[mask].mean()))

    return float(error)


def maximum_calibration_error(
    confidences: list[float], correct: list[int], n_bins: int = 10
) -> float:
    """Worst-case bin gap — catches a single badly miscalibrated region."""
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else conf <= edges[1]
        if not mask.any():
            continue
        worst = max(worst, abs(float(hit[mask].mean()) - float(conf[mask].mean())))
    return float(worst)


def brier_score(confidences: list[float], correct: list[int]) -> float:
    """Mean squared error of the probability estimates. Lower is better."""
    if not confidences:
        return 0.0
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    return float(np.mean((conf - hit) ** 2))


def auroc(scores: list[float], labels: list[int]) -> float:
    """AUROC via the rank-sum identity, with proper tie handling.

    Measures whether confidence *ranks* correct claims above incorrect ones —
    orthogonal to calibration. A system can rank perfectly (AUROC 1.0) while
    being badly calibrated, and vice versa.
    """
    if not scores:
        return 0.5

    y = np.asarray(labels, dtype=int)
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    if positives == 0 or negatives == 0:
        return 0.5

    order = np.argsort(np.asarray(scores, dtype=float))
    ranks = np.empty(len(scores), dtype=float)
    sorted_scores = np.asarray(scores, dtype=float)[order]

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average_rank
        i = j + 1

    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


@dataclass(slots=True)
class SelectivePoint:
    threshold: float
    coverage: float
    risk: float
    n: int


def selective_risk_curve(
    confidences: list[float], correct: list[int], steps: int = 20
) -> list[SelectivePoint]:
    """Risk as a function of coverage when abstaining below a threshold.

    This is the practical payoff of calibration: it answers "if I only trust
    claims above confidence T, how often am I wrong, and how much do I keep?"
    A useful system's risk falls sharply as coverage drops. A system with
    meaningless confidence shows a flat curve — the single most damning plot you
    can draw of an uncalibrated fact-checker.
    """
    if not confidences:
        return []

    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    total = len(conf)
    points: list[SelectivePoint] = []

    for i in range(steps + 1):
        threshold = i / steps
        mask = conf >= threshold
        count = int(mask.sum())
        if count == 0:
            points.append(SelectivePoint(threshold, 0.0, 0.0, 0))
            continue
        points.append(
            SelectivePoint(
                threshold=threshold,
                coverage=count / total,
                risk=1.0 - float(hit[mask].mean()),
                n=count,
            )
        )
    return points


def reliability_bins(
    confidences: list[float], correct: list[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin accuracy vs confidence — the reliability diagram's data."""
    if not confidences:
        return []
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []

    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1]) if i > 0 else conf <= edges[1]
        count = int(mask.sum())
        out.append(
            {
                "bin_lower": float(edges[i]),
                "bin_upper": float(edges[i + 1]),
                "count": float(count),
                "mean_confidence": float(conf[mask].mean()) if count else 0.0,
                "accuracy": float(hit[mask].mean()) if count else 0.0,
            }
        )
    return out


@dataclass
class EvaluationResult:
    """Full metric bundle for one system on one dataset."""

    name: str
    n: int = 0
    accuracy: float = 0.0
    macro_f1: float = 0.0
    ece: float = 0.0
    mce: float = 0.0
    brier: float = 0.0
    auroc: float = 0.5
    abstention_rate: float = 0.0
    accuracy_on_answered: float = 0.0
    per_label: dict[str, dict[str, float]] = field(default_factory=dict)
    selective: list[SelectivePoint] = field(default_factory=list)
    reliability: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "ece": round(self.ece, 4),
            "mce": round(self.mce, 4),
            "brier": round(self.brier, 4),
            "auroc": round(self.auroc, 4),
            "abstention_rate": round(self.abstention_rate, 4),
            "accuracy_on_answered": round(self.accuracy_on_answered, 4),
            "per_label": self.per_label,
            "selective": [
                {
                    "threshold": round(p.threshold, 3),
                    "coverage": round(p.coverage, 4),
                    "risk": round(p.risk, 4),
                    "n": p.n,
                }
                for p in self.selective
            ],
            "reliability": self.reliability,
        }


def macro_f1(predictions: list[str], labels: list[str], classes: list[str]) -> tuple[float, dict]:
    """Macro-averaged F1 with the full per-class breakdown."""
    per_label: dict[str, dict[str, float]] = {}
    f1_scores: list[float] = []

    for cls in classes:
        tp = sum(1 for p, y in zip(predictions, labels, strict=True) if p == cls and y == cls)
        fp = sum(1 for p, y in zip(predictions, labels, strict=True) if p == cls and y != cls)
        fn = sum(1 for p, y in zip(predictions, labels, strict=True) if p != cls and y == cls)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_label[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": float(sum(1 for y in labels if y == cls)),
        }
        f1_scores.append(f1)

    return (sum(f1_scores) / len(f1_scores) if f1_scores else 0.0), per_label


def evaluate(
    name: str,
    predictions: list[str],
    labels: list[str],
    confidences: list[float],
    abstain_label: str = "NEI",
) -> EvaluationResult:
    """Compute the full metric bundle for one system's predictions."""
    if not predictions:
        return EvaluationResult(name=name)

    correct = [1 if p == y else 0 for p, y in zip(predictions, labels, strict=True)]
    classes = sorted(set(labels) | set(predictions))
    f1, per_label = macro_f1(predictions, labels, classes)

    answered = [
        (p, y, c)
        for p, y, c in zip(predictions, labels, confidences, strict=True)
        if p != abstain_label
    ]
    abstentions = len(predictions) - len(answered)

    return EvaluationResult(
        name=name,
        n=len(predictions),
        accuracy=accuracy(predictions, labels),
        macro_f1=f1,
        ece=expected_calibration_error(confidences, correct),
        mce=maximum_calibration_error(confidences, correct),
        brier=brier_score(confidences, correct),
        auroc=auroc(confidences, correct),
        abstention_rate=abstentions / len(predictions),
        accuracy_on_answered=(
            sum(1 for p, y, _ in answered if p == y) / len(answered) if answered else 0.0
        ),
        per_label=per_label,
        selective=selective_risk_curve(confidences, correct),
        reliability=reliability_bins(confidences, correct),
    )


def comparison_table(results: list[EvaluationResult]) -> str:
    """Markdown comparison table — this is the slide that wins the room."""
    if not results:
        return "_no results_"

    header = (
        "| System | n | Accuracy | Macro-F1 | ECE ↓ | Brier ↓ | AUROC ↑ | "
        "Abstain % | Acc. when answered |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    rows = [
        f"| **{r.name}** | {r.n} | {r.accuracy:.3f} | {r.macro_f1:.3f} | {r.ece:.3f} | "
        f"{r.brier:.3f} | {r.auroc:.3f} | {r.abstention_rate * 100:.1f}% | "
        f"{r.accuracy_on_answered:.3f} |"
        for r in results
    ]
    return "\n".join([header, *rows])
