"""Evaluation driver: pipeline vs single-LLM baseline on the same claims."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from veritas.config import Settings, get_settings
from veritas.eval.baseline import SingleLLMBaseline
from veritas.eval.dataset import LabelledClaim, load_dataset
from veritas.eval.metrics import EvaluationResult, comparison_table, evaluate
from veritas.logging import get_logger
from veritas.schemas import ConfidenceFeatures

log = get_logger(__name__)


async def evaluate_pipeline(
    items: list[LabelledClaim], settings: Settings | None = None
) -> tuple[list[dict[str, Any]], list[ConfidenceFeatures], list[int]]:
    """Run the full verification subgraph over each labelled claim."""
    from veritas.graph.nodes import verify_single_claim
    from veritas.state import build_context

    settings = settings or get_settings()
    context = await build_context(settings)
    semaphore = asyncio.Semaphore(max(1, settings.verify_concurrency // 2))

    async def one(item: LabelledClaim) -> dict[str, Any]:
        async with semaphore:
            try:
                result = await verify_single_claim(context, item.claim, run_id="eval")
                claim = result["verified_claims"][0]
                return {
                    "claim": item.claim,
                    "gold": item.label,
                    "predicted": claim.verdict.value,
                    "confidence": claim.confidence,
                    "raw_confidence": claim.raw_confidence,
                    "features": claim.features,
                    "rationale": claim.rationale[:300],
                    "n_evidence": len(claim.evidence_ids),
                    "n_clusters": len(claim.cluster_ids),
                }
            except Exception as exc:
                log.warning("pipeline eval item failed", error=str(exc)[:200])
                return {
                    "claim": item.claim,
                    "gold": item.label,
                    "predicted": "NEI",
                    "confidence": 0.0,
                    "raw_confidence": 0.0,
                    "features": ConfidenceFeatures(),
                    "rationale": f"error: {exc}"[:200],
                    "n_evidence": 0,
                    "n_clusters": 0,
                }

    try:
        log.info("running verification pipeline", n=len(items))
        records = list(await asyncio.gather(*(one(item) for item in items)))
    finally:
        await context.aclose()

    features = [r["features"] for r in records]
    labels = [1 if r["predicted"] == r["gold"] else 0 for r in records]
    return records, features, labels


async def run_evaluation(
    dataset: str = "builtin",
    limit: int = 30,
    include_baseline: bool = True,
    settings: Settings | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Benchmark the pipeline against the single-LLM control."""
    settings = settings or get_settings()
    items, description = load_dataset(dataset, limit)
    started = time.time()

    log.info("evaluation starting", dataset=description, n=len(items))

    records, features, correctness = await evaluate_pipeline(items, settings)
    pipeline_result = evaluate(
        name="VERITAS pipeline",
        predictions=[r["predicted"] for r in records],
        labels=[r["gold"] for r in records],
        confidences=[r["confidence"] for r in records],
    )

    results: list[EvaluationResult] = [pipeline_result]
    baseline_records: list[dict[str, Any]] = []

    if include_baseline:
        from veritas.llm.client import LLMClient

        llm = LLMClient(settings)
        try:
            predictions = await SingleLLMBaseline(llm).predict(items)
        finally:
            await llm.aclose()

        baseline_records = [p.model_dump() for p in predictions]
        results.append(
            evaluate(
                name="Single LLM (no retrieval)",
                predictions=[p.predicted for p in predictions],
                labels=[p.gold for p in predictions],
                confidences=[p.confidence for p in predictions],
            )
        )

    payload: dict[str, Any] = {
        "dataset": description,
        "n": len(items),
        "provider": settings.resolved_provider,
        "duration_seconds": round(time.time() - started, 2),
        "results": [r.to_dict() for r in results],
        "comparison_table": comparison_table(results),
        "verdict": _headline(results),
        "pipeline_records": [
            {k: v for k, v in r.items() if k != "features"} for r in records
        ],
        "baseline_records": baseline_records,
        "label_distribution": _distribution([i.label for i in items]),
    }

    # Feature vectors are retained separately so `veritas calibrate` can reuse
    # an eval run's output as training data instead of re-running the pipeline.
    payload["calibration_data"] = {
        "features": [f.model_dump() for f in features],
        "labels": correctness,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str))
        log.info("evaluation written", path=str(path))

    log.info(
        "evaluation complete",
        pipeline_acc=round(pipeline_result.accuracy, 3),
        pipeline_ece=round(pipeline_result.ece, 3),
        duration=payload["duration_seconds"],
    )
    return payload


def _headline(results: list[EvaluationResult]) -> dict[str, Any]:
    """Plain-language summary of whether the pipeline beat the control."""
    if len(results) < 2:
        return {"comparable": False, "note": "baseline not run"}

    pipeline, baseline = results[0], results[1]
    return {
        "comparable": True,
        "accuracy_delta": round(pipeline.accuracy - baseline.accuracy, 4),
        "ece_delta": round(pipeline.ece - baseline.ece, 4),
        "brier_delta": round(pipeline.brier - baseline.brier, 4),
        "auroc_delta": round(pipeline.auroc - baseline.auroc, 4),
        "abstention_delta": round(pipeline.abstention_rate - baseline.abstention_rate, 4),
        "better_accuracy": pipeline.accuracy > baseline.accuracy,
        "better_calibration": pipeline.ece < baseline.ece,
        "summary": _summary_line(pipeline, baseline),
    }


def _summary_line(pipeline: EvaluationResult, baseline: EvaluationResult) -> str:
    parts = []
    accuracy_delta = pipeline.accuracy - baseline.accuracy
    parts.append(
        f"accuracy {'+' if accuracy_delta >= 0 else ''}{accuracy_delta:.1%} "
        f"({pipeline.accuracy:.1%} vs {baseline.accuracy:.1%})"
    )
    ece_delta = baseline.ece - pipeline.ece
    parts.append(
        f"calibration error {'reduced' if ece_delta > 0 else 'increased'} by "
        f"{abs(ece_delta):.3f} ({pipeline.ece:.3f} vs {baseline.ece:.3f})"
    )
    parts.append(
        f"abstention {pipeline.abstention_rate:.1%} vs {baseline.abstention_rate:.1%}"
    )
    return "; ".join(parts)


def _distribution(labels: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in labels:
        out[label] = out.get(label, 0) + 1
    return out


async def calibrate_from_eval(
    eval_payload: dict[str, Any], output_path: str | Path = "calibration.json"
) -> dict[str, Any]:
    """Fit and persist a calibration bundle from an evaluation run's output."""
    from veritas.verify.calibration import train_calibration

    data = eval_payload.get("calibration_data", {})
    features = [ConfidenceFeatures.model_validate(f) for f in data.get("features", [])]
    labels = list(data.get("labels", []))

    if len(features) < 20:
        raise ValueError(
            f"need at least 20 evaluated claims to calibrate, got {len(features)}. "
            "Re-run `veritas eval` with a larger --limit or a bigger dataset."
        )

    bundle, report = train_calibration(features, labels)
    bundle.save(output_path)
    return report
