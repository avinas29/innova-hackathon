"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { ConfidenceExplanation } from "@/lib/types";
import { cn, confidenceColour } from "@/lib/ui";

const SLIDERS: { key: string; label: string; min: number; max: number }[] = [
  { key: "entail_max", label: "Strongest entailment", min: 0, max: 1 },
  { key: "agreement", label: "Evidence agreement", min: -1, max: 1 },
  { key: "independence", label: "Source independence", min: 0, max: 1 },
  { key: "source_quality", label: "Source quality", min: 0, max: 1 },
  { key: "consistency", label: "Verdict stability", min: 0, max: 1 },
  { key: "sufficiency", label: "Retrieval sufficiency", min: 0, max: 1 },
  { key: "stated_conf", label: "Model's own confidence", min: 0, max: 1 },
];

const DEFAULTS: Record<string, number> = {
  entail_max: 0.9,
  agreement: 0.8,
  independence: 0.6,
  source_quality: 0.8,
  consistency: 1,
  sufficiency: 0.7,
  stated_conf: 0.9,
  n_independent: 1,
};

/**
 * Interactive explainer for the confidence model.
 *
 * Exists because "trust our score" is not an argument. Drag `stated_conf` to
 * 1.0 with everything else at zero and watch the score stay near the floor —
 * that is the ensemble refusing to be talked into certainty by a model's own
 * say-so. Raise `n_independent` and watch the ceiling lift.
 */
export function ConfidenceExplainer() {
  const [values, setValues] = useState<Record<string, number>>(DEFAULTS);
  const [result, setResult] = useState<ConfidenceExplanation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async (next: Record<string, number>) => {
    try {
      setResult(await api.explainConfidence(next));
      setError(null);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : "request failed");
    }
  }, []);

  useEffect(() => {
    void refresh(values);
  }, [values, refresh]);

  const update = (key: string, value: number) =>
    setValues((previous) => ({ ...previous, [key]: value }));

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="panel-title">Confidence model</span>
        <span className="text-xs text-zinc-500">
          seven features, not one opinion
        </span>
      </div>

      <div className="grid gap-6 p-5 lg:grid-cols-2">
        <div className="space-y-3">
          {SLIDERS.map((slider) => (
            <div key={slider.key}>
              <div className="mb-1 flex items-center justify-between text-xs">
                <span className="text-zinc-400">{slider.label}</span>
                <span className="font-mono tabular-nums text-zinc-300">
                  {values[slider.key].toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                min={slider.min}
                max={slider.max}
                step={0.05}
                value={values[slider.key]}
                onChange={(e) => update(slider.key, Number(e.target.value))}
                className="h-1 w-full cursor-pointer appearance-none rounded-full bg-zinc-800 accent-cyan-500"
              />
            </div>
          ))}

          <div className="border-t border-zinc-800 pt-3">
            <div className="mb-1 flex items-center justify-between text-xs">
              <span className="text-zinc-400">Independent sources</span>
              <span className="font-mono tabular-nums text-zinc-300">
                {values.n_independent}
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={10}
              step={1}
              value={values.n_independent}
              onChange={(e) => update("n_independent", Number(e.target.value))}
              className="h-1 w-full cursor-pointer appearance-none rounded-full bg-zinc-800 accent-cyan-500"
            />
          </div>
        </div>

        <div>
          {error ? (
            <p className="text-sm text-rose-400">
              Could not reach the API: {error}
            </p>
          ) : null}

          {result ? (
            <div className="space-y-4">
              <div>
                <div className="stat-label mb-1">Final confidence</div>
                <div className="flex items-baseline gap-3">
                  <span className="font-mono text-4xl font-semibold tabular-nums">
                    {result.final_score.toFixed(3)}
                  </span>
                  {result.capped ? (
                    <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-xs text-amber-400">
                      capped at {result.independence_ceiling.toFixed(2)}
                    </span>
                  ) : null}
                </div>
                <div className="mt-2 h-2 overflow-hidden rounded-full bg-zinc-800">
                  <div
                    className={cn(
                      "h-full rounded-full transition-all",
                      confidenceColour(result.final_score),
                    )}
                    style={{ width: `${result.final_score * 100}%` }}
                  />
                </div>
                {result.capped ? (
                  <p className="mt-2 text-xs text-amber-400/80">
                    {values.n_independent === 0
                      ? "No independent source can license confidence."
                      : `${values.n_independent} independent source${values.n_independent === 1 ? "" : "s"} cannot license more than ${result.independence_ceiling.toFixed(2)}, whatever the other signals say.`}
                  </p>
                ) : null}
              </div>

              <div>
                <div className="stat-label mb-2">
                  Contribution to the log-odds
                </div>
                <div className="space-y-1.5">
                  {result.contributions.map((item) => {
                    const magnitude = Math.min(
                      1,
                      Math.abs(item.contribution) / 3,
                    );
                    return (
                      <div
                        key={item.feature}
                        className="flex items-center gap-2 text-xs"
                      >
                        <span className="w-32 shrink-0 truncate text-zinc-500">
                          {item.feature}
                        </span>
                        <div className="relative h-3 flex-1 rounded bg-zinc-800/60">
                          <div
                            className={cn(
                              "absolute top-0 h-full rounded",
                              item.contribution >= 0
                                ? "left-1/2 bg-emerald-500/70"
                                : "right-1/2 bg-rose-500/70",
                            )}
                            style={{ width: `${magnitude * 50}%` }}
                          />
                          <div className="absolute left-1/2 top-0 h-full w-px bg-zinc-700" />
                        </div>
                        <span className="w-12 shrink-0 text-right font-mono tabular-nums text-zinc-400">
                          {item.contribution >= 0 ? "+" : ""}
                          {item.contribution.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <p className="mt-3 text-xs text-zinc-600">
                  Bias {result.bias.toFixed(2)} · raw{" "}
                  {result.raw_score.toFixed(3)} · calibrated{" "}
                  {result.calibrated_score.toFixed(3)}
                </p>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
