"use client";

import type { RunMetrics } from "@/lib/types";
import { formatDuration, formatNumber } from "@/lib/ui";

function Stat({
  label,
  value,
  hint,
  accent,
}: {
  label: string;
  value: string | number;
  hint?: string;
  accent?: string;
}) {
  return (
    <div className="panel px-4 py-3">
      <div className="stat-label">{label}</div>
      <div className={`stat-value ${accent ?? ""}`}>{value}</div>
      {hint ? (
        <div className="mt-0.5 text-[11px] text-zinc-500">{hint}</div>
      ) : null}
    </div>
  );
}

export function MetricsBar({ metrics }: { metrics: RunMetrics }) {
  const verified =
    metrics.supported + metrics.refuted + metrics.nei || 1;
  const supportRate = ((metrics.supported / verified) * 100).toFixed(0);

  // The headline number for the independence story: how much apparent
  // corroboration survived de-duplication.
  const independence =
    metrics.evidence_items > 0
      ? (metrics.independent_clusters / metrics.evidence_items) * 100
      : 0;

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-3 lg:grid-cols-6">
      <Stat
        label="Supported"
        value={metrics.supported}
        hint={`${supportRate}% of verified`}
        accent="text-emerald-400"
      />
      <Stat label="Refuted" value={metrics.refuted} accent="text-rose-400" />
      <Stat
        label="Not established"
        value={metrics.nei}
        hint="abstained"
        accent="text-amber-400"
      />
      <Stat
        label="Independent sources"
        value={`${metrics.independent_clusters}/${metrics.evidence_items}`}
        hint={`${independence.toFixed(0)}% independent after dedup`}
      />
      <Stat
        label="Mean confidence"
        value={metrics.mean_confidence.toFixed(2)}
        hint="calibrated"
      />
      <Stat
        label="Cost"
        value={formatNumber(
          metrics.tokens.prompt_tokens + metrics.tokens.completion_tokens,
        )}
        hint={`${metrics.tokens.calls} calls · ${formatDuration(metrics.duration_seconds)}`}
      />
    </div>
  );
}
