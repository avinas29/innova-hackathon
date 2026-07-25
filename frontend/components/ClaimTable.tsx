"use client";

import { ChevronDown, ExternalLink, Scale, ShieldAlert } from "lucide-react";
import { useMemo, useState } from "react";

import type { Claim, Evidence, Verdict } from "@/lib/types";
import {
  VERDICT_STYLES,
  cn,
  confidenceColour,
  confidenceTextColour,
} from "@/lib/ui";

type Filter = "ALL" | Verdict;

const FEATURE_LABELS: Record<string, string> = {
  entail_max: "Strongest entailment",
  agreement: "Evidence agreement",
  independence: "Source independence",
  source_quality: "Source quality",
  consistency: "Verdict stability",
  sufficiency: "Retrieval sufficiency",
  stated_conf: "Model's own confidence",
};

function ConfidenceBar({ value }: { value: number }) {
  return (
    <div className="flex items-center gap-2">
      <div className="h-1.5 w-16 overflow-hidden rounded-full bg-zinc-800">
        <div
          className={cn("h-full rounded-full transition-all", confidenceColour(value))}
          style={{ width: `${Math.max(2, value * 100)}%` }}
        />
      </div>
      <span
        className={cn(
          "w-9 text-right font-mono text-xs tabular-nums",
          confidenceTextColour(value),
        )}
      >
        {value.toFixed(2)}
      </span>
    </div>
  );
}

function ClaimDetail({
  claim,
  evidence,
}: {
  claim: Claim;
  evidence: Evidence[];
}) {
  const forClaim = evidence.filter((e) => e.claim_id === claim.id);
  const independent = forClaim.filter((e) => !e.is_derivative);
  const derivative = forClaim.length - independent.length;

  return (
    <div className="space-y-4 border-t border-zinc-800 bg-zinc-950/60 px-4 py-4 text-sm">
      {claim.rationale ? (
        <p className="text-zinc-400">{claim.rationale}</p>
      ) : null}

      {claim.revision ? (
        <div className="rounded-lg border border-sky-500/30 bg-sky-500/5 p-3">
          <div className="mb-1 text-xs font-medium text-sky-400">
            Revised by the reflection agent
          </div>
          <p className="text-zinc-300">{claim.revision}</p>
        </div>
      ) : null}

      {claim.minority_report ? (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-3">
          <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-amber-400">
            <ShieldAlert className="h-3.5 w-3.5" />
            Minority report — preserved, not averaged away
          </div>
          <p className="text-zinc-300">{claim.minority_report}</p>
        </div>
      ) : null}

      {claim.advocate_argument || claim.sceptic_argument ? (
        <div className="grid gap-3 md:grid-cols-2">
          <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-emerald-400">
              <Scale className="h-3.5 w-3.5" />
              Advocate (evidence half A)
            </div>
            <p className="text-zinc-400">{claim.advocate_argument || "—"}</p>
          </div>
          <div className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-3">
            <div className="mb-1 flex items-center gap-1.5 text-xs font-medium text-rose-400">
              <Scale className="h-3.5 w-3.5" />
              Sceptic (evidence half B)
            </div>
            <p className="text-zinc-400">{claim.sceptic_argument || "—"}</p>
          </div>
        </div>
      ) : null}

      <div>
        <div className="mb-2 text-xs font-medium text-zinc-400">
          Why this score
        </div>
        <div className="grid gap-x-6 gap-y-1 sm:grid-cols-2">
          {Object.entries(claim.features).map(([key, value]) => (
            <div key={key} className="flex items-center justify-between gap-3">
              <span className="text-xs text-zinc-500">
                {FEATURE_LABELS[key] ?? key}
              </span>
              <span className="font-mono text-xs tabular-nums text-zinc-300">
                {value.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div>
        <div className="mb-2 flex items-center gap-2 text-xs font-medium text-zinc-400">
          Evidence
          <span className="text-zinc-600">
            {forClaim.length} items → {independent.length} independent
            {derivative > 0 ? ` (${derivative} duplicate)` : ""}
          </span>
        </div>
        <div className="space-y-1.5">
          {forClaim.length === 0 ? (
            <p className="text-xs text-zinc-600">No evidence retrieved.</p>
          ) : null}
          {forClaim
            .slice()
            .sort((a, b) => b.entailment_score - a.entailment_score)
            .map((item) => (
              <div
                key={item.id}
                className={cn(
                  "rounded-lg border p-2.5",
                  item.is_derivative
                    ? "border-zinc-800/60 bg-zinc-900/30 opacity-60"
                    : "border-zinc-800 bg-zinc-900/50",
                )}
              >
                <div className="mb-1 flex items-center gap-2 text-xs">
                  <span
                    className={cn(
                      "font-medium",
                      item.stance === "SUPPORTS" && "text-emerald-400",
                      item.stance === "REFUTES" && "text-rose-400",
                      item.stance === "NEUTRAL" && "text-zinc-500",
                    )}
                  >
                    {item.stance}
                  </span>
                  <span className="font-mono tabular-nums text-zinc-500">
                    {item.entailment_score.toFixed(2)}
                  </span>
                  {item.url ? (
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1 text-zinc-400 hover:text-cyan-400"
                    >
                      {item.domain}
                      <ExternalLink className="h-3 w-3" />
                    </a>
                  ) : (
                    <span className="text-zinc-400">{item.domain}</span>
                  )}
                  {item.is_derivative ? (
                    <span className="rounded bg-zinc-800 px-1.5 py-0.5 text-[10px] text-zinc-400">
                      duplicate — not counted again
                    </span>
                  ) : null}
                </div>
                <p className="line-clamp-3 text-xs text-zinc-400">
                  {item.snippet}
                </p>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

export function ClaimTable({
  claims,
  evidence,
}: {
  claims: Claim[];
  evidence: Evidence[];
}) {
  const [filter, setFilter] = useState<Filter>("ALL");
  const [expanded, setExpanded] = useState<string | null>(null);

  const verified = useMemo(
    () => claims.filter((c) => c.category === "CHECK_WORTHY"),
    [claims],
  );

  const counts = useMemo(() => {
    const out: Record<string, number> = { ALL: verified.length };
    for (const claim of verified) {
      out[claim.verdict] = (out[claim.verdict] ?? 0) + 1;
    }
    return out;
  }, [verified]);

  const visible = useMemo(() => {
    const list =
      filter === "ALL" ? verified : verified.filter((c) => c.verdict === filter);
    return list.slice().sort((a, b) => b.confidence - a.confidence);
  }, [verified, filter]);

  const skipped = claims.length - verified.length;

  return (
    <div className="panel flex h-full flex-col">
      <div className="panel-header">
        <span className="panel-title">Claim verdicts</span>
        <div className="flex gap-1">
          {(["ALL", "SUPPORTED", "REFUTED", "NEI"] as Filter[]).map((option) => (
            <button
              key={option}
              onClick={() => setFilter(option)}
              className={cn(
                "rounded-md px-2 py-1 text-xs transition-colors",
                filter === option
                  ? "bg-zinc-700 text-zinc-100"
                  : "text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300",
              )}
            >
              {option === "ALL" ? "All" : VERDICT_STYLES[option].label}
              <span className="ml-1.5 tabular-nums text-zinc-500">
                {counts[option] ?? 0}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        {visible.length === 0 ? (
          <p className="px-5 py-10 text-center text-sm text-zinc-600">
            No claims in this category.
          </p>
        ) : null}

        {visible.map((claim) => {
          const style = VERDICT_STYLES[claim.verdict];
          const isOpen = expanded === claim.id;
          return (
            <div key={claim.id} className="border-b border-zinc-800/60 last:border-0">
              <button
                onClick={() => setExpanded(isOpen ? null : claim.id)}
                className="flex w-full items-start gap-3 px-4 py-3 text-left transition-colors hover:bg-zinc-800/30"
              >
                <span className={cn("badge mt-0.5 shrink-0", style.className)}>
                  <span className={cn("h-1.5 w-1.5 rounded-full", style.dot)} />
                  {style.label}
                </span>

                <span
                  className={cn(
                    "min-w-0 flex-1 text-sm",
                    claim.retracted
                      ? "text-zinc-600 line-through"
                      : "text-zinc-200",
                  )}
                >
                  {claim.revision || claim.decontextualised || claim.text}
                  {claim.retracted ? (
                    <span className="ml-2 rounded bg-rose-500/10 px-1.5 py-0.5 text-[10px] text-rose-400 no-underline">
                      retracted
                    </span>
                  ) : null}
                </span>

                <span className="shrink-0 pt-0.5">
                  <ConfidenceBar value={claim.confidence} />
                </span>

                <ChevronDown
                  className={cn(
                    "mt-0.5 h-4 w-4 shrink-0 text-zinc-600 transition-transform",
                    isOpen && "rotate-180",
                  )}
                />
              </button>

              {isOpen ? (
                <ClaimDetail claim={claim} evidence={evidence} />
              ) : null}
            </div>
          );
        })}
      </div>

      {skipped > 0 ? (
        <div className="border-t border-zinc-800 px-4 py-2 text-xs text-zinc-500">
          {skipped} claim{skipped === 1 ? "" : "s"} skipped as opinion or
          non-checkable — verifying those would produce a confident verdict about
          something evidence cannot settle.
        </div>
      ) : null}
    </div>
  );
}
