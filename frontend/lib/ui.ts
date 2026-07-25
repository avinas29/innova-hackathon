import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

import type { Verdict } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

/**
 * Trust heatmap colour for a confidence score.
 *
 * The scale is deliberately conservative: nothing reads as "green / trusted"
 * below 0.75, because the independence ceiling means a single-source claim
 * tops out at 0.70. A UI that painted 0.70 green would undo the guard the
 * scoring model works hardest to enforce.
 */
export function confidenceColour(confidence: number): string {
  if (confidence >= 0.85) return "bg-emerald-500";
  if (confidence >= 0.75) return "bg-lime-500";
  if (confidence >= 0.6) return "bg-amber-500";
  if (confidence >= 0.4) return "bg-orange-500";
  return "bg-rose-500";
}

export function confidenceTextColour(confidence: number): string {
  if (confidence >= 0.85) return "text-emerald-400";
  if (confidence >= 0.75) return "text-lime-400";
  if (confidence >= 0.6) return "text-amber-400";
  if (confidence >= 0.4) return "text-orange-400";
  return "text-rose-400";
}

export const VERDICT_STYLES: Record<
  Verdict,
  { label: string; className: string; dot: string }
> = {
  SUPPORTED: {
    label: "Supported",
    className: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
    dot: "bg-emerald-400",
  },
  REFUTED: {
    label: "Refuted",
    className: "bg-rose-500/10 text-rose-400 border-rose-500/30",
    dot: "bg-rose-400",
  },
  NEI: {
    label: "Not established",
    className: "bg-amber-500/10 text-amber-400 border-amber-500/30",
    dot: "bg-amber-400",
  },
};

export const TIER_STYLES: Record<string, string> = {
  PRIMARY: "text-emerald-400",
  HIGH: "text-lime-400",
  MEDIUM: "text-sky-400",
  LOW: "text-amber-400",
  UNKNOWN: "text-zinc-500",
  UNRELIABLE: "text-rose-400",
};

export const NODE_LABELS: Record<string, string> = {
  planner: "Planner",
  researcher: "Researcher",
  synthesiser: "Synthesiser",
  claim_extractor: "Claim extractor",
  verifier: "Verifier",
  contradiction: "Contradiction",
  reflection: "Reflection",
  report: "Report",
  runner: "Runner",
  budget: "Budget",
};

export function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

export function formatNumber(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}
