"use client";

import { useEffect, useRef } from "react";

import type { StreamEvent } from "@/lib/types";
import { NODE_LABELS, cn } from "@/lib/ui";

const NODE_COLOURS: Record<string, string> = {
  planner: "text-violet-400",
  researcher: "text-sky-400",
  synthesiser: "text-cyan-400",
  claim_extractor: "text-teal-400",
  verifier: "text-emerald-400",
  contradiction: "text-orange-400",
  reflection: "text-fuchsia-400",
  report: "text-lime-400",
  runner: "text-zinc-400",
  budget: "text-rose-400",
};

const VERDICT_COLOURS: Record<string, string> = {
  SUPPORTED: "text-emerald-400",
  REFUTED: "text-rose-400",
  NEI: "text-amber-400",
};

export function EventStream({
  events,
  running,
}: {
  events: StreamEvent[];
  running: boolean;
}) {
  const endRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pinnedRef = useRef(true);

  // Auto-scroll, but only while the user is already at the bottom — yanking
  // the viewport away from someone reading earlier output is hostile.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const distance =
      container.scrollHeight - container.scrollTop - container.clientHeight;
    pinnedRef.current = distance < 80;
    if (pinnedRef.current) {
      endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [events]);

  return (
    <div className="panel flex h-full flex-col">
      <div className="panel-header">
        <span className="panel-title">Live pipeline</span>
        <span className="flex items-center gap-2 text-xs text-zinc-500">
          {running ? (
            <>
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-75" />
                <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-500" />
              </span>
              running
            </>
          ) : (
            `${events.length} events`
          )}
        </span>
      </div>

      <div
        ref={containerRef}
        className="flex-1 space-y-1 overflow-y-auto p-3 font-mono text-xs"
      >
        {events.length === 0 ? (
          <p className="px-2 py-8 text-center text-zinc-600">
            Start a run to watch agents work.
          </p>
        ) : null}

        {events.map((event, index) => {
          const verdict = event.verdict as string | undefined;
          return (
            <div
              key={index}
              className="flex animate-fade-in gap-2 rounded px-2 py-1 hover:bg-zinc-800/40"
            >
              <span
                className={cn(
                  "w-24 shrink-0 truncate",
                  NODE_COLOURS[event.node] ?? "text-zinc-500",
                )}
              >
                {NODE_LABELS[event.node] ?? event.node}
              </span>
              <span
                className={cn(
                  "min-w-0 flex-1 break-words",
                  event.level === "warning" && "text-amber-400",
                  event.level === "error" && "text-rose-400",
                  event.level === "info" && "text-zinc-300",
                )}
              >
                {event.message}
              </span>
              {verdict ? (
                <span
                  className={cn(
                    "shrink-0 tabular-nums",
                    VERDICT_COLOURS[verdict],
                  )}
                >
                  {typeof event.confidence === "number"
                    ? (event.confidence as number).toFixed(2)
                    : null}
                </span>
              ) : null}
            </div>
          );
        })}
        <div ref={endRef} />
      </div>
    </div>
  );
}
