"use client";

import { AlertTriangle, Loader2, Search } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { ClaimTable } from "@/components/ClaimTable";
import { ConfidenceExplainer } from "@/components/ConfidenceExplainer";
import { EventStream } from "@/components/EventStream";
import { EvidenceGraph } from "@/components/EvidenceGraph";
import { MetricsBar } from "@/components/MetricsBar";
import { api, subscribeToRun } from "@/lib/api";
import type {
  EvidenceGraph as GraphData,
  HealthResponse,
  RunState,
  StreamEvent,
} from "@/lib/types";
import { cn } from "@/lib/ui";

type Tab = "live" | "claims" | "graph" | "report" | "model";

const TABS: { id: Tab; label: string }[] = [
  { id: "live", label: "Live" },
  { id: "claims", label: "Claims" },
  { id: "graph", label: "Evidence graph" },
  { id: "report", label: "Report" },
  { id: "model", label: "Confidence model" },
];

const EXAMPLES = [
  "The current state of solid-state battery commercialisation",
  "Does intermittent fasting improve metabolic health?",
  "How much has global solar capacity actually grown since 2020?",
];

export default function Home() {
  const [topic, setTopic] = useState("");
  const [run, setRun] = useState<RunState | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [searchOk, setSearchOk] = useState<{
    working: string[];
    any_working: boolean;
  } | null>(null);
  const [tab, setTab] = useState<Tab>("live");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const unsubscribeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
    // Configured != working. A wrong SEARXNG_URL, or DuckDuckGo blocking a
    // datacenter IP, both look healthy in config and return nothing at run
    // time — which surfaces only as every claim coming back "not established".
    api.searchHealth().then(setSearchOk).catch(() => setSearchOk(null));
    return () => unsubscribeRef.current?.();
  }, []);

  /**
   * Fetch the finished run, retrying on transient network failure.
   *
   * The run has already succeeded by this point — minutes of work and real API
   * quota are behind it. Discarding that because one fetch blipped is the worst
   * possible failure. A free-tier instance is slow enough that the first
   * request after a long stream can legitimately fail.
   */
  const loadFinishedRun = useCallback(async (runId: string) => {
    for (let attempt = 0; attempt < 4; attempt++) {
      try {
        const state = await api.getRun(runId);
        setRun(state);
        setError(null);
        if (state.report) {
          setGraph(await api.getGraph(runId).catch(() => null));
          setTab((current) => (current === "live" ? "claims" : current));
        }
        return;
      } catch (exc) {
        const last = attempt === 3;
        if (last) {
          setError(
            exc instanceof Error
              ? `Could not load the finished run: ${exc.message}. The run completed — reload the page to retrieve it.`
              : "could not load run",
          );
          return;
        }
        setError(`Loading results… (retry ${attempt + 1} of 3)`);
        await new Promise((r) => setTimeout(r, 1500 * (attempt + 1)));
      }
    }
  }, []);

  const start = useCallback(
    async (value: string) => {
      const trimmed = value.trim();
      if (!trimmed || starting) return;

      unsubscribeRef.current?.();
      setStarting(true);
      setError(null);
      setEvents([]);
      setGraph(null);
      setRun(null);
      setTab("live");

      try {
        const { run_id } = await api.startRun(trimmed);
        setRun({
          run_id,
          topic: trimmed,
          status: "RUNNING",
          error: "",
          finished: false,
          event_count: 0,
          report: null,
        });

        unsubscribeRef.current = subscribeToRun(
          run_id,
          (event) => setEvents((previous) => [...previous, event]),
          () => void loadFinishedRun(run_id),
        );
      } catch (exc) {
        setError(
          exc instanceof Error
            ? `${exc.message} — is the API running on ${process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"}?`
            : "could not start run",
        );
      } finally {
        setStarting(false);
      }
    },
    [starting, loadFinishedRun],
  );

  const report = run?.report ?? null;
  const running = Boolean(run && !run.finished);
  const offline = health?.config.llm_provider === "fake";

  return (
    <main className="mx-auto min-h-screen max-w-[1600px] px-6 py-8">
      <header className="mb-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-3xl font-semibold tracking-tight">
              VERITAS
              <span className="ml-3 text-base font-normal text-zinc-500">
                multi-agent research &amp; fact verification
              </span>
            </h1>
            <p className="mt-1.5 max-w-3xl text-sm text-zinc-500">
              Every claim is decomposed, checked against{" "}
              <span className="text-zinc-400">independent</span> sources,
              adversarially reviewed, and given a calibrated confidence score.
              &ldquo;Not established&rdquo; is a real answer.
            </p>
          </div>

          {health ? (
            <div className="flex items-center gap-2 text-xs text-zinc-500">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
              {health.config.llm_provider} · {health.config.entailment_backend}{" "}
              entailment ·{" "}
              {searchOk === null ? (
                <span title="checking search providers…">
                  {health.config.search_providers.join(", ") || "no search"}
                </span>
              ) : searchOk.any_working ? (
                <span
                  className="text-emerald-400"
                  title={`Live probe: ${searchOk.working.join(", ")} returning results`}
                >
                  search: {searchOk.working.join(", ")} ✓
                </span>
              ) : (
                <span
                  className="text-rose-400"
                  title="No provider returned results — every claim will come back NEI"
                >
                  search: not working
                </span>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-2 text-xs text-rose-400">
              <span className="h-1.5 w-1.5 rounded-full bg-rose-500" />
              API unreachable
            </div>
          )}
        </div>

        {offline ? (
          <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/5 px-4 py-2.5 text-xs text-amber-400">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              No model API key is configured, so the backend is running its
              deterministic offline provider. Output is heuristic, not model
              output. Set <code className="font-mono">OPENAI_API_KEY</code> or{" "}
              <code className="font-mono">ANTHROPIC_API_KEY</code> for real
              results.
            </span>
          </div>
        ) : null}
      </header>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          void start(topic);
        }}
        className="mb-6"
      >
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-zinc-600" />
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="What should we research and verify?"
              disabled={running || starting}
              className="w-full rounded-lg border border-zinc-800 bg-zinc-900/60 py-3 pl-10 pr-4 text-sm
                placeholder:text-zinc-600 focus:border-cyan-600/60 focus:outline-none focus:ring-1
                focus:ring-cyan-600/40 disabled:opacity-50"
            />
          </div>
          <button
            type="submit"
            disabled={running || starting || !topic.trim()}
            className="flex items-center gap-2 rounded-lg bg-cyan-600 px-5 py-3 text-sm font-medium
              text-white transition-colors hover:bg-cyan-500 disabled:cursor-not-allowed
              disabled:bg-zinc-800 disabled:text-zinc-500"
          >
            {running || starting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Researching
              </>
            ) : (
              "Research"
            )}
          </button>
        </div>

        {!run ? (
          <div className="mt-2.5 flex flex-wrap gap-2">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                onClick={() => {
                  setTopic(example);
                  void start(example);
                }}
                className="rounded-full border border-zinc-800 px-3 py-1 text-xs text-zinc-500
                  transition-colors hover:border-zinc-700 hover:text-zinc-300"
              >
                {example}
              </button>
            ))}
          </div>
        ) : null}

        {error ? (
          <p className="mt-2.5 text-xs text-rose-400">{error}</p>
        ) : null}
      </form>

      {report ? (
        <div className="mb-6">
          <MetricsBar metrics={report.metrics} />
        </div>
      ) : null}

      {run ? (
        <>
          <nav className="mb-4 flex gap-1 border-b border-zinc-800">
            {TABS.map((item) => (
              <button
                key={item.id}
                onClick={() => setTab(item.id)}
                className={cn(
                  "-mb-px border-b-2 px-4 py-2 text-sm transition-colors",
                  tab === item.id
                    ? "border-cyan-500 text-zinc-100"
                    : "border-transparent text-zinc-500 hover:text-zinc-300",
                )}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <section className="min-h-[560px]">
            {tab === "live" ? (
              <div className="h-[620px]">
                <EventStream events={events} running={running} />
              </div>
            ) : null}

            {tab === "claims" ? (
              report ? (
                <div className="h-[620px]">
                  <ClaimTable
                    claims={report.claims}
                    evidence={report.evidence}
                  />
                </div>
              ) : (
                <Pending running={running} />
              )
            ) : null}

            {tab === "graph" ? (
              graph ? (
                <EvidenceGraph graph={graph} />
              ) : (
                <Pending running={running} />
              )
            ) : null}

            {tab === "report" ? (
              report ? (
                <ReportView markdown={report.body_markdown} warnings={report.warnings} />
              ) : (
                <Pending running={running} />
              )
            ) : null}

            {tab === "model" ? <ConfidenceExplainer /> : null}
          </section>
        </>
      ) : (
        <ConfidenceExplainer />
      )}
    </main>
  );
}

function Pending({ running }: { running: boolean }) {
  return (
    <div className="panel flex h-[400px] items-center justify-center gap-3 text-sm text-zinc-600">
      {running ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Available once the run completes.
        </>
      ) : (
        "No data for this run."
      )}
    </div>
  );
}

/**
 * Minimal markdown rendering.
 *
 * A full markdown pipeline is a dependency and an XSS surface; the report is
 * generated by our own backend and uses a handful of constructs, so a small
 * line-based renderer covers it without either.
 */
function ReportView({
  markdown,
  warnings,
}: {
  markdown: string;
  warnings: string[];
}) {
  if (!markdown.trim()) {
    return (
      <div className="panel p-10 text-center text-sm text-zinc-600">
        No report body was generated.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {warnings.length > 0 ? (
        <div className="panel border-amber-500/25 p-4">
          <div className="mb-2 flex items-center gap-2 text-xs font-medium text-amber-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            Run warnings
          </div>
          <ul className="space-y-1 text-xs text-zinc-400">
            {warnings.slice(0, 8).map((warning, index) => (
              <li key={index}>· {warning}</li>
            ))}
          </ul>
        </div>
      ) : null}

      <article className="panel max-h-[640px] overflow-y-auto p-6">
        {markdown.split("\n").map((line, index) => {
          if (line.startsWith("### ")) {
            return (
              <h3 key={index} className="mb-2 mt-5 text-base font-semibold text-zinc-200">
                {line.slice(4)}
              </h3>
            );
          }
          if (line.startsWith("## ")) {
            return (
              <h2 key={index} className="mb-3 mt-6 text-lg font-semibold text-zinc-100">
                {line.slice(3)}
              </h2>
            );
          }
          if (line.startsWith("# ")) {
            return (
              <h1 key={index} className="mb-4 text-2xl font-semibold text-zinc-50">
                {line.slice(2)}
              </h1>
            );
          }
          if (/^[-*]\s+/.test(line)) {
            return (
              <li key={index} className="ml-5 list-disc py-0.5 text-sm text-zinc-300">
                {line.replace(/^[-*]\s+/, "")}
              </li>
            );
          }
          if (!line.trim()) return <div key={index} className="h-2" />;
          return (
            <p key={index} className="py-1 text-sm leading-relaxed text-zinc-300">
              {line}
            </p>
          );
        })}
      </article>
    </div>
  );
}
