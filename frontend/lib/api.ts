/** Typed API client. */

import type {
  ConfidenceExplanation,
  EvidenceGraph,
  HealthResponse,
  RunState,
  StreamEvent,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(detail, response.status);
  }

  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<HealthResponse>("/health"),

  startRun: (topic: string, maxClaims?: number) =>
    request<{ run_id: string; stream_url: string }>("/api/runs", {
      method: "POST",
      body: JSON.stringify({ topic, max_claims: maxClaims }),
    }),

  getRun: (runId: string) => request<RunState>(`/api/runs/${runId}`),

  listRuns: () =>
    request<
      { run_id: string; topic: string; status: string; created_at: string }[]
    >("/api/runs"),

  getGraph: (runId: string) =>
    request<EvidenceGraph>(`/api/runs/${runId}/graph`),

  getReport: async (runId: string) => {
    const response = await fetch(`${API_BASE}/api/runs/${runId}/report`);
    if (!response.ok) throw new ApiError("report unavailable", response.status);
    return response.text();
  },

  explainConfidence: (params: Record<string, number>) => {
    const query = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    );
    return request<ConfidenceExplanation>(
      `/api/confidence/explain?${query.toString()}`,
    );
  },
};

/**
 * Subscribe to a run's SSE stream.
 *
 * The backend replays buffered history on connect, so a late subscriber still
 * sees the whole run. Returns an unsubscribe function.
 */
export function subscribeToRun(
  runId: string,
  onEvent: (event: StreamEvent) => void,
  onDone: () => void,
): () => void {
  const source = new EventSource(`${API_BASE}/api/runs/${runId}/stream`);
  let closed = false;

  const close = () => {
    if (!closed) {
      closed = true;
      source.close();
    }
  };

  const handle = (raw: MessageEvent) => {
    try {
      const event = JSON.parse(raw.data) as StreamEvent;
      onEvent(event);
      if ((event as { terminal?: boolean }).terminal) {
        close();
        onDone();
      }
    } catch {
      /* keep-alive comment frames are not JSON */
    }
  };

  // Named events are dispatched by node name, so listen on each plus the
  // default channel.
  for (const node of [
    "planner",
    "researcher",
    "synthesiser",
    "claim_extractor",
    "verifier",
    "contradiction",
    "reflection",
    "report",
    "runner",
    "budget",
    "message",
  ]) {
    source.addEventListener(node, handle as EventListener);
  }
  source.onmessage = handle;

  source.onerror = () => {
    // EventSource auto-reconnects; only give up once the run is finished.
    if (source.readyState === EventSource.CLOSED) {
      close();
      onDone();
    }
  };

  return close;
}
