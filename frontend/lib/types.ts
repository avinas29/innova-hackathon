/**
 * Domain types mirroring the backend's Pydantic models.
 *
 * Kept hand-written rather than generated: the API surface is small and stable,
 * and a codegen step is one more thing to break during a demo.
 */

export type Verdict = "SUPPORTED" | "REFUTED" | "NEI";
export type Stance = "SUPPORTS" | "REFUTES" | "NEUTRAL";
export type ClaimCategory = "NON_FACTUAL" | "FACTUAL_UNIMPORTANT" | "CHECK_WORTHY";
export type RunStatus =
  | "PENDING"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "BUDGET_EXCEEDED";

export interface ConfidenceFeatures {
  entail_max: number;
  agreement: number;
  independence: number;
  source_quality: number;
  consistency: number;
  sufficiency: number;
  stated_conf: number;
}

export interface Claim {
  id: string;
  text: string;
  decontextualised: string;
  category: ClaimCategory;
  checkworthy_score: number;
  verdict: Verdict;
  raw_confidence: number;
  confidence: number;
  features: ConfidenceFeatures;
  evidence_ids: string[];
  cluster_ids: string[];
  citations: string[];
  rationale: string;
  advocate_argument: string;
  sceptic_argument: string;
  minority_report: string;
  retracted: boolean;
  revision: string;
  error: string;
}

export interface Source {
  id: string;
  url: string;
  domain: string;
  title: string;
  credibility_tier: string;
  credibility_score: number;
  degraded: boolean;
}

export interface Evidence {
  id: string;
  claim_id: string;
  source_id: string;
  url: string;
  domain: string;
  snippet: string;
  stance: Stance;
  entailment_score: number;
  relevance: number;
  reasoning: string;
  cluster_id: string | null;
  is_derivative: boolean;
  credibility_score: number;
}

export interface Contradiction {
  id: string;
  evidence_a: string;
  evidence_b: string;
  claim_id: string;
  score: number;
  explanation: string;
  domain_a: string;
  domain_b: string;
}

export interface RunMetrics {
  total_claims: number;
  checkworthy_claims: number;
  supported: number;
  refuted: number;
  nei: number;
  retracted: number;
  mean_confidence: number;
  unique_sources: number;
  unique_domains: number;
  evidence_items: number;
  independent_clusters: number;
  contradictions: number;
  duration_seconds: number;
  tokens: { prompt_tokens: number; completion_tokens: number; calls: number };
}

export interface ResearchReport {
  run_id: string;
  topic: string;
  status: RunStatus;
  executive_summary: string;
  body_markdown: string;
  claims: Claim[];
  sources: Source[];
  evidence: Evidence[];
  contradictions: Contradiction[];
  metrics: RunMetrics;
  warnings: string[];
}

export interface RunState {
  run_id: string;
  topic: string;
  status: RunStatus;
  error: string;
  finished: boolean;
  event_count: number;
  report: ResearchReport | null;
}

export interface StreamEvent {
  ts: string;
  node: string;
  level: "debug" | "info" | "warning" | "error";
  message: string;
  [key: string]: unknown;
}

export interface GraphNode {
  id: string;
  type: "claim" | "source";
  label: string;
  verdict?: Verdict;
  confidence?: number;
  retracted?: boolean;
  domain?: string;
  url?: string;
  tier?: string;
  credibility?: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  type: "evidence" | "contradiction";
  stance?: Stance;
  weight: number;
  derivative?: boolean;
  cluster?: string | null;
  explanation?: string;
}

export interface EvidenceGraph {
  run_id: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  metrics: RunMetrics;
}

export interface ConfidenceExplanation {
  features: ConfidenceFeatures;
  raw_score: number;
  calibrated_score: number;
  independence_ceiling: number;
  final_score: number;
  capped: boolean;
  bias: number;
  contributions: {
    feature: string;
    value: number;
    weight: number;
    contribution: number;
  }[];
}

export interface HealthResponse {
  status: string;
  version: string;
  active_runs: number;
  config: {
    llm_provider: string;
    model_fast: string;
    model_strong: string;
    entailment_backend: string;
    search_providers: string[];
    openai_key_present: boolean;
    anthropic_key_present: boolean;
  };
}
