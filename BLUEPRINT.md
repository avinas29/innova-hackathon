# VERITAS — Autonomous Multi-Agent Research & Fact-Verification System

**Architecture blueprint · Phases 1–8**
Research conducted 2026-07-25. All GitHub metrics pulled live from the GitHub REST API on that date.

---

## 0. Executive summary — the thesis

Most hackathon submissions for this problem will build **"N agents that debate a topic."** The 2025–2026 literature says that design does not work, and I can prove it with citations. This blueprint deliberately rejects it.

The three decisions that actually determine whether we beat a single LLM:

1. **Verify atomic, decontextualised claims — not paragraphs.** The `decompose → check-worthiness → retrieve → verify` pipeline (FActScore / SAFE / VeriScore / Loki) is the only approach with published human-correlation evidence. A "verify this report" agent is unmeasurable.
2. **Give debating agents *different evidence*, not different personalities.** Naive multi-agent debate is provably a martingale — expected correctness does not improve across rounds when agents share inputs ([Choi et al.](https://arxiv.org/pdf/2606.29270)). Information *asymmetry* is what makes deliberation pay ([arXiv:2607.01661](https://arxiv.org/pdf/2607.01661)).
3. **Confidence must be calibrated against labelled data, not verbalised by an LLM.** Verbalised confidence carries median ECE ≈ 0.2, and models almost never abstain even when abstention is mathematically optimal ([survey](https://aclanthology.org/2024.naacl-long.366.pdf), [I-CALM](https://arxiv.org/pdf/2604.03904)). We fit an isotonic calibrator and report ECE/Brier/AUROC against a single-LLM baseline.

The deliverable that wins the room is not the agent graph. It is **a benchmark table showing our ECE and claim-level accuracy versus a single LLM on the same inputs.** Everything else is scaffolding for that number.

---

## Phase 1 — State of the art

### 1.1 The canonical fact-verification pipeline

Every serious system converges on the same five stages, first unified by **Loki** ([COLING 2025 demo](https://aclanthology.org/2025.coling-demos.4.pdf)):

```
decompose → check-worthiness → query generation → evidence retrieval → verdict
```

| Work | Contribution | What we take | What we reject |
|---|---|---|---|
| [FActScore](https://arxiv.org/abs/2305.14251) | "Decompose-then-verify"; atomic facts; % supported | The atomic-claim unit of account | Wikipedia-only knowledge source |
| [SAFE](https://arxiv.org/abs/2403.18802) (DeepMind) | Search-augmented per-claim verification, multi-step Google queries | Agentic per-claim query loop | GPT-4 per claim = cost blowup |
| [VeriScore](https://arxiv.org/pdf/2406.19276) | Extract & verify **only verifiable** claims | Check-worthiness gate — critical | — |
| [DnDScore](https://arxiv.org/pdf/2412.13175) | Decontextualisation matters as much as decomposition | Claims must carry their own context | — |
| [VeriFastScore](https://arxiv.org/html/2505.16973) | Single-pass joint decompose+verify | Batching insight for latency | Needs a fine-tune we don't have time for |
| [RARR](https://arxiv.org/abs/2210.08726) (Google) | Research→revise; agreement gate; edit unsupported text, preserve the rest | The **repair loop** and attribution report | — |
| [CRAG](https://openreview.net/pdf?id=JnWJbrnaUE) | Plug-and-play retrieval evaluator + query rewrite on bad retrieval | Retrieval-sufficiency gate | — |
| [Self-RAG](https://arxiv.org/abs/2310.11511) | Reflection tokens for when-to-retrieve | Concept only | Requires end-to-end training |
| [MiniCheck](https://aclanthology.org/2024.emnlp-main.499/) (EMNLP'24) | 770M model at GPT-4 grounding accuracy, **400× cheaper** | The entailment scorer | — |
| [AVeriTeC](https://aclanthology.org/2024.fever-1.1.pdf) / [HerO](https://arxiv.org/html/2410.12377) | Real-world claims, QA-decomposed evidence, open-weight systems (AVeriTeC score 0.57) | Question-generation style evidence | — |
| [Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) (Nature'24) | Meaning-level uncertainty detects confabulation | Self-consistency variance feature | Full sampling is too slow for every claim |
| [Anthropic multi-agent research](https://www.anthropic.com/engineering/multi-agent-research-system) | Orchestrator-worker; **+90.2%** over single-agent Opus 4; separate CitationAgent | Orchestrator-worker + citation pass | 15× token cost needs budgeting |

### 1.2 Architecture comparison

Verified against the GitHub API on 2026-07-25. "Hackathon suitability" is my judgement on a 1–5 scale.

| Architecture | Pros | Cons | Performance evidence | Stars | Last push | License | Ease | Hackathon fit | Innovation |
|---|---|---|---|---|---|---|---|---|---|
| **Decompose-verify pipeline** (FActScore/VeriScore/Loki) | Measurable; per-claim outputs map 1:1 to UI; strong human correlation | Slow (many LLM calls); no synthesis | Correlates with human factuality judgements | — | — | MIT (Loki) | 4 | **5** | 2 |
| **Orchestrator-worker research** (Anthropic, GPT-Researcher) | Genuine parallel speedup; scales past one context window | 15× tokens; no verification layer of its own | +90.2% vs single agent (internal eval) | 28.6k (GPT-R) | 2026-07-18 | Apache-2.0 | 4 | **5** | 3 |
| **Multi-agent debate** (Du et al.) | Intuitive; great demo narrative | **Martingale — no expected gain**; stance homogenisation; sycophancy | Weak models fix only 3.6% of stance bias | — | — | — | 3 | **2** | 2 |
| **Asymmetric-evidence deliberation** | Keeps debate's appeal, fixes its math | Needs disjoint retrieval partitions — more plumbing | Deliberation pays under information asymmetry | — | — | — | 3 | **4** | **5** |
| **NLI cross-encoder grounding** (MiniCheck/HHEM) | 400× cheaper than GPT-4 judging; deterministic; auditable | Needs the evidence handed to it; sentence-level | MiniCheck-FT5 770M ≈ GPT-4 on LLM-AggreFact | — | — | mixed¹ | 4 | **5** | 3 |
| **Knowledge-graph verification** (GraphCheck, ClaimPKG) | Great visuals; multi-hop | Entity linking is a project of its own | Competitive on HoVer | — | — | varies | 2 | 2 | 4 |
| **Self-consistency / semantic entropy** | No retrieval needed; catches confabulation | k× sampling cost; detects *uncertainty*, not *falsity* | Nature 2024 | — | — | — | 4 | 4 | 4 |
| **STORM outline-driven curation** | Excellent long-form structure | Research-grade codebase; **stale** | — | 30.3k | 2025-09-30 | MIT | 3 | 3 | 3 |

¹ MiniCheck licensing is split — see §3.2. This materially affects what we can ship.

### 1.3 The debate finding, stated precisely

This is the most important negative result in the research and the core of our differentiation:

- Under a Dirichlet-Categorical belief model, standard multi-agent debate **forms a martingale**: expected correctness does not improve over rounds when agents receive identical inputs ([Choi et al., 2026](https://arxiv.org/pdf/2606.29270)).
- Controlling for aggregation strategy, debate and independent answering show **no significant accuracy difference** — the bottleneck is extracting the right answer from disagreement, not generating disagreement.
- Weak models correct only **3.6%** of stance biases during debate; agents abandon correct judgements to conform ([The Deliberative Illusion](https://arxiv.org/pdf/2606.03032) documents "factual attrition and stance homogenisation").

**Design consequence.** We keep adversarial structure but change two variables: (a) each verifier sees a **disjoint evidence partition**, so disagreement carries information rather than personality; (b) aggregation is a **calibrated statistical model**, not a vote or a consensus round. Minority reports are preserved and surfaced, never averaged away.

---

## Phase 2 — Community research

**Honest limitation:** `reddit.com` blocks my crawler's user agent, so I could not read Reddit threads directly. Rather than invent quotes, I substituted Hacker News, GitHub issue trackers, practitioner blogs, and the repos' own commit/issue history. Anything below that I could not verify is marked *(unverified)*.

Recurring practitioner themes:

| Theme | Evidence | Design response |
|---|---|---|
| **Framework abstraction tax** — role/task metaphors fight you once control flow gets real | AutoGen's 972 open issues; Microsoft's pivot to Agent Framework; CrewAI's 667 open issues | Pick an explicit state machine (LangGraph); keep agents as plain functions |
| **Cost shock** — multi-agent burns tokens far faster than expected | Anthropic: ~15× a chat interaction; token usage alone explains **80%** of performance variance | Hard token budget per run; cheap model for extraction, strong model for adjudication |
| **Agents pick bad sources** | Anthropic: "agents consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources" | Explicit domain-credibility prior, applied *before* the LLM sees ranked evidence |
| **Parallel-write data loss in graph frameworks** | LangGraph docs: without a reducer "the last write wins — which loses data" | Every fan-out key uses an explicit `operator.add` reducer; enforced in tests |
| **Deep research repos go stale fast** | STORM 2025-09-30, Loki 2024-10-03 | Borrow *architecture*, vendor no runtime dependency on them |
| **Search API lock-in / cost** | Tavily PAYG ≈ $96/mo for 10k+10k vs ≈$11 alternative; Tavily acquired by Nebius 2026-02-10 | Provider-agnostic search interface, 4 backends, keyless fallback |

---

## Phase 3 — GitHub deep search

### 3.1 Ranked repositories

All figures from the GitHub API, 2026-07-25.

| # | Repository | Stars | Last push | License | Strengths | Weaknesses | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | [langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research) | 12,423 | **2026-07-25** | MIT | Canonical LangGraph supervisor+workers; actively maintained today | No verification/confidence layer at all | **Study the graph topology, write our own** |
| 2 | [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) | 28,629 | 2026-07-18 | Apache-2.0 | #1 on DeepResearchGym; planner→executor→publisher; battle-tested | Report-centric; citations are not per-claim verified | **Borrow retrieval patterns; don't fork** |
| 3 | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) | 38,105 | **2026-07-25** | MIT | Send API, checkpointing, streaming | Verbose; steep concepts | **Core dependency** |
| 4 | [Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck) | — | 2024-era | mixed¹ | GPT-4-level grounding at 770M | Model licensing split | **Optional backend, not default** |
| 5 | [Libr-AI/OpenFactVerification](https://github.com/Libr-AI/OpenFactVerification) (Loki) | 1,152 | **2024-10-03** | MIT | The cleanest 5-stage pipeline reference | ~21 months stale; 1 open issue = abandoned | **Read the code, reimplement** |
| 6 | [stanford-oval/storm](https://github.com/stanford-oval/storm) | 30,311 | 2025-09-30 | MIT | Outline-driven curation, multi-perspective | ~10 months stale; research-grade | **Take the outline idea only** |
| 7 | [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) | 56,113 | 2026-07-25 | MIT | Biggest community; fastest prototype | Role metaphor obstructs per-claim map-reduce | **Reject** |
| 8 | [microsoft/autogen](https://github.com/microsoft/autogen) | 59,961 | **2026-04-15** | CC-BY-4.0 | Strong conversational patterns | 3 months stale; MS pivoted; **CC-BY-4.0 is not a software licence** | **Reject** |
| 9 | [openai/openai-agents-python](https://github.com/openai/openai-agents-python) | 28,159 | 2026-07-25 | MIT | Minimal, clean, 100+ models via LiteLLM | Handoff model ≠ deterministic DAG | **Reject as orchestrator; use OpenAI SDK directly** |

¹ MiniCheck's small variants (`lytang/MiniCheck-Flan-T5-Large`, RoBERTa/DeBERTa) are usable; **`bespokelabs/Bespoke-MiniCheck-7B` is non-commercial only**. We therefore ship an LLM-based entailment scorer as the default and expose the cross-encoder as an opt-in backend.

### 3.2 Fork, reuse, or build?

**Build from scratch, borrow architecture.** Reasoning:

- No existing repo does research **and** per-claim verification **and** calibrated confidence. GPT-Researcher stops at citations; Loki starts at claims and is abandoned. Bolting them together means owning the seam anyway.
- Forking a 28k-star repo makes our contribution unreadable to a judge diffing the code.
- The differentiator (calibrated confidence + asymmetric evidence) touches every layer; it cannot be a plugin.

**AutoGen's CC-BY-4.0 licence is a genuine trap** — Creative Commons explicitly recommends against using CC licences for software. A judge who checks would be right to flag it.

---

## Phase 4 — Technology selection

| Framework | Fit for *dynamic per-claim fan-out* | Durable state | Streaming | Verdict |
|---|---|---|---|---|
| **LangGraph** | `Send` API = native dynamic map-reduce | Checkpointer, time-travel | Token + state | ✅ **Chosen** |
| CrewAI | Crews are statically declared | Partial | Basic | ❌ |
| AutoGen | Conversational, not DAG | Weak | Basic | ❌ stale + licence |
| OpenAI Agents SDK | Handoffs, not fan-out | External | Good | ❌ (SDK used for models) |
| Haystack | Strong pipelines, weak agent loops | Yes | Limited | ❌ |
| LlamaIndex Workflows | Event-driven, good | Yes | Good | 🥈 runner-up |
| Semantic Kernel | Enterprise .NET-first | Yes | Good | ❌ |

**Why LangGraph, concretely.** Our workload is *N claims × M evidence items, verified independently, then reduced*. That is textbook map-reduce, and `Send` expresses it in one line where CrewAI needs a dynamically-constructed crew per run. We also get: a checkpointer (a 4-minute research run that survives a crash mid-demo), state streaming (the live dashboard), and explicit reducers (no silent parallel-write loss). The cost is verbosity — accepted.

**Where we deliberately don't use it:** claim extraction, entailment scoring, and calibration are plain typed Python. They are pure functions; putting them in a graph would only make them harder to unit-test.

### Selected stack

| Layer | Choice | Why not the alternative |
|---|---|---|
| Orchestration | LangGraph | above |
| API | FastAPI + SSE | SSE beats WebSockets for one-way run streaming |
| Models | OpenAI + Anthropic behind one interface | Provider outage during a demo is a real risk |
| Search | Tavily / Exa / Brave / DuckDuckGo | Keyless DDG fallback means it runs with zero search keys |
| Extraction | trafilatura | Best-in-class boilerplate removal, pure Python |
| Vectors | NumPy + SQLite | FAISS wheels are unreliable on Python 3.13; our corpus is ~10³ vectors — cosine over NumPy is *faster* here |
| DB | SQLite (WAL) → Postgres | Zero-setup for a hackathon; same SQL |
| Observability | OpenTelemetry → Langfuse (opt-in) | Langfuse is MIT + self-hostable; LangSmith self-host is enterprise-only |
| Frontend | Next.js + Tailwind + shadcn | As specified |

---

## Phase 5 — System design

### 5.1 Pipeline

```
                        ┌──────────────┐
   topic ──────────────▶│   PLANNER    │  decompose → research questions + budget
                        └──────┬───────┘
                               │  Send(fan-out per question)
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
   ┌─────────┐           ┌─────────┐            ┌─────────┐
   │  WEB    │           │ ACADEMIC│            │  CODE   │   parallel researchers
   │ Tavily/ │           │ arXiv/  │            │ GitHub  │   (Anthropic orchestrator-worker)
   │ Exa/DDG │           │ S2      │            │         │
   └────┬────┘           └────┬────┘            └────┬────┘
        └──────────────────────┼──────────────────────┘
                               ▼  reducer: operator.add
                    ┌─────────────────────┐
                    │  EVIDENCE COLLECTOR │  fetch → extract → embed → persist
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  SOURCE RANKER      │  credibility prior × relevance
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  DRAFT SYNTHESISER  │  the report under test
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │  CLAIM EXTRACTOR    │  atomic + DECONTEXTUALISED
                    └──────────┬──────────┘
                               ▼
                    ┌─────────────────────┐
                    │ CHECK-WORTHINESS    │  non-factual / unimportant / check-worthy
                    └──────────┬──────────┘
                               │  Send(fan-out per check-worthy claim)
        ┌──────────────────────┼──────────────────────┐
        ▼                                             ▼
  ┌──────────────────────────────────────────────────────┐
  │  PER-CLAIM VERIFICATION SUBGRAPH                     │
  │   query-gen (RARR) → retrieve → INDEPENDENCE CLUSTER │
  │        ↓                                             │
  │   partition clusters into A|B  (information asymmetry)│
  │        ↓                    ↓                        │
  │   ADVOCATE(A)          SCEPTIC(B)   ← disjoint evidence│
  │        └────────┬───────────┘                        │
  │            ADJUDICATOR  → SUPPORTED/REFUTED/NEI      │
  │                 ↓                                    │
  │            CONFIDENCE (7 features → isotonic)        │
  └──────────────────────────┬───────────────────────────┘
                             ▼  reducer: operator.add
                  ┌─────────────────────┐
                  │ CONTRADICTION GRAPH │  pairwise NLI across ALL evidence
                  └──────────┬──────────┘
                             ▼
                  ┌─────────────────────┐
                  │  REFLECTION (RARR)  │──┐ unsupported claims → revise/retract
                  └──────────┬──────────┘  │ bounded ≤2 loops
                             │◀────────────┘
                             ▼
                  ┌─────────────────────┐
                  │  REPORT GENERATOR   │  citations + trust heatmap + minority reports
                  └─────────────────────┘
```

### 5.2 The confidence model — the core contribution

Seven orthogonal features per claim, deliberately **excluding** any single dominant LLM opinion:

| Feature | Signal | Rationale |
|---|---|---|
| `entail_max` | max entailment over evidence clusters | strongest single support |
| `agreement` | (support − refute) / total, **cluster-weighted** | net evidence direction |
| `independence` | log(1 + # independent clusters) | 5 syndicated copies ≠ 5 sources |
| `source_quality` | credibility-weighted mean of supporting domains | counters the content-farm bias |
| `consistency` | 1 − variance over k=3 sampled adjudications | semantic-entropy-lite |
| `sufficiency` | retrieval quality gate (CRAG) | separates "false" from "unknown" |
| `stated_conf` | verbalised LLM confidence | included, **capped at low weight** |

These feed a **logistic model + isotonic calibration** fitted on a labelled dev set. We ship the fitted calibrator and report **ECE, Brier, AUROC, and selective-risk curves** versus a single-LLM baseline on identical claims.

**Abstention is a first-class verdict.** `NEI` / insufficient-evidence is emitted rather than guessed — directly addressing the finding that models "almost never abstain, resulting in utility collapse."

### 5.3 Evidence independence — the anti-echo-chamber layer

Multiple outlets reprinting one wire story is *one* piece of evidence. We collapse it:

1. SimHash over shingles → near-duplicate detection
2. Embedding cosine ≥ τ → semantic clustering
3. Explicit citation/quote detection → derivative marked as child
4. Each cluster contributes **one** weighted vote, weight = best source in cluster

Without this, confidence is systematically overstated. This is the single most defensible technical claim in the system.

### 5.4 Cross-cutting concerns

| Concern | Design |
|---|---|
| Memory | Run-scoped SQLite + vector store; planner persists plan (Anthropic's context-truncation lesson) |
| Caching | Content-addressed SQLite cache on LLM calls, search, and fetches — makes demos instant and reruns free |
| Retry | Tenacity, exponential backoff + jitter, per-provider circuit breaker |
| Failure recovery | LangGraph checkpointer; a dead subagent degrades that claim to NEI rather than failing the run |
| Budget | Hard token + wall-clock ceiling per run, enforced in state |
| Streaming | SSE of graph state deltas → live UI |
| Observability | Structured JSON logs + OTel spans; optional Langfuse |
| Testing | pytest; fake LLM/search providers so the full graph runs offline in CI |

---

## Phase 6 — Innovation

Ranked by (judge impact ÷ build cost):

| # | Feature | Why it lands | Cost |
|---|---|---|---|
| 1 | **Calibration dashboard vs single-LLM baseline** | The only feature that *proves* the problem statement's requirement | M |
| 2 | **Evidence-independence clustering** | Defensible, novel-feeling, visibly changes confidence numbers live | M |
| 3 | **Asymmetric-evidence adversarial verification** | Debate that survives the martingale critique — a judge who knows the literature will notice | M |
| 4 | **Trust heatmap over the report** | Instant visual "wow"; hover a sentence → evidence + score | S |
| 5 | **Contradiction graph** | Shows sources disagreeing with each other, not just with the claim | M |
| 6 | **Minority-report preservation** | Directly counters stance-homogenisation | S |
| 7 | **Hallucination autopsy** | Per-retracted-claim: what was said, what evidence killed it, why | S |
| 8 | **Live verification stream** | Watching claims flip green/red in real time demos the whole thesis | S |
| 9 | Provenance replay (checkpoint time-travel) | Reproducibility story | S |
| 10 | Auto follow-up questions from NEI claims | Shows the system knows what it doesn't know | S |

---

## Phase 7 — Hackathon strategy

**What judges reward:** a working demo, one number that proves the claim, and a technical decision you can defend under questioning. The debate-is-a-martingale finding is that defensible decision.

| Budget | Scope | Demo |
|---|---|---|
| **6 h** | Linear graph: plan → search → draft → extract → verify → report. Heuristic confidence. SQLite. CLI + JSON. | Terminal run with a claim table |
| **12 h** | + parallel researchers, independence clustering, contradiction detection, FastAPI + SSE | Live streaming claim verdicts |
| **24 h** | + asymmetric adversarial verification, isotonic calibration, **eval harness + baseline comparison**, Next.js dashboard with trust heatmap | The ECE table — the winning slide |
| **48 h** | + evidence graph viz, RARR repair loop, Docker/CI, hallucination autopsy, provenance replay | Full product |

**Cut first if time-pressed:** knowledge-graph entity linking, Postgres, multi-tenancy, auth, GitHub/Reddit source agents.
**Never cut:** the eval harness. Without it, "outperforms a single LLM" is an unsupported claim — exactly the failure mode the project exists to solve.

---

## Phase 8 — Implementation plan

### Folder structure

```
jhaat/
├── BLUEPRINT.md
├── README.md
├── docker-compose.yml
├── Makefile
├── .env.example
├── .github/workflows/ci.yml
├── backend/
│   ├── pyproject.toml
│   ├── Dockerfile
│   ├── veritas/
│   │   ├── config.py            # pydantic-settings
│   │   ├── schemas.py           # domain models
│   │   ├── state.py             # LangGraph state + reducers
│   │   ├── llm/{client,cache}.py
│   │   ├── tools/{search,fetch,academic}.py
│   │   ├── evidence/{credibility,dedup,store}.py
│   │   ├── verify/{claims,checkworthy,entailment,contradiction,confidence,calibration}.py
│   │   ├── graph/{build,nodes,prompts}.py
│   │   ├── storage/db.py
│   │   ├── api/{app,routes}.py
│   │   ├── eval/{metrics,baseline,run}.py
│   │   └── cli.py
│   └── tests/
└── frontend/                    # Next.js 15 + Tailwind + shadcn
```

### Database schema

```sql
runs(id, topic, status, created_at, finished_at, config_json, token_usage, error)
claims(id, run_id, text, decontextualised, checkworthy, category, verdict,
       confidence, calibrated_confidence, features_json, rationale)
sources(id, run_id, url, domain, title, credibility_tier, credibility_score, fetched_at)
evidence(id, run_id, claim_id, source_id, snippet, stance, entailment_score,
         cluster_id, is_derivative)
contradictions(id, run_id, evidence_a, evidence_b, score, explanation)
events(id, run_id, ts, node, level, payload_json)     -- powers SSE replay
cache(key, value, created_at)                          -- content-addressed
```

### API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/runs` | Start a research run |
| `GET` | `/api/runs/{id}` | Full run result |
| `GET` | `/api/runs/{id}/stream` | SSE live events |
| `GET` | `/api/runs/{id}/report` | Markdown report |
| `GET` | `/api/runs/{id}/graph` | Evidence + contradiction graph JSON |
| `POST` | `/api/verify` | Verify a single claim (fast path) |
| `POST` | `/api/eval` | Run the benchmark vs baseline |
| `GET` | `/health` | Liveness + provider status |

### Execution flow (per run)

1. `POST /api/runs` → row inserted, graph invoked in a background task
2. Planner emits questions + budget → checkpoint
3. `Send` fan-out to researchers → evidence rows, embeddings
4. Ranker scores sources → draft synthesised
5. Claims extracted, decontextualised, check-worthiness scored
6. `Send` fan-out per check-worthy claim → verification subgraph
7. Contradiction pass over the full evidence set
8. Reflection: unsupported claims revised or retracted (≤2 loops)
9. Report generated; every event streamed over SSE throughout

### Error handling

| Failure | Response |
|---|---|
| Search provider down | Fall through provider chain → DuckDuckGo |
| Fetch 403/timeout | Use search snippet, mark evidence `degraded` |
| LLM rate limit | Backoff + jitter, then fall back to secondary provider |
| Malformed structured output | Re-ask once with the parse error, then heuristic parse |
| Claim verification crash | Claim → `NEI` with `error` rationale; run continues |
| Budget exhausted | Graceful stop, partial report marked incomplete |

---

## Decision log — what I rejected and why

| Rejected | Why |
|---|---|
| Naive multi-agent debate | Martingale; stance homogenisation; 3.6% bias correction |
| CrewAI | Role metaphor can't express per-claim dynamic fan-out |
| AutoGen | 3 months stale + CC-BY-4.0 is not a software licence |
| Forking GPT-Researcher | Buries our contribution; no verification layer to build on |
| Bespoke-MiniCheck-7B as default | Non-commercial licence |
| FAISS | Unreliable py3.13 wheels; NumPy is faster at our scale |
| LLM verbalised confidence as *the* score | ECE ≈ 0.2; kept only as one capped feature |
| Knowledge-graph entity linking | Entity linking is its own project; poor time-to-value |
