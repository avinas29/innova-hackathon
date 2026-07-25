# VERITAS

**Autonomous multi-agent research and fact-verification with calibrated per-claim confidence.**

Every claim in a generated report is decomposed into an atomic, self-contained
assertion, checked against **independent** sources, adversarially reviewed by two
agents holding *different* evidence, and assigned a confidence score that comes
from a fitted statistical model rather than a language model's opinion of itself.

"Not established" is a first-class verdict.

---

## The three decisions that matter

Most systems for this problem are "N agents that debate a topic." That design
does not work, and the 2025–2026 literature says so explicitly. This one is built
around three findings:

**1. Naive multi-agent debate is a martingale.** Under a Dirichlet-Categorical
belief model, expected correctness does not improve across debate rounds when
agents receive identical inputs ([Choi et al.](https://arxiv.org/pdf/2606.29270)).
Weak models correct only 3.6% of stance biases; agents abandon correct judgements
to conform ([The Deliberative Illusion](https://arxiv.org/pdf/2606.03032)).

→ We keep adversarial structure but give each reviewer a **disjoint evidence
partition**. Debate between agents with the same context is theatre. Debate
between agents with different evidence is information aggregation. Minority
positions are preserved and surfaced, never averaged away.

**2. Repetition is not corroboration.** Ten outlets reprinting one wire story
look like ten confirmations. Any confidence score computed over raw evidence
counts is systematically inflated — and most inflated exactly where a claim is
most viral.

→ Evidence is collapsed into **independence clusters** by SimHash, embedding
cosine, same-domain, and verbatim-span detection. Each cluster casts one weighted
vote. The UI shows the collapse: *"9/12 — 75% independent after dedup."*

**3. Verbalised LLM confidence is miscalibrated.** Median ECE around 0.2, and
models almost never abstain even when abstention is mathematically optimal.

→ Confidence is a **measurement**: seven orthogonal features → logistic model →
isotonic calibration fitted on labelled data. The model's own stated confidence
is one capped feature among seven. On top of that sits a hard **independence
ceiling** — one source, however emphatic, cannot license more than 0.70.

Full research, comparison tables and the rejected-alternatives log: **[BLUEPRINT.md](BLUEPRINT.md)**.

---

## Quick start

```bash
make install
```

Runs with **zero API keys** using a deterministic offline provider — that is how
the test suite and CI exercise the whole graph. For real research, add one key:

```bash
cp .env.example .env
```

Then set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, **or** `GEMINI_API_KEY`. Search
falls back to keyless DuckDuckGo, so one model key is enough to start.

### Running free on Gemini

A [free Gemini key](https://aistudio.google.com/apikey) needs no credit card and
works end to end. Set two values in `.env`:

```
GEMINI_API_KEY=your-key
VERITAS_PROFILE=free
```

Gemini is reached through Google's OpenAI-compatible endpoint, so it reuses the
OpenAI code path rather than a second SDK.

**The quota is the real constraint, not the integration.** Free tier allows
10–15 requests per minute, and VERITAS is deliberately parallel — a default run
fires 40–60 calls in seconds and would collect a wall of 429s. Three things
handle that automatically:

| Mechanism | Effect |
|---|---|
| **Per-model pacing** | Sliding-window limiter per model. Quotas are per model, so Flash-Lite's 1000/day pool is not throttled to Flash's 250 — worth roughly **4× usable throughput** |
| **`VERITAS_PROFILE=free`** | Fewer questions, fewer claims, less evidence per claim, no self-consistency resampling. A partial report that finishes beats a thorough one that 429s |
| **Response caching** | Cache hits consume neither quota nor pacing delay, so reruns and demos are free and instant |

Check the budget before spending it:

```bash
make doctor
```

It prints the per-model limits and estimates calls per run and runs per day.

```bash
make doctor                                    # check config and connectivity
make research TOPIC="how much has solar capacity grown since 2020"
make dev                                       # API :8000 + UI :3000
```

Or with Docker:

```bash
docker compose up --build
```

---

## Proving the claim

The problem statement asks the system to outperform a single LLM. That is an
empirical claim, so the benchmark harness is a first-class part of the product,
not a side script:

```bash
make eval        # pipeline vs single-LLM control on identical claims
make calibrate   # fit the confidence model on those results
```

`veritas eval` reports accuracy, macro-F1, **ECE**, Brier, AUROC, abstention
rate, and a selective-risk curve for both systems side by side. The baseline uses
the *strong* model with a clean prompt — beating a strawman would prove nothing.

Where the pipeline should win is not raw accuracy on easy facts (a frontier model
already knows those) but **calibration** and **NEI claims**, where a single model
answers confidently instead of abstaining.

> The shipped 36-claim development set is a smoke test for the harness, **not a
> benchmark**. For a defensible headline number, point `--dataset` at FEVER or
> AVeriTeC via the JSONL loader. The harness prints the dataset name and size
> beside every metric so results cannot be quietly misrepresented.

---

## Architecture

```
PLAN ──Send(×N)──▶ RESEARCHERS (web · academic · reference, parallel)
                        │
                        ▼
                    SYNTHESISE ──▶ EXTRACT CLAIMS ──▶ CHECK-WORTHINESS GATE
                                                              │
                        ┌─────────────────Send(×M)────────────┘
                        ▼
   ╭─ PER-CLAIM VERIFICATION ─────────────────────────────────╮
   │  query-gen (incl. adversarial) → retrieve                │
   │       → INDEPENDENCE CLUSTERING                           │
   │       → partition A | B  ← disjoint evidence              │
   │  ADVOCATE(A)      SCEPTIC(B)                              │
   │            ▼                                              │
   │      ADJUDICATOR → SUPPORTED / REFUTED / NEI              │
   │            ▼                                              │
   │      CONFIDENCE: 7 features → isotonic → ceiling          │
   ╰──────────────────────────┬───────────────────────────────╯
                              ▼
        CONTRADICTION GRAPH ──▶ REFLECTION (≤2) ──▶ REPORT
```

**Orchestration: LangGraph.** The workload is *N claims × M evidence, verified
independently, then reduced* — textbook map-reduce, which `Send` expresses in one
line. Plus durable checkpointing, state streaming, and explicit reducers. Rejected:
CrewAI (role metaphor cannot express dynamic per-claim fan-out), AutoGen (stale;
CC-BY-4.0 is not a software licence), OpenAI Agents SDK (handoffs, not fan-out).

### Layout

```
backend/veritas/
├── config.py            settings; nothing else reads os.environ
├── schemas.py           domain model, framework-free
├── state.py             graph state + reducers + service container
├── prompts.py           every prompt, in one auditable module
├── llm/client.py        OpenAI · Anthropic · deterministic offline
├── tools/               search chain · content extraction · arXiv/S2/Wikipedia
├── evidence/            credibility priors · independence clustering · vectors
├── verify/              claims · check-worthiness · entailment · contradiction
│                        · confidence · calibration
├── graph/               nodes + assembly
├── eval/                metrics · single-LLM baseline · driver · datasets
├── api/                 FastAPI + SSE
└── cli.py               research · verify · eval · calibrate · serve · doctor
```

---

## Design notes worth knowing

**Check-worthiness gating.** Claims are classified non-factual /
factual-unimportant / check-worthy, and only the third class is verified. Running
retrieval-and-entailment over an opinion produces a confident-looking verdict
about something evidence cannot settle.

**Decontextualisation is not optional.** "It grew 40% that year" gets rewritten to
carry its own context before verification. A claim that silently inherits context
is verified against the wrong proposition — worse than not checking it.

**Entailment measures grounding, not truth.** The scorer judges only against
supplied evidence. A model answering from its own knowledge here destroys the
measurement, because that is precisely the single-LLM behaviour we claim to beat.

**Retrieval sufficiency separates "false" from "unknown."** Low sufficiency with no
supporting evidence means *we don't know*. High sufficiency with none means *we
looked properly and it isn't there*. Conflating those is a serious error.

**Adversarial queries are mandatory.** Every claim gets a query that hunts for
refutation. Retrieving only confirming evidence guarantees a falsely confident
verdict.

**Failure is local.** A crashed verification branch yields an NEI claim with the
error attached. Losing one claim is recoverable; losing the run is not.

---

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` | — | at least one for real output |
| `VERITAS_LLM_PROVIDER` | `auto` | `openai` · `anthropic` · `gemini` · `fake`; `auto` takes the first key present |
| `VERITAS_PROFILE` | `default` | `free` shrinks the workload to fit a free-tier quota |
| `VERITAS_RPM_LIMIT` / `VERITAS_DAILY_LIMIT` | `0` | `0` auto-applies Gemini's published free-tier limits |
| `VERITAS_SEARCH_ORDER` | `tavily,exa,brave,duckduckgo` | unconfigured providers skipped |
| `VERITAS_ENTAILMENT_BACKEND` | `llm` | `local` uses a MiniCheck-family cross-encoder¹ |
| `VERITAS_MAX_TOKENS_PER_RUN` | `400000` | hard ceiling, enforced in state |
| `VERITAS_DEDUP_THRESHOLD` | `0.86` | cosine above which evidence is one cluster |

Full list in [.env.example](.env.example).

¹ `pip install 'veritas[local-nli]'`. Default is the LLM backend because
`Bespoke-MiniCheck-7B` is non-commercial-only and the small variants pull ~2GB of
torch wheels — a submission should not ship an unresolved licence question.

---

## Testing

```bash
make check    # ruff + mypy + pytest (116 tests, fully offline)
```

The suite is hermetic: no keys, no network, no flakes. Notable guards:

- **every fan-out state key has an `operator.add` reducer** — without one,
  LangGraph's last-write-wins default silently discards parallel branches and the
  run still "succeeds"
- **SSE frames are dicts, not pre-formatted strings** — `EventSourceResponse`
  formats frames itself; double-wrapping produces a stream no client can parse and
  no error anywhere
- **stated confidence alone cannot exceed 0.25** — the ensemble must not be
  talked into certainty by a model's own say-so
- **the independence ceiling is enforced end to end**, API included

---

## API

| Method | Path | |
|---|---|---|
| `POST` | `/api/runs` | start a run (202, streams over SSE) |
| `GET` | `/api/runs/{id}/stream` | live events; replays history on connect |
| `GET` | `/api/runs/{id}/graph` | evidence + contradiction graph |
| `POST` | `/api/verify` | verify one claim, same subgraph |
| `GET` | `/api/confidence/explain` | interactive score breakdown |
| `POST` | `/api/eval` | benchmark vs the baseline |
| `GET` | `/health` | liveness **and the active provider** |

`/health` exposes the resolved provider so a demo can never pass offline
heuristic output off as model output. The UI shows a banner when it does.

Interactive docs at `/docs`.

---

## Licence

MIT.
