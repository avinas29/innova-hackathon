/**
 * VERITAS — InnovaHack technical deck generator.
 *
 * Dark premium AI aesthetic: near-black canvas, violet primary, cyan accent.
 * Visual motif carried across every slide: rounded translucent cards with a
 * small filled circle badge. No accent stripes or underlines anywhere.
 */
const pptx = require("pptxgenjs");

const pres = new pptx();
pres.layout = "LAYOUT_WIDE"; // 13.3 x 7.5 — must be set before adding slides
pres.author = "Team Peakbuster";
pres.title = "VERITAS — Autonomous Multi-Agent Research & Fact Verification";

// ── Palette ────────────────────────────────────────────────────────────────
const BG = "0A0A12";
const CARD = "16162B";
const CARD2 = "1C1C34";
const VIOLET = "8B5CF6";
const CYAN = "22D3EE";
const GREEN = "34D399";
const RED = "FB7185";
const AMBER = "FBBF24";
const TEXT = "F4F4F5";
const MUTED = "9CA3AF";
const DIM = "6B7280";

const H = "Calibri";
const B = "Calibri";

const W = 13.3;
const shadow = () => ({ type: "outer", color: "000000", blur: 18, offset: 3, angle: 90, opacity: 0.45 });

function slide() {
  const s = pres.addSlide();
  s.background = { color: BG };
  return s;
}

/** Section title + optional kicker. Consistent position on every content slide. */
function heading(s, title, kicker) {
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.6, y: 0.34, w: 12.1, h: 0.28, fontFace: B, fontSize: 11,
      color: CYAN, bold: true, charSpacing: 2, margin: 0,
    });
  }
  s.addText(title, {
    x: 0.6, y: kicker ? 0.62 : 0.45, w: 12.1, h: 0.62,
    fontFace: H, fontSize: 30, bold: true, color: TEXT, margin: 0,
  });
}

/** Rounded card — the repeated motif. */
function card(s, x, y, w, h, fill) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, rectRadius: 0.12,
    fill: { color: fill || CARD },
    line: { color: "2A2A45", width: 1 },
    shadow: shadow(),
  });
}

/** Small filled circle badge with a glyph — used beside every section header. */
function badge(s, x, y, color, glyph) {
  s.addShape(pres.ShapeType.ellipse, {
    x, y, w: 0.34, h: 0.34, fill: { color }, line: { color, width: 0 },
  });
  s.addText(glyph, {
    x, y, w: 0.34, h: 0.34, fontFace: B, fontSize: 13, bold: true,
    color: "0A0A12", align: "center", valign: "middle", margin: 0,
  });
}

function stat(s, x, y, w, value, label, color) {
  s.addText(value, {
    x, y, w, h: 0.72, fontFace: H, fontSize: 40, bold: true,
    color: color || CYAN, align: "center", margin: 0,
  });
  s.addText(label, {
    x, y: y + 0.7, w, h: 0.5, fontFace: B, fontSize: 11,
    color: MUTED, align: "center", margin: 0,
  });
}

/** Arrow connector between pipeline nodes. */
function arrow(s, x, y, w) {
  s.addShape(pres.ShapeType.rightArrow, {
    x, y, w, h: 0.16, fill: { color: "3F3F5E" }, line: { color: "3F3F5E", width: 0 },
  });
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 1 — Vision & Problem
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();

  // Ambient glow, purely decorative
  s.addShape(pres.ShapeType.ellipse, {
    x: 8.6, y: -1.5, w: 6.4, h: 6.4, fill: { color: VIOLET, transparency: 88 }, line: { width: 0 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 9.9, y: 3.0, w: 4.4, h: 4.4, fill: { color: CYAN, transparency: 92 }, line: { width: 0 },
  });

  s.addText("TEAM PEAKBUSTER   ·   INNOVAHACK 2026   ·   DOMAIN 3: GEN AI", {
    x: 0.7, y: 0.55, w: 9.0, h: 0.3, fontFace: B, fontSize: 11,
    color: CYAN, bold: true, charSpacing: 2, margin: 0,
  });

  s.addText("VERITAS", {
    x: 0.7, y: 1.05, w: 8.6, h: 1.35, fontFace: H, fontSize: 76,
    bold: true, color: TEXT, charSpacing: 1, margin: 0,
  });

  s.addText("AI that knows what it doesn't know.", {
    x: 0.7, y: 2.35, w: 8.6, h: 0.55, fontFace: H, fontSize: 25,
    color: VIOLET, italic: true, margin: 0,
  });

  s.addText(
    "Every claim decomposed, verified against independent sources, and scored with calibrated confidence.",
    { x: 0.7, y: 2.98, w: 8.4, h: 0.6, fontFace: B, fontSize: 14, color: MUTED, margin: 0 },
  );

  // The problem, as three hard facts
  card(s, 0.7, 3.85, 3.9, 1.62);
  badge(s, 0.95, 4.08, RED, "!");
  s.addText("Confidently wrong", {
    x: 1.4, y: 4.06, w: 3.0, h: 0.32, fontFace: B, fontSize: 14, bold: true, color: TEXT, margin: 0,
  });
  s.addText("LLMs state falsehoods with the same certainty as facts.", {
    x: 0.95, y: 4.52, w: 3.4, h: 0.8, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0,
  });

  card(s, 4.8, 3.85, 3.9, 1.62);
  badge(s, 5.05, 4.08, AMBER, "%");
  s.addText("Miscalibrated", {
    x: 5.5, y: 4.06, w: 3.0, h: 0.32, fontFace: B, fontSize: 14, bold: true, color: TEXT, margin: 0,
  });
  s.addText("Verbalised confidence carries ~0.2 expected calibration error.", {
    x: 5.05, y: 4.52, w: 3.4, h: 0.8, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0,
  });

  card(s, 8.9, 3.85, 3.7, 1.62);
  badge(s, 9.15, 4.08, VIOLET, "0");
  s.addText("Never abstains", {
    x: 9.6, y: 4.06, w: 2.8, h: 0.32, fontFace: B, fontSize: 14, bold: true, color: TEXT, margin: 0,
  });
  s.addText("Models almost never say \"I don't know\" — even when optimal.", {
    x: 9.15, y: 4.52, w: 3.2, h: 0.8, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0,
  });

  s.addText("Avinash   ·   Atulaya Raj   ·   Mohammad Umam Ali        IIT Guwahati", {
    x: 0.7, y: 5.75, w: 12.0, h: 0.3, fontFace: B, fontSize: 12, color: DIM, margin: 0,
  });

  s.addText(
    "In medicine, law and journalism a fluent hallucination is worse than no answer — because it is trusted.",
    { x: 0.7, y: 6.25, w: 12.0, h: 0.4, fontFace: B, fontSize: 13, color: VIOLET, italic: true, margin: 0 },
  );

  s.addNotes(
    "VERITAS. Team Peakbuster — Avinash, Atulaya Raj, Mohammad Umam Ali, IIT Guwahati.\n\n" +
    "The problem is not that AI is sometimes wrong. It is that AI is wrong with the exact same confidence it is right. " +
    "There is no signal to the user.\n\n" +
    "Three documented facts frame this. One: LLMs state falsehoods fluently. Two: when you ask a model how confident it " +
    "is, that number carries roughly 0.2 expected calibration error — it is close to meaningless. Three: models almost " +
    "never abstain, even when abstention is provably the optimal action.\n\n" +
    "Our thesis: a research system's most valuable output is not the answer. It is an honest, calibrated statement of " +
    "how much you should trust that answer."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 2 — Solution & Innovation
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  heading(s, "We rejected the obvious architecture — with evidence", "Solution & Innovation");

  // The negative result that drives the design
  card(s, 0.6, 1.5, 6.0, 2.25, CARD2);
  badge(s, 0.9, 1.78, RED, "X");
  s.addText("What everyone else builds: multi-agent debate", {
    x: 1.35, y: 1.76, w: 5.0, h: 0.32, fontFace: B, fontSize: 14, bold: true, color: RED, margin: 0,
  });
  s.addText(
    [
      { text: "Debate is a martingale — no expected gain per round", options: { bullet: true, breakLine: true } },
      { text: "Weak models correct only 3.6% of stance biases", options: { bullet: true, breakLine: true } },
      { text: "Agents conform; correct minorities get averaged away", options: { bullet: true } },
    ],
    { x: 0.95, y: 2.25, w: 5.4, h: 1.15, fontFace: B, fontSize: 12, color: MUTED, paraSpaceAfter: 6, margin: 0 },
  );
  s.addText("Choi et al. 2026 · The Deliberative Illusion", {
    x: 0.95, y: 3.36, w: 5.4, h: 0.28, fontFace: B, fontSize: 10, color: DIM, italic: true, margin: 0,
  });

  card(s, 6.9, 1.5, 5.8, 2.25, CARD2);
  badge(s, 7.2, 1.78, GREEN, "✓");
  s.addText("What we built: asymmetric evidence", {
    x: 7.65, y: 1.76, w: 4.8, h: 0.32, fontFace: B, fontSize: 14, bold: true, color: GREEN, margin: 0,
  });
  s.addText(
    [
      { text: "Advocate and Sceptic get DISJOINT evidence sets", options: { bullet: true, breakLine: true } },
      { text: "Disagreement now carries information, not personality", options: { bullet: true, breakLine: true } },
      { text: "Aggregation is statistical, not a consensus vote", options: { bullet: true } },
    ],
    { x: 7.25, y: 2.25, w: 5.2, h: 1.15, fontFace: B, fontSize: 12, color: MUTED, paraSpaceAfter: 6, margin: 0 },
  );
  s.addText("Deliberation pays only under information asymmetry", {
    x: 7.25, y: 3.36, w: 5.2, h: 0.28, fontFace: B, fontSize: 10, color: DIM, italic: true, margin: 0,
  });

  // Four innovations
  const innov = [
    [VIOLET, "Independence clustering", "10 outlets copying 1 wire story = 1 source"],
    [CYAN, "Independence ceiling", "One source can never license >0.70 confidence"],
    [GREEN, "Calibrated confidence", "7 orthogonal features, isotonic-fitted"],
    [AMBER, "Abstention as a verdict", "\"Not established\" is a first-class answer"],
  ];
  innov.forEach(([c, t, d], i) => {
    const x = 0.6 + i * 3.09;
    card(s, x, 4.0, 2.89, 1.5);
    badge(s, x + 0.22, 4.2, c, String(i + 1));
    s.addText(t, {
      x: x + 0.22, y: 4.62, w: 2.5, h: 0.3, fontFace: B, fontSize: 12.5, bold: true, color: TEXT, margin: 0,
    });
    s.addText(d, {
      x: x + 0.22, y: 4.94, w: 2.5, h: 0.5, fontFace: B, fontSize: 10.5, color: MUTED, margin: 0,
    });
  });

  s.addText("Versus ChatGPT · Perplexity · Gemini Deep Research", {
    x: 0.6, y: 5.75, w: 6.0, h: 0.3, fontFace: B, fontSize: 12, bold: true, color: CYAN, margin: 0,
  });
  s.addText(
    "They cite sources. None of them measure source independence, none abstain, " +
    "and none publish a calibrated confidence you can audit.",
    { x: 0.6, y: 6.08, w: 12.1, h: 0.55, fontFace: B, fontSize: 12.5, color: MUTED, margin: 0 },
  );

  s.addNotes(
    "This is the slide that separates us from every other team in this room.\n\n" +
    "The obvious build is 'several agents that debate'. We researched it and found it does not work. Under a " +
    "Dirichlet-Categorical model, debate forms a martingale — expected correctness does not improve across rounds when " +
    "agents share inputs. Empirically, weak models correct only 3.6% of stance biases, and agents abandon correct " +
    "positions to conform.\n\n" +
    "So we changed one variable. Our Advocate and Sceptic receive DISJOINT halves of the evidence. Now disagreement " +
    "means they genuinely saw different things — that is real information aggregation rather than personality clash.\n\n" +
    "The four innovations underneath are what a fact-checker would actually ask for. The most defensible is " +
    "independence clustering: ten outlets reprinting one wire story is ONE piece of evidence. Counting it as ten " +
    "inflates confidence exactly where a claim is most viral — the worst possible place to be overconfident."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 3 — Architecture
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  heading(s, "13 specialised agents on a LangGraph state machine", "Multi-Agent Architecture");

  const layer = (x, y, w, h, label, color) => {
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w, h, rectRadius: 0.1,
      fill: { color: CARD }, line: { color, width: 1.25 }, shadow: shadow(),
    });
    s.addText(label, {
      x, y: y + 0.06, w, h: 0.3, fontFace: B, fontSize: 10.5, bold: true,
      color, align: "center", margin: 0,
    });
  };
  const node = (x, y, w, label, color) => {
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w, h: 0.42, rectRadius: 0.07,
      fill: { color: CARD2 }, line: { color: "34345A", width: 0.75 },
    });
    s.addText(label, {
      x, y, w, h: 0.42, fontFace: B, fontSize: 9.5, color: color || TEXT,
      align: "center", valign: "middle", margin: 0,
    });
  };

  // Row 1 — orchestration + research fan-out
  layer(0.6, 1.45, 2.6, 1.5, "ORCHESTRATION", VIOLET);
  node(0.78, 1.85, 2.24, "Planner Agent", VIOLET);
  node(0.78, 2.36, 2.24, "Budget + Retry Guard");

  layer(3.45, 1.45, 4.3, 1.5, "PARALLEL RESEARCH  (Send fan-out)", CYAN);
  node(3.62, 1.85, 1.28, "Web Search", CYAN);
  node(5.02, 1.85, 1.28, "Academic", CYAN);
  node(6.42, 1.85, 1.16, "Code/Repo", CYAN);
  node(3.62, 2.36, 3.96, "Evidence Collector  →  Source Ranking Agent");

  layer(8.0, 1.45, 4.7, 1.5, "MODELS  ·  4 PROVIDERS", GREEN);
  node(8.18, 1.85, 2.1, "Groq  llama-3.3-70b", GREEN);
  node(10.42, 1.85, 2.1, "Gemini 3.5 Flash", GREEN);
  node(8.18, 2.36, 4.34, "OpenAI  ·  Anthropic  ·  offline deterministic fallback");

  // Row 2 — verification fan-out
  layer(0.6, 3.15, 7.15, 1.5, "PER-CLAIM VERIFICATION  (Send fan-out, N branches)", AMBER);
  node(0.78, 3.55, 1.55, "Claim Extractor", AMBER);
  node(2.45, 3.55, 1.55, "Check-worthy", AMBER);
  node(4.12, 3.55, 1.55, "Entailment", AMBER);
  node(5.79, 3.55, 1.78, "Contradiction", AMBER);
  node(0.78, 4.06, 2.4, "Advocate  |  Sceptic");
  node(3.3, 4.06, 2.0, "Adjudicator");
  node(5.42, 4.06, 2.15, "Confidence Model");

  layer(8.0, 3.15, 4.7, 1.5, "MEMORY  ·  STORAGE", VIOLET);
  node(8.18, 3.55, 2.1, "NumPy Vector Store", VIOLET);
  node(10.42, 3.55, 2.1, "SQLite  (WAL)", VIOLET);
  node(8.18, 4.06, 4.34, "Content-addressed cache  ·  Checkpointer  ·  Event log");

  // Row 3 — output
  layer(0.6, 4.85, 12.1, 1.15, "OUTPUT", CYAN);
  node(0.78, 5.25, 2.3, "Reflection / RARR", CYAN);
  node(3.2, 5.25, 2.3, "Citation Engine", CYAN);
  node(5.62, 5.25, 2.3, "Report Generator", CYAN);
  node(8.04, 5.25, 2.2, "FastAPI  +  SSE", CYAN);
  node(10.36, 5.25, 2.16, "Next.js Dashboard", CYAN);

  s.addText(
    "Dynamic map-reduce: every fan-out key uses an explicit operator.add reducer — LangGraph's default is last-write-wins, " +
    "which silently discards parallel results.",
    { x: 0.6, y: 6.25, w: 12.1, h: 0.5, fontFace: B, fontSize: 11.5, color: MUTED, italic: true, margin: 0 },
  );

  s.addNotes(
    "Thirteen agents, orchestrated by LangGraph as an explicit state machine rather than a chat free-for-all.\n\n" +
    "Two dynamic fan-outs use LangGraph's Send API. The first spawns a researcher per question; the second spawns a " +
    "verification branch per check-worthy claim. That is genuine map-reduce over a workload whose width is not known " +
    "until runtime.\n\n" +
    "Why LangGraph over CrewAI or AutoGen: our workload is N claims by M evidence items verified independently then " +
    "reduced. Send expresses that in one line. CrewAI's role metaphor would need a dynamically constructed crew per run. " +
    "AutoGen we rejected outright — three months stale, and CC-BY-4.0 is not a software licence.\n\n" +
    "The footnote matters more than it looks. LangGraph's default channel behaviour is last-write-wins, so a parallel " +
    "branch writing to an unreduced key silently loses data — the run completes, just with a fraction of the evidence. " +
    "We have a test that fails the build if any fan-out key lacks a reducer."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 4 — Workflow
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  heading(s, "From question to calibrated verdict", "End-to-End Pipeline");

  const steps = [
    ["1", "Plan", "Decompose into research questions", VIOLET],
    ["2", "Research", "Parallel agents, 8 source types", CYAN],
    ["3", "Draft", "Synthesise the report under test", CYAN],
    ["4", "Decompose", "Atomic, decontextualised claims", AMBER],
  ];
  const steps2 = [
    ["5", "Cluster", "Collapse correlated evidence", AMBER],
    ["6", "Adjudicate", "Advocate vs Sceptic, disjoint sets", GREEN],
    ["7", "Calibrate", "7 features → isotonic → score", GREEN],
    ["8", "Report", "Citations + trust heatmap", VIOLET],
  ];

  const drawRow = (arr, y) => {
    arr.forEach(([n, t, d, c], i) => {
      const x = 0.6 + i * 3.16;
      card(s, x, y, 2.82, 1.42);
      badge(s, x + 0.2, y + 0.18, c, n);
      s.addText(t, {
        x: x + 0.64, y: y + 0.18, w: 2.0, h: 0.32, fontFace: B, fontSize: 14,
        bold: true, color: c, margin: 0, valign: "middle",
      });
      s.addText(d, {
        x: x + 0.2, y: y + 0.62, w: 2.45, h: 0.65, fontFace: B, fontSize: 10.5,
        color: MUTED, margin: 0,
      });
      if (i < 3) arrow(s, x + 2.88, y + 0.63, 0.22);
    });
  };
  drawRow(steps, 1.5);
  drawRow(steps2, 3.35);

  // The claim-level decision that makes it different
  card(s, 0.6, 5.15, 12.1, 1.5, CARD2);
  s.addText("The step nobody else has: evidence independence", {
    x: 0.9, y: 5.34, w: 5.4, h: 0.3, fontFace: B, fontSize: 13.5, bold: true, color: CYAN, margin: 0,
  });
  s.addText(
    "SimHash near-duplicates  ·  embedding cosine  ·  same-domain  ·  verbatim-span quotation  →  union-find",
    { x: 0.9, y: 5.7, w: 7.6, h: 0.4, fontFace: B, fontSize: 11.5, color: MUTED, margin: 0 },
  );
  s.addText("Measured on a live run:", {
    x: 0.9, y: 6.12, w: 2.4, h: 0.3, fontFace: B, fontSize: 11, color: DIM, margin: 0,
  });
  s.addText("20 evidence items  →  12 independent clusters", {
    x: 2.85, y: 6.12, w: 5.0, h: 0.3, fontFace: B, fontSize: 11.5, bold: true, color: GREEN, margin: 0,
  });

  stat(s, 8.9, 5.35, 1.7, "0.70", "max conf. 1 source", AMBER);
  stat(s, 10.7, 5.35, 1.7, "0.93", "max conf. 3+ sources", GREEN);

  s.addNotes(
    "Eight stages. The first four build a report; the last four try to tear it down.\n\n" +
    "Stage 4 is the highest-leverage step in the whole system: claims must be atomic AND decontextualised. " +
    "'It grew 40% that year' is unverifiable in isolation and, worse, will be silently verified against the wrong " +
    "proposition. We rewrite it to stand alone before anything touches it.\n\n" +
    "Stage 5 is the one nobody else has. Four independent signals feed a union-find: SimHash for verbatim copies, " +
    "embedding cosine for paraphrase, same-domain, and longest-common-token-run for quotation. On the run shown, " +
    "twenty evidence items collapsed to twelve genuinely independent clusters.\n\n" +
    "That directly changes the published number. One independent source caps confidence at 0.70 no matter how emphatic " +
    "it is — because a single source can simply be wrong and we would have no way to know. Three or more lifts the " +
    "ceiling to 0.93."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 5 — Technical Excellence
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  heading(s, "Production engineering, not a notebook demo", "Technical Implementation");

  const stackCard = (x, y, w, title, color, items) => {
    card(s, x, y, w, 2.05);
    badge(s, x + 0.2, y + 0.18, color, "•");
    s.addText(title, {
      x: x + 0.64, y: y + 0.18, w: w - 0.8, h: 0.32, fontFace: B, fontSize: 13,
      bold: true, color, margin: 0, valign: "middle",
    });
    s.addText(
      items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1 } })),
      { x: x + 0.24, y: y + 0.62, w: w - 0.48, h: 1.3, fontFace: B, fontSize: 10.5, color: MUTED, paraSpaceAfter: 4, margin: 0 },
    );
  };

  stackCard(0.6, 1.5, 3.9, "Backend", VIOLET, [
    "Python 3.13 · FastAPI · LangGraph 1.2",
    "Pydantic v2 typed end-to-end",
    "SQLite WAL · NumPy vector store",
    "SSE streaming · checkpointed runs",
  ]);
  stackCard(4.7, 1.5, 3.9, "Frontend", CYAN, [
    "Next.js 15 · React 19 · TypeScript",
    "Tailwind · static export",
    "Live SSE event stream",
    "Force-directed evidence graph",
  ]);
  stackCard(8.8, 1.5, 3.9, "Infra", GREEN, [
    "Single Docker image, UI + API one port",
    "Render deployed · CI on 3 Python versions",
    "SearXNG self-hosted metasearch",
    "Ruff · mypy · GitHub Actions",
  ]);

  // Hard engineering wins — the details judges probe for
  card(s, 0.6, 3.8, 12.1, 1.85, CARD2);
  s.addText("Engineering depth", {
    x: 0.9, y: 3.98, w: 4.0, h: 0.3, fontFace: B, fontSize: 13, bold: true, color: CYAN, margin: 0,
  });

  const deep = [
    ["Token-per-minute limiting", "Providers cap tokens, not requests"],
    ["Per-model quota tracking", "4× usable free-tier throughput"],
    ["Prompt auto-fitting", "Oversized calls are a permanent 413"],
    ["Server-stated retry delay", "Obeys 'retry in 49s', not a guess"],
    ["Hermetic offline test mode", "230 tests, zero network, zero keys"],
    ["Graceful degradation", "A dead branch → NEI, never a dead run"],
  ];
  deep.forEach(([t, d], i) => {
    const x = 0.9 + (i % 3) * 4.0;
    const y = 4.38 + Math.floor(i / 3) * 0.62;
    s.addText(t, { x, y, w: 3.8, h: 0.26, fontFace: B, fontSize: 11.5, bold: true, color: TEXT, margin: 0 });
    s.addText(d, { x, y: y + 0.24, w: 3.8, h: 0.26, fontFace: B, fontSize: 10, color: DIM, margin: 0 });
  });

  stat(s, 0.6, 5.85, 2.3, "230", "tests passing", GREEN);
  stat(s, 3.0, 5.85, 2.3, "4", "LLM providers", CYAN);
  stat(s, 5.4, 5.85, 2.3, "8", "source types", VIOLET);
  stat(s, 7.8, 5.85, 2.3, "0", "lint errors", GREEN);
  stat(s, 10.2, 5.85, 2.5, "100%", "typed public API", CYAN);

  s.addNotes(
    "Three layers, all deployed and running right now.\n\n" +
    "The engineering-depth box is where I would probe if I were judging, so let me pre-empt it. Every item there is a " +
    "bug we hit and fixed with evidence.\n\n" +
    "Token-per-minute limiting: Groq caps TOKENS, not requests. A request-only limiter sails straight past that. " +
    "Per-model tracking: quotas are per model, so pooling them throttled our high-volume model to the small model's " +
    "budget — fixing that gave roughly four times the usable throughput. Prompt auto-fitting: a single call larger than " +
    "the per-minute cap is a permanent 413 that no retry can fix, so we measure and trim before sending. Retry delay: " +
    "the provider tells you 'retry in 49 seconds'; our old exponential backoff capped at 20 and was guaranteed to fail.\n\n" +
    "And the whole 230-test suite runs with zero network and zero API keys — hermeticity is an enforced property, not " +
    "an accident."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 6 — Demo & Results
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  heading(s, "Built, deployed, and measured on live claims", "Demo & Functionality");

  // Live verdicts
  card(s, 0.6, 1.5, 6.0, 2.5);
  s.addText("Live verification — real output", {
    x: 0.9, y: 1.68, w: 5.4, h: 0.3, fontFace: B, fontSize: 13, bold: true, color: CYAN, margin: 0,
  });

  const verdicts = [
    ["SUPPORTED", GREEN, "0.93", "\"The Eiffel Tower is in Paris, France\""],
    ["REFUTED", RED, "0.85", "\"The Eiffel Tower is in Rome, Italy\""],
    ["NOT ESTABLISHED", AMBER, "0.30", "Thin evidence → the system abstains"],
  ];
  verdicts.forEach(([v, c, conf, claim], i) => {
    const y = 2.1 + i * 0.6;
    s.addShape(pres.ShapeType.roundRect, {
      x: 0.9, y, w: 1.72, h: 0.36, rectRadius: 0.06,
      fill: { color: c, transparency: 82 }, line: { color: c, width: 0.75 },
    });
    s.addText(v, {
      x: 0.9, y, w: 1.72, h: 0.36, fontFace: B, fontSize: 8.5, bold: true,
      color: c, align: "center", valign: "middle", margin: 0,
    });
    s.addText(conf, {
      x: 2.72, y, w: 0.55, h: 0.36, fontFace: B, fontSize: 13, bold: true,
      color: c, valign: "middle", margin: 0,
    });
    s.addText(claim, {
      x: 3.34, y, w: 3.1, h: 0.36, fontFace: B, fontSize: 10, color: MUTED,
      valign: "middle", margin: 0,
    });
  });

  // The moment that proves it isn't keyword matching
  card(s, 6.9, 1.5, 5.8, 2.5, CARD2);
  badge(s, 7.2, 1.7, VIOLET, "★");
  s.addText("It distinguishes the real thing from replicas", {
    x: 7.65, y: 1.68, w: 4.8, h: 0.34, fontFace: B, fontSize: 12.5, bold: true, color: VIOLET, margin: 0,
  });
  s.addText(
    "Verifying the Eiffel Tower, retrieval also surfaced Paris, Texas's Eiffel Tower and a Las Vegas half-scale replica.",
    { x: 7.25, y: 2.12, w: 5.2, h: 0.62, fontFace: B, fontSize: 11, color: MUTED, margin: 0 },
  );
  s.addText(
    "Both were scored as weak refutations — 0.10 — not confused for the subject. All three collapsed to ONE independent cluster (same domain).",
    { x: 7.25, y: 2.78, w: 5.2, h: 0.75, fontFace: B, fontSize: 11, color: TEXT, margin: 0 },
  );
  s.addText("Entailment scoring, not keyword overlap.", {
    x: 7.25, y: 3.56, w: 5.2, h: 0.3, fontFace: B, fontSize: 10.5, color: CYAN, italic: true, margin: 0,
  });

  // Shipped features
  card(s, 0.6, 4.2, 12.1, 1.35);
  const feats = [
    "Live SSE pipeline stream", "Per-claim trust heatmap", "Interactive evidence graph",
    "Confidence explainer", "Citation-backed report", "Minority reports preserved",
  ];
  feats.forEach((f, i) => {
    const x = 0.85 + (i % 3) * 4.0;
    const y = 4.42 + Math.floor(i / 3) * 0.48;
    s.addShape(pres.ShapeType.ellipse, { x, y: y + 0.06, w: 0.16, h: 0.16, fill: { color: GREEN }, line: { width: 0 } });
    s.addText(f, { x: x + 0.28, y, w: 3.6, h: 0.3, fontFace: B, fontSize: 11.5, color: TEXT, valign: "middle", margin: 0 });
  });

  stat(s, 0.6, 5.7, 2.4, "248s", "full research run", CYAN);
  stat(s, 3.1, 5.7, 2.4, "24", "sources / 15 domains", VIOLET);
  stat(s, 5.6, 5.7, 2.4, "44", "model calls / run", GREEN);
  stat(s, 8.1, 5.7, 2.3, "5", "tabs, fully live", AMBER);
  s.addText("innova-hackathon", {
    x: 10.3, y: 5.9, w: 2.4, h: 0.32, fontFace: B, fontSize: 12, bold: true,
    color: CYAN, align: "center", margin: 0,
  });
  s.addText(".onrender.com", {
    x: 10.3, y: 6.2, w: 2.4, h: 0.3, fontFace: B, fontSize: 12, bold: true,
    color: CYAN, align: "center", margin: 0,
  });
  s.addText("deployed & public", {
    x: 10.3, y: 6.5, w: 2.4, h: 0.28, fontFace: B, fontSize: 10, color: MUTED, align: "center", margin: 0,
  });

  s.addNotes(
    "This is not a mock-up. It is deployed at innova-hackathon.onrender.com and you can use it now.\n\n" +
    "Three real outputs. A true claim: SUPPORTED at 0.93. The same claim falsified: REFUTED at 0.85. And critically, " +
    "when evidence is thin the system returns NOT ESTABLISHED rather than guessing.\n\n" +
    "The right-hand panel is my favourite result. Verifying the Eiffel Tower, retrieval also pulled in Paris, Texas's " +
    "Eiffel Tower and a Las Vegas replica. A keyword system would have counted those as confirmations. Ours scored both " +
    "as weak refutations at 0.10 — it understood they were different entities. Then, because all three came from the " +
    "same domain, independence clustering collapsed them to a single source, which is why confidence stayed at 0.70 " +
    "rather than inflating.\n\n" +
    "A full research run: 248 seconds, 24 sources across 15 domains, 44 model calls."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 7 — Scalability & Impact
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();
  heading(s, "Built to scale, priced to reach everyone", "Scalability & Real-World Impact");

  const domain = (x, y, color, title, body) => {
    card(s, x, y, 3.9, 1.55);
    badge(s, x + 0.22, y + 0.2, color, "▸");
    s.addText(title, {
      x: x + 0.66, y: y + 0.19, w: 3.0, h: 0.32, fontFace: B, fontSize: 13,
      bold: true, color, margin: 0, valign: "middle",
    });
    s.addText(body, {
      x: x + 0.22, y: y + 0.66, w: 3.45, h: 0.75, fontFace: B, fontSize: 10.5, color: MUTED, margin: 0,
    });
  };

  domain(0.6, 1.5, RED, "Journalism", "Newsroom fact-checking before publication, with per-claim provenance.");
  domain(4.7, 1.5, CYAN, "Medicine & Law", "Evidence grading where a confident hallucination causes real harm.");
  domain(8.8, 1.5, GREEN, "Research & Academia", "Literature review with citation integrity and contradiction surfacing.");

  // Scaling path
  card(s, 0.6, 3.35, 6.0, 2.3, CARD2);
  s.addText("Scaling path — already architected", {
    x: 0.9, y: 3.53, w: 5.4, h: 0.3, fontFace: B, fontSize: 13, bold: true, color: CYAN, margin: 0,
  });
  s.addText(
    [
      { text: "SQLite → Postgres: connection string, not a rewrite", options: { bullet: true, breakLine: true } },
      { text: "NumPy store → pgvector / FAISS behind one interface", options: { bullet: true, breakLine: true } },
      { text: "Stateless API — horizontal scaling out of the box", options: { bullet: true, breakLine: true } },
      { text: "Checkpointed runs survive restarts mid-flight", options: { bullet: true, breakLine: true } },
      { text: "Provider-agnostic: no single-vendor lock-in", options: { bullet: true } },
    ],
    { x: 0.95, y: 3.95, w: 5.4, h: 1.6, fontFace: B, fontSize: 11.5, color: MUTED, paraSpaceAfter: 5, margin: 0 },
  );

  // Cost — the accessibility argument
  card(s, 6.9, 3.35, 5.8, 2.3);
  s.addText("Runs entirely on free tiers", {
    x: 7.2, y: 3.53, w: 5.2, h: 0.3, fontFace: B, fontSize: 13, bold: true, color: GREEN, margin: 0,
  });
  s.addText(
    "Role-split across two providers draws on two independent quotas at once — deterministic, so results stay cacheable and reproducible.",
    { x: 7.2, y: 3.95, w: 5.2, h: 0.75, fontFace: B, fontSize: 11, color: MUTED, margin: 0 },
  );
  stat(s, 7.1, 4.72, 1.8, "$0", "cost per run", GREEN);
  stat(s, 9.0, 4.72, 1.8, "~15", "runs / day free", CYAN);
  stat(s, 10.9, 4.72, 1.7, "4", "swappable LLMs", VIOLET);

  // Roadmap
  card(s, 0.6, 5.85, 12.1, 1.0);
  s.addText("Next", {
    x: 0.9, y: 6.12, w: 0.7, h: 0.3, fontFace: B, fontSize: 12, bold: true, color: VIOLET, margin: 0,
  });
  s.addText(
    "Fitted calibration on FEVER / AVeriTeC   →   local MiniCheck entailment (400× cheaper)   →   " +
    "knowledge-graph claim linking   →   real-time newsroom API",
    { x: 1.6, y: 6.06, w: 10.85, h: 0.6, fontFace: B, fontSize: 11, color: MUTED, margin: 0 },
  );

  s.addNotes(
    "Three domains where a confident hallucination does real damage — and where per-claim provenance is worth paying for.\n\n" +
    "On scaling, every swap is already behind an interface. SQLite to Postgres is a connection string because the schema " +
    "was written Postgres-compatible from day one. The vector store exposes the same shape an ANN index would, so " +
    "pgvector or FAISS drops in. The API is stateless and runs are checkpointed, so horizontal scaling and mid-flight " +
    "restarts both work today.\n\n" +
    "The accessibility argument matters as much as the technical one. This runs at zero cost on free tiers. We split the " +
    "two model roles across two providers to draw on two independent quotas simultaneously — and we split by role rather " +
    "than round-robin specifically so identical prompts always hit the same provider and stay cacheable and reproducible.\n\n" +
    "A student in any university can run this today for nothing. That is a deliberate design outcome, not a limitation."
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SLIDE 8 — Close
// ═══════════════════════════════════════════════════════════════════════════
{
  const s = slide();

  s.addShape(pres.ShapeType.ellipse, {
    x: -1.6, y: 3.4, w: 6.6, h: 6.6, fill: { color: VIOLET, transparency: 89 }, line: { width: 0 },
  });
  s.addShape(pres.ShapeType.ellipse, {
    x: 9.4, y: -2.0, w: 5.6, h: 5.6, fill: { color: CYAN, transparency: 91 }, line: { width: 0 },
  });

  s.addText("WHY VERITAS SHOULD WIN", {
    x: 0.7, y: 0.6, w: 8.0, h: 0.3, fontFace: B, fontSize: 11, bold: true,
    color: CYAN, charSpacing: 2, margin: 0,
  });

  s.addText("We measured our own system\nthe way we measure claims.", {
    x: 0.7, y: 1.0, w: 8.6, h: 1.5, fontFace: H, fontSize: 36, bold: true,
    color: TEXT, lineSpacing: 42, margin: 0,
  });

  const proof = [
    [GREEN, "Rejected the obvious architecture — with published evidence"],
    [CYAN, "13 agents, 2 dynamic fan-outs, 230 tests, 0 lint errors"],
    [VIOLET, "Calibrated confidence — not an LLM guessing about itself"],
    [AMBER, "Deployed, public, and running on free tiers today"],
  ];
  proof.forEach(([c, t], i) => {
    const y = 2.75 + i * 0.52;
    s.addShape(pres.ShapeType.ellipse, { x: 0.72, y: y + 0.09, w: 0.18, h: 0.18, fill: { color: c }, line: { width: 0 } });
    s.addText(t, { x: 1.08, y, w: 8.3, h: 0.36, fontFace: B, fontSize: 13.5, color: TEXT, valign: "middle", margin: 0 });
  });

  // Links + contact
  card(s, 9.7, 1.0, 3.0, 3.9, CARD2);
  s.addText("Live demo", {
    x: 9.95, y: 1.25, w: 2.5, h: 0.28, fontFace: B, fontSize: 10.5, bold: true, color: CYAN, margin: 0,
  });
  s.addText("innova-hackathon\n.onrender.com", {
    x: 9.95, y: 1.55, w: 2.5, h: 0.6, fontFace: B, fontSize: 11.5, color: TEXT, margin: 0,
  });
  s.addText("Source", {
    x: 9.95, y: 2.3, w: 2.5, h: 0.28, fontFace: B, fontSize: 10.5, bold: true, color: CYAN, margin: 0,
  });
  s.addText("github.com/avinas29\n/innova-hackathon", {
    x: 9.95, y: 2.6, w: 2.5, h: 0.6, fontFace: B, fontSize: 11.5, color: TEXT, margin: 0,
  });
  s.addText("Team Peakbuster", {
    x: 9.95, y: 3.4, w: 2.5, h: 0.28, fontFace: B, fontSize: 10.5, bold: true, color: CYAN, margin: 0,
  });
  s.addText("umam.ali@iitg.ac.in\nr.atulaya@iitg.ac.in\n+91 7060001402", {
    x: 9.95, y: 3.7, w: 2.5, h: 0.95, fontFace: B, fontSize: 10.5, color: MUTED, margin: 0,
  });

  s.addText(
    "Anyone can build an agent that answers. We built one that tells you when it shouldn't.",
    { x: 0.7, y: 5.3, w: 8.6, h: 0.8, fontFace: H, fontSize: 19, bold: true, color: VIOLET, italic: true, margin: 0 },
  );

  s.addText("Avinash   ·   Atulaya Raj   ·   Mohammad Umam Ali        IIT Guwahati   ·   InnovaHack 2026", {
    x: 0.7, y: 6.5, w: 12.0, h: 0.3, fontFace: B, fontSize: 11, color: DIM, margin: 0,
  });

  s.addNotes(
    "To close.\n\n" +
    "The single sentence I want you to remember: anyone can build an agent that answers. We built one that tells you " +
    "when it shouldn't.\n\n" +
    "We started by researching what does NOT work and found published evidence that the obvious multi-agent-debate " +
    "architecture is a martingale. We changed the design because of that evidence rather than building the intuitive " +
    "thing.\n\n" +
    "The system is real: thirteen agents, two dynamic fan-outs, 230 passing tests, zero lint errors, deployed publicly, " +
    "running at zero cost.\n\n" +
    "And the confidence number we publish is a measurement with a ceiling grounded in source independence — not a " +
    "language model guessing about itself.\n\n" +
    "The demo is live right now. Please try to break it. Thank you."
  );
}

pres.writeFile({ fileName: "/Users/avinashbishnoidmn/jhaat/deck/VERITAS-Peakbuster.pptx" })
  .then((f) => console.log("written:", f));
