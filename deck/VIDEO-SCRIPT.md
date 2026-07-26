# VERITAS — 5-Minute Video Script
**Team Peakbuster · InnovaHack 2026**

Total: **5:00** · Slides **0:00–2:45** · Live demo **2:45–4:30** · Close **4:30–5:00**
Target pace: ~145 words/minute. Full script below is ~720 words — do not add more.

---

## ⚠️ THE ONE THING THAT MAKES THIS WORK

A full research run takes **~4 minutes**. You cannot start it on camera and wait.

**So: start the run BEFORE you say a word, and let it finish while you present the slides.**

By the time you reach the demo at 2:45, the run is complete and every tab is populated.
Nothing is faked, nothing is cut, and you never wait on screen.

### Pre-recording checklist (do this 10 minutes before)

| # | Action | Why |
|---|---|---|
| 1 | Open **https://innova-hackathon.onrender.com** and leave it 60s | Wakes the sleeping free instance |
| 2 | Open **https://veritas-searxng.onrender.com/healthz** | Wakes SearXNG too — otherwise first search times out |
| 3 | In a terminal: `cd ~/jhaat/backend && ./.venv/bin/veritas verify "The Eiffel Tower is located in Paris, France."` | **Warms the cache.** The repeat during recording returns instantly |
| 4 | Same for: `...veritas verify "The Eiffel Tower is located in Rome, Italy."` | Your REFUTED example, also cached |
| 5 | Confirm header reads `groq · llm entailment · search: searxng ✓` | Green = retrieval is live |
| 6 | Set screen to 1920×1080, browser zoom **110%**, hide bookmarks bar | Readable when compressed by YouTube |

### Recording layout

- **Screen recording** (OBS / QuickTime) of the full screen
- Slides in **presenter mode on one display**, browser + terminal on the other
- Or simply **alt-tab** between PowerPoint and Chrome — clean enough

---

# SCRIPT

---

## ⏱ 0:00 — 0:10 · BEFORE YOU START TALKING

**SHOW:** Browser, live site.
**DO:** Paste this into the search box and press **Research**:

> `How much has global solar power capacity grown since 2020?`

Let it start. Watch the "Live" tab flicker once. **Now alt-tab to your slides.**

> 🎙 *(say nothing — this is 10 seconds of setup, trim it in editing or talk over it)*

---

## ⏱ 0:10 — 0:45 · SLIDE 1 — The Problem  *(35s)*

**SHOW:** Slide 1 — VERITAS title slide.

> 🎙 **"Hi, we're Team Peakbuster from IIT Guwahati, and this is VERITAS.**
>
> **The problem with AI research tools isn't that they're sometimes wrong. It's that they're wrong with the exact same confidence they're right. There's no signal to the user.**
>
> **Three documented facts. LLMs state falsehoods fluently. When you ask a model how confident it is, that number carries about 0.2 calibration error — it's close to meaningless. And models almost never say 'I don't know,' even when that's the correct answer.**
>
> **In medicine, law, or journalism, a confident hallucination is worse than no answer — because it gets trusted."**

**Point at:** the three red/amber/violet cards as you say each fact.

---

## ⏱ 0:45 — 1:25 · SLIDE 2 — The Innovation  *(40s)*
### 🔑 THIS IS YOUR MOST IMPORTANT 40 SECONDS

**SHOW:** Slide 2 — the red "X" card vs the green "✓" card.

> 🎙 **"The obvious build is 'several AI agents that debate each other.' We researched that first — and found published evidence that it doesn't work.**
>
> **Multi-agent debate forms a martingale. Expected correctness does not improve across rounds when agents share the same inputs. Weak models correct only three-point-six percent of stance biases, and agents abandon correct positions to conform.**
>
> **So we changed one variable. Our Advocate and Sceptic agents receive completely disjoint halves of the evidence. Now when they disagree, it's because they genuinely saw different things — that's real information aggregation, not personality clash.**
>
> **And the number we publish isn't a model guessing about itself. It's seven orthogonal features, statistically calibrated."**

**Point at:** left card → right card → the four numbered innovations.

---

## ⏱ 1:25 — 2:00 · SLIDE 3 — Architecture  *(35s)*

**SHOW:** Slide 3 — the agent diagram.

> 🎙 **"Thirteen specialised agents, orchestrated by LangGraph as an explicit state machine.**
>
> **Two dynamic fan-outs. The first spawns a researcher per question. The second spawns a verification branch per claim — that's genuine map-reduce over a workload whose width isn't known until runtime.**
>
> **We chose LangGraph over CrewAI because our workload is N claims by M evidence items verified independently, then reduced. We rejected AutoGen outright — it's three months stale, and CC-BY-4.0 isn't a software licence.**
>
> **Four LLM providers, eight source types, no vendor lock-in."**

**Point at:** the two fan-out boxes as you mention them.

---

## ⏱ 2:00 — 2:25 · SLIDE 4 — Pipeline  *(25s)*

**SHOW:** Slide 4 — the eight numbered steps.

> 🎙 **"Eight stages. The first four build a report. The last four try to tear it down.**
>
> **Step five is the one nobody else has. Ten news outlets reprinting one wire story is ONE piece of evidence, not ten. We collapse them using four signals — and on a live run, twenty evidence items became twelve genuinely independent sources.**
>
> **That directly caps the score. One source can never license more than 0.70 confidence — because a single source can simply be wrong, and we'd have no way to know."**

**Point at:** the cyan independence card at the bottom, then the 0.70 / 0.93 stats.

---

## ⏱ 2:25 — 2:45 · SLIDE 5 — Engineering  *(20s)*

**SHOW:** Slide 5 — the stack + depth grid.

> 🎙 **"This is production engineering, not a notebook. FastAPI, Next.js, single Docker image, deployed and public.**
>
> **Two-hundred-thirty tests, all passing, with zero network calls — the whole suite runs hermetically. Token-per-minute rate limiting, per-model quota tracking, automatic prompt fitting. Every one of those is a real bug we hit and fixed."**

---

## ⏱ 2:45 — 4:30 · LIVE DEMO  *(105s)*
### Alt-tab to the browser. Your run finished ~30 seconds ago.

---

### 2:45 — 3:00 · The run completed  *(15s)*

**SHOW:** The metrics bar at the top of the page.

> 🎙 **"Here's a run that completed while I was talking. Twenty-four sources across fifteen domains. And look — twenty evidence items reduced to twelve independent ones. That de-duplication is what keeps the confidence honest."**

**Point at:** the "Independent sources" stat card.

---

### 3:00 — 3:25 · Claims tab — the trust heatmap  *(25s)*

**SHOW:** Click the **Claims** tab.

> 🎙 **"Every claim from the report, colour-coded. Green is supported, red refuted, amber not established.**
>
> **That amber matters most. When the evidence is thin, the system abstains instead of guessing. Most AI tools physically cannot do this — they always produce an answer."**

**Point at:** an amber "Not established" row.

---

### 3:25 — 3:55 · Expand a claim — the money shot  *(30s)*

**SHOW:** Click any **green SUPPORTED** claim to expand it.

> 🎙 **"Open any claim and you see the full audit trail. The seven confidence features. The Advocate's argument, the Sceptic's argument — each built from a different half of the evidence.**
>
> **And here — every source, with its entailment score. Notice this one is greyed out and marked 'duplicate, not counted again.' That's the independence clustering working. It found the same content twice and refused to count it as two confirmations."**

**Point at:** the advocate/sceptic panels → then a greyed-out duplicate evidence row.

---

### 3:55 — 4:10 · Evidence graph  *(15s)*

**SHOW:** Click the **Evidence graph** tab. Hover one node.

> 🎙 **"The whole run as a graph. Claims in the centre, sources around them. Green edges support, red refute, orange dashed lines are sources contradicting each other."**

---

### 4:10 — 4:30 · Confidence model — prove it's not an LLM guess  *(20s)*

**SHOW:** Click the **Confidence model** tab.
**DO:** Drag **"Model's own confidence"** slider to **1.00**. Drag **"Source independence"** to **0**.

> 🎙 **"And this is the proof it isn't just asking a model. Watch — I'll set the model's own confidence to maximum and remove the source independence.**
>
> **The score stays low. The system refuses to be talked into certainty by a model's say-so. That's the whole thesis in one slider."**

**Point at:** the "capped at 0.70" amber badge appearing.

---

## ⏱ 4:30 — 5:00 · SLIDE 8 — Close  *(30s)*
### Alt-tab back to slides. Skip slide 6 and 7 — the demo covered them.

**SHOW:** Slide 8 — "Why VERITAS should win."

> 🎙 **"So — we started by researching what doesn't work, found published evidence against the obvious architecture, and changed our design because of it.**
>
> **Thirteen agents. Two-hundred-thirty tests. Deployed, public, running at zero cost on free tiers — any student can run this today.**
>
> **Anyone can build an agent that answers. We built one that tells you when it shouldn't.**
>
> **It's live right now at innova-hackathon dot onrender dot com. Please try to break it. Thank you."**

**END ON:** Slide 8 held for 3 seconds so the URL and contact details are readable.

---

# QUICK REFERENCE CARD
*(print this — it's all you need while recording)*

| Time | Screen | Key line |
|---|---|---|
| 0:00 | Browser — **start the run** | *(silent)* |
| 0:10 | Slide 1 | "Wrong with the same confidence they're right" |
| 0:45 | **Slide 2** | "Debate is a martingale — we changed one variable" |
| 1:25 | Slide 3 | "Thirteen agents, two dynamic fan-outs" |
| 2:00 | Slide 4 | "Ten outlets copying one story is ONE source" |
| 2:25 | Slide 5 | "230 tests, hermetic, zero network" |
| 2:45 | **Browser — metrics** | "20 evidence → 12 independent" |
| 3:00 | Claims tab | "It abstains instead of guessing" |
| 3:25 | **Expand a claim** | "Duplicate — not counted again" |
| 3:55 | Evidence graph | "Orange lines are sources contradicting each other" |
| 4:10 | **Confidence model** | "Score stays low — it refuses to be talked into certainty" |
| 4:30 | Slide 8 | "We built one that tells you when it shouldn't" |

---

# IF SOMETHING GOES WRONG

| Problem | Recovery |
|---|---|
| Run still going at 2:45 | Cut to the **Live** tab — the streaming agent log is itself a great visual. Say: *"you can watch each agent work in real time."* Then continue and return to Claims later |
| Site is asleep / slow | You skipped checklist steps 1–2. Stop, load both URLs, wait 60s, restart recording |
| Search returns nothing | Check `/api/search/health` — if SearXNG is red, the service is asleep |
| A claim errors out | **Leave it in.** Say: *"one branch failed and degraded to 'not established' — the run continues. That's deliberate."* Graceful degradation is a feature, not an excuse |

---

# DELIVERY NOTES

- **Slow down on slide 2.** It's your differentiator. Everything else can be brisk.
- **Say numbers precisely.** "Three-point-six percent," "zero-point-seven-zero." Precision signals rigour.
- **Don't read the slides.** They're visual anchors; the script says more than they show.
- **Never apologise for speed.** If a judge sees pacing, say *"that's deliberate rate-limiting to stay inside the free tier"* — it's an engineering decision, not a defect.
- **Record slides and demo in one continuous take** if you can. Unedited is more convincing than polished.
