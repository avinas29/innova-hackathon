# VERITAS — Read This Word For Word

Numbers are spelled how you say them. Just read.

---

## BEFORE YOU TALK

Paste your topic into the site, press Research, then switch to your slides.
Say nothing for ten seconds. Start talking on Slide 1.

---

## SLIDE 1

Hi, we're Team Peakbuster from IIT Guwahati, and this is VERITAS.

The problem with AI research tools isn't that they're sometimes wrong. It's that they're wrong with the exact same confidence they're right. There's no signal to the user.

Three documented facts. Large language models state falsehoods fluently. When you ask a model how confident it is, that number carries about zero point two calibration error, which makes it close to meaningless. And models almost never say "I don't know", even when that's the correct answer.

In medicine, law, or journalism, a confident hallucination is worse than no answer, because it gets trusted.

---

## SLIDE 2

The obvious thing to build is several AI agents that debate each other. We researched that first, and we found published evidence that it doesn't work.

Multi-agent debate forms a martingale. Expected correctness does not improve across rounds when the agents share the same inputs. Weak models correct only three point six percent of stance biases, and agents abandon correct positions just to conform with the group.

So we changed one variable. Our Advocate and Sceptic agents receive completely disjoint halves of the evidence. Now when they disagree, it's because they genuinely saw different things. That's real information aggregation, not personality clash.

And the confidence number we publish isn't a model guessing about itself. It's seven separate features, statistically calibrated.

---

## SLIDE 3

Thirteen specialised agents, orchestrated by LangGraph as an explicit state machine.

There are two dynamic fan-outs. The first spawns a researcher for every question. The second spawns a verification branch for every claim. That's genuine map-reduce over a workload whose width isn't known until runtime.

We chose LangGraph over CrewAI because our workload is N claims by M evidence items, verified independently and then reduced. We rejected AutoGen outright. It's three months stale, and its licence isn't a software licence at all.

Four language model providers, eight source types, and no vendor lock-in.

---

## SLIDE 4

Eight stages. The first four build a report. The last four try to tear it down.

Step five is the one nobody else has. Ten news outlets reprinting one wire story is one piece of evidence, not ten. We collapse them using four different signals. On a live run, twenty evidence items became twelve genuinely independent sources.

That directly caps the score. One source can never license more than zero point seven confidence, because a single source can simply be wrong, and we'd have no way to know.

---

## SLIDE 5

This is production engineering, not a notebook demo. FastAPI, Next dot js, a single Docker image, deployed and public.

Two hundred and thirty tests, all passing, with zero network calls. The entire suite runs offline. Token per minute rate limiting, per model quota tracking, automatic prompt fitting. Every one of those is a real bug we hit and fixed.

---

## NOW SWITCH TO THE BROWSER

---

## DEMO — THE METRICS AT THE TOP

Here's a run that completed while I was talking. Twenty-four sources across fifteen domains. And look at this: twenty evidence items reduced to twelve independent ones. That de-duplication is what keeps the confidence honest.

---

## DEMO — CLICK THE CLAIMS TAB

Every claim from the report, colour coded. Green is supported, red is refuted, amber is not established.

That amber matters most. When the evidence is thin, the system abstains instead of guessing. Most AI tools physically cannot do this. They always produce an answer.

---

## DEMO — CLICK A GREEN CLAIM TO EXPAND IT

Open any claim and you see the full audit trail. The seven confidence features. The Advocate's argument, and the Sceptic's argument, each built from a different half of the evidence.

And here, every source with its entailment score. Notice this one is greyed out and marked "duplicate, not counted again". That's the independence clustering working. It found the same content twice and refused to count it as two confirmations.

---

## DEMO — CLICK THE EVIDENCE GRAPH TAB

The whole run as a graph. Claims in the centre, sources around them. Green edges support, red edges refute, and the orange dashed lines are sources contradicting each other.

---

## DEMO — CLICK THE CONFIDENCE MODEL TAB

And this is the proof it isn't just asking a model. Watch. I'll set the model's own confidence to maximum, and remove the source independence completely.

The score stays low. The system refuses to be talked into certainty by a model's say so. That's the whole thesis, in one slider.

---

## NOW SWITCH BACK TO SLIDE 8

---

## SLIDE 8

So, we started by researching what doesn't work, we found published evidence against the obvious architecture, and we changed our design because of it.

Thirteen agents. Two hundred and thirty tests. Deployed, public, and running at zero cost on free tiers, so any student can run this today.

Anyone can build an agent that answers. We built one that tells you when it shouldn't.

It's live right now at innova hackathon dot onrender dot com. Please try to break it.

Thank you.

---

## STOP. Hold Slide 8 on screen for three seconds, then end the recording.
