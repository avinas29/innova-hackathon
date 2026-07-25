"""Evaluation datasets.

Two loaders:

* ``builtin`` — a 36-claim balanced development set shipped with the repo, so
  ``veritas eval`` works immediately on a fresh clone with no downloads.
* ``jsonl``   — any file of ``{"claim": ..., "label": ...}`` records, which is
  the shape FEVER, AVeriTeC and HoVer all convert to in one line of jq.

**Be honest about what the builtin set is.** Thirty-six hand-written claims is a
smoke test for the harness, not a benchmark. It is balanced and deliberately
includes an NEI third — the class most systems fail on — but numbers from it
should never be presented as benchmark results. For a defensible headline
figure, run against FEVER or AVeriTeC via the JSONL loader. The harness reports
the dataset name and size alongside every metric so this cannot be accidentally
misrepresented.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LabelledClaim:
    claim: str
    label: str          # SUPPORTED | REFUTED | NEI
    note: str = ""


# Facts chosen for stability: nothing here changes with time, and all of it is
# documented in reference sources any retriever will surface.
_SUPPORTED = [
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by Guido van Rossum.",
    "The Great Barrier Reef is located off the coast of Queensland, Australia.",
    "DNA has a double-helix structure.",
    "The Eiffel Tower is located in Paris, France.",
    "Mount Everest is the highest mountain above sea level on Earth.",
    "The Apollo 11 mission landed humans on the Moon in 1969.",
    "The chemical symbol for gold is Au.",
    "The Amazon River is located in South America.",
    "The Berlin Wall fell in 1989.",
    "Insulin is used in the treatment of diabetes.",
    "The Pacific Ocean is the largest ocean on Earth.",
]

# Each is a minimally-altered version of a true statement: wrong number, wrong
# date, wrong entity. This is deliberate — "basically right but numerically
# wrong" is the hardest and most consequential error class for a fact-checker,
# and a system that waves those through is not doing its job.
_REFUTED = [
    "Water boils at 50 degrees Celsius at standard atmospheric pressure.",
    "The Python programming language was created by James Gosling.",
    "The Great Barrier Reef is located off the coast of Brazil.",
    "Mount Everest is located in Japan.",
    "The Apollo 11 mission landed humans on the Moon in 1975.",
    "The chemical symbol for gold is Ag.",
    "The Berlin Wall fell in 1975.",
    "The Eiffel Tower is located in Rome, Italy.",
    "The Amazon River is the longest river in Europe.",
    "Light travels at approximately 300 kilometres per second in a vacuum.",
    "Humans have 46 pairs of chromosomes in each somatic cell.",
    "The Pacific Ocean is the smallest ocean on Earth.",
]

# Plausible-sounding, specific, and genuinely unestablished. A system that
# confidently labels these SUPPORTED or REFUTED is hallucinating, which is
# exactly what this third of the set is here to catch.
_NEI = [
    "The average mid-sized German office contained 1,240 paperclips in 1997.",
    "More than 60% of professional chess players prefer wooden boards to plastic ones.",
    "The first person to eat a tomato in Britain did so on a Tuesday.",
    "Approximately 3.7% of household cats in Portugal are left-pawed.",
    "The median commute time for dental hygienists in Latvia is 27 minutes.",
    "Roughly one in nine violinists tunes their instrument before drinking coffee.",
    "The most common password among municipal librarians in 2011 contained a colour.",
    "Sixty-two percent of amateur beekeepers name at least one of their hives.",
    "The average lifespan of a stapler in a Canadian law firm is 8.3 years.",
    "Most origami instructors learned the craft from a relative rather than a book.",
    "Approximately 14% of lighthouse keepers historically kept a personal diary.",
    "The typical municipal swimming pool in Belgium is repainted every 6.2 years.",
]


def builtin_dataset() -> list[LabelledClaim]:
    """The shipped 36-claim balanced development set."""
    return [
        *[LabelledClaim(c, "SUPPORTED", "verifiable reference fact") for c in _SUPPORTED],
        *[LabelledClaim(c, "REFUTED", "minimally altered from a true statement") for c in _REFUTED],
        *[LabelledClaim(c, "NEI", "specific but unestablished") for c in _NEI],
    ]


def load_jsonl(path: str | Path) -> list[LabelledClaim]:
    """Load ``{"claim", "label"}`` records from a JSONL file.

    Accepts FEVER's native label spelling (``SUPPORTS`` / ``REFUTES`` /
    ``NOT ENOUGH INFO``) as well as ours.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")

    out: list[LabelledClaim] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} is not valid JSON: {exc}") from exc

        claim = record.get("claim") or record.get("text") or record.get("statement")
        label = record.get("label") or record.get("verdict") or record.get("gold")
        if not claim or not label:
            continue
        out.append(
            LabelledClaim(
                claim=str(claim), label=normalise_label(str(label)), note=str(record.get("note", ""))
            )
        )

    if not out:
        raise ValueError(f"no usable records found in {path}")
    return out


def normalise_label(raw: str) -> str:
    """Map external label vocabularies onto ours."""
    normalised = raw.strip().upper().replace("-", " ").replace("_", " ")
    if normalised in {"SUPPORTS", "SUPPORTED", "TRUE", "SUPPORTING"}:
        return "SUPPORTED"
    if normalised in {"REFUTES", "REFUTED", "FALSE", "CONTRADICTED"}:
        return "REFUTED"
    if normalised in {
        "NOT ENOUGH INFO",
        "NOTENOUGHINFO",
        "NEI",
        "UNVERIFIABLE",
        "CONFLICTING",
        "UNKNOWN",
    }:
        return "NEI"
    return normalised


def load_dataset(name: str, limit: int | None = None) -> tuple[list[LabelledClaim], str]:
    """Resolve a dataset name to claims plus a human-readable description."""
    if name in {"builtin", "default", ""}:
        claims = builtin_dataset()
        description = "builtin balanced development set (hand-written, 36 claims)"
    else:
        claims = load_jsonl(name)
        description = f"{Path(name).name} ({len(claims)} claims)"

    if limit is not None and limit > 0:
        claims = _balanced_sample(claims, limit)
    return claims, description


def _balanced_sample(claims: list[LabelledClaim], limit: int) -> list[LabelledClaim]:
    """Take up to ``limit`` claims, keeping the label mix as even as possible.

    Truncating the head of a sorted dataset would silently evaluate on one label
    only — a mistake that produces spectacular and completely meaningless
    accuracy numbers.
    """
    if limit >= len(claims):
        return claims

    by_label: dict[str, list[LabelledClaim]] = {}
    for claim in claims:
        by_label.setdefault(claim.label, []).append(claim)

    out: list[LabelledClaim] = []
    index = 0
    while len(out) < limit:
        added = False
        for bucket in by_label.values():
            if index < len(bucket) and len(out) < limit:
                out.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return out
