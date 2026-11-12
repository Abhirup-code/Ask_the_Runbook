"""Checks an answer before anyone sees it. An on-call engineer acting on a made-up command or
threshold at 3 a.m. is worse than no answer, so the rules are strict:

1. The answer cites at least one section, as [file#n].
2. Every cited section is one the agent actually retrieved in this conversation.
3. Every number in the answer appears in the sections it cites. No invented thresholds or timings.
   A time span may be written in another unit: "60 minutes" is fine when the runbook says "1h".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from runbook.docs import Section

CITATION = re.compile(r"\[([a-z0-9][a-z0-9._-]*#\d+)\]")
NUMBER = re.compile(r"(?<![\w.#-])\d+(?:[.,]\d+)*")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
DURATION = re.compile(
    r"(?<![\w.#-])(\d+(?:\.\d+)?)\s*"
    r"(ms|milliseconds?|s|secs?|seconds?|m|mins?|minutes?|h|hrs?|hours?|d|days?|w|weeks?)\b",
    re.IGNORECASE,
)
SECONDS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


@dataclass(frozen=True)
class Verdict:
    ok: bool
    cited: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def numbers(text: str) -> set[str]:
    """Numbers as written, minus citations and list markers like "2." at the start of a line."""
    text = CITATION.sub(" ", text)
    text = LIST_MARKER.sub(" ", text)
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def _unit(word: str) -> float:
    word = word.lower()
    if word.startswith("ms") or word.startswith("milli"):
        return SECONDS["ms"]
    if word.startswith("mi") or word == "m":
        return SECONDS["m"]
    return SECONDS[word[0]]


def durations(text: str) -> dict[str, set[float]]:
    """Each number that is a time span, mapped to its length in seconds ("5m" -> {"5": {300}})."""
    found: dict[str, set[float]] = {}
    for value, unit in DURATION.findall(CITATION.sub(" ", text)):
        found.setdefault(value, set()).add(round(float(value) * _unit(unit), 3))
    return found


def check(answer: str, seen: dict[str, Section]) -> Verdict:
    answer = answer.strip()
    if not answer:
        return Verdict(False, problems=["The answer is empty."])
    cited = list(dict.fromkeys(CITATION.findall(answer)))
    problems = []
    if not cited:
        problems.append("The answer cites no section. Cite every claim as [file#n].")
    unseen = [c for c in cited if c not in seen]
    if unseen:
        problems.append(f"It cites sections it never retrieved: {', '.join(unseen)}.")
    sources = " ".join(seen[c].title + " " + seen[c].text for c in cited if c in seen)
    if cited and not unseen:
        source_spans = {s for spans in durations(sources).values() for s in spans}
        answer_spans = durations(answer)
        invented = sorted(
            n
            for n in numbers(answer) - numbers(sources)
            if not (answer_spans.get(n) and answer_spans[n] <= source_spans)
        )
        invented.sort(key=lambda n: (len(n), n))
        if invented:
            problems.append(f"These numbers are not in the cited sections: {', '.join(invented)}.")
    return Verdict(not problems, cited, problems)
