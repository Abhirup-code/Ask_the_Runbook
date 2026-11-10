"""Checks an answer before anyone sees it. An on-call engineer acting on a made-up command or
threshold at 3 a.m. is worse than no answer, so the rules are strict:

1. The answer cites at least one section, as [file#n].
2. Every cited section is one the agent actually retrieved in this conversation.
3. Every number in the answer appears in the sections it cites. No invented thresholds or timings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from runbook.docs import Section

CITATION = re.compile(r"\[([a-z0-9][a-z0-9._-]*#\d+)\]")
NUMBER = re.compile(r"(?<![\w.#-])\d+(?:[.,]\d+)*")
LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)


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
        invented = sorted(numbers(answer) - numbers(sources), key=lambda n: (len(n), n))
        if invented:
            problems.append(f"These numbers are not in the cited sections: {', '.join(invented)}.")
    return Verdict(not problems, cited, problems)
