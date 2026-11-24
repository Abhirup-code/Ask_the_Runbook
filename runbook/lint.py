"""Checks runbooks for the things that make them hard to retrieve or cite.

The agent can only be as good as what it is given, and the ways a runbook goes wrong are quiet
ones: a heading with nothing under it disappears from the index entirely, two sections with the
same title cite identically, and a section longer than the context budget gets trimmed before the
model ever reads the end of it. None of that shows up when you read the file yourself.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from runbook.agent import BUDGET
from runbook.docs import Section, parse, split

SHORT = 80  # a section with less body than this rarely carries an answer on its own


@dataclass(frozen=True)
class Problem:
    where: str  # a section id, or a file name when the section never made it into the index
    message: str

    def __str__(self) -> str:
        return f"{self.where}: {self.message}"


def _empty_headings(stem: str, markdown: str, sections: list[Section]) -> list[Problem]:
    """Headings in the file that no section came from, because nothing was written under them."""
    kept = Counter(s.title for s in sections)
    found: list[Problem] = []
    for level, heading, _ in parse(markdown):
        if level and kept[heading]:
            kept[heading] -= 1
        elif level and not (level == 1 and heading == (sections[0].doc if sections else "")):
            found.append(Problem(f"{stem}.md", f"nothing is written under {heading!r}, so it is not indexed"))
    return found


def check(folder: str | Path) -> list[Problem]:
    """Every problem in a folder of runbooks, in file order."""
    problems: list[Problem] = []
    for path in sorted(Path(folder).glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        sections = split(path.stem, markdown)
        problems += _empty_headings(path.stem, markdown, sections)
        if not sections:
            problems.append(Problem(path.name, "no sections at all, so nothing here can be found or cited"))
            continue
        if sections[0].doc == path.stem:
            problems.append(
                Problem(path.name, "no '# Title' line, so it is named after the file in every citation")
            )
        for title, count in Counter(s.title for s in sections).items():
            if count > 1:
                problems.append(
                    Problem(
                        path.name, f"{count} sections are titled {title!r}; a citation cannot tell them apart"
                    )
                )
        for section in sections:
            if len(section.text) > BUDGET:
                problems.append(
                    Problem(
                        section.id,
                        f"{len(section.text):,} characters, over the {BUDGET:,} budget, so it is trimmed",
                    )
                )
            elif len(section.text) < SHORT:
                problems.append(
                    Problem(
                        section.id, f"only {len(section.text)} characters, likely too little to answer from"
                    )
                )
    return problems


def report(problems: list[Problem], folder: str | Path) -> str:
    if not problems:
        return f"No problems found in {folder}."
    lines = [f"{len(problems)} problem(s) in {folder}:"]
    lines += [f"  {problem}" for problem in problems]
    return "\n".join(lines)
