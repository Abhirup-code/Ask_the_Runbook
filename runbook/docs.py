"""Loads Markdown runbooks and splits them into sections the agent can search and cite."""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

# Every ATX level, not just the first three: a runbook that numbers its steps with #### had the
# whole lot swallowed into its parent section, which then cited as one block and crowded the
# retrieved context. The optional trailing run of #s is Markdown's closing sequence, and leaving
# it on put "2. Check the causes ##" in a title that the index counts twice.
HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    id: str  # "checkout-availability#2": file stem and section number, short enough to cite
    doc: str  # the document's title (its first heading)
    title: str  # this section's heading
    text: str  # the section body, without the heading

    def render(self) -> str:
        return f"[{self.id}] {self.doc} > {self.title}\n{self.text}"


def parse(markdown: str) -> Iterator[tuple[int | None, str, str]]:
    """Each line as (heading level or None, heading text, raw line).

    The one place that decides what a heading is, so the splitter and the linter cannot drift
    apart about it. Lines starting with # inside fenced code blocks are comments (PromQL, shell),
    not headings.
    """
    in_code = False
    for line in markdown.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match:
            yield len(match.group(1)), match.group(2), line
        else:
            yield None, "", line


def split(stem: str, markdown: str) -> list[Section]:
    """Splits one document at its headings, any level from # to ######.

    Lines starting with # inside fenced code blocks are comments (PromQL, shell), not headings.
    Text before the first sub-heading becomes section 0, titled like the document.
    """
    doc_title = stem
    sections: list[tuple[str, list[str]]] = []
    title, body = None, []
    for level, heading, line in parse(markdown):
        if level == 1 and doc_title == stem:
            doc_title = heading
            title = doc_title
            continue
        if level:
            if title is not None:
                sections.append((title, body))
            title, body = heading, []
            continue
        body.append(line)
    if title is not None or any(line.strip() for line in body):
        sections.append((title or doc_title, body))

    out = []
    for title, lines in sections:
        text = "\n".join(lines).strip()
        if text:
            out.append(Section(f"{stem}#{len(out)}", doc_title, title, text))
    return out


def load(folder: str | Path) -> list[Section]:
    """Every section of every .md file in a folder, in a stable order."""
    sections = []
    for path in sorted(Path(folder).glob("*.md")):
        sections += split(path.stem, path.read_text(encoding="utf-8"))
    if not sections:
        raise ValueError(f"No Markdown sections found in {folder}")
    return sections
