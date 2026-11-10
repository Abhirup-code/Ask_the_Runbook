"""Loads Markdown runbooks and splits them into sections the agent can search and cite."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Section:
    id: str  # "checkout-availability#2": file stem and section number, short enough to cite
    doc: str  # the document's title (its first heading)
    title: str  # this section's heading
    text: str  # the section body, without the heading

    def render(self) -> str:
        return f"[{self.id}] {self.doc} > {self.title}\n{self.text}"


def split(stem: str, markdown: str) -> list[Section]:
    """Splits one document at its #, ##, and ### headings.

    Lines starting with # inside fenced code blocks are comments (PromQL, shell), not headings.
    Text before the first sub-heading becomes section 0, titled like the document.
    """
    doc_title = stem
    sections: list[tuple[str, list[str]]] = []
    title, body, in_code = None, [], False
    for line in markdown.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match and len(match.group(1)) == 1 and doc_title == stem:
            doc_title = match.group(2)
            title = doc_title
            continue
        if match:
            if title is not None:
                sections.append((title, body))
            title, body = match.group(2), []
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
