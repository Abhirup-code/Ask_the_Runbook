"""Keyword search over runbook sections with BM25, the ranking classic search engines use.

No embedding model to download and nothing to host: it runs anywhere Python runs, and the
retrieval eval in evals/ keeps it honest.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from runbook.docs import Section

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i if in is it its me my of on or so that the then this "
    "to up was what when where which who why will with you your".split()
)


def stem(word: str) -> str:
    """A deliberately small stemmer: "alerts", "alerting", and "alerted" all become "alert"."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def tokens(text: str) -> list[str]:
    return [stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


class Index:
    """BM25 over section title and body. The title counts twice, because headings say what a
    section is about."""

    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        self.sections = sections
        self.k1, self.b = k1, b
        self.terms = [Counter(tokens(f"{s.title} {s.title} {s.doc} {s.text}")) for s in sections]
        self.lengths = [sum(t.values()) for t in self.terms]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(term for t in self.terms for term in t)
        n = len(sections)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}
        self.by_id = {s.id: s for s in sections}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """The k best sections for a query. Sections that share no words with it are left out."""
        q = set(tokens(query))
        hits = []
        for section, terms, length in zip(self.sections, self.terms, self.lengths, strict=True):
            score = 0.0
            for term in q & terms.keys():
                tf = terms[term]
                score += (
                    self.idf[term]
                    * tf
                    * (self.k1 + 1)
                    / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
                )
            if score > 0:
                hits.append(Hit(section, score))
        hits.sort(key=lambda h: (-h.score, h.section.id))
        return hits[:k]

    def get(self, section_id: str) -> Section | None:
        return self.by_id.get(section_id)
