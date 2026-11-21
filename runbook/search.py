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
    section is about.

    The terms are held as an inverted index: each word points at the sections containing it, and
    how often. A query then touches only the sections that share a word with it, rather than every
    section in the corpus, which is what lets this stay fast as a team's runbooks grow.
    """

    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        self.sections = sections
        self.k1, self.b = k1, b
        counts = [Counter(tokens(f"{s.title} {s.title} {s.doc} {s.text}")) for s in sections]
        self.lengths = [sum(c.values()) for c in counts]
        self.avg = sum(self.lengths) / len(self.lengths)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for n, count in enumerate(counts):
            for term, tf in count.items():
                self.postings.setdefault(term, []).append((n, tf))
        total = len(sections)
        self.idf = {
            term: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }
        self.by_id = {s.id: s for s in sections}

    def _bm25(self, term: str, tf: int, length: int) -> float:
        """One term's contribution to one section's score."""
        return (
            self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.avg))
        )

    def scores(self, query: str) -> dict[int, float]:
        """Every section that shares a word with the query, and its score. Found by walking the
        postings of each query word, so sections with nothing in common are never touched."""
        found: dict[int, float] = {}
        for term in set(tokens(query)):
            for n, tf in self.postings.get(term, ()):
                found[n] = found.get(n, 0.0) + self._bm25(term, tf, self.lengths[n])
        return found

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """The k best sections for a query. Sections that share no words with it are left out."""
        hits = [Hit(self.sections[n], score) for n, score in self.scores(query).items() if score > 0]
        hits.sort(key=lambda h: (-h.score, h.section.id))
        return hits[:k]

    def get(self, section_id: str) -> Section | None:
        return self.by_id.get(section_id)

    def why(self, query: str, section: Section, width: int = 90) -> str:
        """One line from the section showing why it matched: the line with the most query words,
        with those words in CAPITALS. Makes a wrong hit obvious at a glance."""
        wanted = set(tokens(query))
        best, best_score = "", 0
        for line in section.text.splitlines():
            score = sum(1 for word in tokens(line) if word in wanted)
            if score > best_score:
                best, best_score = line.strip(), score
        if not best:
            return ""
        marked = re.sub(
            r"[A-Za-z0-9]+",
            lambda m: m.group(0).upper() if stem(m.group(0).lower()) in wanted else m.group(0),
            best,
        )
        return marked if len(marked) <= width else marked[: width - 3].rstrip() + "..."
