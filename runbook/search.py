"""Keyword search over runbook sections with BM25, the ranking classic search engines use.

No embedding model to download and nothing to host: it runs anywhere Python runs, and the
retrieval eval in evals/ keeps it honest.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from runbook.docs import Section

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i if in is it its me my of on or so that the then this "
    "to up was what when where which who why will with you your".split()
)


VOWELS = "aeiou"


def _is_vowel(word: str, i: int) -> bool:
    """y counts as a vowel only after a consonant, as in "runny" but not "yes"."""
    return word[i] in VOWELS or (word[i] == "y" and i > 0 and word[i - 1] not in VOWELS)


def _has_vowel(word: str) -> bool:
    return any(_is_vowel(word, i) for i in range(len(word)))


def _measure(word: str) -> int:
    """Porter's m: how many vowel-run then consonant-run pairs a stem has."""
    shape = "".join("v" if _is_vowel(word, i) else "c" for i in range(len(word)))
    while "vv" in shape or "cc" in shape:
        shape = shape.replace("vv", "v").replace("cc", "c")
    return shape.count("vc")


def _cvc(word: str) -> bool:
    """Ends consonant, vowel, consonant, with the last not w, x or y: "pag", "fir", "tim"."""
    if len(word) < 3 or _is_vowel(word, -1) or not _is_vowel(word, -2) or _is_vowel(word, -3):
        return False
    return word[-1] not in "wxy"


def stem(word: str) -> str:
    """Steps 1a and 1b of the Porter stemmer: "alerts", "alerting" and "alerted" become "alert".

    The point of the extra machinery over chopping suffixes off is the silent e. Cutting "ing"
    from "paging" leaves "pag", which matches neither "page" nor "pages", so a question about
    being paged found nothing in a runbook that talks about pages. Restoring the e where Porter
    says to conflates 23 of 23 inflected pairs in this corpus, against 8 before, and it stops
    "business" being mangled into "busines" on the way.
    """
    # Step 1a: plurals.
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif not word.endswith("ss") and word.endswith("s") and len(word) > 3:
        word = word[:-1]

    # Step 1b: -eed, -ed and -ing.
    stripped = None
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    elif word.endswith("ed") and _has_vowel(word[:-2]):
        stripped = word[:-2]
    elif word.endswith("ing") and _has_vowel(word[:-3]):
        stripped = word[:-3]

    if stripped is not None and len(stripped) >= 2:
        word = stripped
        if word.endswith(("at", "bl", "iz")):
            word += "e"
        elif word[-1] == word[-2] and word[-1] not in "lsz":
            word = word[:-1]  # "running" to "run", not "runn"
        elif _measure(word) == 1 and _cvc(word):
            word += "e"  # "pag" to "page", "fir" to "fire"
    return word


def words(text: str) -> list[tuple[str, str]]:
    """Each meaningful word as it was written, paired with its stem. Keeping the original lets a
    message name the words someone actually typed rather than stems like "pag"."""
    return [(w, stem(w)) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def tokens(text: str) -> list[str]:
    return [stemmed for _, stemmed in words(text)]


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
        self.index_of = {s.id: n for n, s in enumerate(sections)}

    def coverage(self, query: str, sections: Iterable[Section]) -> float:
        """What fraction of the question's distinct words appear in these sections.

        A high BM25 score can come from one common word repeated often, so the score alone does
        not say whether a result is about the question. Coverage does: a section answering "which
        Kafka topic does checkout publish to?" would contain "kafka" and "topic", not just
        "checkout". Found through the postings, so no extra copy of the terms is kept.
        """
        q = set(tokens(query))
        if not q:
            return 0.0
        here = {self.index_of[s.id] for s in sections if s.id in self.index_of}
        found = sum(1 for term in q if any(n in here for n, _ in self.postings.get(term, ())))
        return found / len(q)

    def missing_words(self, query: str, sections: Iterable[Section]) -> list[str]:
        """The question's words that appear nowhere in these sections, as the asker wrote them.

        This is the other half of coverage. Coverage says how much was found; this says what was
        not, which is the part worth telling someone when the answer is that nobody knows.
        """
        here = {self.index_of[s.id] for s in sections if s.id in self.index_of}
        absent, seen = [], set()
        for word, stemmed in words(query):
            if stemmed in seen:
                continue
            seen.add(stemmed)
            if not any(n in here for n, _ in self.postings.get(stemmed, ())):
                absent.append(word)
        return absent

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

    def breakdown(self, query: str, section: Section) -> list[tuple[str, float]]:
        """Which of the question's words earned this section its score, and how much each was
        worth, biggest first. A ranking nobody can explain is a ranking nobody can fix."""
        n = self.index_of.get(section.id)
        if n is None:
            return []
        scored = [
            (term, self._bm25(term, tf, self.lengths[n]))
            for term in set(tokens(query))
            for i, tf in self.postings.get(term, ())
            if i == n
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored

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
