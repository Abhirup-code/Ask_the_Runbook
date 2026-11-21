"""Two evals that need no LLM, so CI runs both on every change.

Retrieval: for each question, does a section that answers it rank in the top k? The agent can only
cite what search finds, so this is the number that matters first.

Guard: hand-written answers, good and bad, checked against the real runbooks. Every bad answer must
be blocked and every good one let through. The bad ones are the mistakes models really make:
made-up thresholds, invented citation ids, numbers taken from a section they didn't cite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from runbook.docs import Section
from runbook.guard import check
from runbook.search import Index


@dataclass(frozen=True)
class EvalResult:
    total: int
    hits: int
    misses: list[tuple[str, list[str], list[str]]]  # question, expected ids, what search returned
    ranks: tuple[int | None, ...] = ()  # where the first right section ranked, per question (1 = top)
    docs: tuple[str, ...] = ()  # which document each question's answer lives in

    @property
    def score(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def hit_at(self, k: int) -> float:
        found = sum(1 for r in self.ranks if r is not None and r <= k)
        return found / len(self.ranks) if self.ranks else 0.0

    @property
    def mrr(self) -> float:
        """Mean reciprocal rank: 1.0 if the right section is always first, 0.5 if always second."""
        return sum(1 / r for r in self.ranks if r) / len(self.ranks) if self.ranks else 0.0


def load_questions(path: str | Path) -> list[dict]:
    """Every question in the file. An empty expect list means the runbooks do not cover it, and
    the right behaviour is to find nothing worth offering."""
    questions = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("question") or not isinstance(item.get("expect"), list):
            raise ValueError(f"{path}:{n}: each line needs a question and an expect list")
        questions.append(item)
    return questions


def answerable(questions: list[dict]) -> list[dict]:
    return [q for q in questions if q["expect"]]


def unanswerable(questions: list[dict]) -> list[dict]:
    return [q for q in questions if not q["expect"]]


DEPTH = 10  # how far down the ranking to look for the rank metrics


def run(index: Index, questions: list[dict], k: int = 3) -> EvalResult:
    hits, misses, ranks, docs = 0, [], [], []
    for q in questions:
        docs.append(q["expect"][0].split("#")[0])
        ranked = [h.section.id for h in index.search(q["question"], k=max(k, DEPTH))]
        rank = next((n for n, i in enumerate(ranked, 1) if i in q["expect"]), None)
        ranks.append(rank)
        if rank is not None and rank <= k:
            hits += 1
        else:
            misses.append((q["question"], q["expect"], ranked[:k]))
    return EvalResult(len(questions), hits, misses, tuple(ranks), tuple(docs))


def by_document(result: EvalResult, k: int = 3) -> list[str]:
    """One line per runbook: how many of its questions were found. A whole document scoring
    badly points at the document, not at the search."""
    if len(result.docs) != len(result.ranks):
        return []
    per_doc: dict[str, list[bool]] = {}
    for doc, rank in zip(result.docs, result.ranks, strict=True):
        per_doc.setdefault(doc, []).append(rank is not None and rank <= k)
    width = max((len(d) for d in per_doc), default=0)
    return [f"{doc:<{width}}  {sum(found)}/{len(found)}" for doc, found in sorted(per_doc.items())]


def report(result: EvalResult, k: int) -> str:
    lines = [
        f"Retrieval hit@{k}: {result.hits}/{result.total} ({result.score:.0%})",
        f"  hit@1 {result.hit_at(1):.0%}, hit@5 {result.hit_at(5):.0%}, MRR {result.mrr:.2f}",
    ]
    lines += [f"  {line}" for line in by_document(result, k)]
    for question, expected, found in result.misses:
        lines.append(f"  MISS {question}\n       expected {expected}, got {found}")
    return "\n".join(lines)


@dataclass(frozen=True)
class TopScores:
    """What the best hit scored, for a set of questions. Answerable and unanswerable questions
    scored side by side show whether a score threshold could tell them apart at all."""

    answerable: tuple[float, ...]
    unanswerable: tuple[tuple[str, float, str], ...]  # question, top score, top section id

    @property
    def lowest_answerable(self) -> float:
        return min(self.answerable, default=0.0)

    @property
    def highest_unanswerable(self) -> float:
        return max((score for _, score, _ in self.unanswerable), default=0.0)

    @property
    def separable(self) -> bool:
        """True when every unanswerable question scores below every answerable one, so a single
        threshold could reject the first group without losing any of the second."""
        return bool(self.answerable) and self.highest_unanswerable < self.lowest_answerable


def top_scores(index: Index, questions: list[dict]) -> TopScores:
    def top(question: str) -> tuple[float, str]:
        hits = index.search(question, k=1)
        return (hits[0].score, hits[0].section.id) if hits else (0.0, "")

    return TopScores(
        tuple(top(q["question"])[0] for q in answerable(questions)),
        tuple((q["question"], *top(q["question"])) for q in unanswerable(questions)),
    )


def scores_report(scores: TopScores) -> str:
    lines = [
        f"Top score: answerable questions bottom out at {scores.lowest_answerable:.1f}, "
        f"the {len(scores.unanswerable)} uncovered questions peak at {scores.highest_unanswerable:.1f}"
    ]
    for question, score, section_id in sorted(scores.unanswerable, key=lambda t: -t[1]):
        lines.append(f"  {score:5.1f}  [{section_id or '-'}]  {question}")
    lines.append(
        "  A single score threshold would separate them."
        if scores.separable
        else "  The two overlap, so a score threshold alone cannot separate them."
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class AbstainResult:
    """What the coverage rule does to both halves of the question set: how many real answers it
    still lets through, and how many uncovered questions it turns away before the model is asked."""

    kept: int
    answerable: int
    rejected: int
    unanswerable: int
    let_through: tuple[tuple[str, float], ...] = ()  # uncovered questions that got past, and by how much

    @property
    def costs_nothing(self) -> bool:
        return self.kept == self.answerable


def abstain_check(index: Index, questions: list[dict], min_coverage: float, k: int = 4) -> AbstainResult:
    def covered(question: str) -> float:
        return index.coverage(question, [h.section for h in index.search(question, k=k)])

    kept = sum(1 for q in answerable(questions) if covered(q["question"]) >= min_coverage)
    through = tuple(
        (q["question"], covered(q["question"]))
        for q in unanswerable(questions)
        if covered(q["question"]) >= min_coverage
    )
    totals = (len(answerable(questions)), len(unanswerable(questions)))
    return AbstainResult(kept, totals[0], totals[1] - len(through), totals[1], through)


def abstain_report(result: AbstainResult, min_coverage: float) -> str:
    lines = [
        f"Abstain at {min_coverage:.0%} coverage: "
        f"{result.kept}/{result.answerable} answerable questions still get through, "
        f"{result.rejected}/{result.unanswerable} uncovered ones are turned away"
    ]
    for question, covered in sorted(result.let_through, key=lambda t: -t[1]):
        lines.append(f"  through at {covered:.0%}  {question}")
    if not result.costs_nothing:
        lines.append(f"  WARNING {result.answerable - result.kept} answerable question(s) now rejected")
    return "\n".join(lines)


@dataclass(frozen=True)
class GuardResult:
    total: int
    wrong: list[tuple[str, str, list[str]]]  # why, expected outcome, the guard's problems

    @property
    def ok(self) -> bool:
        return not self.wrong


def load_guard_cases(path: str | Path) -> list[dict]:
    cases = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("expect") not in ("pass", "fail") or not item.get("seen") or "answer" not in item:
            raise ValueError(f"{path}:{n}: each line needs expect (pass or fail), seen, and answer")
        cases.append(item)
    return cases


def run_guard(sections: list[Section], cases: list[dict]) -> GuardResult:
    by_id = {s.id: s for s in sections}
    wrong = []
    for case in cases:
        missing = [i for i in case["seen"] if i not in by_id]
        if missing:
            raise ValueError(f"{case['why']}: no section {', '.join(missing)} in the runbooks")
        verdict = check(case["answer"], {i: by_id[i] for i in case["seen"]})
        if verdict.ok != (case["expect"] == "pass"):
            wrong.append((case.get("why", case["answer"]), case["expect"], verdict.problems))
    return GuardResult(len(cases), wrong)


def guard_report(result: GuardResult) -> str:
    right = result.total - len(result.wrong)
    lines = [f"Guard: {right}/{result.total} answers handled correctly"]
    for why, expected, problems in result.wrong:
        lines.append(f"  WRONG {why}\n        expected {expected}, guard said {problems or 'ok'}")
    return "\n".join(lines)
