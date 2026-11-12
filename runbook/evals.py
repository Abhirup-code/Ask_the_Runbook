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
    questions = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not item.get("question") or not item.get("expect"):
            raise ValueError(f"{path}:{n}: each line needs a question and a non-empty expect list")
        questions.append(item)
    return questions


DEPTH = 10  # how far down the ranking to look for the rank metrics


def run(index: Index, questions: list[dict], k: int = 3) -> EvalResult:
    hits, misses, ranks = 0, [], []
    for q in questions:
        ranked = [h.section.id for h in index.search(q["question"], k=max(k, DEPTH))]
        rank = next((n for n, i in enumerate(ranked, 1) if i in q["expect"]), None)
        ranks.append(rank)
        if rank is not None and rank <= k:
            hits += 1
        else:
            misses.append((q["question"], q["expect"], ranked[:k]))
    return EvalResult(len(questions), hits, misses, tuple(ranks))


def report(result: EvalResult, k: int) -> str:
    lines = [
        f"Retrieval hit@{k}: {result.hits}/{result.total} ({result.score:.0%})",
        f"  hit@1 {result.hit_at(1):.0%}, hit@5 {result.hit_at(5):.0%}, MRR {result.mrr:.2f}",
    ]
    for question, expected, found in result.misses:
        lines.append(f"  MISS {question}\n       expected {expected}, got {found}")
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
