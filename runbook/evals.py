"""Retrieval eval: for each question, does a section that answers it rank in the top k?

The agent can only cite what search finds, so this is the number that matters first. It needs no
LLM, so CI runs it on every change and fails if the score drops below the threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from runbook.search import Index


@dataclass(frozen=True)
class EvalResult:
    total: int
    hits: int
    misses: list[tuple[str, list[str], list[str]]]  # question, expected ids, what search returned

    @property
    def score(self) -> float:
        return self.hits / self.total if self.total else 0.0


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


def run(index: Index, questions: list[dict], k: int = 3) -> EvalResult:
    hits, misses = 0, []
    for q in questions:
        found = [h.section.id for h in index.search(q["question"], k=k)]
        if set(found) & set(q["expect"]):
            hits += 1
        else:
            misses.append((q["question"], q["expect"], found))
    return EvalResult(len(questions), hits, misses)


def report(result: EvalResult, k: int) -> str:
    lines = [f"Retrieval hit@{k}: {result.hits}/{result.total} ({result.score:.0%})"]
    for question, expected, found in result.misses:
        lines.append(f"  MISS {question}\n       expected {expected}, got {found}")
    return "\n".join(lines)
