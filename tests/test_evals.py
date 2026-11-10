from pathlib import Path

import pytest

from runbook import evals
from runbook.docs import Section, load
from runbook.search import Index

ROOT = Path(__file__).parent.parent
THRESHOLD = 0.85  # CI's bar for the real corpus; see the README for the known misses


def test_hit_and_miss_are_counted():
    index = Index(
        [Section("d#0", "Doc", "Disk full", "Delete old logs."), Section("d#1", "Doc", "TLS", "Renew.")]
    )
    result = evals.run(
        index, [{"question": "disk full", "expect": ["d#0"]}, {"question": "disk", "expect": ["d#1"]}]
    )
    assert (result.hits, result.total, result.score) == (1, 2, 0.5)
    assert result.misses[0][0] == "disk"
    assert "MISS disk" in evals.report(result, 3)


def test_an_empty_eval_scores_zero():
    assert evals.run(Index([Section("d#0", "D", "T", "x")]), []).score == 0.0


def test_bad_eval_lines_are_rejected(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "x", "expect": []}\n')
    with pytest.raises(ValueError, match="q.jsonl:1"):
        evals.load_questions(path)


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('\n{"question": "x", "expect": ["d#0"]}\n\n')
    assert len(evals.load_questions(path)) == 1


def test_every_expected_section_exists():
    ids = {s.id for s in load(ROOT / "corpus")}
    for q in evals.load_questions(ROOT / "evals/questions.jsonl"):
        assert set(q["expect"]) <= ids, q["question"]


def test_retrieval_on_the_real_corpus_stays_above_the_bar():
    result = evals.run(Index(load(ROOT / "corpus")), evals.load_questions(ROOT / "evals/questions.jsonl"))
    assert result.score >= THRESHOLD, evals.report(result, 3)
