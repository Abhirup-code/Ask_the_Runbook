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


GUARD_SECTIONS = [Section("s#0", "SLOs", "SLO", "Availability is 99.5% over 30 days.")]


def test_the_guard_eval_counts_wrong_verdicts():
    cases = [
        {"expect": "pass", "why": "grounded", "seen": ["s#0"], "answer": "It is 99.5% [s#0]."},
        {"expect": "fail", "why": "made up", "seen": ["s#0"], "answer": "It is 99.9% [s#0]."},
        {"expect": "pass", "why": "wrongly expected to pass", "seen": ["s#0"], "answer": "It is 42 [s#0]."},
    ]
    result = evals.run_guard(GUARD_SECTIONS, cases)
    assert (result.total, len(result.wrong), result.ok) == (3, 1, False)
    report = evals.guard_report(result)
    assert "Guard: 2/3" in report and "WRONG wrongly expected to pass" in report


def test_a_guard_case_must_name_real_sections():
    case = {"expect": "pass", "why": "typo", "seen": ["s#9"], "answer": "x [s#9]"}
    with pytest.raises(ValueError, match="no section s#9"):
        evals.run_guard(GUARD_SECTIONS, [case])


def test_bad_guard_case_lines_are_rejected(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text('\n{"expect": "maybe", "seen": ["s#0"], "answer": "x"}\n')
    with pytest.raises(ValueError, match="g.jsonl:2"):
        evals.load_guard_cases(path)


def test_the_guard_handles_every_real_case():
    result = evals.run_guard(load(ROOT / "corpus"), evals.load_guard_cases(ROOT / "evals/guard_cases.jsonl"))
    assert result.ok, evals.guard_report(result)
