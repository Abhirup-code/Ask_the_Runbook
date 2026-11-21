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
    assert result.ranks == (1, None)


def test_rank_metrics():
    result = evals.EvalResult(4, 2, [], ranks=(1, 2, 4, None))
    assert (result.hit_at(1), result.hit_at(3), result.hit_at(5)) == (0.25, 0.5, 0.75)
    assert result.mrr == (1 + 0.5 + 0.25) / 4
    assert "hit@1 25%, hit@5 75%, MRR 0.44" in evals.report(result, 3)
    empty = evals.EvalResult(0, 0, [])
    assert (empty.hit_at(1), empty.mrr) == (0.0, 0.0)


def test_an_empty_eval_scores_zero():
    assert evals.run(Index([Section("d#0", "D", "T", "x")]), []).score == 0.0


def test_bad_eval_lines_are_rejected(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "x"}\n')
    with pytest.raises(ValueError, match="q.jsonl:1"):
        evals.load_questions(path)
    path.write_text('{"question": "x", "expect": "d#0"}\n')
    with pytest.raises(ValueError, match="q.jsonl:1"):
        evals.load_questions(path)


def test_an_empty_expect_list_means_the_runbooks_do_not_cover_it(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "covered", "expect": ["d#0"]}\n{"question": "not", "expect": []}\n')
    questions = evals.load_questions(path)
    assert len(questions) == 2
    assert [q["question"] for q in evals.answerable(questions)] == ["covered"]
    assert [q["question"] for q in evals.unanswerable(questions)] == ["not"]


def test_blank_lines_are_ignored(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('\n{"question": "x", "expect": ["d#0"]}\n\n')
    assert len(evals.load_questions(path)) == 1


def test_every_expected_section_exists():
    ids = {s.id for s in load(ROOT / "corpus")}
    for q in evals.load_questions(ROOT / "evals/questions.jsonl"):
        assert set(q["expect"]) <= ids, q["question"]


def test_retrieval_on_the_real_corpus_stays_above_the_bar():
    questions = evals.answerable(evals.load_questions(ROOT / "evals/questions.jsonl"))
    result = evals.run(Index(load(ROOT / "corpus")), questions)
    assert result.score >= THRESHOLD, evals.report(result, 3)


def test_the_real_corpus_has_questions_it_cannot_answer():
    questions = evals.load_questions(ROOT / "evals/questions.jsonl")
    assert len(evals.unanswerable(questions)) >= 8


SCORE_SECTIONS = [
    Section("d#0", "Doc", "Disk full", "Delete old logs when the disk fills up."),
    Section("d#1", "Doc", "TLS", "Renew the certificate."),
]


def test_top_scores_separate_when_uncovered_questions_score_lower():
    index = Index(SCORE_SECTIONS)
    scores = evals.top_scores(
        index,
        [{"question": "disk full logs", "expect": ["d#0"]}, {"question": "kafka topic", "expect": []}],
    )
    assert scores.highest_unanswerable == 0.0
    assert scores.lowest_answerable > 0.0
    assert scores.separable
    report = evals.scores_report(scores)
    assert "kafka topic" in report and "would separate them" in report


def test_top_scores_report_says_so_when_the_two_overlap():
    index = Index(SCORE_SECTIONS)
    scores = evals.top_scores(
        index,
        [{"question": "logs", "expect": ["d#0"]}, {"question": "disk full logs", "expect": []}],
    )
    assert not scores.separable
    assert "cannot separate them" in evals.scores_report(scores)


def test_top_scores_of_nothing_are_zero():
    scores = evals.TopScores((), ())
    assert (scores.lowest_answerable, scores.highest_unanswerable, scores.separable) == (0.0, 0.0, False)


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


def test_the_report_breaks_the_score_down_by_document():
    result = evals.EvalResult(3, 2, [], ranks=(1, None, 2), docs=("slos", "slos", "checkout-availability"))
    assert evals.by_document(result, 3) == ["checkout-availability  1/1", "slos                   1/2"]
    assert "slos                   1/2" in evals.report(result, 3)
    assert evals.by_document(evals.EvalResult(0, 0, [])) == []
