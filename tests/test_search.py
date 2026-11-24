from pathlib import Path

import pytest

from runbook.docs import Section, load
from runbook.search import Index, stem, tokens

ROOT = Path(__file__).resolve().parent.parent


def section(i, title, text):
    return Section(f"d#{i}", "Doc", title, text)


SECTIONS = [
    section(0, "Disk full", "Delete old logs when the disk fills up."),
    section(1, "Certificate expired", "Renew the TLS certificate and restart nginx."),
    section(2, "Database slow", "Check slow queries and missing indexes in the database."),
]


def test_stemming_folds_word_forms_together():
    assert stem("alerts") == stem("alerting") == stem("alerted") == "alert"
    assert stem("is") == "is"  # too short to strip


@pytest.mark.parametrize(
    ("base", "inflected"),
    [
        ("page", "paged"),
        ("page", "pages"),
        ("page", "paging"),
        ("fire", "fired"),
        ("fire", "fires"),
        ("time", "timed"),
        ("time", "times"),
        ("cause", "causes"),
        ("failure", "failures"),
        ("minute", "minutes"),
        ("mitigate", "mitigated"),
    ],
)
def test_a_silent_e_survives_its_own_inflections(base, inflected):
    # Chopping the suffix off "paging" leaves "pag", which matches neither "page" nor "pages".
    assert stem(base) == stem(inflected) == base


def test_a_doubled_consonant_is_reduced():
    assert stem("running") == stem("runs") == "run"
    assert stem("stopped") == stem("stops") == "stop"


@pytest.mark.parametrize(
    ("one", "other"),
    [
        ("second", "seconds"),  # no e to restore
        ("checkout", "checkouts"),
        ("burn", "burning"),
        ("slo", "slos"),
    ],
)
def test_words_without_a_silent_e_are_left_alone(one, other):
    assert stem(one) == stem(other) == one


def test_double_s_and_short_words_are_not_stripped():
    assert stem("business") == "business"
    assert stem("miss") == stem("misses") == stem("missing") == "miss"
    assert stem("slos") == "slo"
    assert stem("its") == "its"  # len 3, too short to lose the s


def test_tokens_drop_stopwords_and_punctuation():
    assert tokens("What is the p95 latency?") == ["p95", "latency"]


def test_the_matching_section_ranks_first():
    hits = Index(SECTIONS).search("TLS certificate expired")
    assert hits[0].section.id == "d#1"


def test_sections_with_no_shared_words_are_left_out():
    assert Index(SECTIONS).search("kubernetes pod crashloop") == []


def test_k_limits_the_results():
    assert len(Index(SECTIONS).search("the disk certificate database", k=2)) == 2


def test_postings_list_only_the_sections_holding_a_term():
    index = Index(SECTIONS)
    assert [n for n, _ in index.postings["nginx"]] == [1]
    assert sorted(n for n, _ in index.postings["disk"]) == [0]
    assert "kafka" not in index.postings


def test_scoring_touches_only_sections_sharing_a_word():
    index = Index(SECTIONS)
    assert sorted(index.scores("TLS certificate")) == [1]
    assert index.scores("kubernetes crashloop") == {}


def test_get_finds_a_section_by_id():
    index = Index(SECTIONS)
    assert index.get("d#2").title == "Database slow"
    assert index.get("d#9") is None


def test_why_shows_the_matching_line_with_the_words_in_capitals():
    index = Index(load(ROOT / "corpus"))
    section = index.get("checkout-availability#2")
    line = index.why("chaos switch left on", section)
    assert "CHAOS" in line and "SWITCH" in line
    assert "kubectl" in line.lower()
    assert len(line) <= 90


def test_why_marks_words_through_the_stemmer():
    section = Section("x#0", "Doc", "Alerts", "The alert fired twice.\n")
    assert Index([section]).why("alerting", section) == "The ALERT fired twice."


def test_why_is_empty_when_nothing_matches():
    section = Section("x#0", "Doc", "T", "nothing here")
    assert Index([section]).why("kafka", section) == ""
    assert Index([Section("x#0", "Doc", "T", "")]).why("kafka", Section("x#0", "Doc", "T", "")) == ""


def test_coverage_counts_how_much_of_the_question_a_section_holds():
    index = Index(SECTIONS)
    assert index.coverage("disk full logs", [SECTIONS[0]]) == 1.0
    assert index.coverage("disk kafka", [SECTIONS[0]]) == 0.5
    assert index.coverage("kafka topic", [SECTIONS[0]]) == 0.0


def test_coverage_of_several_sections_is_their_union():
    index = Index(SECTIONS)
    assert index.coverage("disk certificate", [SECTIONS[0]]) == 0.5
    assert index.coverage("disk certificate", [SECTIONS[0], SECTIONS[1]]) == 1.0


def test_coverage_of_a_question_with_no_real_words_is_zero():
    index = Index(SECTIONS)
    assert index.coverage("the and of", SECTIONS) == 0.0


def test_coverage_ignores_sections_from_another_index():
    index = Index(SECTIONS)
    assert index.coverage("disk full", [Section("other#0", "D", "T", "disk full")]) == 0.0


def test_breakdown_names_the_words_that_earned_the_score():
    index = Index(SECTIONS)
    parts = index.breakdown("disk full logs", SECTIONS[0])
    assert {term for term, _ in parts} == {"disk", "full", "log"}
    assert [score for _, score in parts] == sorted((s for _, s in parts), reverse=True)
    # approx, not ==: the two sums add the same floats in a different order.
    assert sum(score for _, score in parts) == pytest.approx(index.search("disk full logs")[0].score)


def test_breakdown_leaves_out_words_the_section_does_not_have():
    index = Index(SECTIONS)
    assert [term for term, _ in index.breakdown("disk kafka", SECTIONS[0])] == ["disk"]
    assert index.breakdown("kafka topic", SECTIONS[0]) == []


def test_breakdown_of_a_section_from_another_index_is_empty():
    index = Index(SECTIONS)
    assert index.breakdown("disk", Section("other#0", "D", "T", "disk")) == []
