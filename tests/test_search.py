from pathlib import Path

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


def test_tokens_drop_stopwords_and_punctuation():
    assert tokens("What is the p95 latency?") == ["p95", "latency"]


def test_the_matching_section_ranks_first():
    hits = Index(SECTIONS).search("TLS certificate expired")
    assert hits[0].section.id == "d#1"


def test_sections_with_no_shared_words_are_left_out():
    assert Index(SECTIONS).search("kubernetes pod crashloop") == []


def test_k_limits_the_results():
    assert len(Index(SECTIONS).search("the disk certificate database", k=2)) == 2


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
