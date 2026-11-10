from runbook.docs import Section
from runbook.search import Index, stem, tokens


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
