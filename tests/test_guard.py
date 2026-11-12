from runbook.docs import Section
from runbook.guard import check, durations, numbers

SEEN = {
    "avail#2": Section(
        "avail#2", "Runbook", "2. Check the usual causes", "If error_rate is not 0, run scripts/chaos.sh 0 0."
    ),
    "slos#0": Section("slos#0", "SLOs", "The SLOs", "Availability objective is 99.5% over 30 days."),
    "alert#1": Section("alert#1", "Alerts", "Page", "Page when the 1h and 5m burn rates pass 14.4."),
}


def test_a_cited_grounded_answer_passes():
    verdict = check(
        "Turn chaos off with scripts/chaos.sh 0 0 [avail#2]. The objective is 99.5% [slos#0].", SEEN
    )
    assert verdict.ok, verdict.problems
    assert verdict.cited == ["avail#2", "slos#0"]


def test_no_citation_fails():
    verdict = check("Restart the pod.", SEEN)
    assert not verdict.ok
    assert "cites no section" in verdict.problems[0]


def test_citing_a_section_it_never_read_fails():
    verdict = check("Scale to 5 replicas [scaling#1].", SEEN)
    assert not verdict.ok
    assert "never retrieved: scaling#1" in verdict.problems[0]


def test_an_invented_number_fails_and_is_named():
    verdict = check("The objective is 99.9% over 30 days [slos#0].", SEEN)
    assert not verdict.ok
    assert "99.9" in verdict.problems[0] and "30" not in verdict.problems[0].split(":")[1]


def test_a_number_must_be_in_a_cited_section_not_just_any_section():
    verdict = check("The objective is 99.5% [avail#2].", SEEN)
    assert not verdict.ok


def test_list_markers_and_citation_numbers_are_not_claims():
    assert numbers("1. Check chaos [avail#2]\n2. Roll back") == set()
    assert numbers("Wait 5,000 ms or 1.5 s") == {"5000", "1.5"}


def test_empty_answer_fails():
    assert check("   ", SEEN).problems == ["The answer is empty."]


def test_the_same_time_span_in_another_unit_passes():
    verdict = check("Page when the burn rate over 60 minutes and 300 seconds passes 14.4 [alert#1].", SEEN)
    assert verdict.ok, verdict.problems
    assert check("Availability is measured over 720 hours [slos#0].", SEEN).ok


def test_a_different_time_span_still_fails():
    verdict = check("Page when the 2 hour burn rate passes 14.4 [alert#1].", SEEN)
    assert not verdict.ok
    assert "2" in verdict.problems[0]


def test_a_bare_number_is_not_a_time_span():
    assert not check("Page when the burn rate passes 60 [alert#1].", SEEN).ok


def test_durations_reads_units():
    assert durations("rate[5m], 1.5 s, 30 days, 250ms, 2 Weeks") == {
        "5": {300},
        "1.5": {1.5},
        "30": {2592000},
        "250": {0.25},
        "2": {1209600},
    }
