from pathlib import Path

from runbook.agent import NOT_FOUND, Agent
from runbook.docs import Section, load
from runbook.llm import Reply, ToolCall
from runbook.search import Index
from tests.fakes import ScriptedLLM, read, say, search

INDEX = Index(load(Path(__file__).parent.parent / "corpus"))


def test_search_read_and_answer_with_a_citation():
    llm = ScriptedLLM(
        search("checkout 500 errors"),
        read("checkout-availability#2"),
        say("Check the chaos switch and turn it off with `scripts/chaos.sh 0 0` [checkout-availability#2]."),
    )
    result = Agent(INDEX, llm).ask("Checkout is failing, what do I check?")
    assert result.answered
    assert [s.id for s in result.citations] == ["checkout-availability#2"]
    assert result.trace == [
        "search_docs({'query': 'Checkout is failing, what do I check?'}) [automatic]",
        "search_docs({'query': 'checkout 500 errors'})",
        "read_section({'id': 'checkout-availability#2'})",
    ]
    # The model saw the tool output: search previews, then the full section.
    tool_messages = [m for m in llm.requests[-1] if m["role"] == "tool"]
    assert "[checkout-availability#2]" in tool_messages[0]["content"]
    assert "rollout undo" in tool_messages[1]["content"]


def test_a_bad_answer_gets_one_retry_with_the_reasons():
    llm = ScriptedLLM(
        search("availability objective"),
        say("The objective is 99.99% [slos#0]."),
        say("The objective is 99.5% [slos#0]."),
    )
    result = Agent(INDEX, llm).ask("What is the availability objective?")
    assert result.answered
    feedback = llm.requests[-1][-1]
    assert feedback["role"] == "user" and "99.99" in feedback["content"]


def test_two_bad_answers_mean_no_answer_and_suggestions():
    llm = ScriptedLLM(search("objective"), say("It is 99.99% [slos#0]."), say("Still 99.99% [slos#0]."))
    result = Agent(INDEX, llm).ask("What is the availability objective?")
    assert not result.answered
    assert "99.99" in result.problems[0]
    assert result.suggestions and result.suggestions[0].id == "slos#0"


def test_citing_a_section_that_was_never_retrieved_is_caught():
    llm = ScriptedLLM(say("Restart it [slos#2]."), say("Restart it [slos#2]."))
    result = Agent(INDEX, llm).ask("Checkout is down")
    assert not result.answered
    assert "never retrieved" in result.problems[0]


def test_not_found_is_an_honest_no():
    llm = ScriptedLLM(search("kafka consumer lag"), say(NOT_FOUND), say(NOT_FOUND))
    result = Agent(INDEX, llm).ask("How do I fix Kafka consumer lag?")
    assert not result.answered
    assert result.problems == ["The runbooks don't cover this."]


def test_unknown_tools_and_ids_are_explained_to_the_model():
    llm = ScriptedLLM(
        Reply(tool_calls=[ToolCall("delete_prod", {}), ToolCall("read_section", {"id": "nope#9"})]),
        say(NOT_FOUND),
    )
    Agent(INDEX, llm, min_coverage=0.0).ask("anything")
    outputs = [m["content"] for m in llm.requests[-1] if m["role"] == "tool"]
    assert "Unknown tool 'delete_prod'" in outputs[0]
    assert "No section with id 'nope#9'" in outputs[1]


def test_a_search_with_no_matches_says_so():
    llm = ScriptedLLM(search("zzzz qqqq"), say(NOT_FOUND))
    Agent(INDEX, llm, min_coverage=0.0).ask("zzzz")
    assert llm.requests[-1][-1]["content"] == "No matching sections."


def test_the_agent_stops_after_max_steps():
    llm = ScriptedLLM(*[search("checkout")] * 3)
    result = Agent(INDEX, llm, max_steps=3, min_coverage=0.0).ask("loop forever")
    assert not result.answered
    assert result.problems == ["Ran out of steps before answering."]


def test_citing_a_bracketed_id_in_read_section_works():
    llm = ScriptedLLM(read("[slos#1]"), say("A burn rate of 14.4 uses the budget in about 2 days [slos#1]."))
    assert Agent(INDEX, llm).ask("What does 14.4 mean?").answered


def test_the_top_results_come_back_in_full_so_a_small_model_sees_the_answer():
    """A real 3B model searched "500s after deploy", saw a preview cut off right before the
    rollback command, and gave up. The top results now come back whole."""
    llm = ScriptedLLM(search("500s after deploy"), say(NOT_FOUND), say(NOT_FOUND))
    Agent(INDEX, llm).ask("Checkout is throwing 500s after a deploy")
    results = [m for m in llm.requests[1] if m["role"] == "tool"][-1]["content"]
    assert "kubectl -n checkout rollout undo deployment/checkout" in results


def test_lower_results_are_cut_and_say_how_to_read_more():
    llm = ScriptedLLM(search("checkout"), say(NOT_FOUND), say(NOT_FOUND))
    Agent(INDEX, llm, results=6).ask("checkout")
    results = [m for m in llm.requests[1] if m["role"] == "tool"][-1]["content"]
    assert "(call read_section for the rest)" in results


def test_the_question_is_searched_before_the_model_answers():
    """A real 1.5B model never called a tool: it answered "not found" or made up ids like
    search_docs#1. Searching first puts the right sections in front of it anyway."""
    llm = ScriptedLLM(
        say(
            "Roll back with `kubectl -n checkout rollout undo deployment/checkout` [checkout-availability#2]."
        )
    )
    result = Agent(INDEX, llm).ask("Checkout is throwing 500s after a deploy. What do I do?")
    assert result.answered
    question = llm.requests[0][-1]
    assert question["role"] == "user"
    assert question["content"].startswith("Runbook sections:")
    assert "rollout undo deployment/checkout" in question["content"]
    assert question["content"].endswith("Question: Checkout is throwing 500s after a deploy. What do I do?")
    assert result.trace[0].endswith("[automatic]")


def test_made_up_citation_ids_are_still_blocked():
    llm = ScriptedLLM(say("Burn rate 14.4 [search_docs#1]."), say("Burn rate 14.4 [search_docs#2]."))
    result = Agent(INDEX, llm).ask("What does a burn rate of 14.4 mean?")
    assert not result.answered
    assert "never retrieved: search_docs#2" in result.problems[0]


def test_a_reflex_not_found_gets_one_nudge_to_check_the_sections():
    """A real 1.5B model replied NOT_FOUND even with the answer in front of it."""
    llm = ScriptedLLM(
        say(NOT_FOUND),
        say("A burn rate of 14.4 uses the budget in about 2 days [slos#1]."),
    )
    result = Agent(INDEX, llm).ask("What does a burn rate of 14.4 mean?")
    assert result.answered
    nudge = llm.requests[1][-1]["content"]
    assert nudge.startswith("Check the runbook sections again: [slos#1]")


def test_not_found_twice_is_accepted():
    llm = ScriptedLLM(say(NOT_FOUND), say(NOT_FOUND))
    result = Agent(INDEX, llm).ask("What does a burn rate of 14.4 mean?")
    assert not result.answered and len(llm.requests) == 2


def test_not_found_with_nothing_retrieved_is_accepted_at_once():
    # With no threshold, a question search cannot match reaches the model, and its first
    # NOT_FOUND is taken at face value: there are no sections for it to check again.
    llm = ScriptedLLM(say(NOT_FOUND))
    assert not Agent(INDEX, llm, min_coverage=0.0).ask("zzzz qqqq").answered
    assert len(llm.requests) == 1


def test_nothing_retrieved_means_the_model_is_never_asked():
    llm = ScriptedLLM()
    assert not Agent(INDEX, llm).ask("zzzz qqqq").answered
    assert llm.requests == []


def test_an_uncovered_question_is_turned_away_before_the_model_is_asked():
    llm = ScriptedLLM()  # no replies: asking the model at all would raise IndexError
    result = Agent(INDEX, llm).ask("How do I rotate the TLS certificate for the checkout endpoint?")
    assert not result.answered
    assert result.problems == ["The runbooks don't cover this."]
    assert llm.requests == []
    assert "abstain:" in result.trace[-1] and "covered" in result.trace[-1]


def test_a_covered_question_passes_the_abstain_rule():
    llm = ScriptedLLM(say("Turn the chaos switch off with `scripts/chaos.sh 0 0` [checkout-availability#2]."))
    result = Agent(INDEX, llm).ask("Is the chaos switch left on for checkout?")
    assert result.answered
    assert len(llm.requests) == 1


def test_the_abstain_threshold_can_be_turned_off():
    llm = ScriptedLLM(say(NOT_FOUND), say(NOT_FOUND))
    result = Agent(INDEX, llm, min_coverage=0.0).ask("How do I add someone to the PagerDuty schedule?")
    assert not result.answered
    assert llm.requests, "with no threshold the model should still be asked"


LONG = Index(
    [
        Section("big#0", "Big", "Checkout errors one", "checkout errors " + "alpha " * 300),
        Section("big#1", "Big", "Checkout errors two", "checkout errors " + "beta " * 300),
        Section("big#2", "Big", "Checkout errors three", "checkout errors " + "gamma " * 300),
    ]
)


def test_long_sections_are_trimmed_to_the_budget():
    llm = ScriptedLLM(say(NOT_FOUND), say(NOT_FOUND))
    Agent(LONG, llm, budget=900, min_coverage=0.0).ask("checkout errors")
    sent = llm.requests[0][-1]["content"]
    assert len(sent) < 1300, sent[:200]
    assert "call read_section for the rest" in sent


def test_the_sections_that_did_not_fit_are_counted_not_hidden():
    llm = ScriptedLLM(say(NOT_FOUND), say(NOT_FOUND))
    Agent(LONG, llm, budget=600, min_coverage=0.0).ask("checkout errors")
    sent = llm.requests[0][-1]["content"]
    assert "more section(s) matched" in sent


def test_a_section_trimmed_away_cannot_be_cited():
    llm = ScriptedLLM(say("Run it [big#2]."), say(NOT_FOUND), say(NOT_FOUND))
    result = Agent(LONG, llm, budget=600, min_coverage=0.0).ask("checkout errors")
    assert not result.answered
    said = " ".join(m["content"] for r in llm.requests for m in r if m["role"] == "user")
    assert "never retrieved" in said and "big#2" in said


def test_the_top_section_is_trimmed_but_never_dropped():
    llm = ScriptedLLM(say(NOT_FOUND), say(NOT_FOUND))
    Agent(LONG, llm, budget=50, min_coverage=0.0).ask("checkout errors")
    sent = llm.requests[0][-1]["content"]
    assert "[big#0]" in sent
