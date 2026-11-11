from pathlib import Path

from runbook.agent import NOT_FOUND, Agent
from runbook.docs import load
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


def test_answering_without_searching_is_caught():
    llm = ScriptedLLM(
        say("Restart it [checkout-availability#2]."), say("Restart it [checkout-availability#2].")
    )
    result = Agent(INDEX, llm).ask("Checkout is down")
    assert not result.answered
    assert "never retrieved" in result.problems[0]


def test_not_found_is_an_honest_no():
    llm = ScriptedLLM(search("kafka consumer lag"), say(NOT_FOUND))
    result = Agent(INDEX, llm).ask("How do I fix Kafka consumer lag?")
    assert not result.answered
    assert result.problems == ["The runbooks don't cover this."]


def test_unknown_tools_and_ids_are_explained_to_the_model():
    llm = ScriptedLLM(
        Reply(tool_calls=[ToolCall("delete_prod", {}), ToolCall("read_section", {"id": "nope#9"})]),
        say(NOT_FOUND),
    )
    Agent(INDEX, llm).ask("anything")
    outputs = [m["content"] for m in llm.requests[-1] if m["role"] == "tool"]
    assert "Unknown tool 'delete_prod'" in outputs[0]
    assert "No section with id 'nope#9'" in outputs[1]


def test_a_search_with_no_matches_says_so():
    llm = ScriptedLLM(search("zzzz qqqq"), say(NOT_FOUND))
    Agent(INDEX, llm).ask("zzzz")
    assert llm.requests[-1][-1]["content"] == "No matching sections."


def test_the_agent_stops_after_max_steps():
    llm = ScriptedLLM(*[search("checkout")] * 3)
    result = Agent(INDEX, llm, max_steps=3).ask("loop forever")
    assert not result.answered
    assert result.problems == ["Ran out of steps before answering."]


def test_citing_a_bracketed_id_in_read_section_works():
    llm = ScriptedLLM(read("[slos#1]"), say("A burn rate of 14.4 uses the budget in about 2 days [slos#1]."))
    assert Agent(INDEX, llm).ask("What does 14.4 mean?").answered


def test_the_top_results_come_back_in_full_so_a_small_model_sees_the_answer():
    """A real 3B model searched "500s after deploy", saw a preview cut off right before the
    rollback command, and gave up. The top results now come back whole."""
    llm = ScriptedLLM(search("500s after deploy"), say(NOT_FOUND))
    Agent(INDEX, llm).ask("Checkout is throwing 500s after a deploy")
    results = llm.requests[-1][-1]["content"]
    assert "kubectl -n checkout rollout undo deployment/checkout" in results


def test_lower_results_are_cut_and_say_how_to_read_more():
    llm = ScriptedLLM(search("checkout"), say(NOT_FOUND))
    Agent(INDEX, llm, results=6).ask("checkout")
    results = llm.requests[-1][-1]["content"]
    assert "(call read_section for the rest)" in results
