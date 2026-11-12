import json

from runbook.__main__ import main
from runbook.llm import LLMError
from tests.fakes import ScriptedLLM, read, say, search


def test_search_prints_ranked_sections(capsys):
    assert main(["search", "rollback a bad deploy"]) == 0
    assert "[checkout-" in capsys.readouterr().out


def test_eval_passes_and_fails_on_the_threshold(capsys):
    assert main(["eval"]) == 0
    out = capsys.readouterr().out
    assert "Retrieval hit@3:" in out and "Guard: 15/15" in out
    assert main(["eval", "--min", "1.01"]) == 1


def test_eval_fails_when_the_guard_gets_one_wrong(tmp_path, capsys):
    cases = tmp_path / "g.jsonl"
    cases.write_text('{"expect": "pass", "seen": ["slos#0"], "answer": "It is 99.99% [slos#0]."}\n')
    assert main(["eval", "--guard-cases", str(cases)]) == 1
    assert "Guard: 0/1" in capsys.readouterr().out


def test_ask_prints_the_answer_sources_and_trace(capsys):
    llm = ScriptedLLM(
        search("burn rate"), read("slos#1"), say("14.4 uses the budget in about 2 days [slos#1].")
    )
    assert main(["ask", "What is 14.4?", "--trace"], llm=llm) == 0
    out = capsys.readouterr().out
    assert "tool: search_docs" in out
    assert "Sources:\n  [slos#1]" in out


def test_ask_without_an_answer_points_to_sections(capsys):
    llm = ScriptedLLM(search("kafka"), say("NOT_FOUND"), say("NOT_FOUND"))
    assert main(["ask", "How do I fix Kafka lag?"], llm=llm) == 1
    assert "couldn't answer" in capsys.readouterr().out


def test_ask_explains_when_ollama_is_down(capsys):
    class Down:
        def chat(self, messages, tools):
            raise LLMError("Could not reach Ollama")

    assert main(["ask", "anything"], llm=Down()) == 2
    assert "Could not reach Ollama" in capsys.readouterr().err


def test_an_empty_docs_folder_is_an_error(tmp_path, capsys):
    assert main(["--docs", str(tmp_path), "search", "x"]) == 2
    assert "No Markdown sections" in capsys.readouterr().err


def test_ask_passes_the_timeout_to_ollama(monkeypatch, capsys):
    seen = {}

    class Recording:
        def __init__(self, url, model, timeout):
            seen.update(url=url, model=model, timeout=timeout)

        def chat(self, messages, tools):
            return say("NOT_FOUND")

    monkeypatch.setattr("runbook.__main__.Ollama", Recording)
    main(["ask", "anything", "--timeout", "600", "--model", "qwen2.5:3b"])
    assert seen["timeout"] == 600 and seen["model"] == "qwen2.5:3b"


def test_ask_json_prints_the_answer_and_its_sources(capsys):
    llm = ScriptedLLM(read("slos#1"), say("14.4 uses the budget in about 2 days [slos#1]."))
    assert main(["ask", "What is 14.4?", "--json"], llm=llm) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["answered"] is True
    assert out["answer"].endswith("[slos#1].")
    assert out["sources"][0]["id"] == "slos#1"
    assert out["closest_sections"] == []


def test_ask_json_never_carries_a_rejected_answer(capsys):
    llm = ScriptedLLM(say("Scale to 12 replicas."), say("Scale to 12 replicas."))
    assert main(["ask", "How do I scale checkout?", "--json"], llm=llm) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["answered"] is False and out["answer"] is None and out["sources"] == []
    assert out["problems"] and out["closest_sections"]
