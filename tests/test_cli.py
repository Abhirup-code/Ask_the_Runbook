from runbook.__main__ import main
from runbook.llm import LLMError
from tests.fakes import ScriptedLLM, read, say, search


def test_search_prints_ranked_sections(capsys):
    assert main(["search", "rollback a bad deploy"]) == 0
    assert "[checkout-" in capsys.readouterr().out


def test_eval_passes_and_fails_on_the_threshold(capsys):
    assert main(["eval"]) == 0
    assert "Retrieval hit@3:" in capsys.readouterr().out
    assert main(["eval", "--min", "1.01"]) == 1


def test_ask_prints_the_answer_sources_and_trace(capsys):
    llm = ScriptedLLM(
        search("burn rate"), read("slos#1"), say("14.4 uses the budget in about 2 days [slos#1].")
    )
    assert main(["ask", "What is 14.4?", "--trace"], llm=llm) == 0
    out = capsys.readouterr().out
    assert "tool: search_docs" in out
    assert "Sources:\n  [slos#1]" in out


def test_ask_without_an_answer_points_to_sections(capsys):
    llm = ScriptedLLM(search("kafka"), say("NOT_FOUND"))
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
