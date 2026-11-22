"""Command line:

python -m runbook ask "Checkout is throwing 500s, what do I check?"
python -m runbook search "rollback"
python -m runbook eval          # retrieval and guard evals
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runbook import evals
from runbook.agent import MIN_COVERAGE, Agent
from runbook.docs import load
from runbook.llm import DEFAULT_MODEL, DEFAULT_TIMEOUT, DEFAULT_URL, LLMError, Ollama
from runbook.search import Index

ROOT = Path(__file__).resolve().parent.parent


def as_json(question: str, result) -> dict:
    """The answer in a shape a chat bot or a script can use. Same rules: no answer without sources."""

    def source(s):
        return {"id": s.id, "doc": s.doc, "title": s.title}

    return {
        "question": question,
        "answered": result.answered,
        "answer": result.answer if result.answered else None,
        "sources": [source(s) for s in result.citations],
        "problems": result.problems,
        "closest_sections": [source(s) for s in result.suggestions],
        "trace": result.trace,
    }


def main(argv: list[str] | None = None, llm=None) -> int:
    parser = argparse.ArgumentParser(
        prog="runbook", description="Answers on-call questions from your runbooks."
    )
    parser.add_argument("--docs", default=str(ROOT / "corpus"), help="Folder of Markdown runbooks")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="Ask a question (needs Ollama)")
    ask.add_argument("question")
    ask.add_argument("--model", default=DEFAULT_MODEL)
    ask.add_argument("--url", default=DEFAULT_URL)
    ask.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="Seconds to wait for each model reply"
    )
    ask.add_argument("--trace", action="store_true", help="Show every tool call the agent made")
    ask.add_argument("--json", action="store_true", help="Print one JSON object, for a chat bot or a script")
    find = sub.add_parser("search", help="Show the best matching sections (no LLM needed)")
    find.add_argument("query")
    find.add_argument("-k", type=int, default=5, help="How many sections to show (default 5)")
    find.add_argument("--why", action="store_true", help="Show the line that matched, words in CAPITALS")
    find.add_argument("--terms", action="store_true", help="Show which query words earned each score")
    ev = sub.add_parser("eval", help="Measure retrieval and the guard (no LLM needed)")
    ev.add_argument("--questions", default=str(ROOT / "evals" / "questions.jsonl"))
    ev.add_argument("--guard-cases", default=str(ROOT / "evals" / "guard_cases.jsonl"))
    ev.add_argument("--min", type=float, default=0.0, help="Exit 1 if retrieval scores below this (0 to 1)")
    args = parser.parse_args(argv)

    try:
        sections = load(args.docs)
        index = Index(sections)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.command == "search":
        hits = index.search(args.query, k=args.k)
        if not hits:
            print("Nothing matched. The runbooks share no words with that question.")
            return 1
        for hit in hits:
            print(f"{hit.score:5.1f}  [{hit.section.id}] {hit.section.doc} > {hit.section.title}")
            if args.why:
                print(f"        {index.why(args.query, hit.section)}")
            if args.terms:
                parts = index.breakdown(args.query, hit.section)
                print("        " + "  +  ".join(f"{term} {score:.1f}" for term, score in parts))
        # The same number the agent abstains on, so "it says it doesn't know" can be looked into.
        covered = index.coverage(args.query, [h.section for h in hits])
        print(
            f"\nCoverage: {covered:.0%} of the question's words appear in these {len(hits)} sections"
            f" (the agent answers from {MIN_COVERAGE:.0%} up)"
        )
        return 0

    if args.command == "eval":
        questions = evals.load_questions(args.questions)
        result = evals.run(index, evals.answerable(questions))
        print(evals.report(result, 3))
        print(evals.scores_report(evals.top_scores(index, questions)))
        abstain = evals.abstain_check(index, questions, MIN_COVERAGE)
        print(evals.abstain_report(abstain, MIN_COVERAGE))
        guard = evals.run_guard(sections, evals.load_guard_cases(args.guard_cases))
        print(evals.guard_report(guard))
        return 0 if result.score >= args.min and guard.ok and abstain.costs_nothing else 1

    try:
        result = Agent(index, llm or Ollama(args.url, args.model, args.timeout)).ask(args.question)
    except LLMError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(as_json(args.question, result), indent=2))
        return 0 if result.answered else 1
    if args.trace:
        for step in result.trace:
            print(f"  tool: {step}")
    if result.answered:
        print(result.answer)
        print("\nSources:")
        for s in result.citations:
            print(f"  [{s.id}] {s.doc} > {s.title}")
        return 0
    print("I couldn't answer that from the runbooks.")
    for problem in result.problems:
        print(f"  {problem}")
    if result.suggestions:
        print("\nThese sections are the closest match:")
        for s in result.suggestions:
            print(f"  [{s.id}] {s.doc} > {s.title}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
