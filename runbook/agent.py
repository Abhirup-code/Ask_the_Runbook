"""The agent: the model searches the runbooks with tools, reads what it needs, and answers with
citations. The guard checks the answer; if it fails, the model gets one chance to fix it, and
after that the agent says it doesn't know and shows where to look instead."""

from __future__ import annotations

from dataclasses import dataclass, field

from runbook.docs import Section
from runbook.guard import check
from runbook.llm import LLM
from runbook.search import Index

NOT_FOUND = "NOT_FOUND"

SYSTEM = f"""You are an on-call assistant. Answer only from the team's runbooks.
- Always call search_docs first. Call read_section to read a section in full before relying on it.
- Answer in a few short sentences or steps, with the exact commands from the runbooks.
- Cite every claim with the section id in square brackets, like [checkout-availability#2].
- Never add a number, threshold, or command that is not in the sections you read.
- If the runbooks do not answer the question, reply with exactly {NOT_FOUND}."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the runbooks. Returns the best sections with their ids and a preview.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "Read one runbook section in full, by its id, like checkout-availability#2.",
            "parameters": {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The section id"}},
                "required": ["id"],
            },
        },
    },
]

PREVIEW = 300


@dataclass
class Result:
    answered: bool
    answer: str = ""
    citations: list[Section] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    suggestions: list[Section] = field(default_factory=list)  # where to look when there is no answer
    trace: list[str] = field(default_factory=list)  # every tool call, for debugging and the eval


class Agent:
    def __init__(self, index: Index, llm: LLM, max_steps: int = 6, results: int = 4):
        self.index, self.llm, self.max_steps, self.results = index, llm, max_steps, results

    def _search(self, query: str, seen: dict[str, Section]) -> str:
        hits = self.index.search(query, k=self.results)
        if not hits:
            return "No matching sections."
        out = []
        for h in hits:
            seen[h.section.id] = h.section
            preview = h.section.text[:PREVIEW] + ("..." if len(h.section.text) > PREVIEW else "")
            out.append(f"[{h.section.id}] {h.section.doc} > {h.section.title}\n{preview}")
        return "\n\n".join(out)

    def _read(self, section_id: str, seen: dict[str, Section]) -> str:
        section = self.index.get(section_id.strip().strip("[]"))
        if section is None:
            return f"No section with id {section_id!r}. Use an id from search_docs."
        seen[section.id] = section
        return section.render()

    def _run_tool(self, name: str, args: dict, seen: dict[str, Section]) -> str:
        if name == "search_docs":
            return self._search(str(args.get("query", "")), seen)
        if name == "read_section":
            return self._read(str(args.get("id", "")), seen)
        return f"Unknown tool {name!r}. Use search_docs or read_section."

    def _give_up(self, question: str, result: Result) -> Result:
        result.suggestions = [h.section for h in self.index.search(question, k=3)]
        return result

    def ask(self, question: str) -> Result:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
        seen: dict[str, Section] = {}
        trace: list[str] = []
        retried = False
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages, TOOLS)
            if reply.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": reply.content,
                        "tool_calls": [
                            {"function": {"name": c.name, "arguments": c.arguments}} for c in reply.tool_calls
                        ],
                    }
                )
                for call in reply.tool_calls:
                    trace.append(f"{call.name}({call.arguments})")
                    output = self._run_tool(call.name, call.arguments, seen)
                    messages.append({"role": "tool", "tool_name": call.name, "content": output})
                continue

            text = reply.content.strip()
            if text.startswith(NOT_FOUND):
                return self._give_up(
                    question, Result(False, problems=["The runbooks don't cover this."], trace=trace)
                )
            verdict = check(text, seen)
            if verdict.ok:
                return Result(True, text, [seen[c] for c in verdict.cited], trace=trace)
            if retried:
                return self._give_up(question, Result(False, problems=verdict.problems, trace=trace))
            retried = True
            messages.append({"role": "assistant", "content": text})
            messages.append(
                {
                    "role": "user",
                    "content": "Your answer broke these rules: "
                    + " ".join(verdict.problems)
                    + f" Fix it using only sections you retrieved, or reply {NOT_FOUND}.",
                }
            )
        return self._give_up(
            question, Result(False, problems=["Ran out of steps before answering."], trace=trace)
        )
