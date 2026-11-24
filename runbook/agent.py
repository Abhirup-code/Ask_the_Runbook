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
- The question comes with the runbook sections that best match it. Call search_docs or read_section
  only if you need more.
- Answer in a few short sentences or steps, with the exact commands from the runbooks.
- Cite every claim with the section id in square brackets, like [checkout-availability#2].
- Never add a number, threshold, or command that is not in the sections you read.
- If a result is about the question, answer from it. Reply with exactly {NOT_FOUND} only when no
  result covers the question."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the runbooks. Returns the best sections in full, with their ids.",
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

FULL = 3  # the best results come back in full: runbook sections are short, and small models rarely
# call read_section, so a cut-off preview hid the answer from them in testing
PREVIEW = 300
# A ceiling on the runbook text in one search result. Today's corpus sends 1,300 to 2,100
# characters, so nothing here is trimmed, but a team that writes long sections would otherwise push
# the system prompt and the earlier turns out of a small model's window without anything saying so.
# The budget is spent from the top down, so what gets trimmed is what ranked worst.
BUDGET = 4000
MIN_ROOM = 200  # less room left than this and a section is not worth showing at all
MORE = "... (call read_section for the rest)"


@dataclass
class Result:
    answered: bool
    answer: str = ""
    citations: list[Section] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    suggestions: list[Section] = field(default_factory=list)  # where to look when there is no answer
    trace: list[str] = field(default_factory=list)  # every tool call, for debugging and the eval


# The question has to share at least this much of its vocabulary with what search found before the
# model is asked anything. Measured on the eval with the four sections the agent actually retrieves:
# every threshold up to 0.40 keeps all 32 answerable questions, 0.35 and 0.40 both turn away 6 of
# the 8 the runbooks don't cover, and 0.45 starts costing real answers. 0.35 buys the full benefit
# with room before that edge. The score cannot do this job at all: an uncovered question can
# outscore a covered one on a single shared word, which the eval prints.
MIN_COVERAGE = 0.35


class Agent:
    def __init__(
        self,
        index: Index,
        llm: LLM,
        max_steps: int = 6,
        results: int = 4,
        min_coverage: float = MIN_COVERAGE,
        budget: int = BUDGET,
    ):
        self.index, self.llm, self.max_steps, self.results = index, llm, max_steps, results
        self.min_coverage, self.budget = min_coverage, budget

    def _fit(self, text: str, rank: int, room: int) -> str | None:
        """The section's text as it should appear, or None when too little budget is left to bother.

        The top hit is never dropped, only trimmed: without it there is nothing to answer from.
        """
        if rank >= FULL and len(text) > PREVIEW:
            text = text[:PREVIEW] + MORE
        if len(text) <= room:
            return text
        if rank and room < MIN_ROOM:
            return None
        return text[: max(room - len(MORE), MIN_ROOM)].rstrip() + MORE

    def _search(self, query: str, seen: dict[str, Section]) -> str:
        hits = self.index.search(query, k=self.results)
        if not hits:
            return "No matching sections."
        out, spent, dropped = [], 0, 0
        for rank, h in enumerate(hits):
            text = self._fit(h.section.text, rank, self.budget - spent)
            if text is None:
                dropped += 1
                continue
            # Only what the model is actually shown counts as retrieved, so the guard keeps
            # rejecting a citation of a section that was trimmed away.
            seen[h.section.id] = h.section
            spent += len(text)
            out.append(f"[{h.section.id}] {h.section.doc} > {h.section.title}\n{text}")
        if dropped:
            out.append(f"({dropped} more section(s) matched. Narrow the search, or read_section by id.)")
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
        seen: dict[str, Section] = {}
        trace: list[str] = []
        retried = nudged = False
        # Search before the model's first turn and put the sections right in the question, the
        # classic RAG prompt. Small models often skip tools and pay little attention to tool
        # messages, but they read the question. The tools stay available for follow-ups.
        trace.append(f"search_docs({{'query': {question!r}}}) [automatic]")
        context = self._search(question, seen)
        # Nothing worth reading means nothing worth asking. Stopping here saves a model call and,
        # more importantly, removes the chance of the model talking its way around thin context.
        covered = self.index.coverage(question, seen.values())
        if covered < self.min_coverage:
            trace.append(f"abstain: the question's words are {covered:.0%} covered by what was found")
            # Naming the words that are nowhere in the corpus turns "no" into something to act on:
            # either the runbook is genuinely missing, or the question used the team's other word
            # for the thing and is worth asking again.
            absent = self.index.missing_words(question, seen.values())
            problem = "The runbooks don't cover this."
            if absent:
                problem += f" Nothing in them mentions: {', '.join(absent)}."
            return self._give_up(question, Result(False, problems=[problem], trace=trace))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Runbook sections:\n\n{context}\n\nQuestion: {question}"},
        ]
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
            if text.startswith(NOT_FOUND) and seen and not nudged:
                # Weak models say NOT_FOUND by reflex. If the search found sections, ask once to
                # check them. The guard still rejects anything they don't support.
                nudged = True
                messages.append({"role": "assistant", "content": text})
                messages.append(
                    {
                        "role": "user",
                        "content": "Check the runbook sections again: "
                        + ", ".join(f"[{i}]" for i in seen)
                        + ". If one answers the question, answer from it with citations."
                        + f" If none does, reply {NOT_FOUND}.",
                    }
                )
                continue
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
