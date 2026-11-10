from runbook.llm import Reply, ToolCall


class ScriptedLLM:
    """Plays back replies in order and records every request, so tests can check the conversation."""

    def __init__(self, *replies: Reply):
        self.replies = list(replies)
        self.requests: list[list[dict]] = []

    def chat(self, messages, tools):
        self.requests.append([dict(m) for m in messages])
        return self.replies.pop(0)


def search(query: str) -> Reply:
    return Reply(tool_calls=[ToolCall("search_docs", {"query": query})])


def read(section_id: str) -> Reply:
    return Reply(tool_calls=[ToolCall("read_section", {"id": section_id})])


def say(text: str) -> Reply:
    return Reply(content=text)
