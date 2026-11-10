"""Talks to a local model through Ollama's chat API, with tool calling. No API key, no cost,
and the runbooks never leave the machine."""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("RUNBOOK_MODEL", "qwen2.5:7b")  # small, free, and good at tool calls


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def chat(self, messages: list[dict], tools: list[dict]) -> Reply: ...


class LLMError(RuntimeError):
    pass


class Ollama:
    def __init__(self, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL, timeout: float = 120.0):
        if not url.startswith(("http://", "https://")):
            raise ValueError("Ollama URL must start with http:// or https://")
        self.url, self.model, self.timeout = url.rstrip("/"), model, timeout

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0},
            }
        ).encode()
        request = urllib.request.Request(  # noqa: S310 (scheme checked in __init__)
            f"{self.url}/api/chat", data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:  # noqa: S310 (scheme checked above)
                data = json.load(resp)
        except OSError as exc:
            raise LLMError(
                f"Could not reach Ollama at {self.url}: {exc}. Is `ollama serve` running?"
            ) from exc
        return parse(data)


def parse(data: dict) -> Reply:
    """Turns an Ollama /api/chat response into a Reply. Arguments may arrive as a dict or a JSON string."""
    if "error" in data:
        raise LLMError(f"Ollama error: {data['error']}")
    message = data.get("message") or {}
    calls = []
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        args = fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(ToolCall(str(fn.get("name", "")), args if isinstance(args, dict) else {}))
    return Reply(str(message.get("content") or ""), calls)
