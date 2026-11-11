import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from runbook.llm import LLMError, Ollama, ToolCall, parse


def test_parse_reads_text_and_tool_calls():
    reply = parse(
        {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search_docs", "arguments": {"query": "500 errors"}}},
                    {"function": {"name": "read_section", "arguments": '{"id": "a#1"}'}},
                ],
            }
        }
    )
    assert reply.tool_calls == [
        ToolCall("search_docs", {"query": "500 errors"}),
        ToolCall("read_section", {"id": "a#1"}),
    ]


def test_parse_survives_broken_arguments():
    reply = parse({"message": {"tool_calls": [{"function": {"name": "x", "arguments": "{not json"}}]}})
    assert reply.tool_calls == [ToolCall("x", {})]
    assert (
        parse({"message": {"tool_calls": [{"function": {"name": "x", "arguments": [1]}}]}})
        .tool_calls[0]
        .arguments
        == {}
    )


def test_parse_reports_ollama_errors():
    with pytest.raises(LLMError, match="model not found"):
        parse({"error": "model not found"})


def test_parse_handles_an_empty_message():
    assert parse({}).content == ""


class FakeOllama(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        FakeOllama.seen = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        payload = json.dumps({"message": {"content": "hello"}}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def test_ollama_sends_the_model_tools_and_no_streaming():
    server = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        reply = Ollama(f"http://127.0.0.1:{server.server_port}", model="m").chat(
            [{"role": "user"}], [{"t": 1}]
        )
    finally:
        server.shutdown()
    assert reply.content == "hello"
    assert FakeOllama.seen["model"] == "m"
    assert FakeOllama.seen["tools"] == [{"t": 1}]
    assert FakeOllama.seen["stream"] is False


def test_an_unreachable_ollama_explains_itself():
    with pytest.raises(LLMError, match="Is `ollama serve` running"):
        Ollama("http://127.0.0.1:1", timeout=2).chat([], [])


def test_only_http_urls_are_allowed():
    with pytest.raises(ValueError):
        Ollama("file:///etc/passwd")


class SlowOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        time.sleep(1.5)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"message": {"content": "late"}}')

    def log_message(self, *args):
        pass


def test_a_slow_model_gets_a_helpful_timeout_message():
    server = HTTPServer(("127.0.0.1", 0), SlowOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(LLMError, match="took longer than 0.5 seconds.*smaller model"):
            Ollama(f"http://127.0.0.1:{server.server_port}", timeout=0.5).chat([], [])
    finally:
        server.shutdown()
