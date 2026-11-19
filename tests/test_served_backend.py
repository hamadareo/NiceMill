"""ServedChatBackend: text-only chat on the session's own OpenAI-compatible server."""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import BaseModel

from pfb_app.config import Settings
from pfb_app.inference.backend import ChatRequest, RawResponse
from pfb_app.inference.served import (ERROR_BODY_CHARS, ServedChatBackend, ServerUnreachable,
                                      lenient_json)
from pfb_app.inference.structured import call_structured_batch, request_hash
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


def reverse(body):
    """The default answer: the user text reversed, with usage."""
    return 200, {"choices": [{"message": {"content": body["messages"][1]["content"][::-1]}}],
                 "usage": {"prompt_tokens": 3, "completion_tokens": 2}}


@pytest.fixture
def server():
    """A stand-in for vLLM's /v1/chat/completions on 127.0.0.1 (an ephemeral port).
    `answers["fn"](body) -> (status, json or str)` decides the answer (default: reverse)."""
    seen, answers = [], {"fn": reverse}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):   # where a followed redirect would land
            seen.append((self.path, None))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.path, body))
            status, answer = answers["fn"](body)
            data = (answer if isinstance(answer, str) else json.dumps(answer)).encode()
            self.send_response(status)
            if 300 <= status < 400:
                self.send_header("Location", "/v1/elsewhere")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen, answers
    httpd.shutdown()
    httpd.server_close()


def req(text):
    return ChatRequest(system="sys", user_text=text, image_paths=[],
                       json_schema={"type": "object"}, max_tokens=64)


def test_requests_are_json_schema_constrained_at_temperature_zero(server):
    url, seen, _ = server
    backend = ServedChatBackend(url, "qwen3vl-8b")
    out = backend.generate([req("abc"), req("xyz")])
    assert [o.text for o in out] == ["cba", "zyx"]   # order kept under concurrency
    assert out[0].prompt_tokens == 3 and out[0].completion_tokens == 2
    path, body = seen[0]
    assert path == "/v1/chat/completions"
    assert body["model"] == "qwen3vl-8b" and body["temperature"] == 0
    assert body["max_tokens"] == 64
    assert body["response_format"] == {"type": "json_schema",
                                       "json_schema": {"name": "output",
                                                       "schema": {"type": "object"}}}
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert backend.model_id == "served:qwen3vl-8b"


def test_images_are_refused():
    backend = ServedChatBackend("http://127.0.0.1:9/v1", "m")
    with pytest.raises(ValueError):
        backend.generate([ChatRequest("s", "t", ["/x.jpg"], {}, 8)])


def content(text):
    return 200, {"choices": [{"message": {"content": text}}]}


def test_one_failing_request_does_not_discard_the_others(server):
    url, _, answers = server
    answers["fn"] = lambda body: ((500, {"error": "boom " + "x" * 1000})
                                  if body["messages"][1]["content"] == "bad" else reverse(body))
    out = ServedChatBackend(url, "m").generate([req("abc"), req("bad"), req("xyz")])
    assert [o.text for o in out] == ["cba", "", "zyx"]
    assert out[0].error is None and out[2].error is None
    assert out[1].error.startswith("ServedHTTPError: HTTP 500: ")
    assert "boom" in out[1].error                    # the start of the server's error body
    assert len(out[1].error) < ERROR_BODY_CHARS + 50  # ... and only the start


def test_a_refused_schema_is_retried_once_without_it_and_read_leniently(server):
    url, seen, answers = server
    def fn(body):
        if "response_format" in body:
            return 400, "response_format json_schema is not supported"
        return content('Here you go:\n```json\n{"a": 1}\n```\nThanks')
    answers["fn"] = fn
    (out,) = ServedChatBackend(url, "m").generate([req("abc")])
    assert out.error is None and json.loads(out.text) == {"a": 1}
    assert ["response_format" in body for _, body in seen] == [True, False]


def test_a_refused_retry_is_reported_with_its_body(server):
    url, seen, answers = server
    answers["fn"] = lambda body: (422, {"detail": "unprocessable"})
    (out,) = ServedChatBackend(url, "m").generate([req("abc")])
    assert "HTTP 422" in out.error and "unprocessable" in out.error
    assert len(seen) == 2   # once with the schema, once without; no more


def test_no_content_is_an_empty_answer(server):
    url, _, answers = server
    answers["fn"] = lambda body: (200, {"choices": [{"message": {"content": None}}]})
    (out,) = ServedChatBackend(url, "m").generate([req("abc")])
    assert out.text == "" and out.error is None


@pytest.mark.parametrize("url", ["http://10.0.0.5:8200/v1", "https://api.openai.com/v1",
                                 "http://localhost.example.com/v1", "file:///tmp/x"])
def test_only_this_machine_is_accepted(url):
    with pytest.raises(ValueError):
        ServedChatBackend(url, "m")
    for local in ("http://localhost:8200/v1", "http://127.0.0.1:8200/v1", "http://[::1]:8200/v1"):
        ServedChatBackend(local, "m")


def test_an_unreachable_server_is_one_error():
    with socket.socket() as s:   # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ServerUnreachable):
        ServedChatBackend(f"http://127.0.0.1:{port}/v1", "m").generate([req("a"), req("b")])


@pytest.mark.parametrize("text, expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('The answer is {"a": {"b": 2}} as requested.', '{"a": {"b": 2}}'),
    ("no json here", "no json here"),
    ("", ""),
])
def test_lenient_json(text, expected):
    assert lenient_json(text) == expected


def test_a_failed_request_is_not_cached_and_is_retried(tmp_path):
    settings = Settings(data_dir=tmp_path / "data")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)

    class Flaky:
        model_id = "flaky"

        def __init__(self):
            self.calls = 0

        def generate(self, reqs):
            self.calls += 1
            if self.calls == 1:
                return [RawResponse("", None, None, 0.0, error="ServedHTTPError: HTTP 500")]
            return [RawResponse('{"x": 1}', None, None, 0.0)]

        def close(self):
            pass

    class Out(BaseModel):
        x: int

    request = req("abc")
    (result,) = call_structured_batch(Flaky(), repo, [request], Out, max_retries=1,
                                      yield_check=lambda: False)
    assert result.parsed == Out(x=1)
    assert result.errors == ["モデルの呼び出しに失敗しました: ServedHTTPError: HTTP 500"]
    assert repo.cache_get(request_hash("flaky", request, 0)) is None   # the failure: not cached


def test_proxy_settings_are_ignored(server, monkeypatch):
    url, seen, _ = server
    for k in ("http_proxy", "HTTP_PROXY", "all_proxy"):
        monkeypatch.setenv(k, "http://127.0.0.1:9")   # nobody listens there
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    (out,) = ServedChatBackend(url, "m").generate([req("abc")])
    assert out.error is None and out.text == "cba" and len(seen) == 1


def test_a_redirect_is_never_followed(server):
    url, seen, answers = server
    answers["fn"] = lambda body: (302, {"moved": True})
    (out,) = ServedChatBackend(url, "m").generate([req("abc")])
    assert out.error is not None and "HTTP 302" in out.error
    assert len(seen) == 1
