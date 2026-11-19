import json

import pytest
from pydantic import BaseModel

from pfb_app.config import Settings
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.structured import (StructuredResult, YieldRequested, call_structured,
                                          call_structured_batch)
from pfb_app.storage.db import Repo, connect, init_schema


class Answer(BaseModel):
    value: int


GOOD = json.dumps({"value": 1})


@pytest.fixture
def repo(tmp_path):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, Settings(data_dir=tmp_path / "data"))


def make_req(text="q"):
    return ChatRequest(system="s", user_text=text, image_paths=[],
                       json_schema=Answer.model_json_schema(), max_tokens=100)


def run(backend, repo, req=None, **kw):
    kw.setdefault("max_retries", 2)
    kw.setdefault("yield_check", lambda: False)
    return call_structured(backend, repo, req or make_req(), Answer, **kw)


def test_retry_then_success(repo):
    fake = FakeBackend(["not json", GOOD])
    res = run(fake, repo)
    assert res.parsed == Answer(value=1)
    assert len(res.raw_responses) == 2
    assert "not json" not in fake.calls[0].user_text
    assert len(res.errors) == 1
    assert res.errors[0] in fake.calls[1].user_text


def test_gives_up_after_max_retries(repo):
    fake = FakeBackend(["bad", "bad", "bad"])
    res = run(fake, repo)
    assert res.parsed is None
    assert len(res.errors) == 3
    assert len(fake.calls) == 3


def test_validate_errors_trigger_retry(repo):
    seen = []

    def validate(m):
        seen.append(m)
        return ["bad value"] if len(seen) == 1 else []

    fake = FakeBackend([GOOD, GOOD])
    res = run(fake, repo, validate=validate)
    assert res.parsed is not None
    assert len(fake.calls) == 2


def test_cache_prevents_regeneration(repo):
    fake = FakeBackend([GOOD])
    run(fake, repo)
    res = run(fake, repo)
    assert res.parsed == Answer(value=1)
    assert len(fake.calls) == 1


def test_yield_raises_before_call(repo):
    fake = FakeBackend([GOOD])
    with pytest.raises(YieldRequested):
        run(fake, repo, yield_check=lambda: True)
    assert fake.calls == []


def test_image_content_changes_hash(repo, tmp_path):
    img = tmp_path / "a.jpg"
    img.write_bytes(b"one")
    req = make_req()
    req.image_paths = [img]
    fake = FakeBackend([GOOD, GOOD])
    run(fake, repo, req)
    img.write_bytes(b"two")
    run(fake, repo, req)
    assert len(fake.calls) == 2


def test_batch_retries_only_failures(repo):
    def responder(req):
        if "a" == req.user_text and "エラー" not in req.user_text:
            return "bad"
        return GOOD

    fake = FakeBackend(responder)
    results = call_structured_batch(fake, repo, [make_req("a"), make_req("b")], Answer,
                                    max_retries=2, yield_check=lambda: False)
    assert all(isinstance(r, StructuredResult) and r.parsed for r in results)
    assert len(fake.calls) == 3
    assert len(results[0].raw_responses) == 2 and len(results[1].raw_responses) == 1


def test_max_tokens_in_hash(repo):
    fake = FakeBackend([GOOD, GOOD])
    run(fake, repo, make_req())
    other = make_req()
    other.max_tokens = 200
    run(fake, repo, other)
    assert len(fake.calls) == 2


def test_short_response_list_raises(repo):
    class Short:
        model_id = "short"

        def generate(self, reqs):
            return []

        def close(self):
            pass

    with pytest.raises(RuntimeError):
        run(Short(), repo)


def test_batch_accepts_one_validator_per_request(repo):
    fake = FakeBackend(lambda req: GOOD)
    rejects_b = lambda m: ["bだけ不合格"]
    accepts = lambda m: []
    results = call_structured_batch(fake, repo, [make_req("a"), make_req("b")], Answer,
                                    max_retries=0, yield_check=lambda: False,
                                    validate=[accepts, rejects_b])
    assert results[0].parsed is not None and results[1].parsed is None
    assert results[1].errors == ["bだけ不合格"]
