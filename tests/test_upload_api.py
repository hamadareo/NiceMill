"""The resumable upload endpoint over HTTP (ui/upload_api.py), on a bare Starlette app."""
import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.services import chunked_upload as cu
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui import upload_api


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        max_upload_mb=1)
    app = Starlette(routes=upload_api.routes(settings))
    with TestClient(app, base_url="http://pfb.test") as c:
        c.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        conn = connect(files.db_path(settings))
        init_schema(conn)
        c.repo = Repo(conn, settings)
        c.user = auth.create_user(c.repo, "tasuke", "手伝う 人", "evaluator",
                                  "correct horse battery")
        yield c


def session_of(client, user=None) -> int:
    """A signed-in session (services.auth) of `user` (the fixture's 評価する人)."""
    cookie = auth.start_session(client.repo, client.settings, user or client.user)
    return auth.check_cookie(client.repo, client.settings, cookie).id


def bound(client, user=None) -> str:
    """An upload token of a signed-in session, as the screens issue it."""
    return cu.issue_token(client.settings, session_id=session_of(client, user))


def headers(client, token=None, **extra):
    return {"X-PFB-Token": token or bound(client), **extra}


def test_create_send_resume_and_finish(client):
    h = headers(client)
    r = client.post("/pfb-upload", json={"name": "後半.mov", "size": 10}, headers=h)
    assert r.status_code == 201
    uid, chunk = r.json()["id"], r.json()["chunk"]
    assert chunk == upload_api.CHUNK_BYTES
    r = client.patch(f"/pfb-upload/{uid}", content=b"01234", headers={**h, "Upload-Offset": "0"})
    assert r.json() == {"offset": 5, "complete": False}
    # the connection dropped: the browser asks where to go on, then sends the rest
    assert client.get(f"/pfb-upload/{uid}", headers=h).json() == {
        "offset": 5, "size": 10, "complete": False}
    r = client.patch(f"/pfb-upload/{uid}", content=b"23456", headers={**h, "Upload-Offset": "2"})
    assert r.status_code == 409 and r.json()["offset"] == 5
    r = client.patch(f"/pfb-upload/{uid}", content=b"56789", headers={**h, "Upload-Offset": "5"})
    assert r.json() == {"offset": 10, "complete": True}
    path = cu.claim(client.settings, h["X-PFB-Token"], uid)
    assert path.read_bytes() == b"0123456789"


def test_without_a_valid_token_nothing_happens(client):
    r = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4})
    assert r.status_code == 401 and "開き直して" in r.json()["error"]
    r = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4},
                    headers={"X-PFB-Token": "1.2.3"})
    assert r.status_code == 401


def test_a_token_needs_a_signed_in_session_that_may_add_work(client):
    body = {"name": "a.mp4", "size": 4}
    legacy = cu.issue_token(client.settings)              # issued before accounts: no session
    r = client.post("/pfb-upload", json=body, headers={"X-PFB-Token": legacy})
    assert r.status_code == 401 and r.json()["error"] == upload_api.SIGNED_OUT
    sid = session_of(client)
    token = cu.issue_token(client.settings, session_id=sid)
    assert client.post("/pfb-upload", json=body, headers={"X-PFB-Token": token}
                       ).status_code == 201
    auth.revoke_session(client.repo, sid)                 # signed out: the phone stops at once
    assert client.post("/pfb-upload", json=body, headers={"X-PFB-Token": token}
                       ).status_code == 401
    viewer = auth.create_user(client.repo, "miru", "見る 人", "viewer", "correct horse battery")
    r = client.post("/pfb-upload", json=body, headers={"X-PFB-Token": bound(client, viewer)})
    assert r.status_code == 401


def test_another_site_is_refused(client):
    h = headers(client, Origin="https://evil.example")
    r = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=h)
    assert r.status_code == 403
    ok = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4},
                     headers=headers(client, Origin="http://pfb.test"))
    assert ok.status_code == 201


def test_limits(client):
    h = headers(client)
    assert client.post("/pfb-upload", json={"name": "a.exe", "size": 4}, headers=h).status_code == 400
    r = client.post("/pfb-upload", json={"name": "a.mp4", "size": 2 * 1024 * 1024}, headers=h)
    assert r.status_code == 413 and "MB" in r.json()["error"]
    uid = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=h).json()["id"]
    r = client.patch(f"/pfb-upload/{uid}", content=b"12345", headers={**h, "Upload-Offset": "0"})
    assert r.status_code == 400
    assert client.get("/pfb-upload/../../app.db", headers=h).status_code == 404
    assert client.get("/pfb-upload/" + "0" * 32, headers=h).status_code == 404


def test_another_token_sees_nothing_and_delete_cancels(client):
    mine, theirs = headers(client), headers(client)
    uid = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=mine).json()["id"]
    assert client.get(f"/pfb-upload/{uid}", headers=theirs).status_code == 404
    assert client.delete(f"/pfb-upload/{uid}", headers=theirs).status_code == 404
    assert client.delete(f"/pfb-upload/{uid}", headers=mine).status_code == 204
    assert client.get(f"/pfb-upload/{uid}", headers=mine).status_code == 404


# --- raw ASGI requests: what the server reads, and from whom (review round 1) ---------------

def raw(method, path, headers=None, body_parts=(), client=("100.64.0.9", 5000), delay=0.0):
    """Call the endpoint with a hand-made request; returns (status, json, pieces read)."""
    import asyncio
    import json as _json
    from starlette.requests import Request

    reads = []
    parts = list(body_parts)

    async def receive():
        if delay:
            await asyncio.sleep(delay)
        reads.append(1)
        body = parts.pop(0) if parts else b""
        return {"type": "http.request", "body": body, "more_body": bool(parts)}

    scope = {"type": "http", "method": method, "path": path, "root_path": "", "query_string": b"",
             "headers": [(k.lower().encode(), str(v).encode()) for k, v in (headers or {}).items()],
             "client": client, "server": ("127.0.0.1", 8640), "scheme": "http",
             "path_params": {}}
    sent = {}

    async def send(message):
        if message["type"] == "http.response.start":
            sent["status"] = message["status"]
        elif message["type"] == "http.response.body":
            sent["body"] = sent.get("body", b"") + message.get("body", b"")

    app = Starlette(routes=upload_api.routes())
    asyncio.run(app(scope, receive, send))
    try:
        data = _json.loads(sent.get("body") or b"null")
    except ValueError:
        data = None
    return sent["status"], data, len(reads)


def test_a_tokenless_big_body_is_never_read(client):
    big = [b" " * (1 << 20)] * 300
    status, data, reads = raw("POST", "/pfb-upload", {"Content-Type": "application/json",
                                                      "Content-Length": str(300 << 20)}, big)
    assert status == 401 and reads == 0
    uid = "0" * 32
    status, _, reads = raw("PATCH", f"/pfb-upload/{uid}", {"Upload-Offset": "0",
                                                           "Content-Length": str(8 << 20)}, big)
    assert status == 401 and reads == 0


def test_a_create_needs_a_small_declared_body(client):
    tok = bound(client)
    status, _, reads = raw("POST", "/pfb-upload", {"X-PFB-Token": tok}, [b'{"name":"a.mp4"}'])
    assert status == 411 and reads == 0                       # no Content-Length
    status, _, reads = raw("POST", "/pfb-upload", {"X-PFB-Token": tok, "Content-Length": "5000"},
                           [b" " * 5000])
    assert status == 413 and reads == 0
    body = b'{"name": "a.mp4", "size": 4}'
    status, data, _ = raw("POST", "/pfb-upload", {"X-PFB-Token": tok,
                                                  "Content-Length": str(len(body))}, [body])
    assert status == 201 and len(data["id"]) == 32


def test_a_stalled_body_is_cut_off(client, monkeypatch):
    monkeypatch.setattr(upload_api, "BODY_TIMEOUT_SEC", 0.2)
    h = headers(client)
    uid = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=h).json()["id"]
    status, data, _ = raw("PATCH", f"/pfb-upload/{uid}", {**h, "Upload-Offset": "0"},
                          [b"12"], delay=1.0)
    assert status == 408 and "続きから" in data["error"]


def test_bodies_at_once_are_limited_per_token(client, monkeypatch):
    h = headers(client)
    uid = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=h).json()["id"]
    owner = cu.check_token(client.settings, h["X-PFB-Token"])
    monkeypatch.setitem(upload_api._sending, owner, upload_api.MAX_PATCHES_PER_OWNER)
    r = client.patch(f"/pfb-upload/{uid}", content=b"12", headers={**h, "Upload-Offset": "0"})
    assert r.status_code == 429
    other = headers(client)                                  # another token is not held up
    uid2 = client.post("/pfb-upload", json={"name": "b.mp4", "size": 2}, headers=other).json()["id"]
    r = client.patch(f"/pfb-upload/{uid2}", content=b"12", headers={**other, "Upload-Offset": "0"})
    assert r.status_code == 200


def test_head_answers_like_get(client):
    h = headers(client)
    uid = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=h).json()["id"]
    assert client.head(f"/pfb-upload/{uid}", headers=h).status_code == 200
    assert client.head(f"/pfb-upload/{'0' * 32}", headers=h).status_code == 404


def test_unknown_ids_leave_no_lock_files(client):
    import os
    part = client.settings.data_dir / "uploads" / cu.PARTIAL
    for _ in range(20):
        client.delete(f"/pfb-upload/{os.urandom(16).hex()}")                     # no token
        client.delete(f"/pfb-upload/{os.urandom(16).hex()}", headers=headers(client))
    assert not part.exists() or not list(part.glob("*.lock"))


def test_negative_offsets_are_refused(client):
    h = headers(client)
    uid = client.post("/pfb-upload", json={"name": "a.mp4", "size": 4}, headers=h).json()["id"]
    r = client.patch(f"/pfb-upload/{uid}", content=b"12", headers={**h, "Upload-Offset": "-5"})
    assert r.status_code == 400 and r.json()["error"] == cu.NEGATIVE_OFFSET


def test_behind_tailscale_serve_the_forwarded_host_counts(client):
    tok = bound(client)
    body = b'{"name": "a.mp4", "size": 4}'
    h = {"X-PFB-Token": tok, "Content-Length": str(len(body)), "Host": "127.0.0.1:8640",
         "Origin": "https://myhost.example.ts.net", "X-Forwarded-Host": "myhost.example.ts.net"}
    status, _, _ = raw("POST", "/pfb-upload", h, [body], client=("127.0.0.1", 40000))
    assert status == 201                                     # from the local proxy
    status, _, _ = raw("POST", "/pfb-upload", h, [body], client=("100.64.0.9", 40000))
    assert status == 403                                     # anyone else: not believed


def test_a_trickled_body_meets_the_whole_body_deadline(client, monkeypatch):
    """Each piece comes in time, but the body as a whole does not (review round 2)."""
    import asyncio
    from starlette.requests import Request
    monkeypatch.setattr(upload_api, "BODY_TIMEOUT_SEC", 0.5)
    assert upload_api.body_deadline_sec(8 << 20) == 0.5 or upload_api.body_deadline_sec(8 << 20) > 80

    async def receive():
        await asyncio.sleep(0.1)
        return {"type": "http.request", "body": b"x", "more_body": True}   # never ends

    scope = {"type": "http", "method": "PATCH", "path": "/", "headers": [(b"content-length", b"10")],
             "query_string": b""}

    async def go():
        return await upload_api._read_capped(Request(scope, receive), 1 << 20)
    import time as _t
    t0 = _t.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(go())
    assert _t.monotonic() - t0 < 2


def test_the_whole_body_deadline_grows_with_its_length():
    assert upload_api.body_deadline_sec(None) == upload_api.BODY_TIMEOUT_SEC
    assert upload_api.body_deadline_sec(8 << 20) == upload_api.BODY_TIMEOUT_SEC   # ~84 s
    assert upload_api.body_deadline_sec(100_000_000) == 1000
