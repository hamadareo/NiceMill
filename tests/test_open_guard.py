"""open mode's safety rail (rel7, 2026-11-14): `auth_enabled: false` opens the app only to a
request that arrives directly from a private / tailnet / loopback peer without the marks of a
public proxy. Everything else meets the normal sign-in gate and a 401 at /pfb-upload.

The rule (ui/open_guard.py) is a pure function of the TCP peer and the request headers; the
screens read them from Streamlit's client context, the upload endpoint from the Starlette
request. Synthetic addresses only (documentation ranges for "public": 203.0.113.0/24,
198.51.100.0/24, 2001:db8::/32)."""
import asyncio

import pytest
import yaml
from starlette.requests import Request

from pfb_app.config import Settings
from pfb_app.services import auth
from pfb_app.services import chunked_upload as cu
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui import auth_ui, open_guard, upload_api
from streamlit.testing.v1 import AppTest
from test_wizard_ui_basic import UI, html, ok, screen_text

pytestmark = pytest.mark.real_auth      # the real gate: no synthetic admin bypass

PUBLIC = "203.0.113.7"
WORK = "安全柵の練習カード"
PW = "correct horse battery"


def allowed(peer, **headers):
    return open_guard.decide(peer, [(k.replace("_", "-"), v) for k, v in headers.items()]).ok


# --- the rule ------------------------------------------------------------------------------

@pytest.mark.parametrize("peer", ["127.0.0.1", "::1", "10.1.2.3", "172.16.0.9", "172.31.255.1",
                                  "192.168.1.20", "127.0.0.1", "100.64.0.1",
                                  "100.127.255.254", "fd7a:115c:a1e0::1", "fe80::1",
                                  "169.254.1.1", "::ffff:127.0.0.1"])
def test_a_direct_private_peer_is_let_in(peer):
    assert allowed(peer, host="myhost.example.ts.net", user_agent="x")


@pytest.mark.parametrize("peer", [PUBLIC, "198.51.100.1", "8.8.8.8", "100.128.0.1",
                                  "172.32.0.1", "2001:db8::1", "::ffff:203.0.113.7",
                                  "100.63.255.255", None, "", "not-an-address"])
def test_a_public_or_unknown_peer_is_not(peer):
    assert not allowed(peer)


def test_the_peer_must_be_known():
    d = open_guard.decide(None, [])
    assert not d.ok and d.reason == "unknown-peer"


@pytest.mark.parametrize("name", ["cf-connecting-ip", "cf-ray", "Cf-Visitor", "CF-IPCountry",
                                  "cdn-loop", "true-client-ip", "fly-client-ip", "via",
                                  "x-envoy-external-address", "ngrok-trace-id",
                                  "x-amzn-trace-id", "fastly-client-ip",
                                  "x-original-forwarded-for"])
def test_the_marks_of_a_public_proxy_refuse_whatever_the_peer(name):
    d = open_guard.decide("127.0.0.1", [(name, "1")])
    assert not d.ok and d.reason.startswith("proxy-header")


@pytest.mark.parametrize("value", [PUBLIC, "198.51.100.4, 100.64.1.2", "100.64.1.2, 203.0.113.7",
                                   "2001:db8::5", "unknown", "_hidden", "", "garbage, 10.0.0.1",
                                   "[2001:db8::5]:443", "203.0.113.7:1234"])
def test_a_public_or_unreadable_forwarded_address_refuses(value):
    for name in ("x-forwarded-for", "x-real-ip", "x-client-ip", "x-cluster-client-ip"):
        assert not open_guard.decide("127.0.0.1", [(name, value)]).ok, (name, value)


def test_forwarded_header_for_tokens_are_read():
    assert not allowed("127.0.0.1", forwarded='for=203.0.113.7;proto=https')
    assert not allowed("127.0.0.1", forwarded='for="[2001:db8::5]:4711"')
    assert not allowed("127.0.0.1", forwarded='by=10.0.0.1, for=_x')
    assert not allowed("127.0.0.1", forwarded='for=100.64.0.1, for=203.0.113.7')
    assert allowed("127.0.0.1", forwarded='for=100.64.0.1;proto=https;host=myhost.ts.net')


def test_what_tailscale_serve_sends_is_let_in():
    """`tailscale serve` -> 127.0.0.1: X-Forwarded-For is the tailnet address of the browser."""
    assert open_guard.decide("127.0.0.1", [
        ("host", "127.0.0.1:8501"), ("x-forwarded-for", "100.101.102.103"),
        ("x-forwarded-host", "myhost.example.ts.net"), ("x-forwarded-proto", "https"),
        ("tailscale-user-login", "someone@example.com"), ("origin", "https://myhost.example.ts.net"),
    ]).ok
    # and the same when uvicorn has already put that address in as the peer
    assert allowed("100.101.102.103", x_forwarded_for="100.101.102.103",
                   x_forwarded_host="myhost.example.ts.net")


def test_a_quick_tunnel_is_refused_in_every_way_it_shows_itself():
    """What cloudflared sends to the app: a public name, the visitor's address."""
    quick = [("host", "random-words.trycloudflare.com"), ("x-forwarded-for", PUBLIC),
             ("cf-connecting-ip", PUBLIC), ("cf-ray", "abc-NRT"), ("cdn-loop", "cloudflared")]
    assert not open_guard.decide("127.0.0.1", quick).ok
    # each mark alone is enough
    for one in quick:
        assert not open_guard.decide("127.0.0.1", [one]).ok, one


@pytest.mark.parametrize("host", ["random-words.trycloudflare.com", "example.org",
                                  "203.0.113.7", "203.0.113.7:8501", "[2001:db8::1]:8501",
                                  "abc.ngrok-free.app"])
def test_a_public_name_in_host_or_forwarded_host_refuses(host):
    assert not allowed("127.0.0.1", host=host)
    assert not allowed("127.0.0.1", host="127.0.0.1:8501", x_forwarded_host=host)


@pytest.mark.parametrize("host", ["127.0.0.1:8501", "localhost:8501", "127.0.0.1:8501",
                                  "myhost", "myhost:8501", "myhost.example.ts.net",
                                  "MYHOST.Example.TS.NET:443", "nas.local", "box.lan",
                                  "x.internal", "[::1]:8501", "[fd7a:115c:a1e0::1]:8501",
                                  "192.168.1.20"])
def test_a_private_host_is_fine(host):
    assert allowed("127.0.0.1", host=host)


def test_no_host_header_is_not_judged():
    assert open_guard.decide("127.0.0.1", []).ok


def test_repeated_headers_are_all_checked():
    assert not open_guard.decide("127.0.0.1", [("x-forwarded-for", "10.0.0.1"),
                                               ("x-forwarded-for", PUBLIC)]).ok


def test_the_reason_never_carries_an_address():
    d = open_guard.decide(PUBLIC, [])
    assert PUBLIC not in d.reason


# --- the screens' side: what Streamlit knows of a browser session ---------------------------

class FakeContext:
    def __init__(self, peer, headers):
        self.remote_ip, self.headers = peer, headers
        self.cookies = {}


def test_streamlit_view_reads_the_clients_context(monkeypatch):
    import streamlit.runtime.context as sc
    monkeypatch.setattr(sc, "_get_client_context",
                        lambda: FakeContext("127.0.0.1", [("Host", "127.0.0.1:8501")]))
    peer, headers = open_guard.streamlit_view()
    assert peer == "127.0.0.1" and headers == [("Host", "127.0.0.1:8501")]
    monkeypatch.setattr(sc, "_get_client_context", lambda: None)
    assert open_guard.streamlit_view() == (None, [])
    assert not open_guard.streamlit_allows()      # no context: closed, never open


def test_streamlit_private_accessor_still_exists():
    """The raw peer (st.context.ip_address hides loopback) comes from a private Streamlit
    function; an upgrade that removes it must fail here, not silently."""
    import streamlit.runtime.context as sc
    assert callable(sc._get_client_context)


# --- the upload endpoint's side -------------------------------------------------------------

def make_request(peer, headers, method="POST", path="/pfb-upload"):
    scope = {"type": "http", "method": method, "path": path, "query_string": b"",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers],
             "client": (peer, 50000) if peer else None, "server": ("127.0.0.1", 8501),
             "scheme": "http"}
    return Request(scope)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def env_open(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3", auth_enabled=False)
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir),
                                   "auth_enabled": False}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    repo.create_assignment(WORK, "document", "調べてまとめる", "観点A", "standard", False)
    auth.create_user(repo, "kanri", "管理 一郎", "admin", PW)
    upload_api.routes(settings)          # the endpoint reads these settings
    return settings, repo


def test_the_upload_endpoint_follows_the_rule(env_open):
    settings, _ = env_open
    token = cu.issue_token(settings)
    headers = [(upload_api.TOKEN_HEADER, token), ("host", "127.0.0.1:8501"),
               ("content-length", "2")]

    async def ask(peer, extra=()):
        request = make_request(peer, headers + list(extra))
        return await upload_api._guarded(request, _ok)

    async def _ok(_token):
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    assert run(ask("127.0.0.1")).status_code == 200            # tailnet, direct
    assert run(ask("127.0.0.1")).status_code == 200
    assert run(ask("127.0.0.1", [("cf-ray", "x")])).status_code == 401      # tunnel marks
    assert run(ask("127.0.0.1", [("x-forwarded-for", PUBLIC)])).status_code == 401
    assert run(ask(PUBLIC)).status_code == 401
    assert run(ask(None)).status_code == 401
    r = run(ask(PUBLIC))
    assert upload_api.SIGNED_OUT in r.body.decode()


def test_with_the_gate_on_the_rule_is_not_involved(env_open):
    settings, _ = env_open
    closed = settings.model_copy(update={"auth_enabled": True})
    assert not upload_api.may_upload(closed, None, direct=True)
    assert upload_api.may_upload(settings, None, direct=True)
    assert not upload_api.may_upload(settings, None, direct=False)   # open, but not direct


def test_a_real_session_still_uploads_when_open_mode_is_refused(env_open):
    """Refused for open mode, a signed-in person is still the gate's person."""
    settings, repo = env_open
    user = auth.find_user(repo, "kanri")
    session = auth.check_cookie(repo, settings, auth.start_session(repo, settings, user))
    assert upload_api.may_upload(settings, session.id, direct=False)


# --- the screens ----------------------------------------------------------------------------

def open_app(monkeypatch, peer, headers=()):
    monkeypatch.setattr(open_guard, "streamlit_view", lambda: (peer, list(headers)))
    monkeypatch.setattr(auth_ui, "_browser_cookie", lambda: None)
    return AppTest.from_file(str(UI / "app.py"), default_timeout=60).run()


def test_a_direct_tailnet_browser_gets_the_open_app(env_open, monkeypatch):
    at = ok(open_app(monkeypatch, "127.0.0.1", [("host", "127.0.0.1:8501")]))
    assert WORK in screen_text(at) + html(at)
    assert auth_ui.OPEN_NOTE in screen_text(at)


@pytest.mark.parametrize("peer,headers", [
    (PUBLIC, []),
    ("127.0.0.1", [("cf-ray", "abc"), ("x-forwarded-for", PUBLIC),
                   ("host", "words.trycloudflare.com")]),
    ("127.0.0.1", [("x-forwarded-for", PUBLIC)]),
    (None, []),
])
def test_a_tunnelled_browser_gets_the_sign_in_gate_and_no_data(env_open, monkeypatch, peer, headers):
    at = ok(open_app(monkeypatch, peer, headers))
    text = screen_text(at) + html(at)
    assert "ログイン" in text
    assert WORK not in text
    assert auth_ui.OPEN_NOTE not in text


def test_a_refused_browser_on_a_fresh_install_sees_the_setup_screen_not_the_app(
        tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3", auth_enabled=False)
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir),
                                   "auth_enabled": False}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    monkeypatch.setattr(auth_ui, "_HINTED", [])    # the once-per-process log line is not ours to use up
    at = ok(open_app(monkeypatch, PUBLIC))
    assert auth_ui.SETUP_TITLE in screen_text(at) + html(at)


def test_a_signed_in_person_keeps_the_normal_box_when_open_mode_refuses(env_open, monkeypatch):
    settings, repo = env_open
    token = auth.start_session(repo, settings, auth.find_user(repo, "kanri"))
    monkeypatch.setattr(open_guard, "streamlit_view", lambda: (PUBLIC, []))
    monkeypatch.setattr(auth_ui, "_browser_cookie", lambda: token)
    at = ok(AppTest.from_file(str(UI / "app.py"), default_timeout=60).run())
    assert auth_ui.OPEN_NOTE not in screen_text(at)
    assert WORK in screen_text(at) + html(at)
