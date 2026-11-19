"""The sign-in cookie, set by the server (HttpOnly), on the same server and origin as the
screens (ui/serve.py mounts it beside /pfb-upload):

    POST /pfb-auth/cookie   X-PFB-Ticket: <one-time ticket>   -> 204 + Set-Cookie
    POST /pfb-auth/clear                                       -> 204 + Set-Cookie (Max-Age=0)

After a sign-in the gate (ui/auth_ui) hands the page a one-time ticket (TICKET_SEC, used once,
kept in this process only), never the cookie itself; the page's script exchanges it here. The
cookie `pfb_auth_<port>` is `HttpOnly; SameSite=Strict; Path=/`, `Secure` over https (also
behind `tailscale serve`, by X-Forwarded-Proto from the local proxy), so no script can read it.
A request whose Origin is another host is refused (the upload endpoint's rule).

With plain `streamlit run ui/app.py` (no routes, `MOUNTED` False) nothing is set: a sign-in then
lasts as long as the page stays open.
"""
import hashlib
import secrets
import threading
import time

from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pfb_app.config import Settings
from pfb_app.services import auth

COOKIE_PATH = "/pfb-auth/cookie"
CLEAR_PATH = "/pfb-auth/clear"
TICKET_HEADER = "x-pfb-ticket"
TICKET_SEC = 60
MOUNTED = False
_tickets: dict[str, tuple[float, str, int]] = {}   # sha256(ticket) -> (until, cookie, max_age)
_lock = threading.Lock()


def port() -> int:
    """The screens' port (the cookie's name carries it)."""
    try:
        import streamlit as st   # noqa: PLC0415
        return int(st.get_option("server.port"))
    except Exception:   # noqa: BLE001
        return 8501


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def ticket_for(cookie: str, max_age: int) -> str:
    """A one-time ticket the page exchanges for `cookie` (within TICKET_SEC)."""
    ticket = secrets.token_urlsafe(24)
    now = time.monotonic()
    with _lock:
        for key in [k for k, v in _tickets.items() if v[0] <= now]:
            _tickets.pop(key, None)
        _tickets[_sha(ticket)] = (now + TICKET_SEC, cookie, int(max_age))
    return ticket


def _take(ticket: str) -> tuple[str, int] | None:
    if not ticket or not ticket.isascii():
        return None
    with _lock:
        held = _tickets.pop(_sha(ticket), None)
    if held is None or held[0] <= time.monotonic():
        return None
    return held[1], held[2]


def _secure(request: Request) -> bool:
    if request.url.scheme == "https":
        return True
    client = request.client.host if request.client else ""
    return client in ("127.0.0.1", "::1", "localhost") \
        and request.headers.get("x-forwarded-proto", "").split(",")[0].strip() == "https"


def _cookie(response: Response, request: Request, value: str, max_age: int) -> Response:
    response.set_cookie(auth.cookie_name(port()), value, max_age=max_age, path="/",
                        httponly=True, samesite="strict", secure=_secure(request))
    response.headers["Cache-Control"] = "no-store"
    return response


def _refused() -> JSONResponse:
    return JSONResponse({"error": "この画面からだけ使えます。"}, status_code=403,
                        headers={"Cache-Control": "no-store"})


async def set_cookie(request: Request):
    from pfb_app.ui.upload_api import _same_origin   # noqa: PLC0415 - one rule for both
    if not _same_origin(request):
        return _refused()
    held = _take(request.headers.get(TICKET_HEADER, ""))
    if held is None:
        return _refused()
    value, max_age = held
    return _cookie(Response(status_code=204), request, value, max_age)


async def clear_cookie(request: Request):
    from pfb_app.ui.upload_api import _same_origin   # noqa: PLC0415
    if not _same_origin(request):
        return _refused()
    return _cookie(Response(status_code=204), request, "", 0)


def routes(settings: Settings | None = None) -> list[Route]:
    return [Route(COOKIE_PATH, set_cookie, methods=["POST"]),
            Route(CLEAR_PATH, clear_cookie, methods=["POST"])]


def mount() -> list[Route]:
    """`routes()` for ui/serve.py, and the mark that tells this process's gate to use them."""
    global MOUNTED
    MOUNTED = True
    return routes()
