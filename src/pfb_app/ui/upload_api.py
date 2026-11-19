"""The resumable upload endpoint, served by the same server and origin as the screens
(services.chunked_upload holds the rules; this is only HTTP around it).

    POST   /pfb-upload            {"name", "size"}       -> 201 {"id", "offset", "size", "chunk"}
    GET    /pfb-upload/<id>  (also HEAD)                 -> {"offset", "size", "complete"}
    PATCH  /pfb-upload/<id>   Upload-Offset: n, body     -> 200 {"offset", "complete"}
    DELETE /pfb-upload/<id>                              -> 204

Every request carries the page's token in `X-PFB-Token` (a header, so a page on another site
cannot send it, and nothing is kept in a cookie). The token carries the signed-in person's auth
session (services.auth): every request checks that it is still valid and may add work, else 401. The Origin and the token are checked before
any byte of the body is read; a create's JSON needs a Content-Length of at most JSON_MAX;
every body is read in capped pieces, each within BODY_TIMEOUT_SEC; at most
MAX_PATCHES_PER_OWNER bodies per token are read at once (429 beyond, the browser retries).
A request whose Origin is another host is refused (X-Forwarded-Host counts only from the
local proxy). Errors are JSON {"error": <plain Japanese>} with 401 (token), 404 (unknown id), 409
(wrong offset; "offset" says where to go on), 413/400 (refused). Disk writes run in a worker
thread, never on the server's event loop (the screens' websockets share it).

`mount()` gives ui/serve.py (st.App) the routes; `MOUNTED` tells the screens the endpoint is
there (with plain `streamlit run ui/app.py` it is not, and they keep Streamlit's uploader).
"""
import json
import time
from urllib.parse import urlsplit

import anyio

from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from pfb_app.config import Settings, load_settings
from pfb_app.services import auth
from pfb_app.services import chunked_upload as cu
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect
from pfb_app.ui import open_guard

PREFIX = "/pfb-upload"
TOKEN_HEADER = "x-pfb-token"
CHUNK_BYTES = 8 * 1024 * 1024        # what the browser sends per request
MOUNTED = False
FORBIDDEN = "この画面からだけ送れます。"
SLOW = "通信が途中で止まりました。続きから送り直します。"
BUSY = "同時に送れる量を超えました。少し待って続きから送ります。"
SIGNED_OUT = "ログインし直してから、もう一度ファイルを選んでください。"
JSON_MAX = 4096                    # bytes of a create request's JSON
BODY_TIMEOUT_SEC = 120             # per piece of a body, and the least for a whole body
BODY_MIN_RATE = 100_000            # bytes/s: a whole body's limit grows with its length
MAX_PATCHES_PER_OWNER = 4          # bodies read at once per token (4 cameras)
_sending: dict[str, int] = {}      # token owner -> bodies being read now (one event loop)

_settings: Settings | None = None


def _get_settings() -> Settings:
    global _settings
    if _settings is None:
        import os   # noqa: PLC0415
        path = os.environ.get("PFB_CONFIG")
        _settings = load_settings(path) if path else Settings()
    return _settings


def _error(status: int, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": message, **extra}, status_code=status,
                        headers={"Cache-Control": "no-store"})


def _same_origin(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True          # same-origin fetches of some browsers, and tools: the token decides
    hosts = {request.headers.get("host")}
    # behind `tailscale serve` the browser's host comes as X-Forwarded-Host from the local
    # proxy (it may rewrite Host to 127.0.0.1:<port>); from anyone else it is not believed
    client = request.client.host if request.client else ""
    if client in ("127.0.0.1", "::1", "localhost"):
        hosts.add(request.headers.get("x-forwarded-host"))
    hosts = {h.split(",")[0].strip().lower() for h in hosts if h}
    return urlsplit(origin).netloc.lower() in hosts


def body_deadline_sec(declared: int | None) -> float:
    """The whole body's time limit: BODY_TIMEOUT_SEC, or longer for a big declared body at
    BODY_MIN_RATE bytes per second (8 MiB: about 84 s, so 120 s)."""
    return max(float(BODY_TIMEOUT_SEC), (declared or 0) / BODY_MIN_RATE)


async def _read_capped(request: Request, limit: int) -> bytes | None:
    """The body, read piece by piece, or None once it passes `limit` bytes. Each piece must
    arrive within BODY_TIMEOUT_SEC and the whole body within `body_deadline_sec` (a sender
    that trickles a byte now and then is cut off too): TimeoutError."""
    length = request.headers.get("content-length", "")
    deadline = time.monotonic() + body_deadline_sec(int(length) if length.isdigit() else None)
    data = bytearray()
    stream = request.stream().__aiter__()
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError
        try:
            with anyio.fail_after(min(BODY_TIMEOUT_SEC, left)):
                part = await stream.__anext__()
        except StopAsyncIteration:
            return bytes(data)
        data += part
        if len(data) > limit:
            return None


def _call(func, *args):
    return run_in_threadpool(func, *args)


def may_upload(settings: Settings, session_id: int | None, direct: bool = True) -> bool:
    """The token's auth session is still valid and its person may add work (管理者,
    評価する人): signing out, a new password, another role or 「使えなくする」 stop it at once.
    Open mode (auth_enabled: false) has no sessions to check: a signed token may upload, but
    only when the request is `direct` (ui.open_guard: from a private / tailnet / loopback peer
    without proxy marks); otherwise it is judged by its session as with the gate on."""
    if not settings.auth_enabled and direct:
        return True
    if session_id is None:
        return False
    conn = connect(files.db_path(settings))
    try:
        user = auth.session_user(Repo(conn, settings), session_id)
    finally:
        conn.close()
    return user is not None and auth.can(user.role, "edit")


async def _guarded(request: Request, handler):
    """Origin and token first: nothing of the body is read for a request that fails them."""
    if not _same_origin(request):
        return _error(403, FORBIDDEN)
    token = request.headers.get(TOKEN_HEADER, "")
    if cu.check_token(_get_settings(), token) is None:
        return _error(401, cu.BAD_TOKEN)
    direct = _get_settings().auth_enabled or open_guard.request_allows(request)
    if not await _call(may_upload, _get_settings(), cu.token_session(_get_settings(), token),
                       direct):
        return _error(401, SIGNED_OUT)
    try:
        return await handler(token)
    except TimeoutError:
        return _error(408, SLOW)
    except cu.BadToken as e:
        return _error(401, str(e))
    except cu.UnknownUpload as e:
        return _error(404, str(e))
    except cu.OffsetMismatch as e:
        return _error(409, str(e), offset=e.offset)
    except cu.UploadRefused as e:
        return _error(413 if "大きすぎ" in str(e) or "多すぎ" in str(e) else 400, str(e))


async def create(request: Request):
    async def handle(token: str):
        length = request.headers.get("content-length", "")
        if not length.isdigit() or int(length) > JSON_MAX:
            return _error(413 if length.isdigit() else 411, cu.WRONG_TYPE)
        raw = await _read_capped(request, JSON_MAX)
        try:
            body = json.loads(raw or b"")
            name, size = str(body["name"]), int(body["size"])
        except (ValueError, KeyError, TypeError):
            return _error(400, cu.WRONG_TYPE)
        up = await _call(cu.create, _get_settings(), token, name, size)
        return JSONResponse({"id": up.id, "offset": up.offset, "size": up.size,
                             "chunk": CHUNK_BYTES}, status_code=201)
    return await _guarded(request, handle)


async def upload(request: Request):
    uid = request.path_params["uid"]

    async def head(token: str):
        up = await _call(cu.status, _get_settings(), token, uid)
        return JSONResponse({"offset": up.offset, "size": up.size, "complete": up.complete},
                            headers={"Cache-Control": "no-store"})

    async def patch(token: str):
        await _call(cu.status, _get_settings(), token, uid)     # owner check before reading
        try:
            offset = int(request.headers.get("upload-offset", ""))
        except ValueError:
            return _error(400, cu.PAST_END)
        if offset < 0:
            return _error(400, cu.NEGATIVE_OFFSET)
        length = request.headers.get("content-length")
        if length is not None and (not length.isdigit() or int(length) > cu.CHUNK_MAX):
            return _error(413, cu.CHUNK_TOO_BIG)
        owner = cu.check_token(_get_settings(), token)
        if _sending.get(owner, 0) >= MAX_PATCHES_PER_OWNER:
            return _error(429, BUSY)
        _sending[owner] = _sending.get(owner, 0) + 1
        try:
            data = await _read_capped(request, cu.CHUNK_MAX)
        finally:
            _sending[owner] -= 1
            if not _sending[owner]:
                del _sending[owner]
        if data is None:
            return _error(413, cu.CHUNK_TOO_BIG)
        new = await _call(cu.append, _get_settings(), token, uid, offset, data)
        up = await _call(cu.status, _get_settings(), token, uid)
        return JSONResponse({"offset": new, "complete": up.complete})

    async def delete(token: str):
        await _call(cu.cancel, _get_settings(), token, uid)
        return Response(status_code=204)

    handler = {"GET": head, "HEAD": head, "PATCH": patch, "DELETE": delete}.get(request.method)
    if handler is None:
        return _error(405, FORBIDDEN)
    return await _guarded(request, handler)


def routes(settings: Settings | None = None) -> list[Route]:
    """The endpoint's routes; `settings` for tests (else PFB_CONFIG, as the screens read it)."""
    global _settings
    if settings is not None:
        _settings = settings
    return [Route(PREFIX, create, methods=["POST"]),
            Route(PREFIX + "/{uid}", upload, methods=["GET", "HEAD", "PATCH", "DELETE"])]


def mount() -> list[Route]:
    """`routes()` for ui/serve.py, and the mark that tells this process's screens to use them."""
    global MOUNTED
    MOUNTED = True
    return routes()
