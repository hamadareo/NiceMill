"""Text-only chat against an already running OpenAI-compatible server (the PFB3x3 session's
own vLLM on localhost). Stdlib only; JSON-schema constrained output, temperature 0.

Used to translate the engine's English observations; never reaches a cloud service: the base
URL must name this machine (the session's ServerHandle.base_url), anything else is refused,
and requests ignore proxy settings and never follow redirects.

Failures are per request: one bad request never discards the answers of the others.
- An HTTP error answer is reported with its status and the start of its body. On 400/422 the
  request is sent once more without `response_format` (a server that rejects the JSON-schema
  option), and the answer is read leniently (code fences and prose around the JSON removed).
- A request that still fails comes back as a RawResponse with `error` set (never cached by
  pfb_app.inference.structured, so a rerun asks again).
- Only when the server cannot be reached at all for every request of a call is an exception
  raised (the caller then records one warning instead of retrying each request).
"""
import json
import logging
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from pfb_app.inference.backend import ChatRequest, RawResponse

log = logging.getLogger(__name__)

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ERROR_BODY_CHARS = 300
RETRY_WITHOUT_SCHEMA = (400, 422)   # the server rejected the request as such
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


class ServedHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body}" if body else f"HTTP {status}")
        self.status = status


class ServerFailed(ConnectionError):
    """The server itself failed a request (strict_server): unreachable, a timeout or a 5xx
    answer (e.g. vLLM out of CUDA memory), as opposed to refusing that one request (4xx)."""


class ServerUnreachable(ServerFailed):
    """No request of a call reached the server."""


# the same instruction the in-process TransformersBackend gives (schema_hint)
SCHEMA_HINT = ("\n\n次のJSONスキーマに従うJSONのみを出力してください。"
               "説明文やコードブロックは付けないでください。\n")


def lenient_json(text: str) -> str:
    """`text` if it is JSON; else the JSON inside a code fence or between the first "{" and the
    last "}" when that parses; else `text` unchanged (the caller reports it)."""
    stripped = text.strip()
    candidates = [stripped] + [m.strip() for m in _FENCE.findall(stripped)]
    first, last = stripped.find("{"), stripped.rfind("}")
    if 0 <= first < last:
        candidates.append(stripped[first:last + 1])
    for c in candidates:
        try:
            json.loads(c)
        except ValueError:
            continue
        return c
    return text


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect could point anywhere: it is an error, never followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def local_opener() -> urllib.request.OpenerDirector:
    """An opener for this machine only: no proxies (http_proxy etc. are ignored), no
    redirects."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _unreachable(e: BaseException) -> bool:
    if isinstance(e, urllib.error.HTTPError):
        return False
    if isinstance(e, urllib.error.URLError):
        return isinstance(e.reason, ConnectionError)
    return isinstance(e, ConnectionError)


def _timed_out(e: BaseException) -> bool:
    """The answer did not come within the client timeout (connecting or reading); the server
    may still be fine (one long generation among many), unlike a refused connection or a 5xx."""
    if isinstance(e, urllib.error.URLError) and not isinstance(e, urllib.error.HTTPError):
        return isinstance(e.reason, TimeoutError)
    return isinstance(e, TimeoutError)


def _server_fault(e: BaseException) -> bool:
    """The server, not the request, failed: unreachable, timed out, or a 5xx answer."""
    if isinstance(e, ServedHTTPError):
        return e.status >= 500
    if isinstance(e, urllib.error.URLError):
        return isinstance(e.reason, (ConnectionError, TimeoutError, OSError))
    return isinstance(e, (ConnectionError, TimeoutError))


def list_models(base_url: str, timeout: float = 5.0) -> list[str]:
    """The model names a local server lists (GET /models). Only this machine is asked."""
    if urllib.parse.urlsplit(base_url).hostname not in LOCAL_HOSTS:
        raise ValueError(f"only this machine can be asked for its models, not {base_url!r}")
    req = urllib.request.Request(base_url.rstrip("/") + "/models")
    with local_opener().open(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [str(d.get("id") or "") for d in data.get("data") or []]


class ServedChatBackend:
    def __init__(self, base_url: str, model: str, *, timeout_sec: float = 300.0,
                 max_concurrency: int = 8, schema_hint: bool = False,
                 strict_server: bool = False):
        """schema_hint: the user text also states the JSON schema (SCHEMA_HINT), as the
        in-process backend does, so the answer follows it even when the server rejects
        `response_format`. strict_server: a request the server itself failed (_server_fault)
        raises ServerFailed for the whole call instead of being one failed answer; a timeout
        counts only when the request timed out twice in a row."""
        host = urllib.parse.urlsplit(base_url).hostname
        if host not in LOCAL_HOSTS:
            raise ValueError(f"ServedChatBackend only talks to this machine, not {base_url!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_id = f"served:{model}"
        self._timeout = timeout_sec
        self._workers = max_concurrency
        self._schema_hint = schema_hint
        # part of the call_log key: an answer given without the schema text is never reused
        # for a request that carries it (and the other way round)
        self.cache_variant = "schema_hint" if schema_hint else None
        self._strict = strict_server
        self._opener = local_opener()

    def _post(self, body: dict) -> dict:
        http = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self._opener.open(http, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                text = e.read().decode("utf-8", errors="replace")
            except OSError:
                text = ""
            raise ServedHTTPError(e.code, " ".join(text.split())[:ERROR_BODY_CHARS]) from None

    def _one(self, req: ChatRequest) -> RawResponse:
        text = req.user_text
        if self._schema_hint:
            text += SCHEMA_HINT + json.dumps(req.json_schema, ensure_ascii=False)
        body = {
            "model": self.model, "temperature": 0, "max_tokens": req.max_tokens,
            "messages": [{"role": "system", "content": req.system},
                         {"role": "user", "content": text}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "output", "schema": req.json_schema}},
        }
        start = time.monotonic()
        try:
            out = self._post(body)
        except ServedHTTPError as e:
            if e.status not in RETRY_WITHOUT_SCHEMA:
                raise
            log.warning("served model refused the request (%s); retrying without response_format",
                        e)
            out = self._post({k: v for k, v in body.items() if k != "response_format"})
        usage = out.get("usage") or {}
        message = (out.get("choices") or [{}])[0].get("message") or {}
        return RawResponse(text=lenient_json(message.get("content") or ""),
                           prompt_tokens=usage.get("prompt_tokens"),
                           completion_tokens=usage.get("completion_tokens"),
                           duration_sec=time.monotonic() - start)

    def _isolated(self, req: ChatRequest) -> RawResponse | Exception:
        """One request; its failure is returned, not raised (the others keep their answers).
        strict_server: a timeout is asked once more before it counts as a server failure, so
        one slow answer among concurrent requests does not end the session."""
        try:
            return self._one(req)
        except Exception as e:
            if not (self._strict and _timed_out(e)):
                return e
            log.warning("served model request timed out (%s); asking once more", e)
        try:
            return self._one(req)
        except Exception as e:
            return e

    def generate(self, reqs: list[ChatRequest]) -> list[RawResponse]:
        if any(r.image_paths for r in reqs):
            raise ValueError("ServedChatBackend is text-only")
        if len(reqs) <= 1:
            outcomes = [self._isolated(r) for r in reqs]
        else:
            with ThreadPoolExecutor(max_workers=min(self._workers, len(reqs))) as pool:
                outcomes = list(pool.map(self._isolated, reqs))
        failures = [o for o in outcomes if isinstance(o, Exception)]
        if failures and len(failures) == len(outcomes) and all(map(_unreachable, failures)):
            raise ServerUnreachable(f"{self.base_url}: {failures[0]}") from failures[0]
        fault = next((f for f in failures if _server_fault(f)), None) if self._strict else None
        if fault is not None:
            raise ServerFailed(f"{self.base_url}: {type(fault).__name__}: {fault}") from fault
        results = []
        for o in outcomes:
            if isinstance(o, Exception):
                message = f"{type(o).__name__}: {o}"
                log.warning("served model request failed: %s", message)
                results.append(RawResponse(text="", prompt_tokens=None, completion_tokens=None,
                                           duration_sec=0.0, error=message))
            else:
                results.append(o)
        return results

    def close(self) -> None:
        pass   # the server belongs to the session
