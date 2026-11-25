"""The safety rail of open mode (`auth_enabled: false`, docs/accounts.md): the open session
is served only to a request that arrives directly from a private, tailnet or loopback peer
without the marks of a public proxy. Everything else meets the normal sign-in gate (and a 401
at /pfb-upload). It is there so that a public tunnel or proxy in front of the screen port cannot
make open mode (everyone an administrator) reachable from the internet.

`decide(peer, headers)` is a pure function of the TCP peer and the request headers:

1. the peer must be an IP address in 127.0.0.0/8, ::1, 10/8, 172.16/12, 192.168/16,
   169.254/16, 100.64/10 (Tailscale), fc00::/7 or fe80::/10 (an IPv4-mapped IPv6 address
   counts as its IPv4); an unknown or unreadable peer is refused;
2. a header that only a public proxy / CDN / tunnel adds refuses (`cf-*`, `cdn-loop`,
   `true-client-ip`, `via`, `x-envoy-*`, `ngrok-*`, `fly-*`, `fastly-*`, `x-amzn-*`,
   `x-azure-*`, `x-original-forwarded-for`, ...);
3. every address a proxy reports (`X-Forwarded-For`, `Forwarded: for=`, `X-Real-IP`,
   `X-Client-IP`, `X-Cluster-Client-IP`) must itself be private; an unreadable one
   ("unknown", "_hidden", empty) refuses. `tailscale serve` passes: it connects from
   127.0.0.1 and reports the tailnet address (100.x) of the browser;
4. the name the browser used (`Host`, `X-Forwarded-Host`) must be internal: a private IP,
   `localhost`, a name without a dot, or one ending in .ts.net / .local / .lan / .internal /
   .home.arpa / .localdomain / .intranet.

Limits, stated plainly (docs/accounts.md): this decides from what the request itself says.
A tunnel that connects from the machine itself, keeps `Host` as an internal name and adds no
header at all (for example `ssh -R` or a bare TCP forwarder) looks exactly like a person on the
tailnet; the rail cannot see it. What it closes is every HTTP proxy, CDN and tunnel that
identifies itself, which includes a Cloudflare quick tunnel. A refusal logs
only its reason, never an address. The first run of a browser session decides (the page's
websocket handshake); the upload endpoint decides per request.
"""
import ipaddress
import logging
import time
from collections.abc import Iterable
from typing import NamedTuple
from urllib.parse import unquote

log = logging.getLogger(__name__)

PRIVATE = tuple(ipaddress.ip_network(n) for n in (
    "127.0.0.0/8", "::1/128", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "169.254.0.0/16", "100.64.0.0/10", "fc00::/7", "fe80::/10"))
INTERNAL_SUFFIXES = (".ts.net", ".local", ".lan", ".internal", ".home.arpa", ".localdomain",
                     ".intranet")
# headers only a public proxy / CDN / tunnel adds
PROXY_PREFIXES = ("cf-", "x-envoy-", "ngrok-", "fly-", "fastly-", "x-amzn-", "x-azure-",
                  "x-vercel-", "x-cloud-trace", "x-akamai-", "akamai-")
PROXY_NAMES = frozenset({"cdn-loop", "true-client-ip", "via", "tailscale-funnel-request", "x-original-forwarded-for",
                         "x-forwarded-for-original", "x-proxyuser-ip", "x-cluster-client-ip-original"})
ADDRESS_HEADERS = ("x-forwarded-for", "x-real-ip", "x-client-ip", "x-cluster-client-ip")
HOST_HEADERS = ("host", "x-forwarded-host")
LOG_EVERY_SEC = 600.0
_logged: dict[str, float] = {}


class Decision(NamedTuple):
    ok: bool
    reason: str      # "" when ok; a short code, never an address


def _private_address(text: str) -> bool:
    """An address (optionally with a port, in brackets for IPv6) that is private."""
    text = text.strip().strip('"')
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return False
        text = text[1:end]
    elif text.count(":") == 1:
        text = text.split(":", 1)[0]
    text = text.split("%", 1)[0]
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip in net for net in PRIVATE)


def _internal_host(value: str) -> bool:
    host = value.strip().lower()
    if not host:
        return True
    if host.startswith("["):
        end = host.find("]")
        return end > 0 and _private_address(host[:end + 1])
    if host.count(":") == 1:
        host = host.split(":", 1)[0]
    elif host.count(":") > 1:       # a bare IPv6 literal
        return _private_address(host)
    host = host.rstrip(".")
    if not host:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return _private_address(host)
    return host == "localhost" or "." not in host or host.endswith(INTERNAL_SUFFIXES)


def _forwarded_for(value: str) -> list[str]:
    """The `for=` values of a `Forwarded` header (all hops)."""
    out = []
    for hop in value.split(","):
        found = False
        for part in hop.split(";"):
            key, _, val = part.strip().partition("=")
            if key.strip().lower() == "for":
                out.append(unquote(val.strip()))
                found = True
        if not found and hop.strip():
            out.append("")      # a hop that says nothing of who: not believed
    return out or [""]


def decide(peer: str | None, headers: Iterable[tuple[str, str]]) -> Decision:
    """Whether the open session may be served to this request (see the module docstring)."""
    if not peer:
        return Decision(False, "unknown-peer")
    if not _private_address(peer):
        return Decision(False, "public-peer")
    pairs = [(k.strip().lower(), v) for k, v in headers]
    for name, _ in pairs:
        if name in PROXY_NAMES or name.startswith(PROXY_PREFIXES):
            return Decision(False, f"proxy-header:{name}")
    for name, value in pairs:
        if name in ADDRESS_HEADERS:
            hops = value.split(",")
        elif name == "forwarded":
            hops = _forwarded_for(value)
        else:
            continue
        if not all(_private_address(h) for h in hops):
            return Decision(False, f"public-forwarded-address:{name}")
    for name, value in pairs:
        if name in HOST_HEADERS and not all(_internal_host(h) for h in value.split(",")):
            return Decision(False, f"public-host:{name}")
    return Decision(True, "")


def _note(reason: str) -> None:
    """One log line per reason per LOG_EVERY_SEC (a refused browser reruns the script often)."""
    now = time.monotonic()
    if now - _logged.get(reason, -1e9) >= LOG_EVERY_SEC:
        _logged[reason] = now
        log.warning("open mode refused a request (%s): it meets the sign-in gate instead", reason)


def allows(peer: str | None, headers: Iterable[tuple[str, str]]) -> bool:
    decision = decide(peer, headers)
    if not decision.ok:
        _note(decision.reason)
    return decision.ok


def streamlit_view() -> tuple[str | None, list[tuple[str, str]]]:
    """(TCP peer, handshake headers) of this browser session, from Streamlit's client
    context. The raw peer is needed (`st.context.ip_address` hides loopback), so this reads
    Streamlit's own accessor; tests/test_open_guard.py fails if an upgrade removes it. No
    context (a bare script run, no browser) gives (None, []): refused."""
    try:
        from streamlit.runtime.context import _get_client_context   # noqa: PLC0415
        context = _get_client_context()
        if context is None:
            return None, []
        peer = context.remote_ip
        return (peer if isinstance(peer, str) else None,
                [(str(k), str(v)) for k, v in context.headers])
    except Exception:   # noqa: BLE001 - closed, never open
        return None, []


def streamlit_allows() -> bool:
    """The rule for this browser session (the screens)."""
    peer, headers = streamlit_view()
    return allows(peer, headers)


def request_allows(request) -> bool:
    """The rule for a Starlette request (the upload endpoint)."""
    client = getattr(request, "client", None)
    return allows(client.host if client else None, request.headers.items())
