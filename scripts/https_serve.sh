#!/usr/bin/env bash
# HTTPS for the PFB app through `tailscale serve`: https://<machine>.<tailnet>.ts.net[:N]/ ->
# http://127.0.0.1:<port> (the screens and the upload endpoint, one origin). docs/https.md.
#
#   scripts/https_serve.sh check  [--port 8501] [--https-port 443]   read-only checks
#   scripts/https_serve.sh on     [--port 8501] [--https-port 443]   map / of that HTTPS port
#   scripts/https_serve.sh off    [--port 8501] [--https-port 443]   unmap it again
#   scripts/https_serve.sh status                                    tailscale serve status
#
# `on` and `off` change the machine's tailscale serve config, which is shared by everything on
# the machine: a deploy step, only with the operator's approval (also for a test port).
# `on` refuses when "/" of that HTTPS port already maps to anything other than
# http://127.0.0.1:<port> (it never replaces someone else's mapping; the same mapping is left
# as it is). `off` refuses unless "/" maps to exactly that. The current mapping is read from
# `tailscale serve status --json`. Nothing here starts or stops the app.
set -uo pipefail
TS="${TAILSCALE:-tailscale}"
cmd="${1:-check}"; shift || true
PORT=8501
HTTPS_PORT=443
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:-}"; shift 2 ;;
        --https-port) HTTPS_PORT="${2:-}"; shift 2 ;;
        *) echo "知らない引数です: $1" >&2; exit 2 ;;
    esac
done
for v in "$PORT" "$HTTPS_PORT"; do
    [[ "$v" =~ ^[0-9]+$ ]] || { echo "ポート番号が正しくありません: $v" >&2; exit 2; }
done
TARGET="http://127.0.0.1:$PORT"

# What "/" of the HTTPS port maps to now: a proxy URL, "other" (a file, text or another kind
# of handler), "tcp:<target>" (the port is a TCP forward), or "" (nothing). Exit 3 when the
# serve config cannot be read.
current_mapping() {
    local json
    json="$("$TS" serve status --json 2>/dev/null)" || return 3
    printf '%s' "$json" | HTTPS_PORT="$HTTPS_PORT" python3 -c '
import json, os, sys
raw = sys.stdin.read().strip()
data = json.loads(raw) if raw else {}
port = os.environ["HTTPS_PORT"]
tcp = (data.get("TCP") or {}).get(port) or {}
if tcp.get("TCPForward"):            # the port is a raw TCP forward: not ours either
    print("tcp:" + str(tcp["TCPForward"]))
    sys.exit(0)
for host, web in (data.get("Web") or {}).items():
    if host.rsplit(":", 1)[-1] != port:
        continue
    root = (web.get("Handlers") or {}).get("/")
    if root is None:
        continue
    print(root.get("Proxy") or "other")
    break
' || return 3
}

status_json() { "$TS" status --json 2>/dev/null; }

check() {
    local ok=0 json name domains operator
    json="$(status_json)" || { echo "NG tailscale が動いていません（tailscale status）"; return 1; }
    name="$(printf '%s' "$json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["Self"]["DNSName"].rstrip("."))')"
    domains="$(printf '%s' "$json" | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin).get("CertDomains") or []))')"
    echo "machine: $name"
    if [ -n "$domains" ]; then
        echo "OK HTTPS certificates are enabled for this tailnet: $domains"
    else
        echo "NG HTTPS certificates are off for this tailnet (CertDomains empty)."
        echo "   An admin turns on 'HTTPS Certificates' in the Tailscale admin console (DNS page)."
        echo "   Note: the certificate puts the machine name ($name) in the public CT logs."
        ok=1
    fi
    operator="$("$TS" debug prefs 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("OperatorUser") or "")' 2>/dev/null)"
    if [ "$(id -u)" = 0 ] || [ "$operator" = "${USER:-}" ]; then
        echo "OK this user may change the serve config"
    else
        echo "NG this user may not change 'tailscale serve' (once, by an admin: sudo tailscale set --operator=${USER:-<user>})"
        ok=1
    fi
    if command -v ss >/dev/null && ss -Hltn "sport = :$PORT" 2>/dev/null | grep -q .; then
        if ss -Hltn "sport = :$PORT" | awk '{print $4}' | grep -qvE '^(127\.0\.0\.1|\[::1\]):'; then
            echo "NG port $PORT also listens beyond 127.0.0.1: start the app with PFB_UI_HOST=127.0.0.1"
            ok=1
        else
            echo "OK the app listens on 127.0.0.1:$PORT only"
        fi
    else
        echo "-- nothing listens on :$PORT yet"
    fi
    local now rc
    now="$(current_mapping)"; rc=$?
    if [ $rc -ne 0 ]; then
        echo "NG the serve config cannot be read"; ok=1
    elif [ -z "$now" ]; then
        echo "-- https port $HTTPS_PORT: / is not mapped"
    elif [ "$now" = "$TARGET" ]; then
        echo "OK https port $HTTPS_PORT: / -> $TARGET"
    else
        echo "!! https port $HTTPS_PORT: / -> $now (not this app; 'on' will refuse)"
    fi
    return $ok
}

case "$cmd" in
    check) check ;;
    status) "$TS" serve status ;;
    on)
        check || { echo "準備が足りないため、何も変えていません" >&2; exit 1; }
        now="$(current_mapping)" || { echo "serve の設定を読めないため、何も変えていません" >&2; exit 1; }
        if [ "$now" = "$TARGET" ]; then
            echo "すでに https:$HTTPS_PORT の / は $TARGET です（何も変えていません）"
            exit 0
        fi
        if [ -n "$now" ]; then
            echo "https:$HTTPS_PORT の / はほかのもの（$now）に使われています。何も変えていません" >&2
            exit 1
        fi
        "$TS" serve --bg --https="$HTTPS_PORT" "$TARGET" || exit 1
        "$TS" serve status
        ;;
    off)
        now="$(current_mapping)" || { echo "serve の設定を読めないため、何も変えていません" >&2; exit 1; }
        if [ "$now" != "$TARGET" ]; then
            echo "https:$HTTPS_PORT の / は $TARGET ではありません（${now:-なし}）。何も変えていません" >&2
            exit 1
        fi
        "$TS" serve --https="$HTTPS_PORT" off && "$TS" serve status
        ;;
    *) echo "使い方: $0 check|on|off|status [--port N] [--https-port N]" >&2; exit 2 ;;
esac
