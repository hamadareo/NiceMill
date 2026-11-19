#!/usr/bin/env bash
# Write the systemd --user units that keep the PFB app running (pfb_app.ops.systemd):
#   <prefix>-worker.service, <prefix>-ui.service   restart on crash; start at boot (lingering)
#   <prefix>-backup.service + .timer                nightly backup of data/app.db (03:30)
#   <prefix>-guard.service + .timer                 status watch every 5 min (ops_status.txt)
#   <prefix>-alert@.service                         operator alert when a unit gives up
#
# By default this only WRITES the files (~/.config/systemd/user) and reloads systemd: nothing is
# enabled or started. Switching the live app from tmux to systemd is a deploy step:
#   scripts/stop.sh   (then wait 90 s)   scripts/install_user_services.sh --enable
# --enable first checks the settings' data (any prefix; pfb_app.ops.systemd preflight): it
# refuses while a worker beats or holds <data_dir>/worker.lock, another process has app.db
# open (/proc/*/fd), the UI's host:port answers, or, when the data is the default (live) one,
# the tmux sessions pfb-ui / pfb-worker exist. A prefix other than pfb needs its own --config
# whose data_dir is not the default one. A relative --config is read from where you run this.
# Units written for another checkout are neither overwritten nor removed without --force.
# Start at boot without anyone logged in needs lingering, once:  loginctl enable-linger "$USER"
#
# Options:
#   --unit-dir DIR   write there instead (a custom folder: systemctl is never called)
#   --prefix NAME    unit name prefix (default pfb; a test instance uses another one)
#   --config YAML    settings for the worker, the UI and the backup (PFB_CONFIG)
#   --host H --port P --max-upload-mb N   the UI's address (default as scripts/start.sh:
#                    PFB_UI_HOST / PFB_UI_PORT, else 127.0.0.1:8501; --host 127.0.0.1 or
#                    PFB_UI_HOST=127.0.0.1 for HTTPS through tailscale serve, docs/https.md)
#   --entry serve|app  the UI entry (default serve: ui/serve.py with the resumable upload;
#                    app: ui/app.py alone), as start.sh's PFB_UI_ENTRY
#   --python PATH    interpreter (default <app>/.venv/bin/python)
#   --enable         enable and start worker, UI and the backup timer
#   --force          overwrite / remove units written for another app folder
#   --uninstall      stop, disable and remove this prefix's units
#   --dry-run        print the units, write nothing
set -euo pipefail
CALLER_PWD="$PWD"   # a relative --config / PFB_CONFIG is the caller's, not the app folder's
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
UNIT_DIR="" PREFIX="pfb" CONFIG="${PFB_CONFIG:-}" MAXUP="4096" ENTRY="${PFB_UI_ENTRY:-serve}"
HOST="${PFB_UI_HOST:-127.0.0.1}" PORT="${PFB_UI_PORT:-8501}"   # as scripts/start.sh
PY="" ENABLE=0 UNINSTALL=0 DRY=0 FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --unit-dir) UNIT_DIR="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --max-upload-mb) MAXUP="$2"; shift 2 ;;
        --entry) ENTRY="$2"; shift 2 ;;
        --python) PY="$2"; shift 2 ;;
        --enable) ENABLE=1; shift ;;
        --force) FORCE=1; shift ;;
        --uninstall) UNINSTALL=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) echo "不明なオプション: $1" >&2; exit 2 ;;
    esac
done
case "$PREFIX" in *[!A-Za-z0-9-]*|"") echo "--prefix は英数字と - だけにしてください" >&2; exit 2 ;; esac
if ! [[ "$HOST" =~ ^[0-9A-Za-z.:-]+$ && "$PORT" =~ ^[0-9]+$ ]]; then
    echo "PFB_UI_HOST / PFB_UI_PORT（--host / --port）が正しくありません" >&2
    exit 2
fi
if [ "$PREFIX" != "pfb" ] && [ -z "$CONFIG" ] && [ "$UNINSTALL" = 0 ] && [ "$DRY" = 0 ]; then
    echo "--prefix $PREFIX には、その設定ファイルを --config で指定してください（既定の設定は本番のデータです）" >&2
    exit 2
fi
if [ -n "$CONFIG" ]; then
    case "$CONFIG" in /*) ;; "~/"*) CONFIG="$HOME/${CONFIG#\~/}" ;; *) CONFIG="$CALLER_PWD/$CONFIG" ;; esac
    if [ ! -f "$CONFIG" ] && [ "$UNINSTALL" = 0 ]; then
        echo "設定ファイルが見つかりません: $CONFIG" >&2
        exit 2
    fi
fi

CUSTOM_DIR=1
if [ -z "$UNIT_DIR" ]; then
    UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
    CUSTOM_DIR=0
fi
UNITS=("$PREFIX-worker.service" "$PREFIX-ui.service" "$PREFIX-backup.service"
       "$PREFIX-backup.timer" "$PREFIX-guard.service" "$PREFIX-guard.timer"
       "$PREFIX-alert@.service")

PY="${PY:-$APP_DIR/.venv/bin/python}"
if [ ! -x "$PY" ]; then
    echo "Python が見つかりません: $PY（先に uv sync を実行するか --python で指定してください）" >&2
    exit 1
fi

if [ "$UNINSTALL" = 1 ]; then
    # only this checkout's units (the same header check as writing; --force overrides)
    CHECK=(--app-dir "$APP_DIR" --prefix "$PREFIX" --out "$UNIT_DIR" --check-target)
    [ "$FORCE" = 1 ] && CHECK+=(--force)
    if ! env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.systemd "${CHECK[@]}"; then
        exit 1
    fi
    if [ "$CUSTOM_DIR" = 0 ]; then
        systemctl --user disable --now "$PREFIX-worker.service" "$PREFIX-ui.service" \
            "$PREFIX-backup.timer" "$PREFIX-guard.timer" 2>/dev/null || true
    fi
    for u in "${UNITS[@]}"; do rm -f "$UNIT_DIR/$u"; done
    [ "$CUSTOM_DIR" = 0 ] && systemctl --user daemon-reload
    echo "$PREFIX のユニットを削除しました（$UNIT_DIR）"
    exit 0
fi

ARGS=(--app-dir "$APP_DIR" --prefix "$PREFIX" --host "$HOST" --port "$PORT"
      --max-upload-mb "$MAXUP" --entry "$ENTRY" --python "$PY")
[ -n "$CONFIG" ] && ARGS+=(--config "$CONFIG")

if [ "$DRY" = 1 ]; then
    TMP="$(mktemp -d)"
    trap 'rm -rf "$TMP"' EXIT
    env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.systemd "${ARGS[@]}" --out "$TMP" >/dev/null
    for u in "${UNITS[@]}"; do echo "### $u"; cat "$TMP/$u"; done
    exit 0
fi

[ "$FORCE" = 1 ] && ARGS+=(--force)
if [ "$ENABLE" = 1 ]; then
    [ "$CUSTOM_DIR" = 1 ] && { echo "--enable は --unit-dir と一緒には使えません" >&2; exit 2; }
    # never a second worker or UI on data that is in use (whatever the prefix)
    if ! env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.systemd "${ARGS[@]}" --preflight; then
        echo "有効にしませんでした（何も書いていません）。" >&2
        exit 1
    fi
fi

env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.systemd "${ARGS[@]}" --out "$UNIT_DIR"
if [ "$CUSTOM_DIR" = 1 ]; then
    echo "ユニットのファイルを書きました（$UNIT_DIR）。systemctl は実行していません。"
    exit 0
fi
systemctl --user daemon-reload
if [ "$ENABLE" = 1 ]; then
    systemctl --user enable --now "$PREFIX-worker.service" "$PREFIX-ui.service" \
        "$PREFIX-backup.timer" "$PREFIX-guard.timer"
    echo "$PREFIX-worker / $PREFIX-ui / $PREFIX-backup.timer / $PREFIX-guard.timer を有効にして起動しました"
else
    echo "ユニットを書きました。有効にするには: $0 --enable（有効にはしていません）"
fi
if command -v loginctl >/dev/null && [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
    echo "注意: ログインしていないときやサーバーの再起動後にも動かすには、一度だけ次を実行してください:"
    echo "  loginctl enable-linger $USER"
fi
