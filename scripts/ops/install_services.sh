#!/usr/bin/env bash
# Switch the app between "tmux by hand" (scripts/start.sh) and "systemd keeps it running"
# (starts by itself after a reboot / power cut, restarts after a crash), with a check first and
# an automatic way back. One method at a time: never both.
#
#   scripts/ops/install_services.sh check                       read-only: which method runs, what is missing
#   scripts/ops/install_services.sh install                     shows the plan and the unit files; changes NOTHING
#   scripts/ops/install_services.sh install --apply [--stop-tmux] [--even-if-busy]
#   scripts/ops/install_services.sh uninstall                   shows the plan; changes NOTHING
#   scripts/ops/install_services.sh uninstall --apply [--start-tmux]
#
# install --apply, in order (every step stops at the first problem):
#   1. lingering is on (else the units would only run while someone is logged in):
#        loginctl enable-linger $USER       (one-time; run it yourself, see docs/operations_ja.md)
#   2. no job is running (else --even-if-busy); the tmux sessions pfb-ui / pfb-worker are
#      stopped by scripts/stop.sh only with --stop-tmux, then it waits until the worker and the
#      screens are really gone (scripts/install_user_services.sh --enable refuses otherwise)
#   3. scripts/install_user_services.sh --enable (units: worker, ui, backup + guard timers)
#   4. waits (up to 4 min) until worker + screens are active, the screens answer and the
#      worker's beat is fresh; if not: ROLLBACK (units removed, tmux started again if this
#      run had stopped it) and exit 1
# A log of every --apply is kept in ~/pfb_deploy_logs/ops-<command>-<time>.log.
#
# Options: --config YAML (default: $PFB_CONFIG, else config/settings.yaml)  --allow-no-linger
#   --prefix NAME + --config YAML: a throwaway instance (its own unit names and data folder; the
#   live tmux sessions are never looked at or touched)
# PFB_PYTHON overrides the interpreter (default .venv/bin/python of this checkout).
set -uo pipefail
cd "$(dirname "$0")/../.."
APP_DIR="$(pwd)"
CMD="${1:-}"; [ $# -gt 0 ] && shift
APPLY=0 STOP_TMUX=0 START_TMUX=0 BUSY_OK=0 NO_LINGER_OK=0 PREFIX="pfb" CONFIG="${PFB_CONFIG:-}"
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) APPLY=1; shift ;;
        --stop-tmux) STOP_TMUX=1; shift ;;
        --start-tmux) START_TMUX=1; shift ;;
        --even-if-busy) BUSY_OK=1; shift ;;
        --allow-no-linger) NO_LINGER_OK=1; shift ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "不明なオプション: $1" >&2; exit 2 ;;
    esac
done
case "$CMD" in check|install|uninstall) ;; *) sed -n '2,10p' "$0"; exit 2 ;; esac
case "$PREFIX" in *[!A-Za-z0-9-]*|"") echo "--prefix は英数字と - だけにしてください" >&2; exit 2 ;; esac
LIVE=1; [ "$PREFIX" != "pfb" ] && LIVE=0
if [ -z "$CONFIG" ] && [ -f "$APP_DIR/config/settings.yaml" ] && [ "$LIVE" = 1 ]; then
    CONFIG="$APP_DIR/config/settings.yaml"
fi
case "$CONFIG" in "") ;; /*) ;; "~/"*) CONFIG="$HOME/${CONFIG#\~/}" ;; *) CONFIG="$PWD/$CONFIG" ;; esac
if [ "$LIVE" = 0 ] && [ -z "$CONFIG" ]; then
    echo "--prefix $PREFIX には --config（別のデータフォルダの設定）が必要です" >&2; exit 2
fi
PY="${PFB_PYTHON:-$APP_DIR/.venv/bin/python}"
[ -x "$PY" ] || { echo "Python が見つかりません: $PY" >&2; exit 1; }
UNIT_SCRIPT="$APP_DIR/scripts/install_user_services.sh"
UNITS=("$PREFIX-worker.service" "$PREFIX-ui.service")
TIMERS=("$PREFIX-backup.timer" "$PREFIX-guard.timer")
export PFB_PYTHON="$PY"

# settings -> host, port, data_dir (one line each)
read_settings() {
    env -u VIRTUAL_ENV PFB_CONFIG="$CONFIG" "$PY" - <<'EOF'
import os
from pathlib import Path
from pfb_app.config import load_settings
c = os.environ.get("PFB_CONFIG")
s = load_settings(Path(c) if c else None)
print(s.host); print(s.port); print(s.data_dir)
EOF
}
SETTINGS_OUT="$(read_settings)" || { echo "設定を読めません: ${CONFIG:-（既定）}" >&2; exit 1; }
HOST="$(sed -n 1p <<<"$SETTINGS_OUT")"; PORT="$(sed -n 2p <<<"$SETTINGS_OUT")"
DATA_DIR="$(sed -n 3p <<<"$SETTINGS_OUT")"
CONFIG_ARGS=(); [ -n "$CONFIG" ] && CONFIG_ARGS=(--config "$CONFIG")

# ---- probes (all read-only) ----
linger() { loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo unknown; }
tmux_up() { [ "$LIVE" = 1 ] && tmux has-session -t "$1" 2>/dev/null; }
unit_active() { systemctl --user is-active --quiet "$1" 2>/dev/null; }
unit_enabled() { systemctl --user is-enabled --quiet "$1" 2>/dev/null; }
port_answers() { (exec 3<>"/dev/tcp/$HOST/$PORT") 2>/dev/null; }
beat_age() {   # seconds since the worker's beat file was touched; 999999 when there is none
    local m; m=$(stat -c %Y "$DATA_DIR/worker.beat" 2>/dev/null) || { echo 999999; return; }
    echo $(( $(date +%s) - m ))
}
running_jobs() {
    env -u VIRTUAL_ENV "$PY" - "$DATA_DIR/app.db" <<'EOF'
import sqlite3, sys
try:
    c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True, timeout=10)
    print(c.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0])
except sqlite3.Error as e:
    # no database / no jobs table: nothing runs. Anything else (locked, unreadable): cannot
    # tell, so it counts as busy, never as idle
    print(0 if ("no such table" in str(e) or "unable to open" in str(e)) else 1)
EOF
}
method() {
    local t=0 s=0
    if tmux_up pfb-ui || tmux_up pfb-worker; then t=1; fi
    if unit_active "$PREFIX-worker.service" || unit_active "$PREFIX-ui.service" \
        || unit_enabled "$PREFIX-worker.service"; then s=1; fi
    if [ $t = 1 ] && [ $s = 1 ]; then echo "both"; elif [ $t = 1 ]; then echo "tmux"
    elif [ $s = 1 ]; then echo "systemd"; else echo "none"; fi
}
say_method() {
    case "$(method)" in
        tmux) echo "いまの起動方法: tmux（手動。再起動すると止まったままになります）" ;;
        systemd) echo "いまの起動方法: systemd（再起動後も自動で起動します）" ;;
        both) echo "いまの起動方法: tmux と systemd の両方が動いています。二重起動です。どちらかを止めてください" ;;
        none) echo "いまの起動方法: どちらでも動いていません" ;;
    esac
}
preflight_problems() {
    env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.systemd --app-dir "$APP_DIR" --prefix "$PREFIX" \
        --host "$HOST" --port "$PORT" --python "$PY" "${CONFIG_ARGS[@]}" --preflight 2>&1
}

# waits are PFB_OPS_WAIT_SEC (default: the given limit) long, polling every PFB_OPS_POLL_SEC (5)
wait_until() {   # wait_until SECONDS "description" command...
    local limit="${PFB_OPS_WAIT_SEC:-$1}" what="$2" poll="${PFB_OPS_POLL_SEC:-5}"; shift 2
    local t=0
    while ! "$@"; do
        t=$((t + poll)); [ $t -gt "$limit" ] && { echo "  待ち切れませんでした: $what" >&2; return 1; }
        sleep "$poll"
    done
}
preflight_clear() { [ -z "$(preflight_problems)" ]; }
healthy() {
    unit_active "$PREFIX-worker.service" && unit_active "$PREFIX-ui.service" \
        && port_answers && [ "$(beat_age)" -lt 90 ]
}

# ---- check ----
if [ "$CMD" = "check" ]; then
    echo "== 起動方法の確認（読み取りだけ。何も変更しません）=="
    say_method
    echo "linger（ログインしていなくても動かす設定）: $(linger)"
    [ "$(linger)" = "yes" ] || echo "  → 一度だけ: loginctl enable-linger $USER"
    for u in "${UNITS[@]}" "${TIMERS[@]}"; do
        printf '  %-28s 有効: %-9s 動作: %s\n' "$u" "$(systemctl --user is-enabled "$u" 2>&1 | head -1)" \
            "$(systemctl --user is-active "$u" 2>&1 | head -1)"
    done
    if [ "$LIVE" = 1 ]; then
        for s in pfb-ui pfb-worker; do
            tmux has-session -t "$s" 2>/dev/null && echo "  tmux $s: あります" || echo "  tmux $s: ありません"
        done
    fi
    echo "実行中の解析: $(running_jobs) 件　画面($HOST:$PORT): $(port_answers && echo 応答あり || echo 応答なし)　ワーカーの応答: $(beat_age) 秒前"
    echo "-- 切り替え前の確認（問題があれば下に出ます。tmux が動いている間は出るのが正常です）--"
    preflight_problems || true
    exit 0
fi

# ---- plans (no --apply) ----
if [ "$APPLY" = 0 ]; then
    echo "== $CMD の予定（--apply を付けるまで、何も変更しません）=="
    say_method
    if [ "$CMD" = "install" ]; then
        echo "手順: 1) linger 確認 2) 解析が動いていないことを確認 3) tmux を止める（--stop-tmux のとき）"
        echo "      4) ユニットを書いて有効化 5) 4分以内に画面とワーカーが動くか確認、だめなら元に戻す"
        echo "linger: $(linger)　実行中の解析: $(running_jobs) 件"
        echo "--- 作られるユニット ---"
        "$UNIT_SCRIPT" --dry-run --prefix "$PREFIX" --python "$PY" --host "$HOST" --port "$PORT" \
            "${CONFIG_ARGS[@]}" | grep -E '^### |^ExecStart=|^Restart|^KillMode|^OnCalendar|^OnUnitActive' || true
    else
        echo "手順: ユニットを止めて無効にして削除"
        [ "$START_TMUX" = 1 ] && echo "      そのあと tmux で起動し直す（--start-tmux）"
    fi
    echo "実行するには同じコマンドに --apply を付けてください。"
    exit 0
fi

# ---- apply ----
LOG_DIR="$HOME/pfb_deploy_logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/ops-$CMD-$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "記録: $LOG"
STOPPED_TMUX=0

rollback() {
    echo "!! 元に戻します"
    "$UNIT_SCRIPT" --uninstall --prefix "$PREFIX" --python "$PY" "${CONFIG_ARGS[@]}" || true
    if [ "$STOPPED_TMUX" = 1 ]; then
        echo "tmux で起動し直します"
        PFB_CONFIG="$CONFIG" scripts/start.sh || true
    fi
    echo "元に戻しました。原因は記録（上、または $LOG）を見てください。"
}

if [ "$CMD" = "install" ]; then
    if [ "$(method)" = "systemd" ] && [ "$LIVE" = 1 ]; then
        echo "すでに systemd で動いています。何もしません（ユニットを書き直すときは uninstall してから）。"
        exit 0
    fi
    if [ "$(linger)" != "yes" ] && [ "$NO_LINGER_OK" = 0 ]; then
        echo "linger が有効ではありません（再起動後に自動で起動しません）。先に一度だけ実行してください:" >&2
        echo "  loginctl enable-linger $USER" >&2
        echo "（できない場合のみ --allow-no-linger で続けられますが、再起動後は動きません）" >&2
        exit 1
    fi
    JOBS="$(running_jobs)"
    if [ "$JOBS" != "0" ] && [ "$BUSY_OK" = 0 ] && [ "$LIVE" = 1 ]; then
        echo "解析が $JOBS 件動いています。終わるのを待つか、--even-if-busy を付けてください" >&2
        echo "（止めた解析はあとで自動的にやり直しになります）" >&2
        exit 1
    fi
    if tmux_up pfb-ui || tmux_up pfb-worker; then
        if [ "$STOP_TMUX" = 0 ]; then
            echo "tmux の pfb-ui / pfb-worker が動いています。systemd に切り替えるには --stop-tmux を付けてください" >&2
            exit 1
        fi
        echo "== tmux を止めます =="
        scripts/stop.sh; STOPPED_TMUX=1
    fi
    echo "ワーカーと画面が完全に止まっているか確かめます（止まってから約90秒かかります。最大4分待ちます）"
    if ! wait_until 240 "ワーカーと画面の停止（preflight）" preflight_clear; then
        preflight_problems >&2
        echo "別の方法（手動の起動など）で動いているようです。何も変更せずに止めました。" >&2
        [ "$STOPPED_TMUX" = 1 ] && rollback
        exit 1
    fi
    echo "== ユニットを書いて有効にします =="
    if ! "$UNIT_SCRIPT" --enable --prefix "$PREFIX" --python "$PY" --host "$HOST" --port "$PORT" \
            "${CONFIG_ARGS[@]}"; then
        rollback; exit 1
    fi
    echo "== 動作確認（最大4分）=="
    if wait_until 240 "ワーカーと画面の起動" healthy; then
        echo "OK: ワーカーと画面が systemd で動いています。確認: scripts/ops/check_ops.sh"
        exit 0
    fi
    systemctl --user status "$PREFIX-worker.service" "$PREFIX-ui.service" --no-pager 2>&1 | tail -20
    rollback; exit 1
fi

# uninstall
echo "== systemd のユニットを止めて削除します =="
if ! "$UNIT_SCRIPT" --uninstall --prefix "$PREFIX" --python "$PY" "${CONFIG_ARGS[@]}"; then exit 1; fi
if [ "$START_TMUX" = 1 ] && [ "$LIVE" = 1 ]; then
    echo "== tmux で起動し直します =="
    PFB_CONFIG="$CONFIG" scripts/start.sh
else
    echo "止めたままです。tmux で動かすなら: PFB_CONFIG=$CONFIG scripts/start.sh"
fi
