#!/usr/bin/env bash
# Check the whole operations setup, read-only: how the app is started, that the units and timers
# run, that boot-time start is on (linger), the free disk, the newest backup (and whether a
# restore was ever tested), the worker and the screens. Changes nothing.
#   scripts/ops/check_ops.sh [--config YAML] [--prefix NAME]
# Exit code: 0 all good, 1 something to look at, 2 something is wrong. Output in Japanese.
set -uo pipefail
cd "$(dirname "$0")/../.."
APP_DIR="$(pwd)"
CONFIG="${PFB_CONFIG:-}" PREFIX="pfb"
while [ $# -gt 0 ]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
        *) echo "不明なオプション: $1" >&2; exit 2 ;;
    esac
done
[ -z "$CONFIG" ] && [ "$PREFIX" = "pfb" ] && [ -f config/settings.yaml ] && CONFIG="$APP_DIR/config/settings.yaml"
PY="${PFB_PYTHON:-$APP_DIR/.venv/bin/python}"
LEVEL=0
mark() {   # mark LEVEL TEXT
    case "$1" in
        0) printf '[OK]   %s\n' "$2" ;;
        1) printf '[注意] %s\n' "$2"; [ $LEVEL -lt 1 ] && LEVEL=1 ;;
        2) printf '[要対応] %s\n' "$2"; LEVEL=2 ;;
    esac
}
echo "== NiceMillの運用チェック（読み取りだけ）$(date '+%Y-%m-%d %H:%M') =="

# how it is started
t=0; s=0
[ "$PREFIX" = "pfb" ] && { tmux has-session -t pfb-ui 2>/dev/null || tmux has-session -t pfb-worker 2>/dev/null; } && t=1
systemctl --user is-active --quiet "$PREFIX-worker.service" 2>/dev/null && s=1
if [ $t = 1 ] && [ $s = 1 ]; then
    mark 2 "tmux と systemd の両方で動いています（二重起動）。どちらかを止めてください"
elif [ $s = 1 ]; then
    mark 0 "systemd で動いています（再起動後も自動で起動します）"
elif [ $t = 1 ]; then
    mark 1 "tmux（手動）で動いています。再起動すると止まったままになります（自動起動にするには docs/operations_ja.md）"
else
    mark 2 "ワーカーが systemd でも tmux でも動いていません"
fi

# systemd units
if [ $s = 1 ] || systemctl --user is-enabled --quiet "$PREFIX-worker.service" 2>/dev/null; then
    for u in "$PREFIX-worker.service" "$PREFIX-ui.service"; do
        en="$(systemctl --user is-enabled "$u" 2>&1 | head -1)"; ac="$(systemctl --user is-active "$u" 2>&1 | head -1)"
        nr="$(systemctl --user show "$u" -p NRestarts --value 2>/dev/null)"
        if [ "$ac" = "active" ] && [ "$en" = "enabled" ]; then
            mark 0 "$u: 動いています（自動起動 有効、これまでの再起動 ${nr:-0} 回）"
        else
            mark 2 "$u: 動作 $ac / 自動起動 $en"
        fi
    done
    for u in "$PREFIX-backup.timer" "$PREFIX-guard.timer"; do
        en="$(systemctl --user is-enabled "$u" 2>&1 | head -1)"; ac="$(systemctl --user is-active "$u" 2>&1 | head -1)"
        if [ "$ac" = "active" ] && [ "$en" = "enabled" ]; then
            next="$(systemctl --user show "$u" -p NextElapseUSecRealtime --value 2>/dev/null)"
            mark 0 "$u: 動いています${next:+（次の実行 $next）}"
        else
            mark 2 "$u: 動作 $ac / 自動起動 $en"
        fi
    done
fi

# starts after a reboot only with linger
lg="$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo unknown)"
if [ "$lg" = "yes" ]; then
    mark 0 "再起動・停電のあとも自動で起動する設定（linger）が有効です"
elif [ $s = 1 ]; then
    mark 2 "linger が無効です。サーバーを再起動すると止まったままになります: loginctl enable-linger $USER"
else
    mark 1 "linger は無効です（systemd に切り替えるときに必要です）"
fi

# the data: worker, screens, disk, backup (python side; its level is the worst of its lines)
if [ -x "$PY" ]; then
    args=(); [ -n "$CONFIG" ] && args=(--settings "$CONFIG")
    out="$(env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.guard "${args[@]}" report 2>&1)"; rc=$?
    echo "$out" | sed -n '4,$p'
    [ $rc -gt $LEVEL ] && LEVEL=$rc
    bargs=(); [ -n "$CONFIG" ] && bargs=(--settings "$CONFIG")
    n="$(env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.backup "${bargs[@]}" list 2>/dev/null | wc -l)"
    echo "       保存しているバックアップ: ${n} 件（scripts/backup.sh list で一覧）"
else
    mark 2 "Python が見つかりません: $PY"
fi

echo
case $LEVEL in
    0) echo "結果: すべて問題ありません。" ;;
    1) echo "結果: 確かめたほうがよいことがあります（上の [注意]）。" ;;
    *) echo "結果: 対応が必要なことがあります（上の [要対応]）。docs/operations_ja.md を見てください。" ;;
esac
exit $LEVEL
