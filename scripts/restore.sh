#!/usr/bin/env bash
# Put a backup back as data/app.db (pfb_app.ops.backup restore).
#   scripts/restore.sh NAME [--force]      NAME: a folder name from `scripts/backup.sh list`
# Stop the app first (systemctl --user stop pfb-ui pfb-worker, or scripts/stop.sh); the
# worker's beat file must be quiet unless --force. The current database is kept as
# data/app.db.before-restore-<time>. Settings: PFB_CONFIG (default: config/settings.yaml).
set -euo pipefail
cd "$(dirname "$0")/.."
# no PFB_CONFIG: this checkout's own config/settings.yaml (the live settings), else the built-ins
if [ -z "${PFB_CONFIG:-}" ] && [ -f config/settings.yaml ]; then
    export PFB_CONFIG="$(pwd)/config/settings.yaml"
fi
[ $# -ge 1 ] || { sed -n '2,6p' "$0"; exit 2; }
PY="${PFB_PYTHON:-$(pwd)/.venv/bin/python}"
NAME="$1"; shift
if [ -t 0 ]; then
    read -r -p "バックアップ $NAME でデータベースを置き換えます。よろしいですか？ [y/N] " ok
    case "$ok" in y|Y|yes) ;; *) echo "やめました"; exit 1 ;; esac
fi
exec env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.backup restore "$NAME" "$@"
