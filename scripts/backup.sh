#!/usr/bin/env bash
# Backups of data/app.db (pfb_app.ops.backup): online SQLite backup + data-dir manifest,
# verified and rotated (backup_keep). Settings: PFB_CONFIG (default: config/settings.yaml), or --settings before the command.
#   scripts/backup.sh run            take one now (the pfb-backup timer runs this nightly)
#   scripts/backup.sh list           the complete backups
#   scripts/backup.sh verify [NAME]  check one (default: the newest)
#   scripts/backup.sh rehearse [NAME]  restore one into a scratch folder and compare (live data untouched)
# Restoring: scripts/restore.sh NAME
set -euo pipefail
cd "$(dirname "$0")/.."
# no PFB_CONFIG: this checkout's own config/settings.yaml (the live settings), else the built-ins
if [ -z "${PFB_CONFIG:-}" ] && [ -f config/settings.yaml ]; then
    export PFB_CONFIG="$(pwd)/config/settings.yaml"
fi
PY="${PFB_PYTHON:-$(pwd)/.venv/bin/python}"
exec env -u VIRTUAL_ENV "$PY" -m pfb_app.ops.backup "$@"
