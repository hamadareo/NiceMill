#!/usr/bin/env bash
# Usage: scripts/provision_pose_weights.sh [--check] [SRC]
#
# Puts the line-call pose model (判定支援) in place: <data>/models/yolo11x-pose.pt, copied from a
# LOCAL file (SRC, or $PFB_POSE_WEIGHTS_SRC) whose sha256 must be the pinned POSE_SHA256 below
# (the same as PFB3x3 scripts/linecall_run.py and pfb_app.engine.pfb3x3.POSE_SHA256). Nothing is
# ever downloaded: without the file the app shows 「判定材料を作る準備ができていません」.
# <data> is $PFB_DATA_DIR or <app>/data (as scripts/setup_pfb3x3_engine.sh, which calls this).
#   --check   only say whether the installed file is the pinned one (writes nothing)
# Idempotent: an installed file with the pinned sha256 is left alone. The copy is written to a
# temp name, checked and only then renamed into place. Exit: 0 ok, 1 error / not ready, 2 usage.
set -euo pipefail
POSE_SHA256="013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8"

CHECK=0 SRC="${PFB_POSE_WEIGHTS_SRC:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) SRC="$1" ;;
  esac
  shift
done

APP="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA="${PFB_DATA_DIR:-$APP/data}"
DEST="$DATA/models/yolo11x-pose.pt"

die() { echo "provision_pose_weights: $*" >&2; exit 1; }
sha_of() { sha256sum -- "$1" | cut -d' ' -f1; }

if [ -f "$DEST" ] && [ "$(sha_of "$DEST")" = "$POSE_SHA256" ]; then
  echo "pose weights: $DEST (up to date)"
  exit 0
fi
if [ "$CHECK" = 1 ]; then
  if [ -e "$DEST" ]; then die "$DEST is not the pinned file (sha256 differs)"; fi
  die "$DEST is missing"
fi

[ -n "$SRC" ] || die "no source: give a local yolo11x-pose.pt as SRC or PFB_POSE_WEIGHTS_SRC"
[ -f "$SRC" ] || die "source not found: $SRC"
[ "$(sha_of "$SRC")" = "$POSE_SHA256" ] || die "$SRC: sha256 is not the pinned $POSE_SHA256"
# Never write into the read-only research tree.
case "$(realpath -m "$DATA")/" in "$(realpath -m "$HOME/extern")"/*) die "refusing to write under ~/extern ($DATA)";; esac

mkdir -p "$DATA/models"
TMP="$DEST.tmp.$$"
trap 'rm -f -- "$TMP"' EXIT
cp -- "$SRC" "$TMP"
[ "$(sha_of "$TMP")" = "$POSE_SHA256" ] || die "the copy of $SRC does not match (disk?)"
mv -- "$TMP" "$DEST"
trap - EXIT
echo "pose weights: copied $SRC -> $DEST (sha256 ok)"
