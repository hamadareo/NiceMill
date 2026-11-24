#!/usr/bin/env bash
# Usage: scripts/setup_pfb3x3_engine.sh [--force] [--source REPO|URL | --archive TAR] [<commit>]
#
# Installs a frozen copy of PFB3x3 at <commit> (default: the pin below) as pfb_app's engine:
#   <app>/engines/pfb3x3/            git archive of <commit> (ENGINE_COMMIT records it)
#   <app>/engines/pfb3x3/.venv       Python 3.12 (uv, relocatable): PFB3x3's deps + torch cu128
#                                    + ultralytics + PyAV (NVDEC); src/ on the path via a .pth
#   <data>/models/yolo11l.pt         YOLO weights (copied from ~/tid_tracker/repo/yolo11l.pt)
#   <data>/models/yolo11x-pose.pt    line-call pose weights, from the local file
#                                    $PFB_POSE_WEIGHTS_SRC (scripts/provision_pose_weights.sh:
#                                    pinned sha256, never downloaded; without it a warning,
#                                    and 判定支援 says 準備ができていません)
# The engine's transcribe command runs in pfb_app's own <app>/.venv (it has faster-whisper).
# <app> is this script's repo (the tree it runs from); <data> is $PFB_DATA_DIR or <app>/data.
#
# Sources (read only; nothing is ever written into them, and never into ~/extern):
#   --source REPO   a local git repository (git -C REPO archive), or a git URL
#                   (git archive --remote=URL); default $PFB3X3_SRC or ~/extern/pfb3x3
#   --archive TAR   a tarball made by `git archive <commit>` (e.g. on the Mac:
#                   git -C <engine checkout> archive --format=tar <commit> > pfb3x3.tar);
#                   its embedded commit id must match <commit>
# <commit> must resolve to a full 40-hex id (a short id works only with a local --source repo).
# Idempotent: rerunning with the pinned commit only checks the install and the weights.
# A different pinned commit is refused unless --force. Everything is built in a staging dir and
# swapped in only when complete, so a failed (re)install leaves the current engine working.
# Exit: 0 ok, 1 error, 2 usage.
set -euo pipefail
# PFB3x3 0bb7c01 (branch fix-players = 58f8341 + the players fixes):
# - `numbers --timeline-s S`: a number read every S s along each person's track (timeline,
#   segments, dominant, dominant_share); the app asks for it on games (number_timeline_s);
# - `track` (engine tracker, basketball): tid3 autocourt beside the tracker, written to
#   <tracks>/<id>/court/geometry.json (reused per video); the app's participation rule and
#   court diagram read it; track output `court` says ok / none and why;
# and everything of 58f8341:
# PFB3x3 58f8341 (branch engine-next = 7bba64a + tid-tracker 545f111 + linecall T11 2f4348c):
# - `track --tracker auto|engine|trackid3x3` (default PFB_TRACKER, else auto): auto takes an
#   existing complete TrackID3x3 run of a TID dataset video (same file) and never starts a fresh
#   one; any other case, or any TID failure, is the engine tracker as before. The JSON and
#   track_stats.json `tracker` say which one ran (the app records it; no TID identity is kept);
# - line-call T11: real outline finding (150 cm window), per-group autocourt h_unc
#   (geometry.json h_unc_group), the ground envelope (frames.jsonl rows hi_raw, frame_would_step,
#   ground_gap_px); the near-line hole stays open, so the app shows a 踏んだ at a line toward the
#   camera as 要確認;
# and everything of 7bba64a (20f0a65 + profiles-engine dbb908b + linecall T7 e533194):
# - `scene --activity` (prompts/check/activity.md): the kind of a non-sport activity, asked in
#   one more request; the app passes --activity only when that prompt file is in the engine;
# - scripts/eval_linecall.py (GT-geometry line-call evaluation; the app does not call it);
# and everything of 20f0a65 (feature/linecall + perf-track, on top of 8815ccf):
# - the sports-names engine of 8815ccf (probe/scene/numbers/teams/quality) and its
#   configs/cue_gate.yaml (every naming cue disabled for qwen3vl-8b and qwen3vl-32b), unchanged;
# - the line-call engine (`linecall` and `live` subcommands);
# - faster `track`: NVDEC decode through PyAV (in-process thread), long side downscaled to 1920
#   with boxes scaled back, YOLO fp16 in batches of 8, `--color fast` colours by default
#   (PFB_TRACK_DECODE/HALF/BATCH select the path; cpu/0/1 = the old pipeline);
# - scripts/serve.sh binds $PFB_SERVE_HOST (default 127.0.0.1), so the shared server is used.
DEFAULT_COMMIT=0bb7c01db8519a4f65c5a11a793a1b271dc45235
export PATH="$HOME/.local/bin:$PATH"

FORCE=0 SRC="${PFB3X3_SRC:-$HOME/extern/pfb3x3}" ARCHIVE="" COMMIT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --force) FORCE=1 ;;
    --source) SRC="${2:?--source needs a value}"; shift ;;
    --archive) ARCHIVE="${2:?--archive needs a value}"; shift ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) [ -z "$COMMIT" ] || { echo "only one <commit>" >&2; exit 2; }; COMMIT="$1" ;;
  esac
  shift
done
COMMIT="${COMMIT:-$DEFAULT_COMMIT}"

APP="$(cd "$(dirname "$0")/.." && pwd -P)"
DATA="${PFB_DATA_DIR:-$APP/data}"
ENGINES="$APP/engines"
ENG="$ENGINES/pfb3x3"
YOLO_SRC="${PFB_YOLO_SRC:-$HOME/tid_tracker/repo/yolo11l.pt}"
TORCH_INDEX=https://download.pytorch.org/whl/cu128
TORCH_PKGS=(torch==2.11.0 torchvision==0.26.0)          # as in ~/tid_tracker/venv (the server)
# av: PyAV with av.codec.hwaccel (NVDEC through its bundled FFmpeg), the version the P2 tracking
# bench and the live engine venv use; fast colours are plain swscale (av + numpy) and the lut
# colours run on torch, so nothing else is needed.
TRACK_PKGS=(ultralytics==8.4.176 lap==0.5.13 av==19.0.2)

die() { echo "setup_pfb3x3_engine: $*" >&2; exit 1; }

# Never install into the read-only research tree.
ABC="$HOME/extern"
for d in "$APP" "$DATA"; do
  case "$(realpath -m "$d")/" in "$(realpath -m "$ABC")"/*) die "refusing to write under $ABC ($d)";; esac
done
command -v uv > /dev/null || die "uv not found (expected ~/.local/bin/uv)"

engine_ok() {  # $1 = engine tree; its venv imports everything the engine needs, from $1/src
  [ -x "$1/.venv/bin/python" ] &&
    (cd "$1" && .venv/bin/python -c '
import os, sys, pfb3x3.engine, ultralytics, torch, lap, scipy, pyarrow, av, yaml
from av.codec.hwaccel import HWAccel
src = os.path.realpath(os.path.join(sys.argv[1], "src"))
assert os.path.realpath(pfb3x3.engine.__file__).startswith(src + os.sep), pfb3x3.engine.__file__
' "$1")
}

whisper_check() {  # the engine's transcribe runs in pfb_app's own venv (warn only)
  "$APP/.venv/bin/python" -c 'import faster_whisper, av, yaml' 2> /dev/null ||
    echo "setup_pfb3x3_engine: warning: $APP/.venv lacks faster-whisper (run uv sync)" >&2
}

copy_weights() {
  [ -f "$YOLO_SRC" ] || die "YOLO weights not found: $YOLO_SRC"
  mkdir -p "$DATA/models"
  if [ -f "$DATA/models/yolo11l.pt" ] && cmp -s "$YOLO_SRC" "$DATA/models/yolo11l.pt"; then
    echo "weights: $DATA/models/yolo11l.pt (up to date)"; return
  fi
  cp "$YOLO_SRC" "$DATA/models/yolo11l.pt.tmp.$$"
  mv "$DATA/models/yolo11l.pt.tmp.$$" "$DATA/models/yolo11l.pt"
  echo "weights: copied $YOLO_SRC -> $DATA/models/yolo11l.pt"
}

pose_weights() {  # linecall's pose model: a local copy with the pinned sha256, never a download
  if [ -z "${PFB_POSE_WEIGHTS_SRC:-}" ] && [ ! -e "$DATA/models/yolo11x-pose.pt" ]; then
    echo "setup_pfb3x3_engine: warning: no yolo11x-pose.pt (set PFB_POSE_WEIGHTS_SRC to a local copy); line calls are not ready" >&2
    return
  fi
  PFB_DATA_DIR="$DATA" "$APP/scripts/provision_pose_weights.sh" || die "pose weights not in place (see above)"
}

PINNED=""
[ -f "$ENG/ENGINE_COMMIT" ] && PINNED="$(cat "$ENG/ENGINE_COMMIT")"

# Resolve <commit> to its full id; pins are compared by the full id only (no prefixes).
if [ -n "$ARCHIVE" ]; then
  [ -f "$ARCHIVE" ] || die "archive not found: $ARCHIVE"
  FULL="$(git get-tar-commit-id < "$ARCHIVE" || true)"
  [ -n "$FULL" ] || die "$ARCHIVE has no embedded commit id (make it with git archive <commit>)"
  [ "$FULL" = "$COMMIT" ] || die "$ARCHIVE is commit $FULL, not $COMMIT (give the full commit id)"
elif [ -d "$SRC" ]; then
  git -C "$SRC" rev-parse --git-dir > /dev/null 2>&1 ||
    die "$SRC is not a git repository; use --archive (see --help)"
  FULL="$(git -C "$SRC" rev-parse --verify "$COMMIT^{commit}")" || die "unknown commit $COMMIT in $SRC"
else
  FULL="$COMMIT"
fi
[[ "$FULL" =~ ^[0-9a-f]{40}$ ]] || die "need a full 40-hex commit id, got $FULL"

if [ -n "$PINNED" ]; then
  if [ "$PINNED" = "$FULL" ]; then
    if [ "$FORCE" = 0 ] && engine_ok "$ENG"; then
      echo "engine: $ENG already at $PINNED"
      copy_weights
      pose_weights
      whisper_check
      exit 0
    fi
  elif [ "$FORCE" = 0 ]; then
    die "$ENG is pinned to $PINNED, not $FULL; rerun with --force to repin"
  fi
fi

# Build the new engine completely in a staging dir; the current engine (if any) stays in place
# and working until the very end, so a failed (re)install changes nothing.
mkdir -p "$ENGINES"
STAGE="$ENGINES/.pfb3x3.new.$$"
OLD="$ENGINES/.pfb3x3.old.$$"
trap 'rm -rf -- "$STAGE"' EXIT
mkdir "$STAGE"
if [ -n "$ARCHIVE" ]; then
  tar -x -C "$STAGE" -f "$ARCHIVE"
elif [ -d "$SRC" ]; then
  git -C "$SRC" archive --format=tar "$FULL" | tar -x -C "$STAGE"
else
  git archive --format=tar --remote="$SRC" "$FULL" | tar -x -C "$STAGE"
fi
[ -f "$STAGE/pyproject.toml" ] && [ -d "$STAGE/src/pfb3x3" ] || die "the archive is not a PFB3x3 tree"
[ -f "$STAGE/src/pfb3x3/engine.py" ] || die "commit $FULL has no src/pfb3x3/engine.py (the embedding entrypoint)"

# Relocatable venvs, so the staged tree can be renamed into place: the package is put on the
# path by a .pth line relative to site-packages (not an absolute editable install), so
# pfb3x3's repo-relative paths (configs/, scripts/) resolve inside the engine copy.
echo "venv: .venv (python 3.12)"
uv venv -q --relocatable --python 3.12 "$STAGE/.venv"
uv pip install -q --python "$STAGE/.venv/bin/python" --index-url "$TORCH_INDEX" "${TORCH_PKGS[@]}"
uv pip install -q --python "$STAGE/.venv/bin/python" -r "$STAGE/pyproject.toml" "${TRACK_PKGS[@]}"
SITE="$("$STAGE/.venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
"$STAGE/.venv/bin/python" -c 'import os, sys; print(os.path.relpath(sys.argv[1], sys.argv[2]))' \
  "$STAGE/src" "$SITE" > "$SITE/pfb3x3_engine_src.pth"

engine_ok "$STAGE" || die "the staged engine does not import (see the errors above)"
copy_weights
pose_weights
echo "$FULL" > "$STAGE/ENGINE_COMMIT"   # after the weights: a pinned engine is a complete one

# Swap: two renames; the old engine is restored if the second one fails.
case "$ENG" in */engines/pfb3x3) ;; *) die "unexpected engine path $ENG" ;; esac
if [ -e "$ENG" ]; then mv -- "$ENG" "$OLD"; fi
if ! mv -- "$STAGE" "$ENG"; then
  [ -e "$OLD" ] && mv -- "$OLD" "$ENG"
  die "could not move the new engine into place; the previous one was kept"
fi
trap - EXIT
rm -rf -- "$OLD"
engine_ok "$ENG" || die "the engine does not import after the move to $ENG"
whisper_check
echo "engine: $ENG pinned to $FULL"
