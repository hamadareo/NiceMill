"""Client for the frozen PFB3x3 engine (`<engine_dir>`, its own Python 3.12 venv).

pfb_app talks to the engine only through subprocesses: `python -m pfb3x3.engine <cmd> ...`
prints one JSON object on its last stdout line and exits 0 (ok), 3 (needs GPU / not enough
memory) or 1 (error). Every path the engine reads or writes is under the submission's
`<sub_dir>/pfb3x3/` (plus `<data_dir>/pfb3x3_runtime/` for the vLLM server and
`<data_dir>/models/` for the YOLO weights); nothing points into the PFB3x3 source tree.

Termination is only ever applied to the child we started: each engine call runs in its own
process group (start_new_session), and the group is signalled only while the Popen handle
says the child has not been reaped, so its pid cannot have been reused.

Every engine call records itself in `call_record` (pfb3x3_runtime/engine_call.pgid), or, while
another call holds that record (a pfb3x3 session runs a work's behavior and names side by
side), in `engine_call.<pid>.pgid` beside it; stop_stale_call stops every recorded group. A
submission's stderr/stdout logs are named after the command (`<root>/logs/<cmd>.err`), so one
submission never runs the same command twice at once (one job per stage and camera).
The one exception is `linecall`, which the worker runs beside the other sessions
(worker.linecall_runner): it records itself in `linecall_call_record` instead, so neither
side ever stops the other's command as a leftover.

Error texts (EngineError and its str()) and CueGate.reason are for logs and the admin only:
callers never show them on the coach's screens.
"""
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

from pfb_app.config import Settings
from pfb_app.inference.structured import YieldRequested
from pfb_app.ops import offline

log = logging.getLogger(__name__)

ENGINE_MODELS = {"standard": "qwen3vl-8b", "high": "qwen3vl-32b"}
CONDITIONS = ("clip_som", "static_som", "static_ref")
# (frame_step_s, frames_per_window) of each condition, mirrored from PFB3x3 windows.CONDITIONS:
# how many windows a behavior run will have (planned_windows), for its progress
WINDOW_PLAN = {"clip_som": (0.5, 12), "static_som": (5.0, 5), "static_ref": (5.0, 5)}
EXIT_NEEDS_GPU = 3
SERVED_CONCURRENCY = 8   # requests of one batch sent to the session's server at once
SERVED_TIMEOUT_SEC = 300.0   # one contribution / pfb answer (a timeout is asked once more)
SERVE_HOST = "127.0.0.1"   # the only address the vLLM server may listen on
TICK_SEC = 10.0
TERM_GRACE_SEC = 30.0
STDERR_TAIL_LINES = 40
OPEN_END_S = 86400.0   # a pasted transcript's open-ended last segment (video seconds)
LOG_PREFIX = "ログ: "   # the line of EngineError naming the engine's stderr log
SERVER_UNAVAILABLE = "server unavailable"   # the engine's own error for a dead server
# exit-1 errors that are never a cue's refusal but a broken call (bad argv, no entry, bad id)
_HARD_ERRORS = ("python -m pfb3x3.engine", "no registry entry", "invalid video id")
PROBE_SUFFIX = "__probe"   # the content check's own registry id: <video id>__probe
VLLM_DEFAULT_VENV = "~/extern/venvs/qwen3vl"  # serve.sh's default (read only)
ENGINE_DOMAINS = ("generic", "sports")   # registry `domain` values pfb_app writes
TARGETS = ("person", "vehicle")          # registry `target`: what the detector looks for
# The naming cues the engine evaluates against the 0.90 precision gate (configs/cue_gate.yaml).
CUES = ("number", "team", "position", "grid", "batting", "appearance", "place")
CUE_GATE_FILE = Path("configs") / "cue_gate.yaml"
ACTIVITY_PROMPT = Path("prompts") / "check" / "activity.md"   # the engine has scene --activity
NO_GATE_RESULT = "評価結果がありません"
# yolo11x-pose.pt (118,481,010 bytes), as PFB3x3 scripts/linecall_run.py POSE_SHA256 pins it.
# The engine would download a missing file: the app never lets it (pose_weights_ready);
# scripts/provision_pose_weights.sh copies it from a local file with this sha256.
POSE_SHA256 = "013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8"
_pose_seen: dict[tuple, str] = {}   # (path, size, mtime_ns, inode) -> sha256, hashed once
_VIDEO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# Inherited variables that would point the engine elsewhere (another venv, the source tree).
_DROP_ENV = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX")
# offline and no usage reports (ops.offline), unbuffered, and no __pycache__ inside the
# frozen engine_dir
_OFFLINE_ENV = {**offline.OFFLINE_ENV, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"}
PFB_APP_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TID_PYTHON = Path.home() / "tid_tracker" / "venv" / "bin" / "python"


def default_whisper_python() -> Path:
    """pfb_app's own venv python (it has faster-whisper); never the ~/extern experiment venvs."""
    own = PFB_APP_ROOT / ".venv" / "bin" / "python"
    return own if own.exists() else Path(sys.executable)

Tick = Callable[[], bool]


class EngineError(RuntimeError):
    """The engine failed (non-zero exit other than 3, no JSON summary, timeout, bad output).

    str(): the message, then 「ログ: <stderr file>」 when known, then the stderr tail. The
    first line alone is the short form (status page); the log line points at the whole log."""

    def __init__(self, message: str, stderr_tail: str = "", returncode: int | None = None,
                 log_path: Path | None = None):
        log_line = f"\n{LOG_PREFIX}{log_path}" if log_path is not None else ""
        super().__init__(f"{message}{log_line}\n{stderr_tail}".rstrip())
        self.message = message
        self.stderr_tail = stderr_tail
        self.returncode = returncode
        self.log_path = log_path


class EngineTimeout(EngineError):
    """The engine command ran longer than its timeout and was terminated."""


class EngineServerDown(YieldRequested):
    """The session's vLLM server died or stopped answering: the job goes back to the queue and
    the session ends (the next session starts a new server), instead of every queued job
    failing against a dead server."""


class EngineNeedsGpu(RuntimeError):
    """Exit 3: not enough free GPU memory right now; retry later."""


class EngineYield(RuntimeError):
    """on_tick asked to give the GPU back; the engine subprocess was terminated."""


TRACKERS = ("engine", "trackid3x3")   # what engine 58f8341's `track` reports as `tracker`
FALLBACK_REASON_MAX = 300              # characters of a TID fallback reason kept in the run


@dataclass(frozen=True)
class TrackResult:
    """tracker: the tracker the engine reports (engine 58f8341: "engine" or "trackid3x3"; None
    from an older engine or an unknown value); tracker_fallback: why TrackID3x3 was not used
    (its `fallback.reason`, shortened), None when it was used or nothing was said."""
    people: int
    labels: list[str]
    track_stats: Path
    boxes: Path
    tracker: str | None = None
    tracker_fallback: str | None = None


@dataclass(frozen=True)
class LinecallResult:
    """The engine's linecall: episodes found, {stepped, clear, undecidable: n}, and its
    episodes.json / provenance.json (in <tracks>/<id>/linecall/; crop, overlay and clip paths
    in episodes.json are relative to that dir). Outputs are 判定材料, never a ruling."""
    episodes: int
    verdicts: dict[str, int]
    episodes_json: Path
    provenance: Path


@dataclass(frozen=True)
class SheetResult:
    sheet: Path
    frame: int
    geometry: dict
    frame_rule: str | None = None        # "auto_start", "manual" or the window rule
    frame_offset_s: float | None = None  # seconds after the task start


@dataclass(frozen=True)
class TranscribeResult:
    segments: int
    transcript: Path


@dataclass(frozen=True)
class BehaviorResult:
    rows: Path
    windows: int
    errors: int
    overlay_dir: Path
    transcript: bool = False   # True only when transcript segments were given to the model

    def load_rows(self) -> list[dict]:
        with open(self.rows, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]


@dataclass(frozen=True)
class ProbeResult:
    """The content check's sampled counts: persons/vehicles are {"median", "p90", "max"},
    camera_motion {"median_shift_frac", "moving"} (None: unknown), frames the sample images."""
    duration_s: float
    persons: dict
    vehicles: dict
    camera_motion: dict
    frames: list[Path]
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    n_frames: int | None = None
    samples: int | None = None


@dataclass(frozen=True)
class SceneResult:
    """The advisory scene verdict: scene group_work | sport | other; sport a sport key (None
    unless scene is sport); sport_text the model's own words. activity: the kind of a
    non-sport activity (group_task | meeting | presentation | workshop | learning | other),
    None when the engine has no activity verdict (an older engine, a sport, or that request
    failed); activity_text the model's own words."""
    scene: str
    sport: str | None
    sport_text: str | None
    numbers_visible: bool
    model: str | None = None
    activity: str | None = None
    activity_text: str | None = None


@dataclass(frozen=True)
class TeamsResult:
    """{P-label: "T1" | "T2" | "other"}; reason "" or too_few / one_colour / not_person (and
    "error" when the engine refused). T1/T2 may swap between runs: store the chosen labels.
    error: the engine's refusal (None: it ran, even when it found no teams)."""
    teams: dict[str, str]
    reason: str
    crops: dict[str, Path]
    strips: dict[str, Path]
    sizes: dict[str, int] = field(default_factory=dict)   # {"T1", "T2", "other": n}
    other_dist: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class NumberRead:
    """One person's number reads, as the engine's `numbers` gives them. number is a digit
    string ("00" and "0" are different numbers), None when the reads did not agree; votes the
    reads of it; frames_agree the distinct frames that read it (fusion needs this); reads the
    answers received; all / frames each answer and its frame; failed the requests that failed;
    skipped {reason: n} picked frames without a request. With a number timeline (numbers
    --timeline-s): timeline (frame, read or None) along the track, segments (first frame,
    last frame, number, reads) of the same number read in a row, dominant the person's number
    over the segments (None: a tie or nothing read) and dominant_share its share of the
    numbered reads; timeline_s 0 without a timeline."""
    number: str | None
    votes: int
    reads: int
    frames_agree: int = 0
    frames: tuple[int, ...] = ()
    all: tuple[str | None, ...] = ()
    failed: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    timeline_s: float = 0.0
    timeline: tuple[tuple[int, str | None], ...] = ()
    segments: tuple[tuple[int, int, str, int], ...] = ()
    dominant: str | None = None
    dominant_share: float | None = None


def _timeline(r: dict) -> dict:
    """The number-timeline fields of one engine numbers entry (NumberRead); malformed parts
    are left out."""
    out: dict = {}
    try:
        out["timeline_s"] = float(r.get("timeline_s") or 0.0)
    except (TypeError, ValueError):
        return out
    tl, segs = [], []
    for item in r.get("timeline") or ():
        try:
            tl.append((int(item[0]), None if item[1] is None else str(item[1])))
        except (TypeError, ValueError, IndexError):
            continue
    for item in r.get("segments") or ():
        try:
            segs.append((int(item[0]), int(item[1]), str(item[2]), int(item[3])))
        except (TypeError, ValueError, IndexError):
            continue
    dom, share = r.get("dominant"), r.get("dominant_share")
    out.update(timeline=tuple(tl), segments=tuple(segs),
               dominant=None if dom is None else str(dom),
               dominant_share=float(share) if isinstance(share, (int, float))
               and not isinstance(share, bool) else None)
    return out


class NumberReads(dict):
    """{P-label: NumberRead} (a dict); error: the engine's refusal (None: it ran)."""

    def __init__(self, reads=(), *, error: str | None = None):
        super().__init__(reads)
        self.error = error


class PositionsResult(dict):
    """{P-label: roster position} (a dict), plus the engine's slots {position: roster rows
    with it, counted from the positions list given} and claims {position: how many labels
    answered it} for every roster position (0 when unclaimed); a position claimed by more
    labels than it has rows is dropped from the dict. error: the engine's refusal."""

    def __init__(self, positions=(), *, slots: dict | None = None, claims: dict | None = None,
                 error: str | None = None):
        super().__init__(positions)
        self.slots = {str(k): int(v) for k, v in (slots or {}).items()}
        self.claims = {str(k): int(v) for k, v in (claims or {}).items()}
        self.error = error


class GridOrder(list):
    """P-labels in start-grid order, front first (a list), plus the frame used and the
    direction of travel (direction_source "tracks" or "override")."""

    def __init__(self, order=(), *, frame: int | None = None, frame_rule: str | None = None,
                 direction: str | None = None, direction_source: str | None = None,
                 error: str | None = None):
        super().__init__(order)
        self.error = error   # the engine's refusal (None: it ran)
        self.frame = frame
        self.frame_rule = frame_rule
        self.direction = direction
        self.direction_source = direction_source


class BattingOrder(list):
    """P-labels by their first turn at bat (a list), plus frames (marked frames asked), errors
    (failed requests), first_frame {label: frame}, seen {label: turns seen} and sequence
    [[frame, label], ..]."""

    def __init__(self, order=(), *, frames: int = 0, errors: int = 0,
                 first_frame: dict | None = None, seen: dict | None = None,
                 sequence: list | None = None, error: str | None = None):
        super().__init__(order)
        self.error = error   # the engine's refusal (None: it ran)
        self.frames = frames
        self.errors = errors
        self.first_frame = dict(first_frame or {})
        self.seen = dict(seen or {})
        self.sequence = list(sequence or [])


@dataclass(frozen=True)
class CueGate:
    """A cue's gate: enabled only if the engine's evaluation passed. reason is for logs and
    the admin only (engine wording); never shown to the coach."""
    enabled: bool
    threshold: float | None
    reason: str


@dataclass(frozen=True)
class ServerHandle:
    pid: int
    port: int
    model: str
    base_url: str


def engine_model_for(model_key: str) -> str:
    """pfb_app model key (standard/high) -> PFB3x3 model name; KeyError for anything else."""
    return ENGINE_MODELS[model_key]


def load_cue_gate(settings: Settings, model_key: str) -> dict[str, CueGate]:
    """{cue: CueGate} for every cue in CUES, from `<engine_dir>/configs/cue_gate.yaml` (written
    by the engine's cue evaluation: {model: {cue: {"enabled", "threshold", ..., "reason"}}})
    under engine_model_for(model_key). A missing or unreadable file, a model or a cue it does
    not cover: that cue is disabled with NO_GATE_RESULT. A cue is enabled only by a literal
    `enabled: true`. The scene check and the school initial-position rule are not gated."""
    off = CueGate(enabled=False, threshold=None, reason=NO_GATE_RESULT)
    section = None
    f = settings.engine_dir / CUE_GATE_FILE
    try:
        data = yaml.safe_load(f.read_text(encoding="utf-8"))
        section = data.get(engine_model_for(model_key)) if isinstance(data, dict) else None
    except FileNotFoundError:
        pass
    except KeyError:
        log.warning("モデル %s の評価結果はありません（%s）", model_key, f)
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
        log.warning("評価結果のファイルを読めません: %s: %s", f, e)
    if not isinstance(section, dict):
        return {cue: off for cue in CUES}
    gate = {}
    for cue in CUES:
        rec = section.get(cue)
        if not isinstance(rec, dict):
            gate[cue] = off
            continue
        try:
            threshold = None if rec.get("threshold") is None else float(rec["threshold"])
        except (TypeError, ValueError):
            gate[cue] = off
            continue
        gate[cue] = CueGate(enabled=rec.get("enabled") is True, threshold=threshold,
                            reason=str(rec.get("reason") or ""))
    return gate


@dataclass(frozen=True)
class _Outcome:
    returncode: int
    summary: dict | None
    stderr_tail: str
    log_path: Path | None = None


class Pfb3x3Engine:
    def __init__(self, settings: Settings, *, tick_sec: float = TICK_SEC,
                 term_grace_sec: float = TERM_GRACE_SEC):
        self._s = settings
        self._tick_sec = tick_sec
        self._grace = term_grace_sec

    # --- paths ---

    @property
    def runtime_dir(self) -> Path:
        return self._s.data_dir / "pfb3x3_runtime"

    @property
    def weights(self) -> Path:
        return self._s.data_dir / "models" / "yolo11l.pt"

    @property
    def pose_weights(self) -> Path:
        """The pose model of linecall (YOLO11x-pose)."""
        return self._s.data_dir / "models" / "yolo11x-pose.pt"

    def pose_ready(self) -> bool:
        """The pose weights are in place with the pinned sha256 (linecall runs only then)."""
        return pose_weights_ready(self.pose_weights)

    @property
    def tid_python(self) -> Path:
        """The TrackID3x3 venv python (court registration of linecall and live)."""
        return self._s.tid_python or DEFAULT_TID_PYTHON

    @property
    def python(self) -> Path:
        return self._s.engine_python or self._s.engine_dir / ".venv" / "bin" / "python"

    @property
    def tracker_python(self) -> Path:
        return self._s.tracker_python or self.python

    @property
    def whisper_python(self) -> Path:
        return self._s.whisper_python or default_whisper_python()

    @property
    def engine_commit(self) -> str | None:
        """The pinned PFB3x3 commit (engine_dir/ENGINE_COMMIT), None when not set up."""
        try:
            return (self._s.engine_dir / "ENGINE_COMMIT").read_text(encoding="utf-8").strip() or None
        except FileNotFoundError:
            return None

    @staticmethod
    def root(sub_dir: Path) -> Path:
        """The engine data root of one submission."""
        return Path(sub_dir) / "pfb3x3"

    def _env(self, root: Path | None, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("PFB_") and k not in _DROP_ENV}
        env.update(_OFFLINE_ENV)
        env["PFB_RUNTIME_DIR"] = str(self.runtime_dir)
        env["PFB_GPU_RESERVE_MIB"] = str(engine_reserve_mib(self._s))
        if root is not None:
            env.update({
                "PFB_VIDEOS": str(root / "videos"),
                "PFB_TRACKS": str(root / "tracks"),
                "PFB_DET_CACHE": str(root / "cache"),
                "PFB_YOLO_WEIGHTS": str(self.weights),
            })
        env.update(extra or {})
        return env

    # --- engine commands ---

    def track(self, sub_dir: Path, video_id: str, video_path: Path, start_s: float,
              end_s: float | None, max_people: int | None, activity: str | None, *,
              transcript: bool = False, device: str = "auto",
              domain: str = "generic", sport: str | None = None, target: str = "person",
              on_tick: Tick | None = None,
              on_entry: Callable[[], None] | None = None) -> TrackResult:
        """Write the registry entry, then run person tracking and tracklet linking.
        domain: the engine's generic | sports; sport: a sport key; target: person | vehicle
        (what the detector tracks). on_entry: called once the entry is written, before the
        tracking starts (e.g. to cache behavior's frames beside it: prefetch_frames)."""
        root = self.root(sub_dir)
        self._write_entry(root, video_id, video_path, start_s, end_s, max_people, activity,
                          transcript, domain=domain, sport=sport, target=target)
        if on_entry is not None:
            on_entry()
        out = self._call(root, "track", video_id,
                         ["--device", device, "--tracker-python", str(self.tracker_python)],
                         on_tick=on_tick,
                         env=({"PFB_TRACK_BATCH": str(int(self._s.track_batch))}
                              if self._s.track_batch else None))
        tracker = out.get("tracker") if out.get("tracker") in TRACKERS else None
        fallback = out.get("fallback")
        reason = fallback.get("reason") if isinstance(fallback, dict) else None
        return TrackResult(people=int(out["people"]), labels=[str(x) for x in out["labels"]],
                           track_stats=self._inside(out["track_stats"], root),
                           boxes=self._inside(out["boxes"], root), tracker=tracker,
                           tracker_fallback=(str(reason)[:FALLBACK_REASON_MAX]
                                             if tracker is not None and reason else None))

    def sheet(self, sub_dir: Path, video_id: str, *, auto_start: bool = True,
              frame: int | None = None, on_tick: Tick | None = None) -> SheetResult:
        """The people sheet image (numbered boxes plus a crop per person). auto_start: on the
        first frame >= the task start where everyone is boxed (within 60 s), the frame
        rule_names(auto_start=True) uses too; EngineError when there is no such frame."""
        root = self.root(sub_dir)
        args = ["--out", str(root / "sheets" / video_id)]
        if frame is not None:   # a frame of the coach's / the app's choice (the engine's --frame)
            args += ["--frame", str(int(frame))]
        elif auto_start:
            args.append("--auto-start")
        out = self._call(root, "sheet", video_id, args, on_tick=on_tick)
        offset = out.get("frame_offset_s")
        return SheetResult(sheet=self._inside(out["sheet"], root), frame=int(out["frame"]),
                           geometry=dict(out.get("geometry") or {}),
                           frame_rule=out.get("frame_rule"),
                           frame_offset_s=None if offset is None else float(offset))

    def quality(self, sub_dir: Path, video_id: str, *, on_tick: Tick | None = None,
                timeout_sec: float | None = None) -> dict[str, dict] | None:
        """Per-label track quality {P-label: {coverage, mean_h_frac, fragments, overlap_rate,
        spread_frac, frames}} (mean_h_frac may be None); None when the engine refuses.
        timeout_sec: this call's limit (None: engine_timeout_sec); over it, EngineTimeout."""
        out, _error = self._soft(self.root(sub_dir), "quality", video_id, [], on_tick,
                                 timeout=timeout_sec)
        if out is None or not isinstance(out.get("quality"), dict):
            return None
        return dict(out["quality"])

    def rule_names(self, sub_dir: Path, video_id: str, positions: dict[str, str], *,
                   auto_start: bool = True, on_tick: Tick | None = None) -> dict[str, str] | None:
        """{P-label: name} from the typed initial positions; None when the rule refuses
        (exit 1 with {"error": ...}, e.g. the margin guard). auto_start: the rule looks at the
        first frame >= the task start where everyone is boxed (within 60 s), not the default
        sheet frame, which can hit the margin guard."""
        args = ["--positions", json.dumps(positions, ensure_ascii=False)]
        if auto_start:
            args.append("--auto-start")
        return self._names(sub_dir, "rule-names", video_id, args, on_tick)

    def propose(self, sub_dir: Path, video_id: str, appearance: dict[str, str], *,
                base_url: str, model: str, on_tick: Tick | None = None) -> dict[str, str] | None:
        """{P-label: name} proposed by the VLM from the typed appearance; None when it fails."""
        return self._names(sub_dir, "propose", video_id,
                           ["--appearance", json.dumps(appearance, ensure_ascii=False),
                            "--base-url", base_url, "--model", model], on_tick)

    # --- content check (detect) ---

    def probe(self, sub_dir: Path, video_id: str, video_path: Path, *, start_s: float = 0.0,
              end_s: float | None = None, samples: int = 8, device: str = "cpu",
              on_tick: Tick | None = None, timeout_sec: float | None = None) -> ProbeResult:
        """Sample `samples` frames of [start_s, end_s] and count people and vehicles in them
        (the detector on the CPU by default: the check never waits for the GPU), plus the
        camera motion. The check has its own registry entry, `<video_id>__probe` (written
        afresh each time), so track()'s entry for the video is never touched. timeout_sec:
        this call's limit (None: engine_timeout_sec); over it, EngineTimeout."""
        root = self.root(sub_dir)
        check_id = probe_id(video_id)
        self._write_entry(root, check_id, video_path, start_s, end_s, None, None, False)
        out = self._call(root, "probe", check_id,
                         ["--samples", str(int(samples)), "--device", device,
                          "--detector-python", str(self.tracker_python)], on_tick=on_tick,
                         timeout=timeout_sec)
        return ProbeResult(duration_s=float(out["duration_s"]),
                           persons=dict(out.get("persons") or {}),
                           vehicles=dict(out.get("vehicles") or {}),
                           camera_motion=dict(out.get("camera_motion") or {}),
                           frames=[self._inside(f, root) for f in out.get("frames") or []],
                           fps=_opt_num(out.get("fps"), float),
                           width=_opt_num(out.get("width"), int),
                           height=_opt_num(out.get("height"), int),
                           n_frames=_opt_num(out.get("n_frames"), int),
                           samples=_opt_num(out.get("samples"), int))

    def scene(self, sub_dir: Path, video_id: str, *, base_url: str, model: str,
              on_tick: Tick | None = None, timeout_sec: float | None = None) -> SceneResult:
        """The advisory scene / sport / numbers-visible verdict from the frames of probe() for
        the same video_id (probe first), plus the kind of a non-sport activity when the engine
        has it (has_activity_verdict). Not a cue: an engine refusal is an EngineError.
        timeout_sec as in probe()."""
        root = self.root(sub_dir)
        args = ["--base-url", base_url, "--model", model]
        if self.has_activity_verdict:
            args.append("--activity")
        out = self._call(root, "scene", probe_id(video_id), args,
                         on_tick=on_tick, timeout=timeout_sec)
        sport, text = out.get("sport"), out.get("sport_text")
        activity, activity_text = out.get("activity"), out.get("activity_text")
        return SceneResult(scene=str(out["scene"]), sport=None if sport is None else str(sport),
                           sport_text=None if text is None else str(text),
                           numbers_visible=out.get("numbers_visible") is True,
                           model=None if out.get("model") is None else str(out["model"]),
                           activity=None if activity is None else str(activity),
                           activity_text=None if activity_text is None else str(activity_text))

    @property
    def has_activity_verdict(self) -> bool:
        """True when the pinned engine has `scene --activity` (its prompt file is there); an
        older engine would refuse the option."""
        return (self._s.engine_dir / ACTIVITY_PROMPT).is_file()

    # --- naming cues (after track; an engine refusal gives an empty result) ---

    def teams(self, sub_dir: Path, video_id: str, *,
              on_tick: Tick | None = None) -> TeamsResult:
        """Torso-colour teams of the tracked people, a crop per person and a strip per team."""
        root = self.root(sub_dir)
        out, error = self._soft(root, "teams", video_id, [], on_tick)
        if out is None:
            return TeamsResult(teams={}, reason="error", crops={}, strips={}, error=error)
        return TeamsResult(
            teams={str(k): str(v) for k, v in (out.get("teams") or {}).items()},
            reason=str(out.get("reason") or ""),
            crops={str(k): self._inside(v, root) for k, v in (out.get("crops") or {}).items()},
            strips={str(k): self._inside(v, root) for k, v in (out.get("strips") or {}).items()},
            sizes={str(k): int(v) for k, v in (out.get("sizes") or {}).items()},
            other_dist=_opt_num(out.get("other_dist"), float))

    def numbers(self, sub_dir: Path, video_id: str, *, base_url: str, model: str,
                on_tick: Tick | None = None, timeline_s: float = 0.0) -> NumberReads:
        """{P-label: NumberRead} of the shirt / car numbers read and voted per person (a
        NumberReads; .error is set when the engine refused). timeline_s > 0: also a number
        read every timeline_s seconds of each person's track (the number timeline)."""
        root = self.root(sub_dir)
        extra = ["--timeline-s", str(float(timeline_s))] if timeline_s and timeline_s > 0 else []
        out, error = self._soft(root, "numbers", video_id,
                                ["--base-url", base_url, "--model", model, *extra], on_tick)
        reads = NumberReads(error=error)
        for label, r in ((out or {}).get("numbers") or {}).items():
            if not isinstance(r, dict):
                continue
            n = r.get("number")
            reads[str(label)] = NumberRead(
                number=None if n is None else str(n), votes=int(r.get("votes") or 0),
                reads=int(r.get("reads") or 0), frames_agree=int(r.get("frames_agree") or 0),
                frames=tuple(int(f) for f in r.get("frames") or ()),
                all=tuple(None if a is None else str(a) for a in r.get("all") or ()),
                failed=int(r.get("failed") or 0),
                skipped={str(k): int(v) for k, v in (r.get("skipped") or {}).items()},
                **_timeline(r))
        return reads

    def positions(self, sub_dir: Path, video_id: str, *, sport: str, positions: list[str],
                  base_url: str, model: str,
                  on_tick: Tick | None = None) -> PositionsResult:
        """{P-label: roster position} from the people sheet (a PositionsResult: also .slots and
        .claims). The engine takes the sport from the registry entry, so a different `sport`
        is written there first (with a warning); no entry (track first) is an EngineError."""
        root = self.root(sub_dir)
        self._set_entry_sport(root, video_id, sport)
        out, error = self._soft(root, "positions", video_id,
                                ["--base-url", base_url, "--model", model,
                                 "--positions", json.dumps([str(p) for p in positions],
                                                           ensure_ascii=False)], on_tick)
        out = out or {}
        return PositionsResult({str(k): str(v) for k, v in (out.get("positions") or {}).items()},
                               slots=out.get("slots"), claims=out.get("claims"), error=error)

    def grid(self, sub_dir: Path, video_id: str, *, base_url: str, model: str,
             on_tick: Tick | None = None) -> GridOrder:
        """P-labels in start-grid order (front first) along the direction of travel."""
        out, error = self._order(sub_dir, "grid", video_id, base_url, model, on_tick)
        return GridOrder([str(x) for x in out.get("order") or []], error=error,
                         frame=_opt_num(out.get("frame"), int), frame_rule=out.get("frame_rule"),
                         direction=out.get("direction"),
                         direction_source=out.get("direction_source"))

    def batting(self, sub_dir: Path, video_id: str, *, base_url: str, model: str,
                on_tick: Tick | None = None) -> BattingOrder:
        """P-labels in the order of their first turn at bat."""
        out, error = self._order(sub_dir, "batting", video_id, base_url, model, on_tick)
        return BattingOrder([str(x) for x in out.get("order") or []], error=error,
                            frames=int(out.get("frames") or 0), errors=int(out.get("errors") or 0),
                            first_frame={str(k): int(v)
                                         for k, v in (out.get("first_frame") or {}).items()},
                            seen={str(k): int(v) for k, v in (out.get("seen") or {}).items()},
                            sequence=[[int(f), str(lab)] for f, lab in out.get("sequence") or []])

    def linecall(self, sub_dir: Path, video_id: str, *,
                 on_tick: Tick | None = None) -> LinecallResult:
        """Line-call evidence (判定材料) of a tracked camera: near-line episodes with a verdict,
        crops and clips in <tracks>/<id>/linecall/ (the app writes only labels.json there).
        On the GPU only: exit 3 -> EngineNeedsGpu; a second run of the same video while one
        runs is exit 1 ("in progress") -> EngineError."""
        root = self.root(sub_dir)
        out = self._call(root, "linecall", video_id,
                         ["--device", "cuda", "--tracker-python", str(self.tracker_python),
                          "--tid-python", str(self.tid_python),
                          "--pose-weights", str(self.pose_weights)], on_tick=on_tick,
                         record=self.linecall_call_record)
        try:
            return LinecallResult(
                episodes=int(out["episodes"]),
                verdicts={str(k): int(v) for k, v in (out.get("verdicts") or {}).items()},
                episodes_json=self._inside(out["episodes_json"], root),
                provenance=self._inside(out["provenance"], root))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise EngineError(f"エンジン linecall の要約が読めません: {type(e).__name__}: {e}") from None

    def transcribe(self, sub_dir: Path, video_id: str, *,
                   on_tick: Tick | None = None) -> TranscribeResult:
        root = self.root(sub_dir)
        out = self._call(root, "transcribe", video_id,
                         ["--whisper-python", str(self.whisper_python)], on_tick=on_tick)
        transcript = out.get("transcript") or root / "tracks" / video_id / "transcript.json"
        return TranscribeResult(segments=int(out["segments"]),
                                transcript=self._inside(transcript, root))

    def write_transcript(self, sub_dir: Path, video_id: str,
                         segments: list[dict]) -> TranscribeResult:
        """Give the engine a transcript pfb_app already has (the teacher's pasted one):
        `<tracks>/<id>/transcript.json`, [{start, end, text}] in video seconds, as the
        engine's transcribe writes it. An open end (inf) becomes the next segment's start or
        the end of the day (the engine only compares times)."""
        _check_video_id(video_id)
        rows = []
        for seg in segments:
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            start, end = float(seg["start"]), float(seg["end"])
            rows.append({"start": start, "end": end if end != float("inf") else OPEN_END_S,
                         "text": text})
        f = self.root(sub_dir) / "tracks" / video_id / "transcript.json"
        _write_atomic(f, json.dumps(rows, ensure_ascii=False))
        return TranscribeResult(segments=len(rows), transcript=f)

    def set_transcript(self, sub_dir: Path, video_id: str, enabled: bool) -> None:
        """Rewrite the registry entry's `transcript` flag (behavior reads the transcript only
        when it is true). The entry is written by track(); EngineError if it does not exist."""
        _check_video_id(video_id)
        f = self.root(sub_dir) / "videos" / f"{video_id}.yaml"
        try:
            entry = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            raise EngineError(f"登録ファイルがありません（先に人物の追跡を実行してください）: {f}") from None
        entry["transcript"] = bool(enabled)
        tmp = f.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")
        tmp.replace(f)

    def behavior(self, sub_dir: Path, video_id: str, *, base_url: str, model: str,
                 condition: str | None = None, on_tick: Tick | None = None) -> BehaviorResult:
        """Behavior observation against a running server; resumable by the engine."""
        condition = condition or self._s.pfb3x3_condition
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choices: {CONDITIONS}")
        root = self.root(sub_dir)
        extra = ([] if not self._s.pfb3x3_behavior_concurrency else
                 ["--max-concurrency", str(int(self._s.pfb3x3_behavior_concurrency))])
        out = self._call(root, "behavior", video_id,
                         ["--frames", str(root / "frames"), "--exp", str(root / "exp"),
                          "--model", model, "--condition", condition, "--base-url", base_url,
                          *extra], on_tick=on_tick)
        return BehaviorResult(rows=self._inside(out["rows"], root), windows=int(out["windows"]),
                              errors=int(out["errors"]),
                              overlay_dir=self._inside(out["overlay_dir"], root),
                              transcript=bool(out.get("transcript", False)))

    def has_frames_command(self) -> bool:
        """True when the pinned engine has `frames` (behavior's frame cache ahead of behavior)."""
        try:
            return "def cmd_frames(" in (self._s.engine_dir / "src" / "pfb3x3" / "engine.py") \
                .read_text(encoding="utf-8")
        except OSError:
            return False

    def prefetch_frames(self, sub_dir: Path, video_id: str, condition: str | None = None, *,
                        on_tick: Tick | None = None) -> int | None:
        """Cache behavior's frames of this camera ahead of behavior (the engine's `frames`:
        every frame of the condition's grid, the files behavior would write itself; NVDEC
        when it can). Needs the registry entry only, so it can run beside `track`. Returns
        how many frames are cached; None when the engine refused (behavior then decodes
        them itself). Yield / GPU signals propagate as for any engine call."""
        condition = condition or self._s.pfb3x3_condition
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choices: {CONDITIONS}")
        root = self.root(sub_dir)
        out, _error = self._soft(root, "frames", video_id,
                                 ["--frames", str(root / "frames"), "--condition", condition],
                                 on_tick)
        return None if out is None else int(out.get("cached") or 0)

    def behavior_plan(self, sub_dir: Path, video_id: str, model: str,
                      condition: str | None = None) -> tuple[Path, int | None]:
        """(the engine's live rows file of a behavior run, the number of windows it will
        have), from what the engine wrote at track time only: the registry entry (start_s,
        end_s), tracks/<id>/meta.json (fps, n_frames) and tracks/<id>/boxes.txt (its largest
        frame). The count is None when any of them is missing or unreadable."""
        condition = condition or self._s.pfb3x3_condition
        root = self.root(sub_dir)
        rows = root / "exp" / model / condition / video_id / "behavior.jsonl"
        tracks = root / "tracks" / video_id
        try:
            entry = yaml.safe_load(
                (root / "videos" / f"{video_id}.yaml").read_text(encoding="utf-8"))
            info = json.loads((tracks / "meta.json").read_text(encoding="utf-8"))
            last = _last_box_frame(tracks / "boxes.txt")
            end_s = entry.get("end_s")
            planned = planned_windows(float(info["fps"]), int(info["n_frames"]),
                                      float(entry.get("start_s") or 0.0),
                                      None if end_s is None else float(end_s), last, condition)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError,
                yaml.YAMLError) as e:
            log.info("区間の数がわかりません video=%s: %s", video_id, e)
            planned = None
        return rows, planned

    def analysed_seconds(self, sub_dir: Path, video_id: str, start_s: float,
                         end_s: float | None) -> float | None:
        """The video seconds a track of this camera covers (from the task start to its end
        or the video's), from tracks/<id>/meta.json; None until the engine has written it."""
        try:
            info = json.loads((self.root(sub_dir) / "tracks" / video_id / "meta.json")
                              .read_text(encoding="utf-8"))
            length = int(info["n_frames"]) / float(info["fps"])
        except (OSError, ValueError, KeyError, TypeError, ZeroDivisionError):
            return None
        seconds = min(length, end_s if end_s is not None else length) - (start_s or 0.0)
        return seconds if seconds > 0 else None

    # --- vLLM server (scripts/serve.sh, scripts/stop_server.sh in engine_dir) ---

    def start_server(self, model: str, util: float, *, on_tick: Tick | None = None,
                     timeout_sec: float | None = None) -> ServerHandle:
        """Start the engine's vLLM server at the fixed utilisation on `pfb3x3_port`; returns once
        it is ready. Exit 3 -> EngineNeedsGpu; a yield while starting stops it again; over
        timeout_sec (None: engine_timeout_sec) EngineTimeout, and it is stopped again."""
        if not 0 < util < 1:
            raise ValueError(f"util must be in (0, 1), got {util}")
        port = self._s.pfb3x3_port
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        env = self._env(None, {"PFB_GPU_UTIL": f"{util:g}", "PFB_SERVE_PORT": str(port),
                               "PFB_SERVE_HOST": SERVE_HOST})   # read by serve.sh's --host fix
        cmd = ["bash", str(self._s.engine_dir / "scripts" / "serve.sh"), model, str(port)]
        try:
            res = self._run(cmd, env, self.runtime_dir / "logs", "serve", on_tick,
                            timeout=self._s.engine_timeout_sec if timeout_sec is None
                            else timeout_sec)
        except BaseException:
            self._cleanup_after_failed_start()
            raise
        if res.returncode == EXIT_NEEDS_GPU:
            raise EngineNeedsGpu(f"GPUの空きが足りないためサーバーを起動できません\n{res.stderr_tail}")
        if res.returncode != 0:
            if res.returncode != 4:   # 4 = refused (port busy / running): nothing was started
                self._cleanup_after_failed_start()
            raise EngineError(f"エンジンのサーバー起動に失敗しました (exit {res.returncode})",
                              res.stderr_tail, res.returncode, res.log_path)
        pid = self._read_pidfile()
        if pid is None:
            raise EngineError("サーバーは起動しましたが serve.pid がありません", res.stderr_tail,
                              log_path=res.log_path)
        return ServerHandle(pid=pid, port=port, model=model,
                            base_url=f"http://localhost:{port}/v1")

    @staticmethod
    def listen_addresses(port: int) -> list[str] | None:
        """The local addresses something listens on at TCP `port` (from `ss -Hltn`), e.g.
        ["127.0.0.1"] or ["0.0.0.0"]; None when ss cannot be run or read."""
        try:
            out = subprocess.run(["ss", "-Hltn", f"sport = :{int(port)}"], capture_output=True,
                                 text=True, timeout=10, check=True).stdout
        except (OSError, subprocess.SubprocessError, ValueError):
            return None
        return parse_listen_addresses(out)

    def _served_concurrency(self) -> int:
        return max(1, int(getattr(self._s, "served_concurrency", SERVED_CONCURRENCY)))

    def translator(self, handle: ServerHandle):
        """A text-only chat backend on the running server (translating the engine's English;
        as before: no schema text, failures per request, so its call_log answers still hit)."""
        from pfb_app.inference.served import ServedChatBackend
        return ServedChatBackend(handle.base_url, handle.model,
                                 max_concurrency=self._served_concurrency())

    def chat_backend(self, handle: ServerHandle):
        """The chat backend contribution and pfb ask on the running server: up to
        settings.served_concurrency requests of a batch at once, the JSON schema also stated in the
        text (as the in-process backend does), and a failure of the server itself raised
        (ServerFailed) instead of failing each request."""
        from pfb_app.inference.served import ServedChatBackend
        return ServedChatBackend(handle.base_url, handle.model, timeout_sec=SERVED_TIMEOUT_SEC,
                                 max_concurrency=self._served_concurrency(), schema_hint=True,
                                 strict_server=True)

    def stop_server(self, handle: ServerHandle) -> None:
        """Stop the server of `handle` (stop_server.sh verifies it is a vllm and stops its tree).
        Nothing is killed when serve.pid is gone or names another pid."""
        self._stop_pidfile(expected=handle.pid)

    def stop_stale_server(self) -> None:
        """Stop a server left behind in our runtime dir (e.g. by a crashed session).
        stop_server.sh only stops serve.pid's tree, and only if it is a vllm process."""
        self._stop_pidfile(expected=None)

    @property
    def call_record(self) -> Path:
        """`<pgid> <start time>` of the running engine command (track, behavior, ...), so that
        the worker can stop it after the session that started it was killed. A call made
        while another holds it records itself in `engine_call.<pid>.pgid` instead."""
        return self.runtime_dir / f"{CALL_RECORD_STEM}{CALL_RECORD_SUFFIX}"

    @property
    def linecall_call_record(self) -> Path:
        """call_record of the linecall command, which runs beside the other sessions."""
        return self.runtime_dir / "linecall_call.pgid"

    def stop_stale_linecall(self) -> bool:
        """stop_stale_call for the linecall command's own record."""
        return self.stop_stale_call(self.linecall_call_record)

    def extra_call_records(self) -> list[Path]:
        """The records of engine calls that ran beside the one in call_record."""
        return sorted(self.runtime_dir.glob(f"{CALL_RECORD_STEM}.*{CALL_RECORD_SUFFIX}"))

    def stop_stale_call(self, record: Path | None = None) -> bool:
        """Stop engine commands left running by a killed session (each whole process group,
        e.g. the tracker holding the GPU). Only the groups recorded in call_record and the
        extra records beside it (or in `record` alone) are signalled, and only if still ours:
        the leader has the recorded start time and runs `pfb3x3.engine`, or, once the leader
        is gone, every live member of the group is an engine or tracker process. Never by
        name. Returns True if a group was stopped."""
        if record is None:
            stopped = [self._stop_recorded(f) for f in [self.call_record,
                                                        *self.extra_call_records()]]
            return any(stopped)
        return self._stop_recorded(record)

    def _stop_recorded(self, f: Path) -> bool:
        try:
            fields = f.read_text(encoding="utf-8").split()
            pgid = int(fields[0])
            start = fields[1] if len(fields) > 1 else None
        except (FileNotFoundError, ValueError, IndexError):
            return False
        ours = _is_our_group(pgid, start)
        if ours:
            log.warning("残っていたエンジンのプロセスグループ %s を停止します", pgid)
            _killpg(pgid, signal.SIGTERM)
            deadline = time.monotonic() + self._grace
            while time.monotonic() < deadline and _group_alive(pgid):
                time.sleep(0.05)
            if _group_alive(pgid):
                _killpg(pgid, signal.SIGKILL)
        else:
            log.info("%s は動いているエンジンを指していないため削除します（停止しません）", f)
        f.unlink(missing_ok=True)
        return ours

    def server_alive(self, handle: ServerHandle) -> bool:
        """Whether the session's server process is still running (not whether it answers)."""
        return _pid_alive(handle.pid)

    def vllm_info(self, model: str) -> dict:
        """{"vllm_version", "vllm_venv"} of the venv serve.sh runs `model` with (the model's
        `venv` in the engine's configs/models.yaml, default ~/extern/venvs/qwen3vl, only read);
        None values when they cannot be found. Read from the installed package metadata, so
        nothing is imported or started."""
        venv = None
        try:
            cfg = yaml.safe_load((self._s.engine_dir / "configs" / "models.yaml")
                                 .read_text(encoding="utf-8")) or {}
            venv = Path(os.path.expanduser((cfg.get(model) or {}).get("venv")
                                           or VLLM_DEFAULT_VENV))
        except (OSError, yaml.YAMLError, AttributeError):
            return {"vllm_version": None, "vllm_venv": None}
        version = None
        for meta in sorted(venv.glob("lib/python*/site-packages/vllm-*.dist-info/METADATA")):
            try:
                version = next((line.split(":", 1)[1].strip() for line in
                                meta.read_text(encoding="utf-8").splitlines()
                                if line.startswith("Version:")), None)
            except OSError:
                continue
            if version:
                break
        return {"vllm_version": version, "vllm_venv": str(venv)}

    # --- internals ---

    def _stop_group(self, pgid: int) -> None:
        """SIGTERM the process group, wait up to the grace period for it to empty, then
        SIGKILL it. Only for a group we created whose id is still reserved: its leader is not
        reaped yet, or it still has members."""
        _killpg(pgid, signal.SIGTERM)
        deadline = time.monotonic() + self._grace
        while time.monotonic() < deadline and _group_alive(pgid):
            time.sleep(0.05)
        if _group_alive(pgid):
            log.warning("エンジンのプロセスグループ %s が SIGTERM 後も残っているため SIGKILL します", pgid)
            _killpg(pgid, signal.SIGKILL)

    def _cleanup_after_failed_start(self) -> None:
        """Stop a half-started server (ours: it is in our runtime dir) without masking the
        error that ended the start."""
        try:
            self.stop_stale_server()
        except Exception:
            log.exception("起動に失敗したエンジンのサーバーを停止できませんでした")

    def _read_pidfile(self) -> int | None:
        try:
            return int((self.runtime_dir / "serve.pid").read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            return None

    def _stop_pidfile(self, expected: int | None) -> None:
        pid = self._read_pidfile()
        if pid is None:
            return
        if expected is not None and pid != expected:
            raise EngineError(f"serve.pid ({pid}) が起動したサーバー ({expected}) と違うため停止しません")
        cmd = ["bash", str(self._s.engine_dir / "scripts" / "stop_server.sh")]
        res = self._run(cmd, self._env(None), self.runtime_dir / "logs", "stop", None,
                        timeout=600)
        if res.returncode != 0:
            raise EngineError(f"エンジンのサーバー停止に失敗しました (exit {res.returncode})",
                              res.stderr_tail, res.returncode, res.log_path)

    def _write_entry(self, root: Path, video_id: str, video_path: Path, start_s: float,
                     end_s: float | None, max_people: int | None, activity: str | None,
                     transcript: bool, *, domain: str = "generic", sport: str | None = None,
                     target: str = "person") -> Path:
        """The engine's registry entry `<root>/videos/<id>.yaml`. `sport` and `target` are
        written for sports entries, and for generic ones only when not the default (so a school
        entry stays as before)."""
        _check_video_id(video_id)
        if domain not in ENGINE_DOMAINS:
            raise ValueError(f"unknown engine domain {domain!r}; choices: {ENGINE_DOMAINS}")
        if target not in TARGETS:
            raise ValueError(f"unknown target {target!r}; choices: {TARGETS}")
        entry = {
            "video_id": video_id, "path": str(video_path), "domain": domain,
            "activity": activity, "start_s": float(start_s),
            "end_s": None if end_s is None else float(end_s),
            "max_people": None if max_people is None else int(max_people),
            "transcript": bool(transcript), "camera_notes": "",
        }
        if domain != "generic" or sport is not None:
            entry["sport"] = sport
        if domain != "generic" or target != "person":
            entry["target"] = target
        d = root / "videos"
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{video_id}.yaml"
        tmp = f.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(entry, allow_unicode=True, sort_keys=False), encoding="utf-8")
        tmp.replace(f)
        return f

    def _set_entry_sport(self, root: Path, video_id: str, sport: str) -> None:
        """Make the registry entry's sport `sport` (the engine reads it from there); a change
        is logged as a warning. No entry: EngineError (track first)."""
        _check_video_id(video_id)
        f = root / "videos" / f"{video_id}.yaml"
        try:
            entry = yaml.safe_load(f.read_text(encoding="utf-8")) or {}
        except FileNotFoundError:
            raise EngineError(f"登録ファイルがありません（先に人物の追跡を実行してください）: {f}") from None
        if entry.get("sport") == sport:
            return
        log.warning("%s の競技を %r から %r に書き換えます", f, entry.get("sport"), sport)
        entry["sport"] = sport
        _write_atomic(f, yaml.safe_dump(entry, allow_unicode=True, sort_keys=False))

    def _order(self, sub_dir: Path, cmd: str, video_id: str, base_url: str, model: str,
               on_tick: Tick | None) -> tuple[dict, str | None]:
        """(the engine's summary of grid / batting, None), or ({}, refusal)."""
        out, error = self._soft(self.root(sub_dir), cmd, video_id,
                                ["--base-url", base_url, "--model", model], on_tick)
        return out or {}, error

    def _names(self, sub_dir: Path, cmd: str, video_id: str, args: list[str],
               on_tick: Tick | None) -> dict[str, str] | None:
        out, _error = self._soft(self.root(sub_dir), cmd, video_id, args, on_tick)
        if out is None or not out.get("names"):
            return None
        return {str(k): str(v) for k, v in out["names"].items()}

    def _soft(self, root: Path, cmd: str, video_id: str, args: list[str],
              on_tick: Tick | None, timeout: float | None = None
              ) -> tuple[dict | None, str | None]:
        """(summary, None), or (None, the engine's error) when the command refused (exit 1
        with {"error"} that is not a broken call). A broken call (argv, registry entry, video
        id) is an EngineError, a dead server EngineServerDown, exit 3 EngineNeedsGpu.
        timeout: as in _call."""
        try:
            return self._call(root, cmd, video_id, args, on_tick=on_tick, soft_error=True,
                              timeout=timeout), None
        except _Refused as e:
            return None, e.error

    def _call(self, root: Path, cmd: str, video_id: str, args: list[str], *,
              on_tick: Tick | None, soft_error: bool = False,
              timeout: float | None = None, record: Path | None = None,
              env: dict[str, str] | None = None) -> dict | None:
        """Run one engine command. soft_error: exit 1 with a JSON {"error": ...} that is a real
        refusal raises _Refused (see _soft); a broken call stays an EngineError and a dead
        server becomes EngineServerDown."""
        _check_video_id(video_id)
        argv = [str(self.python), "-m", "pfb3x3.engine", cmd, "--video-id", video_id,
                "--registry", str(root / "videos"), "--tracks", str(root / "tracks"), *args]
        res = self._run(argv, self._env(root, env), root / "logs", cmd, on_tick,
                        timeout=self._s.engine_timeout_sec if timeout is None else timeout,
                        record=self.call_record if record is None else record)
        if res.returncode == EXIT_NEEDS_GPU:
            raise EngineNeedsGpu(f"エンジン {cmd}: GPUの空きが足りません\n{res.stderr_tail}")
        if res.returncode != 0:
            if soft_error and res.returncode == 1 and res.summary and "error" in res.summary:
                error = str(res.summary["error"])
                if SERVER_UNAVAILABLE in error:
                    raise EngineServerDown(f"PFB3x3のサーバーが応答しません（{cmd}）: {error}")
                if not any(h in error for h in _HARD_ERRORS):
                    log.warning("エンジン %s は結果を出しませんでした: %s", cmd, error)
                    raise _Refused(error)
            reason = f": {res.summary['error']}" if res.summary and "error" in res.summary else ""
            raise EngineError(f"エンジン {cmd} が失敗しました (exit {res.returncode}){reason}",
                              res.stderr_tail, res.returncode, res.log_path)
        if res.summary is None:
            raise EngineError(f"エンジン {cmd} の出力にJSONの要約がありません (no JSON summary)",
                              res.stderr_tail, 0, res.log_path)
        return res.summary

    @staticmethod
    def _inside(value, root: Path) -> Path:
        p = Path(str(value))
        try:
            p.resolve().relative_to(root.resolve())
        except ValueError:
            raise EngineError(f"エンジンの出力パス {p} が {root} の外にあります") from None
        return p

    def _run(self, argv: list[str], env: dict[str, str], log_dir: Path, name: str,
             on_tick: Tick | None, timeout: float | None, record: Path | None = None) -> _Outcome:
        """Run argv in its own process group with stdout/stderr in log_dir/<name>.{out,err}.
        Every tick_sec: on_tick() (heartbeat + yield poll); True -> terminate, EngineYield."""
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path, err_path = log_dir / f"{name}.out", log_dir / f"{name}.err"
        with open(out_path, "w", encoding="utf-8") as out_f, \
                open(err_path, "w", encoding="utf-8") as err_f:
            if not self._s.engine_dir.is_dir():
                raise EngineError(f"エンジンがセットアップされていません: {self._s.engine_dir} "
                                  "(scripts/setup_pfb3x3_engine.sh)")
            try:
                proc = subprocess.Popen(argv, cwd=self._s.engine_dir, env=env, stdout=out_f,
                                        stderr=err_f, stdin=subprocess.DEVNULL,
                                        start_new_session=True)
            except OSError as e:
                raise EngineError(f"エンジンを起動できません ({argv[0]}): {e}") from e
            started = time.monotonic()
            stamp = f"{proc.pid} {_start_time(proc.pid) or ''}".strip() + "\n"
            try:
                if record is not None:
                    record = _claim_record(record, stamp, proc.pid,
                                           shared=record == self.call_record)
                while True:
                    try:
                        proc.wait(timeout=self._tick_sec)
                    except subprocess.TimeoutExpired:
                        pass
                    else:   # children it left in its group (e.g. a tracker) must not outlive
                        if _group_alive(proc.pid):   # it; a group with members keeps its id
                            log.warning("エンジン %s の終了後もプロセスグループ %s が残っているため"
                                        "停止します", name, proc.pid)
                            self._stop_group(proc.pid)
                        break
                    if timeout is not None and time.monotonic() - started > timeout:
                        self._terminate(proc)
                        raise EngineTimeout(
                            f"エンジン {name} がタイムアウトしました ({timeout:g} 秒)",
                            _tail(err_path), proc.returncode, err_path)
                    if on_tick is not None and on_tick():
                        log.info("エンジン %s を中断します（GPUを譲る） pid=%s", name, proc.pid)
                        self._terminate(proc)
                        raise EngineYield(f"エンジン {name} を中断しました（GPUを譲る）")
            finally:
                if proc.poll() is None:   # an exception from on_tick or a signal: never orphan
                    self._terminate(proc)
                if record is not None:
                    _remove_if(record, stamp)
        return _Outcome(proc.returncode, _last_json(out_path), _tail(err_path), err_path)

    def _terminate(self, proc: subprocess.Popen) -> None:
        """Stop the whole process group we created for `proc` (start_new_session: pgid ==
        proc.pid): SIGTERM, wait up to the grace period for the group to empty, then SIGKILL
        the group. The leader is not reaped (no poll/wait) until the end, so even after it has
        exited its pid, and so the pgid, cannot have been reused while we signal the group. Grandchildren holding
        the GPU (the tracker, vLLM workers) therefore never outlive a yield or a timeout."""
        if proc.returncode is not None:   # already reaped: its pid/pgid may be someone else's
            return
        self._stop_group(proc.pid)
        try:
            proc.wait(timeout=30)   # reap the leader only now
        except subprocess.TimeoutExpired:
            log.error("エンジンのプロセス %s が SIGKILL 後も終了しません", proc.pid)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pose_weights_ready(path: Path) -> bool:
    """`path` is a regular file whose sha256 is POSE_SHA256 (hashed once per size, mtime and
    inode, so a page can ask on every run)."""
    try:
        st = Path(path).stat()
        if not Path(path).is_file():
            return False
        key = (str(path), st.st_size, st.st_mtime_ns, st.st_ino)
        if key not in _pose_seen:
            _pose_seen[key] = _file_sha256(Path(path))
        return _pose_seen[key] == POSE_SHA256
    except OSError:
        return False


def parse_listen_addresses(ss_output: str) -> list[str]:
    """The local address (without the port) of each `ss -Hltn` line: "127.0.0.1",
    "0.0.0.0", "[::1]", "*"..."""
    out = []
    for line in ss_output.splitlines():
        fields = line.split()
        if len(fields) >= 4 and ":" in fields[3]:
            out.append(fields[3].rsplit(":", 1)[0])
    return out


def _pid_alive(pid: int) -> bool:
    """True while `pid` exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _killpg(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _group_alive(pgid: int) -> bool:
    """True while a non-zombie process is in group `pgid`. Linux: from /proc; elsewhere it
    falls back to "the group can still be signalled" (zombies count, so this is conservative)."""
    proc_dir = Path("/proc")
    if not (proc_dir / "self" / "stat").exists():
        try:
            os.killpg(pgid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
    for stat in proc_dir.glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
            return True
    return False


def _start_time(pid: int) -> str | None:
    """Field 22 of /proc/<pid>/stat (start time in clock ticks); None without /proc."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _cmdline(pid: int) -> list[str]:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return []


def _is_our_group(pgid: int, start: str | None) -> bool:
    """Whether process group `pgid` is the engine command we recorded (Linux /proc only;
    without /proc nothing is ever treated as ours)."""
    if not Path("/proc/self/stat").exists():
        return False
    leader = _start_time(pgid)
    if leader is not None:   # the leader is alive: it must be the very process we started
        return start is not None and leader == start and "pfb3x3.engine" in _cmdline(pgid)
    members = []
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
            members.append(int(stat.parent.name))
    # a live group keeps its id reserved, so these are the leader's children
    return bool(members) and all(
        any("pfb3x3.engine" in a or "track_people" in a for a in _cmdline(m)) for m in members)


MIN_RESERVE_MIB = 4096   # the engine never leaves less than this to other GPU users


def engine_reserve_mib(settings: Settings) -> int:
    """settings.engine_gpu_reserve_gb in MiB (at least MIN_RESERVE_MIB), as the engine's
    PFB_GPU_RESERVE_MIB."""
    return max(MIN_RESERVE_MIB, int(round(float(settings.engine_gpu_reserve_gb) * 1024)))


CALL_RECORD_STEM, CALL_RECORD_SUFFIX = "engine_call", ".pgid"


def _claim_record(record: Path, stamp: str, pid: int, *, shared: bool) -> Path:
    """Write `stamp` to `record` and return the path written. `shared` (the engine's
    call_record): when another running call holds it, the stamp goes to
    `engine_call.<pid>.pgid` beside it instead (taken atomically: never overwritten)."""
    if not shared:
        _write_atomic(record, stamp)
        return record
    record.parent.mkdir(parents=True, exist_ok=True)
    tmp = record.with_name(f".{record.name}.{os.getpid()}.{pid}.tmp")
    tmp.write_text(stamp, encoding="utf-8")
    try:
        os.link(tmp, record)   # fails when the record exists: another call holds it
        return record
    except FileExistsError:
        extra = record.with_name(f"{CALL_RECORD_STEM}.{pid}{CALL_RECORD_SUFFIX}")
        os.replace(tmp, extra)
        tmp = None
        return extra
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _remove_if(path: Path, text: str) -> None:
    try:
        if path.read_text(encoding="utf-8") == text:
            path.unlink()
    except FileNotFoundError:
        pass


class _Refused(Exception):
    """A soft_error command refused (exit 1 with a real refusal); error: its text."""

    def __init__(self, error: str):
        super().__init__(error)
        self.error = error


def planned_windows(fps: float, n_frames: int, start_s: float, end_s: float | None,
                    last_box_frame: int | None, condition: str) -> int:
    """The number of behavior windows the engine makes (PFB3x3 pipeline.windows_for and
    windows.make_windows): frames every frame_step_s from the task start (1-based) up to the
    last frame found in the video, the task end and the boxes alike, grouped by
    frames_per_window; a final partial group counts with at least 2 frames. No boxes: 0."""
    step_s, per = WINDOW_PLAN[condition]
    start = round(start_s * fps) + 1
    end = None if end_s is None else round(end_s * fps) + 1
    last = min(n_frames, end or n_frames, last_box_frame or 0)
    frames = 0
    while start + round(frames * step_s * fps) <= last:
        frames += 1
    return frames // per + (1 if frames % per >= 2 else 0)


# count_rows answers by file: (size, mtime_ns) -> count; a file is re-read only when it changed
_ROWS: dict[str, tuple[tuple[int, int], int]] = {}
_ROWS_MAX = 64


def count_rows(path: Path) -> int:
    """Distinct window ids in an engine rows file being appended to: blank, partial or
    unreadable lines and repeated ids are not counted; a missing file has none. An unchanged
    file (same size and mtime) is not read again."""
    try:
        st = os.stat(path)
    except OSError:
        return 0
    stamp, key = (st.st_size, st.st_mtime_ns), str(path)
    hit = _ROWS.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    n = _count_rows(path)
    if len(_ROWS) >= _ROWS_MAX:
        _ROWS.clear()
    _ROWS[key] = (stamp, n)
    return n


def _count_rows(path: Path) -> int:
    ids = set()
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and row.get("id") is not None:
                    ids.add(str(row["id"]))
    except OSError:
        return 0
    return len(ids)


def _last_box_frame(path: Path) -> int | None:
    """The largest frame number of a MOT boxes file (frame,track_id,...); None when empty.
    An unreadable line raises ValueError."""
    last = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                frame = int(line.split(",")[0])
                last = frame if last is None else max(last, frame)
    return last


def probe_id(video_id: str) -> str:
    """The registry id of the content check of `video_id` (its own entry and track dir)."""
    return f"{video_id}{PROBE_SUFFIX}"


def _opt_num(value, kind):
    """kind(value), or None for a missing value."""
    return None if value is None else kind(value)


def _check_video_id(video_id: str) -> None:
    if not _VIDEO_ID.fullmatch(video_id):
        raise ValueError(f"invalid video id {video_id!r}")


def _last_json(path: Path) -> dict | None:
    """The last stdout line that parses to a JSON object, scanning backwards past trailing
    noise (torch/vLLM warnings, atexit output); None if no line is a JSON object."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _tail(path: Path, n: int = STDERR_TAIL_LINES) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=n)).rstrip()
    except FileNotFoundError:
        return ""
