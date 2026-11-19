"""Tracking engine (PFB3x3) stages: `track` per camera and `behavior` through the engine.

The engine is only reached through pfb_app.engine.Pfb3x3Engine (subprocesses). While an engine
command runs, `on_tick` refreshes the job's heartbeat and asks the session whether to give the
GPU back; the engine's own signals are turned into the runner's:

- EngineYield    -> YieldRequested (the job goes back to the queue, the session ends);
- EngineNeedsGpu -> GpuUnavailable (a YieldRequested carrying the reason; the session marks
  the key's jobs waiting_gpu);
- EngineError    -> left as it is: execute_job records failed_retryable with its stderr tail.

`on_tick` also writes the job's own progress when it changes (jobs.progress / progress_note,
pipeline/job_progress.py): track by its time against the usual one, behavior by the windows
the engine has written against the windows it will have.
"""
import json
import logging
import re
import statistics
import threading
import time
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

from pfb_app.domain import profiles
from pfb_app.domain.models import Job, Stage, StageRun
from pfb_app.engine import (SERVER_UNAVAILABLE, TARGETS, EngineError, EngineNeedsGpu,
                            EngineServerDown, EngineYield, Pfb3x3Engine)
from pfb_app.engine.pfb3x3 import count_rows
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline.common import StageContext, fail_run, save_run, usable
from pfb_app.pipeline.job_progress import behavior_progress, track_progress
from pfb_app.pipeline.pfb3x3_bridge import (BridgeError, bridge_behavior, engine_run_meta,
                                           english_event_ids, game_clock_offset_s,
                                           provenance_hash, provenance_parts, translate_events,
                                           translation_prompt, usable_row)
from pfb_app.pipeline.tracked_identity import label_mapping, mapped_track_ids, not_current_message
from pfb_app.storage import files

log = logging.getLogger(__name__)

# A pasted transcript is the group's speech as heard at camera 1 (one text for the lesson);
# other cameras transcribe their own audio when auto_transcribe is on, else get none.
PASTED_CAMERA = "cam1"

# PFB3x3 names.from_positions grid keys: f"{column}_{row}" (names.PLACE_COLS x PLACE_ROWS).
PLACE_COLS = ("left", "center", "right")
PLACE_ROWS = ("front", "middle", "back")
POSITION_KEYS = tuple(f"{c}_{r}" for c in PLACE_COLS for r in PLACE_ROWS)
SCHOOL_PLACES = 4   # the school rule defaults: exactly four students and four tracked people
# Words naming a column / a row in a typed initial position (after NFKC, spaces removed).
_POSITION_ALIASES = {
    "left": ("左",), "right": ("右",), "center": ("中央", "真ん中"),
    "front": ("手前", "前"), "back": ("奥", "後ろ", "後"), "middle": ("中段",),
}
_PLACE_WORDS = sorted(((w, part) for part, words in _POSITION_ALIASES.items() for w in words),
                      key=lambda wp: -len(wp[0]))
_PLACE_FILLER = "の側・-_/"
_ENGLISH_PLACE = re.compile(r"(left|center|centre|right)_(front|middle|back)")


class GpuUnavailable(YieldRequested):
    """The engine exited 3 (not enough free GPU memory): requeue and wait for the GPU."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def video_id(submission_id: int, camera: str) -> str:
    """The engine's video id of one camera of a submission (e.g. "s12_cam1")."""
    return f"s{submission_id}_{camera}"


def place_key(raw) -> str | None:
    """The grid key of a typed initial position, or None. English keys (left_front,
    Center_Back, ...) or Japanese with one column word (左 / 右 / 中央・真ん中) and one row word
    (前・手前 / 奥・後・後ろ / 中段), e.g. 左前, 右奥, 中央の手前, 真ん中中段."""
    text = "".join(unicodedata.normalize("NFKC", str(raw or "")).split()).lower()
    m = _ENGLISH_PLACE.fullmatch(text)
    if m:
        return f"{'center' if m.group(1) == 'centre' else m.group(1)}_{m.group(2)}"
    found, rest = set(), text
    for word, part in _PLACE_WORDS:   # longest first: 手前 before 前, 後ろ before 後
        if word in rest:
            found.add(part)
            rest = rest.replace(word, "")
    cols = [p for p in PLACE_COLS if p in found]
    rows = [p for p in PLACE_ROWS if p in found]
    if len(cols) != 1 or len(rows) != 1 or rest.strip(_PLACE_FILLER):
        return None   # anything else in the text (e.g. 右から2番目の前) is not a place
    return f"{cols[0]}_{rows[0]}"


def place_positions(subjects: list[dict], camera: str) -> dict[str, str] | None:
    """{grid key: participant_id} for PFB3x3's initial-place rule with any number of students
    (the names stage). Students without a place for this camera are left out; None when there
    is none, or when a place is not recognised or repeated."""
    found: dict[str, str] = {}
    for s in subjects:
        if not s.get("participant_id"):
            continue
        raw = (s.get("initial_positions") or {}).get(camera)
        if not str(raw or "").strip():
            continue
        key = place_key(raw)
        if key is None or key in found:
            return None
        found[key] = s["participant_id"]
    return {key: found[key] for key in POSITION_KEYS if key in found} or None


def rule_positions(subjects: list[dict], camera: str) -> dict[str, str] | None:
    """{grid key: participant_id} for the school defaults, or None unless exactly four
    students each have a distinct, recognised initial position for this camera."""
    students = [s for s in subjects if s.get("participant_id")]
    if len(students) != SCHOOL_PLACES:
        return None
    found: dict[str, str] = {}
    for s in students:
        key = place_key((s.get("initial_positions") or {}).get(camera))
        if key is None or key in found:
            return None
        found[key] = s["participant_id"]
    return {key: found[key] for key in POSITION_KEYS if key in found}


TRACKER_TEXTS = {"engine": "標準の追跡", "trackid3x3": "TrackID3x3 の追跡（すでにある結果）"}


def tracker_text(output: dict | None) -> str | None:
    """The tracker a track run's output says it used, in words for the admin status page
    (「標準の追跡」 / 「TrackID3x3 の追跡（すでにある結果）」); None when the engine did not
    say (an engine before 58f8341, or an unknown value)."""
    rec = output.get("tracker") if isinstance(output, dict) else None
    used = rec.get("used") if isinstance(rec, dict) else None
    return TRACKER_TEXTS.get(used) if isinstance(used, str) else None


def run_track(ctx: StageContext, job: Job) -> StageRun:
    """Person tracking + linking, the people sheet and the rule defaults for one camera.

    Output: {"people", "labels", "sheet", "track_stats", "default_names" ({P-label:
    participant_id}), "default_source" ("rule" or None), "device", "video_id", "boxes",
    "sheet_frame", "sheet_frame_rule", "sheet_frame_offset_s", "sheet_geometry", "transcript",
    "transcript_segments", "tracker" ({"used": "engine" | "trackid3x3", "fallback": why TID
    was not used, or None} when the engine says, else None)}. VLM name proposals are not made
    in v1 (rule only).

    The sheet and the rule use the same frame: the first frame >= the task start where everyone
    is boxed (--auto-start, within 60 s), else the engine's default frame (with a warning).
    Rule defaults need exactly four students, each with a distinct initial position for this
    camera (左前/左奥/右前/右奥 or left_front/...), and exactly four tracked people; otherwise
    there are no defaults and the teacher maps every label by hand.

    Transcript (registry entry `transcript: true`, so behavior reads it):
    - cam1 of a submission with a pasted transcript: that text is written as the engine's
      transcript.json (no whisper);
    - any camera with auto_transcribe (cam1 only without a pasted one): the engine transcribes
      this camera's own audio;
    - otherwise none. A failed or empty transcript (e.g. no audio) leaves it false: behavior
      then runs on the video alone. `transcript_source`: "pasted", "whisper" or None.
    """
    started = time.monotonic()
    repo, sid, cam = ctx.repo, job.submission_id, job.camera
    submission = repo.get_submission(sid)
    meta = submission["meta"]
    path = (meta.get("cameras") or {}).get(cam) if cam else None
    if not path:
        return fail_run(ctx, sid, Stage.track, f"カメラの動画が登録されていません（{cam}）。",
                        camera=cam, started=started)
    prepare = repo.latest_run(sid, Stage.prepare, cam)
    if not usable(prepare):
        return fail_run(ctx, sid, Stage.track,
                        f"前処理の結果がありません（{cam}）。先に前処理を実行してください。",
                        camera=cam, started=started)
    target = meta.get("target", "person")
    if target not in TARGETS:
        return fail_run(ctx, sid, Stage.track,
                        f"追跡する対象の設定が正しくありません（{cam}）。"
                        "人か車両のどちらかを選び直してください。", camera=cam, started=started)
    engine = _engine(ctx)
    aid = submission["assignment_id"]
    sub_dir = files.submission_dir(ctx.settings, aid, sid)
    vid = video_id(sid, cam)
    start_s = float(meta.get("start_sec") or 0.0)
    end_s = None if meta.get("end_sec") is None else float(meta["end_sec"])
    tick = _on_tick(ctx, job, _track_source(ctx, engine, sub_dir, vid, start_s, end_s, started))
    assignment = repo.get_assignment(aid)
    activity = profiles.engine_activity(assignment)   # 一般/スポーツ: the activity line alone
    sports = assignment.get("domain") == "sports"
    pasted = _pasted_segments(repo, sid) if cam == PASTED_CAMERA else None
    want_transcript = pasted is not None or bool(meta.get("auto_transcribe"))
    warnings: list[str] = []
    prefetch = _FramePrefetch(ctx, engine, sub_dir, vid, job)
    with _engine_signals(), prefetch:
        res = engine.track(sub_dir, vid, Path(path), float(meta.get("start_sec") or 0.0),
                           meta.get("end_sec"), meta.get("max_people"), activity,
                           transcript=want_transcript, device=ctx.device,
                           domain="sports" if sports else "generic",
                           sport=assignment.get("sport") if sports else None,
                           target=target, on_tick=tick, on_entry=prefetch.start)
        if res.people == 0:   # the engine's sheet cannot make a sheet of nobody (KeyError)
            return fail_run(ctx, sid, Stage.track,
                            f"動画に人が見つかりませんでした（{cam}）。人が映っている動画か、"
                            "解析する範囲を確かめてください。", camera=cam, started=started)
        quality = _quality(engine, sub_dir, vid, tick, warnings)
        sheet = _sheet(engine, sub_dir, vid, tick, warnings, res.boxes)
        defaults = _rule_defaults(engine, repo.list_subjects(sid), cam, sub_dir, vid,
                                  res.labels, tick, auto_start=sheet.frame_rule == "auto_start")
        if pasted is not None:   # the teacher's transcript, as the engine's own file
            segments = engine.write_transcript(sub_dir, vid, pasted).segments
            if segments == 0:
                warnings.append("貼り付けた文字起こしに発言がないため、映像だけで観察します")
        elif want_transcript:
            # whisper runs on the CPU: heartbeat only, never a GPU yield in the middle of it
            segments = _transcribe(engine, sub_dir, vid, _heartbeat_tick(ctx, job), warnings)
        else:
            segments = 0
    if want_transcript and segments == 0:
        engine.set_transcript(sub_dir, vid, False)
    output = {
        "people": res.people, "labels": res.labels, "sheet": str(sheet.sheet),
        "track_stats": json.loads(res.track_stats.read_text(encoding="utf-8")),
        "quality": quality, "default_names": defaults, "default_source": "rule" if defaults else None,
        "device": ctx.device, "video_id": vid, "boxes": str(res.boxes),
        "sheet_frame": sheet.frame, "sheet_frame_rule": sheet.frame_rule,
        "sheet_frame_offset_s": sheet.frame_offset_s, "sheet_geometry": sheet.geometry,
        "transcript": segments > 0, "transcript_segments": segments,
        "analysed_s": engine.analysed_seconds(sub_dir, vid, start_s, end_s),
        "transcript_source": (None if segments == 0 else "pasted" if pasted is not None
                              else "whisper"),
        # which tracker found the people (engine 58f8341; None from an older engine)
        "tracker": (None if res.tracker is None
                    else {"used": res.tracker, "fallback": res.tracker_fallback}),
    }
    return save_run(ctx, sid, Stage.track, camera=cam, upstream=[prepare.id],
                    model_id=_model_id(engine, "track"), output=output, warnings=warnings,
                    started=started)


def run_behavior(ctx: StageContext, job: Job) -> StageRun:
    """Behavior observation of one camera through the session's engine server; the engine's
    rows become behavior events of the mapped students (pipeline/pfb3x3_bridge.py).

    The engine's English observations are label-free (本人/別の参加者) and translated to
    Japanese by the session's own served model (ctx.translator in tests).

    Upstream: [identity (the teacher's mapping), prepare, track]. Fails when the engine gave
    no windows, when every window failed, or when its rows cannot be read. A camera where no
    tracked person was seen (all windows skipped), or where nobody is mapped, simply has no
    events."""
    started = time.monotonic()
    if ctx.server is None:
        raise RuntimeError("PFB3x3のサーバーが起動していません（pfb3x3セッションでのみ実行できます）")
    repo, sid, cam = ctx.repo, job.submission_id, job.camera
    submission = repo.get_submission(sid)
    prepare = repo.latest_run(sid, Stage.prepare, cam)
    track = repo.latest_run(sid, Stage.track, cam)
    identity = repo.latest_run(sid, Stage.identity, cam)
    missing = (f"前処理の結果がありません（{cam}）。" if not usable(prepare) else
               f"人物の追跡結果がありません（{cam}）。" if not usable(track) else
               f"人物と参加者の対応づけがありません（{cam}）。" if not usable(identity) else
               "人物と参加者の対応づけが確認されていません。" if not submission["identity_confirmed"]
               else None)
    if not missing and track.id not in mapped_track_ids(identity):
        missing = not_current_message(cam)
    if missing:
        return fail_run(ctx, sid, Stage.behavior, missing, camera=cam, started=started)
    engine = _engine(ctx)
    sub_dir = files.submission_dir(ctx.settings, submission["assignment_id"], sid)
    vid = track.output.get("video_id") or video_id(sid, cam)
    tick = _on_tick(ctx, job, _behavior_source(engine, sub_dir, vid, ctx.server.model))
    with _engine_signals():
        try:
            res = engine.behavior(sub_dir, vid, base_url=ctx.server.base_url,
                                  model=ctx.server.model, on_tick=tick)
        except EngineError as e:   # a dead server would fail every queued job the same way
            if SERVER_UNAVAILABLE in str(e) or not engine.server_alive(ctx.server):
                raise EngineServerDown(f"PFB3x3のサーバーが応答しません（{cam}）: "
                                       f"{_first_line(e)}") from e
            raise
    warnings: list[str] = []
    run_meta = engine_run_meta(res.rows)
    if run_meta is None:
        log.warning("PFB3x3のrun_meta.jsonを読めません: %s", res.rows.parent)
        warnings.append("PFB3x3の実行記録（run_meta.json）を読めないため、"
                        "エンジンのプロンプトと設定は記録されていません")
    parts = provenance_parts(engine.engine_commit, run_meta, translation_prompt(ctx.settings).hash)
    common = dict(camera=cam, upstream=[identity.id, prepare.id, track.id],
                  model_id=_model_id(engine, ctx.server.model),
                  prompt_hash=provenance_hash(parts), started=started)
    try:
        rows = res.load_rows()
        if not rows:
            return fail_run(ctx, sid, Stage.behavior,
                            f"PFB3x3が観察する区間がありませんでした（{cam}）。"
                            "動画の開始・終了時刻を確かめてください。", **common)
        if all(r.get("status") == "skipped" for r in rows if isinstance(r, dict)):
            warnings.append("追跡された人物が映っている区間がありませんでした")
        elif not any(usable_row(r) for r in rows if isinstance(r, dict)):
            return fail_run(ctx, sid, Stage.behavior,
                            f"PFB3x3の行動の観察が、すべての区間で失敗しました（{cam}、"
                            f"{len(rows)}区間、エラー{res.errors}件）。再生成してください。",
                            **common)
        output = bridge_behavior(rows, label_mapping(identity.output), cam,
                                 float(submission["meta"].get("start_sec") or 0.0),
                                 res.transcript, res.overlay_dir, warnings,
                                 clock_offset_s=game_clock_offset_s(
                                     submission["meta"],
                                     repo.get_assignment(submission["assignment_id"])["domain"]))
    except json.JSONDecodeError as e:
        return fail_run(ctx, sid, Stage.behavior,
                        f"PFB3x3の行動の結果を読み取れません（{cam}）: {e}", **common)
    except BridgeError as e:
        return fail_run(ctx, sid, Stage.behavior, f"{e}（{cam}）", **common)
    translator = ctx.translator if ctx.translator is not None else engine.translator(ctx.server)
    stats: dict = {}
    # a regeneration asks again (not from call_log) for what the previous run left English
    previous = repo.latest_run(sid, Stage.behavior, cam)
    retry = english_event_ids(previous.output) if usable(previous) else set()
    output = translate_events(output, translator, repo, ctx.settings, ctx.yield_check, warnings,
                              stats, retry_ids=retry)
    output["provenance"] = {**parts,
                            "translated": 0 < stats["total"] == stats["translated"],
                            "translated_events": stats["translated"],
                            "events": stats["total"],
                            "translator": translator.model_id if stats["total"] else None,
                            **engine.vllm_info(ctx.server.model)}
    if track.output.get("transcript") and not res.transcript:
        warnings.append("文字起こしがエンジンで読み込まれなかったため、映像だけで観察しました")
    return save_run(ctx, sid, Stage.behavior, output=output, warnings=warnings, **common)


# --- internals ---

class _FramePrefetch:
    """behavior's frames of the camera cached beside `track` (engine.prefetch_frames, in a
    thread, its own engine call: NVDEC beside the tracker), so behavior finds them ready. GPU
    track sessions only, and only with an engine that has the command. Its failure is logged
    and never fails the track (behavior then decodes the frames itself). Leaving the block
    waits for it; leaving it with an exception (a yield, a signal) stops it at its next tick."""

    def __init__(self, ctx: StageContext, engine, sub_dir: Path, vid: str, job: Job):
        self._ctx, self._engine, self._sub_dir, self._vid, self._job = ctx, engine, sub_dir, vid, job
        self._thread = None
        self._cancel = threading.Event()

    def start(self) -> None:
        if self._thread is not None or self._ctx.device == "cpu" \
                or not self._engine.has_frames_command():
            return
        self._thread = threading.Thread(target=self._run, name="pfb-frame-prefetch", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        started = time.monotonic()
        try:
            n = self._engine.prefetch_frames(self._sub_dir, self._vid,
                                             on_tick=self._cancel.is_set)
            log.info("行動の記録用のフレームを先に用意しました video=%s 枚数=%s %.0f秒", self._vid,
                     n, time.monotonic() - started)
        except Exception:   # noqa: BLE001 - behavior decodes the frames itself then
            log.exception("行動の記録用のフレームを先に用意できませんでした video=%s", self._vid)

    def __enter__(self) -> "_FramePrefetch":
        return self

    def __exit__(self, exc_type, *_exc) -> None:
        if self._thread is None:
            return
        if exc_type is not None:
            self._cancel.set()
        while self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                _beat(self._ctx, self._job)

def _quality(engine, sub_dir, vid, tick, warnings: list[str]):
    """The engine's per-label track quality, None (and a warning) when it cannot be had; the
    track run never fails on it. Yield / GPU signals are not EngineErrors and propagate."""
    try:
        quality = engine.quality(sub_dir, vid, on_tick=tick)
    except EngineError as e:
        log.warning("quality failed for %s: %s", vid, e)
        quality = None
    if quality is None:
        warnings.append("人の写り方を調べられなかったため、映っている長さで選びました")
    return quality


def _fullest_frame(boxes: Path | None) -> int | None:
    """The frame where the most people are boxed (the earliest of equals), from the engine's
    boxes file (frame,label,x,y,w,h,conf per line); None when it cannot be read."""
    if boxes is None:
        return None
    count: dict[int, int] = {}
    try:
        with open(boxes, encoding="utf-8") as f:
            for line in f:
                part = line.split(",", 2)
                if len(part) >= 2 and part[0].strip().isdigit():
                    count[int(part[0])] = count.get(int(part[0]), 0) + 1
    except OSError:
        return None
    return max(count, key=lambda fr: (count[fr], -fr)) if count else None


def _sheet(engine, sub_dir, vid, tick, warnings: list[str], boxes: Path | None = None):
    try:
        return engine.sheet(sub_dir, vid, auto_start=True, on_tick=tick)
    except EngineError as e:   # nobody has everyone boxed within 60 s of the task start
        log.info("開始直後に全員が写る画面がないため、既定の画面で一覧を作ります video=%s: %s", vid, e)
        warnings.append("開始直後に全員が写っている画面が見つからないため、"
                        "人物一覧は既定の画面で作りました")
        try:
            return engine.sheet(sub_dir, vid, auto_start=False, on_tick=tick)
        except EngineError as again:   # the people appear only later (an intro, a late start)
            frame = _fullest_frame(boxes)
            if frame is None:
                raise
            log.info("最初の画面にも人がいないため、人が最も多く写る画面で一覧を作ります "
                     "video=%s frame=%s: %s", vid, frame, again)
            warnings[-1] = ("動画のはじめには人が写っていないため、人物一覧は人が最も多く"
                            "写っている画面で作りました")
            return engine.sheet(sub_dir, vid, frame=frame, on_tick=tick)


def _pasted_segments(repo, submission_id: int) -> list[dict] | None:
    """The teacher's pasted transcript (a transcribe run with source teacher), else None."""
    run = repo.latest_run(submission_id, Stage.transcribe)
    if usable(run) and run.source == "teacher":
        return list(run.output.get("segments") or [])
    return None


def _transcribe(engine, sub_dir, vid, tick, warnings: list[str]) -> int:
    """Segments the engine transcribed from this camera's audio; 0 (plus a warning) when it
    failed or heard nothing. A yield or a GPU shortage still ends the job."""
    try:
        segments = engine.transcribe(sub_dir, vid, on_tick=tick).segments
    except EngineError as e:
        log.warning("PFB3x3の文字起こしに失敗しました video=%s: %s", vid, e)
        first = next((line for line in str(e).splitlines() if line.strip()), type(e).__name__)
        warnings.append(f"文字起こしに失敗したため、映像だけで観察します: {first}")
        return 0
    if segments == 0:
        warnings.append("音声から発言が見つからなかったため、映像だけで観察します")
    return segments


def _first_line(e: BaseException) -> str:
    return next((line for line in str(e).splitlines() if line.strip()), type(e).__name__)


def _engine(ctx: StageContext) -> Pfb3x3Engine:
    return ctx.engine if ctx.engine is not None else Pfb3x3Engine(ctx.settings)


def _model_id(engine: Pfb3x3Engine, name: str) -> str:
    commit = engine.engine_commit
    return f"pfb3x3:{name}" + (f"@{commit[:12]}" if commit else "")


def _rule_defaults(engine, subjects, camera, sub_dir, vid, labels, tick, *,
                   auto_start: bool) -> dict[str, str]:
    positions = rule_positions(subjects, camera)
    if positions is None or len(labels) != SCHOOL_PLACES:
        return {}   # the rule needs exactly four students with positions and four people
    names = engine.rule_names(sub_dir, vid, positions, auto_start=auto_start, on_tick=tick) or {}
    students, known = set(positions.values()), set(labels)
    return {label: pid for label, pid in names.items() if label in known and pid in students}


def _heartbeat_tick(ctx: StageContext, job: Job):
    """An on_tick that only refreshes the heartbeat (CPU work inside a GPU job)."""
    def tick() -> bool:
        _beat(ctx, job)
        return False
    return tick


def _beat(ctx: StageContext, job: Job) -> None:
    try:
        ctx.repo.heartbeat(job.id)
    except Exception:  # e.g. database locked: the session's heartbeat thread also beats
        log.exception("ハートビートの更新に失敗しました job=%s", job.id)


def _on_tick(ctx: StageContext, job: Job,
             progress: Callable[[], tuple[float | None, str | None]] | None = None):
    """Every engine tick: refresh the heartbeat, write the job's own progress when it has
    changed (`progress` gives (progress, note); asked once per tick), then ask whether to give
    the GPU back. A failing progress source is logged and never stops the job."""
    written: list = []   # the last (progress, note) written
    failed: list = []    # a failure was logged already (once per job)

    def tick() -> bool:
        _beat(ctx, job)
        if progress is not None:
            try:
                value = progress()
                if not written or written[0] != value:
                    # never lower: a reclaimed job's time restarts, its bar must not go back
                    ctx.repo.set_job_progress(job.id, *value, never_lower=True)
                    written[:] = [value]
            except Exception:  # e.g. database locked: the next tick tries again
                if not failed:
                    log.exception("進み具合の更新に失敗しました job=%s", job.id)
                    failed.append(1)
                else:
                    log.debug("進み具合の更新に失敗しました job=%s", job.id, exc_info=True)
        return ctx.yield_check()
    return tick


def _track_source(ctx: StageContext, engine, sub_dir: Path, vid: str, start_s: float,
                  end_s: float | None, started: float):
    """The track job's progress: its time against the expected time. Expected: the usual
    seconds per analysed video second (earlier tracks) x this camera's analysed seconds (the
    task's length, or the video's once the engine has read it); without those, the median
    time of earlier tracks; None without any history. The history is read once per job."""
    try:
        rates = ctx.repo.track_rates()
        durations = ctx.repo.stage_durations(Stage.track, "pfb3x3")
    except Exception:
        log.exception("人物の追跡にかかる時間を読めません")
        rates, durations = [], []
    plain = statistics.median(durations) if durations else None
    rate = statistics.median(rates) if rates else None
    scaled: list[float] = []

    def source() -> tuple[float | None, str]:
        if rate is not None and not scaled:
            seconds = (end_s - start_s if end_s is not None and end_s > start_s
                       else engine.analysed_seconds(sub_dir, vid, start_s, end_s))
            if seconds:
                scaled.append(rate * seconds)
        return track_progress(time.monotonic() - started, scaled[0] if scaled else plain)
    return source


def _behavior_source(engine, sub_dir: Path, vid: str, model: str):
    """The behavior job's progress: the windows written so far / the planned windows, the
    plan read once per job. None when the plan cannot be had."""
    try:
        rows, planned = engine.behavior_plan(sub_dir, vid, model)
    except Exception:
        log.exception("行動の記録の区間を数えられません video=%s", vid)
        return None
    return lambda: behavior_progress(count_rows(rows), planned)


@contextmanager
def _engine_signals():
    try:
        yield
    except EngineYield as e:
        raise YieldRequested(str(e)) from e
    except EngineNeedsGpu as e:
        lines = [line.strip() for line in str(e).splitlines() if line.strip()]
        reason = lines[0] if lines else "GPUの空きが足りません"
        if len(lines) > 1:
            reason += f"（{lines[-1]}）"
        raise GpuUnavailable(reason) from e
