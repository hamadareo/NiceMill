"""Run one claimed job: dispatch its stage, record the job outcome, enqueue the next stages."""
import logging
import sqlite3
import sys
import time
from pathlib import Path

from pfb_app.domain.models import Job, JobStatus, Kind, Stage, StageRun
from pfb_app.engine import EngineServerDown
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import detect, document, linecall, naming, plain, tips, tracking, video
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import (StageContext, fail_run, save_run, submission_engine,
                                     usable)
from pfb_app.preprocess.documents import DocumentError, extract_pages, to_pdf
from pfb_app.preprocess.frames import VideoError, extract_frames, probe_video, validate_interval
from pfb_app.preprocess.transcript import transcribe  # module attribute: tests replace it
from pfb_app.services import proxy
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

CPU_KEY = "cpu"
WHISPER_KEY = "whisper"
TRACK_KEY = "track"          # the tracking engine's track stage (GPU, or CPU as a fallback)
DETECT_KEY = "detect"        # the content check (with the served model, or CPU-only at once)
LINECALL_KEY = "linecall"    # 判定支援: the engine's linecall (GPU only, no server, no CPU fallback)
PFB3X3_PREFIX = "pfb3x3:"    # pfb3x3:<standard|high>: behavior through the engine's vLLM server
NAMES_CPU_KEY = naming.NAMES_CPU_KEY   # a names job whose cues need no vLLM server
_OOM_TEXT = "CUDA out of memory"
# tracking-engine stages served by the pfb3x3:<key> session's vLLM server (text-only chat for
# contribution and pfb, the engine's own requests for behavior)
_SERVED_STAGES = (Stage.behavior, Stage.contribution, Stage.pfb, Stage.plain, Stage.tips)


class SessionMustEnd(Exception):
    """CUDA ran out of memory: the inference session must exit to give the VRAM back."""


def is_cuda_oom(exc: BaseException) -> bool:
    """True for torch.cuda.OutOfMemoryError or any error saying "CUDA out of memory",
    including errors that wrap one. torch is only consulted if it is already loaded."""
    torch = sys.modules.get("torch")
    oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(oom_cls, type) and isinstance(exc, oom_cls):
            return True
        if _OOM_TEXT in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# --- enqueueing ---

def _assignment(repo: Repo, submission_id: int) -> dict:
    return repo.get_assignment(repo.get_submission(submission_id)["assignment_id"])


def pfb3x3_key(model_key: str) -> str:
    return f"{PFB3X3_PREFIX}{model_key}"


def model_key_for(repo: Repo, submission_id: int, stage: Stage,
                  camera: str | None = None) -> str:
    """The session key a job of this stage runs under. names: pfb3x3:<key>, or NAMES_CPU_KEY
    when a camera is given and its wanted cues need no server (naming.needs_server)."""
    if stage == Stage.detect:
        return DETECT_KEY
    if stage == Stage.prepare:
        return CPU_KEY
    if stage == Stage.transcribe:
        return WHISPER_KEY
    if stage == Stage.track:
        return TRACK_KEY
    if stage == Stage.linecall:
        return LINECALL_KEY
    submission = repo.get_submission(submission_id)
    model_key = repo.get_assignment(submission["assignment_id"])["model_key"]
    if stage == Stage.names and camera is not None \
            and not naming.needs_server(repo, submission, camera):
        return NAMES_CPU_KEY
    if stage == Stage.names or (stage in _SERVED_STAGES
                                and submission_engine(submission) == "pfb3x3"):
        # names: the cues ask the session's vLLM server; behavior, contribution and pfb of
        # the tracking engine are all answered by that same server (no second model load)
        return pfb3x3_key(model_key)
    return model_key


def enqueue_once(repo: Repo, submission_id: int, stage: Stage, camera: str | None = None,
                 subject_id: int | None = None) -> int | None:
    """Enqueue unless a queued/running/waiting job already exists for this stage key.

    Returns the new job id, or None when skipped. Nothing is queued for a group whose work is
    to be deleted (services.deletion: meta["delete_requested"]) or whose videos were removed
    (services.retention: the analysis would need them).
    """
    with repo._tx(join=True):   # the checks and the insert at once (retention sets its flag
        # in a write transaction too)
        if repo.has_active_job(submission_id, stage, camera, subject_id):
            return None
        if _delete_requested(repo, submission_id) or _media_removed(repo, submission_id) \
                or _erasing(repo, submission_id):
            return None
        return repo.enqueue(submission_id, stage,
                            model_key_for(repo, submission_id, stage, camera),
                            camera=camera, subject_id=subject_id)


def _delete_requested(repo: Repo, submission_id: int) -> bool:
    from pfb_app.services import deletion   # noqa: PLC0415 - services import this module
    try:
        return deletion.is_requested(repo.get_submission(submission_id)["meta"])
    except KeyError:   # the group is gone: nothing to queue for it
        return True


def _erasing(repo: Repo, submission_id: int) -> bool:
    """services.erasure is editing the group's files (Repo.erasing)."""
    return repo.erasing(submission_id)


def _media_removed(repo: Repo, submission_id: int) -> bool:
    row = repo.conn.execute(
        "SELECT a.media_removed_at FROM assignments a JOIN submissions s"
        " ON s.assignment_id = a.id WHERE s.id=?", (submission_id,)).fetchone()
    return row is not None and bool(row[0])


def enqueue_followups(repo: Repo, job: Job) -> list[int]:
    """Enqueue the stages that follow a successfully finished job. Returns new job ids.

    `detect` starts nothing: prepare waits for the coach to confirm the content check."""
    if job.stage == Stage.detect:
        return []
    sid = job.submission_id
    submission = repo.get_submission(sid)
    kind = Kind(repo.get_assignment(submission["assignment_id"])["kind"])
    targets: list[tuple[Stage, str | None, int | None]] = []
    if kind == Kind.document:
        if job.stage == Stage.prepare:
            targets = [(Stage.evidence, None, None)]
        elif job.stage == Stage.evidence:
            targets = [(Stage.contribution, None, None)]
        elif job.stage == Stage.contribution:
            targets = [(Stage.pfb, None, s["id"]) for s in repo.list_subjects(sid)]
    else:
        targets = _video_followups(repo, job, submission)
    new_ids = [enqueue_once(repo, sid, stage, camera, subject_id)
               for stage, camera, subject_id in targets]
    return [i for i in new_ids if i is not None]


_TRACKED_FRONT = (Stage.prepare, Stage.transcribe, Stage.track, Stage.identity, Stage.names)

needs_names = naming.needs_names


def request_names(repo: Repo, submission_id: int) -> list[int]:
    """Enqueue `names` again for every camera whose latest track run is usable (after a roster
    change), when naming.names_wanted holds for it. A camera with a queued/waiting_gpu names
    job keeps it (one under another key, e.g. CPU-only when a server cue is now wanted, is
    replaced); a running one gets a fresh queued job behind it, since it reads the old
    subjects. Returns the new job ids."""
    from pfb_app.services import deletion   # noqa: PLC0415 - services import this module
    submission = repo.get_submission(submission_id)
    if submission_engine(submission) != "pfb3x3" or deletion.is_requested(submission["meta"]) \
            or _media_removed(repo, submission_id) or _erasing(repo, submission_id):
        return []
    cameras = list(submission["meta"].get("cameras") or {})
    cameras += [c for c in repo.list_cameras(submission_id) if c not in cameras]
    new_ids = []
    for cam in cameras:
        if not usable(repo.latest_run(submission_id, Stage.track, cam)) \
                or not naming.names_wanted(repo, submission, cam):
            continue
        key = model_key_for(repo, submission_id, Stage.names, cam)
        if key in repo.pending_job_keys(submission_id, Stage.names, cam):
            continue
        repo.cancel_pending_jobs(submission_id, Stage.names, cam)
        new_ids.append(repo.enqueue(submission_id, Stage.names, key, camera=cam))
    return new_ids


def _auto_people(repo: Repo, job: Job) -> None:
    """After a camera's track run: its automatic mapping, then the confirmation once every
    camera has one (services.auto_people; it queues behavior itself). Any error is logged and
    leaves the people step to the coach; the finished track job never fails on it."""
    from pfb_app.services import auto_people   # services.review imports this module
    try:
        auto_people.auto_identity(repo, job.submission_id, job.camera)
        auto_people.auto_confirm(repo, job.submission_id)
    except Exception:   # noqa: BLE001 - the coach can still map and confirm by hand
        log.exception("映っている人を自動で選べませんでした submission=%s camera=%s",
                      job.submission_id, job.camera)


def _number_names(repo: Repo, job: Job) -> None:
    """After a names run: the automatic people get the numbers read on them as names
    (services.auto_people.number_names). Any error is logged; the names job never fails on it."""
    from pfb_app.services import auto_people   # services.review imports this module
    try:
        auto_people.number_names(repo, job.submission_id)
    except Exception:   # noqa: BLE001 - names stay as they were; the coach can name people
        log.exception("背番号から名前をつけられませんでした submission=%s camera=%s",
                      job.submission_id, job.camera)


def _tracked_front_followups(repo: Repo, job: Job, submission: dict):
    """Tracking engine up to the mapping: prepare(cam) -> track(cam) -> names(cam)
    (only when naming.names_wanted holds: needs_names plus a gate-enabled cue with its
    input, or a sports work's numbers for names); transcribe, identity and names start
    nothing (a finished names run renames the automatic people by number: _number_names). pfb_app's own
    whisper transcript is not made: the track stage has the engine transcribe each camera.
    A finished track first gets its automatic people (_auto_people): once every camera has
    a current mapping, services.review.confirm_identity queues behavior (those job ids are
    not part of this list). The names stage never holds that up."""
    if job.stage == Stage.track:
        _auto_people(repo, job)
        submission = repo.get_submission(job.submission_id)   # auto people changed its meta
        wanted = naming.names_wanted(repo, submission, job.camera)
        return [(Stage.names, job.camera, None)] if wanted else []
    if job.stage == Stage.names:
        _number_names(repo, job)
        return []
    if job.stage != Stage.prepare:
        return []
    cameras = [job.camera] if job.camera else repo.list_cameras(job.submission_id)
    return [(Stage.track, cam, None) for cam in cameras]


def _video_followups(repo: Repo, job: Job, submission: dict):
    sid = job.submission_id
    if job.stage in _TRACKED_FRONT and submission_engine(submission) == "pfb3x3":
        return _tracked_front_followups(repo, job, submission)
    if job.stage == Stage.prepare:
        meta = submission["meta"]
        if meta.get("auto_transcribe") and repo.latest_run(sid, Stage.transcribe) is None:
            return [(Stage.transcribe, None, None)]
        cameras = [job.camera] if job.camera else repo.list_cameras(sid)
        return [(Stage.identity, cam, None) for cam in cameras]
    if job.stage == Stage.transcribe:
        return [(Stage.identity, cam, None) for cam in repo.list_cameras(sid)]
    if job.stage == Stage.identity:
        gated = _assignment(repo, sid)["require_identity_approval"]
        if gated and not submission["identity_confirmed"]:
            return []  # services.review.confirm_identity enqueues behavior later
        return [(Stage.behavior, job.camera, None)]
    if job.stage == Stage.behavior:
        return [(Stage.contribution, job.camera, None)]
    if job.stage == Stage.contribution:
        for cam in submission["meta"].get("cameras") or {}:  # registered, not just prepared
            run = repo.latest_run(sid, Stage.contribution, cam)
            if not usable(run) or repo.is_stale(run):
                return []  # wait until every camera has an up-to-date contribution
        return [(Stage.pfb, None, s["id"]) for s in repo.list_subjects(sid)
                if is_evaluated(s)]
    return []


# --- the content check's end ---

def mark_check_ready(repo: Repo, submission_id: int) -> bool:
    """A group held for the content check (meta["check"] "pending") becomes "ready": its check
    has ended, whether it worked or not. One atomic update; True when it changed the row."""
    cur = repo.conn.execute(
        "UPDATE submissions SET meta = json_set(meta, '$.check', 'ready')"
        " WHERE id=? AND json_extract(meta, '$.check') = 'pending'", (submission_id,))
    return cur.rowcount == 1


def _detect_ended(repo: Repo, job: Job) -> None:
    """After a detect job's final outcome (done, failed, failed_retryable): the card is ready.
    Never breaks the job's own bookkeeping (the card also promotes, as a fallback)."""
    if job.stage != Stage.detect:
        return
    try:
        mark_check_ready(repo, job.submission_id)
    except sqlite3.Error:
        log.exception("内容確認の完了を記録できませんでした submission=%s", job.submission_id)


# --- execution ---

# A pfb3x3 session's server died while running a job (EngineServerDown): the job is requeued
# for the next session; after SERVER_DOWN_LIMIT such stops in a row (since the teacher last
# re-ran it) it becomes failed_retryable instead.
SERVER_DOWN_LIMIT = 3
SERVER_DOWN_ACTION = "engine_server_down"
SERVER_DOWN_CAP_ACTION = "engine_server_down_cap"
SERVER_DOWN_MESSAGE = "解析サーバーが繰り返し停止しました。詳細画面から再実行してください。"


def execute_job(job: Job, ctx: StageContext) -> None:
    """Run a claimed (running) job and record its outcome; then finish the deletion of works
    that waited for their running step (services.deletion), whatever the outcome.

    - YieldRequested: the job goes back to queued and the exception is re-raised.
    - EngineServerDown (a YieldRequested): the same, up to SERVER_DOWN_LIMIT times in a row;
      then the job becomes failed_retryable (SERVER_DOWN_MESSAGE). Re-raised either way.
    - Other exceptions: the job becomes failed_retryable with the error; a CUDA OOM
      additionally raises SessionMustEnd.
    - A stage that saved a failed StageRun: the job becomes failed (not retryable).

    A failed_retryable job whose reason passes is put back in the queue once
    (services.auto_retry); any other stop is told to the operator (ops.alerts, when set up).
    The model answers the job writes or reuses are recorded as its group's
    (Repo.call_owner), so deleting the work deletes them.
    """
    previous = getattr(ctx.repo, "call_owner", None)
    ctx.repo.call_owner = job.submission_id
    try:
        _execute_job(job, ctx)
    finally:
        ctx.repo.call_owner = previous
        _finish_requested_deletions(ctx)


def _after_failure(ctx: StageContext, job: Job) -> None:
    from pfb_app.services import auto_retry   # noqa: PLC0415 - services import this module
    auto_retry.after_failure(ctx.repo, job.id, ctx.settings)


def _finish_requested_deletions(ctx: StageContext) -> None:
    from pfb_app.services import deletion   # noqa: PLC0415 - services import this module
    try:
        deletion.finish_requested_deletions(ctx.repo, ctx.settings)
    except Exception:   # noqa: BLE001 - never breaks the job's own bookkeeping
        log.exception("削除を頼まれた作業の確認に失敗しました")


def _auto_approve(ctx: StageContext, runs: list[StageRun]) -> None:
    """「確認せずに自動で承認する」: the new feedback texts that need no person's look are
    approved (services.review.auto_approve); a failure here never breaks the job."""
    from pfb_app.services import review   # noqa: PLC0415 - services import this module
    for run in runs:
        try:
            if review.auto_approve(ctx.repo, run):
                log.info("自動で承認しました subject=%s run=%s", run.subject_id, run.id)
        except Exception:   # noqa: BLE001 - the text stays for a person to approve
            log.exception("自動の承認に失敗しました run=%s", run.id)


def _execute_job(job: Job, ctx: StageContext) -> None:
    log.info("ジョブ開始 job=%s stage=%s submission=%s camera=%s subject=%s",
             job.id, job.stage.value, job.submission_id, job.camera, job.subject_id)
    try:
        runs = _run_stage(job, ctx)
    except EngineServerDown as e:   # requeued, but not forever: a server that keeps dying
        target = f"job:{job.id}"
        ctx.repo.audit(SERVER_DOWN_ACTION, target, {"error": str(e)})
        n = ctx.repo.count_audit_since(SERVER_DOWN_ACTION, target, SERVER_DOWN_CAP_ACTION)
        if n >= SERVER_DOWN_LIMIT:
            ctx.repo.audit(SERVER_DOWN_CAP_ACTION, target, {"count": n})
            ctx.repo.finish_job(job.id, JobStatus.failed_retryable,
                                f"{SERVER_DOWN_MESSAGE}\n{e}")
            _detect_ended(ctx.repo, job)
            _after_failure(ctx, job)
            log.error("解析サーバーが%d回停止したため、ジョブを失敗（再試行可）にしました job=%s",
                      n, job.id)
        else:
            ctx.repo.finish_job(job.id, JobStatus.queued)
            log.warning("解析サーバーが停止したためジョブをキューに戻しました（%d/%d） job=%s",
                        n, SERVER_DOWN_LIMIT, job.id)
        raise   # the session ends either way: its server is gone
    except YieldRequested:
        ctx.repo.finish_job(job.id, JobStatus.queued)
        log.info("GPUを譲るためジョブをキューに戻しました job=%s", job.id)
        raise
    except Exception as e:
        message = f"{type(e).__name__}: {e}"
        ctx.repo.finish_job(job.id, JobStatus.failed_retryable, message)
        _detect_ended(ctx.repo, job)
        _after_failure(ctx, job)
        if is_cuda_oom(e):
            log.error("CUDAのメモリ不足 (OOM) job=%s: %s", job.id, message)
            raise SessionMustEnd(message) from e
        log.exception("ジョブ失敗（再試行可） job=%s", job.id)
        return
    failed = [r for r in runs if r.status == "failed"]
    if failed:
        ctx.repo.finish_job(job.id, JobStatus.failed, "\n".join(r.error or "" for r in failed))
        _detect_ended(ctx.repo, job)
        _after_failure(ctx, job)
        log.warning("ジョブ失敗 job=%s: %s", job.id, failed[0].error)
        return
    finished = ctx.repo.finish_job(job.id, JobStatus.done)
    _detect_ended(ctx.repo, job)
    if not finished:
        log.warning("ジョブはすでに実行中ではありません（回収済み）。次段階は投入しません job=%s",
                    job.id)
        return
    if job.stage == Stage.pfb:
        _auto_approve(ctx, runs)
    new_ids = enqueue_followups(ctx.repo, job)
    log.info("ジョブ完了 job=%s 次段階=%s", job.id, new_ids)
    if job.stage == Stage.pfb and Kind(_assignment(ctx.repo, job.submission_id)["kind"]) \
            == Kind.video:
        tips.team_after_pfb(ctx, job.submission_id)   # queues the group's tips once all texts are done


def _run_stage(job: Job, ctx: StageContext) -> list[StageRun]:
    sid = job.submission_id
    kind = Kind(_assignment(ctx.repo, sid)["kind"])
    if job.stage == Stage.detect:
        if kind == Kind.video:
            return [detect.run_detect(ctx, job)]
        # a document's kind comes from the file type: nothing to check, no engine call
        return [save_run(ctx, sid, Stage.detect, output={"kind": Kind.document.value},
                         started=time.monotonic())]
    if job.stage == Stage.prepare:
        return prepare_video(ctx, job) if kind == Kind.video else [prepare_document(ctx, sid)]
    if job.stage == Stage.transcribe:
        return [run_transcribe(ctx, sid)]
    if job.stage == Stage.plain:   # a reworded text: the same for documents and videos
        return [plain.run_plain(ctx, job)]
    if kind == Kind.document:
        if job.stage == Stage.evidence:
            return [document.run_evidence(ctx, sid)]
        if job.stage == Stage.contribution:
            return [document.run_contribution(ctx, sid)]
        if job.stage == Stage.pfb:
            return [document.run_pfb(ctx, sid, job.subject_id)]
    else:
        if job.stage == Stage.track:
            return [tracking.run_track(ctx, job)]
        if job.stage == Stage.linecall:
            return [linecall.run_linecall(ctx, job)]
        if job.stage == Stage.names:
            return [naming.run_names(ctx, job)]
        if job.stage == Stage.identity:
            return [video.run_identity(ctx, sid, job.camera)]
        if job.stage == Stage.behavior:
            if submission_engine(ctx.repo.get_submission(sid)) == "pfb3x3":
                return [tracking.run_behavior(ctx, job)]
            return [video.run_behavior(ctx, sid, job.camera)]
        if job.stage == Stage.contribution:
            return [video.run_contribution(ctx, sid, job.camera)]
        if job.stage == Stage.pfb:
            run = video.run_pfb(ctx, sid, job.subject_id)
            tips.after_pfb(ctx, run)   # next to the feedback; never fails it (pipeline.tips)
            return [run]
        if job.stage == Stage.tips:    # the group's tips: a side job queued after the last text
            return tips.run_team_job(ctx, job)
    raise ValueError(f"この活動の種類（{kind.value}）では実行できない段階です: {job.stage.value}")


def prepare_video(ctx: StageContext, job: Job) -> list[StageRun]:
    """Extract frames for job.camera (or every camera); one prepare run per camera."""
    sid = job.submission_id
    submission = ctx.repo.get_submission(sid)
    meta = submission["meta"]
    cameras: dict[str, str] = meta.get("cameras") or {}
    if job.camera:
        cameras = {job.camera: cameras.get(job.camera, "")}
    if not cameras:
        return [fail_run(ctx, sid, Stage.prepare, "カメラの動画が登録されていません。",
                         camera=job.camera)]
    s = ctx.settings
    runs = []
    for camera, path in cameras.items():
        started = time.monotonic()
        try:
            info = probe_video(path)
            start, end = validate_interval(info, meta.get("start_sec"), meta.get("end_sec"),
                                           s.frame_interval_sec)
            out_dir = files.frames_dir(s, submission["assignment_id"], sid, camera)
            small, frames = proxy.ready(s, path, for_analysis=True), None   # 1080p, if made
            if small is not None:
                try:
                    frames = extract_frames(small, out_dir, start, end, s.frame_interval_sec,
                                            tuple(s.frame_size))
                except VideoError:   # a proxy that cannot serve every time: the original
                    small = None
            if frames is None:
                frames = extract_frames(path, out_dir, start, end, s.frame_interval_sec,
                                        tuple(s.frame_size))
        except VideoError as e:
            runs.append(fail_run(ctx, sid, Stage.prepare, str(e), camera=camera,
                                 started=started))
            break
        output = {"frames": [f.to_dict() for f in frames]}
        if small is not None:
            output["proxy"] = True
        runs.append(save_run(ctx, sid, Stage.prepare, camera=camera, output=output,
                             started=started))
    return runs


def prepare_document(ctx: StageContext, submission_id: int) -> StageRun:
    """Convert to PDF and extract pages; one prepare run with {"pages", "pdf_path"}."""
    started = time.monotonic()
    submission = ctx.repo.get_submission(submission_id)
    aid = submission["assignment_id"]
    try:
        pdf = to_pdf(Path(submission["meta"]["source_path"]),
                     files.submission_dir(ctx.settings, aid, submission_id))
        pages = extract_pages(pdf, files.pages_dir(ctx.settings, aid, submission_id),
                              ctx.settings)
    except DocumentError as e:
        return fail_run(ctx, submission_id, Stage.prepare, str(e), started=started)
    return save_run(ctx, submission_id, Stage.prepare, started=started,
                    output={"pages": [p.to_dict() for p in pages], "pdf_path": str(pdf)})


def run_transcribe(ctx: StageContext, submission_id: int) -> StageRun:
    """Transcribe the first camera's audio with whisper (whisper session only)."""
    started = time.monotonic()
    cameras: dict[str, str] = ctx.repo.get_submission(submission_id)["meta"].get("cameras") or {}
    if not cameras:
        return fail_run(ctx, submission_id, Stage.transcribe, "カメラの動画が登録されていません。",
                        started=started)
    if ctx.yield_check():  # one check per inference call, before whisper is loaded
        raise YieldRequested()
    path = next(iter(cameras.values()))  # cameras are stored in order cam1..camN
    path = proxy.ready(ctx.settings, path) or path   # the same audio, copied into the proxy
    segments = transcribe(path, ctx.settings)
    return save_run(ctx, submission_id, Stage.transcribe, started=started,
                    model_id=ctx.settings.whisper_model,
                    output={"segments": [seg.to_dict() for seg in segments]})
