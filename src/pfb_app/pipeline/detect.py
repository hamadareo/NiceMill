"""The `detect` stage: the content check of a dropped video, before the coach confirms.

One job per submission (camera None). It runs in a `detect` session (with the engine's vLLM
server for the scene verdict) or a `detect:cpu` session (no server: the GPU was busy):

1. `probe` per camera: the detector on the CPU counts people and vehicles in a few sampled
   frames of the meta's start/end, plus the camera motion;
2. `scene` on camera 1 through the session's server, when there is one: group work / sport /
   other, which sport, and whether numbers are visible; with an engine that has it, also the
   kind of a non-sport activity (meeting, presentation, ...), which picks the work's profile.

Output: {"cameras": {camera: probe dict}, "scene": scene dict | None, "scene_skipped": None |
"gpu_busy" | "failed", "duration_s": float}. Without a scene verdict the card asks the coach to
pick the scene. A failed probe fails the run with a plain message (the card then asks for
everything); engine error texts only go to the log. Every engine call is limited to
detect_timeout_sec: a probe over it fails the run, a scene over it gives no verdict.

The probe runs on the CPU and is never stopped for the GPU, but when the GPU is needed while
it runs (ctx.yield_check), the session's idle server is stopped (ctx.release_server) and the
check goes on without a scene verdict.
"""
import logging
import time
from dataclasses import asdict
from pathlib import Path

from pfb_app.domain.models import Job, Stage, StageRun
from pfb_app.engine import (EngineError, EngineNeedsGpu, EngineTimeout, EngineYield,
                            Pfb3x3Engine, ProbeResult, SceneResult)
from pfb_app.pipeline.common import StageContext, fail_run, save_run
from pfb_app.pipeline.tracking import video_id
from pfb_app.storage import files

log = logging.getLogger(__name__)

SCENE_CAMERA = "cam1"
GPU_BUSY = "gpu_busy"
FAILED = "failed"
NO_CAMERAS = "カメラの動画が登録されていません。"
SCENE_FAILED_WARNING = "どんな場面の動画かを自動で判定できませんでした。確認の画面で選んでください。"
PROBE_FAILED = "内容を確かめられませんでした。どんな場面の動画か、下から選んでください。"


def run_detect(ctx: StageContext, job: Job) -> StageRun:
    started = time.monotonic()
    repo, sid = ctx.repo, job.submission_id
    submission = repo.get_submission(sid)
    meta = submission["meta"]
    cameras: dict[str, str] = meta.get("cameras") or {}
    if not cameras:
        return fail_run(ctx, sid, Stage.detect, NO_CAMERAS, started=started)
    engine = ctx.engine if ctx.engine is not None else Pfb3x3Engine(ctx.settings)
    sub_dir = files.submission_dir(ctx.settings, submission["assignment_id"], sid)
    start_s = float(meta.get("start_sec") or 0.0)
    end_s = meta.get("end_sec")
    timeout = ctx.settings.detect_timeout_sec
    probes: dict[str, dict] = {}
    for cam, path in cameras.items():
        try:
            res = engine.probe(sub_dir, video_id(sid, cam), Path(path), start_s=start_s,
                               end_s=None if end_s is None else float(end_s),
                               samples=ctx.settings.detect_samples, device="cpu",
                               on_tick=_probe_tick(ctx, job), timeout_sec=timeout)
            probes[cam] = _probe_dict(res)
        except (EngineError, EngineNeedsGpu, EngineYield, KeyError, TypeError, ValueError):
            log.exception("動画の内容確認（probe）に失敗しました submission=%s camera=%s", sid, cam)
            return fail_run(ctx, sid, Stage.detect, PROBE_FAILED, started=started)
    scene, skipped, warnings = None, None, []
    if ctx.server is None:
        skipped = GPU_BUSY   # detector only: the coach picks the scene
    else:
        cam = SCENE_CAMERA if SCENE_CAMERA in cameras else next(iter(cameras))
        try:
            scene = _scene_dict(engine.scene(sub_dir, video_id(sid, cam),
                                             base_url=ctx.server.base_url,
                                             model=ctx.server.model,
                                             on_tick=_scene_tick(ctx, job),
                                             timeout_sec=timeout))
        except (EngineYield, EngineNeedsGpu):
            log.info("GPUを譲るため、動画の種類の判定を省きました submission=%s", sid)
            skipped = GPU_BUSY
        except EngineTimeout:   # before EngineError: no verdict, the coach picks (as gpu_busy)
            log.warning("動画の種類の判定が%s秒で終わらないため省きました submission=%s",
                        timeout, sid)
            skipped = GPU_BUSY
        except (EngineError, KeyError, TypeError, ValueError):
            log.exception("動画の種類の判定（scene）に失敗しました submission=%s", sid)
            skipped = FAILED
            warnings.append(SCENE_FAILED_WARNING)
    output = {"cameras": probes, "scene": scene, "scene_skipped": skipped,
              "duration_s": max(p["duration_s"] for p in probes.values())}
    model = (scene["model"] or "scene") if scene is not None else "probe"
    return save_run(ctx, sid, Stage.detect, output=output, warnings=warnings,
                    model_id=_model_id(engine, model), started=started)


def _probe_dict(res: ProbeResult) -> dict:
    out = asdict(res)
    out["frames"] = [str(f) for f in res.frames]
    return out


def _scene_dict(res: SceneResult) -> dict:
    """The verdict as stored; the activity keys only when the engine gave an activity (an
    older engine's verdict is stored as before)."""
    out = asdict(res)
    if out.get("activity") is None and out.get("activity_text") is None:
        out.pop("activity", None)
        out.pop("activity_text", None)
    return out


def _model_id(engine, name: str) -> str:
    commit = getattr(engine, "engine_commit", None)
    return f"pfb3x3:{name}" + (f"@{commit[:12]}" if commit else "")


def _beat(ctx: StageContext, job: Job) -> None:
    try:
        ctx.repo.heartbeat(job.id)
    except Exception:  # e.g. database locked: the session's heartbeat thread also beats
        log.exception("ハートビートの更新に失敗しました job=%s", job.id)


def _probe_tick(ctx: StageContext, job: Job):
    """Heartbeat; when the GPU is needed, stop the session's idle server. Never aborts the
    probe (it runs on the CPU)."""
    def tick() -> bool:
        _beat(ctx, job)
        if ctx.server is not None and ctx.release_server is not None and ctx.yield_check():
            ctx.release_server()
        return False
    return tick


def _scene_tick(ctx: StageContext, job: Job):
    """Heartbeat, then whether to give the GPU back (stops the scene call)."""
    def tick() -> bool:
        _beat(ctx, job)
        return ctx.yield_check()
    return tick
