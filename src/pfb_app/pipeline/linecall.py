"""判定支援: the engine's `linecall` for one tracked camera (session key runner.LINECALL_KEY).

Like `track`: one engine subprocess on the GPU (no vLLM server, no CPU fallback). Its outputs
stay in the engine's <tracks>/<video id>/linecall/ (episodes.json, crops/, clips/); the app
writes only labels.json there (services.linecall). Nothing follows the job, and it is not part
of the work's analysis (storage.db.NON_BLOCKING_STAGES / SIDE_STAGES).

- EngineNeedsGpu -> GpuUnavailable (the session marks the key's jobs waiting_gpu);
- EngineYield (the GPU was wanted meanwhile) -> YieldRequested (the job is requeued);
- EngineError (also a second run of the same video) -> failed_retryable with its tail.
- No pose weights with the pinned sha256 (Pfb3x3Engine.pose_ready) -> failed with NOT_READY,
  and the engine is not called: it would download them, which production never does
  (scripts/provision_pose_weights.sh puts them in place).
"""
import time
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.domain.models import Job, Stage, StageRun
from pfb_app.engine import Pfb3x3Engine
from pfb_app.pipeline.common import StageContext, fail_run, save_run, usable
from pfb_app.pipeline.tracking import _engine, _engine_signals, _model_id, _on_tick, video_id
from pfb_app.storage import files

NOT_TRACKED = "判定材料を作るには、先にこのカメラで人を見つける処理を終えてください（{cam}）。"
NOT_READY = "判定材料を作る準備ができていません。管理者に連絡してください。"


def linecall_dir(settings: Settings, assignment_id: int, submission_id: int,
                 camera: str) -> Path:
    """<sub_dir>/pfb3x3/tracks/<video id>/linecall: where the engine writes the camera's
    episodes (not created here)."""
    sub_dir = files.submission_dir(settings, assignment_id, submission_id, create=False)
    return (Pfb3x3Engine.root(sub_dir) / "tracks" / video_id(submission_id, camera)
            / "linecall")


def run_linecall(ctx: StageContext, job: Job) -> StageRun:
    """Line-call evidence (判定材料) of job.camera. Output: {"episodes_json", "episodes",
    "verdicts"} (verdict -> count). Needs a usable track run of that camera (its upstream)."""
    started = time.monotonic()
    repo, sid, cam = ctx.repo, job.submission_id, job.camera
    track = repo.latest_run(sid, Stage.track, cam) if cam else None
    if not usable(track):
        return fail_run(ctx, sid, Stage.linecall, NOT_TRACKED.format(cam=cam or "-"),
                        camera=cam, started=started)
    engine = _engine(ctx)
    if not engine.pose_ready():
        return fail_run(ctx, sid, Stage.linecall, NOT_READY, camera=cam, started=started)
    sub_dir = files.submission_dir(ctx.settings, repo.get_submission(sid)["assignment_id"], sid)
    with _engine_signals():
        res = engine.linecall(sub_dir, video_id(sid, cam), on_tick=_on_tick(ctx, job))
    return save_run(ctx, sid, Stage.linecall, camera=cam, upstream=[track.id],
                    model_id=_model_id(engine, "linecall"), started=started,
                    output={"episodes_json": str(res.episodes_json), "episodes": res.episodes,
                            "verdicts": res.verdicts})
