"""Helpers shared by the document and video pipelines."""
import time
from dataclasses import dataclass
from typing import Any, Callable, Literal

from pfb_app.config import Settings
from pfb_app.domain.models import Stage, StageRun
from pfb_app.inference.backend import InferenceBackend
from pfb_app.storage.db import Repo


@dataclass
class StageContext:
    settings: Settings
    repo: Repo
    backend: InferenceBackend
    yield_check: Callable[[], bool]
    # Tracking engine (PFB3x3) sessions only; the paper engine never reads these.
    engine: Any = None  # pfb_app.engine.Pfb3x3Engine (None: built from settings when needed)
    server: Any = None  # pfb_app.engine.ServerHandle of the running vLLM server (pfb3x3:<key>)
    device: str = "cuda"  # the track stage's device: "cuda", or "cpu" for the CPU fallback
    # Translates the engine's English observations (InferenceBackend); None: the session's
    # served model (engine.translator(server)). Tests inject a fake.
    translator: Any = None
    # detect sessions: stop the session's server now (ctx.server becomes None) when the GPU is
    # needed while a check runs on the CPU. None elsewhere.
    release_server: Callable[[], None] | None = None


# Raw model responses are not copied into run output: the call_log cache already holds them.

def save_run(ctx: StageContext, submission_id: int, stage: Stage, *, camera=None, subject_id=None,
             upstream=(), prompt_hash=None, model_id=None, output=None, error=None,
             warnings=(), started: float | None = None) -> StageRun:
    run = StageRun(
        id=None, submission_id=submission_id, stage=stage, camera=camera, subject_id=subject_id,
        version=0, source="model", upstream_run_ids=list(upstream), prompt_hash=prompt_hash,
        model_id=model_id, output=output, status="failed" if error else "ok", error=error,
        warnings=list(warnings),
        duration_sec=(time.monotonic() - started) if started is not None else 0.0,
        created_at="")
    return ctx.repo.save_stage_run(run)


def fail_run(ctx, submission_id, stage, message, **kw) -> StageRun:
    return save_run(ctx, submission_id, stage, error=message, **kw)


def assignment_of(ctx: StageContext, submission_id: int) -> dict:
    sub = ctx.repo.get_submission(submission_id)
    return ctx.repo.get_assignment(sub["assignment_id"])


def usable(run: StageRun | None) -> bool:
    return run is not None and run.status == "ok" and run.output is not None


def submission_engine(submission: dict) -> Literal["paper", "pfb3x3"]:
    """The video engine chosen for a submission; anything but "pfb3x3" is the paper engine."""
    return "pfb3x3" if submission["meta"].get("engine", "paper") == "pfb3x3" else "paper"
