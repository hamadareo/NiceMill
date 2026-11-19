"""判定支援 (line calls) for the pages: start the engine's linecall for one camera, where its
episodes are, the job's state and the coach's labels.

Labels live in the DB (linecall_labels; one row per episode, the last press wins). Every press
also rewrites the engine's linecall/labels.json atomically from all the camera's rows, in the
engine's format: {"<episode id>": {"label": "correct" | "wrong", "verdict": <the episode's
verdict at that press>}}. A 判定不能 press ("undecidable") stays in the DB only: the engine has no
use for it (pfb3x3.linecall.verdict.label_precision). The rewrite runs inside the DB write
transaction of the press, so two presses at once never leave an older file behind.
"""
import json
import logging
from pathlib import Path
from typing import Literal

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.engine.pfb3x3 import _write_atomic
from pfb_app.pipeline.linecall import linecall_dir
from pfb_app.pipeline.runner import enqueue_once
from pfb_app.storage.db import LINECALL_LABELS, Repo

log = logging.getLogger(__name__)

MIRRORED = ("correct", "wrong")   # the labels the engine counts
_ACTIVE = (JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value)
_FAILED = (JobStatus.failed.value, JobStatus.failed_retryable.value)


def _cameras(repo: Repo, submission_id: int) -> dict:
    return repo.get_submission(submission_id)["meta"].get("cameras") or {}


def request(repo: Repo, submission_id: int, camera: str) -> int | None:
    """「判定材料を作る」: queue linecall for this camera (None when one is already queued or
    running, or the work is to be deleted). ValueError for a camera the group does not have."""
    if camera not in _cameras(repo, submission_id):
        raise ValueError(f"no camera {camera!r} in submission {submission_id}")
    return enqueue_once(repo, submission_id, Stage.linecall, camera)


def blocked(repo: Repo, submission_id: int) -> Literal["deleting", "media_removed",
                                                        "erasing"] | None:
    """Why nothing can be queued for this group (pipeline.runner.enqueue_once's checks), in
    the order the page says it: the work is to be deleted, its videos were removed
    (services.retention), a person's data is being erased (services.erasure); None when a job
    can be queued."""
    from pfb_app.pipeline import runner   # noqa: PLC0415
    from pfb_app.services import deletion   # noqa: PLC0415 - deletion imports the runner
    sub = repo.get_submission(submission_id)
    if deletion.is_requested(sub["meta"]) or deletion.is_pending(repo, sub["assignment_id"]):
        return "deleting"
    if runner._media_removed(repo, submission_id):
        return "media_removed"
    if runner._erasing(repo, submission_id):
        return "erasing"
    return None


def episodes_path(settings: Settings, repo: Repo, submission_id: int, camera: str) -> Path:
    """The camera's episodes.json (it may not exist yet)."""
    aid = repo.get_submission(submission_id)["assignment_id"]
    return linecall_dir(settings, aid, submission_id, camera) / "episodes.json"


def job_state(repo: Repo, submission_id: int, camera: str
              ) -> Literal["none", "active", "done", "failed"]:
    """The camera's latest linecall job: queued / running / waiting for the GPU -> "active";
    without any job, "done" when a usable linecall run exists (else "none")."""
    row = repo.latest_job_row(submission_id, Stage.linecall, camera)
    if row is None:
        run = repo.latest_run(submission_id, Stage.linecall, camera)
        return "done" if run is not None and run.status == "ok" else "none"
    if row["status"] in _ACTIVE:
        return "active"
    return "failed" if row["status"] in _FAILED else "done"


def _verdicts(path: Path) -> dict[str, str]:
    """episode id -> verdict from episodes.json; {} when it is missing or unreadable."""
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    eps = doc.get("episodes") if isinstance(doc, dict) else None
    return {str(e["id"]): str(e.get("verdict")) for e in eps or []
            if isinstance(e, dict) and e.get("id") is not None}


def mirror_records(records: dict[str, dict]) -> dict[str, dict]:
    """The labels.json document for the camera's label rows: correct / wrong only, each with
    the verdict it was given for."""
    return {eid: {"label": rec["label"], "verdict": rec["verdict"]}
            for eid, rec in sorted(records.items())
            if rec["label"] in MIRRORED and rec.get("verdict")}


def save_label(settings: Settings, repo: Repo, submission_id: int, camera: str,
               episode_id: str, label: str, shown_verdict: str) -> None:
    """One press of 「正しい」「違う」「判定不能」 on an episode shown with `shown_verdict`: the
    DB row (with that verdict), then labels.json rewritten from all the camera's rows.
    ValueError (nothing stored) for an unknown label, an episode that is not in the camera's
    episodes.json, or one whose verdict there is no longer `shown_verdict` (a rerun changed
    it: the page reruns and shows the new one). The verdict is read inside the press's write
    transaction. A failed file write is logged and leaves the DB row (the next press writes
    the file again)."""
    if label not in LINECALL_LABELS:
        raise ValueError(f"unknown line-call label: {label!r}")
    path = episodes_path(settings, repo, submission_id, camera)
    with repo._tx():
        verdict = _verdicts(path).get(episode_id)
        if verdict is None:
            raise ValueError(f"no episode {episode_id!r} for submission {submission_id} {camera}")
        if verdict != shown_verdict:
            raise ValueError(f"episode {episode_id!r} is now {verdict!r}, not {shown_verdict!r}")
        repo.set_linecall_label(submission_id, camera, episode_id, label, verdict=verdict)
        doc = mirror_records(repo.linecall_label_records(submission_id, camera))
        try:
            _write_atomic(path.with_name("labels.json"),
                          json.dumps(doc, ensure_ascii=False, indent=1))
        except OSError:
            log.exception("labels.json を書けませんでした submission=%s camera=%s",
                          submission_id, camera)
