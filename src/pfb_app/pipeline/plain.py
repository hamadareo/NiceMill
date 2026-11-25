"""やさしい文章版, the worker's job: one approved text of one person, reworded for one reader level.

A `plain` job (a side stage: it never shows in a work's steps and never holds a work up) has
the person as `subject_id` and the reader level ("low", "mid": `plain_checks.LEVELS`) as its
`camera`. `run_plain` rewords the person's approved text with the model (text only, as the
feedback text itself is made) and saves a `plain` run for that person and level:

- `{"level", "text"}`: the reworded text, after `plain_checks.problems` passed. The model sees
  the problems of an answer that failed and writes it again (the usual retries);
- `{"level", "text": None, "reason"}`: it never passed (or the text is not approved any more).
  This is a normal, finished run, not a failure: the person's handout keeps the approved text
  and the screen says so. A failed job (a model error, a dead server) is the pipeline's own
  business (requeued or retried), as for any stage.

The run's `upstream_run_ids` is the approved feedback run it was made from, so a text that has
been edited, regenerated or approved again since makes the reworded one outdated
(`services.plain_text.state`), never silently used. The approved text itself is never touched.
In a 「あとから訂正」 work the text it is made from is the newest one unless it is reported
wrong (`approved_text`); a correction makes the reworded one outdated the same way.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from pfb_app.domain.models import Job, Stage, StageRun
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.structured import call_structured
from pfb_app.pipeline import plain_checks
from pfb_app.pipeline.common import (StageContext, assignment_of, fail_run, save_run,
                                     submission_engine)
from pfb_app.pipeline.pfb3x3_bridge import clock_is_known
from pfb_app.pipeline.prompts import Prompt, build_user_text
from pfb_app.storage.db import Repo

__all__ = ["Context", "PlainOutput", "approved_text", "context_for", "load_prompt", "run_plain"]

PROMPT_DIR = "plain"
NOT_APPROVED = "元の文が承認されていないか、承認したあとで変わりました"
NOT_USABLE_LIVE = "元の文がないか、誤りの報告を確認中です"      # 「あとから訂正」 works


class PlainOutput(BaseModel):
    text: str


@dataclass(frozen=True)
class Context:
    """What the checks need of one person's approved text."""
    run: StageRun                  # the approved feedback run
    original: str                  # its text
    names: list[str]               # the people and groups of the work (written as they are)
    internal_ids: set[str]
    camera_names: set[str]
    tracked: bool
    clock_known: bool

    def problems(self, level: str, text: str) -> list[str]:
        return plain_checks.problems(
            self.original, text, level, names=self.names, internal_ids=self.internal_ids,
            camera_names=self.camera_names, tracked=self.tracked, clock_known=self.clock_known)

    def fit(self, level: str, text: str) -> str:
        """The model's text as it is kept (trimmed, and for the youngest level with the
        kanji of later school years written in hiragana)."""
        return plain_checks.fit(plain_checks.tidy(text), level, self.kept_names())

    def kept_names(self) -> list[str]:
        """The names the original text has (the model is told to write them as they are)."""
        original = plain_checks.norm(self.original)
        return [n for n in self.names if plain_checks.has_name(plain_checks.norm(n), original)]


def load_prompt(settings, level: str) -> Prompt:
    if level not in plain_checks.LEVELS:
        raise KeyError(level)
    text = (Path(settings.prompts_dir) / PROMPT_DIR / f"{level}.md").read_text(encoding="utf-8")
    return Prompt(text=text, hash=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12])


def approved_text(repo: Repo, subject: dict) -> tuple[StageRun, str] | None:
    """The person's feedback run the version is made from, and its text: the approved one
    while approved right now; in a 「あとから訂正」 work the newest text unless it is reported
    wrong (services.corrections.usable_run). None otherwise."""
    from pfb_app.services import corrections   # noqa: PLC0415 - services import this module
    run = corrections.usable_run(repo, subject)
    text = ((run.output or {}).get("pfb") or "").strip() if run is not None else ""
    return (run, text) if run is not None and run.status == "ok" and text else None


def context_for(repo: Repo, subject: dict) -> Context | None:
    """The checks' context for the person (None: no approved text)."""
    found = approved_text(repo, subject)
    if found is None:
        return None
    run, original = found
    submission = repo.get_submission(subject["submission_id"])
    work = repo.get_assignment(submission["assignment_id"])
    names: list[str] = []
    ids: set[str] = set((run.output or {}).get("evidence_refs") or [])
    for group in repo.list_submissions(work["id"]):
        names.append(group["label"])
        for s in repo.list_subjects(group["id"]):
            if not s.get("erased_at"):
                names.append(s["display_name"])
            if group["id"] == submission["id"] and s["participant_id"]:
                ids.add(s["participant_id"])
    cameras = set(submission["meta"].get("cameras") or {}) | set(repo.list_cameras(submission["id"]))
    return Context(run=run, original=original, names=[n for n in names if n], internal_ids=ids,
                   camera_names=cameras, tracked=submission_engine(submission) == "pfb3x3",
                   clock_known=clock_is_known(submission["meta"], work.get("domain") or "school"))


def _request(ctx: StageContext, prompt: Prompt, level: str, info: Context) -> ChatRequest:
    spec = plain_checks.LEVELS[level]
    sections = [("言いかえる文章", info.original), ("読み手", spec.label)]
    kept = info.kept_names()
    if kept:
        sections.append(("そのまま書く名前", "\n".join(kept)))
    if spec.kanji is not None:
        sections.append(("使ってよい漢字", spec.kanji))
    return ChatRequest(
        system=prompt.text, user_text=build_user_text(sections), image_paths=[],
        json_schema=PlainOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens.get(Stage.plain.value, 2048))


def run_plain(ctx: StageContext, job: Job) -> StageRun:
    """Reword one person's approved text for the job's level; see the module's text."""
    started = time.monotonic()
    sid, level = job.submission_id, job.camera
    common = dict(camera=level, subject_id=job.subject_id, started=started)
    if level not in plain_checks.LEVELS or job.subject_id is None:
        return fail_run(ctx, sid, Stage.plain, "読み手の指定がありません", **common)
    subject = ctx.repo.get_subject(job.subject_id)
    info = context_for(ctx.repo, subject)
    if info is None:
        from pfb_app.services import corrections   # noqa: PLC0415
        live = corrections.mode_of(assignment_of(ctx, sid)) == corrections.CORRECT_LATER
        return save_run(ctx, sid, Stage.plain,
                        output={"level": level, "text": None,
                                "reason": NOT_USABLE_LIVE if live else NOT_APPROVED}, **common)
    prompt = load_prompt(ctx.settings, level)
    common.update(upstream=[info.run.id], prompt_hash=prompt.hash)
    result = call_structured(
        ctx.backend, ctx.repo, _request(ctx, prompt, level, info), PlainOutput,
        max_retries=ctx.settings.max_retries, yield_check=ctx.yield_check,
        validate=lambda out: info.problems(level, info.fit(level, out.text)))
    common["model_id"] = ctx.backend.model_id
    if result.parsed is None:
        reason = "\n".join(result.errors[-3:]) or "作れませんでした"
        return save_run(ctx, sid, Stage.plain,
                        output={"level": level, "text": None, "reason": reason}, **common)
    return save_run(ctx, sid, Stage.plain,
                    output={"level": level, "text": info.fit(level, result.parsed.text)},
                    **common)
