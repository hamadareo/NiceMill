"""次にためしてみること: practice tips written after a person's feedback, and for the group.

A video work's feedback (stage pfb) is followed by one more model call that turns what the
feedback already stands on into one to three small suggestions for that person, and, once
everybody's feedback is written, one call that turns what several people's texts share into
one to three suggestions for the group. Both are stored as `Stage.tips` runs (no new table):

- a person's run: subject_id set; output
  {"tips_status": "generated" | "none", "items": [{"text", "evidence_refs"}], "pfb_run_id",
  "reason"?}. Its upstream is the feedback's upstream (the contribution runs), so it goes
  stale when the analysis is redone, not when the coach words the text again;
- the group's run: subject_id none, camera none; the same output with empty evidence_refs;
  its upstream is the group's latest contribution runs and the feedback runs it was made from
  (so it goes stale when one of those texts is written again). The group's run is made by a
  job of its own (stage tips, a side stage), queued after the last feedback job.

Material: the feedback text and the events it cites (their action, speech and why it counted:
never their time), for the group only the people's feedback texts, without names. A tip is
checked (pipeline.checks): one to three, a suggestion's ending, no digits, time, grades, names
or ids, every cited event is the person's own, and the words stand on the events (a katakana
word must be written there). A failed or refused answer never stops the feedback: it is
stored as "none" (an empty list, with the reason), so an older run's tips never outlive the
feedback they were made for. Hand-written feedback has no cited events: no tips are made.
The coach's confirmation (services.tips) is a newer run with source "teacher".
"""
import logging
import time

from pfb_app.domain.models import (BehaviorEvent, Kind, Stage, StageRun, TeamTipsOutput,
                                   TipsOutput)
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.structured import call_structured
from pfb_app.pipeline import checks
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import StageContext, assignment_of, save_run, submission_engine, usable
from pfb_app.pipeline.prompts import build_user_text, load_prompt, load_prompt_named

log = logging.getLogger(__name__)

GENERATED = "generated"
NONE = "none"
_OOM_TEXT = "CUDA out of memory"
_ACTIVE_JOB = ("queued", "running", "waiting_gpu")
TEAM_PROMPT = "team_tips"
MIN_TEAM_PEOPLE = checks.MIN_TEAM_SUPPORT


def _tokens(ctx: StageContext) -> int:
    return (ctx.settings.max_tokens.get(Stage.tips.value)
            or ctx.settings.max_tokens[Stage.pfb.value])


def tip_context(repo, submission_id: int, extra_ids=()) -> checks.TipContext:
    """What a tip's wording is checked against, for the group: the people's names, the
    participant ids (and `extra_ids`, the events' ids), the camera names, tracker labels."""
    subjects = repo.list_subjects(submission_id)
    ids = {s["participant_id"] for s in subjects if s["participant_id"]} | set(extra_ids)
    return checks.TipContext(
        names=frozenset(s["display_name"].strip() for s in subjects if s["display_name"]),
        internal_ids=frozenset(ids), cameras=frozenset(repo.list_cameras(submission_id)),
        tracked=submission_engine(repo.get_submission(submission_id)) == "pfb3x3")


def _none(ctx: StageContext, submission_id: int, subject_id: int | None, upstream, reason: str,
          started: float, pfb_run_id: int | None = None, **common) -> StageRun:
    out = {"tips_status": NONE, "items": [], "reason": reason}
    if pfb_run_id is not None:
        out["pfb_run_id"] = pfb_run_id
    return save_run(ctx, submission_id, Stage.tips, subject_id=subject_id, upstream=upstream,
                    output=out, started=started, **common)


def _cited(ctx: StageContext, pfb: StageRun) -> dict[str, tuple[BehaviorEvent, str]]:
    """{ref_id: (the event, why it counted)} of the events the feedback cites: the positive
    judgments of the contribution runs it stands on."""
    from pfb_app.pipeline.video import _events_of   # noqa: PLC0415 - video imports a lot
    cited = list((pfb.output or {}).get("evidence_refs") or [])
    found: dict[str, tuple[BehaviorEvent, str]] = {}
    for run_id in pfb.upstream_run_ids:
        contribution = ctx.repo.get_run(run_id)
        if contribution is None or contribution.stage != Stage.contribution \
                or not usable(contribution):
            continue
        behavior = (ctx.repo.get_run(contribution.upstream_run_ids[0])
                    if contribution.upstream_run_ids else None)
        if not usable(behavior):
            continue
        by_id = {e.event_id: e for e in _events_of(behavior)}
        for j in contribution.output["judgments"]:
            if j["ref_id"] in cited and j["is_positive_contribution"] and j["ref_id"] in by_id:
                found[j["ref_id"]] = (by_id[j["ref_id"]], j["contribution_interpretation"])
    return found


def _event_text(e: BehaviorEvent, why: str) -> str:
    return f"行動: {e.observed_behavior}\n発話: {e.utterance or 'なし'}\n貢献の解釈: {why}"


def run_person_tips(ctx: StageContext, submission_id: int, subject_id: int,
                    pfb: StageRun | None) -> StageRun:
    """The tips for one person from their latest feedback run `pfb`, saved as a tips run."""
    started = time.monotonic()
    upstream = list(pfb.upstream_run_ids) if pfb is not None else []
    none = lambda reason, **kw: _none(ctx, submission_id, subject_id, upstream, reason,
                                      started, pfb.id if pfb is not None else None, **kw)
    if pfb is None or not usable(pfb) or (pfb.output or {}).get("pfb_status") != "generated":
        return none("feedback not generated")
    if pfb.source != "model":
        return none("feedback written by a person: no cited events")
    cited = _cited(ctx, pfb)
    refs = [r for r in dict.fromkeys(pfb.output.get("evidence_refs") or []) if r in cited]
    if not refs:
        return none("no cited events")
    assignment = assignment_of(ctx, submission_id)
    domain = assignment.get("domain") or "school"
    from pfb_app.domain import profiles   # noqa: PLC0415
    from pfb_app.pipeline.video import _work_sections   # noqa: PLC0415
    profile = profiles.of_work(assignment)
    prompt = load_prompt(ctx.settings, Kind.video, Stage.tips, domain, profile)
    common = dict(prompt_hash=prompt.hash, model_id=ctx.backend.model_id)
    body = "\n\n".join(f"ref_id: {r}\n{_event_text(*cited[r])}" for r in refs)
    req = ChatRequest(
        system=prompt.text,
        user_text=build_user_text([*_work_sections(assignment),
                                   ("この人のフィードバック", pfb.output["pfb"]),
                                   ("根拠にした行動イベント", body)]),
        image_paths=[], json_schema=TipsOutput.model_json_schema(),
        max_tokens=_tokens(ctx))
    tip_ctx = tip_context(ctx.repo, submission_id, extra_ids=cited)
    sources = {r: _event_text(*cited[r]) for r in refs}

    def validate(out: TipsOutput) -> list[str]:
        return checks.person_tips_errors(out, set(refs), sources, pfb.output["pfb"], tip_ctx)

    result = call_structured(ctx.backend, ctx.repo, req, TipsOutput,
                             max_retries=ctx.settings.max_retries,
                             yield_check=ctx.yield_check, validate=validate)
    if result.parsed is None:
        return none("; ".join(result.errors[-6:]) or "no answer", **common)
    if result.parsed.tips_status != "generated":
        return none("insufficient evidence", **common)
    items = [{"text": t.text.strip(), "evidence_refs": list(dict.fromkeys(t.evidence_refs))}
             for t in result.parsed.tips]
    return save_run(ctx, submission_id, Stage.tips, subject_id=subject_id, upstream=upstream,
                    output={"tips_status": GENERATED, "items": items, "pfb_run_id": pfb.id},
                    started=started, **common)


def after_pfb(ctx: StageContext, pfb: StageRun) -> StageRun | None:
    """Right after a feedback run of a video work: its tips. Never breaks the feedback job:
    a failure is stored as no tips (the exceptions that end a session are let through)."""
    from pfb_app.inference.structured import YieldRequested   # noqa: PLC0415
    if pfb.subject_id is None:
        return None
    try:
        return run_person_tips(ctx, pfb.submission_id, pfb.subject_id, pfb)
    except YieldRequested:
        raise
    except Exception as e:   # noqa: BLE001 - the tips are optional
        if _OOM_TEXT in str(e):
            raise
        log.exception("次にためしてみることを作れませんでした subject=%s", pfb.subject_id)
        return _none(ctx, pfb.submission_id, pfb.subject_id, list(pfb.upstream_run_ids),
                     f"{type(e).__name__}: {e}", time.monotonic(), pfb.id)


# --- the group ---------------------------------------------------------------------------------

def team_people(repo, submission_id: int) -> list[tuple[dict, StageRun]]:
    """The evaluated people whose latest feedback is generated and not outdated, with it."""
    from pfb_app.services import review   # noqa: PLC0415 - services import the pipeline
    out = []
    for subject in repo.list_subjects(submission_id):
        if not is_evaluated(subject):
            continue
        run = repo.latest_run(submission_id, Stage.pfb, subject_id=subject["id"])
        if (usable(run) and (run.output or {}).get("pfb_status") == "generated"
                and not review.is_stale_chain(repo, run)):
            out.append((subject, run))
    return out


def team_upstream(repo, submission_id: int) -> list[int]:
    """What the group's tips stand on: the group's latest contribution runs and every
    person's feedback run they were made from, so a text written again (by the model or by
    the coach) makes them outdated."""
    runs = [repo.latest_run(submission_id, Stage.contribution, cam)
            for cam in repo.list_cameras(submission_id)]
    return [r.id for r in runs if r is not None] + [run.id for _, run in
                                                    team_people(repo, submission_id)]


def run_team_tips(ctx: StageContext, submission_id: int) -> StageRun | None:
    """The group's tips from its people's feedback texts, saved as a tips run (None: fewer
    than two people have a text, nothing is saved)."""
    started = time.monotonic()
    people = team_people(ctx.repo, submission_id)
    if len(people) < MIN_TEAM_PEOPLE:
        return None
    upstream = team_upstream(ctx.repo, submission_id)
    none = lambda reason, **kw: _none(ctx, submission_id, None, upstream, reason, started, **kw)
    assignment = assignment_of(ctx, submission_id)
    domain = assignment.get("domain") or "school"
    from pfb_app.pipeline.video import _work_sections   # noqa: PLC0415
    prompt = load_prompt_named(ctx.settings, Kind.video, TEAM_PROMPT, domain)
    common = dict(prompt_hash=prompt.hash, model_id=ctx.backend.model_id)
    texts = {n: run.output["pfb"] for n, (_, run) in enumerate(people, 1)}
    body = "\n\n".join(f"人{n}:\n{text}" for n, text in texts.items())
    from pfb_app.domain.vocab import words   # noqa: PLC0415
    label = f"{words(domain).people}ごとのフィードバック"
    req = ChatRequest(
        system=prompt.text,
        user_text=build_user_text([*_work_sections(assignment), (label, body)]),
        image_paths=[], json_schema=TeamTipsOutput.model_json_schema(),
        max_tokens=_tokens(ctx))
    tip_ctx = tip_context(ctx.repo, submission_id)

    def validate(out: TeamTipsOutput) -> list[str]:
        checks.trim_team_tips(out, texts)
        return checks.team_tips_errors(out, texts, tip_ctx)

    result = call_structured(ctx.backend, ctx.repo, req, TeamTipsOutput,
                             max_retries=ctx.settings.max_retries,
                             yield_check=ctx.yield_check, validate=validate)
    if result.parsed is None:
        return none("; ".join(result.errors[-6:]) or "no answer", **common)
    if result.parsed.tips_status != "generated":
        return none("insufficient evidence", **common)
    items = [{"text": t.text.strip(), "evidence_refs": []} for t in result.parsed.tips]
    return save_run(ctx, submission_id, Stage.tips, upstream=upstream,
                    output={"tips_status": GENERATED, "items": items},
                    started=started, **common)


def team_wanted(repo, submission_id: int, *, retry_none: bool = True) -> bool:
    """True when the group's tips should be made now: nobody's feedback is still being
    written, two people have a text, and the group has no tips yet or only outdated ones (or,
    with `retry_none`, a failed attempt). Tips the coach kept, confirmed or deleted are never
    replaced by new ones."""
    from pfb_app.services import review   # noqa: PLC0415 - services import the pipeline
    if any(j["stage"] == Stage.pfb.value and j["status"] in _ACTIVE_JOB
           for j in repo.submission_job_rows(submission_id)):
        return False
    if len(team_people(repo, submission_id)) < MIN_TEAM_PEOPLE:
        return False
    latest = repo.latest_run(submission_id, Stage.tips)
    if latest is None or review.is_stale_chain(repo, latest):
        return True
    if latest.source == "teacher":
        return False
    return retry_none and not (latest.output or {}).get("items")


def queue_team_tips(repo, submission_id: int) -> int | None:
    """Queue the group's tips as a job of their own (a side job: the work's analysis never
    waits for it); None: not wanted now, or one is already queued."""
    from pfb_app.pipeline.runner import enqueue_once   # noqa: PLC0415 - runner imports this
    if not team_wanted(repo, submission_id):
        return None
    return enqueue_once(repo, submission_id, Stage.tips)


def team_after_pfb(ctx: StageContext, submission_id: int) -> int | None:
    """After a feedback job is done: the group's tips are queued, once nobody's feedback is
    still being written (the job id, None when not wanted). The tips are made by the job
    (`run_team_job`), so a yield for the GPU or a full GPU puts them back in the queue."""
    return queue_team_tips(ctx.repo, submission_id)


def run_team_job(ctx: StageContext, job) -> list[StageRun]:
    """The group's tips job: the tips are made when they are still wanted (the coach may
    have kept some meanwhile, or a text may be written again). A yield or a full GPU goes up
    to the runner (job back in the queue, session ended); any other failure is stored as no
    tips, with the reason."""
    from pfb_app.inference.structured import YieldRequested   # noqa: PLC0415
    sid = job.submission_id
    if not team_wanted(ctx.repo, sid):
        return []
    try:
        run = run_team_tips(ctx, sid)
    except YieldRequested:
        raise
    except Exception as e:   # noqa: BLE001 - the group's tips are optional
        if _OOM_TEXT in str(e):
            raise
        log.exception("チームのヒントを作れませんでした submission=%s", sid)
        run = _none(ctx, sid, None, team_upstream(ctx.repo, sid),
                    f"{type(e).__name__}: {e}", time.monotonic())
    return [run] if run is not None else []
