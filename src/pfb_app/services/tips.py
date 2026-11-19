"""次にためしてみること: what the coach sees, changes and confirms, and what the PDF takes.

The model's tips are stored by the worker (pipeline.tips) as `Stage.tips` runs: a person's
(subject_id set) and the group's (none). Reading them here:

- `view` is the latest run's tips, shown only while that run is not outdated (its upstream, the
  contribution runs, was not redone) and made something;
- `confirm` is the coach's press (also what 「承認する」 does for the person's tips on screen):
  the lines on screen are saved as a newer run with source "teacher" and output
  "confirmed": true; an empty list is a deletion and is saved too (the tips are then in no
  PDF, and the worker never replaces them with new ones). The press is refused when a newer
  run than the one shown exists (the coach read an older text), and, for a person, while their
  feedback is being written again;
- `confirm_on_approval` runs with every approval of a person's feedback (`review.approve`, the
  press of 「全員まとめて承認」, the worker's automatic approval): the model's own tips are
  confirmed with it when they have no wording fault (`wording`), were made for the text that
  is approved (not an older one, nor one the coach wrote by hand afterwards) and do not follow
  a deletion by the coach (「ヒントを渡さない」) of this analysis, so they reach the PDF unless
  the coach deletes them. Other tips stay unconfirmed for the coach's own look;
- the group's tips stand on the contribution runs and on each person's feedback run they were
  made from (`pipeline.tips.team_upstream`): a text written again makes them outdated;
  `queue_team` asks the worker for them (a job of its own), `team_queued` / `can_ask_team`
  tell the screen;
- `for_pdf` / `team_for_pdf` are the confirmed, current tips that go into the PDFs.

Only an approved person's PDF page is made, so unconfirmed tips are never printed; the PDF
reads these functions and nothing else.
"""
import logging
from dataclasses import dataclass, field

from pfb_app.domain.models import Stage, StageRun
from pfb_app.pipeline import checks
from pfb_app.pipeline.common import usable
from pfb_app.pipeline import tips as worker
from pfb_app.pipeline.tips import tip_context
from pfb_app.services import review
from pfb_app.services.review import BusyError
from pfb_app.storage.db import Repo

__all__ = ["AUDIT", "MAX_PERSON", "MAX_TEAM", "NEWER", "Tips", "TipsError", "can_ask_team",
           "confirm", "confirm_on_approval", "for_pdf", "queue_team", "team_for_pdf",
           "team_queued", "team_title", "view", "wording"]

log = logging.getLogger(__name__)

AUDIT = "tips_confirm"
MAX_PERSON = checks.MAX_PERSON_TIPS
MAX_TEAM = checks.MAX_TEAM_TIPS
NEWER = "ヒントが新しくなりました。確かめてから押してください。"
TOO_MANY = "ヒントは{n}個までです。"
NO_TEXT = "この人の文章がまだありません。"
SPORTS_TEAM = "チームでためしてみること"
GENERAL_TEAM = "みんなでためしてみること"
PERSON_TITLE = "次にためしてみること"


class TipsError(Exception):
    """A refusal of the coach's press, in plain words."""


def team_title(domain: str | None) -> str:
    return SPORTS_TEAM if domain == "sports" else GENERAL_TEAM


@dataclass(frozen=True)
class Tips:
    run: StageRun | None = None          # the latest tips run (any state), None: none made
    items: list[str] = field(default_factory=list)   # the tips to show (none when outdated)
    confirmed: bool = False              # the coach's: they go into the PDF
    stale: bool = False                  # the analysis was redone after they were made
    text_changed: bool = False           # the model's tips were made for an older feedback text

    @property
    def run_id(self) -> int | None:
        return self.run.id if self.run else None


def _latest(repo: Repo, submission_id: int, subject_id: int | None) -> StageRun | None:
    return repo.latest_run(submission_id, Stage.tips, subject_id=subject_id)


def view(repo: Repo, submission_id: int, subject_id: int | None = None) -> Tips:
    """The group's tips (`subject_id` None) or one person's."""
    run = _latest(repo, submission_id, subject_id)
    if run is None or not usable(run):
        return Tips(run)
    if review.is_stale_chain(repo, run):
        return Tips(run, stale=True)
    items = [i["text"] for i in (run.output or {}).get("items") or [] if i.get("text")]
    confirmed = run.source == "teacher" and bool((run.output or {}).get("confirmed"))
    return Tips(run, items, confirmed, text_changed=_text_changed(repo, run, subject_id))


def _text_changed(repo: Repo, run: StageRun, subject_id: int | None) -> bool:
    """A person's model tips whose feedback text is not the person's latest (written again
    by the model or by the coach since): an approval does not confirm them."""
    if subject_id is None or run.source != "model":
        return False
    pfb = repo.latest_run(run.submission_id, Stage.pfb, subject_id=subject_id)
    return pfb is not None and (run.output or {}).get("pfb_run_id") != pfb.id


def wording(repo: Repo, submission_id: int, lines: list[str]) -> list[str]:
    """The plain faults of a person's or the group's tip lines (one line each), for the
    coach to read; they advise, they never refuse."""
    ctx = tip_context(repo, submission_id)
    out: list[str] = []
    for line in lines:
        out += [p for p in checks.tip_wording_problems(line, ctx) if p not in out]
    return out


def _upstream(repo: Repo, submission_id: int, subject_id: int | None,
              latest: StageRun | None = None) -> list[int]:
    """What new tips stand on: a person's latest feedback's upstream; the group's: what its
    tips on screen stand on, else the latest contribution and feedback runs."""
    if subject_id is not None:
        pfb = repo.latest_run(submission_id, Stage.pfb, subject_id=subject_id)
        if pfb is None:
            raise TipsError(NO_TEXT)
        return list(pfb.upstream_run_ids)
    if latest is not None and not review.is_stale_chain(repo, latest):
        return list(latest.upstream_run_ids)    # what the tips on screen stand on
    return worker.team_upstream(repo, submission_id)


def _save(repo: Repo, submission_id: int, subject_id: int | None, lines: list[str],
          latest: StageRun | None, *, audit_from: int | None, confirmed: bool = True) -> StageRun:
    kept = {i["text"]: i for i in ((latest.output or {}).get("items") or [])} if latest else {}
    items = [{"text": line, "evidence_refs": list(kept.get(line, {}).get("evidence_refs", []))}
             for line in lines]
    out = {"tips_status": "generated" if items else "none", "items": items,
           "confirmed": confirmed}
    pfb_id = (latest.output or {}).get("pfb_run_id") if latest else None
    if pfb_id is not None:
        out["pfb_run_id"] = pfb_id
    saved = repo.save_stage_run(StageRun(
        id=None, submission_id=submission_id, stage=Stage.tips, camera=None,
        subject_id=subject_id, version=0, source="teacher",
        upstream_run_ids=_upstream(repo, submission_id, subject_id, latest), prompt_hash=None,
        model_id=None, output=out, status="ok", error=None, warnings=[], duration_sec=0.0,
        created_at=""))
    repo.audit(AUDIT, f"subject:{subject_id}" if subject_id is not None
               else f"submission:{submission_id}",
               {"run_id": saved.id, "from_run_id": audit_from, "count": len(items),
                "confirmed": confirmed})
    return saved


def confirm(repo: Repo, submission_id: int, subject_id: int | None, text_or_lines,
            shown_run_id: int | None, *, draft: bool = False) -> StageRun | None:
    """The coach confirms the tips on screen: `text_or_lines` (a box's text, or its lines),
    which were made from the run `shown_run_id` (None: there was none). Saved as the newest
    run; nothing is saved (None) when nothing was shown and nothing is written. `draft`: kept
    as the coach's words but not confirmed (no PDF carries them; an approval never confirms
    the model's earlier tips in their place)."""
    lines = (checks.split_tips(text_or_lines) if isinstance(text_or_lines, str)
             else checks.split_tips("\n".join(text_or_lines)))
    limit = MAX_PERSON if subject_id is not None else MAX_TEAM
    if len(lines) > limit:
        raise TipsError(TOO_MANY.format(n=limit))
    with repo._tx(join=True):
        latest = _latest(repo, submission_id, subject_id)
        if (latest.id if latest else None) != shown_run_id:
            raise TipsError(NEWER)
        if subject_id is not None and review.busy_for_subject(repo, repo.get_subject(subject_id)):
            raise BusyError("いまは文章を作っているため、できませんでした。終わってから押してください。")
        if latest is None and not lines:
            return None
        return _save(repo, submission_id, subject_id, lines, latest, audit_from=shown_run_id,
                     confirmed=not draft)


def _after_deletion(repo: Repo, submission_id: int, subject_id: int, run: StageRun) -> bool:
    """The coach's last decision on this person's tips (before `run`) was a deletion, made
    for this same analysis: new tips are the coach's to confirm."""
    earlier = [r for r in repo.run_history(submission_id, Stage.tips, None, subject_id)
               if r.id != run.id and r.version < run.version and r.source == "teacher"]
    if not earlier:
        return False
    last = earlier[-1]
    return (not (last.output or {}).get("items") and usable(last)
            and not review.is_stale_chain(repo, last))


def confirm_on_approval(repo: Repo, subject_id: int) -> bool:
    """With a person's approval: confirm the model's own current tips when they have no
    wording fault (`wording`); tips the coach already confirmed or deleted, outdated ones and
    ones with a fault are left as they are. Never raises (an approval never fails for tips)."""
    try:
        subject = repo.get_subject(subject_id)
        sid = subject["submission_id"]
        with repo._tx(join=True):
            now = view(repo, sid, subject_id)
            if now.run is None or now.run.source != "model" or not now.items or now.stale:
                return False
            if now.text_changed or _after_deletion(repo, sid, subject_id, now.run):
                return False
            if wording(repo, sid, now.items):
                return False
            _save(repo, sid, subject_id, now.items, now.run, audit_from=now.run.id)
        return True
    except Exception:   # noqa: BLE001 - the tips are optional
        log.exception("承認と一緒にヒントを確定できませんでした subject=%s", subject_id)
        return False


def for_pdf(repo: Repo, submission_id: int, subject_id: int) -> list[str]:
    """The person's confirmed tips that are current: what their handout prints."""
    now = view(repo, submission_id, subject_id)
    return now.items if now.confirmed else []


def team_for_pdf(repo: Repo, submission_id: int) -> list[str]:
    """The group's confirmed tips that are current."""
    now = view(repo, submission_id, None)
    return now.items if now.confirmed else []


def team_queued(repo: Repo, submission_id: int) -> bool:
    """The group's tips are being made (a job is queued or running)."""
    return repo.has_active_job(submission_id, Stage.tips, None, None)


def can_ask_team(repo: Repo, submission_id: int) -> bool:
    """The coach may ask for the group's tips: all texts are written, two people have one,
    and there are no tips (or only outdated ones) and no job already."""
    return (worker.team_wanted(repo, submission_id, retry_none=False)
            and not team_queued(repo, submission_id))


def queue_team(repo: Repo, submission_id: int) -> int | None:
    """The coach's press: the group's tips are queued (the job id; None: not wanted now, or
    already queued)."""
    return worker.queue_team_tips(repo, submission_id)
