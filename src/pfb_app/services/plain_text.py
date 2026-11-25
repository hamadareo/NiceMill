"""やさしい文章版: an approved feedback text, reworded for a younger reader.

The coach picks who reads the handouts (「13歳以上」: the approved text as it is, 「9〜12歳」,
「6〜8歳」, `plain_checks.LEVELS`), asks for the reworded texts, reads and edits them next to the
approved one, and the handout PDFs, the one-person ZIP and the phone pages take the picked version.

What is stored, and where:
- the reworded text of one person and level is a `plain` stage run (subject_id, `camera` = the
  level; the worker's `pipeline.plain.run_plain` writes it, a coach's edit adds a run with
  source "teacher"). Its `upstream_run_ids` is the approved feedback run it was made from. No new
  table: erasing a person or deleting a work removes the runs with everything else of theirs;
- the work's choice (the level, and the people who keep the approved text) is the app setting
  "plain:<work id>" (`choice` / `choose`).

The approved text is never changed, always stays the coach's version and is what the
evaluator PDF, the review screen and the growth record read. In a 「あとから訂正」 work the text
a version is made from is the person's newest text unless it is reported wrong
(services.corrections.usable_run): a correction outdates the reworded text, and the handout
then carries the corrected text until the version is made again. A person's handout takes the
reworded text only while it is `ready`: made from the text that is approved now (`outdated`
otherwise), passed the checks (`fallback`: it never did, so the approved text is used) and not
switched off for that person. Whatever is not ready falls back to the approved text, so a
handout never fails and never carries an unchecked or stale sentence. A coach's own edit has to
pass the same checks (`save_edit` refuses with the reasons in plain words).
"""
from __future__ import annotations

from dataclasses import dataclass

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline import plain as pipeline_plain
from pfb_app.pipeline import plain_checks
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.runner import enqueue_once
from pfb_app.storage.db import Repo

ORIGINAL = plain_checks.ORIGINAL
LEVELS = plain_checks.LEVELS
CHOICES = (ORIGINAL, *LEVELS)
AUDIT_EDIT = "plain_edit"
SETTING = "plain:{}"

READY, MAKING, FALLBACK, OUTDATED, NONE, UNAVAILABLE = (
    "ready", "making", "fallback", "outdated", "none", "unavailable")

LABELS = {ORIGINAL: plain_checks.ORIGINAL_LABEL, **{k: v.label for k, v in LEVELS.items()}}
WHY_ORIGINAL = {
    "off": "この人は、元の文を使う設定です",
    FALLBACK: "意味を変えずに言いかえられなかったため、元の文を使います",
    OUTDATED: "元の文が変わったため、元の文を使います（作り直してください）",
    NONE: "まだ作っていないため、元の文を使います",
    MAKING: "作っている途中のため、元の文を使います",
    UNAVAILABLE: "承認した文章がないため、元の文を使います",
}
NOT_APPROVED = "この人の文章は、承認が変わったため言いかえできません。承認してからやり直してください。"
# 「あとから訂正」 works: the text it is made from is the newest one unless reported wrong
NOT_USABLE_LIVE = "この人の文は、まだないか、誤りの報告を確認中のため、言いかえできません。"
UNAVAILABLE_LIVE = "元の文がないか、誤りの報告を確認中のため、元の文を使います"
BUSY = "この人の言いかえを作っている途中です。できてから直してください。"
NOTHING = "いまは頼めませんでした。少し待ってから、もう一度押してください。"
UNKNOWN_LEVEL = "読み手の選びかたが正しくありません。"


class Refused(Exception):
    """Not done (a plain Japanese message; `reasons` the checks' words for a refused edit)."""

    def __init__(self, message: str, reasons: list[str] | None = None):
        super().__init__(message)
        self.reasons = reasons or []


# --- the work's choice -------------------------------------------------------------------------

@dataclass(frozen=True)
class Choice:
    level: str = ORIGINAL
    off: frozenset[int] = frozenset()      # people who keep the approved text


def choice(repo: Repo, assignment_id: int) -> Choice:
    """The work's choice (the approved text for everybody until a coach picks a level)."""
    raw = repo.get_setting(SETTING.format(assignment_id))
    if not isinstance(raw, dict) or raw.get("level") not in CHOICES:
        return Choice()
    off = raw.get("off")
    return Choice(raw["level"], frozenset(int(i) for i in off) if isinstance(off, list) else frozenset())


def choose(repo: Repo, assignment_id: int, level: str, off=()) -> Choice:
    """Save the work's choice (an unknown level is refused)."""
    if level not in CHOICES:
        raise Refused(UNKNOWN_LEVEL)
    repo.get_assignment(assignment_id)
    saved = Choice(level, frozenset(int(i) for i in off))
    repo.set_setting(SETTING.format(assignment_id),
                     {"level": level, "off": sorted(saved.off)})
    return saved


# --- who has what ------------------------------------------------------------------------------

@dataclass
class State:
    kind: str                        # READY, MAKING, FALLBACK, OUTDATED, NONE or UNAVAILABLE
    text: str | None = None          # the reworded text (READY only)
    source: str | None = None        # "model" or "teacher" (READY only)
    run: StageRun | None = None


def state(repo: Repo, subject: dict, level: str) -> State:
    """The person's version for `level` (a level of LEVELS)."""
    found = pipeline_plain.approved_text(repo, subject)
    if found is None:
        return State(UNAVAILABLE)
    approved, _ = found
    sid, subject_id = subject["submission_id"], subject["id"]
    run = repo.latest_run(sid, Stage.plain, level, subject_id)
    if repo.has_active_job(sid, Stage.plain, level, subject_id):
        return State(MAKING, run=run)
    if run is None:
        return State(NONE)
    if run.upstream_run_ids != [approved.id]:
        return State(OUTDATED, run=run)
    text = (run.output or {}).get("text") if run.status == "ok" else None
    if not text:
        return State(FALLBACK, run=run)
    return State(READY, text=text, source=run.source, run=run)


def people(repo: Repo, assignment_id: int) -> list[tuple[dict, str]]:
    """[(subject, group label)] of the people with an approved text now, in name order (the
    people a handout is made for)."""
    from pfb_app.export.pdf import name_order   # noqa: PLC0415 - export imports this module
    work = repo.get_assignment(assignment_id)
    video = work["kind"] == Kind.video.value
    out = []
    for group in repo.list_submissions(assignment_id):
        for subject in repo.list_subjects(group["id"]):
            if (not video or is_evaluated(subject)) \
                    and pipeline_plain.approved_text(repo, subject) is not None:
                out.append((subject, group["label"]))
    return sorted(out, key=lambda p: name_order(p[0]["display_name"]))


@dataclass
class Usage:
    level: str
    plain: list[str]                       # people whose handout takes the reworded text
    original: list[tuple[str, str]]        # (name, why the approved text is used)


def usage(repo: Repo, assignment_id: int) -> Usage:
    """Whose handout takes the reworded text under the work's choice, and why not the others."""
    picked = choice(repo, assignment_id)
    used: list[str] = []
    kept: list[tuple[str, str]] = []
    for subject, _label in people(repo, assignment_id):
        name = subject["display_name"]
        if picked.level == ORIGINAL:
            continue
        if subject["id"] in picked.off:
            kept.append((name, WHY_ORIGINAL["off"]))
            continue
        shown = state(repo, subject, picked.level)
        if shown.kind == READY:
            used.append(name)
        else:
            kept.append((name, WHY_ORIGINAL[shown.kind]))
    return Usage(picked.level, used, kept)


def making(repo: Repo, assignment_id: int, level: str) -> int:
    """How many people's versions for `level` are being made now."""
    return sum(1 for subject, _ in people(repo, assignment_id)
               if state(repo, subject, level).kind == MAKING)


def pending(repo: Repo, assignment_id: int, level: str) -> int:
    """How many people (who do not keep the approved text) have no version for `level` that is
    ready or being made: what 「言いかえた文章を作る」 would ask for."""
    off = choice(repo, assignment_id).off
    return sum(1 for subject, _ in people(repo, assignment_id)
               if subject["id"] not in off
               and state(repo, subject, level).kind in (NONE, OUTDATED, FALLBACK))


def handout_texts(repo: Repo, assignment_id: int) -> dict[int, str]:
    """{subject id: the text the person's handout shows} for the people who get the reworded
    one (empty while the work's choice is the approved text)."""
    picked = choice(repo, assignment_id)
    if picked.level == ORIGINAL:
        return {}
    out = {}
    for subject, _label in people(repo, assignment_id):
        if subject["id"] in picked.off:
            continue
        shown = state(repo, subject, picked.level)
        if shown.kind == READY:
            out[subject["id"]] = shown.text
    return out


def handout_notes(repo: Repo, assignment_id: int) -> dict[int, dict]:
    """{subject id: {"label", "text"}} for the evaluator PDF: which version each person's
    handout carries, and the reworded text itself when it does (nothing while the choice is the
    approved text)."""
    picked = choice(repo, assignment_id)
    if picked.level == ORIGINAL:
        return {}
    texts = handout_texts(repo, assignment_id)
    notes = {}
    for subject, _label in people(repo, assignment_id):
        sid = subject["id"]
        notes[sid] = ({"label": LABELS[picked.level], "text": texts[sid]} if sid in texts
                      else {"label": LABELS[ORIGINAL], "text": None})
    return notes


# --- asking for the texts, and a coach's edit --------------------------------------------------

def request(repo: Repo, assignment_id: int, level: str, subject_ids=None, *,
            redo: bool = False) -> int:
    """Ask the worker for the reworded texts of `level`: everybody with an approved text (or
    `subject_ids`) who has none that is ready (`redo`: also those who have one that the model
    made; a coach's edit is replaced only for the people named in `subject_ids`). Returns how
    many were queued; Refused when there were people to do but nothing could be queued."""
    if level not in LEVELS:
        raise Refused(UNKNOWN_LEVEL)
    wanted = set(subject_ids) if subject_ids is not None else None
    queued = eligible = 0
    for subject, _label in people(repo, assignment_id):
        if wanted is not None and subject["id"] not in wanted:
            continue
        shown = state(repo, subject, level)
        if shown.kind == MAKING:
            continue
        if shown.kind == READY and not (redo and (wanted is not None or shown.source != "teacher")):
            continue
        eligible += 1
        if enqueue_once(repo, subject["submission_id"], Stage.plain, level, subject["id"]) is not None:
            queued += 1
    if eligible and not queued:
        raise Refused(NOTHING)
    return queued


def save_edit(repo: Repo, subject_id: int, level: str, text: str) -> StageRun:
    """A coach's own wording of the reworded text: saved as the newest version when it passes
    the same checks as a model's (Refused with the reasons otherwise). The same text as the
    version now ready changes nothing."""
    if level not in LEVELS:
        raise Refused(UNKNOWN_LEVEL)
    with repo._tx(join=True):
        subject = repo.get_subject(subject_id)
        info = pipeline_plain.context_for(repo, subject)
        if info is None:
            from pfb_app.services import corrections   # noqa: PLC0415
            live = corrections.mode_of(corrections._work_of(repo, subject)) == corrections.CORRECT_LATER  # noqa: SLF001
            raise Refused(NOT_USABLE_LIVE if live else NOT_APPROVED)
        sid = subject["submission_id"]
        if repo.has_active_job(sid, Stage.plain, level, subject_id):
            raise Refused(BUSY)
        text = plain_checks.tidy(text)
        shown = state(repo, subject, level)
        if shown.kind == READY and shown.text == text:
            return shown.run
        reasons = info.problems(level, text)
        if reasons:
            raise Refused("この文は使えません。" + "".join(f"\n・{r}" for r in reasons), reasons)
        run = repo.save_stage_run(StageRun(
            id=None, submission_id=sid, stage=Stage.plain, camera=level, subject_id=subject_id,
            version=0, source="teacher", upstream_run_ids=[info.run.id], prompt_hash=None,
            model_id=None, output={"level": level, "text": text}, status="ok", error=None,
            warnings=[], duration_sec=0.0, created_at=""))
        repo.audit(AUDIT_EDIT, f"subject:{subject_id}", {"run_id": run.id, "level": level})
    return run
