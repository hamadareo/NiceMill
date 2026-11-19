"""A short ready-made text for a person the analysis found too few scenes of (UX round 2, L6).

A person whose feedback came back 「根拠不足」 (pfb_status insufficient_positive_evidence) used to
be left out of the PDF unless the coach wrote a text from scratch. On a video work step 5 now
offers a short text built only from what the data supports: the activity's name (the work's
sport, else a short activity line) and that the person appears in the video. It never claims
effort, teamwork or finishing; the analysis recorded none of that for this person.

- `offer` gives the text for a person's latest feedback run when it is fresh and too thin (video
  works only; a document has no scenes), else None. Nothing is stored for an offer.
- `approve_short` is a person's press of 「この文で承認する」: in one write transaction it checks
  that the text on screen is still the one offered for that very run, saves it as the person's
  newest feedback run (source "teacher", model_id `MODEL_ID`, warnings [`MARK`], the thin run's
  upstream, so it goes stale exactly like other texts), approves it with `review.approve` (every
  approval gate) and writes the audit row `AUDIT` {"run_id", "from_run_id"}. Any refusal rolls
  everything back.
- Never automatic, never in 「全員まとめて承認」: `review._skip_reason` refuses a run with
  `MODEL_ID` or `MARK` (SKIP_SHORT) in `bulk_approve` and `auto_approve`, and the `MARK` warning
  is a text warning too; step 5 never offers it for the bulk press either.
- The PDFs take it like any approved text (`export.pdf.collect_approved`); the evaluator PDF
  says （短い文）, not （手書き）, since nobody wrote it.
"""
from pfb_app.domain.models import Kind, PfbOutput, Stage, StageRun
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import review
from pfb_app.services.content_check import sport_phrase
from pfb_app.services.review import ApprovalError, BusyError
from pfb_app.storage.db import Repo

__all__ = ["AUDIT", "FALLBACK", "MARK", "MODEL_ID", "NOTE", "SHORT_SOURCE", "SKIP_SHORT",
           "UNSUPPORTED", "approve_short", "compose", "is_short", "offer"]

MARK = review.SHORT_MARK          # 「短い文」: the chip, and the short run's warnings marker
MODEL_ID = review.SHORT_MODEL_ID  # the short run's model_id: which template made it
SKIP_SHORT = review.SKIP_SHORT
AUDIT = "short_text"              # audit action of an approved short text
NOTE = "映っている場面が少なく、自動の文章は作れませんでした。短い文を用意しました。"
SHORT_SOURCE = "用意した短い文"     # where the text came from (詳細画面)
# claims the analysis never records for a person with too few scenes: an activity line that
# carries one is not used (its words would read as said about the person)
UNSUPPORTED = ("最後まで", "仲間", "一緒", "協力", "頑張", "がんば", "努力", "積極", "熱心", "丁寧",
               "助け", "活躍", "貢献", "支え", "みんなで", "力を合わせ", "工夫", "粘り", "諦め",
               "あきらめ", "上手", "成長", "リーダー", "励ま")
CHANGED = "文章が新しくなりました。確かめてから押してください。"
NOT_OFFERED = "この人には短い文を用意できません。"

FALLBACK = "この活動に参加している様子が、動画に映っていました。"
_SPORT = "{activity}に参加している様子が、動画に映っていました。"
_LINE = "「{activity}」の活動に参加している様子が、動画に映っていました。"
MAX_LINE = 20                     # chars: a longer activity line is a description, not a name


def _line(task_text: str | None) -> str | None:
    """The work's activity line when it reads as a name: one short line, not a sentence or a
    list, and no word of `UNSUPPORTED`."""
    lines = (task_text or "").strip().splitlines()
    line = lines[0].strip() if len(lines) == 1 else ""
    if (not line or len(line) > MAX_LINE or any(c in line for c in "。、，,！？!?「」")
            or any(w in line for w in UNSUPPORTED)):
        return None
    return line


def compose(assignment: dict) -> str:
    """The short text for the work: its sport, else its activity line, else a plain one."""
    if assignment.get("sport"):
        return _SPORT.format(activity=sport_phrase(assignment["sport"]))
    line = _line(assignment.get("task_text"))
    return _LINE.format(activity=line) if line else FALLBACK


def is_short(run: StageRun | None) -> bool:
    """The run is a short text a person approved with 「この文で承認する」."""
    return run is not None and run.model_id == MODEL_ID and MARK in run.warnings


def _thin(repo: Repo, run: StageRun | None) -> bool:
    return (run is not None and run.status == "ok"
            and (run.output or {}).get("pfb_status") != "generated"
            and not review.is_stale_chain(repo, run))


def offer(repo: Repo, subject: dict, run: StageRun | None) -> str | None:
    """The short text offered for the subject whose latest feedback is `run`: a video work's
    evaluated person whose text is fresh and too thin; None otherwise."""
    submission = repo.get_submission(subject["submission_id"])
    work = repo.get_assignment(submission["assignment_id"])
    if Kind(work["kind"]) != Kind.video or not is_evaluated(subject) or not _thin(repo, run):
        return None
    return compose(work)


def approve_short(repo: Repo, subject_id: int, run_id: int, text: str) -> StageRun:
    """「この文で承認する」: store `text` (the short text shown for the thin run `run_id`) as the
    subject's newest feedback and approve it, in one transaction. ApprovalError when the text
    or the run is no longer the one offered, or a gate of `review.approve` fails; BusyError
    while the analysis runs. Nothing is kept on a refusal."""
    with repo._tx(join=True):
        subject = repo.get_subject(subject_id)
        sid = subject["submission_id"]
        if review.busy_for_subject(repo, subject):   # other people's texts may be written
            raise BusyError(review._busy_message("短い文を承認"))
        run = repo.latest_run(sid, Stage.pfb, subject_id=subject_id)
        if run is None or run.id != run_id:
            raise ApprovalError(CHANGED)
        offered = offer(repo, subject, run)
        if offered is None:
            raise ApprovalError(NOT_OFFERED)
        if (text or "").strip() != offered:
            raise ApprovalError(CHANGED)
        out = PfbOutput(pfb_status="generated", pfb=offered, evidence_refs=[])
        saved = repo.save_stage_run(StageRun(
            id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subject_id,
            version=0, source="teacher", upstream_run_ids=list(run.upstream_run_ids),
            prompt_hash=None, model_id=MODEL_ID, output=out.model_dump(), status="ok",
            error=None, warnings=[MARK], duration_sec=0.0, created_at=""))
        review.approve(repo, subject_id)
        repo.audit(AUDIT, f"subject:{subject_id}", {"run_id": saved.id, "from_run_id": run.id})
    return saved
