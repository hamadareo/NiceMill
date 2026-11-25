"""「あとから訂正」の作業（review_mode = correct_later）: 承認を待たずに最新の文を出力し、
訂正は版として残す。

- 最新の文（subject の最新の pfb 実行）がそのまま出力の対象。承認の操作は要らない。
- 訂正は `text_versions` に追記だけする（削除しない）。種類 kind:
    edit        文を直した（新しい pfb 実行を作る。前の実行の番号を prev_run_id に残す）
    revert      前の版に戻した（前の実行の写しを新しい最新の実行にする）
    regenerate  この人だけ作り直した
    report      「この文は間違い」の報告（理由つき。版は増えない）
    resolve     報告を確認して閉じた（版は増えない）
- 「版」は、edit / revert / regenerate の回数（0 = AI が作ったまま）。
- 確認状況: 訂正なし / 訂正済み 版N / 誤りの報告あり・確認中（報告のあとに edit・revert・
  regenerate・resolve がまだないもの）。
- 画面から作る新しい作業は「あとから訂正」で始まる（`NEW_WORK`）。前からある作業（NULL・approve）
  はそのまま。
- 作り直し（「この人だけ作り直す」、解析のやり直し）でできた新しい文は、ワーカーが `after_pfb` で
  版として残す（前の文の実行番号つきなので「前の版に戻す」で戻せる）。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from pfb_app.domain.models import Stage, StageRun
from pfb_app.services import review
from pfb_app.storage.db import _now

log = logging.getLogger(__name__)

APPROVE, CORRECT_LATER = "approve", "correct_later"
MODES = (APPROVE, CORRECT_LATER)
NEW_WORK = CORRECT_LATER      # the mode of a work made from the screens (since 2026-11-23)
MODE_WORDS = {CORRECT_LATER: "あとから訂正", APPROVE: "承認して出す"}
REDO_ASKED, REDO_ANALYSIS = "作り直し", "解析のやり直し"   # why a regenerate row was written
BUMPS = ("edit", "revert", "regenerate")
REASONS = ("人物の取り違え", "動作が違う", "背番号が違う", "その他")
STATE_NONE, STATE_REPORT = "訂正なし", "誤りの報告あり・確認中"
KIND_WORD = {"edit": "文を訂正", "revert": "前の版に戻した", "regenerate": "作り直した"}


class CorrectionError(ValueError):
    """A correction that cannot be recorded (plain Japanese message)."""


def mode_of(work: dict | None) -> str:
    return (work or {}).get("review_mode") or APPROVE


def is_correct_later(repo, assignment_id: int) -> bool:
    return mode_of(repo.get_assignment(assignment_id)) == CORRECT_LATER


def approval_enabled(repo, assignment_id: int) -> bool:
    """False for a 「あとから訂正」 work: the screens show no approve button / auto-approval
    notice and no 「承認」 word."""
    return not is_correct_later(repo, assignment_id)


def set_mode(repo, assignment_id: int, mode: str) -> None:
    if mode not in MODES:
        raise CorrectionError("設定が正しくありません。")
    repo.set_review_mode(assignment_id, mode)
    repo.audit("review_mode", f"assignment:{assignment_id}", {"mode": mode})


@dataclass
class Version:
    id: int
    version: int          # the version in force after this row
    kind: str
    run_id: int | None
    prev_run_id: int | None
    text: str | None
    prev_text: str | None
    reason: str | None
    note: str | None
    actor: str | None
    created_at: str


def history(repo, subject_id: int) -> list[Version]:
    rows = repo.conn.execute(
        "SELECT id, version, kind, run_id, prev_run_id, text, prev_text, reason, note, actor,"
        " created_at FROM text_versions WHERE subject_id=? ORDER BY id", (subject_id,)).fetchall()
    return [Version(*tuple(r)) for r in rows]


def version_no(repo, subject_id: int) -> int:
    return sum(1 for v in history(repo, subject_id) if v.kind in BUMPS)


def open_report(repo, subject_id: int) -> Version | None:
    open_ = None
    for v in history(repo, subject_id):
        if v.kind == "report":
            open_ = v
        elif open_ is not None:     # an edit, revert, regenerate or resolve closes it
            open_ = None
    return open_


@dataclass
class State:
    version: int
    label: str                 # the status line
    corrected_at: str | None   # when the newest version was made (None: version 0)
    reported: Version | None
    lines: list[str]           # the correction record, oldest first


def state(repo, subject_id: int) -> State:
    rows = history(repo, subject_id)
    bumps = [v for v in rows if v.kind in BUMPS]
    n = len(bumps)
    rep = open_report(repo, subject_id)
    if rep is not None:
        label = STATE_REPORT
    elif n:
        label = f"訂正済み 版{n}"
    else:
        label = STATE_NONE
    lines = []
    for v in rows:
        if v.kind not in BUMPS:
            continue
        lines.append(f"版{v.version}（{review_when(v.created_at)}）{KIND_WORD[v.kind]}"
                     + (f"：{v.reason}" if v.reason else "") + (f"（{v.note}）" if v.note else ""))
    return State(n, label, bumps[-1].created_at if bumps else None, rep, lines)


# --- what the other features use in place of 「承認した文」 -------------------------------------
# 成長の記録, 記録への質問, ヒント, やさしい文章版, 解説動画 and 試合・活動のまとめ use, in a
# correct_later work, the newest text unless someone reported it wrong (see docs/correct_later.md).
NO_TEXT_LIVE = "文がまだありません"
REPORTED_LIVE = "誤りの報告を確認中です"


def _work_of(repo, subject: dict) -> dict:
    return repo.get_assignment(repo.get_submission(subject["submission_id"])["assignment_id"])


def usable_run(repo, subject: dict, work: dict | None = None) -> StageRun | None:
    """The person's text that the other features build on.

    approve: the approved run while the person is approved now (review.subject_status), else
    None. correct_later: the newest feedback run when it is ok and has a text (what the PDF
    carries), unless it is 「誤りの報告あり・確認中」: nothing new is built on a text someone said
    is wrong until the report is closed or the text is corrected (None)."""
    from pfb_app.domain.models import SubjectStatus   # noqa: PLC0415
    work = work if work is not None else _work_of(repo, subject)
    if mode_of(work) != CORRECT_LATER:
        if review.subject_status(repo, subject["id"]) != SubjectStatus.approved:
            return None
        return repo.get_run(subject["approved_run_id"]) if subject.get("approved_run_id") else None
    run = review._latest_pfb(repo, subject)   # noqa: SLF001
    if run is None or run.status != "ok" or not ((run.output or {}).get("pfb") or "").strip():
        return None
    if open_report(repo, subject["id"]) is not None:
        return None
    return run


def unusable_why(repo, subject: dict) -> str:
    """Why a correct_later person's text is not used (usable_run gave None)."""
    return REPORTED_LIVE if open_report(repo, subject["id"]) is not None else NO_TEXT_LIVE


def review_when(stamp: str | None) -> str:
    from pfb_app.export.sheets import local_date, local_time   # noqa: PLC0415
    if not stamp:
        return ""
    d, t = local_date(stamp), local_time(stamp)
    return f"{d} {t.split(' ', 1)[-1]}" if d and t else ""


def _row(repo, subject_id, kind, *, run_id=None, prev_run_id=None, text=None, prev_text=None,
         reason=None, note=None, actor=None):
    version = version_no(repo, subject_id) + (1 if kind in BUMPS else 0)
    repo.conn.execute(
        "INSERT INTO text_versions (subject_id, version, kind, run_id, prev_run_id, text,"
        " prev_text, reason, note, actor, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (subject_id, version, kind, run_id, prev_run_id, text, prev_text, reason, note, actor,
         _now()))
    repo.audit("correction_" + kind, f"subject:{subject_id}", {"version": version, "reason": reason})


def _latest(repo, subject_id: int) -> StageRun | None:
    return review._latest_pfb(repo, repo.get_subject(subject_id))


def edit_text(repo, subject_id: int, text: str, *, actor: str | None = None,
              reason: str | None = None, note: str | None = None) -> StageRun:
    """The correction: the new text becomes the newest pfb run (the one the PDF carries)."""
    prev = _latest(repo, subject_id)
    if prev is None:
        raise CorrectionError("直す文がまだありません。")
    with repo._tx(join=True):
        run = review.write_manual_pfb(repo, subject_id, text)
        _row(repo, subject_id, "edit", run_id=run.id, prev_run_id=prev.id, text=run.output["pfb"],
             prev_text=(prev.output or {}).get("pfb"), reason=reason, note=note, actor=actor)
    return run


def report_wrong(repo, subject_id: int, reason: str, *, note: str = "",
                 actor: str | None = None) -> None:
    if reason not in REASONS:
        raise CorrectionError("理由を選んでください。")
    cur = _latest(repo, subject_id)
    if cur is None:
        raise CorrectionError("報告する文がまだありません。")
    with repo._tx(join=True):
        _row(repo, subject_id, "report", run_id=cur.id, text=(cur.output or {}).get("pfb"),
             reason=reason, note=note or None, actor=actor)


def resolve_report(repo, subject_id: int, *, actor: str | None = None,
                   note: str | None = None) -> None:
    if open_report(repo, subject_id) is None:
        raise CorrectionError("確認中の報告はありません。")
    cur = _latest(repo, subject_id)
    with repo._tx(join=True):
        _row(repo, subject_id, "resolve", run_id=cur.id if cur else None, note=note, actor=actor)


def record_regenerated(repo, subject_id: int, prev_text: str | None, run: StageRun, *,
                       actor: str | None = None, prev_run_id: int | None = None,
                       reason: str = REDO_ASKED) -> None:
    """Call after 「この人だけ作り直す」 made `run`: the new text is a new version. With
    `prev_run_id` (the run it replaced) 「前の版に戻す」 can bring the earlier text back."""
    with repo._tx(join=True):
        _row(repo, subject_id, "regenerate", run_id=run.id,
             prev_run_id=prev_run_id, text=(run.output or {}).get("pfb"), prev_text=prev_text,
             reason=reason, actor=actor)


def _asked_by(repo, submission_id: int, job_id: int | None) -> tuple[bool, str | None]:
    """(whether a person asked for the job — review.regenerate_from's audit entry lists it —,
    their login name)."""
    if job_id is None:
        return False, None
    rows = repo.conn.execute(
        "SELECT detail, actor FROM audit_log WHERE action='regenerate' AND target=?"
        " ORDER BY id DESC LIMIT 200", (f"submission:{submission_id}",)).fetchall()
    for detail, actor in rows:
        try:
            ids = (json.loads(detail or "{}") or {}).get("job_ids") or []
        except ValueError:
            continue
        if job_id in ids:
            return True, actor
    return False, None


def after_pfb(repo, runs, job_id: int | None = None) -> int:
    """The worker's hook after a feedback job (pipeline.runner): in a 「あとから訂正」 work, a
    new text that replaces an earlier text of the person becomes a version (kind regenerate),
    with the run it replaced, so the record explains every change of what the PDF carries.
    The person's first text is version 0 and is not recorded (nor is a new text when no
    earlier run of the person had one). Returns how many were recorded."""
    done = 0
    for run in runs:
        if Stage(run.stage) != Stage.pfb or run.subject_id is None or run.status != "ok":
            continue
        sub = repo.get_submission(run.submission_id)
        if mode_of(repo.get_assignment(sub["assignment_id"])) != CORRECT_LATER:
            continue
        earlier = [r for r in repo.run_history(run.submission_id, Stage.pfb,
                                               subject_id=run.subject_id)
                   if r.id < run.id and r.status == "ok"]
        if not any((r.output or {}).get("pfb") for r in earlier):
            continue          # no text before it: the person's first text (version 0)
        prev = earlier[-1]    # the run it replaces (its text may be none: too few scenes)
        asked, actor = _asked_by(repo, run.submission_id, job_id)
        record_regenerated(repo, run.subject_id, (prev.output or {}).get("pfb"), run,
                           actor=actor, prev_run_id=prev.id,
                           reason=REDO_ASKED if asked else REDO_ANALYSIS)
        done += 1
    return done


PDF_PEOPLE = "people"    # the audit row of a PDF download: who that PDF carried


def pdf_people(repo, assignment_id: int) -> set[int] | None:
    """Who the last received participants' PDF carried (its download's audit row), None when
    that download did not record it (before 2026-11-23, or a saved PDF of a work whose videos
    were removed)."""
    row = repo.conn.execute(
        "SELECT detail FROM audit_log WHERE action='pdf_downloaded' AND target=?"
        " ORDER BY id DESC LIMIT 1", (f"assignment:{assignment_id}",)).fetchone()
    try:
        people = json.loads(row[0] or "{}").get(PDF_PEOPLE) if row else None
    except (TypeError, ValueError):
        return None
    return {int(i) for i in people} if isinstance(people, list) else None


def in_pdf_now(repo, assignment_id: int) -> set[int] | None:
    """Who a participants' PDF made now carries (the export's own report, no pictures); None
    when it cannot be read (logged)."""
    from pfb_app.export.sheets import build_report   # noqa: PLC0415 - sheets imports this module
    from pfb_app.storage.db import Repo              # noqa: PLC0415
    try:
        plain = repo if type(repo) is Repo else Repo(repo.conn, repo.settings)
        return {s.subject_id for s in build_report(plain, assignment_id, with_pictures=False).exported}
    except Exception:   # noqa: BLE001 - the notice then errs on the side of 「再出力が必要」
        log.exception("reading who the PDF carries failed")
        return None


def in_pdf(repo, assignment_id: int, subject_ids) -> set[int]:
    """Of `subject_ids`, those whose change matters to the PDF: the received PDF carried them,
    or a PDF made now carries them. A person in neither (say, not confirmed as one person)
    changes nothing a participant gets. When the download did not record who it carried, or
    the PDF now cannot be read, every one counts (as before 2026-11-23)."""
    ids = set(subject_ids)
    if not ids:
        return set()
    carried = pdf_people(repo, assignment_id)
    if carried is None:
        return ids
    keep = ids & carried
    if ids - carried:
        now = in_pdf_now(repo, assignment_id)
        keep |= (ids - carried) if now is None else (ids - carried) & now
    return keep


def changed_since(repo, assignment_id: int, stamp: str | None, *,
                  pdf_only: bool = False) -> dict[int, str]:
    """{subject id: when} of the people whose text or check state changed after `stamp` (the
    last PDF download): a newer feedback run, or a correction / report row. Empty without a
    stamp (nothing was output yet). `pdf_only`: only the people whose change matters to the
    PDF (`in_pdf`: 「再出力が必要」 is said for them only)."""
    if not stamp:
        return {}
    out: dict[int, str] = {}
    rows = repo.conn.execute(
        "SELECT r.subject_id, MAX(r.created_at) FROM stage_runs r JOIN submissions s"
        " ON r.submission_id = s.id WHERE s.assignment_id=? AND r.stage=? AND r.subject_id"
        " IS NOT NULL GROUP BY r.subject_id", (assignment_id, Stage.pfb.value)).fetchall()
    rows += repo.conn.execute(
        "SELECT v.subject_id, MAX(v.created_at) FROM text_versions v JOIN subjects t"
        " ON v.subject_id = t.id JOIN submissions s ON t.submission_id = s.id"
        " WHERE s.assignment_id=? GROUP BY v.subject_id", (assignment_id,)).fetchall()
    for sid, when in rows:
        if when and when > stamp and when > out.get(sid, ""):
            out[sid] = when
    if pdf_only and out:
        keep = in_pdf(repo, assignment_id, out)
        out = {sid: when for sid, when in out.items() if sid in keep}
    return out


def revert(repo, subject_id: int, *, actor: str | None = None) -> StageRun:
    """Back to the text the newest version replaced (the earlier run is copied as the newest
    run; nothing is deleted)."""
    bumps = [v for v in history(repo, subject_id) if v.kind in BUMPS]
    if not bumps or bumps[-1].prev_run_id is None:
        raise CorrectionError("戻せる前の版がありません。")
    last = bumps[-1]
    target = repo.get_run(last.prev_run_id)
    cur = _latest(repo, subject_id)
    if target is None:
        raise CorrectionError("前の版が見つかりません。")
    with repo._tx(join=True):
        copy = repo.save_stage_run(StageRun(
            id=None, submission_id=target.submission_id, stage=Stage.pfb, camera=target.camera,
            subject_id=subject_id, version=0, source=target.source,
            upstream_run_ids=list(target.upstream_run_ids), prompt_hash=target.prompt_hash,
            model_id=target.model_id, output=target.output, status=target.status, error=None,
            warnings=list(target.warnings), duration_sec=0.0, created_at=""))
        _row(repo, subject_id, "revert", run_id=copy.id, prev_run_id=cur.id if cur else None,
             text=(target.output or {}).get("pfb"), prev_text=(cur.output or {}).get("pfb") if cur else None,
             reason=f"版{last.version}を取り消し", actor=actor)
    return copy
