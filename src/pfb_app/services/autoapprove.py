"""機械チェックだけで通す「自動承認」: which feedback texts pass, which wait for a person, and why.

A person's feedback text (the newest `pfb` run of a subject) is approved by the app without a
person's look only when EVERY check of `evaluate` passes. A text that fails any check stays
「人の確認待ち」, and the failed checks are its plain reasons (shown on the review screen). Pure
decision, no writing: `review.auto_approve` (the worker) and `apply_now` (a press) approve
through `review._approve_unseen(auto=True)`, which calls `evaluate` again inside its write
transaction and stores the result of every check (`auto_approvals`, with the time and the policy).

This is not a person's check and is never shown or recorded as one: the subject's
`approved_auto` is 1, the record says 「自動承認（機械チェック済み・人は未確認）」, the handouts
say so too (export.labels), and `revoke` puts every automatic approval of a work back to
「人の確認待ち」 at once. A machine check cannot guarantee that a sentence is true.

Checks (ids are stored; each reuses what the app already has):
  base         the approval gates of 「全員まとめて承認」 (review._skip_reason: fresh, generated, no
               text warnings, identity confirmed, no short ready-made text, not set apart)
               and: not undone by a person before, a video work (documents are always read)
  evidence     the cited scene ids exist, were judged as good scenes (checks.evidence_ref_errors)
               and are events of this very person
  scenes       at least `policy.min_scenes` different scenes are cited
  confidence   every cited scene was recorded with at least `policy.min_confidence`
  identity     in every cited scene the person's number timeline reads the person's own number
               (pipeline.identity_check): 別人かも (another number read) or 判定不能 (nothing read
               near the time, no number for the label) in ONE scene stops it; the same number on
               two people, or two numbers on one person, stops it too
  claims       an action stated in the text (シュート・ドリブル・パス・守り・走る) must have been
               re-checked by the claim check (pipeline.claim_check) as 確認できた in a cited scene;
               a text that states an action nobody re-checked is held (so is a text with a
               scene the claim check weakened, or one that already carries 要確認)
  wording      no comparison, ranking or grading word (review_video.claims._COMPARE,
               services.ask's evaluating and rank words)
  jersey       a jersey number is written 「N番？」 (never 「N番」, 「#N」, 「背番号」)
  player       not probably a non-player (referee, off the court) by the names stage's team
               verdict and the participation rule
  text         not empty, at least `policy.min_chars` characters, no sentence repeated, not the
               same text as another person's in the group
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from pfb_app.domain.models import Kind, PfbOutput, Stage, StageRun
from pfb_app.pipeline import checks, claim_check, motion_check
from pfb_app.pipeline import identity_check as ident
from pfb_app.pipeline.common import usable
from pfb_app.services import ask, review
from pfb_app.storage.db import Repo

POLICY_VERSION = "auto-approve/1"
LABEL = "自動承認（機械チェック済み・人は未確認）"        # the mark, wherever it is shown
SHORT_LABEL = "自動承認・人は未確認"                       # where the long mark does not fit
SWITCH = "機械チェックを通った文を自動承認する"             # the work's switch
AUDIT_APPROVED = review.AUTO_APPROVED
AUDIT_REVOKED = "auto_approval_revoked"

_ORDER = {"high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class Policy:
    min_scenes: int = 2             # the prompt asks the model for two or three scenes
    min_confidence: str = "high"    # of the recorded scenes (Confidence)
    min_chars: int = 20             # shorter than that is no feedback text

    def as_dict(self) -> dict:
        return {"version": POLICY_VERSION, "min_scenes": self.min_scenes,
                "min_confidence": self.min_confidence, "min_chars": self.min_chars}

    @classmethod
    def from_settings(cls, settings) -> "Policy":
        if settings is None:
            return cls()
        return cls(settings.auto_approve_min_scenes, settings.auto_approve_min_confidence,
                   settings.auto_approve_min_chars)


@dataclass(frozen=True)
class Check:
    id: str
    label: str      # what the check looks at, in plain words
    ok: bool
    detail: str     # when it failed: why the text waits for a person; when it passed: what held

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "ok": self.ok, "detail": self.detail}


@dataclass
class Verdict:
    checks: list[Check] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.ok for c in self.checks)

    @property
    def held(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    @property
    def reasons(self) -> list[str]:
        """The plain reasons the text waits for a person (empty when it passed)."""
        return [c.detail for c in self.held]


# --- the words -----------------------------------------------------------------------------------

def _compare_re() -> re.Pattern:
    from pfb_app.review_video import claims   # noqa: PLC0415 - review_video imports services
    return claims._COMPARE   # noqa: SLF001 (the one list of comparison words)


_GRADING = tuple(ask._EVALUATING)   # noqa: SLF001
_RANK = ask._RANK_RE                # noqa: SLF001
# a jersey number said as a fact: 「7番」 (not 「7番？」, 「1番目」), 「#7」, 「背番号」
_JERSEY_FACT = re.compile(r"(?<![0-9])\d{1,3}\s*番(?![？?目手組線地])|[#＃]\s*\d+|背番号")
_HEDGED = re.compile(r"要確認|ように見え")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[。！？!?])", _norm(text)) if s.strip()]


def wording_problems(text: str) -> list[str]:
    """Comparison, ranking and grading words found in the text."""
    norm = _norm(text)
    found = [m.group(0) for m in _compare_re().finditer(norm)]
    found += [w for w in _GRADING if w in norm]
    found += [m.group(0) for m in _RANK.finditer(norm)]
    return list(dict.fromkeys(found))


def jersey_problems(text: str) -> list[str]:
    """Jersey numbers written as a fact (the rule: always 「N番？」)."""
    return list(dict.fromkeys(m.group(0).strip() for m in _JERSEY_FACT.finditer(_norm(text))))


def actions_stated(text: str) -> set[str]:
    """Every action the text states: shoot / dribble / pass / defend (picture re-check) and the
    movement words run / walk / move (tracks)."""
    from pfb_app.review_video import claims   # noqa: PLC0415
    return set().union(*(claims.sentence_actions(s) | motion_check.motion_actions(s)
                         for s in _sentences(text))) if text else set()


# --- the context of one text ------------------------------------------------------------------

@dataclass
class _Ctx:
    refs: list[str]
    events: dict[str, tuple[StageRun, dict]]   # cited event id -> (its behavior run, the event)
    positive: set[str]
    every: set[str]


def _context(repo: Repo, run: StageRun) -> _Ctx:
    out = PfbOutput.model_validate(run.output)
    contribs = [c for c in review._upstream(repo, run, Stage.contribution) if usable(c)]   # noqa: SLF001
    positive, every = review._positive_refs(contribs)   # noqa: SLF001
    events: dict[str, tuple[StageRun, dict]] = {}
    for c in contribs:
        for b in review._upstream(repo, c, Stage.behavior):   # noqa: SLF001
            if not usable(b):
                continue
            for e in (b.output or {}).get("events") or []:
                events.setdefault(e.get("event_id"), (b, e))
    return _Ctx(list(dict.fromkeys(out.evidence_refs)), events, positive, every)


def _check(id_: str, label: str, errors: list[str], ok_detail: str) -> Check:
    detail = "。".join(errors[:3]) + "。" if errors else ok_detail
    return Check(id_, label, not errors, detail)


# --- the checks ---------------------------------------------------------------------------------

def _c_evidence(subject: dict, ctx: _Ctx, out: PfbOutput) -> Check:
    errors = []
    if checks.evidence_ref_errors(out, ctx.positive):
        if any(r not in ctx.every for r in ctx.refs):
            errors.append("根拠にした場面が、記録の中に見つかりません")
        if any(r in ctx.every and r not in ctx.positive for r in ctx.refs):
            errors.append("根拠にした場面に、よかった場面として記録されていないものがあります")
        if not ctx.refs:
            errors.append("根拠にした場面がありません")
    bad = bool(errors)
    others = [r for r in ctx.refs if r in ctx.events and ctx.events[r][1].get("participant")
              != subject.get("participant_id")]
    if others:
        errors.append("根拠の場面に、別の人の記録が混ざっています")
    missing = [r for r in ctx.refs if r not in ctx.events]
    if missing and not bad:
        errors.append("根拠にした場面の記録が見つかりません")
    return _check("evidence", "根拠の場面が実在し、この人の記録である", errors,
                  f"根拠の場面{len(ctx.refs)}件は、この人の記録です")


def _c_scenes(ctx: _Ctx, policy: Policy) -> Check:
    n = len(ctx.refs)
    errors = [f"根拠の場面が{n}件で、必要な{policy.min_scenes}件に足りません"] if n < policy.min_scenes else []
    return _check("scenes", f"根拠の場面が{policy.min_scenes}件以上ある", errors, f"根拠の場面は{n}件です")


def _c_confidence(ctx: _Ctx, policy: Policy) -> Check:
    need = _ORDER[policy.min_confidence]
    low = [r for r in ctx.refs if r in ctx.events
           and _ORDER.get(str(ctx.events[r][1].get("confidence")), 0) < need]
    errors = ["確信度の低い場面が含まれています"] if low else []
    return _check("confidence", "根拠の場面の確信度が基準以上である", errors, "確信度は基準以上です")


def _c_identity(repo: Repo, subject: dict, ctx: _Ctx) -> Check:
    label = "根拠の場面の人物が、全場面で同じ人と確かめられている"
    errors: list[str] = []
    states: dict[str, str] = {}
    facts = []
    seen: set[int] = set()
    for r in ctx.refs:
        if r not in ctx.events:
            continue
        behavior = ctx.events[r][0]
        if behavior.id not in seen:
            seen.add(behavior.id)
            states.update(ident.event_states(repo, behavior))
            f = ident.person_facts(repo, behavior, subject.get("participant_id") or "")
            if f is not None:
                facts.append(f)
    cited = [states.get(r, ident.UNKNOWN) for r in ctx.refs if r in ctx.events]
    if any(s == ident.FOREIGN for s in cited):
        errors.append("根拠の場面に、別の人かもしれない場面があります")
    if any(s == ident.UNKNOWN for s in cited) or not cited:
        errors.append("人物が同じ人か判定できない場面があります")
    if any(f.conflict for f in facts):
        errors.append("同じ人に、ちがう背番号が読み取られています")
    if any(f.duplicate for f in facts):
        errors.append("同じ背番号の人がほかにもいます")
    return _check("identity", label, errors, "根拠の全場面で、同じ人と確かめられました")


def _verdicts(repo: Repo, ctx: _Ctx) -> dict[str, list[dict]]:
    """The re-checks recorded for the cited scenes ({event id: [verdict, ...]}): the table
    claim_checks (pipeline.claim_scan: one verdict per action of the scene), and a verdict kept
    in the behavior run's output under 「claim_checks」 (the older form)."""
    out: dict[str, list[dict]] = {}
    rows: dict[int, dict] = {}
    for r in ctx.refs:
        if r not in ctx.events:
            continue
        behavior = ctx.events[r][0]
        if behavior.id not in rows:
            rows[behavior.id] = repo.claim_checks(behavior.id)
        got = [v for v in rows[behavior.id].get(r, {}).values() if isinstance(v, dict)]
        old = ((behavior.output or {}).get("claim_checks") or {}).get(r)
        if isinstance(old, dict):
            got.append(old)
        if got:
            out[r] = got
    return out


def _c_claims(repo: Repo, ctx: _Ctx, text: str) -> Check:
    label = "動作の断定（シュート・走るなど）が、再確認で確かめられている"
    stated = actions_stated(text)
    recorded = _verdicts(repo, ctx)
    every = [v for vs in recorded.values() for v in vs]
    errors: list[str] = []
    if any(v.get("show") != "keep" for v in every):
        errors.append("動作を弱めた場面が含まれています")
    if _HEDGED.search(_norm(text)) or claim_check.SAFE_TEXT in text:
        errors.append("弱めた言い方（要確認）が文に含まれています")
    supported = {v.get("action") for v in every
                 if v.get("status") == "supported" and v.get("show") == "keep"}
    unverified = sorted(stated - supported)
    if unverified:
        names = {"shoot": "シュート", "dribble": "ドリブル", "pass": "パス", "defend": "守り",
                 "run": "走る", "walk": "歩く", "move": "移動する"}
        errors.append("動作の断定（" + "、".join(names.get(a, a) for a in unverified)
                      + "）が、再確認で確かめられていません")
    return _check("claims", label, errors,
                  "動作の断定はないか、すべて再確認で確かめられています")


def _c_wording(text: str) -> Check:
    words = wording_problems(text)
    errors = ["比較・順位づけ・評価の言葉があります（" + "、".join(words[:4]) + "）"] if words else []
    return _check("wording", "比較・順位づけ・評価の言葉がない", errors, "比較や評価の言葉はありません")


def _c_jersey(text: str) -> Check:
    bad = jersey_problems(text)
    errors = ["背番号が「N番？」の形ではありません（" + "、".join(bad[:3]) + "）"] if bad else []
    return _check("jersey", "背番号は「N番？」の形である", errors, "背番号の断定はありません")


def _c_player(repo: Repo, subject: dict, ctx: _Ctx) -> Check:
    errors: list[str] = []
    seen: set[int] = set()
    for r in ctx.refs:
        if r not in ctx.events or ctx.events[r][0].id in seen:
            continue
        seen.add(ctx.events[r][0].id)
        f = ident.person_facts(repo, ctx.events[r][0], subject.get("participant_id") or "")
        if f is not None and (f.other_team or f.unsure):
            errors.append("選手以外（審判・コート外）かもしれません")
            break
    return _check("player", "選手以外（審判・コート外）の疑いがない", errors, "選手以外の疑いはありません")


def _c_text(repo: Repo, subject: dict, text: str, policy: Policy) -> Check:
    errors: list[str] = []
    flat = checks.normalize_text(text)
    if not flat:
        errors.append("文が空です")
    elif len(flat) < policy.min_chars:
        errors.append(f"文が短すぎます（{policy.min_chars}文字未満）")
    parts = [checks.normalize_text(s) for s in _sentences(text)]
    if len(parts) != len(set(parts)):
        errors.append("同じ文がくり返されています")
    for other in repo.list_subjects(subject["submission_id"]):
        if other["id"] == subject["id"]:
            continue
        run = review._latest_pfb(repo, other)   # noqa: SLF001
        if run is not None and flat and checks.normalize_text(
                (run.output or {}).get("pfb") or "") == flat:
            errors.append("ほかの人と同じ文です")
            break
    return _check("text", "文が空でも極端に短くもなく、重複もない", errors, "文の長さと重複に問題はありません")


def evaluate(repo: Repo, subject: dict, run: StageRun | None,
             policy: Policy = Policy()) -> Verdict:
    """Every check for the subject's latest feedback text; `Verdict.passed` only when all pass."""
    base = review._skip_reason(repo, subject, run)   # noqa: SLF001
    submission = repo.get_submission(subject["submission_id"])
    if base is None and review._kind(repo, submission) != Kind.video:   # noqa: SLF001
        base = "動画以外の作業は、人が確認します。"
    if base is None and repo.auto_approval_undone(subject["id"], run.id):
        base = "自動承認が取り消されたため、人が確認します。"
    verdict = Verdict([Check("base", "承認できる状態の文である（新しく、気になる点がない）",
                             base is None, base or "承認できる状態です")])
    if base is not None or run is None:
        return verdict
    out = PfbOutput.model_validate(run.output)
    text = out.pfb or ""
    ctx = _context(repo, run)
    verdict.checks += [
        _c_evidence(subject, ctx, out), _c_scenes(ctx, policy), _c_confidence(ctx, policy),
        _c_identity(repo, subject, ctx), _c_claims(repo, ctx, text), _c_wording(text),
        _c_jersey(text), _c_player(repo, subject, ctx), _c_text(repo, subject, text, policy)]
    return verdict


# --- the work's automatic approvals ---------------------------------------------------------------

def apply_now(repo: Repo, assignment_id: int, policy: Policy = Policy()) -> tuple[list[int], int]:
    """「いま待っている文にも機械チェックをかける」: run the checks on the texts that wait now and
    approve (automatically) those that pass. Returns (approved subject ids, texts that wait on)."""
    if not repo.get_assignment(assignment_id).get("auto_approve"):
        return [], 0
    approved: list[int] = []
    left = 0
    for sub in repo.list_submissions(assignment_id):
        for subject in repo.list_subjects(sub["id"]):
            if review.subject_status(repo, subject["id"]).value != "review":
                continue
            run = review._latest_pfb(repo, subject)   # noqa: SLF001
            if run is None:
                continue
            new, _ = review._approve_unseen(repo, subject["id"], run.id, auto=True, policy=policy)   # noqa: SLF001
            if new:
                approved.append(subject["id"])
            else:
                left += 1
    return approved, left


def revoke(repo: Repo, assignment_id: int, subject_ids=None) -> list[int]:
    """「自動承認を取り消して人の確認待ちに戻す」: every subject of the work whose CURRENT approval
    is an automatic one (or only `subject_ids` of them) goes back to waiting for a person.
    A person's own approval is never touched. Returns the subject ids put back."""
    back: list[int] = []
    with repo._tx(join=True):   # noqa: SLF001
        for sub in repo.list_submissions(assignment_id):
            for subject in repo.list_subjects(sub["id"]):
                if subject_ids is not None and subject["id"] not in subject_ids:
                    continue
                if subject["approval_status"] == "approved" and subject.get("approved_auto"):
                    repo.set_approval(subject["id"], None)
                    back.append(subject["id"])
        if back:
            repo.audit(AUDIT_REVOKED, f"assignment:{assignment_id}", {"subjects": back})
    return back


@dataclass(frozen=True)
class Listed:
    subject_id: int
    name: str
    text: str
    approved_at: str
    policy_version: str
    checks: list[dict]
    recorded: bool      # False: approved by the older rule (no check results were kept)


def listing(repo: Repo, assignment_id: int) -> list[Listed]:
    """The texts of the work whose current approval is an automatic one, with their records."""
    out: list[Listed] = []
    for sub in repo.list_submissions(assignment_id):
        for subject in repo.list_subjects(sub["id"]):
            if not (subject["approval_status"] == "approved" and subject.get("approved_auto")):
                continue
            if review.subject_status(repo, subject["id"]).value != "approved":
                continue
            rec = repo.active_auto_approval(subject["id"], subject["approved_run_id"])
            run = repo.get_run(subject["approved_run_id"])
            out.append(Listed(subject["id"], subject["display_name"],
                              ((run.output or {}).get("pfb") or "") if run else "",
                              subject.get("approved_at") or "",
                              rec["policy_version"] if rec else "（記録なし）",
                              rec["checks"] if rec else [], rec is not None))
    return out


def counts(repo: Repo, assignment_id: int) -> tuple[int, int]:
    """(自動承認 N件, 人の確認待ち M件) over the work's evaluated people."""
    auto = sum(1 for _ in listing(repo, assignment_id))
    waiting = 0
    for sub in repo.list_submissions(assignment_id):
        for subject in repo.list_subjects(sub["id"]):
            if subject.get("participant_id") is not None and subject.get("excluded"):
                continue
            if review.subject_status(repo, subject["id"]).value == "review":
                waiting += 1
    return auto, waiting
