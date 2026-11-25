"""「同一人物の確認」: only a person confirmed to be one player is printed (correct_later outputs).

The tracker links pieces of track into one label and sometimes links different players. What can
be checked, from the app's own data (nothing new is computed from the video):
- the number the names stage read along the label's track (`cues.number[label].timeline`: one
  read per ~6 s, mostly empty, sometimes another number) and its dominant number;
- `pipeline.identity_check.event_states`: for each event of the person, SAME (the number read
  at that time is the dominant one), FOREIGN (another number is read there) or UNKNOWN (nothing
  is read near it);
- the labels of the person (more than one with different dominant numbers = a mix), and whether
  another label reads the same number (duplicate).

Verdicts (a strict default; `settings.identity_gate` may override a threshold):
  ok      the dominant number has at least `min_reads` reads and `min_share` of the numbered
          reads, there are at least `min_same_scenes` SAME events, no FOREIGN event, no conflict
          or duplicate;
  mixed   a FOREIGN event, or the person's labels read different dominant numbers: another
          player is in it;
  doubt   anything else: nothing in the data confirms that the label is one player.
A person is printed (`decide`) when one of these holds (since 2026-11-23):
  number   the verdict is `ok`: only the SAME scenes are used (pictures, counts, the timeline),
           and the timeline's on-screen time is limited to the spans where the dominant number
           is read;
  checked  every camera mapping the person's text stands on was decided by a person on
           「人を確かめる」 (the step's 「これで決定して次へ」 / 「参加者の特定を確認」, recorded in the
           group's meta PEOPLE_CHECKED, or a mapping saved by a person: source "teacher"), and the
           person is in it: every scene, as in an approved work (no scene is checked one by one;
           a mix-up is corrected with 「この文は間違い」, decided 2026-11-23);
  no_clue  the work has no clue for anybody (`work_has_clue`: no evaluated person's number is
           read, and nobody decided a mapping on 「人を確かめる」): everyone is printed with the
           usual 「あとから訂正」 marks, every scene used, and step 5 suggests 「人を確かめる」.
Anyone else is not printed (the reason says why).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pfb_app.domain.models import Stage
from pfb_app.pipeline import identity_check as ident
from pfb_app.pipeline.common import usable

OK, DOUBT, MIXED = "ok", "doubt", "mixed"
WORDS = {OK: "同一人物と確かめられた", DOUBT: "疑わしい（確認が取れていない）", MIXED: "別人混入が確か"}
NUMBER, CHECKED, NO_CLUE = "number", "checked", "no_clue"     # how a printed person passed
PEOPLE_CHECKED = "people_checked"   # submissions.meta: identity run ids a person decided on 「人を確かめる」
CHECK_HINT = "人を確かめると、取り違えを防げます"


@dataclass(frozen=True)
class Policy:
    min_reads: int = 3          # reads of the dominant number
    min_share: float = 0.6      # of the numbered reads
    min_same_scenes: int = 3    # events of the person read as SAME
    allow_foreign: int = 0


def policy(settings=None) -> Policy:
    over = getattr(settings, "identity_gate", None) or {}
    return Policy(**{k: v for k, v in over.items() if k in Policy.__dataclass_fields__})


@dataclass
class Verdict:
    state: str
    reasons: list[str]
    facts: dict
    same_refs: set[str] = field(default_factory=set)
    ranges: dict[str, list[tuple[int, int]]] = field(default_factory=dict)   # camera -> frames read as the dominant number
    foreign_refs: set[str] = field(default_factory=set)   # events where another number is read

    def restrict(self, camera: str, frames: np.ndarray) -> np.ndarray:
        """The frames inside the spans where the dominant number is read."""
        spans = self.ranges.get(camera) or []
        if not len(frames) or not spans:
            return frames[:0]
        keep = np.zeros(len(frames), dtype=bool)
        for a, b in spans:
            keep |= (frames >= a) & (frames <= b)
        return frames[keep]


def assess(repo, sub, subject: dict, chain, pol: Policy | None = None) -> Verdict:
    from pfb_app.services import auto_people   # noqa: PLC0415
    pol = pol or policy(repo.settings)
    pid = subject.get("participant_id") or ""
    reads_dom = numbered = labels_n = foreign = same = unknown = 0
    doms: set[str] = set()
    duplicate = False
    same_refs: set[str] = set()
    foreign_refs: set[str] = set()
    ranges: dict[str, list[tuple[int, int]]] = {}
    seen: set[int] = set()
    for _con, source in chain:
        if source.id in seen or not source.camera:
            continue
        seen.add(source.id)
        cam = sub.camera(source.camera)
        states = ident.event_states(repo, source)
        for e in (source.output or {}).get("events") or []:
            if e.get("participant") != pid:
                continue
            st = states.get(e.get("event_id"), ident.UNKNOWN)
            if st == ident.SAME:
                same += 1
                same_refs.add(e.get("event_id"))
            elif st == ident.FOREIGN:
                foreign += 1
                foreign_refs.add(e.get("event_id"))
            else:
                unknown += 1
        if cam is None or cam.track is None:
            continue
        run = auto_people._number_run(repo, sub.sid, source.camera, cam.track)
        reads = ((run.output or {}).get("cues") or {}).get("number") if run else None
        mine = cam.labels_of(pid)
        labels_n += len(mine)
        for lb in mine:
            r = (reads or {}).get(lb)
            if not isinstance(r, dict) or r.get("dominant") is None:
                continue
            dom = str(r["dominant"])
            doms.add(dom)
            tl = [str(n) for _f, n in (r.get("timeline") or []) if n is not None]
            numbered += len(tl)
            reads_dom += sum(1 for n in tl if n == dom)
            for seg in r.get("segments") or []:
                if str(seg[2]) == dom:
                    ranges.setdefault(source.camera, []).append((int(seg[0]), int(seg[1])))
            for other_lb, o in (reads or {}).items():
                if other_lb not in mine and isinstance(o, dict) and str(o.get("dominant")) == dom:
                    duplicate = True
    share = reads_dom / numbered if numbered else 0.0
    facts = {"labels": labels_n, "dominant": sorted(doms), "reads": reads_dom, "numbered": numbered,
             "share": round(share, 2), "scenes_same": same, "scenes_foreign": foreign,
             "scenes_unknown": unknown, "duplicate": duplicate}
    reasons: list[str] = []
    if foreign > pol.allow_foreign:
        state = MIXED
        reasons.append(f"別の人かもしれない場面が{foreign}件あります")
    elif len(doms) > 1:
        state = MIXED
        reasons.append("同じ人に、ちがう背番号が読み取られています")
    else:
        state = OK
        if not doms:
            state = DOUBT
            reasons.append("背番号が読めておらず、同じ人と確かめる手がかりがありません")
        else:
            if reads_dom < pol.min_reads or share < pol.min_share:
                state = DOUBT
                reasons.append(f"背番号の読み取りが{numbered}回中{reads_dom}回で、確かめるには足りません")
            if duplicate:
                state = DOUBT
                reasons.append("同じ背番号を読まれた人がほかにもいます")
            if same < pol.min_same_scenes:
                state = DOUBT
                reasons.append(f"同じ人と確かめられた場面が{same}件で、必要な{pol.min_same_scenes}件に足りません")
    return Verdict(state, reasons, facts, same_refs, ranges, foreign_refs)


# --- who is printed: the number, a person's check on 「人を確かめる」, or no clue at all -----------

def checked_ids(submission: dict) -> set[int]:
    """The identity runs a person decided on 「人を確かめる」 (review.mark_people_checked)."""
    raw = (submission.get("meta") or {}).get(PEOPLE_CHECKED) or []
    return {int(i) for i in raw if isinstance(i, int) or (isinstance(i, str) and i.isdigit())}


def _decided(run, checked: set[int]) -> bool:
    return usable(run) and (run.source == "teacher" or run.id in checked)


def _mapped(output: dict | None) -> set[str]:
    return {p.get("participant_id") for p in (output or {}).get("participants") or []
            if isinstance(p, dict) and p.get("participant_id")}


def checked_by_person(repo, submission: dict, subject: dict, chain) -> bool:
    """Every camera mapping the person's text stands on (the identity run of each behavior run
    with an event of theirs) was decided by a person on 「人を確かめる」, and maps them."""
    pid = subject.get("participant_id")
    if not pid:
        return False
    checked = checked_ids(submission)
    seen = False
    for _con, source in chain:
        if not source.camera or not any(e.get("participant") == pid
                                        for e in (source.output or {}).get("events") or []):
            continue
        ups = [repo.get_run(i) for i in source.upstream_run_ids or []]
        identity = next((r for r in ups if r is not None and Stage(r.stage) == Stage.identity),
                        None)
        if identity is None or not _decided(identity, checked) or pid not in _mapped(identity.output):
            return False
        seen = True
    return seen


def work_has_clue(repo, work: dict) -> bool:
    """Something in the work can tell people apart: an evaluated person's number is read (the
    names stage's number cues on a label mapped to them), or a person decided a camera's mapping
    on 「人を確かめる」. False only when there is no clue for anybody (the no_clue case)."""
    from pfb_app.pipeline.auto_people import is_evaluated   # noqa: PLC0415
    from pfb_app.pipeline.tracked_identity import label_mapping   # noqa: PLC0415
    from pfb_app.services import auto_people   # noqa: PLC0415
    for submission in repo.list_submissions(work["id"]):
        sid = submission["id"]
        checked = checked_ids(submission)
        pids = {s["participant_id"] for s in repo.list_subjects(sid)
                if is_evaluated(s) and s.get("participant_id")}
        cams = sorted(set((submission.get("meta") or {}).get("cameras") or {})
                      | set(repo.list_cameras(sid)))
        for cam in cams:
            identity = repo.latest_run(sid, Stage.identity, cam)
            if _decided(identity, checked):
                return True
            track = repo.latest_run(sid, Stage.track, cam)
            if not usable(identity) or not usable(track):
                continue
            try:
                mapping = label_mapping(identity.output)
            except (KeyError, TypeError, ValueError):
                continue
            run = auto_people._number_run(repo, sid, cam, track)   # noqa: SLF001
            reads = ((run.output or {}).get("cues") or {}).get("number") if run else None
            for label, read in (reads or {}).items():
                if (mapping.get(label) in pids and isinstance(read, dict)
                        and read.get("dominant") is not None):
                    return True
    return False


@dataclass
class Decision:
    passed: bool
    how: str                        # NUMBER, CHECKED, NO_CLUE ("" when not printed)
    verdict: Verdict
    keep: set[str] | None = None    # the scenes to use (None: all of them)

    @property
    def restrict(self) -> bool:
        """The on-screen time is limited to the spans of the read number."""
        return self.how == NUMBER


def decide(repo, submission: dict, subject: dict, chain, verdict: Verdict,
           clue: bool) -> Decision:
    """Whether the person is printed, and with which scenes (see the module's text)."""
    if verdict.state == OK:
        return Decision(True, NUMBER, verdict, set(verdict.same_refs))
    if checked_by_person(repo, submission, subject, chain):
        return Decision(True, CHECKED, verdict, None)    # every scene, as an approved work
    if not clue:
        return Decision(True, NO_CLUE, verdict, None)
    return Decision(False, "", verdict)
