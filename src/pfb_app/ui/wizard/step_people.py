"""Step 3, 人を確かめる（任意） (tracking method) / 参加者を確認 (paper method).

For the tracking method the step is optional: the people are chosen and confirmed
automatically (services.auto_people) and the work moves on without it; the mapping grid stays
here for a coach who wants to change who is evaluated.

One group at a time (「このグループへ」 switches when the work has several), and for the
tracking method one camera at a time (a camera switcher when the group has several):

- While the camera is being read or tracked: the waiting state 「人を見つけています（カメラ1）」
  and the progress panel (progress_panel: the overall bar, the stages, the time left),
  redrawn every 10 seconds (live.watch); the page reruns by itself when the sheet is ready
  (and at least every 10 points of the percentage, but never while a mapping form is shown).
- Then the mapping grid (people_grid): the team question, 「相手チームも評価する」 (sports), a
  filter, 「残りを対象外にする」 and one card per person (a student, a new person or 対象外),
  filled from the saved mapping, the names stage's proposals (推定, with the reason), the
  initial-position rule or the team defaults. 「これで決定して次へ」 saves the camera's mapping
  (`save_tracked_identity`, which makes the new people first; skipped when nothing changed)
  and moves to the next camera; on the last one it also confirms the group
  (`confirm_identity`) and the screen follows the derived step, usually 解析を待つ.
- A names job (the proposals) never holds the coach up: the choices can be made and decided
  meanwhile, and the screen fills in the proposals when they arrive (live.watch), without
  touching what the coach already chose.
- Changing a mapping after the analysis has results (or approvals) shows a warning, and the
  button stays disabled until the teacher ticks the confirmation.

The paper method shows the identity results read-only and 「参加者の特定を確認」
(`confirm_identity`); corrections go through the admin pages (管理用の画面).

A stopped group (nothing to choose from) is titled 「解析が止まりました」: its reason, then
「もう一度ためす」 as the only primary when the derivation offers it (step_wait.retry_button),
and the quiet link 「くわしく見る（管理用の画面）」.
"""
import hashlib
import json
from dataclasses import dataclass
from html import escape

import streamlit as st
from pydantic import ValidationError

from pfb_app.domain.models import Confidence, Stage
from pfb_app.domain.vocab import mark_name
from pfb_app.pipeline.common import submission_engine, usable
from pfb_app.pipeline.tracked_identity import OUTSIDE, ROSTER_RESET, roster_was_reset
from pfb_app.services import review
from pfb_app.services.review import ApprovalError, BusyError
from pfb_app.storage.db import Repo
from pfb_app.ui.common import (MappingForm, flash, mapping_form, mapping_from_choices,
                               validation_text)
from pfb_app.ui.wizard import live, people_grid, progress_panel, router, step_wait, theme
from pfb_app.ui.wizard.progress import SubProgress, View, camera_label, job_line, plain_service

DECIDE = "これで決定して次へ"
CONFIRM = "参加者の特定を確認"
MAP_TITLE = "映っている人が誰か選んでください"
MAP_GUIDE = "写真の番号ごとに参加者の名前を選び、分からない人は「対象外」にします。"
SUGGESTED = "推定した名前が入っています。確かめてから決定してください。"
RETRACKED = "人を見つけ直したため、新しい写真でもう一度選んでください。"
REDO_WARNING = ("このつなぎ方を変えると、「したことの書き出し」から先の解析がやり直しになります。"
                "承認した参加者も、承認し直しになります。")
REDO_CHECK = "解析がやり直しになり、承認し直しになることを確かめました"
BUSY_NOTE = "解析が進んでいる間は決定できません。終わると押せるようになります。"
STOPPED_GUIDE = "止まった理由を読んで、「もう一度ためす」か管理用の画面で直してください。"
BUSY = "いまは解析が進んでいるため決定できませんでした。終わってから、もう一度押してください。"
UNMAPPED = "{names}に人物が割り当てられていません。このままだと「場面が少ない」になります。"
UNMAPPED_CHECK = "このまま進めます（割り当てのない参加者は「場面が少ない」になります）"
SAVED_NOT_CONFIRMED = "{camera}のつなぎ方は保存しましたが、確定できませんでした: {why}"
UNSET_PEOPLE = "まだ選んでいない人がいます: "
REMOVED_NAMED = "使われなかった名前を{n}件取り除きました。"
CONFIDENCE = {Confidence.high.value: "高い", Confidence.medium.value: "ふつう",
              Confidence.low.value: "低い"}

_NEXT_CAMERA = "_pfb_next_camera"   # {submission id: camera} to select on the next run
_CHOICES = people_grid.CHOICES      # {(track run id, identity run id): {label: choice}}, kept
                                    # across cameras; a mapping saved elsewhere starts afresh

TRACKING_HELP = [
    "この段階は任意です。映っている人は自動で選ばれ、そのまま解析が進みます。"
    "評価する人を変えたいときだけ、ここで選び直します。",
    "写真の番号ごとに、その人の名前を選びます。評価しない人（指導する人や見学者など）は「対象外」にします。",
    "「推定」は名簿などから自動で入れた名前です。理由を見て、違うときは選び直します。",
    "名簿にない人は「名前をつける」で名前を入れられます。残りは「残りを対象外にする」でまとめて対象外にできます。",
    "カメラが複数あるときは、カメラごとに決めます。最後のカメラで解析が先へ進みます。",
    "決めたあとで直すと、解析がやり直しになります。承認した参加者も承認し直しになります。",
    "うまくいかないとき：写真がない、写っている人が足りないときは、管理用の画面で"
    "人を見つけるところからやり直します。",
]
PAPER_HELP = [
    "解析が見分けた参加者の一覧です。名前と見た目が合っているか確かめます。",
    "合っていれば「参加者の特定を確認」を押します。解析が先へ進みます。",
    "うまくいかないとき：名前と見た目が違うところは、「管理用の画面で直す」から直してから確認します。",
]


# What a track run's warnings say, in the screen's words (never the run's own text, which can
# carry an engine's message): (the start of the warning, the sentence); anything else is not shown.
TRACK_NOTES = (
    ("開始直後に全員が写っている画面が見つからない",
     "全員が同時に映る場面がなかったため、写真に写っていない人がいます。"),
    ("人の写り方を調べられなかった", "人の写り方を調べられなかったため、映っている長さで選びました。"),
    ("音声から発言が見つからなかった", "音声に聞き取れる話し声がなかったため、映像だけで観察します。"),
    ("文字起こしに失敗した", "音声を文字にできなかったため、映像だけで観察します。"),
    ("貼り付けた文字起こしに発言がない", "貼り付けた文字起こしに発言がないため、映像だけで観察します。"),
)


def track_notes(warnings: list[str]) -> list[str]:
    """The plain sentences for a track run's warnings, once each, in order."""
    out: list[str] = []
    for w in warnings or []:
        line = next((text for key, text in TRACK_NOTES if str(w).startswith(key)), None)
        if line is not None and line not in out:
            out.append(line)
    return out


# --- the cameras of a tracking group ---------------------------------------------------------

@dataclass
class Camera:
    camera: str
    busy: list[dict]            # active jobs reading or tracking this camera
    form: MappingForm | None    # None while there is no usable track result

    @property
    def label(self) -> str:
        return camera_label(self.camera)

    @property
    def decided(self) -> bool:
        return self.form is not None and self.form.current


def cameras(repo: Repo, sub: dict) -> list[Camera]:
    active = [r for r in live.active_jobs(repo, sub["id"]) if r["stage"] in live.FRONT_STAGES]
    out = []
    for cam in sub["meta"].get("cameras") or {}:
        busy = [r for r in active if r["camera"] in (cam, None)]
        out.append(Camera(cam, busy, None if busy else mapping_form(repo, sub["id"], cam)))
    return out


def _camera_word(c: Camera) -> tuple[str, str]:
    if c.busy:
        return "追跡中", "processing"
    if c.form is None:
        return "結果がありません", "stopped"
    return ("決定済み", "done") if c.decided else ("未決定", "your_turn")


def _busy_line(rows: list[dict], kind: str) -> str:
    rows = sorted(rows, key=lambda r: (r["status"] != "running", r["id"]))
    r = rows[0]
    return job_line(r["stage"], r["status"], r["camera"], kind)


def _pick_camera(sid: int, cams: list[Camera]) -> Camera:
    """The camera switcher; it starts on the first camera still to decide (or being
    tracked), and after a decision moves on to the next one."""
    key = f"pfb_cam_{sid}"
    names = [c.camera for c in cams]
    pending = st.session_state.get(_NEXT_CAMERA, {}).pop(sid, None)
    if pending in names:
        st.session_state[key] = pending
    elif st.session_state.get(key) not in names:
        st.session_state[key] = next(
            (c.camera for c in cams if c.form is not None and not c.decided),
            next((c.camera for c in cams if c.busy), names[0]))
    chosen = st.radio(theme.L("カメラ"), names, format_func=camera_label, horizontal=True, key=key)
    states = "".join(f'<li class="pfb-camstate"><span class="pfb-camstate__name">{escape(c.label)}'
                     f'</span>{theme.status_chip(chip, word)}</li>'
                     for c in cams for word, chip in [_camera_word(c)])
    st.html(f'<ul class="pfb-camstates" aria-label="カメラごとの様子">{states}</ul>')
    return cams[names.index(chosen)]


# --- the live part ---------------------------------------------------------------------------

def _signature(aid: int, sid: int, form: bool):
    """What reruns the page. With a form on screen only this group's cameras count (a track
    finished or started again), so unsaved choices are never thrown away because another
    group moved on."""
    def signature(view: View) -> tuple:
        repo = view.repo
        cams = tuple((c.camera, bool(c.busy), c.form.track.id if c.form else None,
                      _names_state(repo, sid, c))
                     for c in cameras(repo, repo.get_submission(sid)))
        if form:
            return (cams,)
        progress = view.progress
        return (router.default_step(progress),
                tuple((s.submission_id, s.step, s.status) for s in progress.subs), cams,
                progress_panel.bucket(progress))
    return signature


def _names_state(repo: Repo, sid: int, c: "Camera") -> tuple | None:
    """The names stage of a camera with a form: a job waiting or running, and its newest run
    (when it finishes, the page reruns and the proposals fill the people not yet chosen)."""
    if c.form is None:
        return None
    run = repo.latest_run(sid, Stage.names, c.camera)
    return c.form.names_pending, run.id if run is not None else None


def _live(aid: int, sid: int, settings, kind: str):
    def draw(view: View) -> None:
        repo = view.repo
        sub = repo.get_submission(sid)
        rows = [r for r in live.active_jobs(repo, sid) if r["stage"] in live.FRONT_STAGES]
        lines = []
        for cam in sub["meta"].get("cameras") or {}:
            mine = [r for r in rows if r["camera"] in (cam, None)]
            if mine:
                status = "waiting_gpu" if all(r["status"] == "waiting_gpu" for r in mine) \
                    else "processing"
                lines.append(theme.status_chip(status, _busy_line(mine, kind)))
        if lines:
            items = "".join(f"<li>{line}</li>" for line in lines)
            st.html(f'<ul class="pfb-livelines" aria-live="polite">{items}</ul>')
        progress_panel.draw(view, settings)
    return draw


def _watch(screen: router.Screen, sub: dict, form: bool = False) -> None:
    aid = screen.progress.assignment_id
    live.watch("people", screen.view, _signature(aid, sub["id"], form),
               _live(aid, sub["id"], screen.settings, screen.progress.kind))


def _stopped(screen: router.Screen, p: SubProgress, primary: bool = False) -> None:
    """Why the group has stopped, 「もう一度ためす」 when it can be retried (the screen's primary
    when nothing else can be done here), and the quiet link to the admin pages."""
    st.error(theme.L(p.reason))
    if p.action == "retry":
        step_wait.retry_button(screen.repo, screen.progress.assignment_id,
                               f"pfb_people_retry_{p.submission_id}", primary=primary,
                               disabled=screen.pending)
    router.detail_link("status", router.DETAILS)


# --- tracking: the mapping form --------------------------------------------------------------

def _results_exist(repo: Repo, sub: dict) -> bool:
    """The group was confirmed before, or has analysis results after the mapping."""
    if sub["identity_confirmed"]:
        return True
    return any(repo.latest_run(sub["id"], Stage.behavior, cam) is not None
               for cam in sub["meta"].get("cameras") or {})


def _approved(repo: Repo, sid: int) -> int:
    return sum(1 for s in repo.list_subjects(sid) if s["approval_status"] == "approved")


def _choices_key(form: MappingForm) -> tuple[int, int | None]:
    return people_grid.choices_key(form)


def _form(screen: router.Screen, sub: dict, cam: Camera,
          cams: list[Camera]) -> dict[str, str | None]:
    """The team question, the roster, the filter and the bulk action, then the grid of
    people; returns the choices of every person."""
    form, repo, sid, meta = cam.form, screen.repo, sub["id"], sub["meta"] or {}
    if form.identity is not None and not form.current:
        st.warning(theme.L(ROSTER_RESET if roster_was_reset(meta, form.identity)
                           else RETRACKED))
    elif form.suggested:
        st.caption(theme.L(SUGGESTED))
    if form.names_pending:
        st.caption(theme.L(people_grid.NAMES_PENDING))
    people_grid.team_question(form, sid, cam.camera, repo=repo, meta=meta)
    people_grid.opponents_toggle(form, sid, screen.progress.assignment_id, repo=repo)
    assignment = screen.snap.get_assignment(screen.progress.assignment_id)
    people_grid.roster_expander(repo, sid, form.domain, assignment.get("sport"),
                                decided=any(c.decided for c in cams))
    values = people_grid.current(form)
    if people_grid.bulk_outside(form, values):    # drawn before the filter counts are made
        values = people_grid.current(form)
    filt = people_grid.filter_bar(f"pfb_filter_{form.track.id}",
                                  people_grid.counts_of(form, values))
    people_grid.sheet(form, cam.label)
    for note in track_notes(form.track.warnings):
        st.caption(theme.L(note))
    return people_grid.person_grid(form, filt, repo=repo, sid=sid, values=values)


def _rest(cams: list[Camera], cam: Camera, results: bool) -> list[Camera]:
    """The cameras to go through after `cam`, in switcher order. On a first mapping only the
    undecided ones; once there are results every later camera too, so that the group is
    confirmed (and analysed again) once, after the last camera."""
    i = next(n for n, c in enumerate(cams) if c.camera == cam.camera)
    return ([c for c in cams[i + 1:] if results or not c.decided]
            + [c for c in cams[:i] if not c.decided])


def _unmapped(cams: list[Camera], cam: Camera, chosen: dict[str, str | None],
              final: bool, own: set[str]) -> list[str]:
    """The work's own people (`own`: the roster or the registered students; not the people
    made on the mapping screen) no person is chosen for: on this camera, or on the last one
    across all."""
    mapped = {v for v in chosen.values() if v and v != OUTSIDE}
    if final:
        for c in cams:
            if c.camera != cam.camera and c.decided:
                mapped |= {v for v in c.form.saved.values() if v != OUTSIDE}
    return [name for pid, name in cam.form.students.items()
            if pid in own and pid not in mapped]


def choices_key(chosen: dict[str, str | None]) -> str:
    """A short digest of the choices on screen: a confirmation given for other choices does
    not carry over."""
    raw = json.dumps(sorted(chosen.items()), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _decide(screen: router.Screen, sub: dict, cam: Camera, rest: list[Camera],
            chosen: dict[str, str | None], changed: bool) -> None:
    repo, sid = screen.repo, sub["id"]
    mapping, unset = mapping_from_choices(chosen)
    if unset:
        st.error(UNSET_PEOPLE + "、".join(mark_name(label) for label in unset))
        return
    saved = False
    try:
        if changed:
            review.save_tracked_identity(repo, sid, cam.camera, mapping)
            saved = True
        people_grid.drop_kept(cam.form)                                      # now saved
        if rest:
            st.session_state.setdefault(_NEXT_CAMERA, {})[sid] = rest[0].camera
            flash(f"{cam.label}のつなぎ方を決定しました。次は{rest[0].label}です。", raw=True)
        else:
            done = review.confirm_identity(repo, sid)
            removed = getattr(done, "removed_named", 0)
            flash(f"{sub['label']}の人物のつなぎ方を決定しました。解析を続けます。"
                  + (REMOVED_NAMED.format(n=removed) if removed else ""), raw=True)
            router.go(None)
            router.state()["submission_id"] = None
    except (BusyError, ApprovalError) as e:
        why = theme.L(BUSY) if isinstance(e, BusyError) else people_grid.marks(
            camera_label(theme.L(plain_service(str(e)))))
        if not saved:
            st.error(why)
            return
        # the saved camera now shows as decided: say why after the rerun
        flash(SAVED_NOT_CONFIRMED.format(camera=cam.label, why=why), "error", raw=True)
    except ValidationError as e:
        st.error(people_grid.marks(camera_label(plain_service(validation_text(e)))))
        return
    st.rerun()


def _tracking(screen: router.Screen, p: SubProgress, sub: dict) -> None:
    repo = screen.snap      # reads; the decision writes through screen.repo
    if people_grid.sync_opponents(screen.repo, repo, screen.progress.assignment_id, sub):
        st.rerun()          # a group added after 「相手チームも評価する」 was turned on
    cams = cameras(repo, sub)
    if not cams:
        with theme.step_card(step_wait.STOPPED_TITLE, STOPPED_GUIDE):
            _stopped(screen, p, primary=True)
        theme.help_expander(TRACKING_HELP)
        return
    cam = _pick_camera(sub["id"], cams) if len(cams) > 1 else cams[0]
    moving = p.status in ("processing", "waiting_gpu")
    if cam.form is None:
        tracking = bool(cam.busy) or moving
        stopped = not tracking and p.status == "stopped"
        if tracking:
            title, guide = "人を見つけています", ("見つけ終わると、この画面は写真を見て参加者と"
                                                 "つなぐ画面に自動で切り替わります。")
        elif stopped:
            title, guide = step_wait.STOPPED_TITLE, STOPPED_GUIDE
        else:
            title, guide = MAP_TITLE, MAP_GUIDE
        with theme.step_card(title, guide):
            if tracking:
                _watch(screen, sub)
            else:
                _stopped(screen, p, primary=stopped)
            if not stopped:    # a stopped group has nothing to decide: no disabled button
                st.button(theme.L(DECIDE), key="pfb_people_decide", type="primary",
                          disabled=True)
            if tracking:
                st.caption(live.NOTHING_TO_PRESS)
        theme.help_expander(TRACKING_HELP)
        return
    if router.state()["step"] is None:
        router.go(screen.step)    # keep this screen while choices may be unsaved
    with theme.step_card(MAP_TITLE, MAP_GUIDE, key="people"):
        if p.status == "stopped":     # e.g. another camera's tracking failed: say why first
            _stopped(screen, p)
        chosen = _form(screen, sub, cam, cams)
        changed = not cam.form.current or chosen != cam.form.saved
        results = _results_exist(repo, sub)
        rest = _rest(cams, cam, results)
        busy = repo.has_blocking_jobs(sub["id"])   # a names job never holds the coach up
        if busy or cam.form.names_pending:         # proposals fill in when they arrive
            _watch(screen, sub, form=True)
        if busy:
            st.caption(theme.L(BUSY_NOTE))
        checks = []
        if changed and results:
            approved = _approved(repo, sub["id"])
            st.warning(theme.L(REDO_WARNING + (f"（いま承認済みの参加者: {approved}人）" if approved else "")))
            checks.append(st.checkbox(theme.L(REDO_CHECK), key=f"pfb_redo_{cam.form.track.id}"))
        own = {s["participant_id"] for s in review.roster_subjects(
            repo.list_subjects(sub["id"]), sub["meta"])}
        unmapped = ([] if None in chosen.values()
                    else _unmapped(cams, cam, chosen, final=not rest, own=own))
        if unmapped:
            st.warning(theme.L(UNMAPPED).format(names="、".join(f"{n}さん" for n in unmapped)))
            checks.append(st.checkbox(
                theme.L(UNMAPPED_CHECK), key=f"pfb_unmapped_{cam.form.track.id}_{choices_key(chosen)}"))
        if st.button(theme.L(DECIDE), key="pfb_people_decide", type="primary",
                     disabled=busy or not all(checks), width="stretch"):
            _decide(screen, sub, cam, rest, chosen, changed)
    theme.help_expander(TRACKING_HELP)


# --- paper: the identity results -------------------------------------------------------------

def _identity_rows(repo: Repo, sub: dict, camera: str) -> list[dict] | None:
    run = repo.latest_run(sub["id"], Stage.identity, camera)
    if not usable(run):
        return None
    names = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sub["id"])
             if s["participant_id"]}
    return [{theme.L("参加者"): names.get(r.get("participant_id"), r.get("participant_id")),
             "見た目": r.get("appearance", ""), "初期位置": r.get("initial_position", ""),
             "確からしさ": CONFIDENCE.get(r.get("identity_confidence"), ""),
             "確かめた点": "、".join(r.get("verification_points") or [])}
            for r in run.output.get("participants") or []]


def _paper(screen: router.Screen, p: SubProgress, sub: dict) -> None:
    repo = screen.snap
    cams = list(sub["meta"].get("cameras") or {})
    busy = [r for r in live.active_jobs(repo, sub["id"]) if r["stage"] in live.FRONT_STAGES]
    if busy or (p.step == "people" and p.status in ("processing", "waiting_gpu")):
        with theme.step_card("参加者を特定しています",
                             "特定が終わると、この画面は結果を確かめる画面に自動で切り替わります。"):
            _watch(screen, sub)
            st.button(theme.L(CONFIRM), key="pfb_people_confirm", type="primary", disabled=True)
            st.caption(live.NOTHING_TO_PRESS)
        theme.help_expander(PAPER_HELP)
        return
    tables = {cam: _identity_rows(repo, sub, cam) for cam in cams}
    missing = [cam for cam, rows in tables.items() if rows is None]
    if p.status == "stopped" and (missing or not cams):   # nothing to confirm yet
        with theme.step_card(step_wait.STOPPED_TITLE, STOPPED_GUIDE):
            _stopped(screen, p, primary=True)
        theme.help_expander(PAPER_HELP)
        return
    with theme.step_card("参加者を確かめてください",
                         "解析が見分けた参加者の名前と見た目が合っているか確かめます。"):
        if missing or not cams or p.status == "stopped":   # why, above the table
            _stopped(screen, p)
        for cam, rows in tables.items():
            if rows is not None:
                if len(cams) > 1:
                    st.html(f'<h3 class="pfb-subhead">{escape(camera_label(cam))}</h3>')
                st.dataframe(rows, hide_index=True)
        if sub["identity_confirmed"]:
            st.success(theme.L("参加者の特定は確認済みです。"))
        router.detail_link("review", "管理用の画面で直す")
        if st.button(theme.L(CONFIRM), key="pfb_people_confirm", type="primary",
                     disabled=bool(missing) or not cams or sub["identity_confirmed"],
                     width="stretch"):
            try:
                review.confirm_identity(screen.repo, sub["id"])
            except ApprovalError as e:
                st.error(camera_label(plain_service(str(e))))
            else:
                flash(f"{sub['label']}の参加者の特定を確認しました。解析を続けます。", raw=True)
                router.go(None)
                router.state()["submission_id"] = None
                st.rerun()
    theme.help_expander(PAPER_HELP)


def render(screen: router.Screen) -> None:
    shown = router.shown_group(screen, "people")
    router.group_picker(screen.progress.subs, shown)
    sub = screen.snap.get_submission(shown.submission_id)
    if submission_engine(sub) == "pfb3x3":
        _tracking(screen, shown, sub)
    else:
        _paper(screen, shown, sub)
