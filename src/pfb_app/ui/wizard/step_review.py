"""Step 5 (documents: 4), 確認して承認: one student at a time.

Video works go group by group (「このグループへ」 when there are several); a document work is
one list, one student per document. Within it:

- the counter 「承認 n / 全 m」 and a student switcher that starts on the first student to
  check, and moves on to the next one after each approval;
- the student's feedback text, large, and below it the evidence it cites: when, what the
  student did, why it counted, and up to three frame images per item (`frames_for`), 320 px
  wide, each with 「大きく見る」;
- 「承認する」 (`approve`), the one primary button. The hanamaru stamp is drawn once for the
  student just approved (it is still, not animated, from then on and under reduced motion);
- 「直す」 opens an editor with the text; while it is open the primary is 「直して承認」
  (`write_manual_pfb`, then `approve`), and 「やめる」 closes it. On a video work a person with
  too little evidence gets a short ready-made text (`short_text.offer`, chip 「短い文」) with
  「この文で承認する」 (`short_text.approve_short`, the primary for that person) and 「書き直す」,
  which opens the same editor with that text, and the tertiary 「この人は入れない」 (the people
  panel's `set_person_excluded`; 「評価に戻す」 under 「映っている人」 undoes it, for every
  method); the short text is never offered for
  「全員まとめて承認」. Documents with too little evidence get 「自分で書く」 (an empty editor);
- 「この人だけ作り直す」 (`regenerate_from` for this student's feedback). The student shows 解析中
  and the screen refreshes itself (live.watch) until the new text is there;
- when the analysis is not finished in a way only 管理用の画面 can fix (records left in English, or
  people not confirmed), a plain message and 「根拠を管理用の画面で直す」, which opens 確認・承認 with
  this student selected;
- when everyone is approved or has too little evidence (and no short text is waiting): a short
  line and 「続ける」 to PDFを受け取る (or to the next group to check);
- at the top, 「全員まとめて承認」 (`bulk_approve`) when two or more texts need no look (no
  warnings, enough evidence, no gate): the primary then, bound to the texts shown; the
  students it left are listed once with the plain reason. The work option 「確認せずに自動で
  承認する」 (`set_auto_approve`, off by default) sits below it; students the worker approved
  carry the 「自動で承認」 chip (word and mark) in the list, the switcher and their heading.
  A kept draft that is the same as the text it would replace does not count as editing;
- at the foot, 「映っている人」: the people panel for tracking groups; for the other groups
  only the people taken out (「この人は入れない」), each with 「評価に戻す」.

Refusals of the services (another tab, a running analysis) are shown as plain messages.
Every action keeps this step on screen (`router.go`), so the screen does not jump away when
the work's derived step moves on.
"""
from dataclasses import dataclass
from html import escape

import streamlit as st
from pydantic import ValidationError

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import submission_engine
from pfb_app.preprocess.frames import VideoError
from pfb_app.services import review, short_text
from pfb_app.services.auto_people import set_person_excluded, short_people, unused_roster
from pfb_app.services.content_check import add_next_part, analysed_part, next_part_group
from pfb_app.services.review import ApprovalError, BusyError
from pfb_app.storage.db import NON_BLOCKING_STAGES, Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import cited_evidence, flash, get_repo, validation_text
from pfb_app.ui.wizard import bars, live, router, step_tips, theme, zoom
from pfb_app.ui.wizard.people_panel import EXCLUDED, RESTORE, RESTORED, UNNAMED, people_panel
from pfb_app.ui.wizard.progress import REDO_CONTRIBUTION, View, contribution_redo_needed, job_line

APPROVE = "承認する"
FIX_APPROVE = "直して承認"
CONFIRM_FIX = "このまま直して承認"
CONTINUE = "続ける"
FIX = "直す"
WRITE = "自分で書く"
CANCEL = "やめる"
REDO = "この人だけ作り直す"
BULK = "全員まとめて承認"
AUTO_OPTION = "確認せずに自動で承認する"
AUTO_CHIP = "自動で承認"
GATE_LINK = "根拠を管理用の画面で直す"
USE_SHORT = "この文で承認する"
GROWTH_LINK, GROWTH_PAGE = "成長の記録を見る", "pages/growth.py"   # this person across works
REWRITE = "書き直す"
DECLINE = "この人は入れない"      # the people panel's 「この人は評価しない」 (set_person_excluded)
DECLINE_NOTE = "この人には文章を渡しません。あとでこの画面の下の「映っている人」から戻せます。"
SHORT_CHIP = short_text.MARK      # 「短い文」

TITLE = "一人ずつ確かめて承認してください"
PEOPLE = "映っている人"
SHORT_NOTE = ("映っている時間が短い人が{n}人いて、評価には入れていません。"
              "評価したい人は、下の「映っている人」から選べます。")
PART_ADDED = "つづきは、別のグループとして追加されています。"
PART_DONE = "つづきを別のグループとして追加しました。内容を確かめてください。"
GUIDE = "文章と、その根拠になった場面を見て、よければ「承認する」を押します。"
INSUFFICIENT = "映っている場面が少ないため、自動の文章は作られませんでした。"
INSUFFICIENT_MORE = "「自分で書く」から書いて承認できます。"
STALE = ("この文章を作ったあとで解析が更新されました。"
         "「この人だけ作り直す」か、「直す」で書き直して承認してください。")
FAILED = "文章を作れませんでした。「この人だけ作り直す」か、「自分で書く」で書いてください。"
PENDING = "まだ文章ができていません。解析が終わると、ここに出ます。"
REDOING = "新しい文章を作っています。できると、ここに出ます。この画面は自動で新しくなります。"
# per-person results as they finish: a first text not written yet (its job is queued/running)
WRITING = "この人の文章を作っています。できると、ここに出ます。この画面は自動で新しくなります。"
WRITING_LINE = "ほかの{n}人の文章を作っています。できた人から、ここで確かめられます。"
SKIP_WRITING = "まだ文章を作っています。"
ENGLISH_GATE = ("したことの書き出しの一部が英語のままのため、まだ承認できません。"
                "管理用の画面で根拠を直してください。")
IDENTITY_GATE = ("映っている人と参加者のつなぎ方がまだ確定していないため、まだ承認できません。"
                 "管理用の画面で確かめてください。")
BUSY = "いまは解析が進んでいるため、できませんでした。終わってから、もう一度押してください。"
EMPTY = "文章が空です。書いてから押してください。"
TEACHER_TEXT = "あなたが書いた文章です。"
EDIT_NOTE = "あなたが書いた文章として保存し、そのまま承認します。"
NOBODY = "確認する参加者がいません。"
NEWER = "文章が新しくなりました。確かめてから押してください。"
BULK_LINE = "気になる点がなく根拠が足りている{n}人の文章を、まとめて承認できます。"
BULK_DONE = "{n}人を承認しました。"
BULK_NONE = "まとめて承認できる人はいませんでした。"
BULK_LEFT = "まとめて承認しなかった人（一人ずつ確かめてください）"
AUTO_HELP = ("オンにすると、これから作られる文章のうち、気になる点がなく根拠が足りているものは、"
             "確認せずに承認されます。気になる点がある文章は、この画面で確かめます。")
AUTO_LINE = "確認せずに自動で承認した人"
SKIP_EDITING = "文章を直している途中です。"
SHORT_CAPTION = "活動の名前と、動画に映っていたことだけから作った短い文です。"

# the categories of review.checks' text warnings, in plain words
WARNING_TEXT = (
    ("内部ID", "文章に記号や番号のような呼び方が入っています。"),
    ("カメラ名", "文章にカメラの名前が入っています。"),
    ("人物の番号", "文章に人物の番号が入っています。"),
    ("色の語", "色で人を表す言葉が入っています。"),
    ("確信度", "文章に確かさの度合い（数字や割合）が入っています。"),
    ("改善指導", "直すところを指示する言い方が入っています。良かった点を伝える文章にします。"),
)
WARNING_OTHER = "文章に気になる点があります。"
WARNING_FIX = "参加者に渡す前に「直す」で直してください。"
WARNING_CONFIRM = "このままでよければ、もう一度「このまま直して承認」を押してください。"

HELP_FIX = "言い回しを変えたいときは「直す」で書き直して「直して承認」を押します。"
HELP_WRITE = "根拠が足りない参加者は「自分で書く」で書けます。"          # documents
HELP_SHORT = ("映っている場面が少ない人には「短い文」を用意します。「この文で承認する」で承認するか、"
              "「書き直す」で書き直します。文章を渡さない人は「この人は入れない」を押します。"
              "短い文は「全員まとめて承認」には入りません。")   # videos
HELP = [
    "一人ずつ、文章と根拠の場面を見て、よければ「承認する」を押します。次の参加者へ進みます。",
    HELP_FIX,
    "気になる点のない文章が二人分以上あるときは「全員まとめて承認」で一度に承認できます。"
    "まとめて承認しなかった人は、理由と一緒に出ます。",
    "「確認せずに自動で承認する」をオンにすると、これから作られる気になる点のない文章は自動で承認され、"
    "「自動で承認」の印がつきます。",
    "全員が済むと「PDFを受け取る」へ進めます。",
    "うまくいかないとき：根拠がおかしいときは「この人だけ作り直す」か「根拠を管理用の画面で直す」を使います。",
]


def help_lines(kind: Kind) -> list[str]:
    """「この画面の使い方」 for the work: a document's thin text is written by hand
    (「自分で書く」), a video's gets a short text (「短い文」)."""
    extra = HELP_SHORT if kind == Kind.video else HELP_WRITE
    return [*HELP[:2], extra, *HELP[2:]]

EDITING = "_pfb_review_editing"   # subject id whose editor is open
STAMP = router.REVIEW_STAMP       # (subject id, name) approved on the last action: stamp once
_NEXT = "_pfb_review_next"        # {unit key: subject id or None (the first to check)}
_ACTION = router.REVIEW_ACTION    # (action, subject id, run id) accepted by a button callback
_WARNED = "_pfb_review_warned"    # (subject id, text) whose warnings the teacher has seen
_DRAFT = "_pfb_draft_{}"          # the editor's text per student, kept while switching
_EDIT = "pfb_edit_{}"             # the editor widget's key per student
_SHOWN_SHORT = "_pfb_review_short_shown"   # (subject id, run id, short text on screen) pressed

# a state: the word for it and the chip that repeats it
WORDS = {
    "approved": ("承認済み", "approved"),
    "review": ("確認できます", "your_turn"),
    "stale": ("内容が古くなっています", "your_turn"),
    "failed": ("作れませんでした", "stopped"),
    "insufficient": ("場面が少ない", "insufficient"),
    "regenerating": ("解析中", "processing"),
    "writing": ("文章を書いています", "processing"),   # its first text is being written
    "short": ("確認できます", "your_turn"),     # too thin, a short text offered (短い文)
    "pending": ("まだです", "insufficient"),
    "gated": ("管理用の画面で直す必要があります", "stopped"),
}
SETTLED = {"approved", "insufficient"}
_BUSY = {"regenerating", "writing"}   # a text on its way: the screen watches for it


# --- who is in the list, and where they stand --------------------------------------------------

@dataclass
class Student:
    subject: dict
    state: str              # a key of WORDS
    run: StageRun | None    # the latest feedback run
    job: dict | None        # its active feedback job (being written again)
    gate: str | None        # why only 管理用の画面 can help, or None
    short: str | None = None   # the short text offered (state "short"), else None

    @property
    def id(self) -> int:
        return self.subject["id"]

    @property
    def name(self) -> str:
        return self.subject["display_name"]

    @property
    def run_id(self) -> int | None:
        return self.run.id if self.run else None

    @property
    def text(self) -> str:
        return ((self.run.output or {}).get("pfb") or "") if self.run else ""

    @property
    def auto(self) -> bool:
        """Approved by the worker (確認せずに自動で承認する), not by a person."""
        return self.state == "approved" and bool(self.subject.get("approved_auto"))

    @property
    def short_run(self) -> bool:
        """Its latest text is a short text a person approved (「この文で承認する」)."""
        return short_text.is_short(self.run)

    @property
    def marked(self) -> bool:
        """It carries the 「短い文」 chip: a short text offered, or one approved."""
        return self.state == "short" or self.short_run

    @property
    def start(self) -> str:
        """What the editor starts from: the short text offered, nothing for a missing text,
        else the current text."""
        if self.state == "short":
            return self.short or ""
        return "" if self.state in ("insufficient", "failed") else self.text

    @property
    def bulk_ok(self) -> bool:
        """Its text can go with 「全員まとめて承認」: to check, no text warnings, not a short
        text (those are always a person's own press)."""
        return (self.state == "review" and not review.text_warnings(self.run)
                and not self.short_run)


@dataclass
class Unit:
    """What one screen walks through: a video group, or every document of the work."""
    key: str
    label: str
    submission_ids: list[int]
    students: list[Student]


def _gate(repo: Repo, sub: dict, kind: Kind) -> str | None:
    if kind != Kind.video:
        return None
    if not sub["identity_confirmed"]:
        return IDENTITY_GATE
    if review.untranslated_behavior(repo, sub["id"]):
        return ENGLISH_GATE
    return None


def _state(repo: Repo, subject: dict, run: StageRun | None, job: dict | None, busy: bool,
           gate: str | None) -> str:
    stale = run is not None and review.is_stale_chain(repo, run)
    if (subject["approval_status"] == "approved" and run is not None and not stale
            and subject["approved_run_id"] == run.id):
        return "approved"
    if job is not None and run is None and not busy:
        return "writing"
    if job is not None or busy:
        return "regenerating"
    if gate:
        return "gated"
    if run is None:
        return "pending"
    if run.status == "failed":
        return "failed"
    if stale:
        return "stale"
    if (run.output or {}).get("pfb_status") != "generated":
        return "insufficient"
    return "review"


def students_of(repo: Repo, sub: dict, kind: Kind) -> list[Student]:
    """The group's students and where each stands (`repo`: the page run's or the tick's
    snapshot). A text built on an outdated contribution is gated: making it again here would
    only give another outdated text."""
    jobs = live.latest_jobs(repo, sub["id"])
    active = [r for r in jobs.values() if r["status"] in live.ACTIVE]
    upstream_busy = any(r["stage"] != Stage.pfb.value and r["stage"] not in NON_BLOCKING_STAGES
                        for r in active)   # all will change (a names job is not upstream)
    gate = _gate(repo, sub, kind)
    out = []
    for subject in repo.list_subjects(sub["id"]):
        if kind == Kind.video and not is_evaluated(subject):
            continue
        job = jobs.get((Stage.pfb.value, None, subject["id"]))
        job = job if job is not None and job["status"] in live.ACTIVE else None
        run = repo.latest_run(sub["id"], Stage.pfb, subject_id=subject["id"])
        mine = gate or (REDO_CONTRIBUTION if run is not None and review.is_stale_chain(repo, run)
                        and contribution_redo_needed(repo, run) else None)
        state = _state(repo, subject, run, job, upstream_busy, mine)
        short = short_text.offer(repo, subject, run) if state == "insufficient" else None
        out.append(Student(subject, "short" if short else state, run, job, mine, short))
    return out


def _unit_students(repo: Repo, ids: list[int], kind: Kind) -> list[Student]:
    return [s for sid in ids for s in students_of(repo, repo.get_submission(sid), kind)]


def _unit(screen: router.Screen, kind: Kind) -> Unit:
    """The unit on screen; for videos with several groups the group picker comes first."""
    repo, progress = screen.snap, screen.progress
    if kind == Kind.video:
        shown = router.shown_group(screen, "review")
        router.group_picker(progress.subs, shown)
        ids = [shown.submission_id]
        return Unit(str(shown.submission_id), shown.label, ids,
                    _unit_students(repo, ids, kind))
    ids = [s.submission_id for s in progress.subs]
    return Unit("all", "", ids, _unit_students(repo, ids, kind))


def _first(students: list[Student]) -> Student:
    """The first student to check, else the first not approved, else the first."""
    return next((s for s in students if s.state == "review"),
                next((s for s in students if s.state != "approved"), students[0]))


def _todo(students: list[Student], but: int) -> Student | None:
    """The next student who needs the teacher (not settled, not being written again)."""
    rest = [s for s in students
            if s.id != but and s.state not in SETTLED | {"regenerating", "writing"}]
    return next((s for s in rest if s.state == "review"), rest[0] if rest else None)


# --- the parts of the screen -------------------------------------------------------------------

def _pin(screen: router.Screen, unit: Unit) -> None:
    """Keep this step and this group on screen after an action, whatever the derived step
    becomes (a finished group would otherwise give way to the next one at once)."""
    router.go(screen.step)
    if unit.key != "all":
        router.state()["submission_id"] = int(unit.key)


def _counter(unit: Unit) -> None:
    """「承認 n / 全 m」 with its bar (bars.approval_bar_html)."""
    done = sum(1 for s in unit.students if s.state == "approved")
    st.html(bars.approval_bar_html(done, len(unit.students), unit.label))


def _label(student: Student) -> str:
    word = AUTO_CHIP if student.auto else WORDS[student.state][0]
    return f"{student.name}（{word}・{SHORT_CHIP}）" if student.marked else f"{student.name}（{word}）"


def _switcher(unit: Unit) -> Student:
    key = f"pfb_student_{unit.key}"
    by_id = {s.id: s for s in unit.students}
    pending = st.session_state.get(_NEXT, {})
    if unit.key in pending:
        wanted = pending.pop(unit.key)
        st.session_state[key] = wanted if wanted in by_id else _first(unit.students).id
    elif st.session_state.get(key) not in by_id:
        st.session_state[key] = _first(unit.students).id
    if len(unit.students) == 1:
        return unit.students[0]
    ids = list(by_id)
    label = lambda i: _label(by_id[i])   # noqa: E731
    if len(ids) <= 6:
        chosen = st.radio(theme.L("参加者"), ids, format_func=label, horizontal=True, key=key)
    else:
        chosen = st.selectbox(theme.L("参加者"), ids, format_func=label, key=key)
    return by_id[chosen]


def _stamp_line() -> None:
    just = st.session_state.pop(STAMP, None)
    if just:
        st.html(f'<div class="pfb-stamped">{theme.approval_stamp(animate=True)}'
                f'<span class="pfb-stamped__text">{escape(just[1])}さんを承認しました。</span></div>')


def _head(student: Student) -> None:
    word, chip = WORDS[student.state]
    mark = (theme.approval_stamp(animate=False) if student.state == "approved"
            else theme.status_chip(chip, word))
    if student.auto:
        mark += theme.status_chip("approved", AUTO_CHIP)
    if student.marked:
        mark += theme.status_chip("insufficient", SHORT_CHIP)
    st.html(f'<div class="pfb-who"><span class="pfb-who__name">{escape(student.name)}さん</span>'
            f'{mark}</div>')


FRAME_WIDTH = 320     # px: an evidence frame on the screen (「大きく見る」 for the whole frame)


def _evidence(repo: Repo, run: StageRun, kind: Kind) -> None:
    items = cited_evidence(repo, run)
    if not items:
        return
    st.html('<h3 class="pfb-subhead">この文章の根拠</h3>')
    when = "場面" if kind == Kind.video else "書いた所"
    for n, item in enumerate(items, 1):
        st.html(f'<dl class="pfb-evidence"><dt>{when}</dt>'
                f'<dd class="pfb-evidence__when">{escape(item.when)}</dd>'
                f'<dt>したこと</dt><dd>{escape(item.what)}</dd>'
                f'<dt>よかった理由</dt><dd>{escape(item.why or "（書かれていません）")}</dd></dl>')
        if item.frames:
            with keyed.container(horizontal=True, gap="small", key=f"pfb_frames_{item.ref_id}"):
                for k, f in enumerate(item.frames, 1):
                    name = f"{when}{n}" + (f"の{k}枚目" if len(item.frames) > 1 else "")
                    with st.container(width=FRAME_WIDTH, gap=None):
                        zoom.picture(f.path, alt=f"{when} {item.when}、{f.caption}",
                                     caption=f.caption, width=FRAME_WIDTH, name=name,
                                     key=f"frame_{item.ref_id}_{k - 1}")


def _text(student: Student) -> None:
    text = student.short if student.state == "short" else student.text
    st.html(f'<div class="pfb-text">{escape(text)}</div>')
    if student.marked:
        st.caption(theme.L(SHORT_CAPTION))
    elif student.run.source == "teacher":
        st.caption(theme.L(TEACHER_TEXT))


def plain_warnings(warnings: list[str], tail: str = WARNING_FIX) -> list[str]:
    """The text warnings of a feedback run (or of a teacher's text) in plain words, one per
    category; the hand-written marker is not a warning."""
    out = []
    for w in warnings:
        if w in (review.MANUAL_WARNING, short_text.MARK):   # markers: shown as chips, if at all
            continue
        line = next((text for key, text in WARNING_TEXT if w.startswith(key)), WARNING_OTHER)
        if line not in out:
            out.append(line)
    return [line + tail for line in out]


def _plain(error: Exception) -> str:
    """A service refusal in the screen's words."""
    if isinstance(error, BusyError):
        return BUSY
    text = validation_text(error) if isinstance(error, ValidationError) else str(error)
    if "空" in text and isinstance(error, ValidationError):
        return EMPTY
    if "古く" in text:
        return STALE
    if "根拠が足りません" in text:
        return INSUFFICIENT + INSUFFICIENT_MORE
    if review.UNTRANSLATED in text:
        return ENGLISH_GATE
    if "人物の特定" in text:
        return IDENTITY_GATE
    if "失敗" in text:
        return FAILED
    if "まだありません" in text:
        return PENDING
    return text.replace("PFB", "文章")


def _request(action: str, subject_id: int, run_id: int | None) -> None:
    """A button's on_click (before the page runs): take the action only if the text on screen
    when it was pressed is still the student's latest; otherwise drop it and say so."""
    repo = get_repo()
    try:
        subject = repo.get_subject(subject_id)
        run = repo.latest_run(subject["submission_id"], Stage.pfb, subject_id=subject_id)
    except KeyError:
        run = None
    finally:
        repo.conn.close()
    if (run.id if run else None) != run_id:
        st.session_state.pop(_ACTION, None)
        flash(NEWER, "warning")
        return
    st.session_state[_ACTION] = (action, subject_id, run_id)


def _request_short(subject_id: int, run_id: int | None, shown: str) -> None:
    """「この文で承認する」's on_click: `_request`, keeping the short text the screen showed so
    the service compares against what was read, not what the page builds now."""
    _request("short", subject_id, run_id)
    if st.session_state.get(_ACTION) == ("short", subject_id, run_id):
        st.session_state[_SHOWN_SHORT] = (subject_id, run_id, shown)


def _button(label: str, action: str, student: Student, **kw) -> None:
    """An action button bound to this student and the text shown (its run id)."""
    st.button(label, key=f"pfb_review_{action}_{student.id}_{student.run_id}",
              on_click=_request, args=(action, student.id, student.run_id), **kw)


def _approved(screen: router.Screen, unit: Unit, student: Student) -> None:
    """After an approval: the stamp once, the next student to check, this step kept."""
    st.session_state[STAMP] = (student.id, student.name)
    st.session_state.setdefault(_NEXT, {})[unit.key] = None
    _close_editor(student)
    _pin(screen, unit)
    st.rerun()


def _approve(screen: router.Screen, unit: Unit, student: Student) -> None:
    if _kind_of(screen) == Kind.video and not step_tips.confirm_shown(screen, student):
        return              # the tips box was refused (said on screen): nothing is approved
    try:
        review.approve(screen.repo, student.id)
    except (BusyError, ApprovalError) as e:
        st.error(theme.L(_plain(e)))
        return
    _approved(screen, unit, student)


def _kind_of(screen: router.Screen) -> Kind:
    return Kind(screen.progress.kind)


def _fix_and_approve(screen: router.Screen, unit: Unit, student: Student, text: str) -> None:
    try:
        review.write_manual_pfb(screen.repo, student.id, text)
    except (BusyError, ValidationError) as e:
        st.error(theme.L(_plain(e)))
        return
    try:
        review.approve(screen.repo, student.id)
    except (BusyError, ApprovalError) as e:   # the text is saved; say why it is not approved
        _close_editor(student)
        _pin(screen, unit)
        flash(f"{student.name}さんの文章は保存しましたが、承認できませんでした: {_plain(e)}", "error", raw=True)
        st.rerun()
    _approved(screen, unit, student)


def _approve_short(screen: router.Screen, unit: Unit, student: Student, shown: str) -> None:
    """「この文で承認する」 (or an unchanged 「直して承認」 after 「書き直す」): the short text
    that was on screen, stored and approved by this press (the service refuses another)."""
    try:
        short_text.approve_short(screen.repo, student.id, student.run_id, shown)
    except (BusyError, ApprovalError) as e:
        st.error(theme.L(_plain(e)))
        return
    _approved(screen, unit, student)


def _decline(screen: router.Screen, unit: Unit, student: Student) -> None:
    """「この人は入れない」: the people panel's exclusion (「評価に戻す」 under 「映っている人」
    at the foot of this screen undoes it, for every method), so the person leaves the totals and the work can complete."""
    try:
        set_person_excluded(screen.repo, student.id, True)
    except (ValueError, BusyError) as e:
        st.error(theme.L(_plain(e)))
        return
    _close_editor(student)
    st.session_state.setdefault(_NEXT, {})[unit.key] = None
    _pin(screen, unit)
    flash(f"{student.name}さんは入れないことにしました。", raw=True)
    st.rerun()


def _redo(screen: router.Screen, unit: Unit, student: Student) -> None:
    try:
        review.regenerate_from(screen.repo, student.subject["submission_id"], Stage.pfb,
                               subject_id=student.id)
    except (BusyError, ValidationError) as e:
        st.error(theme.L(_plain(e)))
        return
    _close_editor(student)     # a kept draft would outlive the text it was written against
    _pin(screen, unit)
    flash(f"{student.name}さんの文章を作り直しています。できると、ここに出ます。", raw=True)
    st.rerun()


def _open_editor(screen: router.Screen, unit: Unit, student: Student) -> None:
    """Open the editor: from the student's unsaved draft, else from the current text."""
    st.session_state[EDITING] = student.id
    st.session_state.pop(_EDIT.format(student.id), None)
    _pin(screen, unit)
    st.rerun()


def _close_editor(student: Student) -> None:
    st.session_state.pop(EDITING, None)
    st.session_state.pop(_DRAFT.format(student.id), None)
    st.session_state.pop(_EDIT.format(student.id), None)   # else `_sync_drafts` revives it
    st.session_state.pop(_WARNED, None)


def _editor(screen: router.Screen, unit: Unit, student: Student, request: str | None) -> None:
    draft = _DRAFT.format(student.id)
    text = st.text_area(theme.L("参加者に渡す文章"), value=st.session_state.get(draft, student.start),
                        key=_EDIT.format(student.id), height=180)
    st.session_state[draft] = text      # survives switching to another student and back
    st.caption(theme.L(EDIT_NOTE))
    warned = st.session_state.get(_WARNED) == (student.id, text.strip())
    if warned:
        for line in plain_warnings(review.manual_pfb_warnings(screen.repo, student.id, text),
                                   WARNING_CONFIRM):
            st.warning(theme.L(line))
    with keyed.container(horizontal=True, key="pfb_review_actions"):
        _button(CONFIRM_FIX if warned else FIX_APPROVE, "fix", student, type="primary")
        cancel = st.button(theme.L(CANCEL), key="pfb_review_cancel")
    if request == "fix" and student.state == "short" and text.strip() == (student.short or ""):
        _approve_short(screen, unit, student, text)   # unchanged: it is still the short text
    elif request == "fix":
        found = [] if warned else review.manual_pfb_warnings(screen.repo, student.id, text)
        if found:                  # show them and ask for a second press
            st.session_state[_WARNED] = (student.id, text.strip())
            st.rerun()
        _fix_and_approve(screen, unit, student, text)
    elif cancel:
        _close_editor(student)
        st.rerun()


def _actions(screen: router.Screen, unit: Unit, student: Student, all_done: bool,
             request: str | None, write_primary: bool = False, taken: bool = False) -> None:
    """The buttons under a student: exactly one primary on screen (with all_done the
    completion line carries it for settled students, unless there is nowhere to go on to:
    then `write_primary` makes 「自分で書く」 the primary; with `taken` 「全員まとめて承認」
    carries it). Every button is bound to the text shown; `request` is the action its
    callback accepted."""
    state = student.state
    nxt = _todo(unit.students, student.id)
    kind = "secondary" if taken else "primary"
    with keyed.container(horizontal=True, key="pfb_review_actions"):
        if state in SETTLED:
            if not all_done:
                _button(CONTINUE, "next", student, type=kind, disabled=nxt is None,
                        help=theme.L("次に確かめる参加者へ進みます"))
        elif state == "short":
            st.button(USE_SHORT, key=f"pfb_review_short_{student.id}_{student.run_id}",
                      type=kind, on_click=_request_short,
                      args=(student.id, student.run_id, student.short or ""),
                      help=theme.L("この短い文を、この人の文章として承認します"))
            _button(REWRITE, "write", student)
            _button(DECLINE, "decline", student, type="tertiary")
        else:
            _button(APPROVE, "approve", student, type=kind, disabled=state != "review")
        if state in ("review", "stale", "approved"):
            _button(FIX, "edit", student)
        if state in ("insufficient", "failed"):
            _button(WRITE, "write", student, type="primary" if write_primary else "secondary")
        if state in ("review", "stale", "failed", "insufficient", "short"):
            _button(REDO, "redo", student, type="tertiary", icon=":material/refresh:")
    if request == "approve" and state == "review":
        _approve(screen, unit, student)
    elif request == "short" and state == "short":
        pressed = st.session_state.pop(_SHOWN_SHORT, None)
        shown = pressed[2] if pressed and pressed[:2] == (student.id, student.run_id) else ""
        _approve_short(screen, unit, student, shown)
    elif request == "decline" and state == "short":
        _decline(screen, unit, student)
    elif request in ("edit", "write"):
        _open_editor(screen, unit, student)
    elif request == "redo":
        _redo(screen, unit, student)
    elif request == "next" and nxt is not None:
        st.session_state.setdefault(_NEXT, {})[unit.key] = nxt.id
        st.rerun()


def _body(screen: router.Screen, unit: Unit, student: Student, kind: Kind,
          editing: bool, all_done: bool, request: str | None, stuck: bool = False,
          taken: bool = False) -> None:
    repo, state = screen.snap, student.state
    _head(student)
    if state == "approved":
        st.page_link(GROWTH_PAGE, label=GROWTH_LINK, icon=":material/timeline:",
                     query_params={"subject": student.id})
    if state == "regenerating":
        st.info(theme.L(REDOING))
        _watch(screen, unit, student, kind)
    elif state == "writing":
        st.info(theme.L(WRITING))
        _watch(screen, unit, student, kind)
    elif state == "pending":
        st.info(theme.L(PENDING))
    elif state == "gated":
        st.warning(theme.L(student.gate))
        router.detail_link("review", GATE_LINK, {"subject": student.id})
    elif state == "failed":
        st.error(theme.L(FAILED))
    elif state == "insufficient":
        st.info(theme.L(INSUFFICIENT))
    elif state == "short":
        st.info(theme.L(short_text.NOTE))
    elif state == "stale":
        st.warning(theme.L(STALE))
    if editing:
        _editor(screen, unit, student, request)
        return
    shown = state in ("review", "stale", "approved", "gated") and bool(student.text)
    if shown or state == "short":
        _text(student)
    if state in ("review", "stale"):
        for line in plain_warnings(student.run.warnings):
            st.warning(theme.L(line))
    _actions(screen, unit, student, all_done, request, write_primary=stuck, taken=taken)
    if state == "short":
        st.caption(theme.L(DECLINE_NOTE))
    if kind == Kind.video and state in ("review", "approved") and shown:
        step_tips.person(screen, student, lambda: _pin(screen, unit))
    if shown and student.run.source != "teacher":
        # below the buttons, so that 「承認する」 is in the first view at 1280x800
        _evidence(repo, student.run, kind)
    if state not in _BUSY and any(s.state in _BUSY for s in unit.students):
        _watch(screen, unit, student, kind)


def _onward(screen: router.Screen, unit: Unit):
    """(the next group to check or None, whether PDFを受け取る can be opened)."""
    progress = screen.progress
    others = [s for s in progress.subs if s.submission_id not in unit.submission_ids]
    nxt = next((s for s in others if s.step == "review"), None)
    return nxt, progress.reachable(progress.number("pdf"))


def _completion(screen: router.Screen, unit: Unit, primary: bool) -> None:
    """The completion line and 「続ける」 (to the next group or the PDF); with nowhere to go on
    to (only students with too little evidence, nothing approved) the line alone, and
    「自分で書く」 is the primary."""
    progress = screen.progress
    pdf = progress.number("pdf")
    others = [s for s in progress.subs if s.submission_id not in unit.submission_ids]
    nxt, can_pdf = _onward(screen, unit)
    if nxt is not None:
        line = f"{unit.label}は全員の確認が終わりました。次は{nxt.label}です。"
    elif any(s.step != "pdf" for s in others) and can_pdf:
        line = (f"{unit.label}は全員の確認が終わりました。ほかのグループは解析が終わると確認できます。"
                "承認した参加者の分は、いまPDFを受け取れます。")
    elif can_pdf:
        line = "全員の確認が終わりました。「PDFを受け取る」へ進めます。"
    else:
        line = "自動の文章が作られた参加者はいません。自分で書いて承認すると、PDFを受け取れます。"
    st.html(f'<p class="pfb-done">{escape(theme.L(line))}</p>')
    if nxt is None and not can_pdf:
        return
    if st.button(theme.L(CONTINUE), key="pfb_review_done", type="primary" if primary else "secondary",
                 help="次のグループへ進みます" if nxt else "PDFを受け取る画面へ進みます"):
        if nxt is not None:
            router.state()["submission_id"] = nxt.submission_id
        else:
            router.go(None if pdf == screen.derived else pdf)
        st.rerun()


# --- less approval work: 全員まとめて承認, 確認せずに自動で承認する -------------------------------

def _request_bulk(shown: dict[int, int | None], left: list[tuple[int, str]]) -> None:
    """「全員まとめて承認」's on_click: keep the texts the screen showed (only those it offered
    for the press) and who it left with why; the service approves only texts that are still
    the students' latest (`review.bulk_approve`)."""
    st.session_state[router.REVIEW_BULK] = (shown, left)


def _bulk_done(screen: router.Screen, unit: Unit, shown: dict[int, int | None],
               left: list[tuple[int, str]]) -> None:
    """Approve what the press offered. A student it approved can only have had a draft equal
    to the text (`_editing`): that draft (and its editor) is dropped; every other open editor
    and kept draft stays as it is (their students were never offered)."""
    result = review.bulk_approve(screen.repo, shown)
    for sid in result.approved:   # a kept draft equal to the text it approved: nothing to keep
        st.session_state.pop(_DRAFT.format(sid), None)
        st.session_state.pop(_EDIT.format(sid), None)
        if st.session_state.get(EDITING) == sid:
            st.session_state.pop(EDITING, None)
    names = {s.id: s.name for s in unit.students}
    skipped = [*left, *result.skipped]
    st.session_state[router.REVIEW_BULK_DONE] = (
        len(result.approved), [(names.get(sid, ""), why) for sid, why in skipped])
    st.session_state.setdefault(_NEXT, {})[unit.key] = None
    _pin(screen, unit)
    st.rerun()


def _bulk_report() -> None:
    """What the last 「全員まとめて承認」 did, once: how many, and who was left and why."""
    done = st.session_state.pop(router.REVIEW_BULK_DONE, None)
    if not done:
        return
    n, left = done
    head = (f'<div class="pfb-stamped">{theme.approval_stamp(animate=True)}'
            f'<span class="pfb-stamped__text">{escape(BULK_DONE.format(n=n))}</span></div>'
            if n else f'<p class="pfb-done">{escape(theme.L(BULK_NONE))}</p>')
    rows = "".join(f"<li>{escape(name)}さん: {escape(theme.L(why))}</li>" for name, why in left)
    tail = (f'<p class="pfb-bulk__left">{escape(theme.L(BULK_LEFT))}</p><ul>{rows}</ul>'
            if left else "")
    st.html(f'<div class="pfb-bulk" role="status">{head}{tail}</div>')


def _writing_line(unit: Unit) -> None:
    """While some first texts are still written: how many, and that they come here."""
    n = sum(1 for s in unit.students if s.state == "writing")
    if n:
        st.html(f'<p class="pfb-writing" role="status" style="margin:0;font-size:0.9375rem;'
                f'color:var(--pfb-muted)">{theme.status_chip("processing", WORDS["writing"][0])} '
                f'{escape(theme.L(WRITING_LINE.format(n=n)))}</p>')


def _auto_line(unit: Unit) -> None:
    """Who the worker approved (確認せずに自動で承認する): the chip, word and mark, and names."""
    auto = [s.name for s in unit.students if s.auto]
    if auto:
        names = "、".join(f"{escape(n)}さん" for n in auto)
        st.html(f'<p class="pfb-auto">{theme.status_chip("approved", AUTO_CHIP)} '
                f'{escape(theme.L(AUTO_LINE))}: {names}</p>')


def _sync_drafts() -> None:
    """Before anything reads the drafts: an editor's text the page has not run with yet (typed,
    then another student picked in the same press) becomes that student's kept draft."""
    for key in list(st.session_state):
        sid = key[len(_EDIT) - 2:] if isinstance(key, str) and key.startswith(_EDIT[:-2]) else ""
        if sid.isdigit() and isinstance(st.session_state[key], str):
            st.session_state[_DRAFT.format(int(sid))] = st.session_state[key]


def _editing(student: Student) -> bool:
    """A draft of the student's is kept (while switching, or in the open editor) that differs
    from the text it would replace; an editor just opened, before it kept its draft, counts."""
    draft = st.session_state.get(_DRAFT.format(student.id))
    if draft is None:
        return st.session_state.get(EDITING) == student.id
    return draft.strip() != student.start.strip()


def _left_reason(student: Student) -> str:
    """Why the screen does not offer a student for 「全員まとめて承認」, in plain words."""
    if _editing(student):
        return SKIP_EDITING
    if student.state == "short" or (student.state == "review" and student.short_run):
        return short_text.SKIP_SHORT
    if student.state == "writing":
        return SKIP_WRITING
    if student.state == "regenerating":
        return review.SKIP_BUSY if student.job is not None else review.SKIP_ANALYSIS
    if student.state == "gated":
        return {IDENTITY_GATE: review.SKIP_IDENTITY,
                ENGLISH_GATE: review.SKIP_ENGLISH}.get(student.gate, review.SKIP_STALE)
    return {"review": review.SKIP_WARNINGS, "stale": review.SKIP_STALE,
            "failed": review.SKIP_FAILED, "insufficient": review.SKIP_INSUFFICIENT,
            "pending": review.SKIP_PENDING}.get(student.state, review.SKIP_PENDING)


def _bulk(screen: router.Screen, unit: Unit, editing: bool) -> bool:
    """「全員まとめて承認」 when two or more texts need no look, the primary then; hidden while
    an editor is open (its 「直して承認」 is the one, and the press must never take the text
    under edit). Students with a kept draft are not offered. True when it is shown."""
    if editing:
        return False
    ready = [s for s in unit.students if s.bulk_ok and not _editing(s)]
    if len(ready) < 2:
        return False
    shown = {s.id: s.run_id for s in ready}
    left = [(s.id, _left_reason(s)) for s in unit.students
            if s.state != "approved" and s.id not in shown]
    st.caption(theme.L(BULK_LINE.format(n=len(ready))))
    st.button(theme.L(BULK), key=f"pfb_review_bulk_{unit.key}", type="primary",
              on_click=_request_bulk, args=(shown, left),
              help=theme.L("気になる点のある人や根拠が足りない人は、まとめて承認しません"))
    return True


def _set_auto(assignment_id: int, key: str) -> None:
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    repo = get_repo()
    try:
        review.set_auto_approve(repo, assignment_id, bool(st.session_state.get(key)))
    finally:
        repo.conn.close()


def _auto_option(screen: router.Screen) -> None:
    """The work option 「確認せずに自動で承認する」 (also on 管理用の画面), read from the work each run."""
    aid = screen.progress.assignment_id
    key = f"pfb_auto_approve_{aid}"
    st.session_state[key] = bool(screen.snap.get_assignment(aid).get("auto_approve"))
    st.toggle(theme.L(AUTO_OPTION), key=key, on_change=_set_auto, args=(aid, key),
              help=theme.L(AUTO_HELP), disabled=screen.pending)


# --- the live part -----------------------------------------------------------------------------

def _watch(screen: router.Screen, unit: Unit, student: Student, kind: Kind) -> None:
    ids = list(unit.submission_ids)

    def signature(view: View) -> tuple:
        return tuple((s.id, s.state, s.run.id if s.run else None)
                     for s in _unit_students(view.repo, ids, kind))

    def draw(view: View) -> None:
        repo = view.repo
        mine = next((s for s in _unit_students(repo, ids, kind) if s.id == student.id), None)
        if mine is not None and mine.job is not None:
            status = "waiting_gpu" if mine.job["status"] == "waiting_gpu" else "processing"
            line = job_line(mine.job["stage"], mine.job["status"], None, kind)
            st.html(f'<ul class="pfb-livelines" aria-live="polite"><li>'
                    f'{theme.status_chip(status, line)}</li></ul>')
        live.show_worker_warning(repo, screen.settings, ids)

    live.watch("review", screen.view, signature, draw)


def _request_for(student: Student) -> str | None:
    """The action a button callback accepted for this student and the text shown, if any
    (taken once; one for another student or text is dropped)."""
    req = st.session_state.pop(_ACTION, None)
    if req and req[1] == student.id and req[2] == student.run_id:
        return req[0]
    return None


def render(screen: router.Screen) -> None:
    kind = Kind(screen.progress.kind)
    unit = _unit(screen, kind)
    with theme.step_card(TITLE, GUIDE):
        _part_notes(screen, unit, kind)
        if not unit.students:
            st.info(theme.L(NOBODY))
            st.button(theme.L(APPROVE), key="pfb_review_approve", type="primary", disabled=True)
        else:
            _sync_drafts()            # before anything reads a draft (the bulk press too)
            pressed = st.session_state.pop(router.REVIEW_BULK, None)
            if pressed is not None:
                _bulk_done(screen, unit, *pressed)
            _counter(unit)
            _writing_line(unit)
            _auto_line(unit)
            _bulk_report()
            top = st.container()      # 全員まとめて承認 and the option, above the switcher
            student = _switcher(unit)
            editing = (st.session_state.get(EDITING) == student.id
                       and student.state not in ("regenerating", "writing", "gated", "pending"))
            with top:
                taken = _bulk(screen, unit, editing)
                _auto_option(screen)
            _stamp_line()
            all_done = all(s.state in SETTLED for s in unit.students)
            request = _request_for(student)
            stuck = False
            if all_done:
                nxt, can_pdf = _onward(screen, unit)
                stuck = nxt is None and not can_pdf
                _completion(screen, unit, primary=not editing)
            _body(screen, unit, student, kind, editing, all_done, request,
                  stuck=stuck and not editing, taken=taken)
            if kind == Kind.video:
                step_tips.team(screen, unit, lambda: _pin(screen, unit))
        _people(screen, unit, kind)
    from pfb_app.ui.wizard import dashboard   # noqa: PLC0415 - 試合・活動のまとめ
    dashboard.entry_link(screen.snap.get_assignment(screen.progress.assignment_id))
    theme.help_expander(help_lines(kind))


def _part_notes(screen: router.Screen, unit: Unit, kind: Kind) -> None:
    """A group analysed only in part (a video longer than 20 minutes: the first 20): say so,
    and offer 「つづきも解析する」, the rest as one more group of the work (content_check)."""
    if kind != Kind.video:
        return
    repo, aid = screen.repo, screen.progress.assignment_id
    for sid in unit.submission_ids:
        part = analysed_part(repo, sid)
        if part is None:
            continue
        st.info(theme.L(part.sentence))
        if next_part_group(repo, sid) is not None:
            st.caption(theme.L(PART_ADDED))
        elif st.button(theme.L(part.next_label), key=f"pfb_next_part_{sid}"):
            try:
                add_next_part(repo, screen.settings, aid, sid)
            except (ValueError, VideoError) as exc:
                st.error(str(exc))
                return
            flash(PART_DONE)
            router.go(None)                  # the derived step: the new group's check card
            router.state()["submission_id"] = None
            st.rerun()


def _people(screen: router.Screen, unit: Unit, kind: Kind) -> None:
    """「映っている人」 for the unit's groups, also when nobody is evaluated (so 「評価に戻す」
    stays reachable): the people panel for tracking groups; for the other groups only the
    people taken out of the evaluation (「この人は入れない」), each with 「評価に戻す」, and no
    section while there is nobody to bring back."""
    if kind != Kind.video:
        return
    repo = screen.repo
    subs = [repo.get_submission(sid) for sid in unit.submission_ids]
    tracked = [sub for sub in subs if submission_engine(sub) == "pfb3x3"]
    declined = [person for sub in subs if submission_engine(sub) != "pfb3x3"
                for person in _declined(repo, sub)]
    short = sum(len(v.people) for v in (short_people(repo, sub["id"]) for sub in tracked)
                if v.much)
    if short:    # said above the panel, which is closed: why fewer people than were found
        st.caption(theme.L(SHORT_NOTE.format(n=short)))
    if tracked or declined:
        with st.expander(theme.L(PEOPLE), expanded=False):
            for sub in tracked:
                people_panel(repo, screen.settings, sub)
            for person in declined:
                _restore_row(repo, person)


def _declined(repo: Repo, sub: dict) -> list[dict]:
    """The group's people taken out of the evaluation (the people panel's 「評価しない」 cards:
    a participant id, excluded, and not a roster row that nobody maps)."""
    idle = {s["id"] for s in unused_roster(repo, sub["id"])}
    return [s for s in repo.list_subjects(sub["id"])
            if s.get("participant_id") and not is_evaluated(s) and s["id"] not in idle
            and not s.get("erased_at")]   # removed on request: never offered back


def _restore_row(repo: Repo, person: dict) -> None:
    """One taken-out person: the name marked 「評価しない」 and the panel's 「評価に戻す」
    (`set_person_excluded(..., False)`, secondary: the screen keeps its one primary)."""
    shown = person["display_name"] or UNNAMED
    st.html(f'<p class="pfb-person__mark"><strong>{escape(shown)}</strong>（{EXCLUDED}）</p>')
    if st.button(RESTORE, key=f"pfb_review_restore_{person['id']}"):
        try:
            set_person_excluded(repo, person["id"], False)
        except (ValueError, BusyError) as exc:
            st.error(str(exc))
            return
        flash(RESTORED)
        st.rerun()
