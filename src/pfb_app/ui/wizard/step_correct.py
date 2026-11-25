"""Step 5 (documents: 4) of a 「あとから訂正」 work (review_mode correct_later): 文を読む・直す.

Nothing waits for an approval: the newest text of each person goes into the PDF as it is, and
every correction is kept as a version (services.corrections). One person at a time, in the
order of the poster:

① AIの文を読む — the person's 「よかった所」, large, with its check state as a chip
   (訂正なし / 訂正済み 版N / 誤りの報告あり・確認中) and 「再出力が必要」 when the text or its
   state changed after the PDF was received;
② 根拠の場面を見る — the scenes the text was made from and their frames (for a corrected text,
   those of the AI text it replaced, said so);
③ 直せる — 「文を直す」 (corrections.edit_text, with an optional reason and note),
   「この文は間違い」 (corrections.report_wrong: a reason from REASONS and a note),
   「確認して報告を閉じる」 (resolve_report) while a report is open, 「前の版に戻す」 (revert)
   when the newest version replaced an earlier text, and 「この人だけ作り直す」
   (review.regenerate_from; the worker records the new text as a version,
   corrections.after_pfb). Below, 「訂正の記録」: every row, oldest first, nothing deleted.

The one primary is 「PDFを受け取る」 (to step 6), or the open form's own press while one is
open. Every press is bound to the text shown (step_review._request): a text that changed in
between is not acted on. The screen never shows the words 承認・渡す・下書き・未承認.
"""
from __future__ import annotations

import hashlib
from html import escape

import streamlit as st
from pydantic import ValidationError

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.export import identity_gate
from pfb_app.services import corrections, review
from pfb_app.services.review import BusyError
from pfb_app.ui import access, keyed
from pfb_app.ui.common import cited_evidence, flash
from pfb_app.ui.wizard import router, step_pdf, step_review, step_tips, theme, zoom
from pfb_app.ui.wizard.step_review import Student, Unit

TITLE = "一人ずつ、AIの文を読んでください"
GUIDE = ("① AIの文を読む　② 根拠の場面を見る　③ 違っていれば直す（直した記録が残ります）。"
         "人の確認を待たずに、PDFをすぐ受け取れます。")
STEP_READ = "① AIの文を読む（参加者ごとの「よかった所」）"
STEP_SCENES = "② 根拠の場面を見る（文のもとになったコマ）"
STEP_FIX = "③ 直せる（違っていれば直す。直した記録が残ります）"
PDF = "PDFを受け取る"
FIX = "文を直す"
WRITE = "自分で書く"
SAVE = "直した文を保存する"
CONFIRM_SAVE = "このまま保存する"
WRONG = "この文は間違い"
SEND_REPORT = "報告する"
RESOLVE = "確認して報告を閉じる"
REVERT = "前の版に戻す"
REDO = step_review.REDO                       # 「この人だけ作り直す」
CANCEL = "やめる"
REISSUE = "再出力が必要"
REISSUE_LINE = ("PDFを受け取ったあとで、{n}人の文か確認状況が変わりました（{names}）。"
                "「PDFを受け取る」で受け取り直してください（再出力が必要）。")
COUNTER = "文ができた人 {n} / 全 {m}"
COUNTER_MORE = "訂正済み {c}人・確認中の報告 {r}人"
NOBODY = "読む参加者がいません。"
NO_PDF_YET = "文ができた人がいると、PDFを受け取れます。"
FIXED_NOTE = "直した文です（版{v}）。"
FIXED_SCENES = ("下は、直す前のAIの文の根拠になった場面です。"
                "直した文のPDFには、場面の写真は入りません（どの場面のことか、わからないため）。")
NO_SCENES = "この文には、根拠の場面がついていません。"
EDIT_LABEL = "PDFに入れる文"
REASON_LABEL = "直した理由（任意）"
REASON_NONE = "（選ばない）"
NOTE_LABEL = "メモ（任意。PDFの訂正の記録にも入ります）"
REPORT_REASON = "どこが違いますか"
REPORT_NOTE = "メモ（任意）"
REPORT_HELP = ("報告すると、PDFの確認状況が「誤りの報告あり・確認中」になります。"
               "文を直すか、「確認して報告を閉じる」を押すと終わります。")
REPORT_OPEN = "誤りの報告（確認中）：{reason}"
EDIT_NOTE = "保存すると、この文が新しい版になります（PDFに入る人なら、次に受け取るPDFに入ります）。前の文は記録に残ります。"
HISTORY = "訂正の記録（{n}件）"
WRITING = step_review.WRITING
REDOING = "新しい文を作っています。できると、ここに出ます。この画面は自動で新しくなります。"
PENDING = step_review.PENDING
FAILED = "文章を作れませんでした。「この人だけ作り直す」か、「自分で書く」で書いてください。"
NO_TEXT = "映っている場面が少ないため、AIの文は作られませんでした。「自分で書く」で書けます。"
STALE = ("この文を作ったあとで解析が更新されました。「この人だけ作り直す」で、"
         "新しい解析から作り直せます（いまの文は記録に残ります）。")
GATES = {   # step_review's gate messages, in words without 承認
    step_review.ENGLISH_GATE: "したことの書き出しの一部が英語のままです。管理用の画面で直せます。",
    step_review.IDENTITY_GATE: ("映っている人と参加者のつなぎ方がまだ確定していません。"
                                "管理用の画面で確かめてください。"),
}
SAVED = "{name}さんの文を直しました（版{v}）。次に受け取るPDFに入ります。"
# a person the PDF leaves out (not confirmed as one person, say): never 「PDFに入ります」
SAVED_OUT = ("{name}さんの文を直しました（版{v}）。この人は、いまはPDFに入っていません"
             "（理由は「PDFを受け取る」の画面に出ます）。")
REPORTED = "{name}さんの文の誤りを報告しました。確認状況が「誤りの報告あり・確認中」になります。"
RESOLVED = "{name}さんの報告を閉じました。"
REVERTED = "{name}さんの文を前の版に戻しました（版{v}）。"
REDOING_FLASH = "{name}さんの文を作り直しています。できると、ここに出ます。"
NEWER = step_review.NEWER
WARNING_FIX = "違っていれば「文を直す」で直せます。"
NO_CLUE_HELP = ("背番号がだれにも読めず、「人を確かめる」もしていない動画の作業では、同じ人かどうかを確かめずに"
                "全員をPDFに入れます。そのときは一人ずつの画面に「人を確かめると、取り違えを防げます」と出ます。")
BUSY = step_review.BUSY
EMPTY = step_review.EMPTY

HELP = [
    "一人ずつ、AIが作った文と、その根拠になった場面を見ます。人の確認を待たずに、PDFをすぐ受け取れます。",
    "違っていたら「文を直す」で直します。直した文が新しい版になり、前の文は「訂正の記録」に残ります。",
    "直すのがむずかしいときは「この文は間違い」で、どこが違うか（人物の取り違え・動作が違う・背番号が違う・"
    "その他）を知らせます。PDFの確認状況が「誤りの報告あり・確認中」になります。",
    "「前の版に戻す」で、直す前の文に戻せます。何も消えません。",
    "「この人だけ作り直す」で、その人の文をAIに作り直させます。新しい文も版として残ります。",
    "PDFを受け取ったあとで直したときは「再出力が必要」と出ます。もう一度受け取ってください。",
    "動画の作業では、背番号で同じ人と確かめられた人と、「人を確かめる」で評価する人が確かめた人がPDFに入ります。"
    + NO_CLUE_HELP,
    "「次にためしてみること」（動画の作業）は、AIの案がそのままPDFに入ります（気になる点がある案は入りません）。"
    "直すときは、直して「ヒントを確定する」を押します。",
]

_OPEN = "_pfb_correct_open"          # (subject id, "edit" | "report"): the form that is open
_WARNED = "_pfb_correct_warned"      # (subject id, text) whose warnings were shown once
_TEXT, _REASON, _NOTE = "pfb_fix_text_{}", "pfb_fix_reason_{}", "pfb_fix_note_{}"
_RREASON, _RNOTE = "pfb_report_reason_{}", "pfb_report_note_{}"
_BUSY = ("writing", "regenerating")


# --- where a person stands --------------------------------------------------------------------

def live_state(student: Student) -> str:
    """writing / regenerating / pending / failed (as step 5 of the other mode), else "text"
    (a text the PDF prints) or "none" (no text: too few scenes)."""
    if student.state in ("writing", "regenerating", "pending", "failed"):
        return student.state
    return "text" if student.text else "none"


def _state_word(student: Student, st_: corrections.State | None) -> tuple[str, str]:
    """(the word, the chip kind) of the person's state."""
    state = live_state(student)
    if state == "text" and st_ is not None:
        if st_.reported is not None:
            return st_.label, "your_turn"
        return st_.label, ("guess" if st_.version else "done")
    return {"writing": ("文章を書いています", "processing"),
            "regenerating": ("作り直しています", "processing"),
            "pending": ("まだです", "insufficient"),
            "failed": ("作れませんでした", "stopped"),
            "none": ("文がありません", "insufficient")}.get(state, ("", "insufficient"))


def _actor() -> str | None:
    who = access.actor()
    return who[1] if who is not None and who[1] != "open" else None


# --- the parts of the screen ------------------------------------------------------------------

def _counter(unit: Unit, states: dict[int, corrections.State]) -> None:
    ready = sum(1 for s in unit.students if live_state(s) == "text")
    fixed = sum(1 for s in unit.students if states[s.id].version)
    reported = sum(1 for s in unit.students if states[s.id].reported is not None)
    label = f'<span class="pfb-counter__label">{escape(unit.label)}</span>' if unit.label else ""
    st.html(f'<p class="pfb-counter" role="status">{label}'
            f'<b>{escape(COUNTER.format(n=ready, m=len(unit.students)))}</b>'
            f'<span class="pfb-counter__auto">　{escape(COUNTER_MORE.format(c=fixed, r=reported))}'
            f'</span></p>')


def _reissue_line(unit: Unit, changed: dict[int, str]) -> None:
    names = [s.name for s in unit.students if s.id in changed]
    if names:
        st.warning(theme.L(REISSUE_LINE.format(
            n=len(names), names="、".join(f"{n}さん" for n in names))), icon=":material/sync:")


def _pdf_button(screen: router.Screen, primary: bool) -> None:
    progress = screen.progress
    pdf = progress.number("pdf")
    can = progress.reachable(pdf)
    if st.button(theme.L(PDF), key="pfb_correct_pdf", type="primary" if primary else "secondary",
                 disabled=not can, icon=":material/picture_as_pdf:",
                 help=theme.L("PDFを受け取る画面へ進みます") if can else theme.L(NO_PDF_YET)):
        router.go(pdf)
        st.rerun()


def _switcher(unit: Unit, states: dict[int, corrections.State], changed: dict[int, str]) -> Student:
    key = f"pfb_student_{unit.key}"
    by_id = {s.id: s for s in unit.students}
    if st.session_state.get(key) not in by_id:
        st.session_state[key] = unit.students[0].id
    if len(by_id) == 1:
        return unit.students[0]

    def label(i: int) -> str:
        s = by_id[i]
        word = _state_word(s, states[i])[0]
        return f"{s.name}（{word}・{REISSUE}）" if i in changed else f"{s.name}（{word}）"
    ids = list(by_id)
    if len(ids) <= 6:
        chosen = st.radio(theme.L("参加者"), ids, format_func=label, horizontal=True, key=key)
    else:
        # the box keeps showing the words it had when the person was picked (「選手7（訂正なし）」
        # after a correction): a key that changes with the words draws it again; the person
        # chosen stays in `key`
        words = hashlib.sha1("|".join(label(i) for i in ids).encode("utf-8")).hexdigest()[:10]
        chosen = st.selectbox(theme.L("参加者"), ids, index=ids.index(st.session_state[key]),
                              format_func=label, key=f"{key}_{words}")
        st.session_state[key] = chosen
    return by_id[chosen]


def _head(student: Student, st_: corrections.State, changed: bool) -> None:
    word, chip = _state_word(student, st_)
    mark = theme.status_chip(chip, word) if word else ""
    if changed:
        mark += theme.status_chip("your_turn", REISSUE)
    st.html(f'<div class="pfb-who"><span class="pfb-who__name">{escape(student.name)}さん</span>'
            f'{mark}</div>')


def _subhead(text: str) -> None:
    st.html(f'<h3 class="pfb-subhead">{escape(theme.L(text))}</h3>')


def evidence_run(repo, student: Student) -> StageRun | None:
    """The run whose scenes the screen shows: the text's own, else (a corrected text carries
    none) the newest earlier run of the person that cites scenes."""
    if student.run is not None and (student.run.output or {}).get("evidence_refs"):
        return student.run
    if student.run is None:
        return None
    for run in reversed(repo.run_history(student.subject["submission_id"], Stage.pfb,
                                         subject_id=student.id)):
        if run.status == "ok" and (run.output or {}).get("evidence_refs"):
            return run
    return None


def _scenes(repo, run: StageRun | None, kind: Kind) -> None:
    items = cited_evidence(repo, run) if run is not None else []
    if not items:
        st.caption(theme.L(NO_SCENES))
        return
    when = "場面" if kind == Kind.video else "書いた所"
    for n, item in enumerate(items, 1):
        st.html(f'<dl class="pfb-evidence"><dt>{when}</dt>'
                f'<dd class="pfb-evidence__when">{escape(item.when)}</dd>'
                f'<dt>したこと</dt><dd>{escape(item.what)}</dd>'
                f'<dt>よかった理由</dt><dd>{escape(item.why or "（書かれていません）")}</dd></dl>')
        if item.frames:
            with keyed.container(horizontal=True, gap="small", key=f"pfb_cframes_{item.ref_id}"):
                for k, f in enumerate(item.frames, 1):
                    name = f"{when}{n}" + (f"の{k}枚目" if len(item.frames) > 1 else "")
                    with st.container(width=step_review.FRAME_WIDTH, gap=None):
                        zoom.picture(f.path, alt=f"{when} {item.when}、{f.caption}",
                                     caption=f.caption, width=step_review.FRAME_WIDTH, name=name,
                                     key=f"cframe_{item.ref_id}_{k - 1}")


def _when(stamp: str) -> str:
    return corrections.review_when(stamp) or ""


def _history(repo, student: Student) -> None:
    rows = corrections.history(repo, student.id)
    if not rows:                       # the chip already says 「訂正なし」
        return
    with st.expander(theme.L(HISTORY.format(n=len(rows))), expanded=False):
        items = []
        for v in rows:
            if v.kind in corrections.BUMPS:
                what = f"版{v.version} {corrections.KIND_WORD[v.kind]}"
                what += f"：{v.reason}" if v.reason else ""
            elif v.kind == "report":
                what = f"誤りの報告：{v.reason}"
            else:
                what = "報告を確認して閉じた"
            what += f"（{v.note}）" if v.note else ""
            who = f"　{v.actor}" if v.actor and v.actor != "open" else ""
            body = ""
            if v.kind in corrections.BUMPS:
                body = (f'<details class="pfb-history__texts"><summary>前の文と新しい文</summary>'
                        f'<p><b>前</b>：{escape(v.prev_text or "（文なし）")}</p>'
                        f'<p><b>新</b>：{escape(v.text or "（文なし）")}</p></details>')
            items.append(f'<li><span class="pfb-history__when">{escape(_when(v.created_at))}</span>　'
                         f'{escape(what)}{escape(who)}{body}</li>')
        st.html(f'<ul class="pfb-history">{"".join(items)}</ul>')


# --- the actions --------------------------------------------------------------------------------

def _pin(screen: router.Screen, unit: Unit) -> None:
    step_review._pin(screen, unit)


def _close(student: Student) -> None:
    st.session_state.pop(_OPEN, None)
    st.session_state.pop(_WARNED, None)
    for key in (_TEXT, _REASON, _NOTE, _RREASON, _RNOTE):
        st.session_state.pop(key.format(student.id), None)


def _done(screen: router.Screen, unit: Unit, student: Student, message: str) -> None:
    _close(student)
    _pin(screen, unit)
    flash(message, raw=True)
    st.rerun()


def _plain(error: Exception) -> str:
    if isinstance(error, BusyError):
        return BUSY
    if isinstance(error, ValidationError):
        return EMPTY
    return str(error)


def _save(screen: router.Screen, unit: Unit, student: Student, text: str) -> None:
    reason = st.session_state.get(_REASON.format(student.id))
    note = (st.session_state.get(_NOTE.format(student.id)) or "").strip()
    try:
        corrections.edit_text(screen.repo, student.id, text, actor=_actor(),
                              reason=None if reason in (None, REASON_NONE) else reason,
                              note=note or None)
    except (BusyError, ValidationError, corrections.CorrectionError) as e:
        st.error(theme.L(_plain(e)))
        return
    v = corrections.version_no(screen.repo, student.id)
    aid = screen.progress.assignment_id
    now = corrections.in_pdf_now(screen.repo, aid)
    words = SAVED_OUT if now is not None and student.id not in now else SAVED
    _done(screen, unit, student, words.format(name=student.name, v=v))


def _editor(screen: router.Screen, unit: Unit, student: Student, request: str | None) -> None:
    key = _TEXT.format(student.id)
    if key not in st.session_state:
        st.session_state[key] = student.text
    text = st.text_area(theme.L(EDIT_LABEL), key=key, height=180)
    st.selectbox(theme.L(REASON_LABEL), [REASON_NONE, *corrections.REASONS],
                 key=_REASON.format(student.id))
    st.text_input(theme.L(NOTE_LABEL), key=_NOTE.format(student.id), max_chars=120)
    st.caption(theme.L(EDIT_NOTE))
    warned = st.session_state.get(_WARNED) == (student.id, text.strip())
    if warned:
        for line in step_review.plain_warnings(
                review.manual_pfb_warnings(screen.snap, student.id, text),
                "このままでよければ、もう一度「このまま保存する」を押してください。"):
            st.warning(theme.L(line))
    with keyed.container(horizontal=True, key="pfb_correct_form"):
        _button(CONFIRM_SAVE if warned else SAVE, "save", student, type="primary")
        cancel = st.button(theme.L(CANCEL), key="pfb_correct_cancel")
    if request == "save":
        if not text.strip():
            st.error(theme.L(EMPTY))
            return
        found = [] if warned else step_review.plain_warnings(
            review.manual_pfb_warnings(screen.snap, student.id, text))
        if found:                  # show them and ask for a second press
            st.session_state[_WARNED] = (student.id, text.strip())
            st.rerun()
        _save(screen, unit, student, text)
    elif cancel:
        _close(student)
        st.rerun()


def _report_form(screen: router.Screen, unit: Unit, student: Student, request: str | None) -> None:
    st.radio(theme.L(REPORT_REASON), list(corrections.REASONS), index=None, horizontal=True,
             key=_RREASON.format(student.id))
    st.text_input(theme.L(REPORT_NOTE), key=_RNOTE.format(student.id), max_chars=120)
    st.caption(theme.L(REPORT_HELP))
    with keyed.container(horizontal=True, key="pfb_correct_form"):
        _button(SEND_REPORT, "send", student, type="primary")
        cancel = st.button(theme.L(CANCEL), key="pfb_correct_cancel")
    if request == "send":
        reason = st.session_state.get(_RREASON.format(student.id))
        note = (st.session_state.get(_RNOTE.format(student.id)) or "").strip()
        try:
            corrections.report_wrong(screen.repo, student.id, reason or "", note=note,
                                     actor=_actor())
        except corrections.CorrectionError as e:
            st.error(theme.L(str(e)))
            return
        _done(screen, unit, student, REPORTED.format(name=student.name))
    elif cancel:
        _close(student)
        st.rerun()


def _button(label: str, action: str, student: Student, **kw) -> None:
    """A press bound to this person and the text shown (step_review._request)."""
    st.button(theme.L(label), key=f"pfb_correct_{action}_{student.id}_{student.run_id}",
              on_click=step_review._request, args=(action, student.id, student.run_id), **kw)


def _can_revert(repo, student: Student) -> bool:
    bumps = [v for v in corrections.history(repo, student.id) if v.kind in corrections.BUMPS]
    return bool(bumps) and bumps[-1].prev_run_id is not None


def _act(screen: router.Screen, unit: Unit, student: Student, request: str | None,
         st_: corrections.State) -> None:
    """The presses that act at once (the forms open on 「文を直す」 and 「この文は間違い」)."""
    repo = screen.repo
    if request in ("fix", "write"):
        st.session_state[_OPEN] = (student.id, "edit")
        st.session_state.pop(_TEXT.format(student.id), None)
        _pin(screen, unit)
        st.rerun()
    elif request == "wrong":
        st.session_state[_OPEN] = (student.id, "report")
        _pin(screen, unit)
        st.rerun()
    elif request == "resolve":
        try:
            corrections.resolve_report(repo, student.id, actor=_actor())
        except corrections.CorrectionError as e:
            st.error(theme.L(str(e)))
            return
        _done(screen, unit, student, RESOLVED.format(name=student.name))
    elif request == "revert":
        try:
            corrections.revert(repo, student.id, actor=_actor())
        except corrections.CorrectionError as e:
            st.error(theme.L(str(e)))
            return
        _done(screen, unit, student, REVERTED.format(
            name=student.name, v=corrections.version_no(repo, student.id)))
    elif request == "redo":
        try:
            review.regenerate_from(repo, student.subject["submission_id"], Stage.pfb,
                                   subject_id=student.id)
        except (BusyError, ValidationError) as e:
            st.error(theme.L(step_review._plain(e)))
            return
        _done(screen, unit, student, REDOING_FLASH.format(name=student.name))


def _fix_buttons(screen: router.Screen, student: Student, st_: corrections.State,
                 state: str) -> None:
    with keyed.container(horizontal=True, key="pfb_correct_actions"):
        if state == "text":
            _button(FIX, "fix", student, icon=":material/edit:")
            if st_.reported is None:
                _button(WRONG, "wrong", student, icon=":material/report:")
            else:
                _button(RESOLVE, "resolve", student, icon=":material/task_alt:")
        elif state in ("none", "failed"):
            _button(WRITE, "write", student, icon=":material/edit:")
        if state in ("text", "none") and _can_revert(screen.snap, student):
            _button(REVERT, "revert", student, icon=":material/undo:")
        if state in ("text", "none", "failed"):
            _button(REDO, "redo", student, type="tertiary", icon=":material/refresh:")


def _body(screen: router.Screen, unit: Unit, student: Student, kind: Kind,
          states: dict[int, corrections.State], changed: dict[int, str],
          no_clue: bool = False) -> None:
    repo = screen.snap
    st_ = states[student.id]
    state = live_state(student)
    _head(student, st_, student.id in changed)
    if no_clue:      # nothing tells people apart in this work: everyone is printed
        st.caption(theme.L(identity_gate.CHECK_HINT + "。"))
    request = step_review._request_for(student)
    opened = st.session_state.get(_OPEN)
    form = opened[1] if opened and opened[0] == student.id and state not in _BUSY else None
    if state in _BUSY:
        st.info(theme.L(WRITING if state == "writing" else REDOING))
        step_review._watch(screen, unit, student, kind)
        return
    if student.gate:
        st.caption(theme.L(GATES.get(student.gate, student.gate)))
    _subhead(STEP_READ)
    if state == "pending":
        st.info(theme.L(PENDING))
    elif state == "failed":
        st.error(theme.L(FAILED))
    elif state == "none":
        st.info(theme.L(NO_TEXT))
    else:
        if form != "edit":
            st.html(f'<div class="pfb-text">{escape(student.text)}</div>')
        if student.run.source == "teacher":
            st.caption(theme.L(FIXED_NOTE.format(v=st_.version) if st_.version
                               else step_review.TEACHER_TEXT))
        if student.state == "stale":
            st.warning(theme.L(STALE))
        for line in step_review.plain_warnings(student.run.warnings, WARNING_FIX):
            st.caption(theme.L(line))
        if st_.reported is not None:
            rep = st_.reported
            st.warning(theme.L(REPORT_OPEN.format(reason=rep.reason))
                       + (f"（{rep.note}）" if rep.note else ""), icon=":material/report:")
    if state in ("text", "none") and student.run is not None:
        _subhead(STEP_SCENES)
        shown = evidence_run(repo, student)
        if shown is not None and shown.id != student.run.id:
            st.caption(theme.L(FIXED_SCENES))
        _scenes(repo, shown, kind)
    if state == "pending":
        return
    _subhead(STEP_FIX)
    if form == "edit":
        _editor(screen, unit, student, request)
    elif form == "report":
        _report_form(screen, unit, student, request)
    else:
        _fix_buttons(screen, student, st_, state)
        _act(screen, unit, student, request, st_)
    if kind == Kind.video and state == "text":      # 次にためしてみること, changed later as well
        step_tips.person(screen, student, lambda: step_review._pin(screen, unit), live=True)
    _history(repo, student)
    if any(live_state(s) in _BUSY for s in unit.students):
        step_review._watch(screen, unit, student, kind)


def render(screen: router.Screen) -> None:
    kind = Kind(screen.progress.kind)
    unit = step_review._unit(screen, kind)
    work = screen.snap.get_assignment(screen.progress.assignment_id)
    changed = corrections.changed_since(screen.snap, work["id"], work.get("pdf_downloaded_at"),
                                        pdf_only=True)
    with theme.step_card(TITLE, GUIDE):
        step_review._part_notes(screen, unit, kind)
        if not unit.students:
            st.info(theme.L(NOBODY))
            _pdf_button(screen, primary=True)
        else:
            states = {s.id: corrections.state(screen.snap, s.id) for s in unit.students}
            opened = st.session_state.get(_OPEN)
            _counter(unit, states)
            _reissue_line(unit, changed)
            if kind == Kind.video:      # people out of the PDF for want of a check
                found = step_pdf.who_live(screen.snap, work["id"])
                if found is not None and step_pdf.gate_left_out(found[1]):
                    step_pdf.more_by_check(screen, "pfb_correct_to_people")
            top = st.container()
            student = _switcher(unit, states, changed)
            form_open = bool(opened and opened[0] == student.id
                             and live_state(student) not in _BUSY)
            with top:
                _pdf_button(screen, primary=not form_open)
            no_clue = kind == Kind.video and not identity_gate.work_has_clue(screen.snap, work)
            _body(screen, unit, student, kind, states, changed, no_clue)
            if kind == Kind.video:
                step_tips.team(screen, unit, lambda: step_review._pin(screen, unit), live=True)
        step_review._people(screen, unit, kind)
    from pfb_app.ui.wizard import dashboard   # noqa: PLC0415 - 試合・活動のまとめ
    dashboard.entry_link(work)
    theme.help_expander(HELP)
