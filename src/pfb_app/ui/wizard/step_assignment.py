"""Step 1, 活動を決める: create the assignment, or look at (and, before any analysis, edit) it.

The fields are those of the 活動 page in plain words; creation goes through the same glue
(`create_assignment_from_inputs` in ui/common). After analysis has started the task text and rubric are only
shown (`update_assignment` refuses then, as results would no longer match them).

A new work starts at the drop zone (step_drop). Dropped documents come here with the kind
fixed (`render_new(..., docs)`, with the optional roster CSV for the name guessing) and are
registered right after the work is created; dropped videos go to the content check instead.
The old video fields of this form (the identity checkbox) are now on the check card or in
管理用の画面.

「作業の設定」 (an existing work, `_settings`): 文の出し方 — 「あとから訂正」 (correct_later:
the newest texts go out at once, corrections are kept as versions; a new work's default) or
「承認して出す」 (approve: only approved texts) — and the work's 出典 (credits), one per line,
printed on its PDFs. Both are saved at once and can be changed at any time.
"""
import logging
from html import escape

import streamlit as st

from pfb_app.config import Settings
from pfb_app.services import corrections
from pfb_app.services.assignments import BusyError, update_assignment
from pfb_app.services.intake import register_documents
from pfb_app.storage.db import Repo
from pfb_app.ui import access
from pfb_app.ui.common import (MODEL_LABELS, UNEXPECTED, IntakeFailed,
                               create_assignment_from_inputs, flash, get_repo, read_roster,
                               show_intake_error, upload_note)
from pfb_app.ui.wizard import router, theme
from pfb_app.ui.wizard.progress import DOCUMENT_STEPS

log = logging.getLogger(__name__)

NEW_KEYS = ("a_title", "a_task", "a_rubric", "a_rubric_file", "a_roster", "a_model")
LOCKED = ("解析が始まったため、活動の内容と評価の観点は変更できません。"
          "変えるときは、ホームから新しく始めてください。")
KINDS = {"video": "動画（グループ活動や練習などの録画）", "document": "書類（レポートや企画書など）"}

SETTINGS = "作業の設定"
MODE = "文の出し方"
MODE_NOTES = {
    corrections.CORRECT_LATER: ("人の確認を待たずに、AIが作った文をそのままPDFにできます。違っていたところは"
                                "「文を読む・直す」であとから直せます（直した記録が残り、PDFには文の版と確認状況が入ります）。"),
    corrections.APPROVE: "確かめて承認した人の文だけをPDFに入れます。",
}
MODE_SAVED = "文の出し方を「{mode}」にしました。"
CREDITS = "出典（動画や資料の出どころ・ライセンス。1行に1つ）"
CREDITS_NOTE = "PDFの終わりに小さく入ります。空のときは「出典：作業の設定で未入力」と入ります。"
CREDITS_SAVE = "出典を保存する"
CREDITS_SAVED = "出典を保存しました。次に受け取るPDFから入ります。"
CREDITS_CLEARED = "出典を空にしました。"
CREDITS_MAX = 600

HELP = [
    "活動の名前と、評価する内容（参加者に伝えた問いや指示）を書きます。評価の観点があれば入れます。",
    "フィードバックは、ここに書いた活動の内容と評価の観点に沿って作られます。",
    "解析が始まった後は活動の内容を変えられません。変えるときは新しく始めてください。",
    "「作業の設定」で、文の出し方（あとから訂正／承認して出す）と、PDFに入れる出典を決められます。いつでも変えられます。",
    "うまくいかないとき：評価の観点のファイルが読めないときは、その内容を「評価の観点」の欄に"
    "書き写してください。",
]


def _dropped(docs: list[tuple[str, bytes]]) -> None:
    """The dropped documents (the kind is fixed by them) and a way back to the drop zone."""
    names = "".join(f"<li>{escape(name)}</li>" for name, _ in docs)
    st.html(f'<dl class="pfb-facts"><dt>ファイルの種類</dt><dd>{escape(KINDS["document"])}</dd>'
            f'<dt>入れた書類</dt><dd>{len(docs)}件<ul class="pfb-names__list">{names}</ul>'
            f'</dd></dl>')
    if st.button("ファイルを入れ直す", key="a_redrop", icon=":material/arrow_back:"):
        st.session_state.pop(router.DROP_DOCS, None)
        st.rerun()


def render_new(repo: Repo, settings: Settings, docs: list[tuple[str, bytes]]) -> None:
    """Step 1 of a new work from documents dropped on the drop zone (videos go to the content
    check instead): the kind is fixed, and the documents are registered as soon as the work
    is created. The stepper starts at 1 with nothing done yet."""
    theme.stepper(DOCUMENT_STEPS, current=0, done=set())
    with theme.step_card("活動を決めましょう",
                         "どんな活動で何を評価するかを書くと、フィードバックはその内容と"
                         "評価の観点に沿って作られます。"):
        title = st.text_input(theme.L("活動の名前"), key="a_title",
                              placeholder="例：ペーパータワー（Aチーム）")
        _dropped(docs)
        task_text = st.text_area(theme.L("活動の内容（参加者に伝えた問いや指示）"), key="a_task")
        rubric_text = st.text_area(theme.L("評価の観点（任意）"), key="a_rubric",
                                   help="評価の観点があれば、ここに書くかファイルで入れます。")
        rubric_file = st.file_uploader(theme.L("評価の観点のファイル（任意、PDFまたはWord）"),
                                       type=["pdf", "docx", "doc"], key="a_rubric_file")
        st.caption(upload_note("PDFかWordのファイル"))
        roster_file = st.file_uploader(theme.L("名簿のCSV（任意、「name」列）"), type=["csv"],
                                       key="a_roster")
        st.caption(theme.L("名簿があると、書類のファイル名から名前を読み取るのが確かになります。")
                   + upload_note("CSVファイル"))
        model_key = st.selectbox(theme.L("解析の細かさ"), list(settings.models), key="a_model",
                                 format_func=lambda k: MODEL_LABELS.get(k, k),
                                 help="高精度は時間がかかりますが、より細かく読み取ります。")
        if st.button(theme.L("これで決定して次へ"), key="a_save", type="primary"):
            _create(repo, settings, title, task_text, rubric_text, rubric_file, roster_file,
                    model_key, docs)
    theme.help_expander(HELP)


def _roster(upload) -> list[str] | None:
    """The names of the roster CSV (None without one); IntakeFailed when it cannot be read."""
    if upload is None:
        return None
    try:
        return read_roster(upload)
    except UnicodeDecodeError as e:          # a ValueError too: caught first
        raise IntakeFailed("名簿を読めません。UTF-8で保存したCSVを使ってください。", str(e)) from e
    except ValueError as e:
        raise IntakeFailed(f"名簿を読めません: {e}") from e


def _create(repo, settings, title, task_text, rubric_text, rubric_file, roster_file, model_key,
            docs) -> None:
    if not title.strip() or not task_text.strip():
        st.error(theme.L("活動の名前と活動の内容を入力してください。"))
        return
    try:
        roster = _roster(roster_file)        # read first: a bad roster creates nothing
        aid = create_assignment_from_inputs(repo, settings, title, "document", task_text,
                                            rubric_text, rubric_file, model_key, False)
    except IntakeFailed as e:
        show_intake_error(e)
        return
    for key in NEW_KEYS:            # the next 「新しく始める」 starts from an empty form
        st.session_state.pop(key, None)
    st.session_state.pop(router.DROP_DOCS, None)
    from pfb_app.ui.wizard import step_files   # noqa: PLC0415
    try:
        report = register_documents(repo, settings, aid, docs, roster)
    except Exception as e:  # noqa: BLE001 - the work exists: step 2 takes the documents again
        log.exception("書類を登録できませんでした assignment=%s", aid)
        flash(f"活動を決定しました。{UNEXPECTED}", "error")
        router.open_work(aid, 2)
        st.rerun()
    if report.created:
        flash(f"活動を決定しました。{len(report.created)}件の書類を登録して、解析を始めています。")
    else:
        flash("活動を決定しました。書類を読めなかったので、もう一度入れてください。", "warning")
    shown = None
    if report.errors or report.duplicate_names:
        # step 2 says what could not be taken (unreadable files, repeated names)
        st.session_state[step_files.REPORT] = report
        shown = 2
    router.open_work(aid, shown)
    st.rerun()


def _save(repo: Repo, a: dict, task: str, rubric: str) -> bool:
    """Save the edited texts (if changed); False after showing why it could not be done."""
    old = (a["task_text"].strip(), (a["rubric_text"] or "").strip())
    if (task.strip(), rubric.strip()) == old:
        return True
    try:
        update_assignment(repo, a["id"], task, rubric)
    except BusyError:
        st.error(theme.L(LOCKED))
        return False
    except ValueError:
        st.error(theme.L("活動の内容を入力してください。"))
        return False
    flash("活動の内容を決定しました。")
    return True


def _set_mode(assignment_id: int, key: str) -> None:
    """The radio's on_change: store 文の出し方 (services.corrections.set_mode, audited)."""
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    mode = st.session_state.get(key)
    repo = get_repo()
    try:
        corrections.set_mode(repo, assignment_id, mode)
    except corrections.CorrectionError as e:
        flash(str(e), "error")
        return
    finally:
        repo.conn.close()
    flash(MODE_SAVED.format(mode=corrections.MODE_WORDS[mode]))


def _settings(screen: router.Screen, a: dict) -> None:
    """「作業の設定」: 文の出し方 and the work's 出典, each saved by its own press."""
    st.html(f'<h3 class="pfb-subhead">{escape(SETTINGS)}</h3>')
    key = f"pfb_review_mode_{a['id']}"
    mode = corrections.mode_of(a)
    st.session_state[key] = mode         # always the stored one (another tab may change it)
    modes = [corrections.CORRECT_LATER, corrections.APPROVE]
    st.radio(theme.L(MODE), modes, format_func=corrections.MODE_WORDS.get, horizontal=True,
             key=key, on_change=_set_mode, args=(a["id"], key), disabled=screen.pending)
    st.caption(theme.L(MODE_NOTES[mode]))
    credits = st.text_area(theme.L(CREDITS), a.get("credits") or "", max_chars=CREDITS_MAX,
                           key=f"pfb_credits_{a['id']}", height=90,
                           placeholder="例：映像 〇〇（CC BY 4.0）")
    st.caption(theme.L(CREDITS_NOTE))
    if st.button(theme.L(CREDITS_SAVE), key=f"pfb_credits_save_{a['id']}",
                 disabled=screen.pending):
        if access.refuse_unless_edit():
            st.rerun()
        screen.repo.set_credits(a["id"], credits)
        flash(CREDITS_SAVED if (credits or "").strip() else CREDITS_CLEARED)
        st.rerun()


def render(screen: router.Screen) -> None:
    """Step 1 of an existing work: editable until analysis starts, then view only; below,
    「作業の設定」 (`_settings`)."""
    repo = screen.repo
    a = repo.get_assignment(screen.progress.assignment_id)
    locked = repo.assignment_has_jobs(a["id"])
    guidance = ("解析が始まったので、この活動の内容は見るだけです。" if locked else
                "活動の内容と評価の観点を見直して、よければ「これで決定して次へ」を押します。")
    with theme.step_card("活動の内容", guidance):
        st.html(f'<dl class="pfb-facts"><dt>{escape(theme.L("活動の名前"))}</dt><dd>{escape(a["title"])}</dd>'
                f'<dt>ファイルの種類</dt><dd>{escape(KINDS[a["kind"]])}</dd>'
                f'<dt>解析の細かさ</dt>'
                f'<dd>{escape(MODEL_LABELS.get(a["model_key"], a["model_key"]))}</dd></dl>')
        if locked:
            st.info(theme.L(LOCKED))
            st.text_area(theme.L("活動の内容"), a["task_text"], disabled=True,
                         key=f"a_view_task_{a['id']}")
            st.text_area(theme.L("評価の観点"), a["rubric_text"] or "（なし）", disabled=True,
                         key=f"a_view_rubric_{a['id']}")
            if st.button(theme.L("続ける"), key="a_continue", type="primary"):
                router.go(None)
                st.rerun()
        else:
            task = st.text_area(theme.L("活動の内容（参加者に伝えた問いや指示）"), a["task_text"],
                                key=f"a_edit_task_{a['id']}")
            rubric = st.text_area(theme.L("評価の観点（任意）"), a["rubric_text"] or "",
                                  key=f"a_edit_rubric_{a['id']}")
            if st.button(theme.L("これで決定して次へ"), key="a_update", type="primary"):
                if _save(repo, a, task, rubric):
                    router.go(None)
                    st.rerun()
        _settings(screen, a)
    theme.help_expander(HELP)
