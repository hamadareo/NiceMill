"""Step 6 (documents: 5), PDFを受け取る.

The primary 「参加者に渡すPDFを受け取る」 downloads the student PDF (the existing export) and,
once its bytes are made, records the download (`mark_pdf_downloaded`, which the home
screen's 完了 needs); the
secondary 「評価者用（根拠つき）PDF」 downloads the teacher PDF. Both are made only when the
button is pressed (a deferred download), so the screen itself stays quick and the PDF always
has the approvals of that moment. The screen says who is in the PDF and who is not yet, and
why, in plain words.

「データの扱い」: a work whose consent record says 「未確認」 gets a gentle reminder above the
downloads (nothing is blocked). A work whose videos were removed (services.retention) hands
out the PDFs saved before they went (services.kept) while no approval is newer; otherwise its
new PDFs are made as usual, without pictures. It offers no 成長のまとめ page.

解説動画 (ui/wizard/explainer_view.section, video works, 管理者 and 評価する人): pick an approved
person, 「解説動画を作る」 (secondary), progress, the player and its download; then the group's
「追跡の様子」.
"""
import logging
from html import escape
from typing import Callable

import streamlit as st

from pfb_app.domain.models import Kind, Stage, SubjectStatus
from pfb_app.export.pdf import (ExportError, collect_approved, render_student_pdf,
                                render_student_zip, render_teacher_pdf)
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.services import consent, kept, review, short_text
from pfb_app.services.assignments import mark_pdf_downloaded
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import org_settings, plain_view, router, theme

STUDENT = "参加者に渡すPDFを受け取る"
TEACHER = "評価者用（根拠つき）PDF"
EACH = "一人ずつのPDF（ZIPでまとめて）"
TITLE = "PDFを受け取ってください"
GUIDE = "承認した参加者のフィードバックを、参加者に渡すPDFにまとめて受け取ります。"
ONLY_APPROVED = "このPDFには、承認した参加者だけが入ります。ほかの参加者は、承認すると次に受け取るPDFに入ります。"
EVERYONE = "全員の分が入っています。"
NOBODY = "承認した参加者がまだいません。「確認して承認」で承認すると受け取れます。"
AGAIN = "参加者用のPDFは受け取り済みです。承認を変えたときは、もう一度受け取ってください。"
GROWTH = "成長のまとめのページを入れる"
GROWTH_SOME = "承認した文章のある作業が2件以上ある人の分だけ、そのPDFのあとに1ページ増えます（いま{n}人まで）。"
GROWTH_NONE = "いまは、承認した文章のある作業が2件以上ある人がいません（「成長の記録」でつなげます）。"
CANNOT = "PDFを用意できませんでした。もう一度開いても直らないときは、管理者に知らせてください。"
KEPT_NOTE = "この作業の動画は消してあります。消す前に保存したPDFを渡します。"
NO_PICTURES = "この作業の動画は消してあります。新しく作るPDFには、写真が入りません。"

log = logging.getLogger(__name__)

HELP = [
    "「参加者に渡すPDFを受け取る」で、参加者に配るPDFを保存します。参加者ごとに名前と文章、"
    "よかった場面の写真、場面の種類と時間の流れの図が入ります。",
    "「一人ずつのPDF（ZIPでまとめて）」には、一人ずつのPDFと、スマートフォンで開けるページが入ります。",
    "「評価者用（根拠つき）PDF」には、全員の一覧と、どの場面を根拠にしたかが入ります。記録として残せます。",
    "「団体の設定」で、PDFに入れる団体の名前・ロゴ・見出しの色を決められます。",
    "「成長のまとめのページを入れる」をオンにすると、承認した文章のある作業が2件以上ある人に、"
    "作業ごとの記録をまとめたページが1枚増えます（はじめはオフです）。",
    "「解説動画を作る」（動画の作業）で、承認した人のよかった場面ごとに、追跡した枠、AIに見せた範囲と画像、"
    "場面の説明をつけた動画ができます。できあがるまで数分かかることがあります。",
    "「読み手に合わせた文章」で、子ども向けに言いかえた文章をPDFに入れられます。意味は変えず、元の文はいつでも残ります。",
    "まだ入っていない参加者は、「確認して承認」で承認すると次に受け取るPDFに入ります。",
    "参加者に渡す前に、受け取ったPDFを開いて確かめてください。",
    "うまくいかないとき：PDFを受け取れないときは、画面を開き直してください。直らないときは、"
    "管理者に知らせてください。",
]

# why a student is not in the PDF yet
WHY_NOT = {SubjectStatus.review: "確認待ち", SubjectStatus.insufficient: "場面が少ない",
           SubjectStatus.processing: "解析中", SubjectStatus.waiting_gpu: "解析の順番待ち",
           SubjectStatus.pending: "解析待ち", SubjectStatus.failed: "止まっています"}
WHY_NOT_SHORT = "短い文の承認待ち"   # too few scenes, with a short text offered on step 5 (L6)


def pdf_data(assignment_id: int, which: str, growth: bool = False) -> Callable[[], bytes]:
    """The bytes of the student ("student") or teacher PDF, or the one-person ZIP ("each"),
    made when they are asked for (on its own connection: a deferred download runs on another
    thread). A handout download (the PDF or the ZIP) is recorded only once its bytes are made. A PDF that cannot be made (nobody
    approved any more, say) is logged and the download fails (no empty file, no record)."""
    render = {"student": render_student_pdf, "each": render_student_zip}.get(
        which, render_teacher_pdf)
    options = {"growth": True} if growth and which in ("student", "each") else {}

    def make() -> bytes:
        repo = get_repo()
        try:
            data = render(repo, assignment_id, **options)
            if which in ("student", "each"):
                mark_pdf_downloaded(repo, assignment_id)
            return data
        except ExportError as e:
            log.warning("the %s PDF of assignment %s could not be made: %s", which,
                        assignment_id, e)
            raise
        finally:
            repo.conn.close()
    return make


def kept_data(assignment_id: int, which: str, record: kept.Kept) -> Callable[[], bytes]:
    """As pdf_data, from the PDFs saved before the work's videos were removed."""
    def make() -> bytes:
        settings = get_settings()
        if which == "student":
            data = kept.student_pdf(settings, assignment_id, record)
        elif which == "each":
            data = kept.student_zip(settings, assignment_id, record)
        else:
            data = kept.evaluator_pdf(settings, assignment_id, record)
            if data is None:
                raise ExportError("保存した評価者用のPDFがありません")
        if which in ("student", "each"):
            record_download(assignment_id)
        return data
    return make


def record_download(assignment_id: int) -> None:
    """Record a student-PDF download on its own connection (管理用の画面's PDF出力, whose bytes
    are made before its download button is shown)."""
    repo = get_repo()
    try:
        mark_pdf_downloaded(repo, assignment_id)
    finally:
        repo.conn.close()


def exportable(repo: Repo, assignment_id: int) -> int | None:
    """How many approved feedback texts the PDFs would hold (0: nothing to export), or None
    when they cannot be collected (the cause is logged)."""
    try:
        return len(collect_approved(repo, assignment_id))
    except Exception:   # noqa: BLE001 - keep the screen alive; the download stays off
        log.exception("collecting the approved feedback for the PDFs failed")
        return None


def who(repo: Repo, assignment_id: int,
        kind: Kind) -> tuple[list[tuple[int, str]], list[tuple[str, str]]]:
    """([(subject id, name)] in the PDF, [(name, why not yet)])."""
    included, waiting = [], []
    for sub in repo.list_submissions(assignment_id):
        for subject in repo.list_subjects(sub["id"]):
            if kind == Kind.video and not is_evaluated(subject):
                continue
            status = review.subject_status(repo, subject["id"])
            if status == SubjectStatus.approved:
                included.append((subject["id"], subject["display_name"]))
            elif status == SubjectStatus.insufficient and short_text.offer(
                    repo, subject, repo.latest_run(sub["id"], Stage.pfb, subject_id=subject["id"])):
                waiting.append((subject["display_name"], WHY_NOT_SHORT))
            else:
                waiting.append((subject["display_name"], WHY_NOT[status]))
    return included, waiting


def _names(title: str, names: list[str]) -> None:
    items = "".join(f"<li>{escape(n)}</li>" for n in names)
    st.html(f'<div class="pfb-names"><p class="pfb-names__title">{escape(title)}</p>'
            f'<ul class="pfb-names__list">{items}</ul></div>')


def _growth_option(repo: Repo, assignment_id: int, included: list[int]) -> bool:
    """「成長のまとめのページを入れる」 (off by default; export.growth decides who gets it).
    The count is of `included` (the people listed as in the PDF); 「まで」: an earlier work's
    text may have been outdated since, and the PDF then leaves its page out."""
    from pfb_app.export import growth   # noqa: PLC0415
    try:
        count = growth.handout_count(repo, assignment_id, included)
    except Exception:   # noqa: BLE001 - the option is extra: the PDFs never depend on it
        log.exception("counting the growth pages failed")
        return False
    on = st.checkbox(theme.L(GROWTH), value=False, key="pfb_pdf_growth")
    st.caption(theme.L(GROWTH_SOME.format(n=count) if count else GROWTH_NONE))
    return on


def render(screen: router.Screen) -> None:
    repo, progress = screen.snap, screen.progress     # one read of the work for this run
    aid = progress.assignment_id
    included, waiting = who(repo, aid, Kind(progress.kind))
    work = repo.get_assignment(aid)
    title = work["title"]
    removed = bool(work.get("media_removed_at"))
    saved = kept.current(repo, aid) if removed else None
    with theme.step_card(TITLE, GUIDE):
        if consent.needs_reminder(work):
            st.info(theme.L(consent.REMINDER), icon=":material/info:")
        if included:
            _names(theme.L(f"PDFに入っている参加者（{len(included)}人）"), [f"{n}さん" for _, n in included])
        if waiting:
            _names(theme.L(f"まだ入っていない参加者（{len(waiting)}人）"),
                   [f"{n}さん（{why}）" for n, why in waiting])
        ready = exportable(repo, aid)
        if ready is None:
            st.error(theme.L(CANNOT))
            st.button(theme.L(STUDENT), key="pfb_pdf_student", type="primary", disabled=True)
        elif not ready:
            st.info(theme.L(NOBODY))
            st.button(theme.L(STUDENT), key="pfb_pdf_student", type="primary", disabled=True)
        else:
            if waiting:
                st.info(theme.L(ONLY_APPROVED))
            else:
                st.success(theme.L(EVERYONE))
            if removed:
                st.caption(theme.L(KEPT_NOTE if saved is not None else NO_PICTURES))
            growth = False if removed else _growth_option(repo, aid, [i for i, _ in included])
            if not removed:
                plain_view.section(repo, work)   # 読み手に合わせた文章 (管理者 and 評価する人)

            def data(which: str, growth: bool = False):
                if saved is not None and (which != "teacher" or saved.evaluator):
                    return kept_data(aid, which, saved)
                return pdf_data(aid, which, growth)
            with keyed.container(horizontal=True, key="pfb_pdf_actions"):
                st.download_button(theme.L(STUDENT), data("student", growth),
                                   file_name=f"{title}_{theme.L('参加者')}用.pdf", mime="application/pdf",
                                   key="pfb_pdf_student", type="primary",
                                   icon=":material/download:")
                st.download_button(theme.L(TEACHER), data("teacher"),
                                   file_name=f"{title}_{theme.L('評価者')}用.pdf", mime="application/pdf",
                                   key="pfb_pdf_teacher", icon=":material/download:")
                st.download_button(theme.L(EACH), data("each", growth),
                                   file_name=f"{title}_{theme.L('参加者')}ごと.zip",
                                   mime="application/zip", key="pfb_pdf_each",
                                   icon=":material/folder_zip:")
            if progress.complete:
                st.caption(theme.L(AGAIN))
    from pfb_app.ui.wizard import ask_view, dashboard, explainer_view   # noqa: PLC0415
    explainer_view.section(repo, work)    # 解説動画 (video works; 管理者 and 評価する人)
    dashboard.entry_link(work)            # 試合・活動のまとめ
    ask_view.entry_link(work)             # ききたいことを聞く
    if access.is_admin():                 # 団体の設定: 管理者 only
        org_settings.render(screen.settings)
    theme.help_expander(HELP)
