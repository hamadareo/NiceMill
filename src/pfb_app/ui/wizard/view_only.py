"""A work as a 見るだけの人 sees it (router, in place of the step screens): the approved feedback
texts, who is not in them yet and why, and the participants' PDF to download. Nothing can be
changed here, and the download is not recorded as handed out (that is the coach's step)."""
from html import escape

import streamlit as st

from pfb_app.domain.models import Kind
from pfb_app.export.pdf import ExportError, collect_approved, render_student_pdf
from pfb_app.services import kept
from pfb_app.storage.db import Repo
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import step_pdf, theme
from pfb_app.ui.wizard.progress import WorkProgress

NOTE = "見るだけの人として開いています。文章を直したり承認したりはできません。"
TITLE = "承認したフィードバック"
GUIDE = "承認された文章を読めます。参加者に渡すPDFも受け取れます。"
NONE_YET = "承認されたフィードバックは、まだありません。"
WAITING = "まだ入っていない参加者"
CSS = """<style>
.pfb-viewonly { margin: 0.5rem 0; padding: 0.75rem 1rem; border: 1px solid var(--pfb-line);
  border-radius: 9px; background: var(--pfb-paper, #fff); }
.pfb-viewonly__name { margin: 0 0 0.25rem; font-weight: 700; color: var(--pfb-ink); }
.pfb-viewonly__text { margin: 0; white-space: pre-wrap; line-height: 1.7; }
</style>"""


def _pdf(aid: int, saved: kept.Kept | None):
    def make() -> bytes:
        if saved is not None:
            return kept.student_pdf(get_settings(), aid, saved)
        repo = get_repo()
        try:
            return render_student_pdf(repo, aid)
        finally:
            repo.conn.close()
    return make


def render(snap: Repo, progress: WorkProgress) -> None:
    aid = progress.assignment_id
    st.info(theme.L(NOTE), icon=":material/visibility:")
    try:
        entries = collect_approved(snap, aid)
    except (ExportError, KeyError):
        entries = []
    _, waiting = step_pdf.who(snap, aid, Kind(progress.kind))
    work = snap.get_assignment(aid)
    saved = kept.current(snap, aid) if work.get("media_removed_at") else None
    st.html(CSS)
    with theme.step_card(theme.L(TITLE), theme.L(GUIDE)):
        if not entries:
            st.info(theme.L(NONE_YET))
        for e in entries:
            st.html(f'<div class="pfb-viewonly"><p class="pfb-viewonly__name">'
                    f'{escape(e.display_name)}さん</p>'
                    f'<p class="pfb-viewonly__text">{escape(e.pfb_text)}</p></div>')
        if waiting:
            names = "".join(f"<li>{escape(n)}さん（{escape(why)}）</li>" for n, why in waiting)
            st.html(f'<div class="pfb-names"><p class="pfb-names__title">'
                    f'{escape(theme.L(WAITING))}（{len(waiting)}人）</p>'
                    f'<ul class="pfb-names__list">{names}</ul></div>')
        if entries:
            st.download_button(theme.L(step_pdf.STUDENT), _pdf(aid, saved),
                               file_name=f"{work['title']}_{theme.L('参加者')}用.pdf",
                               mime="application/pdf", key="pfb_view_pdf", type="primary",
                               icon=":material/download:")
