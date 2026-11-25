"""PDF出力：評価者用と参加者用。"""
import streamlit as st

from pfb_app.domain import profiles
from pfb_app.domain.vocab import localize
from pfb_app.export.pdf import (ExportError, collect_approved, render_student_pdf,
                                render_student_zip, render_teacher_pdf)
from pfb_app.ui.common import get_repo, select_assignment
from pfb_app.ui.wizard.step_pdf import record_download
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

repo = get_repo()
st.title("PDF出力")
assignment = select_assignment(repo)
if assignment is not None:
    if assignment.get("review_mode") == "correct_later":    # 「あとから訂正」: no approvals
        from pfb_app.export.sheets import build_report      # noqa: PLC0415
        approved = build_report(repo, assignment["id"], with_pictures=False).exported
        st.write(f"PDFに入る文（あとから訂正）: {len(approved)}件")
        empty = "PDFに入れられる文がありません。"
    else:
        approved = collect_approved(repo, assignment["id"])
        st.write(f"承認済み: {len(approved)}件")
        empty = "承認済みのフィードバックがありません。「確認・承認」ページで承認してください。"
    if not approved:
        st.info(empty)
    else:
        kinds = {"teacher": ("評価者用PDF", render_teacher_pdf),
                 "student": ("参加者用PDF", render_student_pdf),
                 "each": ("一人ずつのPDF（ZIPでまとめて）", render_student_zip)}
        wording = profiles.of_work(assignment)   # the work's profile words, as in the wizard
        for key, (label, render) in kinds.items():
            # the work's profile words (sports: コーチ用 / 選手用)
            label = localize(label, wording)
            # Generated on click and offered at once: nothing is cached across reruns, so an
            # approval change can never leave outdated bytes available for download.
            if st.button(f"{label}を作成", key=f"make-{key}"):
                try:
                    pdf = render(repo, assignment["id"])
                except ExportError as e:
                    st.error(str(e))
                else:
                    # a handout download is recorded (the wizard's 完了 needs it)
                    record = ({"on_click": record_download, "args": (assignment["id"],)}
                              if key in ("student", "each") else {})
                    if key == "each":
                        people = localize("参加者", wording)
                        name, mime = f"{assignment['title']}_{people}ごと.zip", "application/zip"
                    else:
                        name = f"{assignment['title']}_{label.removesuffix('PDF')}.pdf"
                        mime = "application/pdf"
                    st.download_button(f"{label}をダウンロード", pdf, file_name=name, mime=mime,
                                       key=f"dl-{key}", **record)
