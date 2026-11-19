"""活動の一覧・作成・削除。"""
import streamlit as st

from pfb_app.services import deletion
from pfb_app.ui.common import (KIND_LABELS, MODEL_LABELS, IntakeFailed,
                               create_assignment_from_inputs, flash, get_repo, get_settings,
                               show_flash, show_intake_error, upload_note)
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

ACTIVE = ("解析が待機中または実行中のため、ここでは削除できません。"
          "ホームの「削除」から「解析を止めて削除」を使うか、解析が終わってから削除してください。")

repo = get_repo()
settings = get_settings()
st.title("活動")
show_flash("school")

assignments = repo.list_assignments()
if assignments:
    st.dataframe(
        [{"ID": a["id"], "活動名": a["title"], "種類": KIND_LABELS[a["kind"]],
          "モデル": MODEL_LABELS.get(a["model_key"], a["model_key"]),
          "登録数": len(repo.list_submissions(a["id"]))} for a in assignments],
        hide_index=True)
else:
    st.info("活動はまだありません。下のフォームから作成してください。")

st.subheader("活動を作成")
with st.form("create_assignment"):
    title = st.text_input("活動名")
    kind = st.radio("種類", list(KIND_LABELS), format_func=KIND_LABELS.get, horizontal=True)
    task_text = st.text_area("活動の内容")
    rubric_text = st.text_area("評価の観点（任意）")
    rubric_file = st.file_uploader("評価の観点のファイル（任意、PDFまたはWord）",
                                   type=["pdf", "docx", "doc"])
    st.caption(upload_note("PDFかWordのファイル"))
    model_key = st.selectbox("モデル", list(settings.models), format_func=lambda k: MODEL_LABELS.get(k, k))
    require_identity = st.checkbox("動画では、人物の確認後に行動の観察を始める", value=True)
    submitted = st.form_submit_button("作成")
if submitted:
    if not title.strip() or not task_text.strip():
        st.error("活動名と活動の内容を入力してください。")
    else:
        try:
            create_assignment_from_inputs(repo, settings, title, kind, task_text, rubric_text,
                                          rubric_file, model_key, require_identity)
        except IntakeFailed as e:
            show_intake_error(e)
        else:
            flash("活動を作成しました。")
            st.rerun()

if assignments:
    st.subheader("活動を削除")
    target = st.selectbox("削除する活動", [a["id"] for a in assignments], key="delete_target",
                          format_func=lambda i: next(a["title"] for a in assignments if a["id"] == i))
    confirm = st.checkbox("登録したファイルと結果もすべて削除されることを理解しました", key="confirm_delete")
    if st.button("削除", disabled=not confirm):
        try:   # the same deletion as the home screen's (folder, own uploads, audit)
            deletion.delete_work(repo, settings, target)
        except KeyError:
            flash("この活動はすでに削除されています。")
            st.rerun()
        except deletion.WorkActive:
            st.error(ACTIVE)
        else:
            flash("活動を削除しました。")
            st.rerun()
