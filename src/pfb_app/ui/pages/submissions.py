"""ファイルの登録：書類の一括アップロード・名簿・名前の修正、動画グループ。"""
import streamlit as st

from pfb_app.pipeline.common import submission_engine
from pfb_app.services.intake import (DOCUMENT_EXTENSIONS, VIDEO_EXTENSIONS, list_inbox_videos,
                                     rename_subject)
from pfb_app.ui.common import (IntakeFailed, engine_ready, flash, get_repo, get_settings,
                               participants_table, register_documents_from_inputs,
                               register_group_from_picks, select_assignment, show_flash,
                               show_intake_error, upload_note)
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

repo = get_repo()
settings = get_settings()
st.title("ファイルの登録")
show_flash("school")
assignment = select_assignment(repo)


def documents_section(assignment: dict) -> None:
    st.subheader("書類をアップロード")
    uploads = st.file_uploader("書類（PDFまたはWord、複数可）",
                               type=[e.lstrip(".") for e in DOCUMENT_EXTENSIONS],
                               accept_multiple_files=True)
    st.caption(upload_note("PDFかWordのファイル（いくつでも）"))
    roster_file = st.file_uploader("名簿のCSV（任意、「name」列）", type=["csv"])
    st.caption(upload_note("CSVファイル"))
    if st.button("登録", disabled=not uploads):
        try:
            report = register_documents_from_inputs(repo, settings, assignment["id"], uploads,
                                                    roster_file)
        except IntakeFailed as e:
            show_intake_error(e)
            return
        st.session_state["intake_report"] = report
        st.rerun()
    report = st.session_state.pop("intake_report", None)
    if report is not None:
        if report.created:
            st.success(f"{len(report.created)}件を登録しました。")
        for name, message in report.errors:
            st.error(f"{name}: {message}")
        if report.duplicate_names:
            st.warning("同じ名前の書類があります: " + "、".join(report.duplicate_names))

    rows = [{"subject_id": s["id"], "登録したファイル": sub["meta"].get("original_filename", sub["label"]),
             "名前": s["display_name"]}
            for sub in repo.list_submissions(assignment["id"])
            for s in repo.list_subjects(sub["id"])]
    if not rows:
        return
    st.subheader("推定した名前の確認・修正")
    names = [r["名前"] for r in rows]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        st.warning("名前が重複しています: " + "、".join(duplicates))
    edited = st.data_editor(rows, hide_index=True, disabled=["subject_id", "登録したファイル"],
                            column_config={"subject_id": None}, key=f"names-{assignment['id']}-{len(rows)}")
    if st.button("名前を保存"):
        changed = 0
        for old, new in zip(rows, edited):
            if new["名前"] != old["名前"]:
                try:
                    rename_subject(repo, old["subject_id"], new["名前"])
                    changed += 1
                except ValueError as e:
                    st.error(str(e))
                    return
        flash(f"{changed}件の名前を保存しました。")
        st.rerun()


ENGINE_LABELS = {"paper": "論文方式（標準）", "pfb3x3": "トラッキング方式（PFB3x3）"}
PAPER_HINT = ("論文方式では、モデルが外見と初期位置（カメラごと）を手がかりに人物を特定します。"
              "できるだけ書いてください。")
TRACKING_HINT = ("トラッキング方式（PFB3x3）では、外見と初期位置は任意です。人物を追跡したあと、"
                 "あなたが人物一覧の画像を見て、番号の付いた人物と参加者を対応づけます。"
                 "ちょうど4人の参加者全員について初期位置を「左前・左奥・右前・右奥」のいずれか"
                 "（重ならないように）で書くと（追跡する人数は参加者の人数のまま）、"
                 "対応づけに初期値が入ります。"
                 "それ以外はあなたがすべて選びます。")


def video_section(assignment: dict) -> None:
    st.subheader("動画グループを登録")
    engine = st.radio("分析の方式", list(ENGINE_LABELS), format_func=ENGINE_LABELS.get,
                      horizontal=True, key="engine")
    if engine == "pfb3x3" and not engine_ready(settings):
        st.warning("トラッキング方式のエンジンがまだセットアップされていません"
                   "（scripts/setup_pfb3x3_engine.sh）。登録はできますが、人物の追跡は失敗します。")
    count = int(st.number_input("カメラの台数", 1, 4, 1))
    inbox = list_inbox_videos(settings)
    source = st.radio("動画の選び方", ["ブラウザからアップロード", "inboxフォルダから選ぶ"], horizontal=True)
    from_inbox = source == "inboxフォルダから選ぶ"
    if from_inbox:
        st.caption(f"inboxフォルダ: {settings.inbox_dir}（{len(inbox)}件）。選んだ動画はコピーせず、そのまま参照します。")
    with st.form("video_group"):
        label = st.text_input("グループ名")
        picks = []
        for i in range(1, count + 1):
            if from_inbox:
                picks.append(st.selectbox(f"カメラ{i}の動画", inbox, index=None,
                                          format_func=lambda p: p.name, key=f"inbox{i}"))
            else:
                picks.append(st.file_uploader(f"カメラ{i}の動画", type=[e.lstrip(".") for e in VIDEO_EXTENSIONS],
                                              key=f"upload{i}"))
                st.caption(upload_note("MP4・MOVなどの動画ファイル"))
        c1, c2 = st.columns(2)
        start = c1.number_input("開始（秒）", min_value=0.0, value=0.0)
        end = c2.number_input("終了（秒、0なら最後まで）", min_value=0.0, value=0.0)
        transcript = st.text_area("文字起こし（任意、「00:05 発言」の形式）")
        auto = st.checkbox("文字起こしがなければ、音声から自動で作る", value=True)
        people = None
        if engine == "pfb3x3":
            people = st.number_input("追跡する人数（空欄なら参加者の人数、0なら制限しない）",
                                     min_value=0, max_value=20, value=None, step=1,
                                     key="max_people")
            st.caption("空欄のままにすると、下の表の参加者の人数だけを追跡します。活動の間いちばん"
                       "長く写っているその人数だけに番号を付け、背景の人は追跡しません。"
                       "見つかった全員に番号を付けるときは0を入れてください。")
        st.write("参加者（初期位置は、カメラごとに書いてください）")
        st.caption(TRACKING_HINT if engine == "pfb3x3" else PAPER_HINT)
        table = st.data_editor(participants_table(count), num_rows="dynamic", key="participants")
        submitted = st.form_submit_button("登録")
    if not submitted:
        return
    try:
        register_group_from_picks(repo, settings, assignment["id"], label, picks, from_inbox,
                                  start, end, transcript, auto, table, engine, people)
    except IntakeFailed as e:
        show_intake_error(e)
        return
    flash("動画グループを登録しました。")
    st.rerun()


def video_list(assignment: dict) -> None:
    subs = repo.list_submissions(assignment["id"])
    if not subs:
        return
    st.subheader("登録済みの動画グループ")
    for sub in subs:
        with st.expander(sub["label"]):
            cams = sub["meta"].get("cameras", {})
            st.write("分析の方式: " + ENGINE_LABELS.get(submission_engine(sub), "不明"))
            st.write("カメラ: " + "、".join(f"{c}（{p.rsplit('/', 1)[-1]}）" for c, p in cams.items()))
            st.dataframe([{"参加者ID": s["participant_id"], "表示名": s["display_name"],
                           "外見": s["appearance"]} for s in repo.list_subjects(sub["id"])],
                         hide_index=True)


if assignment is not None:
    if assignment["kind"] == "document":
        documents_section(assignment)
    else:
        video_section(assignment)
        video_list(assignment)
