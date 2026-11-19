"""Step 2, 動画を入れる / 書類を入れる: register the work's videos or documents.

Registration goes through the same glue as the ファイルの登録 page (`register_group_from_picks`,
`register_documents_from_inputs` in ui/common), which calls the intake services unchanged, so
it enqueues the analysis exactly as that page does. After a registration the screen stays here
with a summary, 「続ける」 to the next step and 「もう1件追加」 for more. Files can always be added;
replacing or deleting goes through 管理用の画面 (the old pages).

A work started from the drop zone (its groups went through the content check) adds later
video groups the same way: a drop of the group's videos (「入れて確かめる」), then that group's
own check card on step 1 (`add_videos_to_work`). Works set up before the check keep the
registration form above.
"""
import streamlit as st

from pfb_app.services.content_check import add_videos_to_work, used_video_paths
from pfb_app.services.intake import DOCUMENT_EXTENSIONS, VIDEO_EXTENSIONS, list_inbox_videos
from pfb_app.ui.common import (IntakeFailed, engine_ready, flash, participants_table,
                               register_documents_from_inputs, register_group_from_picks,
                               show_intake_error, upload_note)
from pfb_app.ui.wizard import router, theme

# the analysis method of a video group (submissions.meta["engine"]); tracking is recommended
METHODS = {"pfb3x3": "トラッキング方式（おすすめ）", "paper": "論文方式"}
METHOD_CAPTIONS = ["人物を自動で追跡し、あとで写真を見て参加者とつなぎます。",
                   "外見と位置の説明を手がかりに、人物を見分けます。"]
TRACKING_HINT = ("トラッキング方式では、外見と初期位置は書かなくてもかまいません。参加者がちょうど"
                 "4人で、全員の初期位置を「左前・左奥・右前・右奥」のどれか（重ならないように）で"
                 "書くと、つなぎ方の初期値が入ります。それ以外はあなたがすべて選びます。")
PAPER_HINT = ("論文方式では、外見と初期位置（カメラごと）を手がかりに人物を見分けます。"
              "できるだけ書いてください。")
PEOPLE_CAPTION = ("空欄のままにすると、下の表の参加者の人数だけを追跡します。活動の間いちばん長く"
                  "写っているその人数だけに番号を付け、背景の人は追跡しません。見つかった全員に"
                  "番号を付けるときは0を入れてください。")
SOURCES = {"upload": "このパソコンから送る", "inbox": "サーバーの受け取り用フォルダから選ぶ"}
MORE = "もう1件追加"

REPORT = "_pfb_intake_report"

VIDEO_HELP = [
    "1つのグループ（同じ活動を撮った動画）ごとに登録します。カメラが複数あれば一緒に入れます。",
    "参加者の表に、評価する人の名前を書きます。追跡する人数は、空欄なら参加者の人数になります。",
    "「登録して解析を始める」を押すと解析が始まります。ほかのグループは「もう1件追加」で入れます。",
    "うまくいかないとき：登録できなかったときは、出てきた理由を読みます。動画の形式と大きさ"
    "（入れる欄の下に書いてあります）を確かめて、もう一度入れてください。",
]
DROP_HELP = [
    "同じ活動の別の動画（別の時間や別のグループ）を入れます。同じ場面を複数のカメラで撮った"
    "動画は、まとめて入れます。",
    "入れると中身を自動で確かめ、活動を決める画面で、解析の範囲や名簿を決めてから解析を始めます。",
    "うまくいかないとき：入れられなかったときは、出てきた理由を読みます。動画の形式と大きさ"
    "（入れる欄の下に書いてあります）を確かめて、もう一度入れてください。",
]
DOCUMENT_HELP = [
    "参加者の書類（PDFまたはWord）をまとめて選びます。名前はファイル名から読み取ります。",
    "名簿のCSV（「name」列）を入れると、名前の読み取りが確かになります。",
    "「登録して解析を始める」を押すと解析が始まります。名前を直すときは管理用の画面を使います。",
    "うまくいかないとき：読めなかった書類は理由が出ます。PDFかWordで保存し直して、"
    "もう一度入れてください。",
]


def _adding(aid: int) -> bool:
    return st.session_state.get(router.ADDING) == aid


def _registered(message: str, raw: bool = False) -> None:
    """After a registration: stay on step 2 and show the summary with 「続ける」."""
    st.session_state.pop(router.ADDING, None)
    flash(message, raw=raw)
    router.go(2)
    st.rerun()


def _summary(screen: router.Screen) -> None:
    progress = screen.progress
    router.group_list(progress.subs)
    if progress.kind == "document":
        router.detail_link("submissions", "名前を直すときは管理用の画面へ")
    left, right = st.columns(2)
    with left:
        if st.button(theme.L("続ける"), key="f_continue", type="primary", width="stretch"):
            router.go(None)
            st.rerun()
    with right:
        if st.button(theme.L(MORE), key="f_more", icon=":material/add:", width="stretch"):
            st.session_state[router.ADDING] = progress.assignment_id
            st.rerun()


def _cancel_button() -> None:
    if st.button(theme.L("やめる"), key="f_cancel"):
        st.session_state.pop(router.ADDING, None)
        st.rerun()


# --- video ----------------------------------------------------------------------------------

def _video_form(screen: router.Screen) -> None:
    repo, settings, aid = screen.repo, screen.settings, screen.progress.assignment_id
    engine = st.radio(theme.L("解析の方式"), list(METHODS), format_func=METHODS.get, key="v_engine",
                      captions=[theme.L(c) for c in METHOD_CAPTIONS], horizontal=True)
    if engine == "pfb3x3" and not engine_ready(settings):
        st.warning(theme.L("トラッキング方式の準備がまだ済んでいません。登録はできますが、人を見つける"
                   "段階で止まります。管理者に準備を頼んでください。"))
    count = int(st.number_input(theme.L("カメラの台数"), 1, 4, 1, key="v_count"))
    source = st.radio(theme.L("動画の選び方"), list(SOURCES), format_func=SOURCES.get, key="v_source",
                      horizontal=True)
    from_inbox = source == "inbox"
    inbox = list_inbox_videos(settings) if from_inbox else []
    if from_inbox:
        st.caption(f"サーバーの受け取り用フォルダにある動画です（{len(inbox)}件）。"
                   "選んだ動画はコピーせず、そのまま使います。")
    with st.form("v_group", enter_to_submit=False):
        label = st.text_input(theme.L("グループ名"), key="v_label", placeholder="例：A班")
        picks = []
        for i in range(1, count + 1):
            if from_inbox:
                picks.append(st.selectbox(f"カメラ{i}の動画", inbox, index=None,
                                          format_func=lambda p: p.name, key=f"v_inbox{i}",
                                          placeholder="選んでください" if inbox
                                          else "動画がありません"))
            else:
                picks.append(st.file_uploader(
                    f"カメラ{i}の動画", type=[e.lstrip(".") for e in VIDEO_EXTENSIONS],
                    key=f"v_upload{i}"))
        if not from_inbox:
            st.caption(upload_note("MP4・MOVなどの動画ファイル"))
        c1, c2 = st.columns(2)
        start = c1.number_input("活動の始まり（秒）", min_value=0.0, value=0.0, key="v_start")
        end = c2.number_input("活動の終わり（秒、0なら最後まで）", min_value=0.0, value=0.0,
                              key="v_end")
        transcript = st.text_area(theme.L("文字起こし（任意、「00:05 発言」の形式）"), key="v_transcript")
        auto = st.checkbox(theme.L("文字起こしがなければ、音声から自動で作る"), value=True, key="v_auto")
        people = None
        if engine == "pfb3x3":
            people = st.number_input(theme.L("追跡する人数（空欄なら参加者の人数、0なら制限しない）"),
                                     min_value=0, max_value=20, value=None, step=1,
                                     key="v_people")
            st.caption(theme.L(PEOPLE_CAPTION))
        st.write(theme.L("参加者（初期位置は、カメラごとに書いてください）"))
        st.caption(TRACKING_HINT if engine == "pfb3x3" else PAPER_HINT)
        table = st.data_editor(participants_table(count), num_rows="dynamic",
                               key=f"v_participants{count}")
        submitted = st.form_submit_button(theme.L("登録して解析を始める"), type="primary",
                                          width="stretch")
    if screen.progress.subs:
        _cancel_button()
    if not submitted:
        return
    try:
        register_group_from_picks(repo, settings, aid, label, picks, from_inbox, start, end,
                                  transcript, auto, table, engine, people)
    except IntakeFailed as e:
        show_intake_error(e)
        return
    _registered(f"「{label.strip() or '無題'}」を登録しました。解析を始めています。",
                raw=True)     # the label is the user's own text


def _uses_check(screen: router.Screen) -> bool:
    """The work's groups went through the content check (a work started from the drop zone)."""
    return any("check" in (s["meta"] or {})
               for s in screen.snap.list_submissions(screen.progress.assignment_id))


def _video_drop(screen: router.Screen) -> None:
    """A later group of a checked work: drop its videos; its check card follows on step 1."""
    from pfb_app.ui.wizard import step_drop   # noqa: PLC0415
    repo, settings, aid = screen.repo, screen.settings, screen.progress.assignment_id
    with theme.step_card("もう1つ動画を入れましょう",
                         "同じ活動の別の動画を入れると、中身を確かめてからこの活動に加えます。"):
        source, picks = step_drop.pick_files(settings, "drop_more", documents=False,
                                             used=used_video_paths(repo))
        st.caption(theme.L("同じ場面を複数のカメラで撮った動画は、まとめて入れます（4つまで）。"))
        clicked = st.button(step_drop.PRIMARY, key="drop_more_go", type="primary",
                            disabled=not picks, width="stretch")
        _cancel_button()
    theme.help_expander(DROP_HELP)
    if not clicked:
        return
    sid = step_drop.drop_videos(settings, picks, source == "inbox",
                                lambda paths: add_videos_to_work(repo, settings, aid, paths))
    if sid is None:
        return
    for key in ("drop_more_source", "drop_more_files", "drop_more_inbox"):
        st.session_state.pop(key, None)
    step_drop.big_upload.reset("drop_more")
    st.session_state.pop(router.ADDING, None)
    flash(step_drop.VIDEO_DROPPED)
    router.go(None)                    # the derived step: this group's check card
    router.state()["submission_id"] = None
    st.rerun()


def _video(screen: router.Screen) -> None:
    aid = screen.progress.assignment_id
    if screen.progress.subs and not _adding(aid):
        with theme.step_card("動画を入れました",
                             "ほかのグループがあれば「もう1件追加」で入れ、なければ「続ける」を"
                             "押します。"):
            _summary(screen)
        theme.help_expander(VIDEO_HELP)
        return
    if _uses_check(screen):
        _video_drop(screen)
        return
    title = "もう1グループ入れましょう" if screen.progress.subs else "動画を入れましょう"
    with theme.step_card(title, "1つのグループの動画（カメラが複数あるときは、同じ活動を撮った"
                                "動画すべて）と参加者を入れます。"):
        _video_form(screen)
    theme.help_expander(VIDEO_HELP)


# --- documents ------------------------------------------------------------------------------

def _show_report() -> None:
    """What the last registration could not take: unreadable files and repeated names."""
    report = st.session_state.pop(REPORT, None)
    if report is None:
        return
    for name, message in report.errors:
        st.error(f"{name}: {message}")
    if report.duplicate_names:
        st.warning("同じ名前の書類があります: " + "、".join(report.duplicate_names)
                   + "。管理用の画面で名前を直してください。")


def _document_form(screen: router.Screen) -> None:
    repo, settings, aid = screen.repo, screen.settings, screen.progress.assignment_id
    uploads = st.file_uploader(theme.L("書類（PDFまたはWord、いくつでも）"),
                               type=[e.lstrip(".") for e in DOCUMENT_EXTENSIONS],
                               accept_multiple_files=True, key="d_files")
    st.caption(upload_note("PDFかWordのファイル（いくつでも）"))
    roster_file = st.file_uploader(theme.L("名簿のCSV（任意、「name」列）"), type=["csv"], key="d_roster")
    st.caption(upload_note("CSVファイル"))
    clicked = st.button(theme.L("登録して解析を始める"), key="d_register", type="primary",
                        disabled=not uploads, width="stretch")
    if screen.progress.subs:
        _cancel_button()
    if not clicked:
        return
    try:
        report = register_documents_from_inputs(repo, settings, aid, uploads, roster_file)
    except IntakeFailed as e:
        show_intake_error(e)
        return
    st.session_state[REPORT] = report
    if not report.created:
        st.rerun()          # nothing readable: the form again, with the reasons
    _registered(f"{len(report.created)}件の書類を登録しました。解析を始めています。")


def _documents(screen: router.Screen) -> None:
    aid = screen.progress.assignment_id
    _show_report()
    if screen.progress.subs and not _adding(aid):
        with theme.step_card("書類を入れました",
                             "ほかの書類があれば「もう1件追加」で入れ、なければ「続ける」を"
                             "押します。"):
            _summary(screen)
        theme.help_expander(DOCUMENT_HELP)
        return
    with theme.step_card("書類を入れましょう",
                         "参加者の書類をまとめて選ぶと、名前はファイル名から読み取ります。"):
        _document_form(screen)
    theme.help_expander(DOCUMENT_HELP)


def _cleared(screen: router.Screen) -> bool:
    """A work whose videos were removed takes no new files: say so and offer a new work."""
    if not screen.snap.get_assignment(screen.progress.assignment_id).get("media_removed_at"):
        return False
    from pfb_app.services.intake import CLEARED   # noqa: PLC0415
    with theme.step_card("この作業には、もう動画を入れられません", CLEARED):
        if st.button(theme.L(NEW_WORK), key="pfb_files_new_work", type="primary",
                     icon=":material/add:"):
            router.start_new()
            st.rerun()
    return True


NEW_WORK = "新しい作業として始める"


def render(screen: router.Screen) -> None:
    if _cleared(screen):
        return
    if screen.progress.kind == "document":
        _documents(screen)
    else:
        _video(screen)
