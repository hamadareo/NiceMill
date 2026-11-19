"""Wizard: the drop zone (first screen of a new work) and the content check card on step 1.

AppTest on ui/app.py over a synthetic database: generated videos (flat grey frames) and
invented names only; the detect runs are canned outputs, no model or engine runs.
"""
import re

import pytest

from pfb_app.domain.models import JobStatus, Stage
from pfb_app.services import content_check
from pfb_app.services.content_check import (CheckChoice, add_videos_to_work, confirm_check,
                                            start_work_from_videos)
from pfb_app.services.review import BusyError
from pfb_app.services.roster import parse_roster_text
from pfb_app.ui.wizard import live, step_check, step_drop
from pfb_app.ui.wizard.progress import CHECKED, CHECKING, work_progress
from test_content_check import _finish_detect_job, detect_done, probe
from test_wizard_a11y import ENGLISH_UI, INTERNAL, _rendered
from test_wizard_ui_basic import (PRIMARY_LABELS, app, button, engine_set_up, env,  # noqa: F401
                                  html, make_video, no_internal_terms, ok, pdf_bytes, primaries,
                                  screen_text, shown_step)

SOCCER = "サッカーの試合のようです（推定）。映っている人は約14人。"
DISAGREE = "この動画はグループ活動のようですが、この試合・練習はサッカーの試合として進めます。"
NEW = {"assignment_id": None, "step": 1, "submission_id": None}


def video_bytes(settings, name="試合A.mp4") -> bytes:
    return make_video(settings.data_dir.parent / "src" / name).read_bytes()


def held_work(settings, repo, name="試合A.mp4"):
    """A work started from one dropped video, its check still running: (aid, sid)."""
    return start_work_from_videos(repo, settings, [make_video(settings.inbox_dir / name)],
                                  "standard")


def ready_work(settings, repo, **detect):
    aid, sid = held_work(settings, repo)
    detect_done(repo, sid, **detect)
    return aid, sid


def sports_choice(**kw) -> CheckChoice:
    base = dict(title="練習試合", activity="サッカーの試合", rubric_text="", domain="sports",
                sport="soccer", method="pfb3x3", max_people=16, target="person",
                start_sec=0.0, end_sec=None, clock_offset_min=0.0)
    base.update(kw)
    return CheckChoice(**base)


def primary(at):
    (b,) = [b for b in at.button if b.proto.type == "primary"]
    return b


def number(at, label: str):
    return next(w for w in at.number_input if w.label == label)


def jobs(repo, sid, stage):
    return [j for j in repo.submission_job_rows(sid) if j["stage"] == stage]


def drop(at, *files):
    """Put files on the drop zone and press 「入れて確かめる」."""
    at.file_uploader(key="drop_files").set_value(list(files)).run()
    ok(at)
    at.button(key="drop_go").click().run()
    return ok(at)


# --- the drop zone ----------------------------------------------------------------------------

def test_new_work_shows_drop_zone_title(env):
    at = ok(app())
    at.button(key="pfb_new").click().run()
    ok(at)
    assert step_drop.TITLE == "評価したいファイルを入れてください"
    assert step_drop.TITLE in html(at)
    assert primaries(at) == ["入れて確かめる"]
    assert at.button(key="drop_go").disabled                 # nothing dropped yet
    uploader = at.file_uploader(key="drop_files")
    assert uploader.accept_multiple_files
    types = " ".join(uploader.allowed_type)
    assert all(t in types for t in ("mp4", "mov", "pdf", "docx"))
    assert any(e.label == "この画面の使い方" for e in at.expander)
    assert at.query_params.get("work") == "new"
    no_internal_terms(at)
    # nothing to decide but the files: no 解析の細かさ (it is on the check card), no source
    # radio while the server folder is empty
    assert not at.selectbox and not at.radio
    assert step_drop.PHONE_TIP in html(at) and 'class="pfb-phone-tip"' in html(at)
    styles = "".join(el.proto.body for el in at.get("html"))
    assert "このファイルは入れられません。1ファイル200 MBまで" in styles    # the server's limit


def test_the_phone_tip_and_the_limit_message_copy():
    assert step_drop.PHONE_TIP == ("スマホの4K動画は大きく、送るのに時間がかかります。"
                                   "カメラの設定を「1080p・30fps」にすると、速く送れて解析も速くなります。")
    from pfb_app.ui.common import too_large_text
    assert too_large_text(4096) == ("このファイルは入れられません。1ファイル4096 MBまでの、"
                                    "入れられる種類のファイルを選んでください。")


def test_inbox_option_only_when_it_has_videos_newest_preselected(env):
    import os
    settings, repo = env
    old = make_video(settings.inbox_dir / "b_old.mp4")
    new = make_video(settings.inbox_dir / "a_new.mp4")
    os.utime(old, (1_000_000, 1_000_000))
    at = ok(app(session=NEW))
    source = at.radio(key="drop_source")
    assert source.value == "upload"
    assert source.options == ["この端末から送る", "受け取り用フォルダの動画から選ぶ"]
    assert "受け取り用フォルダに新しい動画があります（2件）。" in [c.value for c in at.caption]
    source.set_value("inbox").run()
    ok(at)
    assert at.multiselect(key="drop_inbox").value == [new]      # the newest, already picked
    assert not at.button(key="drop_go").disabled
    at.button(key="drop_go").click().run()
    ok(at)
    (a,) = repo.list_assignments()
    assert a["model_key"] == "standard"                       # the default, changed on the card
    (sub,) = repo.list_submissions(a["id"])
    assert list(sub["meta"]["cameras"].values()) == [str(new)]


def test_drop_video_creates_held_work_and_opens_check(env):
    settings, repo = env
    at = ok(app(session=NEW))
    drop(at, ("試合A.mp4", video_bytes(settings), "video/mp4"))
    (a,) = repo.list_assignments()
    assert (a["kind"], a["title"], a["domain"]) == ("video", "試合A", "school")
    assert a["require_identity_approval"]
    (sub,) = repo.list_submissions(a["id"])
    assert sub["meta"]["check"] == "pending"
    assert [j["stage"] for j in repo.submission_job_rows(sub["id"])] == ["detect"]
    progress = work_progress(repo, a["id"])
    assert progress.current == 1 and progress.subs[0].step == "assignment"
    assert (progress.subs[0].status, progress.subs[0].reason) == ("processing", CHECKING)
    assert at.session_state["wizard"]["assignment_id"] == a["id"]
    assert shown_step(at) == 1
    assert step_check.CHECKING_TITLE in html(at)
    assert any("中身を確かめています" in s.value for s in at.success)
    assert at.query_params.get("work") == str(a["id"])


def test_drop_videos_from_the_server_folder(env):
    settings, repo = env
    videos = [make_video(settings.inbox_dir / n) for n in ("cam_a.mp4", "cam_b.mp4")]
    at = ok(app(session=NEW))
    at.radio(key="drop_source").set_value("inbox").run()
    at.multiselect(key="drop_inbox").set_value(videos).run()
    at.button(key="drop_go").click().run()
    ok(at)
    (a,) = repo.list_assignments()
    (sub,) = repo.list_submissions(a["id"])
    assert list(sub["meta"]["cameras"].values()) == [str(v) for v in videos]
    assert shown_step(at) == 1


def test_mixed_drop_error(env):
    settings, repo = env
    at = ok(app(session=NEW))
    drop(at, ("試合A.mp4", video_bytes(settings), "video/mp4"),
         ("山田太郎.pdf", pdf_bytes(), "application/pdf"))
    assert [e.value for e in at.error] == [step_drop.MIXED]
    assert step_drop.MIXED == "書類と動画は別々に入れてください"
    assert repo.list_assignments() == []
    assert not (settings.data_dir / "uploads").exists() or \
        not list((settings.data_dir / "uploads").iterdir())


def test_more_than_four_videos_is_refused(env):
    settings, repo = env
    data = video_bytes(settings)
    at = ok(app(session=NEW))
    drop(at, *[(f"c{i}.mp4", data, "video/mp4") for i in range(5)])
    assert any("カメラは4台まで" in e.value for e in at.error)
    assert repo.list_assignments() == []
    uploads = settings.data_dir / "uploads"
    assert not uploads.exists() or not list(uploads.iterdir())


def test_drop_documents_goes_to_form_with_kind_fixed(env):
    settings, repo = env
    at = ok(app(session=NEW))
    drop(at, ("山田太郎.pdf", pdf_bytes(), "application/pdf"),
         ("佐藤花子.pdf", pdf_bytes(), "application/pdf"))
    assert repo.list_assignments() == []                     # nothing yet: the form first
    assert len(at.session_state[step_drop.DROP_DOCS]) == 2
    assert "活動を決めましょう" in html(at)
    assert not [r for r in at.radio if r.key == "a_kind"]    # the kind is fixed by the drop
    assert "書類（レポートや企画書など）" in html(at) and "山田太郎.pdf" in html(at)
    assert primaries(at) == ["これで決定して次へ"]
    no_internal_terms(at)
    at.text_input(key="a_title").input("探究レポート")
    at.text_area(key="a_task").input("調べてまとめる")
    at.button(key="a_save").click().run()
    ok(at)
    (a,) = repo.list_assignments()
    assert (a["kind"], a["title"]) == ("document", "探究レポート")
    names = sorted(s["label"] for s in repo.list_submissions(a["id"]))
    assert names == ["佐藤花子", "山田太郎"]
    assert all(jobs(repo, s["id"], "prepare") for s in repo.list_submissions(a["id"]))
    assert step_drop.DROP_DOCS not in at.session_state
    assert at.session_state["wizard"]["assignment_id"] == a["id"]
    assert shown_step(at) == 3                                # analysing: no detect card
    assert any("2件の書類を登録" in s.value for s in at.success)


def test_dropped_documents_are_forgotten_on_the_way_home(env):
    at = ok(app(session=NEW))
    drop(at, ("山田太郎.pdf", pdf_bytes(), "application/pdf"))
    assert "活動を決めましょう" in html(at)
    at.button(key="pfb_home").click().run()
    ok(at)
    assert step_drop.DROP_DOCS not in at.session_state
    at.button(key="pfb_new").click().run()
    ok(at)
    assert step_drop.TITLE in html(at)


def test_documents_roster_csv_guides_the_names(env):
    settings, repo = env
    at = ok(app(session=NEW))
    drop(at, ("探究山田太郎まとめ.pdf", pdf_bytes(), "application/pdf"))
    at.text_input(key="a_title").input("探究レポート")
    at.text_area(key="a_task").input("調べてまとめる")
    at.file_uploader(key="a_roster").set_value(
        ("名簿.csv", "name\n山田太郎\n".encode(), "text/csv")).run()
    at.button(key="a_save").click().run()
    ok(at)
    (a,) = repo.list_assignments()
    assert [s["label"] for s in repo.list_submissions(a["id"])] == ["山田太郎"]


def test_an_unreadable_documents_roster_creates_nothing(env):
    settings, repo = env
    at = ok(app(session=NEW))
    drop(at, ("山田太郎.pdf", pdf_bytes(), "application/pdf"))
    at.text_input(key="a_title").input("探究レポート")
    at.text_area(key="a_task").input("調べてまとめる")
    at.file_uploader(key="a_roster").set_value(
        ("名簿.csv", "氏名\n山田太郎\n".encode(), "text/csv")).run()
    at.button(key="a_save").click().run()
    ok(at)
    assert any("「name」列" in e.value for e in at.error)
    assert repo.list_assignments() == []
    assert step_drop.DROP_DOCS in at.session_state              # still on the form


def test_redrop_from_the_documents_form(env):
    at = ok(app(session=NEW))
    drop(at, ("山田太郎.pdf", pdf_bytes(), "application/pdf"))
    at.button(key="a_redrop").click().run()
    ok(at)
    assert step_drop.DROP_DOCS not in at.session_state
    assert step_drop.TITLE in html(at)


# --- the check card ---------------------------------------------------------------------------

def test_check_waiting_disables_primary(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 1
    page = html(at)
    assert step_check.CHECKING_TITLE in page
    assert "動画の内容を確かめています。" in page and "試合A.mp4" in page
    assert primaries(at) == ["この内容で始める"]
    assert primary(at).disabled
    assert live.NOTHING_TO_PRESS in [c.value for c in at.caption]
    assert any(e.label == "名簿（任意）" for e in at.expander)  # the roster can be entered now
    assert not [r for r in at.radio if (r.key or "").startswith("chk_profile_")]


def test_check_waiting_moves_on_when_the_check_ends(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    detect_done(repo, sid)
    at.run()                          # what the 10-second watch does once the work moved on
    ok(at)
    assert SOCCER in html(at)
    assert not primary(at).disabled
    progress = work_progress(repo, aid)
    assert (progress.subs[0].status, progress.subs[0].reason) == ("your_turn", CHECKED)
    assert progress.your_turn and progress.turn_step == 1


def facts(at) -> list[tuple[str, str]]:
    (block,) = re.findall(r'<dl class="pfb-facts pfb-checkfacts"[^>]*>(.*?)</dl>', html(at), re.S)
    return re.findall(r"<dt>([^<]*)</dt><dd>([^<]*)</dd>", block)


def change(at):
    (e,) = [e for e in at.expander if e.label == step_check.CHANGE]
    return e


def test_check_ready_shows_sentence_and_prefill(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 1
    assert SOCCER in html(at) and 'class="pfb-verdict__text"' in html(at)
    # the short card: the sentence, three facts, one primary; the rest in 「内容を変える」
    assert facts(at) == [("種目", "サッカー"), ("評価する人", "約14人"), ("解析する部分", "全体（約1分）")]
    assert step_check.CHANGE == "内容を変える" and not change(at).proto.expanded
    assert primaries(at) == ["この内容で始める"] and not primary(at).disabled
    assert at.radio(key=f"chk_profile_{sid}").value == "sports"
    assert at.selectbox(key=f"chk_sport_{sid}").value == "soccer"
    day = content_check.work_day(repo.get_assignment(aid))
    assert at.text_input(key=f"chk_title_{sid}").value == f"サッカーの試合 {day.month}月{day.day}日"
    activity = at.text_area(key=f"chk_activity_{sid}")
    assert (activity.label, activity.value) == ("試合・練習の内容", "サッカーの試合")
    assert at.text_area(key=f"chk_rubric_{sid}").label == "見てほしい観点（任意）"
    # sports: tracking only, so no radio with a single option
    assert not [r for r in at.radio if (r.key or "").startswith("chk_method_")]
    model = at.selectbox(key=f"chk_model_{sid}")
    assert (model.label, model.value, model.options) == ("解析の細かさ", "standard",
                                                         ["標準", "高精度"])
    assert at.number_input(key=f"chk_people_{sid}_sports").value == 16   # p90 of the check
    assert at.number_input(key=f"chk_people_{sid}_sports").max == 30
    assert at.number_input(key=f"chk_start_{sid}").value == 0.0
    assert at.number_input(key=f"chk_end_{sid}").value is None          # a short video
    assert at.text_area(key=f"chk_roster_{sid}_text")                    # inside 内容を変える
    assert [e.label for e in at.expander if e.label != "この画面の使い方"] == [step_check.CHANGE]
    assert not [r for r in at.radio if (r.key or "").startswith("chk_target_")]


def test_the_facts_follow_the_changes(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, duration_s=3000.0)
    at = ok(app(session={"assignment_id": aid}))
    assert facts(at)[2] == ("解析する部分", "はじめの20分")
    at.number_input(key=f"chk_end_{sid}").set_value(10.0).run()
    at.selectbox(key=f"chk_sport_{sid}").set_value("basketball").run()
    ok(at)
    assert facts(at)[0] == ("種目", "バスケットボール")
    assert facts(at)[2] == ("解析する部分", "はじめの10分")
    day = content_check.work_day(repo.get_assignment(aid))
    assert at.text_input(key=f"chk_title_{sid}").value == (
        f"バスケットボールの試合 {day.month}月{day.day}日")      # follows until edited
    at.text_input(key=f"chk_title_{sid}").input("練習試合").run()
    at.selectbox(key=f"chk_sport_{sid}").set_value("soccer").run()
    ok(at)
    assert at.text_input(key=f"chk_title_{sid}").value == "練習試合"


def test_the_model_choice_moves_to_the_card(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.selectbox(key=f"chk_model_{sid}").set_value("high").run()
    primary(at).click().run()
    ok(at)
    assert repo.get_assignment(aid)["model_key"] == "high"


def test_a_coach_given_title_is_kept_on_the_card(env):
    settings, repo = env
    aid = repo.create_assignment("ペーパータワー", "video", "", "", "standard", True)
    sid = add_videos_to_work(repo, settings, aid, [make_video(settings.inbox_dir / "x.mp4")])
    detect_done(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    assert at.text_input(key=f"chk_title_{sid}").value == "ペーパータワー"


def test_scene_change_follows_with_vocabulary_and_prefill(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"chk_profile_{sid}").set_value("general").run()
    ok(at)
    activity = at.text_area(key=f"chk_activity_{sid}")
    assert (activity.label, activity.value) == ("活動の内容", "グループ活動")
    assert at.text_area(key=f"chk_rubric_{sid}").label == "評価の観点（任意）"
    assert at.radio(key=f"chk_method_{sid}_school").options == [
        "トラッキング方式（おすすめ）", "論文方式"]
    assert at.number_input(key=f"chk_people_{sid}_school").max == 20
    assert facts(at)[0] == ("場面", "グループ活動")
    # an edited field is the coach's: it no longer follows the scene
    at.text_area(key=f"chk_activity_{sid}").input("紙の塔づくり").run()
    at.radio(key=f"chk_profile_{sid}").set_value("sports").run()
    ok(at)
    assert at.text_area(key=f"chk_activity_{sid}").value == "紙の塔づくり"


def test_long_video_defaults_to_the_first_20_minutes(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, duration_s=3000.0)
    at = ok(app(session={"assignment_id": aid}))
    assert at.number_input(key=f"chk_end_{sid}").value == 20.0
    assert any("はじめの20分を解析します" in i.value for i in at.info)


def test_motion_note_and_motor_racing_target(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, sport="motor_racing",
                          cameras={"cam1": probe(moving=True, vehicles=(9, 10, 12))})
    at = ok(app(session={"assignment_id": aid}))
    assert any("カメラが大きく動く動画は、人の取り違えが増えます" in w.value for w in at.warning)
    target = at.radio(key=f"chk_target_{sid}")
    assert (target.options, target.value) == (["人", "車・バイク"], "vehicle")
    assert at.number_input(key=f"chk_people_{sid}_sports").label == "追跡する台数（30台まで）"


def test_gpu_busy_card_requires_scene_pick(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, scene=None, skipped="gpu_busy")
    at = ok(app(session={"assignment_id": aid}))
    assert "映っている人は約14人。どんな場面の動画か、下から選んでください。" in html(at)
    assert change(at).proto.expanded                       # the pick is inside: opened
    assert facts(at)[0] == ("場面", "まだ選ばれていません")
    assert at.radio(key=f"chk_profile_{sid}").value is None
    assert primary(at).disabled
    assert step_check.PICK_SCENE in [c.value for c in at.caption]
    at.radio(key=f"chk_profile_{sid}").set_value("sports").run()
    ok(at)
    assert at.selectbox(key=f"chk_sport_{sid}").value is None
    assert primary(at).disabled
    assert step_check.PICK_SPORT in [c.value for c in at.caption]
    at.selectbox(key=f"chk_sport_{sid}").set_value("basketball").run()
    ok(at)
    assert not primary(at).disabled
    assert at.text_area(key=f"chk_activity_{sid}").value == "バスケットボールの試合"


def test_failed_check_still_gives_a_card_and_never_stops_the_work(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    _finish_detect_job(repo, sid, JobStatus.failed, "RuntimeError: detector crashed")
    progress = work_progress(repo, aid)
    assert not progress.stopped and progress.subs[0].status == "your_turn"
    at = ok(app(session={"assignment_id": aid}))
    assert "内容を確かめられませんでした。どんな場面の動画か、下から選んでください。" in html(at)
    assert "crashed" not in screen_text(at)
    assert primary(at).disabled


def test_confirm_sets_sports_words_and_enqueues_prepare(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.number_input(key=f"chk_offset_{sid}").set_value(45.0).run()
    primary(at).click().run()
    ok(at)
    a = repo.get_assignment(aid)
    day = content_check.work_day(a)
    assert (a["title"], a["task_text"], a["domain"], a["sport"], a["model_key"]) == (
        f"サッカーの試合 {day.month}月{day.day}日", "サッカーの試合", "sports", "soccer",
        "standard")
    meta = repo.get_submission(sid)["meta"]
    assert (meta["check"], meta["max_people"], meta["target"], meta["clock_offset_min"]) == (
        "confirmed", 16, "person", 45.0)
    assert jobs(repo, sid, "prepare")
    assert any(s.value == step_check.DECIDED for s in at.success)
    assert shown_step(at) == 3                       # the videos are being read
    assert at.button(key="pfb_nav_1").label == "動画の中身"   # started from the drop zone
    assert "生徒" not in screen_text(at)


def test_offset_only_for_sports(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    offset = number(at, "この動画の始まりの試合時間（分）")
    assert offset.value is None                     # blank by default: not a known game clock
    at.radio(key=f"chk_profile_{sid}").set_value("general").run()
    ok(at)
    assert not [w for w in at.number_input if w.label == step_check.OFFSET_LABEL]
    primary(at).click().run()
    ok(at)
    a = repo.get_assignment(aid)
    assert (a["domain"], a["sport"], a["task_text"]) == ("school", None, "グループ活動")
    meta = repo.get_submission(sid)["meta"]
    assert meta["clock_offset_min"] is None and meta["max_people"] == 16   # the check's count
    assert at.button(key="pfb_nav_1").label == "動画の中身"


def test_paper_method_for_a_school_work_needs_a_roster(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, scene="group_work", sport=None)
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"chk_method_{sid}_school").set_value("paper").run()
    ok(at)
    assert not [w for w in at.number_input if (w.key or "").startswith("chk_people_")]
    assert any("初期位置（カメラ1）" in i.value for i in at.info)       # the paper hint
    assert primary(at).disabled
    assert step_check.PAPER_NEEDS_ROSTER in [c.value for c in at.caption]
    assert step_check.PAPER_NEEDS_ROSTER == "「内容を変える」で名簿を入れると、論文方式で始められます。"
    at.text_area(key=f"chk_roster_{sid}_text").input(
        "名前,外見,初期位置（カメラ1）\n山田,赤い帽子,左前\n佐藤,眼鏡,右奥").run()
    ok(at)
    assert not primary(at).disabled
    primary(at).click().run()
    ok(at)
    meta = repo.get_submission(sid)["meta"]
    assert meta["engine"] == "paper" and "max_people" not in meta
    subjects = {s["display_name"]: s for s in repo.list_subjects(sid)}
    assert subjects["山田"]["initial_positions"] == {"cam1": "左前"}


def test_confirm_error_is_shown_and_nothing_changes(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.text_area(key=f"chk_activity_{sid}").input("").run()
    ok(at)
    assert primary(at).disabled                       # said before pressing, not after
    assert step_check.EMPTY_ACTIVITY.format(work="試合・練習") in [c.value for c in at.caption]
    assert repo.get_submission(sid)["meta"]["check"] == "ready"
    assert not jobs(repo, sid, "prepare")


def test_roster_entered_while_checking_is_applied_on_confirm(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.text_area(key=f"chk_roster_{sid}_text").input("背番号,名前\n10,山田\n7,佐藤").run()
    ok(at)
    detect_done(repo, sid)
    at.run()
    ok(at)
    assert at.text_area(key=f"chk_roster_{sid}_text").value.startswith("背番号")
    primary(at).click().run()
    ok(at)
    names = sorted(s["display_name"] for s in repo.list_subjects(sid))
    assert names == ["佐藤", "山田"]


def test_a_roster_file_given_while_checking_survives_the_card(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.file_uploader(key=f"chk_roster_{sid}_file").set_value(
        ("名簿.csv", "背番号,名前\n10,山田\n7,佐藤\n".encode(), "text/csv")).run()
    ok(at)
    detect_done(repo, sid)
    at.run()
    ok(at)
    assert SOCCER in html(at)
    assert not primary(at).disabled
    primary(at).click().run()
    ok(at)
    assert sorted(s["display_name"] for s in repo.list_subjects(sid)) == ["佐藤", "山田"]


def test_a_second_tab_never_replaces_the_confirmed_roster(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    first, second = (ok(app(session={"assignment_id": aid})) for _ in range(2))
    for at, text in ((first, "背番号,名前\n10,山田"), (second, "背番号,名前\n9,鈴木")):
        at.text_area(key=f"chk_roster_{sid}_text").input(text).run()
        ok(at)
    primary(first).click().run()
    ok(first)
    primary(second).click().run()
    ok(second)
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["山田"]
    assert shown_step(second) == 3                   # its rerun shows the group as decided
    assert len(jobs(repo, sid, "prepare")) == 1


def test_a_confirm_refused_meanwhile_says_so_and_follows_the_work(env, monkeypatch):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))

    def refused(*args, **kwargs):
        raise BusyError("この動画は、もう決定して解析を始めています。")

    monkeypatch.setattr(step_check, "confirm_check", refused)
    primary(at).click().run()
    ok(at)
    assert any("もう決定して" in e.value for e in at.error)
    assert at.session_state["wizard"]["step"] is None
    assert at.session_state["wizard"]["submission_id"] is None


def test_confirm_check_applies_the_roster_inside_its_transaction(env, monkeypatch):
    """A second tab that passed the early checks before the first tab's confirm landed: its
    roster must never replace the people the first confirm brought in."""
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    confirm_check(repo, settings, sid, sports_choice(), parse_roster_text("10 山田"))
    real, calls = content_check._not_confirmed, []

    def raced(sub):                   # the early check ran before the first confirm landed
        calls.append(sub["id"])
        if len(calls) > 1:
            real(sub)

    monkeypatch.setattr(content_check, "_not_confirmed", raced)
    with pytest.raises(BusyError):
        confirm_check(repo, settings, sid, sports_choice(), parse_roster_text("9 鈴木"))
    assert len(calls) == 2
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["山田"]


def test_confirm_keeps_what_the_roster_wrote_to_the_meta(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    confirm_check(repo, settings, sid, sports_choice(), parse_roster_text("10 山田\n7 佐藤"))
    meta = repo.get_submission(sid)["meta"]
    assert meta["check"] == "confirmed"
    assert "roster_run_floor" in meta                 # written by apply_roster, kept by confirm


def test_the_wait_estimate_ignores_a_content_check(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    assert not live.remaining_jobs(repo, aid)
    assert live.estimate_text(600, label="解析にかかる時間の目安") == "解析にかかる時間の目安: 約10分"
    assert live.estimate_text(600) == "残り時間の目安: 約10分"


def test_an_unusable_roster_disables_the_primary(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    at.text_area(key=f"chk_roster_{sid}_text").input("背番号,背番号\n10,7").run()
    ok(at)
    assert primary(at).disabled
    assert "「内容を変える」の選手名簿を直すか、空にすると、始められます。" in [
        c.value for c in at.caption]


# --- later groups -----------------------------------------------------------------------------

def _confirmed_sports_work(settings, repo):
    aid, sid = ready_work(settings, repo)
    confirm_check(repo, settings, sid, sports_choice(title="試合A"))
    return aid, sid


def test_second_group_shows_disagreement_note(env):
    settings, repo = env
    aid, _ = _confirmed_sports_work(settings, repo)
    second = add_videos_to_work(repo, settings, aid, [make_video(settings.inbox_dir / "後半.mp4")])
    detect_done(repo, second, scene="group_work", sport=None)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 1
    assert any(i.value == DISAGREE for i in at.info)
    assert not [r for r in at.radio if (r.key or "").startswith("chk_profile_")]   # the work's
    page = html(at)
    assert "練習試合" not in page and "試合A" in page and "後半.mp4" in page
    assert number(at, step_check.OFFSET_LABEL).value is None        # blank: not known
    at.number_input(key=f"chk_offset_{second}").set_value(50.0).run()
    primary(at).click().run()
    ok(at)
    meta = repo.get_submission(second)["meta"]
    assert meta["check"] == "confirmed" and meta["clock_offset_min"] == 50.0
    assert repo.get_assignment(aid)["sport"] == "soccer"


def test_a_later_group_starts_from_the_first_groups_roster(env):
    settings, repo = env
    aid, first = ready_work(settings, repo)
    confirm_check(repo, settings, first, sports_choice(title="試合A"),
                  parse_roster_text("背番号,名前\n10,山田\n7,佐藤\n"))
    second = add_videos_to_work(repo, settings, aid, [make_video(settings.inbox_dir / "後半.mp4")])
    detect_done(repo, second)
    at = ok(app(session={"assignment_id": aid}))
    text = at.text_area(key=f"chk_roster_{second}_text")
    assert text.value == "名前\t背番号\n山田\t10\n佐藤\t7\n"
    assert step_check.EARLIER_ROSTER.format(roster="選手名簿") in screen_text(at)
    text.input("名前\t背番号\n山田\t10\n高橋\t5\n").run()       # the coach can still edit it
    ok(at)
    primary(at).click().run()
    ok(at)
    assert sorted(s["display_name"] for s in repo.list_subjects(second)) == ["山田", "高橋"]
    assert sorted(s["display_name"] for s in repo.list_subjects(first)) == ["佐藤", "山田"]


def test_step_2_adds_a_later_group_through_the_drop(env):
    settings, repo = env
    aid, _ = _confirmed_sports_work(settings, repo)
    at = ok(app(session={"assignment_id": aid, "step": 2, "submission_id": None}))
    button(at, "もう1件追加").click().run()
    ok(at)
    assert primaries(at) == ["入れて確かめる"]
    assert "同じ試合・練習の別の動画" in html(at)
    at.file_uploader(key="drop_more_files").set_value(
        ("後半.mp4", video_bytes(settings, "後半.mp4"), "video/mp4")).run()
    at.button(key="drop_more_go").click().run()
    ok(at)
    subs = repo.list_submissions(aid)
    assert len(subs) == 2 and subs[1]["meta"]["check"] == "pending"
    assert subs[1]["label"] == "後半"
    assert shown_step(at) == 1 and step_check.CHECKING_TITLE in html(at)


def test_a_work_set_up_before_the_check_keeps_the_registration_form(engine_set_up):
    settings, repo = engine_set_up
    aid = repo.create_assignment("ペーパータワー", "video", "塔を作る", "", "standard", True)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 2
    assert primaries(at) == ["登録して解析を始める"]           # the form, not the drop


def test_a_failed_check_of_a_confirmed_group_never_stops_it(env):
    settings, repo = env
    aid, sid = held_work(settings, repo)
    confirm_check(repo, settings, sid, sports_choice())          # confirmed during the check
    _finish_detect_job(repo, sid, JobStatus.failed, "RuntimeError: boom")
    progress = work_progress(repo, aid)
    assert not progress.stopped
    assert progress.subs[0].step == "people"


# --- copy -------------------------------------------------------------------------------------

def test_no_internal_terms_on_check_card(env):
    settings, repo = env
    screens = {}
    aid, _ = held_work(settings, repo, "a.mp4")
    screens["checking"] = app(session={"assignment_id": aid})
    aid, _ = ready_work(settings, repo)
    screens["ready"] = app(session={"assignment_id": aid})
    aid, _ = ready_work(settings, repo, scene=None, skipped="gpu_busy")
    screens["pick"] = app(session={"assignment_id": aid})
    aid, _ = _confirmed_sports_work(settings, repo)
    second = add_videos_to_work(repo, settings, aid, [make_video(settings.inbox_dir / "b.mp4")])
    detect_done(repo, second, scene="group_work", sport=None)
    screens["later"] = app(session={"assignment_id": aid})
    screens["drop"] = app(session=NEW)
    for name, at in screens.items():
        ok(at)
        text = _rendered(at)
        assert not INTERNAL.search(text), (name, INTERNAL.search(text).group())
        assert not ENGLISH_UI.search(text), (name, ENGLISH_UI.search(text).group())
        no_internal_terms(at)
        assert len(primaries(at)) == 1 and primaries(at)[0] in PRIMARY_LABELS, (name,
                                                                               primaries(at))
        assert any(e.label == "この画面の使い方" for e in at.expander), name
        for g in re.findall(r'<p class="pfb-card-guide">([^<]*)</p>', html(at)):
            assert g.endswith("。") and g.count("。") == 1, (name, g)
        assert not re.search(r"色|赤|青|白|黒", re.sub(r"<[^>]+>", " ", html(at))), name
        assert Stage.detect.value not in text


# --- review fixes (L3 round 2) ----------------------------------------------------------------

def _reasons(at) -> list[str]:
    return [c.value for c in at.caption if "始められます" in c.value]


def test_other_scene_opens_the_expander_and_needs_an_activity(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, scene="other", sport=None)
    at = ok(app(session={"assignment_id": aid}))
    assert change(at).proto.expanded                       # nothing to start from: opened
    assert at.text_area(key=f"chk_activity_{sid}").value == ""
    assert primary(at).disabled
    assert _reasons(at) == [step_check.EMPTY_ACTIVITY.format(work="活動")]
    assert step_check.EMPTY_ACTIVITY.startswith("「内容を変える」")
    at.text_area(key=f"chk_activity_{sid}").input("発表の練習").run()
    ok(at)
    assert not primary(at).disabled
    primary(at).click().run()
    ok(at)
    assert repo.get_submission(sid)["meta"]["check"] == "confirmed"


def test_a_cleared_title_disables_the_primary(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    assert not change(at).proto.expanded
    at.text_input(key=f"chk_title_{sid}").input("  ").run()
    ok(at)
    assert primary(at).disabled
    assert _reasons(at) == [step_check.EMPTY_TITLE.format(work="試合・練習")]
    assert step_check.EMPTY_TITLE.startswith("「内容を変える」")


def test_end_before_start_disables_with_minutes(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, duration_s=3000.0)
    at = ok(app(session={"assignment_id": aid}))
    at.number_input(key=f"chk_start_{sid}").set_value(30.0).run()
    at.number_input(key=f"chk_end_{sid}").set_value(10.0).run()
    ok(at)
    assert primary(at).disabled
    (reason,) = _reasons(at)
    assert reason == step_check.END_BEFORE_START.format(start="30", end="10")
    assert reason.startswith("「内容を変える」") and "30分" in reason and "10分" in reason


def test_start_after_the_video_disables_with_minutes(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, duration_s=3000.0)
    at = ok(app(session={"assignment_id": aid}))
    at.number_input(key=f"chk_end_{sid}").set_value(70.0).run()
    at.number_input(key=f"chk_start_{sid}").set_value(60.0).run()
    ok(at)
    assert primary(at).disabled
    assert _reasons(at) == [step_check.START_AFTER_END.format(start="60", length="50")]


def test_confirm_errors_point_to_the_expander(env, monkeypatch):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    at = ok(app(session={"assignment_id": aid}))

    def refused(*args, **kwargs):
        raise ValueError("カメラ1（a.mp4）: 解析の終わりが動画より後です")

    monkeypatch.setattr(step_check, "confirm_check", refused)
    primary(at).click().run()
    ok(at)
    assert [e.value for e in at.error] == [
        "「内容を変える」で直してください：カメラ1（a.mp4）: 解析の終わりが動画より後です"]


def test_the_paper_reason_points_to_the_expander():
    assert step_check.PAPER_NEEDS_ROSTER.startswith("「内容を変える」")
    assert step_check.PAPER_NEEDS_ROSTER.endswith("始められます。")


def test_a_second_work_on_the_same_day_gets_a_number(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo)
    day = content_check.work_day(repo.get_assignment(aid))
    base = f"サッカーの試合 {day.month}月{day.day}日"
    repo.create_assignment(base, "video", "x", "", "standard", True)
    repo.create_assignment(f"{base}（2）", "video", "x", "", "standard", True)
    at = ok(app(session={"assignment_id": aid}))
    assert at.text_input(key=f"chk_title_{sid}").value == f"{base}（3）"


def test_the_inbox_skips_videos_already_in_use(env):
    import os
    settings, repo = env
    used = make_video(settings.inbox_dir / "b_used.mp4")
    start_work_from_videos(repo, settings, [used], "standard")
    older = make_video(settings.inbox_dir / "a_older.mp4")
    os.utime(older, (1_000_000, 1_000_000))
    at = ok(app(session=NEW))
    assert step_drop.INBOX_FOUND.format(n=1) in [c.value for c in at.caption]
    assert step_drop.INBOX_FOUND == "受け取り用フォルダに新しい動画があります（{n}件）。"
    source = at.radio(key="drop_source")
    assert source.options == ["この端末から送る", "受け取り用フォルダの動画から選ぶ"]  # no 2nd count
    source.set_value("inbox").run()
    ok(at)
    assert at.multiselect(key="drop_inbox").value == [older]   # the newest one not in use


def test_the_inbox_preselects_nothing_when_every_video_is_used(env):
    settings, repo = env
    used = make_video(settings.inbox_dir / "used.mp4")
    start_work_from_videos(repo, settings, [used], "standard")
    at = ok(app(session=NEW))
    assert step_drop.INBOX_ALL_USED in [c.value for c in at.caption]
    at.radio(key="drop_source").set_value("inbox").run()
    ok(at)
    assert at.multiselect(key="drop_inbox").value == []
    assert at.button(key="drop_go").disabled


def test_add_more_never_preselects_this_works_own_video(env):
    import os
    settings, repo = env
    old = make_video(settings.inbox_dir / "first_half.mp4")
    new = make_video(settings.inbox_dir / "second_half.mp4")
    os.utime(old, (1_000_000, 1_000_000))
    aid, sid = start_work_from_videos(repo, settings, [new], "standard")
    confirm_check(repo, settings, sid, sports_choice())
    at = ok(app(session={"assignment_id": aid, "step": 2, "submission_id": None}))
    button(at, "もう1件追加").click().run()
    ok(at)
    at.radio(key="drop_more_source").set_value("inbox").run()
    ok(at)
    assert at.multiselect(key="drop_more_inbox").value == [old]


def test_the_upload_styles_tell_refused_from_failed():
    from pfb_app.ui.common import UPLOAD_FAILED, upload_refused_style
    assert UPLOAD_FAILED == "送れませんでした。通信を確かめて、ファイルを押すともう一度送ります。"
    css = upload_refused_style()
    assert '[aria-invalid="true"]:not([role="button"])' in css
    assert '[aria-invalid="true"][role="button"]' in css and UPLOAD_FAILED in css
    assert '[role="alert"] { display: none' not in css           # still announced
    from pfb_app.ui.common import upload_watch_script
    js = upload_watch_script()
    assert 'aria-live' in js and UPLOAD_FAILED in js and "data-pfb-send" in js
    # the aria-live line is hidden inline (its class rule exists on upload screens only) and
    # emptied once no chip is in error (round 2 merge, L3 review LOW)
    assert 'said.style.cssText = "position: absolute; width: 1px;' in js and "clip-path" in js
    assert 'said.textContent = ""' in js


# --- activity-type profiles on the card ---------------------------------------------------------

def test_card_offers_six_kinds_and_suggests_a_meeting(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, scene="group_work", sport=None, activity="meeting")
    at = ok(app(session={"assignment_id": aid}))
    assert "話し合い・会議のようです（推定）。映っている人は約14人。" in html(at)
    kind = at.radio(key=f"chk_profile_{sid}")
    assert kind.label == step_check.KIND_QUESTION
    assert kind.options == ["一般", "話し合い・会議", "スポーツ", "発表・パフォーマンス",
                            "研修・ワークショップ", "学習活動"]
    assert kind.value == "meeting"
    assert facts(at)[0] == ("場面", "話し合い・会議")
    assert at.text_input(key=f"chk_title_{sid}").label == "会議の名前"
    assert at.text_area(key=f"chk_activity_{sid}").value == "話し合い・会議"
    assert primaries(at) == ["この内容で始める"]
    primary(at).click().run()
    ok(at)
    a = repo.get_assignment(aid)
    assert (a["profile"], a["domain"], a["task_text"]) == ("meeting", "school", "話し合い・会議")


def test_changing_the_kind_changes_words_title_and_profile(env):
    settings, repo = env
    aid, sid = ready_work(settings, repo, scene="group_work", sport=None)
    at = ok(app(session={"assignment_id": aid}))
    assert at.radio(key=f"chk_profile_{sid}").value == "general"
    assert facts(at)[0] == ("場面", "グループ活動")
    at.radio(key=f"chk_profile_{sid}").set_value("workshop").run()
    ok(at)
    day = content_check.work_day(repo.get_assignment(aid))
    assert at.text_input(key=f"chk_title_{sid}").value == f"研修 {day.month}月{day.day}日"
    assert at.text_input(key=f"chk_title_{sid}").label == "研修の名前"
    assert at.text_area(key=f"chk_activity_{sid}").value == "研修・ワークショップ"
    assert facts(at)[0] == ("場面", "研修・ワークショップ")
    assert not [w for w in at.number_input if w.label == step_check.OFFSET_LABEL]
    at.radio(key=f"chk_profile_{sid}").set_value("presentation").run()
    ok(at)
    primary(at).click().run()
    ok(at)
    a = repo.get_assignment(aid)
    assert (a["profile"], a["domain"], a["sport"]) == ("presentation", "school", None)


def test_no_school_words_on_a_card_of_each_kind(env):
    from test_neutral_wording import clean, everything
    settings, repo = env
    for activity in ("meeting", "presentation", "workshop", "learning"):
        aid, sid = ready_work(settings, repo, scene="group_work", sport=None, activity=activity)
        at = ok(app(session={"assignment_id": aid}))
        clean(everything(at), activity)
        no_internal_terms(at)


def test_the_pick_wording_matches_the_kind_question():
    assert step_check.KIND_QUESTION == "どんな場面の動画ですか"
    for text in (step_check.PICK_SCENE, content_check.PICK, content_check.NO_PROBE_SENTENCE):
        assert "どんな場面の動画か" in text, text
