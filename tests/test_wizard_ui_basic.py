"""Wizard UI: router, home screen, steps 1 and 2, navigation (AppTest on ui/app.py)."""
import re
from pathlib import Path

import cv2
import numpy as np
from pdfhelp import make_pdf_bytes
import pytest
import yaml
from streamlit.testing.v1 import AppTest

import pfb_app.ui
from pfb_app.config import Settings
from pfb_app.services.intake import register_documents, register_video_group
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.wizard.progress import work_progress

UI = Path(pfb_app.ui.__file__).parent
# the global list of primary button labels (global-constraints.md)
PRIMARY_LABELS = ("新しく始める", "続ける", "これで決定して次へ", "参加者の特定を確認",
                  "この内容で始める",   # the content check card
                  "登録して解析を始める", "承認する", "直して承認", "このまま直して承認",
                  "参加者に渡すPDFを受け取る",
                  "自分で書く",   # step 5 when only students with too little evidence are left
                  "全員まとめて承認",   # step 5 when two or more texts need no look
                  "この文で承認する",   # step 5: the short text of a person with too few scenes
                  "入れて確かめる",   # the drop zone of a new work (and a later video group)
                  "映っている人を自動で選び直す", "もう一度解析する",   # a stopped step 4
                  "もう一度ためす",   # a stopped step 3 or 4 (services.retry)
                  "新しい作業として始める",   # step 2 of a work whose videos were removed
                  # 「あとから訂正」 works (step_correct, step_pdf._live)
                  "PDFを受け取る", "直した文を保存する", "このまま保存する", "報告する",
                  "参加者用のPDFを受け取る")
FORBIDDEN = ("stage_run", "job", "engine", "P-label", "model_key", "エンジン", "ジョブ",
             "PFB3x3", "pfb3x3",
             # internal words replaced by plain ones (UX round 2, audit #6)
             "GPUの空き", "GPUのメモリ", "人の選定", "根拠の抜き出し",
             "対応づけを待って",
             # round-2 copy pass: 管理用の画面 / 場面が少ない / まだ選んでいません / 人を見つける.
             # The admin pages (ui/pages/) keep their own labels; they are not scanned here.
             "詳細画面", "根拠不足", "未割り当て", "人物の追跡")


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox",
                        engine_dir=tmp_path / "engines" / "pfb3x3")   # never the real engine
    cfg = tmp_path / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir)}), encoding="utf-8")
    monkeypatch.setenv("PFB_CONFIG", str(cfg))
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return settings, Repo(conn, settings)


@pytest.fixture
def engine_set_up(env):
    """The tracking method counts as set up (FakeEngine-style settings: ENGINE_COMMIT)."""
    settings, _ = env
    settings.engine_dir.mkdir(parents=True, exist_ok=True)
    (settings.engine_dir / "ENGINE_COMMIT").write_text("fake\n", encoding="utf-8")
    return env


def make_video(path: Path, seconds: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(seconds * 10):
        writer.write(np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def app(session: dict | None = None, query: dict | None = None,
        extra: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(UI / "app.py"), default_timeout=60)
    if session is not None:
        at.session_state["wizard"] = session
    for key, value in (extra or {}).items():
        at.session_state[key] = value
    if query:
        at.query_params.update(query)
    return at.run()


def ok(at: AppTest) -> AppTest:
    assert not at.exception, [e.value for e in at.exception]
    return at


def primaries(at: AppTest) -> list[str]:
    """Primary buttons, download buttons included (step 6's primary is one)."""
    return [b.label for b in [*at.button, *at.download_button] if b.proto.type == "primary"]


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def html(at: AppTest) -> str:
    bodies = [el.proto.body for el in at.get("html")]
    return "\n".join(re.sub(r"<style>.*?</style>", "", b, flags=re.S) for b in bodies)


def shown_step(at: AppTest) -> int:
    """The step the stepper marks as shown (いまここ)."""
    cur = re.findall(r'pfb-navstep [^"]*pfb-step--current', html(at))
    assert len(cur) == 1, html(at)
    marks = re.findall(r'<div class="pfb-navstep ([^"]*)"', html(at))
    return next(i for i, cls in enumerate(marks, 1) if "pfb-step--current" in cls)


def screen_text(at: AppTest) -> str:
    parts = [html(at)]
    for kind in ("markdown", "caption", "title", "header", "subheader", "info", "warning",
                 "error", "success", "expander"):
        parts += [getattr(e, "value", "") or getattr(e, "label", "") for e in at.get(kind)]
    parts += [b.label + (b.help or "") for b in at.button]
    for w in list(at.radio) + list(at.selectbox):
        parts += [w.label, *map(str, w.options)]
    for w in list(at.text_input) + list(at.text_area) + list(at.number_input) + list(at.checkbox):
        parts.append(w.label)
    return "\n".join(str(p) for p in parts)


def no_internal_terms(at: AppTest) -> None:
    text = screen_text(at)
    for word in FORBIDDEN:
        assert word not in text, word


def pdf_bytes(text="これは提出された書類の本文です。" * 5) -> bytes:
    return make_pdf_bytes([text])


def doc_assignment(repo: Repo, title="探究レポート") -> int:
    return repo.create_assignment(title, "document", "調べてまとめる", "", "standard", False)


def video_assignment(repo: Repo, title="ペーパータワー") -> int:
    return repo.create_assignment(title, "video", "塔を高く作る", "協力", "standard", True)


def register(settings, repo, aid, label="A班", engine="pfb3x3") -> int:
    video = make_video(settings.inbox_dir / f"{label}.mp4")
    return register_video_group(repo, settings, aid, label, {"cam1": video}, 0.0, None, "",
                                False, [], engine=engine)


# --- home -----------------------------------------------------------------------------------

def test_home_without_work_shows_start(env):
    at = ok(app())
    assert primaries(at) == ["新しく始める"]
    assert any(e.label == "この画面の使い方" for e in at.expander)
    no_internal_terms(at)


def test_home_lists_works_with_reason_chip_and_continue(env):
    settings, repo = env
    waiting = video_assignment(repo, "待つ作業")
    register(settings, repo, waiting)                 # analysis queued: not the teacher's turn
    turn = video_assignment(repo, "あなたの番の作業")  # no files yet: the teacher's turn
    at = ok(app())
    page = html(at)
    assert page.index("あなたの番の作業") < page.index("待つ作業")    # your turn first
    assert "動画を入れてください。" in page and "あなたの番" in page
    assert 'class="pfb-mini"' in page
    # one primary button: 「続ける」 on the first work that waits for the teacher
    assert primaries(at) == ["続ける"]
    assert at.button(key=f"pfb_open_{turn}").proto.type == "primary"
    assert at.button(key=f"pfb_open_{waiting}").proto.type == "secondary"
    assert at.button(key="pfb_new").proto.type == "secondary"
    no_internal_terms(at)
    at.button(key=f"pfb_open_{turn}").click().run()
    ok(at)
    assert at.session_state["wizard"]["assignment_id"] == turn
    assert shown_step(at) == 2
    assert at.query_params.get("work") == str(turn)


def test_home_card_shows_stage_and_percent(env):
    from test_wizard_progress import group, tracked_to_review
    from test_wizard_ui_people_wait import confirmed_and_analysing, worker_alive
    from pfb_app.services.assignments import mark_pdf_downloaded
    from pfb_app.services.review import approve
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    jid = repo.conn.execute("SELECT id FROM jobs WHERE submission_id=?", (sid,)).fetchone()[0]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    # 1 + 1 + 4 + 0.5 done and 30 % of the behavior's 10, of 22.5
    repo.set_job_progress(jid, 0.3, "12 / 40 区間")
    done = video_assignment(repo, "終わった作業")
    dsid, subjects = group(repo, done, students=(("Pa", "A子"),))
    tracked_to_review(repo, dsid, subjects)
    approve(repo, subjects["Pa"])
    mark_pdf_downloaded(repo, done)
    at = ok(app())
    page = html(at)
    assert "したことの書き出し・42%" in page
    assert page.count('<div class="pfb-cardbar">') == 1          # the complete work: no bar
    bar = page[page.index('<div class="pfb-cardbar">'):]
    assert "進行中" in bar[:bar.index("</p>")]                    # a chip word next to the bar
    assert "削除" in [b.label for b in at.button]                  # the delete stays usable
    assert len([b for b in at.button if b.label == "削除"]) == 2


def test_stepper_shows_the_percent_while_analysing(env):
    from test_wizard_progress import group, tracked_to_review
    from test_wizard_ui_people_wait import confirmed_and_analysing, worker_alive
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    jid = repo.conn.execute("SELECT id FROM jobs WHERE submission_id=?", (sid,)).fetchone()[0]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    repo.set_job_progress(jid, 0.3, None)
    at = ok(app(session={"assignment_id": aid}))
    states = re.findall(r'<div class="pfb-navstate[^"]*">(.*?)</div>', html(at))
    assert shown_step(at) == 4 and "42%" in states[3]
    assert not any("%" in s for n, s in enumerate(states) if n != 3)
    other = video_assignment(repo, "見直しの作業")
    osid, subjects = group(repo, other)
    tracked_to_review(repo, osid, subjects)
    at = ok(app(session={"assignment_id": other}))
    assert shown_step(at) == 5
    assert not any("%" in s for s in re.findall(r'<div class="pfb-navstate[^"]*">(.*?)</div>',
                                                html(at)))


def test_home_shows_a_stopped_work_with_a_link_to_fix_it(env):
    settings, repo = env
    aid = video_assignment(repo, "止まった作業")
    sid = register(settings, repo, aid)
    job = repo.claim_next_job({"cpu"})
    assert job is not None and job.submission_id == sid
    repo.finish_job(job.id, "failed", "VideoError: 動画を読めません")
    at = ok(app())
    page = html(at)
    assert "止まっています" in page and "動画を読めません" in page
    links = [link.proto.label for link in at.get("page_link")]
    assert "くわしく見る（管理用の画面）" in links and f"pfb_retry_{aid}" in [b.key for b in at.button]
    no_internal_terms(at)


def test_home_start_is_primary_when_nothing_waits_for_the_teacher(env):
    settings, repo = env
    register(settings, repo, video_assignment(repo))
    at = ok(app())
    assert primaries(at) == ["新しく始める"]


# --- step 1 ---------------------------------------------------------------------------------

def test_creating_an_assignment_from_dropped_documents(env):
    """A new work starts at the drop zone; documents lead to step 1's form with the kind
    fixed, and are registered as soon as the work is created."""
    settings, repo = env
    at = ok(app())
    at.button(key="pfb_new").click().run()
    ok(at)
    assert primaries(at) == ["入れて確かめる"]
    assert any(e.label == "この画面の使い方" for e in at.expander)
    assert at.query_params.get("work") == "new"
    no_internal_terms(at)
    at.file_uploader(key="drop_files").set_value(
        ("山田太郎.pdf", pdf_bytes(), "application/pdf")).run()
    at.button(key="drop_go").click().run()
    ok(at)
    assert primaries(at) == ["これで決定して次へ"]
    no_internal_terms(at)
    at.text_input(key="a_title").input("探究レポート")
    at.text_area(key="a_task").input("調べてまとめる")
    at.button(key="a_save").click().run()
    ok(at)
    (a,) = repo.list_assignments()
    assert (a["title"], a["kind"], a["task_text"]) == ("探究レポート", "document", "調べてまとめる")
    assert [s["label"] for s in repo.list_submissions(a["id"])] == ["山田太郎"]
    assert at.session_state["wizard"] == {"assignment_id": a["id"], "step": None,
                                          "submission_id": None}
    assert shown_step(at) == 3
    assert any("1件の書類を登録" in s.value for s in at.success)


DROPPED = {"_pfb_drop_docs": [("山田太郎.pdf", b"%PDF-1.4")]}   # documents on the drop zone


def test_step_1_needs_title_and_task(env):
    at = ok(app(session={"assignment_id": None, "step": 1, "submission_id": None},
                extra=DROPPED))
    at.button(key="a_save").click().run()
    ok(at)
    assert any("入力してください" in e.value for e in at.error)
    assert at.session_state["wizard"]["assignment_id"] is None


def test_step_1_is_editable_before_analysis_and_view_only_after(env):
    settings, repo = env
    aid = video_assignment(repo)
    at = ok(app(session={"assignment_id": aid, "step": 1, "submission_id": None}))
    assert shown_step(at) == 1
    assert primaries(at) == ["これで決定して次へ"]
    at.text_area(key=f"a_edit_task_{aid}").input("紙で一番高い塔を作る")
    at.button(key="a_update").click().run()
    ok(at)
    assert repo.get_assignment(aid)["task_text"] == "紙で一番高い塔を作る"
    assert shown_step(at) == 2                       # back to the derived step

    register(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid, "step": 1, "submission_id": None}))
    assert shown_step(at) == 1
    assert at.text_area(key=f"a_view_task_{aid}").disabled
    assert not [w for w in at.text_area if w.key == f"a_edit_task_{aid}"]
    assert any("変更できません" in i.value for i in at.info)
    assert primaries(at) == ["続ける"]
    no_internal_terms(at)
    at.button(key="a_continue").click().run()
    ok(at)
    assert shown_step(at) == 3


# --- step 2 ---------------------------------------------------------------------------------

def register_in_ui(at: AppTest, video: Path, label="B班", method=None) -> AppTest:
    if method is not None:
        at.radio(key="v_engine").set_value(method).run()
    at.radio(key="v_source").set_value("inbox").run()
    box = at.selectbox(key="v_inbox1")      # form widgets: set together, then submit
    box.select_index(box.options.index(video.name))
    at.text_input(key="v_label").input(label)
    button(at, "登録して解析を始める").click().run()
    return ok(at)


def test_step_2_tracking_is_the_default_method(engine_set_up):
    settings, repo = engine_set_up
    aid = video_assignment(repo)
    at = ok(app(session={"assignment_id": aid}))
    radio = at.radio(key="v_engine")
    assert radio.options == ["トラッキング方式（おすすめ）", "論文方式"]
    assert radio.value == "pfb3x3"
    assert not at.warning                              # set up: no warning
    captions = "\n".join(c.value for c in at.caption)
    assert "外見と初期位置は書かなくてもかまいません" in captions
    assert primaries(at) == ["登録して解析を始める"]
    no_internal_terms(at)


def test_registering_a_video_group_moves_the_derived_step_to_3(engine_set_up):
    settings, repo = engine_set_up
    aid = video_assignment(repo)
    video = make_video(settings.inbox_dir / "group.mp4")
    at = ok(app(session={"assignment_id": aid}))
    register_in_ui(at, video)
    (sub,) = repo.list_submissions(aid)
    assert sub["label"] == "B班"
    assert sub["meta"]["engine"] == "pfb3x3"
    assert sub["meta"]["max_people"] is None            # blank: the participants (none here)
    progress = work_progress(repo, aid)
    assert progress.current == 3 and progress.keys[2] == "people"
    # the screen stays on step 2 with the summary: 「続ける」, and 「もう1件追加」
    assert shown_step(at) == 2
    assert any("「B班」を登録しました" in s.value for s in at.success)
    assert primaries(at) == ["続ける"]
    assert any(b.label == "もう1件追加" for b in at.button)
    assert "B班" in html(at)
    no_internal_terms(at)
    button(at, "続ける").click().run()
    ok(at)
    assert shown_step(at) == 3
    assert at.session_state["wizard"]["step"] is None


def test_add_another_group_and_paper_method(engine_set_up):
    settings, repo = engine_set_up
    aid = video_assignment(repo)
    register(settings, repo, aid, "A班")
    at = ok(app(session={"assignment_id": aid, "step": 2, "submission_id": None}))
    assert primaries(at) == ["続ける"]
    button(at, "もう1件追加").click().run()
    ok(at)
    assert primaries(at) == ["登録して解析を始める"]
    assert any(b.label == "やめる" for b in at.button)
    video = make_video(settings.inbox_dir / "c.mp4")
    register_in_ui(at, video, "C班", method="paper")
    subs = {s["label"]: s for s in repo.list_submissions(aid)}
    assert set(subs) == {"A班", "C班"}
    assert subs["C班"]["meta"]["engine"] == "paper"
    assert "max_people" not in subs["C班"]["meta"]
    assert primaries(at) == ["続ける"]


def test_step_2_warns_when_tracking_is_not_set_up(env):
    settings, repo = env
    at = ok(app(session={"assignment_id": video_assignment(repo)}))
    assert any("準備がまだ済んでいません" in w.value for w in at.warning)
    no_internal_terms(at)


def test_step_2_documents(env):
    settings, repo = env
    aid = repo.create_assignment("探究レポート", "document", "調べてまとめる", "", "standard",
                                 False)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 2
    assert [at.button(key=f"pfb_nav_{n}").label for n in range(1, 6)] == [
        "活動を決める", "書類を入れる", "解析を待つ", "確認して承認", "PDFを受け取る"]
    assert primaries(at) == ["登録して解析を始める"]
    assert not at.radio                               # no method choice for documents
    no_internal_terms(at)


# --- router ---------------------------------------------------------------------------------

def test_reload_lands_on_the_derived_step(engine_set_up):
    settings, repo = engine_set_up
    empty = video_assignment(repo, "まだ空")
    at = ok(app(session={"assignment_id": empty}))   # a session holding only the work
    assert shown_step(at) == 2
    busy = video_assignment(repo, "登録ずみ")
    register(settings, repo, busy)
    at = ok(app(session={"assignment_id": busy}))
    assert shown_step(at) == 3
    # a browser reload starts a new session: only the URL remembers the work
    at = ok(app(query={"work": str(busy)}))
    assert at.session_state["wizard"]["assignment_id"] == busy
    assert shown_step(at) == 3


def test_a_link_without_the_work_in_the_url_goes_home(env):
    settings, repo = env
    aid = video_assignment(repo)
    at = ok(app(session={"assignment_id": aid}))
    assert at.query_params.get("work") == str(aid)
    at.query_params.clear()                           # what a sidebar link does
    at.run()
    ok(at)
    assert at.session_state["wizard"]["assignment_id"] is None
    assert any(b.label == "新しく始める" for b in at.button)


def test_the_sidebar_home_always_leads_home(env):
    settings, repo = env
    aid = video_assignment(repo)
    at = ok(app(session={"assignment_id": aid}))
    at.button(key="pfb_side_home").click().run()     # from inside a work
    ok(at)
    assert at.session_state["wizard"]["assignment_id"] is None
    assert "work" not in at.query_params
    assert any(b.label == "新しく始める" for b in at.button)
    # (from a detail page it also switches pages: checked in a browser, as AppTest drops the
    # entry script's sidebar after switch_page)


def test_unknown_work_in_the_url_goes_home(env):
    at = ok(app(query={"work": "999"}))
    assert at.session_state["wizard"]["assignment_id"] is None
    assert any("見つかりませんでした" in s.value for s in at.success)
    assert "work" not in at.query_params


def test_stepper_opens_reachable_steps_only(engine_set_up):
    settings, repo = engine_set_up
    aid = video_assignment(repo)
    register(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert at.button(key="pfb_nav_3").disabled          # shown now
    assert not at.button(key="pfb_nav_1").disabled
    assert at.button(key="pfb_nav_5").disabled          # not reached yet
    at.button(key="pfb_nav_1").click().run()
    ok(at)
    assert shown_step(at) == 1
    assert at.session_state["wizard"]["step"] == 1
    at.button(key="pfb_nav_3").click().run()            # back to the derived step
    ok(at)
    assert shown_step(at) == 3
    assert at.session_state["wizard"]["step"] is None
    at.button(key="pfb_home").click().run()
    ok(at)
    assert at.session_state["wizard"]["assignment_id"] is None
    assert "work" not in at.query_params


def test_a_step_out_of_reach_falls_back_to_the_derived_step(env):
    settings, repo = env
    aid = video_assignment(repo)
    at = ok(app(session={"assignment_id": aid, "step": 5, "submission_id": None}))
    assert shown_step(at) == 2


def test_old_pages_are_still_reachable_under_details(env):
    _, repo = env
    repo.create_assignment("練習試合", "video", "パスをつなぐ", "", "standard", False,
                           domain="sports", sport="basketball")   # 判定支援 is shown
    at = ok(app())
    assert '<p class="pfb-sidenav__section">管理用の画面（くわしい操作）</p>' in "".join(
        el.proto.body for el in at.sidebar.get("html"))
    sections = [el.proto.body for el in at.sidebar.get("html")
                if "pfb-sidenav__section" in el.proto.body]
    assert sections == ['<p class="pfb-sidenav__section">管理用の画面（くわしい操作）</p>',
                        '<p class="pfb-sidenav__section">判定支援</p>']   # 判定支援 after it
    assert [link.proto.label for link in at.sidebar.get("page_link")] == [
        "はじめての方へ",   # under ホーム (onboarding lane)
        "成長の記録",       # one person across works (growth lane)
        "ききたいことを聞く",   # a work's records answer a question (ask lane)
        "活動", "ファイルの登録", "処理状況", "確認・承認", "PDF出力", "この解析の確かさ",
        "データの扱い",     # consent and retention (data lane)
        "利用する人",       # who may sign in (accounts lane; 管理者 only)
        "試合後の判定材料", "ライブ判定"]   # 判定支援 last (linecall lane)
    for page, title in (("pages/assignments.py", "活動"), ("pages/submissions.py", "ファイルの登録"),
                        ("pages/status.py", "処理状況"), ("pages/review.py", "確認・承認"),
                        ("pages/export.py", "PDF出力"),
                        ("pages/linecall_review.py", "試合後の判定材料"),
                        ("pages/linecall_live.py", "ライブ判定")):
        at.switch_page(page).run()
        ok(at)
        assert at.title[0].value == title


def test_the_line_call_section_is_hidden_without_a_sports_work(env):
    _, repo = env
    video_assignment(repo)                                   # a general work only
    at = ok(app())
    sidebar = "".join(el.proto.body for el in at.sidebar.get("html"))
    labels = [link.proto.label for link in at.sidebar.get("page_link")]
    assert "判定支援" not in sidebar and "試合後の判定材料" not in labels
    assert "ライブ判定" not in labels
    at.switch_page("pages/linecall_review.py").run()          # still reachable by its address
    assert ok(at).title[0].value == "試合後の判定材料"


def test_each_step_screen_has_exactly_one_primary_button(engine_set_up):
    settings, repo = engine_set_up
    video = video_assignment(repo)
    doc = repo.create_assignment("レポート", "document", "まとめる", "", "standard", False)
    started = video_assignment(repo, "登録ずみ")
    register(settings, repo, started)
    docs = doc_assignment(repo, "提出ずみ")
    register_documents(repo, settings, docs, [("山田太郎.pdf", pdf_bytes())], None)
    screens = [
        dict(session=None),                                                      # home
        dict(session={"assignment_id": None, "step": 1, "submission_id": None}),   # new
        dict(session={"assignment_id": video, "step": 1, "submission_id": None}),  # editable
        dict(session={"assignment_id": started, "step": 1, "submission_id": None}),  # view
        dict(session={"assignment_id": video}),                                  # 2, video
        dict(session={"assignment_id": doc}),                                    # 2, documents
        dict(session={"assignment_id": started, "step": 2, "submission_id": None}),  # summary
        dict(session={"assignment_id": started, "step": 2, "submission_id": None},   # adding
             extra={"_pfb_adding": started}),
        dict(session={"assignment_id": docs, "step": 2, "submission_id": None}),   # summary
        dict(session={"assignment_id": docs, "step": 2, "submission_id": None},    # adding
             extra={"_pfb_adding": docs}),
        dict(session={"assignment_id": started}),                  # 3, tracking: waiting
        dict(session={"assignment_id": docs}),                     # 3, documents: waiting
    ]
    for screen in screens:
        at = ok(app(**screen))
        session = screen["session"] or {}
        waiting_docs = session.get("assignment_id") == docs and not session.get("step")
        if waiting_docs:   # the analysis runs: nothing to press
            assert primaries(at) == [], primaries(at)
            continue
        assert len(primaries(at)) == 1, (screen, primaries(at))
        assert primaries(at)[0] in PRIMARY_LABELS, primaries(at)
        assert any(e.label == "この画面の使い方" for e in at.expander), screen
        no_internal_terms(at)


# --- review round 1 -------------------------------------------------------------------------

def test_registering_documents_from_the_wizard(env):
    settings, repo = env
    aid = doc_assignment(repo)
    at = ok(app(session={"assignment_id": aid}))
    at.file_uploader(key="d_files").set_value([
        ("山田太郎.pdf", pdf_bytes(), "application/pdf"),
        ("山田太郎 (1).pdf", pdf_bytes(), "application/pdf"),
        ("壊れ.pdf", b"garbage", "application/pdf")])
    at.run()
    at.button(key="d_register").click().run()
    ok(at)
    assert len(repo.list_submissions(aid)) == 2
    assert any("2件の書類を登録しました" in m.value for m in at.success)
    assert any(e.value.startswith("壊れ.pdf: ") for e in at.error)       # kept: unreadable
    assert any("同じ名前の書類があります" in w.value for w in at.warning)  # kept: duplicates
    assert primaries(at) == ["続ける"]                                   # the summary
    assert "山田太郎" in html(at)
    no_internal_terms(at)
    button(at, "もう1件追加").click().run()
    ok(at)
    assert primaries(at) == ["登録して解析を始める"]
    assert not at.error and not at.warning          # the report was shown once
    button(at, "やめる").click().run()
    ok(at)
    assert primaries(at) == ["続ける"]


def test_unreadable_roster_is_a_plain_message(env):
    settings, repo = env
    aid = doc_assignment(repo)
    at = ok(app(session={"assignment_id": aid}))
    at.file_uploader(key="d_files").set_value(("山田.pdf", pdf_bytes(), "application/pdf"))
    at.file_uploader(key="d_roster").set_value(("名簿.csv", "name\n山田".encode("cp932"),
                                                "text/csv"))
    at.run()
    at.button(key="d_register").click().run()
    ok(at)
    assert not repo.list_submissions(aid)
    assert [e.value for e in at.error] == ["名簿を読めません。UTF-8で保存したCSVを使ってください。"]
    assert any(e.label == "くわしい情報（管理者向け）" for e in at.expander)


def test_an_unexpected_failure_is_a_plain_message_with_the_detail_folded(engine_set_up,
                                                                       monkeypatch):
    settings, repo = engine_set_up
    import pfb_app.ui.common as common

    def boom(*a, **k):
        raise RuntimeError("database is locked")
    monkeypatch.setattr(common, "register_video_group", boom)
    aid = video_assignment(repo)
    video = make_video(settings.inbox_dir / "group.mp4")
    at = ok(app(session={"assignment_id": aid}))
    register_in_ui(at, video)
    assert not repo.list_submissions(aid)
    assert [e.value for e in at.error] == [common.UNEXPECTED]
    assert "database is locked" not in "\n".join(e.value for e in at.error)
    assert any(e.label == "くわしい情報（管理者向け）" for e in at.expander)
    assert "RuntimeError: database is locked" in "\n".join(c.value for c in at.code)


def test_step_2_never_shows_server_paths_or_english_folder_names(engine_set_up):
    settings, repo = engine_set_up
    make_video(settings.inbox_dir / "group.mp4")
    at = ok(app(session={"assignment_id": video_assignment(repo)}))
    at.radio(key="v_source").set_value("inbox").run()
    ok(at)
    text = screen_text(at)
    assert "受け取り用フォルダ" in text
    assert "inbox" not in text and str(settings.inbox_dir) not in text


def test_back_to_the_new_form_after_creating_goes_home(env):
    settings, repo = env
    at = ok(app(session={"assignment_id": None, "step": 1, "submission_id": None},
                extra=DROPPED))
    at.text_input(key="a_title").input("一つだけ")
    at.text_area(key="a_task").input("課題")
    at.button(key="a_save").click().run()
    ok(at)
    (a,) = repo.list_assignments()
    assert at.query_params.get("work") == str(a["id"])
    at.query_params["work"] = "new"                     # the browser's back button
    at.run()
    ok(at)
    assert at.session_state["wizard"]["assignment_id"] is None
    assert at.session_state["wizard"]["step"] is None    # home, not a second blank form
    assert any(b.label == "新しく始める" for b in at.button)
    at.button(key="pfb_new").click().run()             # a new start is empty again
    ok(at)
    assert "評価したいファイルを入れてください" in html(at)   # the drop zone, nothing kept
    assert "_pfb_drop_docs" not in at.session_state
    assert len(repo.list_assignments()) == 1


def test_leaving_a_work_forgets_its_add_form(engine_set_up):
    settings, repo = engine_set_up
    aid = video_assignment(repo)
    register(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid, "step": 2, "submission_id": None}))
    button(at, "もう1件追加").click().run()
    assert primaries(at) == ["登録して解析を始める"]
    at.button(key="pfb_home").click().run()
    at.button(key=f"pfb_open_{aid}").click().run()
    at.button(key="pfb_nav_2").click().run()
    ok(at)
    assert primaries(at) == ["続ける"]                 # the summary, not the form


def test_compact_stepper_for_phones(engine_set_up):
    """Below 40rem the stepper is one line, a thin bar and 「ほかの段階」 (CSS picks which)."""
    settings, repo = engine_set_up
    aid = video_assignment(repo)
    register(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    html = " ".join(el.proto.body for el in at.get("html"))
    assert re.search(r'pfb-navline__text">3/6 人を確かめる（任意）<', html)
    assert 'role="progressbar"' in html and 'aria-valuenow="3"' in html
    assert any(p.proto.popover.label == "ほかの段階" for p in at.get("popover"))
    assert at.button(key="pfb_navm_3").disabled         # shown now
    assert not at.button(key="pfb_navm_1").disabled
    assert at.button(key="pfb_navm_5").disabled         # not reached yet
    at.button(key="pfb_navm_1").click().run()
    ok(at)
    assert shown_step(at) == 1
