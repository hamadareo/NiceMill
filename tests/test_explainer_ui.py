"""解説動画 on the screens (AppTest): step 6's section (make, progress, player + download, an
outdated video, a failure), the dashboard's one-person view, and nothing for anyone but 管理者
and 評価する人. Synthetic works only."""
import pytest

from pdf_synthetic import video_group, work
from pfb_app.services import explainer
from test_dashboard_ui import page as dashboard_page
from test_neutral_wording import clean, everything
from test_wizard_ui_basic import FORBIDDEN, app, env, ok, primaries  # noqa: F401 (env)

STUDENT = "選手に渡すPDFを受け取る"   # a sports work
MAKE = "解説動画を作る"


@pytest.fixture(autouse=True)
def short(monkeypatch):
    monkeypatch.setattr(explainer, "MAX_MOMENTS", 1)
    monkeypatch.setattr(explainer, "TRACKING_MAX_S", 3.0)


def _work(env, **kw):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=2, seconds=30, **kw)
    return settings, repo, aid, sid


def step6(aid):
    return ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _neutral(at, where):
    text = everything(at)
    clean(text, where)
    assert not [w for w in FORBIDDEN if w in text], where


def test_step_6_makes_a_video_and_then_plays_it(env):
    settings, repo, aid, sid = _work(env)
    at = step6(aid)
    text = everything(at)
    assert "解説動画" in text and "AIの注目の可視化は研究中です" in text
    assert "追跡の様子" in text
    who = next(w for w in at.selectbox if w.label == "だれの動画か")
    assert len(who.options) == 2 and all(o.endswith("さん") for o in who.options)
    assert _button(at, MAKE).proto.type != "primary"
    assert primaries(at) == [STUDENT]
    _neutral(at, "step 6 explainer")
    _button(at, MAKE).click()
    at = ok(at.run())
    first = repo.list_subjects(sid)[0]
    job = repo.explainer_latest(sid, "person", first["id"])
    assert job is not None and job["status"] == "queued"
    assert any("順番を待っています" in i.value for i in at.info)
    assert primaries(at) == [STUDENT]
    explainer.run_pending(repo, settings)
    at = step6(aid)
    assert at.get("video"), "the player"
    assert "解説動画を受け取る" in [b.label for b in at.download_button]
    assert any(b.label == "作り直す" for b in at.button)
    assert primaries(at) == [STUDENT]
    _neutral(at, "step 6 explainer done")


def test_an_outdated_or_failed_video_is_not_played(env, monkeypatch):
    settings, repo, aid, sid = _work(env)
    first = repo.list_subjects(sid)[0]
    explainer.request(repo, sid, first["id"], session_id=None, actor=None)
    explainer.run_pending(repo, settings)
    repo.rename_subject(first["id"], "新しい名前")
    at = step6(aid)
    assert not at.get("video")
    assert "作り直してください" in everything(at)
    from pfb_app.ops import disk

    def low(*a, **k):
        raise disk.DiskLow("low")
    monkeypatch.setattr(disk, "ensure_room", low)
    explainer.request(repo, sid, first["id"], session_id=None, actor=None)
    explainer.run_pending(repo, settings)
    at = step6(aid)
    assert any(explainer.LOW_DISK in w.value for w in at.warning)
    assert any(b.label == "もう一度作る" for b in at.button)


def test_the_tracking_video_from_step_6(env):
    settings, repo, aid, sid = _work(env)
    at = step6(aid)
    _button(at, "追跡の様子の動画を作る").click()
    ok(at.run())
    assert repo.explainer_latest(sid, "tracking", None)["status"] == "queued"
    explainer.run_pending(repo, settings)
    at = step6(aid)
    assert "追跡の様子の動画を受け取る" in [b.label for b in at.download_button]


def test_the_dashboards_person_view_offers_the_video(env):
    settings, repo, aid, sid = _work(env)
    first = repo.list_subjects(sid)[0]
    at = dashboard_page({"work": aid})
    assert not any(b.label == MAKE for b in at.button)      # everyone: no person's video
    who = next(w for w in at.selectbox if w.label == "見る人")
    who.set_value(first["id"])
    at = ok(at.run())
    assert any(b.label == MAKE for b in at.button)
    assert primaries(at) == ["人ごとの表をCSVで受け取る"]
    _neutral(at, "dashboard explainer")


def test_nobody_but_editors_sees_or_makes_one(env, monkeypatch):
    from pfb_app.ui import access
    settings, repo, aid, sid = _work(env)
    from pfb_app.ui.wizard import explainer_view
    monkeypatch.setattr(explainer_view.access, "can_edit", lambda: False)
    at = step6(aid)
    assert not any(b.label in (MAKE, "追跡の様子の動画を作る") for b in at.button)
    assert "解説動画を作る" not in everything(at)
    assert access is explainer_view.access


def test_while_the_video_is_prepared_the_button_waits_with_the_truth(env, monkeypatch):
    from pfb_app.services import moment_clip
    settings, repo, aid, sid = _work(env)
    explainer._probe.cache_clear()
    monkeypatch.setattr(moment_clip, "probe",
                        lambda p: moment_clip.Probe(3840, 2160, "hevc", 30.0))
    at = step6(aid)
    make = _button(at, MAKE)
    assert make.disabled
    assert any("動画の準備をしています（0%）" in str(e.proto) for e in at.get("progress"))
    assert primaries(at) == [STUDENT]


def _progress_script():
    from pfb_app.ui.wizard import explainer_view
    import streamlit as st
    explainer_view._progress(st.session_state["sid"], None)


def test_the_progress_line_stops_for_a_session_that_may_no_longer_edit(env, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from pfb_app.ui.wizard import explainer_view
    settings, repo, aid, sid = _work(env)
    explainer.request(repo, sid, None, session_id=None, actor=None)
    at = AppTest.from_function(_progress_script, default_timeout=60)
    at.session_state["sid"] = sid
    at = ok(at.run())
    assert any("順番を待っています" in i.value for i in at.info)
    monkeypatch.setattr(explainer_view.access, "can_edit", lambda: False)
    at = ok(at.run())
    assert not any("順番を待っています" in i.value for i in at.info)
    assert not at.get("progress")


def test_step_6s_explainer_blocks_are_never_keyed_siblings(env):
    from test_keyed_containers import assert_alone_in_an_unkeyed_block
    settings, repo, aid, sid = _work(env)
    explainer.request(repo, sid, None, session_id=None, actor=None)     # a progress fragment
    at = step6(aid)
    assert_alone_in_an_unkeyed_block(at)
