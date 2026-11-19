"""The admin status page (処理状況) says which tracker found the people of each camera, when
the engine reported it (engine 58f8341), and nothing when it did not (no TID data, older
engines)."""
from streamlit.testing.v1 import AppTest

from pfb_app.domain.models import Stage, StageRun
from test_wizard_ui_basic import UI, env, ok, register, video_assignment  # noqa: F401 (env)

TITLE = "人を見つける方法"


def save_track(repo, sid, output, camera="cam1"):
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.track, camera=camera, subject_id=None,
        version=0, source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def status_page() -> AppTest:
    return ok(AppTest.from_file(str(UI / "pages" / "status.py"), default_timeout=60).run())


def captions(at: AppTest) -> list[str]:
    return [c.value for c in at.caption]


def test_status_page_names_the_tracker(env):
    settings, repo = env
    aid = video_assignment(repo, "練習試合")
    sid = register(settings, repo, aid, "A班")
    save_track(repo, sid, {"people": 3, "tracker": {"used": "trackid3x3", "fallback": None}})
    other = register(settings, repo, aid, "B班")
    save_track(repo, other, {"people": 3, "tracker": {"used": "engine",
                                                      "fallback": "not a TID dataset video"}})
    at = status_page()
    assert TITLE in [s.value for s in at.subheader]
    assert "練習試合／A班（カメラ1）: TrackID3x3 の追跡（すでにある結果）" in captions(at)
    assert "練習試合／B班（カメラ1）: 標準の追跡" in captions(at)


def test_status_page_without_tracker_data(env):
    settings, repo = env
    aid = video_assignment(repo, "練習試合")
    sid = register(settings, repo, aid, "A班")
    save_track(repo, sid, {"people": 3})            # an engine before 58f8341
    register(settings, repo, aid, "B班")            # no track run at all
    at = status_page()
    assert TITLE not in [s.value for s in at.subheader]
    assert not any("の追跡" in c for c in captions(at))
