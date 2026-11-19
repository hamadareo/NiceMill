"""The progress panel of the waiting screens: the held overall bar, the stage rows ending with
the review line, and the form signature that ignores the bars.

Only synthetic data: invented names, tiny black PNGs, throwaway databases."""
import re

from pfb_app.domain.models import Stage
from pfb_app.ui.wizard import progress_panel, step_people
from pfb_app.ui.wizard.bars import StageRow
from pfb_app.ui.wizard.progress import View, work_progress
from test_wizard_ui_basic import app, env, html, ok, shown_step, video_assignment  # noqa: F401
from test_wizard_ui_people_wait import (confirmed_and_analysing, group, job, save, track,
                                        worker_alive)


def test_panel_holds_the_bar_across_ticks(env):
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = group(repo, aid)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    jid = job(repo, sid, Stage.track, "running", camera="cam1", model="track")
    # fallback weights: content check 1 + preparation 1 + half of the tracking 4 = 4 / 22.5
    repo.set_job_progress(jid, 0.5, "人物を追跡しています")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 3
    assert "全体の進み具合 17%" in html(at)
    repo.set_job_progress(jid, 0.1, "人物を追跡しています")      # a reclaimed job: 10 %
    ok(at.run())
    assert "全体の進み具合 17%" in html(at) and "全体の進み具合 10%" not in html(at)
    # a real redo: the tracking finished once, then a new tracking starts from nothing
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (jid,))
    track(settings, repo, sid)
    job(repo, sid, Stage.track, "running", camera="cam1", model="track")
    ok(at.run())
    assert "全体の進み具合 8%" in html(at)                        # lower, once
    assert "やり直しています" in html(at)
    ok(at.run())
    assert "全体の進み具合 8%" in html(at)


def test_stage_rows_end_with_the_review_line(env):
    settings, repo = env
    aid = video_assignment(repo)
    confirmed_and_analysing(settings, repo, aid)
    rows = progress_panel.stage_rows(work_progress(repo, aid).analysis)
    assert [r.name for r in rows] == ["内容の確認", "準備", "人を見つける", "評価する人を選ぶ",
                                      "したことの書き出し", "よかった点を選ぶ", "文章づくり",
                                      "確認と承認"]
    assert rows[-1] == StageRow("確認と承認", "queued", 0, True, "承認 0 / 全 3")


def test_unknown_progress_rows_say_it_is_moving(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    repo.conn.execute("UPDATE jobs SET status='running' WHERE submission_id=?", (sid,))
    row = progress_panel.stage_rows(work_progress(repo, aid).analysis)[4]
    assert (row.name, row.status, row.known, row.note) == ("したことの書き出し", "running", False,
                                                            "進んでいます")


def test_form_signature_ignores_progress(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid, cameras=("cam1", "cam2"))
    track(settings, repo, sid, "cam1")                  # cam1: the form
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam2")
    jid = job(repo, sid, Stage.track, "running", camera="cam2", model="track")
    repo.set_job_progress(jid, 0.2, "人物を追跡しています")
    form = step_people._signature(aid, sid, form=True)
    plain = step_people._signature(aid, sid, form=False)
    before, plain_before = form(View.load(repo, aid)), plain(View.load(repo, aid))
    repo.set_job_progress(jid, 0.9, "人物を追跡しています")
    assert form(View.load(repo, aid)) == before          # unsaved choices are never thrown away
    assert plain(View.load(repo, aid)) != plain_before   # the page follows every 10 points


def test_held_percent_never_goes_back_and_drops_once_on_a_redo(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)
    jid = repo.conn.execute("SELECT id FROM jobs WHERE submission_id=? AND stage='behavior'",
                            (sid,)).fetchone()[0]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    state: dict = {}
    monkeypatch.setattr(progress_panel.st, "session_state", state)
    repo.set_job_progress(jid, 0.3, None)
    assert progress_panel.held_percent(aid, work_progress(repo, aid).analysis) == 42
    repo.set_job_progress(jid, 0.0, None)
    assert progress_panel.held_percent(aid, work_progress(repo, aid).analysis) == 42
    assert state[f"_pfb_hold_{aid}"][1] is False


def test_waiting_for_the_coach_shows_your_turn_without_a_percentage(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = group(repo, aid)
    track(settings, repo, sid)                          # tracked, nobody mapped: your turn
    analysis = work_progress(repo, aid).analysis
    assert analysis.waiting_for_you and not analysis.active
    at = ok(app(session={"assignment_id": aid}))
    states = re.findall(r'<div class="pfb-navstate[^"]*">(.*?)</div>', html(at))
    assert shown_step(at) == 3 and "あなたの番" in states[2]
    assert not any("%" in s for s in states)
