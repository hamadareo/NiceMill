"""The content check: the card, the coach's confirm and later files of the same work."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage, StageRun
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import runner
from pfb_app.pipeline.common import StageContext
from pfb_app.services.content_check import (CheckChoice, add_next_part, add_videos_to_work,
                                            analysed_part, build_card,
                                            card_sentence, confirm_check, estimate_for,
                                            later_note, start_work_from_videos)
from pfb_app.services.intake import register_video_group, save_uploaded_video
from pfb_app.services.review import BusyError, regenerate_from
from pfb_app.services.roster import parse_roster_text
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

SOCCER = "サッカーの試合のようです（推定）。映っている人は約14人。"
MOTION = "カメラが大きく動く動画は、人の取り違えが増えます"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def make_video(path: Path, seconds=4, fps=10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for i in range(seconds * fps):
        w.write(np.full((120, 160, 3), (i * 7) % 256, dtype=np.uint8))
    w.release()
    return path


def videos(settings, *names) -> list[Path]:
    return [make_video(settings.inbox_dir / n) for n in names]


def probe(median=14, p90=16, mx=18, moving=False, vehicles=(0, 0, 0)):
    return {"duration_s": 4.0, "persons": {"median": median, "p90": p90, "max": mx},
            "vehicles": dict(zip(("median", "p90", "max"), vehicles)),
            "camera_motion": {"median_shift_frac": 0.2 if moving else 0.0,
                              "moving": moving}, "frames": []}


def detect_done(repo, sid, *, cameras=None, scene="sport", sport="soccer", numbers=False,
                skipped=None, duration_s=4.0, activity=None):
    scene_out = None if scene is None else {
        "scene": scene, "sport": sport, "sport_text": sport, "numbers_visible": numbers,
        "model": "fake"}
    if scene_out is not None and activity is not None:   # an engine with scene --activity
        scene_out.update(activity=activity, activity_text=activity)
    output = {"cameras": cameras or {"cam1": probe()}, "scene": scene_out,
              "scene_skipped": skipped if scene is None else None, "duration_s": duration_s}
    _finish_detect_job(repo, sid)
    return repo.save_stage_run(_run(sid, Stage.detect, output))


def _run(sid, stage, output=None, *, camera=None, subject_id=None, error=None, duration=0.0):
    return StageRun(id=None, submission_id=sid, stage=stage, camera=camera,
                    subject_id=subject_id, version=0, source="model", upstream_run_ids=[],
                    prompt_hash=None, model_id="fake", output=output,
                    status="failed" if error else "ok", error=error, warnings=[],
                    duration_sec=duration, created_at="")


def _finish_detect_job(repo, sid, status=JobStatus.done, error=None):
    for job in repo.submission_job_rows(sid):
        if job["stage"] == "detect" and job["status"] in ("queued", "running"):
            repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (job["id"],))
            assert repo.finish_job(job["id"], status, error)


def jobs(repo, sid, stage=None):
    return [j for j in repo.submission_job_rows(sid) if stage is None or j["stage"] == stage]


def choice(**kw) -> CheckChoice:
    base = dict(title="練習試合", activity="サッカーの試合", rubric_text="", domain="sports",
                sport="soccer", method="pfb3x3", max_people=16, target="person",
                start_sec=0.0, end_sec=None, clock_offset_min=0.0)
    base.update(kw)
    return CheckChoice(**base)


# --- starting a work ---------------------------------------------------------------------------

def test_start_work_holds_for_check(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "試合A.mp4", "b.mp4"),
                                      "standard")
    a = repo.get_assignment(aid)
    assert (a["kind"], a["title"], a["task_text"], a["domain"]) == ("video", "試合A", "", "school")
    meta = repo.get_submission(sid)["meta"]
    assert meta["check"] == "pending" and meta["engine"] == "pfb3x3"
    assert sorted(meta["cameras"]) == ["cam1", "cam2"]
    assert [(j["stage"], j["camera"]) for j in jobs(repo, sid)] == [("detect", None)]
    assert not jobs(repo, sid, "prepare")


def test_start_work_at_most_four_cameras(repo, settings):
    paths = videos(settings, *(f"c{i}.mp4" for i in range(5)))
    with pytest.raises(ValueError, match="カメラは4台まで"):
        start_work_from_videos(repo, settings, paths, "standard")
    assert repo.list_assignments() == []


def test_start_work_unreadable_video_leaves_nothing(repo, settings):
    bad = settings.inbox_dir / "bad.mp4"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a video")
    with pytest.raises(Exception):
        start_work_from_videos(repo, settings, [bad], "standard")
    assert repo.list_assignments() == []


def test_register_without_hold_is_unchanged(repo, settings):
    aid = repo.create_assignment("授業", "video", "課題文", "", "standard", True)
    sid = register_video_group(repo, settings, aid, "g", {"a": videos(settings, "x.mp4")[0]},
                               None, None, None, False, [], engine="pfb3x3")
    meta = repo.get_submission(sid)["meta"]
    assert "check" not in meta and "clock_offset_min" not in meta
    assert [j["stage"] for j in jobs(repo, sid)] == ["prepare"]


def test_register_hold_keeps_the_clock_offset(repo, settings):
    aid = repo.create_assignment("授業", "video", "", "", "standard", True)
    sid = register_video_group(repo, settings, aid, "後半", {"a": videos(settings, "x.mp4")[0]},
                               None, None, None, False, [], engine="pfb3x3",
                               hold_for_check=True, clock_offset_min=45)
    meta = repo.get_submission(sid)["meta"]
    assert meta["check"] == "pending" and meta["clock_offset_min"] == 45.0


# --- the card ----------------------------------------------------------------------------------

def test_card_waiting_while_the_check_runs(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    card = build_card(repo, settings, sid)
    assert card.state == "waiting" and not card.needs_pick
    assert repo.get_submission(sid)["meta"]["check"] == "pending"


def test_card_soccer_sentence_and_prefill(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, numbers=True)
    card = build_card(repo, settings, sid)
    assert card.state == "ready" and not card.needs_pick
    assert card.sentence == SOCCER
    assert (card.scene, card.sport, card.people) == ("sport", "soccer", 14)
    assert card.domain == "sports" and card.max_people == 16
    assert card.method == "pfb3x3" and card.target == "person"
    assert card.title.startswith("サッカーの試合 ") and card.activity
    assert card.roster_roles == ["number", "name", "position", "team"]
    assert card.end_sec is None and card.notes == []
    assert repo.get_submission(sid)["meta"]["check"] == "ready"


def test_card_sentence_variants():
    assert card_sentence("sport", "soccer", 14, "person", False) == SOCCER
    assert card_sentence("sport", "motor_racing", 12, "vehicle", False) == \
        "モーターレースのようです（推定）。映っている車は約12台。"
    assert card_sentence("sport", "other", 8, "person", False) == \
        "スポーツの試合のようです（推定）。映っている人は約8人。"
    assert card_sentence("group_work", None, 0, "person", False) == \
        "グループ活動のようです（推定）。人が見つかりませんでした。"
    assert card_sentence("sport", "motor_racing", 0, "vehicle", False) == \
        "モーターレースのようです（推定）。車が見つかりませんでした。"
    assert card_sentence(None, None, 9, "person", True, ask=False) == "映っている人は約9人。"
    assert card_sentence("group_work", None, 4, "person", False) == \
        "グループ活動のようです（推定）。映っている人は約4人。"
    assert card_sentence("other", None, 2, "person", False) == \
        "グループ活動やスポーツではないようです（推定）。映っている人は約2人。"
    assert card_sentence(None, None, 9, "person", True) == \
        "映っている人は約9人。どんな場面の動画か、下から選んでください。"
    assert card_sentence(None, None, None, "person", True) == \
        "内容を確かめられませんでした。どんな場面の動画か、下から選んでください。"


def test_card_motor_racing_counts_vehicles(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "race.mp4"), "standard")
    detect_done(repo, sid, sport="motor_racing", numbers=True,
                cameras={"cam1": probe(median=3, p90=4, mx=5, vehicles=(12, 14, 15))})
    card = build_card(repo, settings, sid)
    assert card.target == "vehicle" and card.people == 12 and card.max_people == 14
    assert card.roster_roles == ["number", "name", "grid"]
    assert card.sentence == "モーターレースのようです（推定）。映っている車は約12台。"
    assert card.activity == "モーターレース"


def test_card_other_sport_wording(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, sport="other")
    card = build_card(repo, settings, sid)
    assert card.sentence == "スポーツの試合のようです（推定）。映っている人は約14人。"
    assert card.activity == "スポーツの試合" and card.domain == "sports"


def test_card_sports_nobody_found_has_no_cap(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, cameras={"cam1": probe(median=0, p90=0, mx=1)})
    card = build_card(repo, settings, sid)
    assert card.max_people is None
    assert card.sentence == "サッカーの試合のようです（推定）。人が見つかりませんでした。"
    confirm_check(repo, settings, sid, choice(max_people=card.max_people))
    assert repo.get_submission(sid)["meta"]["max_people"] == 30


def test_card_roster_roles(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, sport="baseball_softball")
    assert build_card(repo, settings, sid).roster_roles == ["name", "number", "position",
                                                            "batting"]
    _, sid2 = start_work_from_videos(repo, settings, videos(settings, "b.mp4"), "standard")
    detect_done(repo, sid2, scene="group_work", sport=None)
    card = build_card(repo, settings, sid2)
    assert card.domain == "school" and card.roster_roles == ["name", "appearance", "place"]
    assert card.max_people == 18   # school: persons.max


def test_card_sports_cap_is_thirty(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, cameras={"cam1": probe(median=30, p90=41, mx=44)})
    assert build_card(repo, settings, sid).max_people == 30


def test_card_long_video_defaults_to_20_minutes(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, duration_s=3600.0)
    card = build_card(repo, settings, sid)
    assert card.duration_s == 3600.0
    assert card.start_sec == 0.0 and card.end_sec == 1200.0
    assert any("20分" in n for n in card.notes)


def test_card_motion_note(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4", "b.mp4"),
                                    "standard")
    detect_done(repo, sid, cameras={"cam1": probe(), "cam2": probe(moving=True)})
    card = build_card(repo, settings, sid)
    assert card.moving is True
    assert MOTION in card.notes
    assert "短い部分だけを解析すると、取り違えが減ります" in card.notes


def test_card_motion_unknown(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    p = probe()
    p["camera_motion"] = {"median_shift_frac": None, "moving": None}
    detect_done(repo, sid, cameras={"cam1": p})
    card = build_card(repo, settings, sid)
    assert card.moving is None and MOTION not in card.notes


def test_card_gpu_busy_needs_pick(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene=None, skipped="gpu_busy")
    card = build_card(repo, settings, sid)
    assert card.state == "ready" and card.needs_pick
    assert card.scene is None and card.people == 14
    assert card.sentence == "映っている人は約14人。どんな場面の動画か、下から選んでください。"


def test_card_scene_failed_needs_pick(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene=None, skipped="failed")
    card = build_card(repo, settings, sid)
    assert card.needs_pick and card.sentence.startswith("映っている人は約14人。")


def test_card_failed_detect_still_renders(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    _finish_detect_job(repo, sid)
    repo.save_stage_run(_run(sid, Stage.detect, error="Traceback: engine exploded"))
    card = build_card(repo, settings, sid)
    assert card.state == "failed" and card.needs_pick
    assert card.people is None and card.max_people is None
    assert card.sentence == "内容を確かめられませんでした。どんな場面の動画か、下から選んでください。"
    assert "Traceback" not in card.sentence and all("Traceback" not in n for n in card.notes)
    assert card.title.startswith("動画 ") and card.method == "pfb3x3"
    assert repo.get_submission(sid)["meta"]["check"] == "ready"


def test_card_failed_job_without_a_run(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    _finish_detect_job(repo, sid, JobStatus.failed, "KeyError: 'secret internal'")
    card = build_card(repo, settings, sid)
    assert card.state == "failed" and card.needs_pick
    assert "secret" not in card.sentence


def test_card_zero_people(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene="group_work", sport=None,
                cameras={"cam1": probe(median=0, p90=0, mx=0)})
    card = build_card(repo, settings, sid)
    assert card.max_people is None and card.people == 0
    assert card.sentence == "グループ活動のようです（推定）。人が見つかりませんでした。"


def test_estimate_none_without_history(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    assert build_card(repo, settings, sid).estimate_sec is None


def test_estimate_from_history(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4", "b.mp4"),
                                      "standard")
    for stage, sec in ((Stage.prepare, 10), (Stage.track, 100), (Stage.behavior, 200),
                       (Stage.contribution, 20), (Stage.pfb, 5)):
        repo.save_stage_run(_run(sid, stage, {}, camera="old", duration=sec))
    detect_done(repo, sid)
    card = build_card(repo, settings, sid)
    # (10 + 100 + 200 + 20) × 2 cameras + 5 × about 14 people
    assert card.estimate_sec == pytest.approx(330 * 2 + 5 * 14)
    assert estimate_for(repo, "pfb3x3", 1, 0) == pytest.approx(330)


# --- confirming --------------------------------------------------------------------------------

def test_confirm_first_group_sets_work_and_enqueues_prepare(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4", "b.mp4"),
                                      "standard")
    detect_done(repo, sid)
    confirm_check(repo, settings, sid, choice(rubric_text="パスをよく見る", start_sec=1.0,
                                              end_sec=3.0, clock_offset_min=45))
    a = repo.get_assignment(aid)
    assert (a["title"], a["task_text"], a["rubric_text"], a["domain"], a["sport"]) == (
        "練習試合", "サッカーの試合", "パスをよく見る", "sports", "soccer")
    meta = repo.get_submission(sid)["meta"]
    assert meta["check"] == "confirmed" and meta["engine"] == "pfb3x3"
    assert (meta["max_people"], meta["target"]) == (16, "person")
    assert (meta["start_sec"], meta["end_sec"], meta["clock_offset_min"]) == (1.0, 3.0, 45.0)
    assert sorted(j["camera"] for j in jobs(repo, sid, "prepare")) == ["cam1", "cam2"]
    assert not jobs(repo, sid, "names")


def test_confirm_needs_title_and_activity(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    with pytest.raises(ValueError):
        confirm_check(repo, settings, sid, choice(title="  "))
    with pytest.raises(ValueError):
        confirm_check(repo, settings, sid, choice(activity=""))
    assert repo.get_submission(sid)["meta"]["check"] == "pending"
    assert not jobs(repo, sid, "prepare")


@pytest.mark.parametrize("kw", [dict(max_people=31), dict(domain="school", sport=None,
                                                          max_people=21),
                                dict(max_people=0), dict(sport="cricket"),
                                dict(method="paper"), dict(target="horse"),
                                dict(end_sec=99.0), dict(clock_offset_min=-1)])
def test_confirm_rejects_bad_choices(repo, settings, kw):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    with pytest.raises(ValueError):
        confirm_check(repo, settings, sid, choice(**kw))
    assert repo.get_submission(sid)["meta"]["check"] == "pending"
    assert not jobs(repo, sid, "prepare")


def test_confirm_without_cap_uses_the_domain_limit(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice(domain="school", sport=None, max_people=None,
                                              activity="話し合い"))
    meta = repo.get_submission(sid)["meta"]
    assert meta["max_people"] == 20
    assert repo.get_assignment(repo.get_submission(sid)["assignment_id"])["sport"] is None


def test_confirm_school_paper_method(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice(domain="school", sport=None, method="paper",
                                              max_people=None, activity="話し合い"))
    meta = repo.get_submission(sid)["meta"]
    assert meta["engine"] == "paper" and "max_people" not in meta and "target" not in meta
    assert len(jobs(repo, sid, "prepare")) == 1


def test_confirm_while_the_check_still_runs(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice())
    assert repo.get_submission(sid)["meta"]["check"] == "confirmed"
    assert len(jobs(repo, sid, "prepare")) == 1
    assert build_card(repo, settings, sid).state == "waiting"
    assert repo.get_submission(sid)["meta"]["check"] == "confirmed"


def test_confirm_twice_is_busy(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    confirm_check(repo, settings, sid, choice())
    with pytest.raises(BusyError):
        confirm_check(repo, settings, sid, choice(title="別の名前"))
    assert len(jobs(repo, sid, "prepare")) == 1


def test_confirm_a_group_registered_without_check_is_busy(repo, settings):
    aid = repo.create_assignment("授業", "video", "課題文", "", "standard", True)
    sid = register_video_group(repo, settings, aid, "g", {"a": videos(settings, "x.mp4")[0]},
                               None, None, None, False, [], engine="pfb3x3")
    with pytest.raises(BusyError):
        confirm_check(repo, settings, sid, choice())


def test_confirm_applies_the_roster(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    roster = parse_roster_text("背番号\t名前\n10\t山田\n7\t佐藤")
    confirm_check(repo, settings, sid, choice(), roster=roster)
    assert sorted(s["display_name"] for s in repo.list_subjects(sid)) == ["佐藤", "山田"]
    assert not jobs(repo, sid, "names")


def _later_group(repo, settings, aid, name="b.mp4"):
    return add_videos_to_work(repo, settings, aid, videos(settings, name))


def test_confirm_later_group_keeps_work(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    confirm_check(repo, settings, sid, choice())
    before = repo.get_assignment(aid)
    sid2 = _later_group(repo, settings, aid)
    detect_done(repo, sid2, sport="basketball")
    card = build_card(repo, settings, sid2)
    assert (card.domain, card.sport, card.title, card.activity) == (
        "sports", "soccer", "練習試合", "サッカーの試合")
    confirm_check(repo, settings, sid2, choice(title="違う", activity="違う", domain="school",
                                               sport=None, max_people=10,
                                               clock_offset_min=45))
    assert repo.get_assignment(aid) == before
    meta = repo.get_submission(sid2)["meta"]
    assert meta["check"] == "confirmed" and meta["clock_offset_min"] == 45.0
    assert meta["max_people"] == 10
    assert len(jobs(repo, sid2, "prepare")) == 1


def test_later_group_cap_follows_the_work_domain(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice())
    sid2 = _later_group(repo, settings, aid)
    with pytest.raises(ValueError):   # a sports work: up to 30, whatever the choice says
        confirm_check(repo, settings, sid2, choice(domain="school", sport=None, max_people=31))
    confirm_check(repo, settings, sid2, choice(domain="school", sport=None, max_people=25))
    assert repo.get_submission(sid2)["meta"]["max_people"] == 25


def test_later_note_on_disagreement(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    assert later_note(repo, sid) is None
    confirm_check(repo, settings, sid, choice())
    assert later_note(repo, sid) is None
    sid2 = _later_group(repo, settings, aid)
    assert later_note(repo, sid2) is None            # not checked yet
    detect_done(repo, sid2, sport="basketball")
    note = ("この動画はバスケットボールの試合のようですが、"
            "この試合・練習はサッカーの試合として進めます。")
    assert later_note(repo, sid2) == note
    assert note in build_card(repo, settings, sid2).notes
    sid3 = _later_group(repo, settings, aid, "c.mp4")
    detect_done(repo, sid3)                          # soccer again: no note
    assert later_note(repo, sid3) is None
    sid4 = _later_group(repo, settings, aid, "d.mp4")
    detect_done(repo, sid4, scene=None, skipped="gpu_busy")
    assert later_note(repo, sid4) is None


def test_later_note_school_work(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice(domain="school", sport=None, max_people=4,
                                              activity="話し合い"))
    sid2 = _later_group(repo, settings, aid)
    detect_done(repo, sid2, scene="other", sport=None)
    assert later_note(repo, sid2) is None
    sid3 = _later_group(repo, settings, aid, "c.mp4")
    detect_done(repo, sid3)
    assert later_note(repo, sid3) == ("この動画はサッカーの試合のようですが、"
                                      "この活動はグループ活動として進めます。")


def test_later_group_never_needs_a_pick(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice())
    sid2 = _later_group(repo, settings, aid)
    detect_done(repo, sid2, scene=None, skipped="gpu_busy")
    card = build_card(repo, settings, sid2)
    assert card.state == "ready" and not card.needs_pick
    assert (card.scene, card.sport, card.domain) == ("sport", "soccer", "sports")
    assert card.sentence == "映っている人は約14人。"
    sid3 = _later_group(repo, settings, aid, "c.mp4")
    _finish_detect_job(repo, sid3)
    repo.save_stage_run(_run(sid3, Stage.detect, error="boom"))
    card = build_card(repo, settings, sid3)
    assert card.state == "failed" and not card.needs_pick
    assert card.sentence == "内容を確かめられませんでした。"


def test_confirm_waits_for_the_first_group(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    sid2 = _later_group(repo, settings, aid)
    with pytest.raises(BusyError, match="先に最初の動画を決定してください"):
        confirm_check(repo, settings, sid2, choice(title="後半"))
    assert repo.get_submission(sid2)["meta"]["check"] == "pending"
    assert not jobs(repo, sid2, "prepare")
    confirm_check(repo, settings, sid, choice())
    confirm_check(repo, settings, sid2, choice(title="後半"))
    assert repo.get_assignment(aid)["title"] == "練習試合"


def test_add_videos_to_work(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    with pytest.raises(ValueError, match="カメラは4台まで"):
        add_videos_to_work(repo, settings, aid, videos(settings, *(f"c{i}.mp4" for i in range(5))))
    assert len(repo.list_submissions(aid)) == 1
    src = make_video(settings.inbox_dir / "tmp" / "後半.mp4")
    with open(src, "rb") as f:
        up = save_uploaded_video(settings, "後半.mp4", f)
    assert up.name != "後半.mp4"
    sid2 = add_videos_to_work(repo, settings, aid, [up, videos(settings, "d.mp4")[0]])
    sub = repo.get_submission(sid2)
    assert sub["label"] == "後半" and sub["meta"]["check"] == "pending"
    assert sub["meta"]["auto_transcribe"] is True and sub["meta"]["engine"] == "pfb3x3"
    assert sorted(sub["meta"]["cameras"]) == ["cam1", "cam2"]
    assert [j["stage"] for j in jobs(repo, sid2)] == ["detect"]


def test_start_work_title_without_the_upload_prefix(repo, settings):
    src = make_video(settings.inbox_dir / "tmp" / "決勝.mp4")
    with open(src, "rb") as f:
        up = save_uploaded_video(settings, "決勝.mp4", f)
    aid, _ = start_work_from_videos(repo, settings, [up], "standard")
    assert repo.get_assignment(aid)["title"] == "決勝"


# --- the check's end in the runner ---------------------------------------------------------------

def _run_detect_job(repo, settings, sid, monkeypatch, outcome):
    def fake_stage(job, ctx):
        if isinstance(outcome, BaseException):
            raise outcome
        return [repo.save_stage_run(_run(sid, Stage.detect, outcome[0], error=outcome[1]))]
    monkeypatch.setattr(runner, "_run_stage", fake_stage)
    job_id = jobs(repo, sid, "detect")[-1]["id"]
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (job_id,))
    ctx = StageContext(settings, repo, None, lambda: False)
    try:
        runner.execute_job(repo.get_job(job_id), ctx)
    except YieldRequested:
        pass
    return repo.get_job(job_id)


@pytest.mark.parametrize("outcome,status", [
    (({"cameras": {"cam1": probe()}, "scene": None, "scene_skipped": "gpu_busy",
       "duration_s": 4.0}, None), "done"),
    ((None, "内容を確かめられませんでした。"), "failed"),
    (RuntimeError("engine exploded"), "failed_retryable"),
])
def test_runner_marks_the_check_ready_when_detect_ends(repo, settings, monkeypatch, outcome,
                                                       status):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    job = _run_detect_job(repo, settings, sid, monkeypatch, outcome)
    assert job.status.value == status
    meta = repo.get_submission(sid)["meta"]
    assert meta["check"] == "ready" and sorted(meta["cameras"]) == ["cam1"]
    assert meta["engine"] == "pfb3x3"


def test_runner_keeps_the_check_pending_on_yield(repo, settings, monkeypatch):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    job = _run_detect_job(repo, settings, sid, monkeypatch, YieldRequested("gpu"))
    assert job.status.value == "queued"
    assert repo.get_submission(sid)["meta"]["check"] == "pending"


def test_runner_never_touches_a_confirmed_check(repo, settings, monkeypatch):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice())
    _run_detect_job(repo, settings, sid, monkeypatch, (None, "x"))
    assert repo.get_submission(sid)["meta"]["check"] == "confirmed"


def test_card_promotion_survives_a_locked_database(repo, settings, monkeypatch):
    import sqlite3

    from pfb_app.services import content_check
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)

    def locked(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    monkeypatch.setattr(content_check, "mark_check_ready", locked)
    assert build_card(repo, settings, sid).state == "ready"
    assert repo.get_submission(sid)["meta"]["check"] == "pending"


# --- regenerate before the confirm ---------------------------------------------------------------

def test_regenerate_only_the_check_before_the_confirm(repo, settings):
    from pydantic import ValidationError
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    with pytest.raises(ValidationError, match="動画の内容を決定してから"):
        regenerate_from(repo, sid, Stage.prepare)
    assert not jobs(repo, sid, "prepare")
    assert regenerate_from(repo, sid, Stage.detect)
    _finish_detect_job(repo, sid)
    confirm_check(repo, settings, sid, choice())
    for j in jobs(repo, sid, "prepare"):
        repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (j["id"],))
    assert regenerate_from(repo, sid, Stage.prepare)


# --- the short card: default title, facts and the model choice (ux round 2, L3) ---------------

def test_default_title_is_the_scene_and_the_day():
    from datetime import date

    from pfb_app.services.content_check import default_title
    day = date(2026, 10, 22)
    assert default_title("sport", "soccer", day) == "サッカーの試合 10月22日"
    assert default_title("sport", "motor_racing", day) == "モーターレース 10月22日"
    assert default_title("sport", None, day) == "スポーツの試合 10月22日"
    assert default_title("group_work", None, date(2026, 3, 5)) == "グループ活動 3月5日"
    assert default_title("other", None, day) == "動画 10月22日"
    assert default_title(None, None, day) == "動画 10月22日"


def test_card_title_defaults_to_scene_and_day_not_the_file_name(repo, settings):
    from pfb_app.services.content_check import work_day
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "IMG_4821.mp4"),
                                      "standard")
    assert repo.get_assignment(aid)["title"] == "IMG_4821"     # home tells works apart meanwhile
    detect_done(repo, sid)
    card = build_card(repo, settings, sid)
    day = work_day(repo.get_assignment(aid))
    assert card.auto_title and card.day == day
    assert card.title == f"サッカーの試合 {day.month}月{day.day}日"


def test_card_title_of_an_uploaded_file_is_also_a_default(repo, settings):
    path = save_uploaded_video(settings, "IMG_1.mp4",
                               make_video(settings.inbox_dir / "x.mp4").open("rb"))
    aid, sid = start_work_from_videos(repo, settings, [path], "standard")
    detect_done(repo, sid, scene="group_work", sport=None)
    card = build_card(repo, settings, sid)
    assert card.auto_title and card.title.startswith("グループ活動 ")


def test_card_failed_check_title_is_a_plain_default(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    _finish_detect_job(repo, sid, JobStatus.failed, "boom")
    card = build_card(repo, settings, sid)
    assert card.needs_pick and card.title.startswith("動画 ")


def test_card_keeps_a_title_the_coach_gave(repo, settings):
    aid = repo.create_assignment("ペーパータワー", "video", "", "", "standard", True)
    sid = add_videos_to_work(repo, settings, aid, videos(settings, "a.mp4"))
    detect_done(repo, sid)
    card = build_card(repo, settings, sid)
    assert card.sets_work and not card.auto_title and card.title == "ペーパータワー"


def test_check_facts_three_lines():
    from pfb_app.services.content_check import check_facts
    facts = check_facts(scene="sport", sport="soccer", people=14, target="person",
                        start_sec=0.0, end_sec=1200.0, estimate_sec=35 * 60)
    assert facts == [("種目", "サッカー"), ("評価する人", "約14人"),
                     ("解析する部分", "はじめの20分（解析の目安 約35分）")]


@pytest.mark.parametrize("kw,expected", [
    (dict(scene="group_work", sport=None), ("場面", "グループ活動")),
    (dict(scene="other", sport=None), ("場面", "その他の動画")),
    (dict(scene=None, sport=None), ("場面", "まだ選ばれていません")),
    (dict(scene="sport", sport=None), ("種目", "まだ選ばれていません")),
    (dict(scene="sport", sport="other"), ("種目", "その他のスポーツ")),
])
def test_check_facts_scene_line(kw, expected):
    from pfb_app.services.content_check import check_facts
    assert check_facts(people=3, target="person", start_sec=0.0, end_sec=None,
                       estimate_sec=None, **kw)[0] == expected


@pytest.mark.parametrize("kw,expected", [
    (dict(people=None), ("評価する人", "動画から自動で数えます")),
    (dict(people=0), ("評価する人", "見つかりませんでした")),
    (dict(people=12, target="vehicle"), ("評価する車・バイク", "約12台")),
    (dict(people=14, limit=10), ("評価する人", "約10人")),
])
def test_check_facts_people_line(kw, expected):
    from pfb_app.services.content_check import check_facts
    base = dict(scene="sport", sport="soccer", target="person", start_sec=0.0, end_sec=None,
                estimate_sec=None)
    base.update(kw)
    assert check_facts(**base)[1] == expected


@pytest.mark.parametrize("start,end,duration,expected", [
    (0.0, None, None, "全体"),
    (0.0, None, 754.0, "全体（約13分）"),
    (0.0, 90.0, None, "はじめの1.5分"),
    (300.0, None, None, "5分から最後まで"),
    (300.0, 1500.0, None, "5分〜25分"),
])
def test_check_facts_part_line(start, end, duration, expected):
    from pfb_app.services.content_check import check_facts
    facts = check_facts(scene="sport", sport="soccer", people=3, target="person",
                        start_sec=start, end_sec=end, estimate_sec=None, duration_s=duration)
    assert facts[2] == ("解析する部分", expected)


def test_confirm_can_change_the_works_model(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    confirm_check(repo, settings, sid, choice(model_key="high"))
    assert repo.get_assignment(aid)["model_key"] == "high"


def test_confirm_without_a_model_keeps_the_works(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "high")
    confirm_check(repo, settings, sid, choice())
    assert repo.get_assignment(aid)["model_key"] == "high"


def test_confirm_rejects_an_unknown_model(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    with pytest.raises(ValueError):
        confirm_check(repo, settings, sid, choice(model_key="huge"))
    assert repo.get_assignment(aid)["model_key"] == "standard"
    assert not jobs(repo, sid, "prepare")


def test_a_later_group_never_changes_the_works_model(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    confirm_check(repo, settings, sid, choice())
    sid2 = _later_group(repo, settings, aid)
    confirm_check(repo, settings, sid2, choice(model_key="high"))
    assert repo.get_assignment(aid)["model_key"] == "standard"


def test_add_videos_refuses_a_video_the_work_already_uses(repo, settings):
    (path,) = videos(settings, "a.mp4")
    aid, sid = start_work_from_videos(repo, settings, [path], "standard")
    with pytest.raises(ValueError, match="もう入っています"):
        add_videos_to_work(repo, settings, aid, [path])
    assert len(repo.list_submissions(aid)) == 1


def test_used_video_paths_lists_every_works_cameras(repo, settings):
    from pfb_app.services.content_check import used_video_paths
    a, b = videos(settings, "a.mp4", "b.mp4")
    start_work_from_videos(repo, settings, [a], "standard")
    assert used_video_paths(repo) == {a.resolve()}


def test_numbered_title():
    from pfb_app.services.content_check import numbered
    assert numbered("動画 10月22日", set()) == "動画 10月22日"
    assert numbered("動画 10月22日", {"動画 10月22日"}) == "動画 10月22日（2）"
    assert numbered("動画 10月22日", {"動画 10月22日", "動画 10月22日（2）"}) == "動画 10月22日（3）"


# --- activity-type profiles ---------------------------------------------------------------------

def test_card_meeting_profile_from_the_activity_verdict(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene="group_work", sport=None, activity="meeting")
    card = build_card(repo, settings, sid)
    assert card.profile == "meeting" and card.domain == "school" and not card.needs_pick
    assert card.sentence == "話し合い・会議のようです（推定）。映っている人は約14人。"
    assert card.title.startswith("会議 ") and card.activity == "話し合い・会議"


@pytest.mark.parametrize("scene,activity,profile", [
    ("sport", None, "sports"), ("group_work", None, "general"), ("other", None, "general"),
    ("group_work", "group_task", "general"), ("other", "presentation", "presentation"),
    ("group_work", "workshop", "workshop"), ("group_work", "learning", "learning"),
])
def test_card_profile_follows_the_verdict(repo, settings, scene, activity, profile):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene=scene, sport="soccer" if scene == "sport" else None,
                activity=activity)
    assert build_card(repo, settings, sid).profile == profile


def test_card_without_a_verdict_has_no_profile(repo, settings):
    _, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene=None, skipped="gpu_busy")
    card = build_card(repo, settings, sid)
    assert card.needs_pick and card.profile is None


def test_confirm_stores_the_profile(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene="group_work", sport=None, activity="meeting")
    confirm_check(repo, settings, sid, choice(domain="school", sport=None, title="定例会",
                                              activity="週の定例会", profile="presentation"))
    work = repo.get_assignment(aid)
    assert work["profile"] == "presentation" and work["domain"] == "school"


def test_confirm_without_a_profile_uses_the_domain_default(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    confirm_check(repo, settings, sid, choice())
    assert repo.get_assignment(aid)["profile"] == "sports"


@pytest.mark.parametrize("kw", [dict(profile="sports", domain="school", sport=None),
                                dict(profile="meeting"), dict(profile="zzz")])
def test_confirm_rejects_a_profile_of_another_domain(repo, settings, kw):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    with pytest.raises(ValueError):
        confirm_check(repo, settings, sid, choice(**kw))
    assert repo.get_assignment(aid)["profile"] == "general"     # nothing changed
    assert not jobs(repo, sid, "prepare")


def test_later_group_keeps_the_work_profile(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, scene="group_work", sport=None, activity="workshop")
    confirm_check(repo, settings, sid, choice(domain="school", sport=None, title="研修",
                                              activity="新人研修", profile="workshop"))
    later = _later_group(repo, settings, aid)
    detect_done(repo, later, scene="group_work", sport=None, activity="meeting")
    card = build_card(repo, settings, later)
    assert not card.sets_work and card.profile == "workshop"
    confirm_check(repo, settings, later, choice(domain="sports", profile="sports"))
    assert repo.get_assignment(aid)["profile"] == "workshop"
    detect_done(repo, _later_group(repo, settings, aid, "c.mp4"))
    note = later_note(repo, repo.list_submissions(aid)[-1]["id"])
    assert note == "この動画はサッカーの試合のようですが、この研修は研修・ワークショップとして進めます。"


def test_profile_facts_and_titles():
    from datetime import date

    from pfb_app.services.content_check import check_facts, default_title
    day = date(2026, 10, 23)
    assert default_title("group_work", None, day, "learning") == "学習活動 10月23日"
    assert default_title("group_work", None, day, "general") == "グループ活動 10月23日"
    assert default_title("sport", "soccer", day, "sports") == "サッカーの試合 10月23日"
    facts = check_facts(scene="group_work", sport=None, people=3, target="person",
                        start_sec=0.0, end_sec=None, estimate_sec=None, profile="meeting")
    assert facts[0] == ("場面", "話し合い・会議")
    assert check_facts(scene="group_work", sport=None, people=3, target="person",
                       start_sec=0.0, end_sec=None, estimate_sec=None,
                       profile="general")[0] == ("場面", "グループ活動")


# --- a group analysed in part -------------------------------------------------------------------

def _long_group(repo, settings, duration=1800.0):
    # a real file as long as the check says (1 frame a second keeps it small)
    video = make_video(settings.inbox_dir / "long.mp4", seconds=int(duration), fps=1)
    aid, sid = start_work_from_videos(repo, settings, [video], "standard")
    detect_done(repo, sid, duration_s=duration)
    card = build_card(repo, settings, sid)
    confirm_check(repo, settings, sid, choice(start_sec=card.start_sec, end_sec=card.end_sec))
    return aid, sid


def test_a_long_video_says_which_part_was_analysed(repo, settings):
    aid, sid = _long_group(repo, settings)
    part = analysed_part(repo, sid)
    assert (part.start_sec, part.end_sec, part.duration_s) == (0.0, 1200.0, 1800.0)
    assert part.sentence == "動画は約30分です。はじめの20分だけを解析しました。"
    assert part.next_label == "つづきも解析する（20分〜）"


def test_a_later_part_names_its_range(repo, settings):
    from pfb_app.services.content_check import AnalysedPart
    assert AnalysedPart(1200.0, 2400.0, 3600.0).sentence == \
        "動画は約60分です。20分〜40分だけを解析しました。"


def test_no_message_when_the_whole_video_was_analysed(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid, duration_s=600.0)
    confirm_check(repo, settings, sid, choice())
    assert analysed_part(repo, sid) is None
    aid2, sid2 = _long_group(repo, settings, duration=1210.0)   # 10 s left over: not worth it
    assert analysed_part(repo, sid2) is None


def test_next_part_is_one_more_group_from_where_the_analysis_ended(repo, settings):
    aid, sid = _long_group(repo, settings)
    sid2 = add_next_part(repo, settings, aid, sid)
    sub, first = repo.get_submission(sid2), repo.get_submission(sid)
    assert sub["assignment_id"] == aid and sub["meta"]["check"] == "pending"
    assert sub["meta"]["cameras"] == first["meta"]["cameras"] and sub["meta"]["start_sec"] == 1200.0
    assert [j["stage"] for j in jobs(repo, sid2)] == ["detect"]
    with pytest.raises(ValueError, match="もう追加されています"):
        add_next_part(repo, settings, aid, sid)
    with pytest.raises(ValueError, match="つづきがありません"):
        add_next_part(repo, settings, aid + 1, sid)


def test_next_part_of_a_whole_video_is_refused(repo, settings):
    aid, sid = start_work_from_videos(repo, settings, videos(settings, "a.mp4"), "standard")
    detect_done(repo, sid)
    confirm_check(repo, settings, sid, choice())
    with pytest.raises(ValueError, match="つづきがありません"):
        add_next_part(repo, settings, aid, sid)
