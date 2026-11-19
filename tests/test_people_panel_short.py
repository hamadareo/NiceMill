"""Step 5 when the tracker found more people than are evaluated, and when a video was analysed
only in part (synthetic data; the services are replaced where they would start a job)."""
import json

from pfb_app.domain.models import Stage
from pfb_app.services import auto_people, content_check
from test_people_panel import no_internal_terms
from test_wizard_ui_basic import (app, env, ok, primaries, screen_text,  # noqa: F401
                                  video_assignment)
from test_wizard_ui_review_pdf import reviewable
from test_wizard_ui_people_wait import save

EXTRA = ["P7", "P8", "P9", "P10"]


def auto_group(settings, repo, aid, extra=EXTRA):
    """A reviewable group whose automatic mapping left `extra` labels out (on screen 12 s)."""
    sid = reviewable(settings, repo, aid)   # three people mapped; its label P4 is left over
    track = repo.latest_run(sid, Stage.track, "cam1")
    out = {**track.output, "labels": [*track.output["labels"], *extra],
           "quality": {lb: {"coverage": 0.05, "mean_h_frac": 0.15, "frames": 120}
                       for lb in extra}}
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?", (json.dumps(out), track.id))
    repo.conn.execute("UPDATE stage_runs SET source='auto' WHERE submission_id=? AND stage='identity'",
                      (sid,))
    boxes = settings.data_dir / f"boxes_{sid}.txt"
    boxes.write_text("1,1,0,0,10,10,0.9\n1,4,0,0,10,10,0.9\n", encoding="utf-8")
    (settings.data_dir / "meta.json").write_text(json.dumps(
        {"fps": 10.0, "n_frames": 3000, "width": 1280, "height": 720}))
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?",
                      (json.dumps({**out, "boxes": str(boxes)}), track.id))
    return sid


def test_many_people_left_out_are_explained_above_the_closed_panel(env):
    settings, repo = env
    aid = video_assignment(repo)
    auto_group(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    page = screen_text(at)
    assert "映っている時間が短い人が5人いて、評価には入れていません。" in page
    assert "下の「映っている人」から選べます" in page
    no_internal_terms(at)


def test_the_panel_lists_them_with_how_long_they_were_on_screen(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = auto_group(settings, repo, aid)
    called = []
    monkeypatch.setattr(auto_people, "include_person",
                        lambda repo_, settings_, sid_, cam, lb: called.append((sid_, cam, lb)) or [])
    at = ok(app(session={"assignment_id": aid}))
    page = screen_text(at)
    assert "映っている時間が短い人" in page and page.count("約12秒映っていました") == 4 \
        and "約1秒映っていました" in page   # P4: one box
    button = at.button(key="pfb_people_incl_cam1_P7")
    assert button.label == "この人も評価する" and button.proto.type == "secondary"
    assert primaries(at) == ["全員まとめて承認"]
    button.click().run()
    ok(at)
    assert called == [(sid, "cam1", "P7")]


def test_one_short_person_is_listed_but_not_announced(env):
    settings, repo = env
    aid = video_assignment(repo)
    auto_group(settings, repo, aid, extra=["P7"])
    at = ok(app(session={"assignment_id": aid}))
    page = screen_text(at)
    assert "約12秒映っていました" in page and "評価には入れていません" not in page


def test_no_explanation_when_the_coach_chose_the_people(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = auto_group(settings, repo, aid)
    repo.conn.execute("UPDATE stage_runs SET source='teacher' WHERE submission_id=? "
                      "AND stage='identity'", (sid,))
    page = screen_text(ok(app(session={"assignment_id": aid})))
    assert "評価には入れていません" not in page and "映っている時間が短い人" not in page


def long_video(repo, sid, duration=1800.0, end=1200.0):
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "start_sec": 0.0, "end_sec": end})
    save(repo, sid, Stage.detect, {"cameras": {"cam1": {}}, "scene": None,
                                   "duration_s": duration})


def test_a_long_video_says_only_the_first_part_was_analysed(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    long_video(repo, sid)
    added = []
    monkeypatch.setattr(content_check, "add_next_part",
                        lambda *a: added.append(a[2:]) or 99)
    import pfb_app.ui.wizard.step_review as step_review
    monkeypatch.setattr(step_review, "add_next_part", content_check.add_next_part)
    at = ok(app(session={"assignment_id": aid}))
    assert "動画は約30分です。はじめの20分だけを解析しました。" in screen_text(at)
    button = at.button(key=f"pfb_next_part_{sid}")
    assert button.label == "つづきも解析する（20分〜）" and button.proto.type == "secondary"
    assert primaries(at) == ["全員まとめて承認"]
    no_internal_terms(at)
    button.click().run()
    ok(at)
    assert added == [(aid, sid)]


def test_no_button_once_the_continuation_is_added(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    long_video(repo, sid)
    other = repo.get_submission(sid)
    monkeypatch.setattr(content_check, "next_part_group", lambda repo_, sid_: other)
    import pfb_app.ui.wizard.step_review as step_review
    monkeypatch.setattr(step_review, "next_part_group", content_check.next_part_group)
    at = ok(app(session={"assignment_id": aid}))
    page = screen_text(at)
    assert "はじめの20分だけを解析しました" in page and "つづきは、別のグループとして追加されています" in page
    assert not any(b.key == f"pfb_next_part_{sid}" for b in at.button)


def test_a_whole_video_has_no_message(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    long_video(repo, sid, duration=1200.0, end=1200.0)
    assert "だけを解析しました" not in screen_text(ok(app(session={"assignment_id": aid})))


def test_the_continuation_button_really_adds_a_held_group(env):
    """End to end with a real (1 frame a second) video: the click adds the work's next group,
    held for the content check from where the first part ended, and the screen follows."""
    import cv2
    import numpy as np
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    path = settings.inbox_dir / "long.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1, (160, 120))
    for i in range(1800):
        w.write(np.full((120, 160, 3), i % 256, dtype=np.uint8))
    w.release()
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "cameras": {"cam1": str(path)}})
    long_video(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    at.button(key=f"pfb_next_part_{sid}").click().run()
    ok(at)
    later = [s for s in repo.list_submissions(aid) if s["id"] != sid]
    assert len(later) == 1
    assert later[0]["meta"]["start_sec"] == 1200.0 and later[0]["meta"]["check"] == "pending"
    assert later[0]["label"].endswith("（20分〜）")
