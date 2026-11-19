"""Automatic people on a game: only the people who take part are evaluated. Spectators, the
bench and officials (off the court, or standing in one place all game) are not mapped
(試合に参加していない人), unsure people are evaluated with 要確認, and the coach can include a
person who was left out."""
import json

import numpy as np

from pfb_app.domain.models import Stage
from pfb_app.pipeline.tracked_identity import label_mapping
from pfb_app.services import auto_people
from test_auto_people_service import repo, save, work  # noqa: F401 - fixture

FPS = 10.0


def _rows(n: int, points, frames_each=30, start=1):
    """MOT rows of label P{n} whose box bottom centre walks through `points` (court cm with
    the identity homography)."""
    out, f = [], start
    for x, y in points:
        for _ in range(frames_each):
            out.append(f"{f},{n},{x - 20:.1f},{y - 180:.1f},40.0,180.0,0.9")
            f += 1
    return out


PLAYER = [(200, 200), (700, 500), (1200, 900), (400, 800)]
PLAYER2 = [(1300, 300), (800, 900), (300, 600), (900, 200)]
SPECTATOR = [(-650, 0), (-640, 20), (-655, 10), (-650, 5)]
HALF = [(-500, 300), (-500, 900), (-480, 600), (300, 300), (900, 800)]   # 40 % on the court


def game_track(repo, tmp_path, sid, people: dict[str, list], *, geometry=True, camera="cam1"):
    d = tmp_path / camera
    d.mkdir()
    lines = []
    for label, pts in people.items():
        lines += _rows(int(label[1:]), pts)
    (d / "boxes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"fps": FPS, "n_frames": 120, "width": 1920,
                                             "height": 1080}))
    if geometry:
        (d / "court").mkdir()
        (d / "court" / "geometry.json").write_text(json.dumps(
            {"H": np.eye(3).tolist(), "lens": None, "source": "autocourt"}))
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    labels = list(people)
    out = {"people": len(labels), "labels": labels, "video_id": f"s{sid}_{camera}",
           "boxes": str(d / "boxes.txt"), "default_names": {}, "default_source": None,
           "quality": {lb: {"coverage": 0.9, "mean_h_frac": 0.2} for lb in labels}}
    return save(repo, sid, Stage.track, out, camera=camera, upstream=[prep.id])


def mapped(repo, sid, camera="cam1"):
    return {lb for lb, pid in label_mapping(repo.latest_run(sid, Stage.identity, camera)
                                            .output).items() if pid}


def test_spectators_are_not_evaluated(repo, tmp_path):
    sid = work(repo, domain="sports")
    t = game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P3"}
    assert len(repo.list_subjects(sid)) == 2                  # nobody made for the spectator
    part = auto_people.participation_view(repo, sid)
    assert [(p.camera, p.label, p.track_run_id) for p in part.not_playing] == [("cam1", "P2", t.id)]
    assert part.check == {}


def test_unsure_people_are_evaluated_and_marked(repo, tmp_path):
    sid = work(repo, domain="sports")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": HALF, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P2", "P3"}
    part = auto_people.participation_view(repo, sid)
    pid = label_mapping(repo.latest_run(sid, Stage.identity, "cam1").output)["P2"]
    subject = next(s for s in repo.list_subjects(sid) if s["participant_id"] == pid)
    assert set(part.check) == {subject["id"]}
    assert part.not_playing == []


def test_without_geometry_nobody_is_dropped(repo, tmp_path):
    sid = work(repo, domain="sports")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR}, geometry=False)
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P2"}
    assert auto_people.participation_view(repo, sid).not_playing == []


def test_the_rule_is_for_games_only(repo, tmp_path):
    sid = work(repo)                       # a seated group: everyone sits still
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P2", "P3"}


def test_including_a_person_left_out(repo, tmp_path):
    sid = work(repo, domain="sports")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P3"}
    auto_people.include_person(repo, repo.settings, sid, "cam1", "P2")
    assert mapped(repo, sid) == {"P1", "P2", "P3"}
    assert auto_people.participation_view(repo, sid).not_playing == []
    # choosing the people again (same track run) keeps the coach's choice
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P2", "P3"}


def test_including_needs_a_left_out_label(repo, tmp_path):
    import pytest
    sid = work(repo, domain="sports")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    with pytest.raises(ValueError):
        auto_people.include_person(repo, repo.settings, sid, "cam1", "P1")


def test_a_new_track_run_forgets_old_inclusions(repo, tmp_path):
    sid = work(repo, domain="sports")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    auto_people.include_person(repo, repo.settings, sid, "cam1", "P2")
    (tmp_path / "cam1").rename(tmp_path / "old")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P3"}


def test_the_spectators_events_never_reach_anyone(repo, tmp_path):
    from pfb_app.pipeline.pfb3x3_bridge import bridge_behavior
    sid = work(repo, domain="sports")
    game_track(repo, tmp_path, sid, {"P1": PLAYER, "P2": SPECTATOR, "P3": PLAYER2})
    auto_people.auto_identity(repo, sid, "cam1")
    mapping = label_mapping(repo.latest_run(sid, Stage.identity, "cam1").output)
    rows = [{"window_id": "s1_cam1_w0000", "status": "ok", "events": [
        {"label": "P2", "clock": "0:01", "observed_action": "P2 jumps and shoots",
         "confidence": "high"},
        {"label": "P1", "clock": "0:02", "observed_action": "P1 passes", "confidence": "high"}]}]
    out = bridge_behavior(rows, mapping, "cam1", 0.0, False, tmp_path)
    assert [e["participant"] for e in out["events"]] == [mapping["P1"]]
