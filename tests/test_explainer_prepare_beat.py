"""The explainer's preparation of every part (_prepare: frames read, the AI region checked)
runs before the first scene is drawn. It must beat the job's heartbeat and honour a stop
request part by part, or a long preparation looks dead to the worker's supervisor
(explainer.HUNG_SEC) and is stopped and failed (rel5 finding, fixed in rel7). Synthetic only."""
import pytest

from pfb_app.export import explainer_video as ev
from test_explainer_video import FPS, SH, SW, make_video, write_tracks, load_boxes


@pytest.fixture
def scene(tmp_path):
    video = make_video(tmp_path / "v.mp4", 4, 30)
    boxes = load_boxes(write_tracks(tmp_path / "tracks", 4, 30, False))
    return tmp_path, video, boxes


def plan_of(video, boxes, n):
    src = ev.Source(video, FPS, (SW, SH), boxes, 0.0, 300)
    parts = [ev.Part(f"開始から 00:0{k}", None, "a", "b", src, float(k), ["P1"], [])
             for k in range(1, n + 1)]
    return ev.PersonPlan("x", "t", "", "", "ほかの人", parts)


def test_every_part_is_prepared_under_a_beat_and_a_progress_note(scene, monkeypatch):
    tmp_path, video, boxes = scene
    order = []
    monkeypatch.setattr(ev, "_prepare", lambda part: order.append("prepare") or None)
    notes = []

    def stop():
        order.append("stop")
        return False

    ev.render_person(plan_of(video, boxes, 3), tmp_path / "o.mp4", stop=stop,
                     progress=lambda f, note: notes.append((f, note)))
    first_scene = order.index("prepare")
    assert order[:first_scene * 2 or 1].count("stop") >= 1
    # a beat (stop) comes before each of the three preparations
    prepares = [i for i, o in enumerate(order) if o == "prepare"]
    assert len(prepares) == 3
    assert all(order[i - 1] == "stop" for i in prepares)
    assert [n for _, n in notes[:3]] == ["場面を確かめています 1 / 3", "場面を確かめています 2 / 3",
                                         "場面を確かめています 3 / 3"]
    assert all(0.0 <= f <= 1.0 for f, _ in notes)


def test_a_stop_during_the_preparation_ends_it_at_once(scene, monkeypatch):
    tmp_path, video, boxes = scene
    calls = []
    monkeypatch.setattr(ev, "_prepare", lambda part: calls.append(part.when) or None)
    beats = iter([False, True])         # the second part's turn: stopped
    with pytest.raises(ev.Cancelled):
        ev.render_person(plan_of(video, boxes, 3), tmp_path / "o.mp4",
                         stop=lambda: next(beats, True))
    assert len(calls) == 1               # the other parts were not prepared
    assert not (tmp_path / "o.mp4").exists()


def test_a_part_without_a_source_is_skipped_without_a_beat_needed(scene):
    tmp_path, video, boxes = scene
    part = ev.Part("開始から 00:08", None, "a", "b", None, None, [], [])
    beats = []
    ev.render_person(ev.PersonPlan("x", "t", "", "", "ほかの人", [part]), tmp_path / "o.mp4",
                     stop=lambda: beats.append(1) or False)
    assert (tmp_path / "o.mp4").exists()
