"""People nobody evaluates because they were on screen too briefly: the selection bar of a long
video, the list (short_people) and 「この人も評価する」 for them."""
import json

from pfb_app.domain.models import Stage
from pfb_app.pipeline.tracked_identity import label_mapping
from pfb_app.services import auto_people
from test_auto_people_service import repo, save, track, work  # noqa: F401 - fixture

LABELS = [f"P{n}" for n in range(1, 9)]
FPS = 10.0
# a long game with cuts away from the court (2026-11-14 trial, 30 min): nobody is boxed half
# of the time; P6..P8 are on screen only briefly
COVERAGE = {"P1": 0.34, "P2": 0.31, "P3": 0.27, "P4": 0.22, "P5": 0.18,
            "P6": 0.06, "P7": 0.04, "P8": 0.02}


def quality(coverage=COVERAGE, frames=3000):
    return {lb: {"coverage": c, "mean_h_frac": 0.15, "frames": round(c * frames)}
            for lb, c in coverage.items()}


def long_game(repo, tmp_path, *, domain="school", coverage=COVERAGE):
    sid = work(repo, domain=domain)
    (tmp_path / "meta.json").write_text(json.dumps({"fps": FPS, "n_frames": 3000,
                                                    "width": 1280, "height": 720}))
    run = track(repo, tmp_path, sid, labels=LABELS, quality=quality(coverage))
    return sid, run


def mapped(repo, sid):
    out = repo.latest_run(sid, Stage.identity, "cam1").output
    return {lb for lb, pid in label_mapping(out).items() if pid}


def test_a_long_video_evaluates_everybody_who_was_on_screen_for_minutes(repo, tmp_path):
    sid, _ = long_game(repo, tmp_path)
    auto_people.auto_identity(repo, sid, "cam1")
    assert mapped(repo, sid) == {"P1", "P2", "P3", "P4", "P5"}      # bar 0.17, not 0.25


def test_the_short_ones_are_listed_with_how_long_they_were_boxed(repo, tmp_path):
    sid, run = long_game(repo, tmp_path)
    auto_people.auto_identity(repo, sid, "cam1")
    view = auto_people.short_people(repo, sid)
    assert [(p.label, p.seconds, p.track_run_id) for p in view.people] == [
        ("P6", 18.0, run.id), ("P7", 12.0, run.id), ("P8", 6.0, run.id)]
    assert (view.found, view.evaluated, view.much) == (8, 5, True)


def test_one_missing_person_is_not_worth_a_sentence(repo, tmp_path):
    cov = {**COVERAGE, "P6": 0.2, "P7": 0.2, "P8": 0.02}
    sid, _ = long_game(repo, tmp_path, coverage=cov)
    auto_people.auto_identity(repo, sid, "cam1")
    view = auto_people.short_people(repo, sid)
    assert [p.label for p in view.people] == ["P8"] and not view.much


def test_including_a_short_person_evaluates_them_and_survives_choosing_again(repo, tmp_path):
    sid, _ = long_game(repo, tmp_path)
    auto_people.auto_identity(repo, sid, "cam1")
    auto_people.include_person(repo, repo.settings, sid, "cam1", "P7")
    assert mapped(repo, sid) == {"P1", "P2", "P3", "P4", "P5", "P7"}
    assert [p.label for p in auto_people.short_people(repo, sid).people] == ["P6", "P8"]
    auto_people.auto_identity(repo, sid, "cam1")                    # the same track run
    assert "P7" in mapped(repo, sid)


def test_including_refuses_somebody_already_evaluated_or_unknown(repo, tmp_path):
    import pytest
    sid, _ = long_game(repo, tmp_path)
    auto_people.auto_identity(repo, sid, "cam1")
    for label in ("P1", "P99"):
        with pytest.raises(ValueError):
            auto_people.include_person(repo, repo.settings, sid, "cam1", label)


def test_a_mapping_the_coach_made_leaves_nobody_in_the_list(repo, tmp_path):
    from pfb_app.services import review
    sid, _ = long_game(repo, tmp_path)
    auto_people.auto_identity(repo, sid, "cam1")
    pid = repo.list_subjects(sid)[0]["participant_id"]
    mapping = {lb: None for lb in LABELS} | {"P1": pid}
    review.save_tracked_identity(repo, sid, "cam1", mapping)       # the coach decides
    assert auto_people.short_people(repo, sid).people == []
