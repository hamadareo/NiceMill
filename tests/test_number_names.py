"""Names from read numbers: automatic people become 「#10」 / 「10号車」 after a names run
(synthetic data only; the coach's own names always win)."""
from types import SimpleNamespace

from pfb_app.domain.models import Stage
from pfb_app.pipeline import runner
from pfb_app.pipeline.tracked_identity import label_mapping
from pfb_app.services import auto_people, review
from pfb_app.ui.common import mapping_form
from test_mapping_model import (ROSTER, names_run, read, repo, save, settings,  # noqa: F401
                                track, work)


def auto_work(repo, labels=("P1", "P2", "P3", "P4"), *, sport="soccer", cameras=("cam1",)):
    sid = work(repo, sport=sport, cameras=cameras)
    trs = {}
    for cam in cameras:
        trs[cam] = track(repo, sid, camera=cam, labels=list(labels))
        auto_people.auto_identity(repo, sid, cam)
    return sid, trs


def by_label(repo, sid, camera="cam1"):
    names = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    mapping = label_mapping(repo.latest_run(sid, Stage.identity, camera).output)
    return {lb: names[pid] for lb, pid in mapping.items() if pid}


def sources(repo, sid):
    return dict(repo.get_submission(sid)["meta"].get(auto_people.NAME_SOURCE) or {})


def test_auto_people_get_their_read_numbers(repo):
    sid, trs = auto_work(repo)
    assert by_label(repo, sid) == {"P1": "選手1", "P2": "選手2", "P3": "選手3", "P4": "選手4"}
    names_run(repo, sid, trs["cam1"], numbers={
        "P1": read("10"), "P2": read("00", agree=2), "P3": read("7", agree=1),
        "P4": read(None, agree=0)})
    assert auto_people.number_names(repo, sid) == 2
    assert by_label(repo, sid) == {"P1": "#10", "P2": "#00", "P3": "選手3", "P4": "選手4"}
    assert sources(repo, sid) == {"auto1": "number", "auto2": "number"}
    assert auto_people.number_names(repo, sid) == 0               # nothing new


def test_zero_and_double_zero_are_different_numbers(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("0"), "P2": read("00")})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid) == {"P1": "#0", "P2": "#00"}


def test_the_same_number_on_both_teams_names_the_team(repo):
    sid, trs = auto_work(repo, ("P1", "P2", "P3"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("10"),
                                               "P3": read("4")},
              teams={"P1": "T1", "P2": "T2", "P3": "T1"})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid) == {"P1": "#10（チーム1）", "P2": "#10（チーム2）", "P3": "#4"}


def test_an_unclear_repeat_keeps_the_default_names(repo):
    sid, trs = auto_work(repo, ("P1", "P2", "P3"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("10"),
                                               "P3": read("4")})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid) == {"P1": "選手1", "P2": "選手2", "P3": "#4"}


def test_cars_are_named_by_car_number(repo):
    sid, trs = auto_work(repo, ("P1",), sport="motor_racing")
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid) == {"P1": "10号車"}


def test_the_coach_name_always_wins(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    subjects = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    auto_people.rename_person(repo, subjects["auto1"], "山田")       # before the numbers
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("7")})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid) == {"P1": "山田", "P2": "#7"}
    auto_people.rename_person(repo, subjects["auto2"], "佐藤")       # after them
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("8")})
    assert auto_people.number_names(repo, sid) == 0
    assert by_label(repo, sid) == {"P1": "山田", "P2": "佐藤"}
    assert sources(repo, sid) == {"auto1": "coach", "auto2": "coach"}


def test_a_name_set_before_sources_were_kept_counts_as_the_coach(repo):
    sid, trs = auto_work(repo, ("P1",))
    repo.rename_subject(repo.list_subjects(sid)[0]["id"], "鈴木")
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    assert auto_people.number_names(repo, sid) == 0
    assert by_label(repo, sid) == {"P1": "鈴木"}


def test_a_newer_read_updates_a_number_name(repo):
    sid, trs = auto_work(repo, ("P1",))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    auto_people.number_names(repo, sid)
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("11")})
    assert auto_people.number_names(repo, sid) == 1
    assert by_label(repo, sid) == {"P1": "#11"}


def test_a_name_someone_else_has_is_not_taken(repo):
    sid, trs = auto_work(repo, ("P1",))
    review.add_named_subject(repo, sid, "#10")
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    assert auto_people.number_names(repo, sid) == 0
    assert by_label(repo, sid) == {"P1": "選手1"}


def test_roster_people_are_never_renamed(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1"])
    review.save_tracked_identity(repo, sid, "cam1", {"P1": "r1"})
    names_run(repo, sid, tr, numbers={"P1": read("10")})
    assert auto_people.number_names(repo, sid) == 0
    assert {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}["r1"] == "山田"


def test_a_school_work_is_left_alone(repo):
    sid = work(repo, domain="school")
    tr = track(repo, sid, labels=["P1"])
    auto_people.auto_identity(repo, sid, "cam1")
    names_run(repo, sid, tr, numbers={"P1": read("10")})
    assert auto_people.number_names(repo, sid) == 0


def test_a_finished_names_job_names_the_people(repo):
    sid, trs = auto_work(repo, ("P1",))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    job = SimpleNamespace(submission_id=sid, stage=Stage.names, camera="cam1", subject_id=None)
    assert runner.enqueue_followups(repo, job) == []
    assert by_label(repo, sid) == {"P1": "#10"}


def test_a_failing_number_naming_never_fails_the_names_job(repo, monkeypatch, caplog):
    sid, trs = auto_work(repo, ("P1",))

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(auto_people, "number_names", boom)
    job = SimpleNamespace(submission_id=sid, stage=Stage.names, camera="cam1", subject_id=None)
    assert runner.enqueue_followups(repo, job) == []
    assert "boom" in caplog.text


def test_removed_people_lose_their_name_source(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("7")})
    auto_people.number_names(repo, sid)
    review._remove_added(repo, sid, ["auto2"])
    assert sources(repo, sid) == {"auto1": "number"}


def test_number_notes_name_the_source(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    auto_people.number_names(repo, sid)
    ids = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    assert auto_people.name_notes(repo, sid) == {ids["auto1"]: "背番号から"}
    car, trs = auto_work(repo, ("P1",), sport="motor_racing")
    names_run(repo, car, trs["cam1"], numbers={"P1": read("3")})
    auto_people.number_names(repo, car)
    assert list(auto_people.name_notes(repo, car).values()) == ["番号から"]


def test_a_disabled_number_gate_proposes_no_roster_rows(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1"])
    run = names_run(repo, sid, tr, numbers={"P1": read("10")})
    out = {**run.output, "gate": {"number": {"enabled": False, "threshold": None}}}
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?",
                      (__import__("json").dumps(out), run.id))
    form = mapping_form(repo, sid, "cam1")
    assert form.proposals == {} and form.reasons == {}


# --- several cameras: one name per camera's person, the lowest camera keeps the plain one ---

def test_the_same_number_on_two_cameras_names_the_camera(repo):
    sid, trs = auto_work(repo, ("P1",), cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        names_run(repo, sid, trs[cam], numbers={"P1": read("10")}, camera=cam)
    assert auto_people.number_names(repo, sid) == 2
    assert by_label(repo, sid, "cam1") == {"P1": "#10"}
    assert by_label(repo, sid, "cam2") == {"P1": "#10（カメラ2）"}
    assert auto_people.number_names(repo, sid) == 0               # stable


def test_team_and_camera_go_in_one_bracket(repo):
    sid, trs = auto_work(repo, ("P1", "P2"), cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        names_run(repo, sid, trs[cam], numbers={"P1": read("10"), "P2": read("10")},
                  teams={"P1": "T1", "P2": "T2"}, camera=cam)
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid, "cam1") == {"P1": "#10（チーム1）", "P2": "#10（チーム2）"}
    assert by_label(repo, sid, "cam2") == {"P1": "#10（チーム1・カメラ2）",
                                           "P2": "#10（チーム2・カメラ2）"}


def test_cars_on_two_cameras(repo):
    sid, trs = auto_work(repo, ("P1",), sport="motor_racing", cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        names_run(repo, sid, trs[cam], numbers={"P1": read("3")}, camera=cam)
    auto_people.number_names(repo, sid)
    assert (by_label(repo, sid, "cam1"), by_label(repo, sid, "cam2")) == (
        {"P1": "3号車"}, {"P1": "3号車（カメラ2）"})


def test_a_freed_number_goes_to_the_next_camera(repo):
    # cam2's person is made first (auto1), so it is renamed before cam1's person (auto2)
    sid = work(repo, cameras=("cam1", "cam2"))
    trs = {cam: track(repo, sid, camera=cam, labels=["P1"]) for cam in ("cam1", "cam2")}
    auto_people.auto_identity(repo, sid, "cam2")
    auto_people.auto_identity(repo, sid, "cam1")
    for cam in ("cam1", "cam2"):
        names_run(repo, sid, trs[cam], numbers={"P1": read("10")}, camera=cam)
    auto_people.number_names(repo, sid)
    assert (by_label(repo, sid, "cam1"), by_label(repo, sid, "cam2")) == (
        {"P1": "#10"}, {"P1": "#10（カメラ2）"})
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("11")})   # cam1 no longer reads 10
    assert auto_people.number_names(repo, sid) == 2
    assert (by_label(repo, sid, "cam1"), by_label(repo, sid, "cam2")) == (
        {"P1": "#11"}, {"P1": "#10"})


def test_one_person_mapped_on_two_cameras_gets_one_name(repo):
    sid, trs = auto_work(repo, ("P1",), cameras=("cam1", "cam2", "cam3"))
    review.save_tracked_identity(repo, sid, "cam2", {"P1": "auto1"})   # the cam1 person
    for cam in ("cam1", "cam2", "cam3"):
        names_run(repo, sid, trs[cam], numbers={"P1": read("10")}, camera=cam)
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid, "cam1") == by_label(repo, sid, "cam2") == {"P1": "#10"}
    assert by_label(repo, sid, "cam3") == {"P1": "#10（カメラ3）"}


def test_an_unclear_repeat_in_one_camera_stays_in_that_camera(repo):
    sid, trs = auto_work(repo, ("P1", "P2"), cameras=("cam1", "cam2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("10")})
    names_run(repo, sid, trs["cam2"], numbers={"P1": read("10"), "P2": read("4")},
              camera="cam2")
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid, "cam1") == {"P1": "選手1", "P2": "選手2"}
    assert by_label(repo, sid, "cam2") == {"P1": "#10", "P2": "#4"}


# --- after the coach saves a mapping ---

def test_a_person_made_on_the_mapping_screen_gets_the_read_number(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2"])
    names_run(repo, sid, tr, numbers={"P1": read("10"), "P2": read("8")})   # names done
    review.save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": review.AUTO + "P2"})
    assert by_label(repo, sid) == {"P1": "山田", "P2": "#8"}


def test_a_failing_number_naming_never_fails_the_save(repo, monkeypatch, caplog):
    sid = work(repo, ROSTER)
    track(repo, sid, labels=["P1"])

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(auto_people, "number_names", boom)
    assert review.save_tracked_identity(repo, sid, "cam1", {"P1": "r1"}) is not None
    assert "boom" in caplog.text


def test_a_failing_number_naming_never_fails_reselect(repo, monkeypatch, caplog):
    sid, trs = auto_work(repo, ("P1",))

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(auto_people, "number_names", boom)
    monkeypatch.setattr(auto_people, "Pfb3x3Engine",
                        lambda settings: SimpleNamespace(quality=lambda *a, **k: None))
    assert isinstance(auto_people.reselect(repo, repo.settings, sid), list)
    assert "boom" in caplog.text


def test_a_name_changed_outside_rename_person_is_kept(repo):
    sid, trs = auto_work(repo, ("P1",))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    auto_people.number_names(repo, sid)
    repo.rename_subject(repo.list_subjects(sid)[0]["id"], "山田")   # not via rename_person
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("11")})
    assert auto_people.number_names(repo, sid) == 0
    assert by_label(repo, sid) == {"P1": "山田"}


# --- swaps, rotations and a plain name someone keeps ---

def test_two_people_swap_their_numbers(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("11")})
    auto_people.number_names(repo, sid)
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("11"), "P2": read("10")})
    assert auto_people.number_names(repo, sid) == 2
    assert by_label(repo, sid) == {"P1": "#11", "P2": "#10"}


def test_three_people_rotate_their_numbers(repo):
    sid, trs = auto_work(repo, ("P1", "P2", "P3"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("11"),
                                               "P3": read("12")})
    auto_people.number_names(repo, sid)
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("11"), "P2": read("12"),
                                               "P3": read("10")})
    assert auto_people.number_names(repo, sid) == 3
    assert by_label(repo, sid) == {"P1": "#11", "P2": "#12", "P3": "#10"}


def test_a_swap_waits_behind_a_name_someone_keeps(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10"), "P2": read("11")})
    auto_people.number_names(repo, sid)
    review.add_named_subject(repo, sid, "#12")
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("12"), "P2": read("10")})
    assert auto_people.number_names(repo, sid) == 0    # P1 cannot move, so P2 cannot either
    assert by_label(repo, sid) == {"P1": "#10", "P2": "#11"}


def test_a_coach_typed_plain_name_sends_the_next_camera_to_its_own_name(repo):
    sid, trs = auto_work(repo, ("P1",), cameras=("cam1", "cam2"))
    ids = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    auto_people.rename_person(repo, ids["auto1"], "#10")             # the cam1 person
    for cam in ("cam1", "cam2"):
        names_run(repo, sid, trs[cam], numbers={"P1": read("10")}, camera=cam)
    auto_people.number_names(repo, sid)
    assert (by_label(repo, sid, "cam1"), by_label(repo, sid, "cam2")) == (
        {"P1": "#10"}, {"P1": "#10（カメラ2）"})


def test_a_kept_plain_name_sends_the_next_camera_to_its_own_name(repo):
    sid, trs = auto_work(repo, ("P1",), cameras=("cam1", "cam2"))
    names_run(repo, sid, trs["cam1"], numbers={"P1": read("10")})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid, "cam1") == {"P1": "#10"}
    names_run(repo, sid, trs["cam1"], numbers={"P1": read(None, agree=0)})   # no read now
    names_run(repo, sid, trs["cam2"], numbers={"P1": read("10")}, camera="cam2")
    assert auto_people.number_names(repo, sid) == 1
    assert (by_label(repo, sid, "cam1"), by_label(repo, sid, "cam2")) == (
        {"P1": "#10"}, {"P1": "#10（カメラ2）"})
    assert auto_people.number_names(repo, sid) == 0               # stable


def test_a_kept_plain_name_on_the_same_first_camera_keeps_the_default(repo):
    sid, trs = auto_work(repo, ("P1", "P2"))
    ids = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    auto_people.rename_person(repo, ids["auto1"], "#10")
    names_run(repo, sid, trs["cam1"], numbers={"P2": read("10")})
    assert auto_people.number_names(repo, sid) == 0
    assert by_label(repo, sid) == {"P1": "#10", "P2": "選手2"}


def test_the_number_timeline_names_by_the_dominant_number(repo):
    """With a number timeline, the number the person wears most of the time names them (work
    6: the 3 sampled reads of 選手2 were 3, 3, 1, but #1 for 70 % of the track); a label whose
    number keeps changing (no number for half of it) gets no name."""
    sid, trs = auto_work(repo, ("P1", "P2", "P3"))
    tl = lambda dom, share: {"timeline_s": 6.0, "segments": [[1, 50, "x", 2]],  # noqa: E731
                             "dominant": dom, "dominant_share": share}
    names_run(repo, sid, trs["cam1"], numbers={
        "P1": {**read(None, agree=1), **tl("1", 0.70)},
        "P2": {**read("9", agree=2), **tl("6", 0.26)},
        "P3": {**read("5"), **tl("5", 0.64)}})
    auto_people.number_names(repo, sid)
    assert by_label(repo, sid) == {"P1": "#1", "P2": "選手2", "P3": "#5"}
