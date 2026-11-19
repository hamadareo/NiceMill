"""The mapping model (Task 10): fused proposals, the team answer, defaults and default names,
people named on the mapping screen, and the subjects made for automatic names."""
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline import naming
from pfb_app.services import review
from pfb_app.pipeline.tracked_identity import ROSTER_RESET, mapping_is_current
from pfb_app.services.review import (AUTO, OPP, ApprovalError, BusyError, add_named_subject,
                                     confirm_identity, materialise_choices,
                                     save_tracked_identity, set_evaluate_opponents,
                                     set_team_choice)
from pfb_app.services.roster import apply_roster, parse_roster_text
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.common import OUTSIDE, mapping_form, mapping_from_choices

KEY = "pfb3x3:standard"
LABELS = ["P1", "P2", "P3", "P4", "P5"]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=tmp_path / "engine")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def work(repo, rows=(), *, domain="sports", sport="soccer", cameras=("cam1",)):
    aid = repo.create_assignment("試合", Kind.video, "練習試合", "", "standard", True,
                                 domain=domain, sport=sport if domain == "sports" else None)
    sid = repo.create_submission(aid, "G1", {"cameras": {c: f"/{c}.mp4" for c in cameras},
                                             "engine": "pfb3x3", "start_sec": 0.0})
    for pid, name, attrs in rows:
        repo.add_subject(sid, pid, name, None, {}, attrs)
    return sid


ROSTER = [("r1", "山田", {"name": "山田", "number": "10"}),
          ("r2", "佐藤", {"name": "佐藤", "number": "7"}),
          ("r3", "鈴木", {"name": "鈴木", "number": "5"})]


def save(repo, sid, stage, output, camera="cam1", upstream=(), status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=None, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "x", warnings=[],
        duration_sec=0.0, created_at=""))


def track(repo, sid, camera="cam1", labels=LABELS, defaults=None):
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    return save(repo, sid, Stage.track, {"labels": list(labels), "people": len(labels),
                                         "default_names": defaults or {}},
                camera=camera, upstream=[prep.id])


def read(number, agree=3):
    return {"number": number, "votes": agree, "reads": agree, "frames_agree": agree,
            "frames": [], "all": [], "failed": 0, "skipped": {}}


def names_run(repo, sid, tr, *, numbers=None, teams=None, threshold=None, camera="cam1",
              track_run_id=None, fingerprint=None, strips=None, team_reason="", crops=None):
    cues, status = {}, {}
    if numbers is not None:
        cues["number"], status["number"] = numbers, {"status": "ok"}
    if teams is not None:
        cues["team"] = {"teams": teams, "reason": team_reason, "crops": crops or {},
                        "strips": strips or {}, "sizes": {}, "other_dist": None}
        status["team"] = {"status": "ok"}
    return save(repo, sid, Stage.names, {
        "track_run_id": tr.id if track_run_id is None else track_run_id,
        "labels": list(tr.output["labels"]), "teams": dict(teams or {}),
        "team_reason": team_reason if teams is not None else None,
        "crops": dict(crops or {}), "strips": dict(strips or {}), "cues": cues, "status": status,
        "disabled": {}, "gate": {"number": {"enabled": True, "threshold": threshold}},
        "model": "qwen3vl-8b", "roster_floor": 0,
        "roster_fingerprint": fingerprint or naming.roster_fingerprint(repo.list_subjects(sid))},
        camera=camera, upstream=[tr.id])


def names(form, prefill=None):
    """{label: what the select shows} of the form's prefill."""
    return {label: form.name(v) for label, v in (prefill or form.prefill).items()}


def jobs(repo):
    return repo.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


# --- proposals ------------------------------------------------------------------------------

def test_proposals_prefill_and_reasons(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2", "P3"])
    names_run(repo, sid, tr, numbers={"P1": read("10"), "P2": read("07", agree=2),
                                      "P3": read(None, agree=0)})
    form = mapping_form(repo, sid, "cam1")
    assert {k: p.row for k, p in form.proposals.items()} == {"P1": "r1", "P2": "r2"}
    assert form.prefill == {"P1": "r1", "P2": "r2"}       # P3: no proposal, so 未割り当て
    assert form.reasons == {"P1": "背番号10から推定", "P2": "背番号7から推定"}
    assert form.suggested and not form.current and form.saved == {}
    # proposals are only pre-fills: nothing is saved
    assert repo.latest_run(sid, Stage.identity, "cam1") is None
    # the gate's threshold for numbers raises the frames that must agree (max(2, threshold))
    names_run(repo, sid, tr, numbers={"P1": read("10"), "P2": read("7", agree=2)}, threshold=3)
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P1": "r1"} and set(form.reasons) == {"P1"}


def test_an_explicit_min_frames_agree_wins_over_the_threshold():
    gate = {"number": {"enabled": True, "threshold": 2, "min_frames_agree": 4}}
    assert review._min_frames_agree({"gate": gate}) == 4
    gate = {"number": {"enabled": True, "threshold": 3}}          # frames_agree k (Task 9)
    assert review._min_frames_agree({"gate": gate}) == 3
    assert review._min_frames_agree({"gate": {"number": {"threshold": 1}}}) == 2
    assert review._min_frames_agree({}) == 2


def test_rule_defaults_come_after_proposals(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2"], defaults={"P1": "r3", "P2": "r3"})
    names_run(repo, sid, tr, numbers={"P1": read("10")})
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P1": "r1", "P2": "r3"}
    assert form.reasons == {"P1": "背番号10から推定", "P2": "初期位置から推定"}


def test_stale_names_run_ignored(repo):
    sid = work(repo, ROSTER)
    old = track(repo, sid, labels=["P1", "P2"])
    names_run(repo, sid, old, numbers={"P1": read("10")})
    tr = track(repo, sid, labels=["P1", "P2"])          # tracked again: the run is stale
    form = mapping_form(repo, sid, "cam1")
    assert form.proposals == {} and form.prefill == {}
    # a run whose output names another track run is not used either
    names_run(repo, sid, tr, numbers={"P1": read("10")}, track_run_id=old.id)
    assert mapping_form(repo, sid, "cam1").proposals == {}
    # nor one built from other roster rows
    names_run(repo, sid, tr, numbers={"P1": read("10")}, fingerprint="other")
    assert mapping_form(repo, sid, "cam1").proposals == {}
    names_run(repo, sid, tr, numbers={"P1": read("10")})
    assert mapping_form(repo, sid, "cam1").prefill == {"P1": "r1"}


def test_saved_mapping_wins_over_late_names_run(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2", "P3"])
    save_tracked_identity(repo, sid, "cam1", {"P1": "r2", "P2": None, "P3": "r1"})
    names_run(repo, sid, tr, numbers={"P1": read("10"), "P2": read("7")})   # arrives late
    form = mapping_form(repo, sid, "cam1")
    assert form.current and form.prefill == form.saved == {"P1": "r2", "P2": OUTSIDE,
                                                           "P3": "r1"}
    assert form.reasons == {} and not form.suggested


def test_names_pending(repo):
    sid = work(repo, ROSTER)
    track(repo, sid, labels=["P1"])
    assert not mapping_form(repo, sid, "cam1").names_pending
    repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    assert mapping_form(repo, sid, "cam1").names_pending


# --- defaults and default names -------------------------------------------------------------

def test_no_roster_defaults_to_auto_players(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1", "P2", "P10"])
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P1": AUTO + "P1", "P2": AUTO + "P2", "P10": AUTO + "P10"}
    assert names(form) == {"P1": "選手1", "P2": "選手2", "P10": "選手3"}
    assert form.options[-1] == OUTSIDE and AUTO + "P2" in form.options
    assert form.options_for("P2") == [AUTO + "P2", OUTSIDE]
    assert form.index("P10") == form.options.index(AUTO + "P10")
    assert form.reasons == {}          # a default name is not a proposal
    # keeping the defaults gives exactly the names shown
    plain = materialise_choices(repo, sid, "cam1", dict(form.prefill))
    by_id = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    assert {label: by_id[pid] for label, pid in plain.items()} == names(form)
    save_tracked_identity(repo, sid, "cam1", plain)
    form = mapping_form(repo, sid, "cam1")
    assert form.current and names(form) == {"P1": "選手1", "P2": "選手2", "P10": "選手3"}


def test_school_without_people_uses_participant_names(repo):
    sid = work(repo, domain="school")
    track(repo, sid, labels=["P1", "P2"])
    assert names(mapping_form(repo, sid, "cam1")) == {"P1": "参加者1", "P2": "参加者2"}


def test_school_with_people_is_unchanged(repo):
    sid = work(repo, [("s1", "青木", {})], domain="school")
    track(repo, sid, labels=["P1", "P2"], defaults={"P2": "s1"})
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P2": "s1"} and form.options == ["s1", OUTSIDE]
    assert form.suggested


def test_team_answer_sets_other_team_outside(repo):
    sid = work(repo)
    tr = track(repo, sid, labels=["P1", "P2", "P3", "P4"])
    teams = {"P1": "T1", "P2": "T2", "P3": "T1", "P4": "T2"}
    names_run(repo, sid, tr, teams=teams)
    form = mapping_form(repo, sid, "cam1")
    assert form.teams == teams and form.own_team is None
    assert set(form.prefill.values()) == {AUTO + l for l in ["P1", "P2", "P3", "P4"]}
    before = jobs(repo)
    set_team_choice(repo, sid, "cam1", "T1")
    assert jobs(repo) == before                         # an answer never starts any work
    stored = repo.get_submission(sid)["meta"]["own_team"]["cam1"]
    assert stored["labels"] == ["P1", "P3"] and "T1" not in str(stored)   # a label set
    form = mapping_form(repo, sid, "cam1")
    assert form.own_team == "T1"
    assert form.prefill == {"P1": AUTO + "P1", "P2": OUTSIDE, "P3": AUTO + "P3",
                            "P4": OUTSIDE}
    assert names(form)["P3"] == "選手2"
    # the engine may swap T1/T2 in a new run: the answer follows the people
    names_run(repo, sid, tr, teams={"P1": "T2", "P2": "T1", "P3": "T2", "P4": "T1"})
    form = mapping_form(repo, sid, "cam1")
    assert form.own_team == "T2" and form.prefill["P2"] == OUTSIDE
    set_team_choice(repo, sid, "cam1", None)
    assert mapping_form(repo, sid, "cam1").own_team is None
    with pytest.raises(ValueError):
        set_team_choice(repo, sid, "cam1", "other")


def test_team_answer_filters_proposals_to_own_team(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2", "P3"])
    names_run(repo, sid, tr, numbers={"P2": read("10"), "P3": read("7")},
              teams={"P1": "T1", "P2": "T2", "P3": "T1"})
    assert mapping_form(repo, sid, "cam1").prefill == {"P2": "r1", "P3": "r2"}
    set_team_choice(repo, sid, "cam1", "T1")
    form = mapping_form(repo, sid, "cam1")      # the opponent's #10 is not our 山田
    assert form.prefill == {"P2": OUTSIDE, "P3": "r2"}
    assert set(form.proposals) == {"P3"}


def test_evaluate_opponents_names(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid)
    names_run(repo, sid, tr, numbers={"P1": read("10")},
              teams={"P1": "T1", "P2": "T1", "P3": "T2", "P4": "T2", "P5": "other"})
    set_team_choice(repo, sid, "cam1", "T1")
    form = mapping_form(repo, sid, "cam1")
    assert not form.evaluate_opponents
    assert form.prefill == {"P1": "r1", "P3": OUTSIDE, "P4": OUTSIDE, "P5": OUTSIDE}
    set_evaluate_opponents(repo, sid, True)
    form = mapping_form(repo, sid, "cam1")
    assert form.evaluate_opponents
    assert form.prefill == {"P1": "r1", "P2": AUTO + "P2", "P3": OPP + "P3", "P4": OPP + "P4",
                            "P5": OUTSIDE}
    assert names(form) == {"P1": "山田", "P2": "選手（番号なし）1", "P3": "相手選手1",
                           "P4": "相手選手2", "P5": OUTSIDE}
    assert form.options_for("P3") == ["r1", "r2", "r3", AUTO + "P3", OPP + "P3", OUTSIDE]
    mapping, unset = mapping_from_choices(materialise_choices(repo, sid, "cam1",
                                                              dict(form.prefill)))
    assert unset == []
    save_tracked_identity(repo, sid, "cam1", mapping)
    shown = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    assert sorted(shown.values()) == sorted(["山田", "佐藤", "鈴木", "選手（番号なし）1",
                                             "相手選手1", "相手選手2"])
    set_evaluate_opponents(repo, sid, False)
    assert repo.get_submission(sid)["meta"]["evaluate_opponents"] is False


def test_referee_cluster_outside(repo):
    sid = work(repo)
    tr = track(repo, sid, labels=["P1", "P2", "P3"])
    names_run(repo, sid, tr, teams={"P1": "T1", "P2": "T2", "P3": "other"})
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P1": AUTO + "P1", "P2": AUTO + "P2", "P3": OUTSIDE}
    set_evaluate_opponents(repo, sid, True)
    assert mapping_form(repo, sid, "cam1").prefill["P3"] == OUTSIDE


def test_team_strips_and_crops(repo, tmp_path):
    sid = work(repo)
    tr = track(repo, sid, labels=["P1", "P2"])
    strip = tmp_path / "t1.jpg"
    strip.write_bytes(b"x")
    names_run(repo, sid, tr, teams={"P1": "T1", "P2": "T2"},
              strips={"T1": str(strip), "T2": str(tmp_path / "gone.jpg")})
    form = mapping_form(repo, sid, "cam1")
    assert form.strips == {"T1": strip} and form.crops == {} and form.team_reason == ""


# --- people named on the screen and automatic people ----------------------------------------

def test_add_named_subject(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2"])
    run = names_run(repo, sid, tr, numbers={"P1": read("10")})
    assert add_named_subject(repo, sid, " 田中 ") == "n1"
    assert add_named_subject(repo, sid, "高橋") == "n2"
    subjects = {s["participant_id"]: s for s in repo.list_subjects(sid)}
    assert subjects["n1"]["display_name"] == "田中" and subjects["n1"]["roster"] == {}
    # a coach-added person never voids the proposals (Task 9's fingerprint)
    assert naming.current_names_run(repo, sid, "cam1").id == run.id
    form = mapping_form(repo, sid, "cam1")
    assert form.name("n1") == "田中" and "n2" in form.options_for("P2")
    assert form.prefill == {"P1": "r1"}       # named people are not a roster
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": "n2"})
    with pytest.raises(ValueError):
        add_named_subject(repo, sid, "  ")


def test_add_named_subject_rejects_duplicate(repo):
    sid = work(repo, ROSTER)
    with pytest.raises(ValueError):
        add_named_subject(repo, sid, "山田")
    with pytest.raises(ValueError):
        add_named_subject(repo, sid, "佐藤 ")          # trimmed before the comparison
    add_named_subject(repo, sid, "田中")
    with pytest.raises(ValueError):
        add_named_subject(repo, sid, "田中")
    assert len(repo.list_subjects(sid)) == 4


def test_named_people_without_roster_keep_auto_defaults(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1", "P2"])
    add_named_subject(repo, sid, "選手1")      # a typed name that looks like a default
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P1": AUTO + "P1", "P2": AUTO + "P2"}
    assert names(form) == {"P1": "選手2", "P2": "選手3"}


def test_materialise_numbers_continue_across_cameras(repo):
    sid = work(repo, cameras=("cam1", "cam2"))
    track(repo, sid, "cam1", labels=["P1", "P2"])
    track(repo, sid, "cam2", labels=["P1", "P2", "P3"])
    plain = materialise_choices(repo, sid, "cam1", {"P1": AUTO + "P1", "P2": AUTO + "P2"})
    assert all(not v.startswith((AUTO, OPP)) for v in plain.values())
    save_tracked_identity(repo, sid, "cam1", plain)
    form = mapping_form(repo, sid, "cam2")
    assert names(form) == {"P1": "選手3", "P2": "選手4", "P3": "選手5"}
    set_evaluate_opponents(repo, sid, True)
    plain2 = materialise_choices(repo, sid, "cam2", {"P1": AUTO + "P1", "P2": OPP + "P2",
                                                     "P3": OUTSIDE})
    assert plain2["P3"] == OUTSIDE
    by_id = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    assert (by_id[plain2["P1"]], by_id[plain2["P2"]]) == ("選手3", "相手選手1")
    mapping, _ = mapping_from_choices(plain2)
    save_tracked_identity(repo, sid, "cam2", mapping)
    # saving cam1 again without its second automatic person removes that person
    save_tracked_identity(repo, sid, "cam1", {"P1": plain["P1"], "P2": None})
    left = {s["display_name"] for s in repo.list_subjects(sid)}
    assert left == {"選手1", "選手3", "相手選手1"}


def test_save_accepts_automatic_options(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1", "P2"])
    save_tracked_identity(repo, sid, "cam1", {"P1": AUTO + "P1", "P2": None})
    form = mapping_form(repo, sid, "cam1")
    assert names(form) == {"P1": "選手1", "P2": OUTSIDE}
    # a mapping that cannot be saved leaves no automatic person behind
    with pytest.raises(review.ValidationError):
        save_tracked_identity(repo, sid, "cam1", {"P1": AUTO + "P1"})   # P2 is missing
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["選手1"]


def test_unmapped_automatic_person_with_runs_is_kept(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1"])
    save_tracked_identity(repo, sid, "cam1", {"P1": AUTO + "P1"})
    subj = repo.list_subjects(sid)[0]
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subj["id"],
        version=0, source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output={"pfb_status": "generated", "pfb": "x", "evidence_refs": []}, status="ok",
        error=None, warnings=[], duration_sec=0.0, created_at=""))
    save_tracked_identity(repo, sid, "cam1", {"P1": None})
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["選手1"]


# --- names jobs never hold the coach up -----------------------------------------------------

def test_decide_allowed_while_names_job_waiting_gpu(repo):
    sid = work(repo, ROSTER)
    track(repo, sid, labels=["P1", "P2"])
    jid = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='waiting_gpu' WHERE id=?", (jid,))
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": None})
    confirm_identity(repo, sid)
    assert repo.get_submission(sid)["identity_confirmed"]


# --- review round 1 -------------------------------------------------------------------------

def by_label(repo, sid, camera="cam1"):
    """{label: display name} of the camera's saved mapping."""
    shown = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    run = repo.latest_run(sid, Stage.identity, camera)
    return {p["label"]: shown.get(p["participant_id"], OUTSIDE)
            for p in run.output["participants"]}


def test_shown_names_are_the_saved_names(repo):
    sid = work(repo)
    tr = track(repo, sid, labels=["P1", "P2", "P3"])
    names_run(repo, sid, tr, teams={"P1": "T1", "P2": "T2", "P3": "T1"})
    set_team_choice(repo, sid, "cam1", "T1")
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill["P2"] == OUTSIDE
    chosen = {**form.prefill, "P2": AUTO + "P2"}        # the coach switches 対象外 → 選手
    shown = form.names_for(chosen)
    assert [shown[chosen[l]] for l in ["P1", "P2", "P3"]] == ["選手1", "選手2", "選手3"]
    mapping, _ = mapping_from_choices(chosen)
    save_tracked_identity(repo, sid, "cam1", mapping)
    assert by_label(repo, sid) == {l: shown[chosen[l]] for l in ["P1", "P2", "P3"]}


def test_team_reason_and_crops_without_teams(repo, tmp_path):
    sid = work(repo)
    tr = track(repo, sid, labels=["P1", "P2"])
    crop = tmp_path / "p1.jpg"
    crop.write_bytes(b"x")
    names_run(repo, sid, tr, teams={}, team_reason="one_colour", crops={"P1": str(crop)})
    form = mapping_form(repo, sid, "cam1")
    assert form.teams == {} and form.team_reason == "one_colour"
    assert form.crops == {"P1": crop}


def test_referees_are_never_proposed(repo):
    sid = work(repo, ROSTER)
    tr = track(repo, sid, labels=["P1", "P2", "P3"])
    names_run(repo, sid, tr, numbers={"P1": read("7"), "P3": read("10")},
              teams={"P1": "T1", "P2": "T1", "P3": "other"})
    form = mapping_form(repo, sid, "cam1")
    assert form.prefill == {"P1": "r2", "P3": OUTSIDE} and set(form.proposals) == {"P1"}


def test_roster_replacement_keeps_people_made_on_the_screen(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1"])
    kept = add_named_subject(repo, sid, "田中")
    add_named_subject(repo, sid, "山田")
    add_named_subject(repo, sid, "＃10")                 # the same as a roster row's 「#10」
    auto = materialise_choices(repo, sid, "cam1", {"P1": AUTO + "P1"})["P1"]
    roster = parse_roster_text("背番号,名前\n10,\n7,山田\n")
    assert apply_roster(repo, sid, roster, "sports") == 2
    shown = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    assert shown == {kept: "田中", auto: "選手1", "r1": "#10", "r2": "山田"}
    meta = repo.get_submission(sid)["meta"]
    assert meta["added_people"] == {kept: "named", auto: "auto"}
    floor = meta["roster_run_floor"]
    # the same roster again changes nothing, whatever people were made on the screen
    add_named_subject(repo, sid, "高橋")
    assert apply_roster(repo, sid, roster, "sports") == 2
    assert repo.get_submission(sid)["meta"]["roster_run_floor"] == floor
    assert len(repo.list_subjects(sid)) == 5


def test_unmapped_warning_names_only_the_roster(repo):
    sid = work(repo, ROSTER)
    track(repo, sid, labels=["P1", "P2"])
    add_named_subject(repo, sid, "田中")
    run = save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": None})
    assert run.warnings == ["人物が割り当てられていない選手がいます（行動の記録がないため根拠不足になります）: "
                            "佐藤、鈴木"]


def test_auto_option_without_a_plan_and_opponents_off(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1"])
    form = mapping_form(repo, sid, "cam1")
    assert form.name(AUTO + "P99") == "新しい選手"
    school = work(repo, domain="school")
    track(repo, school, labels=["P1"])
    assert mapping_form(repo, school, "cam1").name(OPP + "P99") == "新しい参加者"
    with pytest.raises(review.ValidationError, match="相手チームも評価する"):
        save_tracked_identity(repo, sid, "cam1", {"P1": OPP + "P1"})
    assert repo.list_subjects(sid) == []
    set_evaluate_opponents(repo, sid, True)
    save_tracked_identity(repo, sid, "cam1", {"P1": OPP + "P1"})
    assert by_label(repo, sid) == {"P1": "相手選手1"}


def audits(repo, action):
    return repo.conn.execute("SELECT COUNT(*) FROM audit_log WHERE action=?",
                             (action,)).fetchone()[0]


def test_failed_save_cleanup_never_hides_the_error(repo, monkeypatch):
    sid = work(repo)
    track(repo, sid, labels=["P1", "P2"])

    def broken(*args):
        raise RuntimeError("disk")
    monkeypatch.setattr(review, "_remove_added", broken)
    with pytest.raises(review.ValidationError):
        save_tracked_identity(repo, sid, "cam1", {"P1": AUTO + "P1"})   # P2 is missing
    assert audits(repo, "save_tracked_identity") == 0
    assert audits(repo, "materialise_choices") == 0


def test_add_named_subject_waits_for_jobs(repo):
    sid = work(repo)
    repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    add_named_subject(repo, sid, "田中")                 # a names job never holds it up
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    with pytest.raises(BusyError):
        add_named_subject(repo, sid, "高橋")


def test_confirm_removes_named_people_nobody_uses(repo):
    sid = work(repo, cameras=("cam1", "cam2"))
    track(repo, sid, "cam1", labels=["P1"])
    track(repo, sid, "cam2", labels=["P1"])
    used = add_named_subject(repo, sid, "田中")
    add_named_subject(repo, sid, "高橋")
    save_tracked_identity(repo, sid, "cam1", {"P1": used})
    save_tracked_identity(repo, sid, "cam2", {"P1": None})
    result = confirm_identity(repo, sid)
    assert result.removed_named == 1 and len(result) == 2     # behavior for both cameras
    assert [s["display_name"] for s in repo.list_subjects(sid)] == ["田中"]
    assert repo.get_submission(sid)["meta"]["added_people"] == {used: "named"}


# --- final review: a mapping saved before a roster change is not current ----------------------

def test_mapping_saved_before_a_roster_change_cannot_be_confirmed(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1", "P2"])
    apply_roster(repo, sid, parse_roster_text("名前\n山田\n佐藤\n"), "sports")
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1", "P2": "r2"})
    assert mapping_is_current(repo, sid, "cam1")
    apply_roster(repo, sid, parse_roster_text("名前\n高橋\n鈴木\n"), "sports")
    assert not mapping_is_current(repo, sid, "cam1")
    assert not mapping_form(repo, sid, "cam1").current
    with pytest.raises(ApprovalError) as e:
        confirm_identity(repo, sid)
    assert ROSTER_RESET.rstrip("。") in str(e.value) and "cam1" in str(e.value)
    assert not repo.get_submission(sid)["identity_confirmed"]


def test_roster_ids_are_never_reused(repo):
    sid = work(repo)
    track(repo, sid, labels=["P1"])
    apply_roster(repo, sid, parse_roster_text("名前\n山田\n佐藤\n"), "sports")
    assert {s["participant_id"] for s in repo.list_subjects(sid)} == {"r1", "r2"}
    apply_roster(repo, sid, parse_roster_text("名前\n高橋\n"), "sports")
    assert {s["participant_id"]: s["display_name"]
            for s in repo.list_subjects(sid)} == {"r3": "高橋"}
    apply_roster(repo, sid, parse_roster_text("名前\n山田\n佐藤\n"), "sports")
    assert {s["participant_id"] for s in repo.list_subjects(sid)} == {"r4", "r5"}
    assert repo.get_submission(sid)["meta"]["roster_id_max"] == 5
    # the same roster again is no change (the ids are not compared)
    floor = repo.get_submission(sid)["meta"]["roster_run_floor"]
    assert apply_roster(repo, sid, parse_roster_text("名前\n山田\n佐藤\n"), "sports") == 2
    assert {s["participant_id"] for s in repo.list_subjects(sid)} == {"r4", "r5"}
    assert repo.get_submission(sid)["meta"]["roster_run_floor"] == floor


def test_confirm_cancels_the_pending_names_jobs_of_every_camera(repo):
    repo.settings = repo.settings.model_copy(update={"number_names": False})
    sid = work(repo, ROSTER, cameras=("cam1", "cam2"))
    track(repo, sid, "cam1", labels=["P1"])
    track(repo, sid, "cam2", labels=["P1"])
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1"})
    save_tracked_identity(repo, sid, "cam2", {"P1": "r2"})
    waiting = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='waiting_gpu' WHERE id=?", (waiting,))
    queued = repo.enqueue(sid, Stage.names, "names:cpu", camera="cam2")
    running = repo.enqueue(sid, Stage.names, KEY, camera="cam2")
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (running,))
    confirm_identity(repo, sid)
    left = {r[0] for r in repo.conn.execute("SELECT id FROM jobs WHERE stage='names'")}
    assert left == {running}                              # a running one is left alone
    assert waiting not in left and queued not in left


def test_names_jobs_that_read_numbers_for_names_are_kept(repo):
    sid = work(repo, ROSTER, cameras=("cam1", "cam2"))     # sports: numbers name people
    track(repo, sid, "cam1", labels=["P1"])
    track(repo, sid, "cam2", labels=["P1"])
    waiting = repo.enqueue(sid, Stage.names, KEY, camera="cam1")
    save_tracked_identity(repo, sid, "cam1", {"P1": "r1"})
    save_tracked_identity(repo, sid, "cam2", {"P1": "r2"})
    confirm_identity(repo, sid)
    left = {r[0] for r in repo.conn.execute("SELECT id FROM jobs WHERE stage='names'")}
    assert left == {waiting}
