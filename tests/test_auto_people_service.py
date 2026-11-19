"""Automatic people: the auto identity after track, its confirmation and 「映っている人を自動で選び直す」."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline import runner
from pfb_app.pipeline.tracked_identity import label_mapping
from pfb_app.services import auto_people, review
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema

LABELS = [f"P{n}" for n in range(1, 48)]   # the work-1 shape: 47 tracked labels


@pytest.fixture
def repo(tmp_path):
    settings = Settings(data_dir=tmp_path / "data",
                        prompts_dir=Path(__file__).resolve().parents[1] / "prompts")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), source="model"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def work(repo, *, domain="school", cameras=("cam1",), roster=(), **meta):
    aid = repo.create_assignment("課題", Kind.video, "練習", "", "standard", False,
                                 domain=domain, sport="basketball" if domain == "sports" else None)
    sid = repo.create_submission(aid, "G1", {
        "cameras": {c: f"/x/{c}.mp4" for c in cameras}, "engine": "pfb3x3", **meta})
    for pid, name in roster:
        repo.add_subject(sid, pid, name, None, {})
    return sid


def boxes(tmp_path, camera, first: dict[str, int], last=100) -> str:
    """A boxes.txt: each label boxed from its first frame to `last`, left to right by number."""
    path = tmp_path / f"{camera}_boxes.txt"
    lines = []
    for label, start in first.items():
        n = int(label[1:])
        lines += [f"{f},{n},{10 * n},5,8,20,0.9" for f in range(start, last + 1)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def track(repo, tmp_path, sid, camera="cam1", labels=LABELS, *, quality=None, first=None,
          defaults=None, with_quality=True):
    prep = repo.latest_run(sid, Stage.prepare, camera) or save(
        repo, sid, Stage.prepare, {"frames": []}, camera=camera)
    output = {"people": len(labels), "labels": list(labels), "video_id": f"s{sid}_{camera}",
              "boxes": boxes(tmp_path, camera, first or {lb: 1 for lb in labels}),
              "default_names": defaults or {}, "default_source": "rule" if defaults else None}
    if with_quality:
        output["quality"] = quality
    return save(repo, sid, Stage.track, output, camera=camera, upstream=[prep.id])


def q(**coverage) -> dict:
    """Quality where the named labels have the given coverage and a good height; the rest
    of LABELS are fragments."""
    out = {lb: {"coverage": 0.05, "mean_h_frac": 0.1} for lb in LABELS}
    out.update({lb: {"coverage": c, "mean_h_frac": 0.2} for lb, c in coverage.items()})
    return out


def people(repo, sid):
    return {s["participant_id"]: (s["display_name"], s["excluded"])
            for s in repo.list_subjects(sid)}


def names_by_label(repo, sid, camera="cam1"):
    subjects = {s["participant_id"]: s["display_name"] for s in repo.list_subjects(sid)}
    mapping = label_mapping(repo.latest_run(sid, Stage.identity, camera).output)
    return {lb: subjects[pid] for lb, pid in mapping.items() if pid is not None}


def queued(repo, sid):
    return [(r["stage"], r["camera"]) for r in repo.conn.execute(
        "SELECT stage, camera FROM jobs WHERE submission_id=? AND status='queued' ORDER BY id",
        (sid,))]


# --- auto_identity ---

def test_auto_identity_maps_selected_and_names_them(repo, tmp_path):
    sid = work(repo)
    first = {lb: 50 for lb in LABELS} | {"P40": 1, "P5": 2, "P12": 3}
    t = track(repo, tmp_path, sid, quality=q(P5=0.9, P12=0.6, P40=0.3), first=first)
    run = auto_people.auto_identity(repo, sid, "cam1")
    assert (run.source, run.upstream_run_ids, run.output["track_run_id"]) == ("auto", [t.id], t.id)
    assert names_by_label(repo, sid) == {"P40": "参加者1", "P5": "参加者2", "P12": "参加者3"}
    mapping = label_mapping(run.output)
    assert sum(pid is None for pid in mapping.values()) == 44
    added = review.added_people(repo.get_submission(sid)["meta"])
    assert added == {"auto1": "auto", "auto2": "auto", "auto3": "auto"}
    assert all(not excluded for _, excluded in people(repo, sid).values())


def test_auto_identity_without_a_usable_track_saves_nothing(repo):
    sid = work(repo)
    assert auto_people.auto_identity(repo, sid, "cam1") is None
    assert repo.latest_run(sid, Stage.identity, "cam1") is None


def test_sports_names_are_senshu(repo, tmp_path):
    sid = work(repo, domain="sports")
    track(repo, tmp_path, sid, quality=q(P3=0.8))
    auto_people.auto_identity(repo, sid, "cam1")
    assert names_by_label(repo, sid) == {"P3": "選手1"}


def test_names_continue_across_cameras(repo, tmp_path):
    sid = work(repo, cameras=("cam1", "cam2"))
    track(repo, tmp_path, sid, "cam1", quality=q(P1=0.9, P2=0.9, P3=0.9))
    track(repo, tmp_path, sid, "cam2", quality=q(P7=0.9, P8=0.9))
    auto_people.auto_identity(repo, sid, "cam1")
    auto_people.auto_identity(repo, sid, "cam2")
    assert names_by_label(repo, sid, "cam2") == {"P7": "参加者4", "P8": "参加者5"}
    assert set(people(repo, sid)) == {"auto1", "auto2", "auto3", "auto4", "auto5"}


def test_auto_identity_uses_rule_defaults(repo, tmp_path):
    roster = (("s1", "A子"), ("s2", "B太"), ("s3", "C美"), ("s4", "D郎"))
    sid = work(repo, roster=roster)
    defaults = {"P1": "s1", "P2": "s2", "P3": "s3", "P4": "s4"}
    track(repo, tmp_path, sid, labels=["P1", "P2", "P3", "P4"], quality=None, defaults=defaults)
    run = auto_people.auto_identity(repo, sid, "cam1")
    assert label_mapping(run.output) == defaults and run.source == "auto"
    assert people(repo, sid) == {pid: (name, False) for pid, name in roster}
    assert review.added_people(repo.get_submission(sid)["meta"]) == {}


def test_unmapped_roster_subjects_are_excluded(repo, tmp_path):
    sid = work(repo, roster=(("s1", "A子"), ("s2", "B太")))
    track(repo, tmp_path, sid, quality=q(P1=0.9, P2=0.8))
    auto_people.auto_identity(repo, sid, "cam1")
    assert people(repo, sid) == {"s1": ("A子", True), "s2": ("B太", True),
                                 "auto1": ("参加者1", False), "auto2": ("参加者2", False)}


# --- auto_confirm ---

def test_auto_confirm_queues_behavior(repo, tmp_path):
    sid = work(repo)
    track(repo, tmp_path, sid, quality=q(P1=0.9))
    auto_people.auto_identity(repo, sid, "cam1")
    ids = auto_people.auto_confirm(repo, sid)
    assert len(ids) == 1 and queued(repo, sid) == [("behavior", "cam1")]
    assert repo.get_submission(sid)["identity_confirmed"]


def test_auto_confirm_waits_for_every_camera(repo, tmp_path):
    sid = work(repo, cameras=("cam1", "cam2"))
    track(repo, tmp_path, sid, "cam1", quality=q(P1=0.9))
    auto_people.auto_identity(repo, sid, "cam1")
    assert auto_people.auto_confirm(repo, sid) == []          # cam2 has no mapping yet
    repo.enqueue(sid, Stage.track, "track", camera="cam2")
    track(repo, tmp_path, sid, "cam2", quality=q(P2=0.9))
    auto_people.auto_identity(repo, sid, "cam2")
    assert auto_people.auto_confirm(repo, sid) == []          # cam2's track job is still active
    assert not repo.get_submission(sid)["identity_confirmed"]
    repo.conn.execute("DELETE FROM jobs WHERE submission_id=?", (sid,))
    assert len(auto_people.auto_confirm(repo, sid)) == 2
    assert queued(repo, sid) == [("behavior", "cam1"), ("behavior", "cam2")]


def test_auto_confirm_is_never_an_error_for_a_document(repo):
    aid = repo.create_assignment("課題", Kind.document, "", "", "standard", False)
    sid = repo.create_submission(aid, "G1", {"cameras": {}})
    assert auto_people.auto_confirm(repo, sid) == []


# --- the runner ---

def test_a_finished_track_job_evaluates_people_automatically(repo, tmp_path):
    sid = work(repo)
    track(repo, tmp_path, sid, quality=q(P1=0.9, P2=0.9))
    job = SimpleNamespace(submission_id=sid, stage=Stage.track, camera="cam1", subject_id=None)
    assert runner.enqueue_followups(repo, job) == []          # names are not wanted here
    assert repo.latest_run(sid, Stage.identity, "cam1").source == "auto"
    assert repo.get_submission(sid)["identity_confirmed"]
    assert queued(repo, sid) == [("behavior", "cam1")]


def test_a_failing_auto_identity_leaves_step_3_to_the_coach(repo, tmp_path, monkeypatch,
                                                           caplog):
    sid = work(repo)
    track(repo, tmp_path, sid, quality=q(P1=0.9))

    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(auto_people, "auto_identity", boom)
    job = SimpleNamespace(submission_id=sid, stage=Stage.track, camera="cam1", subject_id=None)
    assert runner.enqueue_followups(repo, job) == []
    assert repo.latest_run(sid, Stage.identity, "cam1") is None and queued(repo, sid) == []
    assert "boom" in caplog.text


# --- reselect ---

class StubEngine:
    def __init__(self, quality=None):
        self.calls, self._quality = [], quality

    def quality(self, sub_dir, video_id, *, on_tick=None, timeout_sec=None):
        self.calls.append(video_id)
        self.timeouts = [*getattr(self, "timeouts", []), timeout_sec]
        if isinstance(self._quality, Exception):
            raise self._quality
        return self._quality


def test_reselect_old_track_run_without_quality(repo, tmp_path, monkeypatch):
    sid = work(repo, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        first = {lb: 90 for lb in LABELS} | {"P2": 1, "P9": 5}   # two people seen all along
        t = track(repo, tmp_path, sid, cam, first=first, with_quality=False)
        ident = save(repo, sid, Stage.identity, {
            "participants": [{"participant_id": None, "label": lb, "display_name": None,
                              "note": ""} for lb in LABELS],
            "labels": LABELS, "track_run_id": t.id}, camera=cam, upstream=[t.id],
            source="teacher")
        save(repo, sid, Stage.behavior, {"events": []}, camera=cam, upstream=[ident.id])
    repo.set_identity_confirmed(sid, True)
    stub = StubEngine(None)
    monkeypatch.setattr(auto_people, "Pfb3x3Engine", lambda settings: stub)
    ids = auto_people.reselect(repo, repo.settings, sid)
    assert stub.calls == [f"s{sid}_cam1", f"s{sid}_cam2"]
    assert names_by_label(repo, sid, "cam1") == {"P2": "参加者1", "P9": "参加者2"}
    assert names_by_label(repo, sid, "cam2") == {"P2": "参加者3", "P9": "参加者4"}
    assert repo.latest_run(sid, Stage.identity, "cam1").source == "auto"
    assert repo.get_submission(sid)["identity_confirmed"]
    assert len(ids) == 2 and queued(repo, sid) == [("behavior", "cam1"), ("behavior", "cam2")]
    action = repo.conn.execute("SELECT action FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    assert action[0] == "auto_people_reselect"


def test_reselect_quality_has_a_short_timeout_and_falls_back_to_boxes(repo, tmp_path,
                                                                      monkeypatch):
    from pfb_app.engine.pfb3x3 import EngineTimeout
    sid = work(repo)
    first = {lb: 90 for lb in LABELS} | {"P2": 1}
    track(repo, tmp_path, sid, first=first, with_quality=False)
    stub = StubEngine(EngineTimeout("エンジン quality がタイムアウトしました (120 秒)"))
    monkeypatch.setattr(auto_people, "Pfb3x3Engine", lambda settings: stub)
    auto_people.reselect(repo, repo.settings, sid)
    assert stub.timeouts == [120]
    assert names_by_label(repo, sid) == {"P2": "参加者1"}
    action = repo.conn.execute("SELECT detail FROM audit_log WHERE action='auto_identity'"
                               " ORDER BY id DESC LIMIT 1").fetchone()
    assert '"boxes"' in action[0]


def test_reselect_keeps_mapped_subjects_and_excludes_dropped(repo, tmp_path, monkeypatch):
    sid = work(repo)
    first = {lb: 1 for lb in LABELS} | {"P1": 1, "P2": 2, "P3": 3}
    track(repo, tmp_path, sid, quality=q(P1=0.9, P2=0.4, P3=0.8), first=first)
    auto_people.auto_identity(repo, sid, "cam1")
    assert names_by_label(repo, sid) == {"P1": "参加者1", "P2": "参加者2", "P3": "参加者3"}
    subjects = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    save(repo, sid, Stage.pfb, {"text": "x"}, subject_id=subjects["auto2"])   # P2 has a run
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "max_people": 1})
    monkeypatch.setattr(auto_people, "Pfb3x3Engine", lambda settings: pytest.fail("no engine"))
    auto_people.reselect(repo, repo.settings, sid)
    assert names_by_label(repo, sid) == {"P1": "参加者1"}     # the same subject is kept
    assert people(repo, sid) == {"auto1": ("参加者1", False), "auto2": ("参加者2", True)}


def test_reselect_busy(repo, tmp_path):
    sid = work(repo)
    track(repo, tmp_path, sid, quality=q(P1=0.9))
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    with pytest.raises(review.BusyError):
        auto_people.reselect(repo, repo.settings, sid)
    assert repo.latest_run(sid, Stage.identity, "cam1") is None


# --- names and exclusion ---

def two_people(repo, tmp_path, *, contribution=True):
    """A work with two automatic people and (optionally) a current contribution run."""
    sid = work(repo)
    track(repo, tmp_path, sid, quality=q(P1=0.9, P2=0.8))
    auto_people.auto_identity(repo, sid, "cam1")
    auto_people.auto_confirm(repo, sid)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    subjects = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    if contribution:
        ident = repo.latest_run(sid, Stage.identity, "cam1")
        beh = save(repo, sid, Stage.behavior, {"events": []}, camera="cam1", upstream=[ident.id])
        save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1", upstream=[beh.id])
    return sid, subjects


def test_rename_changes_only_the_name(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    pfb = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "良い。"},
               subject_id=subjects["auto1"])
    repo.set_approval(subjects["auto1"], pfb.id)
    auto_people.rename_person(repo, subjects["auto1"], "  山田花子 ")
    subject = repo.get_subject(subjects["auto1"])
    assert subject["display_name"] == "山田花子"
    assert (subject["approval_status"], subject["approved_run_id"]) == ("approved", pfb.id)
    assert repo.latest_run(sid, Stage.pfb, subject_id=subjects["auto1"]).output["pfb"] == "良い。"


def test_rename_rejects_blank_long_and_taken(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    with pytest.raises(ValueError, match="名前を入れてください。"):
        auto_people.rename_person(repo, subjects["auto1"], "  ")
    with pytest.raises(ValueError, match="40文字まで"):
        auto_people.rename_person(repo, subjects["auto1"], "あ" * 41)
    with pytest.raises(review.NameTaken):
        auto_people.rename_person(repo, subjects["auto1"], "参加者２")   # NFKC of 参加者2
    auto_people.rename_person(repo, subjects["auto1"], "参加者1")        # its own name is fine
    assert repo.get_subject(subjects["auto1"])["display_name"] == "参加者1"


def test_rename_is_allowed_while_jobs_run(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    auto_people.rename_person(repo, subjects["auto1"], "山田")
    assert repo.get_subject(subjects["auto1"])["display_name"] == "山田"


def roster_work(repo, tmp_path):
    """Two automatic people on a roster work without rule defaults: both roster subjects end
    up excluded and mapped by no camera."""
    sid = work(repo, roster=(("s1", "山田"), ("s2", "佐藤")))
    track(repo, tmp_path, sid, quality=q(P1=0.9, P2=0.8))
    auto_people.auto_identity(repo, sid, "cam1")
    subjects = {s["participant_id"]: s["id"] for s in repo.list_subjects(sid)}
    return sid, subjects


def test_rename_from_the_roster_replaces_the_unused_roster_row(repo, tmp_path):
    sid, subjects = roster_work(repo, tmp_path)
    auto_people.rename_person(repo, subjects["auto1"], "山田", roster=True)
    assert repo.get_subject(subjects["auto1"])["display_name"] == "山田"
    assert "s1" not in people(repo, sid)                       # the roster row is gone
    assert people(repo, sid)["s2"] == ("佐藤", True)


def test_rename_from_the_roster_keeps_a_roster_row_with_runs(repo, tmp_path):
    sid, subjects = roster_work(repo, tmp_path)
    save(repo, sid, Stage.pfb, {"text": "x"}, subject_id=subjects["s1"])
    auto_people.rename_person(repo, subjects["auto1"], "山田", roster=True)
    assert repo.get_subject(subjects["auto1"])["display_name"] == "山田"
    assert people(repo, sid)["s1"] == ("山田", True)


def test_typed_name_of_an_evaluated_person_is_still_taken(repo, tmp_path):
    sid, subjects = roster_work(repo, tmp_path)
    auto_people.rename_person(repo, subjects["auto1"], "山田", roster=True)
    for roster in (False, True):
        with pytest.raises(review.NameTaken):
            auto_people.rename_person(repo, subjects["auto2"], "山田", roster=roster)
    assert repo.get_subject(subjects["auto2"])["display_name"] == "参加者2"


def test_typed_name_of_an_unused_roster_row_is_taken(repo, tmp_path):
    sid, subjects = roster_work(repo, tmp_path)
    with pytest.raises(review.NameTaken):
        auto_people.rename_person(repo, subjects["auto1"], "山田")
    assert "s1" in people(repo, sid)


def test_excluded_gets_no_feedback_job(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    assert auto_people.set_person_excluded(repo, subjects["auto2"], True) == []
    job = SimpleNamespace(submission_id=sid, stage=Stage.contribution, camera="cam1",
                          subject_id=None)
    runner.enqueue_followups(repo, job)
    pfb_jobs = [r["subject_id"] for r in repo.conn.execute(
        "SELECT subject_id FROM jobs WHERE submission_id=? AND stage='pfb'", (sid,))]
    assert pfb_jobs == [subjects["auto1"]]


def test_excluding_cancels_the_queued_feedback_job(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=subjects["auto1"])
    repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=subjects["auto2"])
    auto_people.set_person_excluded(repo, subjects["auto1"], True)
    left = [r["subject_id"] for r in repo.conn.execute(
        "SELECT subject_id FROM jobs WHERE submission_id=? AND stage='pfb'", (sid,))]
    assert left == [subjects["auto2"]]


def test_excluding_a_running_feedback_job_is_busy(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.pfb, "pfb3x3:standard", subject_id=subjects["auto1"])
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    with pytest.raises(review.BusyError, match="解析が進んでいる間"):
        auto_people.set_person_excluded(repo, subjects["auto1"], True)
    assert not repo.get_subject(subjects["auto1"])["excluded"]


def test_restore_queues_missing_feedback(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    auto_people.set_person_excluded(repo, subjects["auto2"], True)
    ids = auto_people.set_person_excluded(repo, subjects["auto2"], False)
    assert len(ids) == 1 and not repo.get_subject(subjects["auto2"])["excluded"]
    assert queued(repo, sid) == [("pfb", None)]
    assert auto_people.set_person_excluded(repo, subjects["auto2"], False) == []   # no second


def test_restore_with_a_usable_feedback_queues_nothing(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "x"},
         subject_id=subjects["auto2"])
    auto_people.set_person_excluded(repo, subjects["auto2"], True)
    assert auto_people.set_person_excluded(repo, subjects["auto2"], False) == []


def test_restore_while_busy_queues_nothing(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    auto_people.set_person_excluded(repo, subjects["auto2"], True)
    repo.enqueue(sid, Stage.behavior, "pfb3x3:standard", camera="cam1")
    assert auto_people.set_person_excluded(repo, subjects["auto2"], False) == []
    assert not repo.get_subject(subjects["auto2"])["excluded"]
    assert queued(repo, sid) == [("behavior", "cam1")]


def test_restore_without_a_current_contribution_queues_nothing(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path, contribution=False)
    auto_people.set_person_excluded(repo, subjects["auto2"], True)
    assert auto_people.set_person_excluded(repo, subjects["auto2"], False) == []


def test_teacher_mapping_clears_excluded(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    auto_people.set_person_excluded(repo, subjects["auto2"], True)
    review.save_tracked_identity(repo, sid, "cam1", {**dict.fromkeys(LABELS), "P1": "auto1", "P2": "auto2"})
    assert not repo.get_subject(subjects["auto2"])["excluded"]


def test_teacher_mapping_excludes_dropped_auto_people_with_runs(repo, tmp_path):
    sid, subjects = two_people(repo, tmp_path)
    save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "x"},
         subject_id=subjects["auto2"])
    review.save_tracked_identity(repo, sid, "cam1", {**dict.fromkeys(LABELS), "P1": "auto1"})
    assert people(repo, sid) == {"auto1": ("参加者1", False), "auto2": ("参加者2", True)}


# --- the excluded column ---

def test_excluded_migration_is_guarded(tmp_path):
    conn = connect(tmp_path / "x.db")
    init_schema(conn)
    init_schema(conn)
    repo = Repo(conn)
    sid = work(repo)
    subject = repo.add_subject(sid, "s1", "A子", None, {})
    assert repo.get_subject(subject)["excluded"] is False
    repo.set_excluded(subject, True)
    assert repo.get_subject(subject)["excluded"] is True
    repo.set_excluded(subject, False)
    assert repo.list_subjects(sid)[0]["excluded"] is False


# --- 「もう一度解析する」 ---

def test_rerun_behavior_queues_every_camera(repo, tmp_path):
    sid = work(repo, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        track(repo, tmp_path, sid, cam, quality=q(P1=0.9))
        auto_people.auto_identity(repo, sid, cam)
    auto_people.auto_confirm(repo, sid)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    ids = auto_people.rerun_behavior(repo, sid)
    assert len(ids) == 2 and queued(repo, sid) == [("behavior", "cam1"), ("behavior", "cam2")]
    action = repo.conn.execute("SELECT action FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
    assert action[0] == "rerun_behavior"


def test_rerun_behavior_busy(repo, tmp_path):
    sid = work(repo)
    track(repo, tmp_path, sid, quality=q(P1=0.9))
    repo.enqueue(sid, Stage.contribution, "pfb3x3:standard", camera="cam1")
    with pytest.raises(review.BusyError) as e:
        auto_people.rerun_behavior(repo, sid)
    assert "ジョブ" not in str(e.value)
    assert queued(repo, sid) == [("contribution", "cam1")]
