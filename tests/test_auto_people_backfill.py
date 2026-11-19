"""scripts/auto_people_backfill.py: which tracking groups get their people chosen (temp DB)."""
import importlib.util
from pathlib import Path

import yaml

from pfb_app.domain.models import Kind, Stage
from pfb_app.services import auto_people
from test_auto_people_service import q, repo, save, track, work  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "auto_people_backfill", ROOT / "scripts" / "auto_people_backfill.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def teacher_mapping(repo, sid, camera="cam1"):
    t = repo.latest_run(sid, Stage.track, camera)
    return save(repo, sid, Stage.identity, {
        "participants": [{"participant_id": None, "label": lb, "display_name": None}
                         for lb in t.output["labels"]],
        "labels": t.output["labels"], "track_run_id": t.id}, camera=camera,
        upstream=[t.id], source="teacher")


def test_candidates(repo, tmp_path):
    backfill = _script()
    good = work(repo, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        track(repo, tmp_path, good, cam, quality=q(P1=0.9))
    half = work(repo, cameras=("cam1", "cam2"))            # one camera already automatic
    for cam in ("cam1", "cam2"):
        track(repo, tmp_path, half, cam, quality=q(P1=0.9))
    auto_people.auto_identity(repo, half, "cam1")
    confirmed = work(repo)
    track(repo, tmp_path, confirmed, quality=q(P1=0.9))
    repo.set_identity_confirmed(confirmed, True)
    untracked = work(repo, cameras=("cam1", "cam2"))
    track(repo, tmp_path, untracked, "cam1", quality=q(P1=0.9))
    busy = work(repo)
    track(repo, tmp_path, busy, quality=q(P1=0.9))
    repo.enqueue(busy, Stage.behavior, "pfb3x3:standard", camera="cam1")
    coach = work(repo, cameras=("cam1", "cam2"))
    for cam in ("cam1", "cam2"):
        track(repo, tmp_path, coach, cam, quality=q(P1=0.9))
    teacher_mapping(repo, coach, "cam1")
    done = work(repo)                                       # every camera mapped already
    track(repo, tmp_path, done, quality=q(P1=0.9))
    auto_people.auto_identity(repo, done, "cam1")
    paper = work(repo, engine="paper")
    track(repo, tmp_path, paper, quality=q(P1=0.9))
    aid = repo.create_assignment("書類", Kind.document, "練習", "", "standard", False)
    repo.create_submission(aid, "D1", {"engine": "pfb3x3"})
    assert backfill.candidates(repo) == [(good, 2), (half, 2)]


def test_dry_run_then_apply(repo, tmp_path, monkeypatch, capsys):
    backfill = _script()
    repo.add_subject(sid := work(repo), "s1", "山田", None, {})
    track(repo, tmp_path, sid, quality=q(P1=0.9, P2=0.8))
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump({"data_dir": str(repo.settings.data_dir),
                                             "prompts_dir": str(repo.settings.prompts_dir)}),
                             encoding="utf-8")
    monkeypatch.setattr(auto_people, "Pfb3x3Engine", lambda s: None)   # quality is saved
    assert backfill.main(["--settings", str(settings_file)]) == 0
    out = capsys.readouterr().out
    assert f"submission {sid} (1 camera(s))" in out and "dry run" in out
    assert repo.latest_run(sid, Stage.identity, "cam1") is None
    assert backfill.main(["--settings", str(settings_file), "--apply"]) == 0
    out = capsys.readouterr().out
    assert f"submission {sid} (1 camera(s)): confirmed, 1 job(s) queued" in out
    assert "山田" not in out and "参加者" not in out
    assert repo.latest_run(sid, Stage.identity, "cam1").source == "auto"
    assert repo.get_submission(sid)["identity_confirmed"]
    assert backfill.main(["--settings", str(settings_file)]) == 0     # nothing left
    assert "0 group(s)" in capsys.readouterr().out


def test_no_database(tmp_path, capsys):
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump({"data_dir": str(tmp_path / "none")}),
                             encoding="utf-8")
    assert _script().main(["--settings", str(settings_file)]) == 2
    assert not (tmp_path / "none").exists()
