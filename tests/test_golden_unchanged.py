"""Golden snapshot: what 一般 and スポーツ send to the models and the engine is exactly what it was
before activity-type profiles (ux-int 857685e).

tests/fixtures/golden_unchanged.json was written by this file run against 857685e
(GOLDEN_WRITE=1 writes it); it holds the contribution and feedback requests (system prompt
hash and user_text) of a general and a sports work, and the engine registry YAML the tracking
stage writes for both. This file uses only code that exists on both sides. Synthetic data only.
"""
import json
import os
from pathlib import Path

import pytest
import yaml

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.engine import Pfb3x3Engine
from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.video import run_contribution, run_pfb
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from fakes.engine_env import install_fake_engine
from test_pfb3x3_runner import claim_run, make_video
from test_pipeline_video import ev, events_run, judg, pfb_json, save, sports_events_run

GOLDEN = Path(__file__).parent / "fixtures" / "golden_unchanged.json"
PROMPTS = Path(__file__).resolve().parents[1] / "prompts"


def _requests(tmp_path, domain: str) -> dict:
    settings = Settings(data_dir=tmp_path / domain / "data", prompts_dir=PROMPTS)
    tmp_path.mkdir(parents=True, exist_ok=True)
    conn = connect(tmp_path / f"{domain}.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    sport = "soccer" if domain == "sports" else None
    aid = repo.create_assignment("ペーパータワー", Kind.video, "紙で塔を作る", "協力する",
                                 "standard", False, domain=domain, sport=sport)
    sid = repo.create_submission(aid, "G1", {})
    pa = repo.add_subject(sid, "Pa", "A子", "", {})
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    run_events = sports_events_run if domain == "sports" else events_run
    run_events(repo, sid, "cam1", [ev("00:10", "Pa", "紙を折る"), ev("00:15", "Pa", "塔を支える")])
    out = {}
    backend = FakeBackend(lambda r: judg(["cam1-0001", "cam1-0002"], {"cam1-0001"}))
    run = run_contribution(StageContext(settings, repo, backend, lambda: False), sid, "cam1")
    out["contribution"] = {"prompt_hash": run.prompt_hash, "user_text": backend.calls[0].user_text}
    backend = FakeBackend(lambda r: pfb_json(["cam1-0001"]))
    run = run_pfb(StageContext(settings, repo, backend, lambda: False), sid, pa)
    out["pfb"] = {"prompt_hash": run.prompt_hash, "user_text": backend.calls[0].user_text}
    return out


def _registry(tmp_path, monkeypatch, domain: str) -> str:
    fake = install_fake_engine(tmp_path / domain, monkeypatch)
    settings = Settings(data_dir=tmp_path / domain / "data", prompts_dir=PROMPTS,
                        engine_dir=fake.engine_dir, engine_python=fake.engine_python)
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    sport = "soccer" if domain == "sports" else None
    aid = repo.create_assignment("ペーパータワー", Kind.video, "紙で塔を作る", "", "standard", True,
                                 domain=domain, sport=sport)
    video = make_video(tmp_path / domain / "c.mp4")
    sid = repo.create_submission(aid, "G1", {"cameras": {"cam1": str(video)}, "start_sec": 1.0,
                                             "end_sec": None, "engine": "pfb3x3"})
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    repo.enqueue(sid, Stage.track, "track", camera="cam1")
    ctx = StageContext(settings, repo, FakeBackend(["{}"]), lambda: False,
                       engine=Pfb3x3Engine(settings, tick_sec=0.05, term_grace_sec=2),
                       device="cuda")
    claim_run(repo, ctx, {"track"})
    f = files.submission_dir(settings, aid, sid) / "pfb3x3" / "videos" / f"s{sid}_cam1.yaml"
    entry = yaml.safe_load(f.read_text(encoding="utf-8"))
    entry["path"] = Path(entry["path"]).name     # the tmp dir differs per run
    return yaml.safe_dump(entry, allow_unicode=True, sort_keys=False)


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    out = {}
    for domain in ("school", "sports"):
        out[domain] = _requests(tmp_path / "req", domain)
        out[domain]["registry"] = _registry(tmp_path / "reg", monkeypatch, domain)
    return out


def test_general_and_sports_model_and_engine_input_is_unchanged(snapshot):
    if os.environ.get("GOLDEN_WRITE"):
        GOLDEN.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n",
                          encoding="utf-8")
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert snapshot == golden
