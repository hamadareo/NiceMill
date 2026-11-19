import json
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.inference.fake import FakeBackend
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline.document import StageContext, run_contribution, run_evidence, run_pfb
from pfb_app.storage.db import Repo, connect, init_schema

QUOTE1 = "私たちは地域の防災について調べました"
QUOTE2 = "避難所までの距離を地図で測りました"
RUBRIC = "ルーブリック秘密項目XYZ"
PAGE_TEXT = f"{QUOTE1}。\n{QUOTE2}。"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


@pytest.fixture
def env(tmp_path, settings):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("課題", Kind.document, "防災を調べる課題", RUBRIC, "standard", False)
    sid = repo.create_submission(aid, "A", {})
    subj = repo.add_subject(sid, None, "提出者", None, {})
    page = {"number": 1, "text": PAGE_TEXT, "image_path": None}
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.prepare, camera=None, subject_id=None, version=0,
        source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output={"pages": [page], "pdf_path": "x.pdf"}, status="ok", error=None, warnings=[],
        duration_sec=0.0, created_at=""))
    return repo, sid, subj


def ctx_for(settings, repo, fn, yield_check=lambda: False):
    backend = FakeBackend(fn)
    return StageContext(settings, repo, backend, yield_check), backend


def evid(*quotes, source="text"):
    return json.dumps({"items": [
        {"page": 1, "quote": q, "observed_content": "内容", "evidence_source": source,
         "confidence": "high"} for q in quotes]}, ensure_ascii=False)


def judg(ids, true_ids=()):
    return json.dumps({"judgments": [
        {"ref_id": i, "contribution_interpretation": "解釈",
         "is_positive_contribution": i in true_ids, "rubric_criteria": []} for i in ids]})


def pfb_json(refs):
    return json.dumps({"pfb_status": "generated", "pfb": "調べたことが課題を進めました。",
                       "evidence_refs": refs}, ensure_ascii=False)


def responder(req):
    if "抜き出す担当" in req.system:
        return evid(QUOTE1, QUOTE2)
    if "判定する担当" in req.system:
        return judg(["p1-001", "p1-002"], ["p1-001"])
    return pfb_json(["p1-001"])


def test_document_happy_path(settings, env):
    repo, sid, subj = env
    ctx, _ = ctx_for(settings, repo, responder)
    e = run_evidence(ctx, sid)
    c = run_contribution(ctx, sid)
    p = run_pfb(ctx, sid, subj)
    assert [r.status for r in (e, c, p)] == ["ok"] * 3
    prep = repo.latest_run(sid, Stage.prepare)
    assert e.upstream_run_ids == [prep.id]
    assert c.upstream_run_ids == [e.id] and p.upstream_run_ids == [c.id]
    assert p.subject_id == subj and p.model_id == "fake" and p.prompt_hash
    assert [i["item_id"] for i in e.output["items"]] == ["p1-001", "p1-002"]
    assert all(i["quote_verified"] for i in e.output["items"])
    assert p.output["evidence_refs"] == ["p1-001"] and p.warnings == []
    assert repo.latest_run(sid, Stage.pfb, subject_id=subj).id == p.id


def test_evidence_flags_unverified_quote(settings, env):
    repo, sid, _ = env
    ctx, _ = ctx_for(settings, repo, lambda r: evid(QUOTE1, "ページに存在しない引用文章です"))
    e = run_evidence(ctx, sid)
    assert e.status == "ok"
    assert [i["quote_verified"] for i in e.output["items"]] == [True, False]
    assert len(e.warnings) == 1 and "p1-002" in e.warnings[0]


def test_evidence_figure_skips_quote_check(settings, env):
    repo, sid, _ = env
    ctx, _ = ctx_for(settings, repo, lambda r: evid("", source="figure"))
    e = run_evidence(ctx, sid)
    assert e.output["items"][0]["quote_verified"] is True and e.warnings == []


def test_evidence_without_prepare_fails(settings, tmp_path):
    conn = connect(tmp_path / "b.db")
    init_schema(conn)
    repo = Repo(conn, settings)
    aid = repo.create_assignment("t", Kind.document, "x", "y", "standard", False)
    sid = repo.create_submission(aid, "A", {})
    ctx, backend = ctx_for(settings, repo, lambda r: evid())
    e = run_evidence(ctx, sid)
    assert e.status == "failed" and e.error and backend.calls == []


def test_contribution_rejects_misaligned_then_fails(settings, env):
    repo, sid, _ = env
    ctx, _ = ctx_for(settings, repo, lambda r: evid(QUOTE1, QUOTE2))
    run_evidence(ctx, sid)
    ctx2, _ = ctx_for(settings, repo, lambda r: judg(["p1-001"]))
    c = run_contribution(ctx2, sid)
    assert c.status == "failed" and "p1-002" in c.error and c.output is None


def test_pfb_insufficient_without_model_call(settings, env):
    repo, sid, subj = env
    ctx, _ = ctx_for(settings, repo, lambda r: evid(QUOTE1))
    run_evidence(ctx, sid)
    ctx, _ = ctx_for(settings, repo, lambda r: judg(["p1-001"]))
    run_contribution(ctx, sid)
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    p = run_pfb(ctx, sid, subj)
    assert backend.calls == []
    assert p.status == "ok" and p.model_id is None
    assert p.output["pfb_status"] == "insufficient_positive_evidence"


def test_pfb_only_sees_true_items(settings, env):
    repo, sid, subj = env
    ctx, _ = ctx_for(settings, repo, responder)
    run_evidence(ctx, sid)
    run_contribution(ctx, sid)
    run_pfb(ctx, sid, subj)
    user_text = ctx.backend.calls[-1].user_text
    assert "p1-001" in user_text and "p1-002" not in user_text
    assert QUOTE2 not in user_text


def test_pfb_bad_refs_fail_and_text_warnings(settings, env):
    repo, sid, subj = env
    ctx, _ = ctx_for(settings, repo, responder)
    run_evidence(ctx, sid)
    run_contribution(ctx, sid)
    ctx2, _ = ctx_for(settings, repo, lambda r: pfb_json(["p1-002"]))
    assert run_pfb(ctx2, sid, subj).status == "failed"
    bad = json.dumps({"pfb_status": "generated", "pfb": "p1-001 の赤い図が良かった",
                      "evidence_refs": ["p1-001"]}, ensure_ascii=False)
    ctx3, _ = ctx_for(settings, repo, lambda r: bad)
    ctx3.backend.model_id = "fake2"  # avoid the call_log cache hit from the previous call
    p = run_pfb(ctx3, sid, subj)
    assert p.status == "ok" and len(p.warnings) >= 2


def test_rubric_not_in_evidence_prompt(settings, env):
    repo, sid, _ = env
    ctx, backend = ctx_for(settings, repo, lambda r: evid(QUOTE1))
    run_evidence(ctx, sid)
    for req in backend.calls:
        assert RUBRIC not in req.system + req.user_text
    assert "防災を調べる課題" in backend.calls[0].user_text
    assert "p1" in backend.calls[0].user_text
    run_contribution(ctx_for(settings, repo, lambda r: judg(["p1-001"]))[0], sid)


def test_yield_propagates(settings, env):
    repo, sid, _ = env
    ctx, _ = ctx_for(settings, repo, lambda r: evid(QUOTE1), yield_check=lambda: True)
    with pytest.raises(YieldRequested):
        run_evidence(ctx, sid)


def test_pfb_missing_evidence_run_fails_gracefully(settings, env):
    repo, sid, subj = env
    bad = repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.evidence, camera=None, subject_id=None, version=0,
        source="model", upstream_run_ids=[], prompt_hash=None, model_id=None, output=None,
        status="failed", error="x", warnings=[], duration_sec=0.0, created_at=""))
    repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=Stage.contribution, camera=None, subject_id=None,
        version=0, source="teacher", upstream_run_ids=[bad.id], prompt_hash=None, model_id=None,
        output={"judgments": json.loads(judg(["p1-001"], ["p1-001"]))["judgments"]},
        status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    p = run_pfb(ctx, sid, subj)
    assert p.status == "failed" and p.error and backend.calls == []


def test_pfb_dangling_ref_id_fails_gracefully(settings, env):
    repo, sid, subj = env
    ctx, _ = ctx_for(settings, repo, responder)
    run_evidence(ctx, sid)
    c = run_contribution(ctx, sid)
    edited = repo.save_stage_run(c.model_copy(update={
        "id": None, "source": "teacher",
        "output": {"judgments": json.loads(judg(["p1-009"], ["p1-009"]))["judgments"]}}))
    ctx2, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    p = run_pfb(ctx2, sid, subj)
    assert edited.version == 2
    assert p.status == "failed" and "p1-009" in p.error and backend.calls == []
