"""次にためしてみること: the worker's part (pipeline.tips) and where the runner calls it."""
import json
from pathlib import Path

import pytest
from tips_helpers import (FEEDBACK, PA_EVENTS, TIP, ctx_for, group, save, team_json,
                          tips_json)

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import tips
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


@pytest.fixture
def repo(tmp_path, settings):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, settings)


def latest(repo, sid, subject_id=None):
    return repo.latest_run(sid, Stage.tips, subject_id=subject_id)


def test_person_tips_are_made_from_the_feedback_and_its_events(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    run = tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    assert run.status == "ok" and run.source == "model" and run.subject_id == subj["Pa"]
    assert run.output == {"tips_status": "generated", "pfb_run_id": pfb["Pa"].id, "items": [
        {"text": TIP["Pa"], "evidence_refs": ["cam1-0001"]}]}
    assert run.upstream_run_ids == pfb["Pa"].upstream_run_ids
    assert run.prompt_hash and run.model_id == "fake"
    assert latest(repo, sid, subj["Pa"]).id == run.id
    sent = backend.calls[0].user_text
    assert FEEDBACK["Pa"] in sent and all(t in sent for t in PA_EVENTS)
    # only the person's own events, no time, no other person
    assert "ディフェンス" not in sent and "00:" not in sent and "山田花子" not in sent


def test_sports_work_uses_the_sports_prompt(settings, repo):
    _, sid, subj, pfb = group(repo, domain="sports", sport="basketball")
    ctx, backend = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    assert "コーチ" in backend.calls[0].system
    assert "競技" in backend.calls[0].user_text


def test_general_work_prompt_has_no_coach_words(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    assert "コーチ" not in backend.calls[0].system and "評価者" in backend.calls[0].system


@pytest.mark.parametrize("answer, why", [
    (lambda r: "not json", "JSON"),
    (lambda r: tips_json("Pa", refs=["cam1-0003"]), "存在しない根拠ID"),     # another person's event
    (lambda r: tips_json("Pa", text="上手にドリブルしてみる"), "評価する"),
    (lambda r: tips_json("Pa", text="ダンクを決めてみる"), "ダンク"),
    (lambda r: tips_json("Pa", text="山田花子さんのドリブルを続けてみる"), "名前"),
    (lambda r: tips_json("Pa", text="ドリブルを3回続けてみる"), "数字"),
])
def test_a_bad_answer_gives_no_tips_and_the_reason(settings, repo, answer, why):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, answer)
    run = tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    assert run.status == "ok" and run.output["tips_status"] == "none"
    assert run.output["items"] == [] and why in run.output["reason"]
    assert len(backend.calls) == 1 + settings.max_retries     # asked again with the faults


def test_a_second_answer_that_fixes_it_is_taken(settings, repo):
    _, sid, subj, pfb = group(repo)
    answers = [tips_json("Pa", text="上手にドリブルしてみる"), tips_json("Pa")]
    ctx, backend = ctx_for(settings, repo, lambda r: answers[len(backend.calls) - 1])
    run = tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    assert run.output["items"][0]["text"] == TIP["Pa"]
    assert "評価する" in backend.calls[1].user_text     # the retry carries what was wrong


def test_the_model_may_decline(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, _ = ctx_for(settings, repo, lambda r: json.dumps(
        {"tips_status": "insufficient_evidence", "tips": []}))
    run = tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    assert run.output["tips_status"] == "none" and "insufficient" in run.output["reason"]


def test_no_model_call_for_feedback_without_cited_events(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    hand = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "手書きの文章です。",
                                       "evidence_refs": []}, subject_id=subj["Pa"],
                upstream=pfb["Pa"].upstream_run_ids, source="teacher")
    assert tips.run_person_tips(ctx, sid, subj["Pa"], hand).output["tips_status"] == "none"
    thin = save(repo, sid, Stage.pfb, {"pfb_status": "insufficient_positive_evidence",
                                       "pfb": None, "evidence_refs": []}, subject_id=subj["Pa"])
    assert tips.run_person_tips(ctx, sid, subj["Pa"], thin).output["tips_status"] == "none"
    assert backend.calls == []


def test_after_pfb_never_breaks_the_feedback(settings, repo):
    _, sid, subj, pfb = group(repo)

    def boom(req):
        raise RuntimeError("backend exploded")
    ctx, _ = ctx_for(settings, repo, boom)
    run = tips.after_pfb(ctx, pfb["Pa"])
    assert run.output["tips_status"] == "none" and "backend exploded" in run.output["reason"]


def test_after_pfb_lets_the_session_ending_errors_through(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, _ = ctx_for(settings, repo, lambda r: tips_json("Pa"), yield_check=lambda: True)
    with pytest.raises(YieldRequested):
        tips.after_pfb(ctx, pfb["Pa"])

    def oom(req):
        raise RuntimeError("CUDA out of memory. Tried to allocate")
    ctx, _ = ctx_for(settings, repo, oom)
    with pytest.raises(RuntimeError):
        tips.after_pfb(ctx, pfb["Pa"])


def test_a_newer_feedback_run_replaces_older_tips(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, _ = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    first = tips.run_person_tips(ctx, sid, subj["Pa"], pfb["Pa"])
    thin = save(repo, sid, Stage.pfb, {"pfb_status": "insufficient_positive_evidence",
                                       "pfb": None, "evidence_refs": []}, subject_id=subj["Pa"])
    second = tips.after_pfb(ctx, thin)
    assert latest(repo, sid, subj["Pa"]).id == second.id != first.id
    assert second.output["items"] == []


# --- the group ---------------------------------------------------------------------------------

def test_team_tips_come_from_the_texts_without_names(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: team_json())
    run = tips.run_team_tips(ctx, sid)
    assert run.subject_id is None and run.camera is None and run.source == "model"
    assert run.output == {"tips_status": "generated", "items": [
        {"text": "ボールを持った仲間を見て、声をかけながら動いてみる", "evidence_refs": []}]}
    contribution = repo.latest_run(sid, Stage.contribution, "cam1")
    assert run.upstream_run_ids == [contribution.id, pfb["Pa"].id, pfb["Pz"].id]
    sent = backend.calls[0].user_text
    assert FEEDBACK["Pa"] in sent and FEEDBACK["Pz"] in sent
    assert "人1" in sent and "人2" in sent
    assert "山田花子" not in sent and "鈴木一郎" not in sent and "Pa" not in sent
    assert "先生" not in backend.calls[0].system


def test_team_tips_need_two_people_with_a_text(settings, repo):
    _, sid, subj, pfb = group(repo)
    save(repo, sid, Stage.pfb, {"pfb_status": "insufficient_positive_evidence", "pfb": None,
                                "evidence_refs": []}, subject_id=subj["Pz"])
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    assert tips.run_team_tips(ctx, sid) is None
    assert latest(repo, sid) is None and backend.calls == []


def test_team_tip_from_one_person_only_is_refused(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, _ = ctx_for(settings, repo, lambda r: team_json(people=(1,)))
    run = tips.run_team_tips(ctx, sid)
    assert run.output["tips_status"] == "none" and "2人以上" in run.output["reason"]


def test_a_tip_for_one_person_is_dropped_and_the_shared_one_is_kept(settings, repo):
    _, sid, subj, pfb = group(repo)
    answer = json.dumps({"tips_status": "generated", "tips": [
        {"text": "ボールを持った仲間を見て、声をかけながら動いてみる", "people": [1, 2]},
        {"text": "声をかけながら動いてみる", "people": [1]}]}, ensure_ascii=False)
    ctx, backend = ctx_for(settings, repo, lambda r: answer)
    run = tips.run_team_tips(ctx, sid)
    assert run.output["tips_status"] == "generated"
    assert [i["text"] for i in run.output["items"]] == [
        "ボールを持った仲間を見て、声をかけながら動いてみる"]
    assert len(backend.calls) == 1   # no second ask


def made_by_job(ctx, repo, sid):
    """The group's tips as the worker makes them: the job queued after the last text, run."""
    from pfb_app.pipeline.runner import execute_job
    job_id = tips.team_after_pfb(ctx, sid)
    if job_id is None:
        return None
    execute_job(repo.claim_next_job({repo.get_job(job_id).model_key}), ctx)
    return latest(repo, sid)


def test_team_after_pfb_waits_for_the_last_text_and_keeps_what_is_there(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: team_json())
    job = repo.enqueue(sid, Stage.pfb, "standard", None, subj["Pz"])
    assert made_by_job(ctx, repo, sid) is None and backend.calls == []   # Pz is still queued
    repo.conn.execute("UPDATE jobs SET status='done'")
    first = made_by_job(ctx, repo, sid)
    assert first is not None and first.output["items"]
    assert made_by_job(ctx, repo, sid) is None            # tips already there: kept
    assert len(backend.calls) == 1


def test_team_after_pfb_tries_again_after_a_failure_but_never_over_the_coachs(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: "nope")
    assert made_by_job(ctx, repo, sid).output["tips_status"] == "none"
    ctx2, _ = ctx_for(settings, repo, lambda r: team_json())
    ctx2.backend.model_id = "fake2"        # not the cached failure
    assert made_by_job(ctx2, repo, sid).output["items"]
    from pfb_app.services import tips as service
    service.confirm(repo, sid, None, "", latest(repo, sid).id)     # the coach deleted them
    assert made_by_job(ctx2, repo, sid) is None


def test_no_group_tips_while_the_texts_are_outdated(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, _ = ctx_for(settings, repo, lambda r: team_json())
    made_by_job(ctx, repo, sid)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    repo.save_stage_run(con.model_copy(update={"id": None, "version": 0}))    # redone
    # the feedback is outdated too: no person is left to stand on
    assert tips.team_after_pfb(ctx, sid) is None


# --- the runner --------------------------------------------------------------------------------

def test_the_runner_makes_the_tips_next_to_the_feedback(settings, repo, monkeypatch):
    from pfb_app.domain.models import Job, JobStatus
    from pfb_app.pipeline import runner, video
    _, sid, subj, pfb = group(repo)
    monkeypatch.setattr(video, "run_pfb", lambda ctx, s, sub: pfb["Pa"])
    ctx, backend = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    job = Job(id=1, submission_id=sid, stage=Stage.pfb, camera=None, subject_id=subj["Pa"],
              model_key="standard", status=JobStatus.running, attempts=1, heartbeat_at=None,
              error=None)
    assert runner._run_stage(job, ctx) == [pfb["Pa"]]
    assert latest(repo, sid, subj["Pa"]).output["items"][0]["text"] == TIP["Pa"]


def test_a_feedback_job_leaves_person_and_group_tips_behind(settings, repo):
    """Two feedback jobs through execute_job: each leaves its person's tips; the group's follow
    the last one only."""
    from pfb_app.pipeline.runner import execute_job
    _, sid, subj, _ = group(repo, feedback=False)

    def answer(req):
        if "「次にためしてみること」を書く" in req.system:
            return tips_json("Pa" if "ドリブル" in req.user_text else "Pz")
        if "ためしてみること」を書く" in req.system:
            return team_json()
        mine = "Pa" if "ドリブルでボール" in req.user_text else "Pz"
        return json.dumps({"pfb_status": "generated", "pfb": FEEDBACK[mine],
                           "evidence_refs": ["cam1-0001"] if mine == "Pa" else ["cam1-0003"]},
                          ensure_ascii=False)
    ctx, backend = ctx_for(settings, repo, answer)
    for who in ("Pa", "Pz"):
        repo.enqueue(sid, Stage.pfb, "standard", None, subj[who])
    first = repo.claim_next_job({"standard"})
    second = repo.claim_next_job({"standard"})     # both running at once, as a parallel session
    execute_job(first, ctx)
    assert latest(repo, sid, subj["Pa"]).output["items"][0]["text"] == TIP["Pa"]
    assert latest(repo, sid) is None                # the other text is still being written
    execute_job(second, ctx)
    assert latest(repo, sid, subj["Pz"]).output["items"][0]["text"] == TIP["Pz"]
    assert latest(repo, sid) is None                # queued as a job of its own
    execute_job(repo.claim_next_job({"standard"}), ctx)
    assert latest(repo, sid).output["items"][0]["text"].endswith("動いてみる")
