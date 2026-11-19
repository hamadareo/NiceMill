"""次にためしてみること, review fixes of rel8: what an approval confirms (the text it was made
for, a deletion), what the group's tips stand on (each person's text), and the group's tips as a
job of their own (a yield or a full GPU never loses them; the coach can ask for them)."""
from pathlib import Path

import pytest
from tips_helpers import TIP, ctx_for, group, save, team_json, tips_json

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.inference.structured import YieldRequested
from pfb_app.pipeline import runner
from pfb_app.pipeline import tips as worker
from pfb_app.services import review, tips
from pfb_app.storage.db import SIDE_STAGES, Repo, connect, init_schema


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


@pytest.fixture
def repo(tmp_path, settings):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, settings)


@pytest.fixture
def made(settings, repo):
    aid, sid, subj, pfb = group(repo)
    for pid in ("Pa", "Pz"):
        ctx, _ = ctx_for(settings, repo, lambda r, pid=pid: tips_json(pid))
        worker.run_person_tips(ctx, sid, subj[pid], pfb[pid])
    return aid, sid, subj, pfb


def newer_pfb(repo, sid, subj, pfb, who="Pa", source="model"):
    """A newer feedback run of the same person (same analysis)."""
    return save(repo, sid, Stage.pfb, dict(pfb[who].output), subject_id=subj[who],
                upstream=pfb[who].upstream_run_ids, source=source)


# --- an approval confirms only tips made for the text it approves ------------------------------

def test_a_text_written_by_hand_does_not_inherit_the_models_tips(repo, made):
    _, sid, subj, _ = made
    review.write_manual_pfb(repo, subj["Pa"], "ドリブルでボールを運んでいました。")
    now = tips.view(repo, sid, subj["Pa"])
    assert now.items == [TIP["Pa"]] and now.text_changed       # still there to read and confirm
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []
    assert not tips.view(repo, sid, subj["Pa"]).confirmed
    shown = tips.view(repo, sid, subj["Pa"])                    # the coach can confirm them
    tips.confirm(repo, sid, subj["Pa"], "\n".join(shown.items), shown.run_id)
    assert tips.for_pdf(repo, sid, subj["Pa"]) == [TIP["Pa"]]


def test_tips_of_an_older_model_text_are_not_confirmed(repo, made):
    _, sid, subj, pfb = made
    newer_pfb(repo, sid, subj, pfb)                              # the feedback was written again
    assert tips.view(repo, sid, subj["Pa"]).text_changed
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []


def test_tips_of_the_current_text_are_not_marked_as_changed(repo, made):
    _, sid, subj, _ = made
    assert not tips.view(repo, sid, subj["Pa"]).text_changed


def test_new_tips_after_a_deletion_wait_for_the_coach(repo, made, settings):
    _, sid, subj, pfb = made
    shown = tips.view(repo, sid, subj["Pa"])
    tips.confirm(repo, sid, subj["Pa"], "", shown.run_id)       # 「ヒントを渡さない」
    again = newer_pfb(repo, sid, subj, pfb)                      # the person's text redone
    ctx, _ = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    worker.run_person_tips(ctx, sid, subj["Pa"], again)
    now = tips.view(repo, sid, subj["Pa"])
    assert now.items == [TIP["Pa"]] and now.run.source == "model"
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []             # the earlier deletion stands
    assert not tips.view(repo, sid, subj["Pa"]).confirmed
    tips.confirm(repo, sid, subj["Pa"], TIP["Pa"], now.run_id)   # the coach may still confirm
    assert tips.for_pdf(repo, sid, subj["Pa"]) == [TIP["Pa"]]


def test_tips_after_a_confirmation_are_confirmed_again_with_the_approval(repo, made, settings):
    _, sid, subj, pfb = made
    shown = tips.view(repo, sid, subj["Pa"])
    tips.confirm(repo, sid, subj["Pa"], TIP["Pa"], shown.run_id)   # not a deletion
    again = newer_pfb(repo, sid, subj, pfb)
    ctx, _ = ctx_for(settings, repo, lambda r: tips_json("Pa"))
    worker.run_person_tips(ctx, sid, subj["Pa"], again)
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == [TIP["Pa"]]


def test_a_draft_is_kept_but_never_confirmed_by_an_approval(repo, made):
    _, sid, subj, _ = made
    shown = tips.view(repo, sid, subj["Pa"])
    tips.confirm(repo, sid, subj["Pa"], "山田花子さんに声をかける", shown.run_id, draft=True)
    now = tips.view(repo, sid, subj["Pa"])
    assert now.items == ["山田花子さんに声をかける"] and not now.confirmed
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []
    assert not tips.view(repo, sid, subj["Pa"]).confirmed


# --- the group's tips stand on every person's text ---------------------------------------------

def test_group_tips_stand_on_each_persons_feedback(repo, made, settings):
    _, sid, subj, pfb = made
    ctx, _ = ctx_for(settings, repo, lambda r: team_json())
    run = worker.run_team_tips(ctx, sid)
    contribution = repo.latest_run(sid, Stage.contribution, "cam1")
    assert run.upstream_run_ids == [contribution.id, pfb["Pa"].id, pfb["Pz"].id]
    assert not tips.view(repo, sid, None).stale


@pytest.mark.parametrize("how", ["by hand", "again"])
def test_group_tips_go_out_of_date_when_a_text_changes(repo, made, settings, how):
    _, sid, subj, pfb = made
    ctx, _ = ctx_for(settings, repo, lambda r: team_json())
    worker.run_team_tips(ctx, sid)
    shown = tips.view(repo, sid, None)
    tips.confirm(repo, sid, None, "\n".join(shown.items), shown.run_id)
    assert tips.team_for_pdf(repo, sid) == shown.items
    if how == "by hand":
        review.write_manual_pfb(repo, subj["Pz"], "守りで声をかけていました。")
    else:
        newer_pfb(repo, sid, subj, pfb, "Pz")
    assert tips.view(repo, sid, None).stale
    assert tips.team_for_pdf(repo, sid) == []


def test_the_coachs_confirmation_keeps_what_the_group_tips_stand_on(repo, made, settings):
    _, sid, subj, _ = made
    ctx, _ = ctx_for(settings, repo, lambda r: team_json())
    run = worker.run_team_tips(ctx, sid)
    saved = tips.confirm(repo, sid, None, "声をかけながら動いてみる", run.id)
    assert saved.upstream_run_ids == run.upstream_run_ids


def test_group_tips_written_by_hand_stand_on_the_current_texts(repo, made):
    _, sid, subj, pfb = made
    saved = tips.confirm(repo, sid, None, "声をかけながら動いてみる", None)
    assert pfb["Pa"].id in saved.upstream_run_ids and pfb["Pz"].id in saved.upstream_run_ids
    review.write_manual_pfb(repo, subj["Pa"], "ドリブルで運んでいました。")
    assert tips.view(repo, sid, None).stale


# --- the group's tips are a job ----------------------------------------------------------------

def team_job(repo, sid):
    return repo.claim_next_job({repo.get_assignment(
        repo.get_submission(sid)["assignment_id"])["model_key"]})


def test_the_group_job_is_a_side_stage():
    assert Stage.tips.value in SIDE_STAGES


def test_the_last_feedback_queues_the_group_job_and_the_job_makes_the_tips(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: team_json())
    other = repo.enqueue(sid, Stage.pfb, "standard", None, subj["Pz"])
    assert worker.team_after_pfb(ctx, sid) is None               # a text is still being written
    repo.conn.execute("UPDATE jobs SET status='done' WHERE id=?", (other,))
    job_id = worker.team_after_pfb(ctx, sid)
    assert job_id is not None and backend.calls == []             # queued, nothing made yet
    assert worker.team_after_pfb(ctx, sid) is None               # one at a time
    job = team_job(repo, sid)
    assert job.stage == Stage.tips and job.subject_id is None
    runner.execute_job(job, ctx)
    assert repo.get_job(job.id).status == JobStatus.done
    assert repo.latest_run(sid, Stage.tips).output["items"]
    assert worker.team_after_pfb(ctx, sid) is None               # made: kept


def test_a_yield_in_the_group_job_puts_it_back_in_the_queue(settings, repo):
    _, sid, subj, pfb = group(repo)
    ctx, _ = ctx_for(settings, repo, lambda r: team_json(), yield_check=lambda: True)
    worker.team_after_pfb(ctx, sid)
    job = team_job(repo, sid)
    with pytest.raises(YieldRequested):
        runner.execute_job(job, ctx)
    assert repo.get_job(job.id).status == JobStatus.queued
    assert repo.latest_run(sid, Stage.tips) is None
    ctx2, _ = ctx_for(settings, repo, lambda r: team_json())
    runner.execute_job(team_job(repo, sid), ctx2)
    assert repo.latest_run(sid, Stage.tips).output["items"]


def test_a_full_gpu_in_the_group_job_ends_the_session_and_keeps_the_job(settings, repo):
    _, sid, _, _ = group(repo)

    def oom(req):
        raise RuntimeError("CUDA out of memory. Tried to allocate")
    ctx, _ = ctx_for(settings, repo, oom)
    worker.team_after_pfb(ctx, sid)
    job = team_job(repo, sid)
    with pytest.raises(runner.SessionMustEnd):
        runner.execute_job(job, ctx)
    assert repo.get_job(job.id).status == JobStatus.queued      # the automatic retry keeps it
    assert repo.latest_run(sid, Stage.tips) is None


def test_another_failure_in_the_group_job_is_stored_as_no_tips(settings, repo):
    _, sid, _, _ = group(repo)

    def boom(req):
        raise RuntimeError("backend exploded")
    ctx, _ = ctx_for(settings, repo, boom)
    worker.team_after_pfb(ctx, sid)
    job = team_job(repo, sid)
    runner.execute_job(job, ctx)
    assert repo.get_job(job.id).status == JobStatus.done
    run = repo.latest_run(sid, Stage.tips)
    assert run.output["tips_status"] == "none" and "backend exploded" in run.output["reason"]


def test_the_group_job_leaves_what_the_coach_kept(settings, repo):
    _, sid, subj, _ = group(repo)
    ctx, backend = ctx_for(settings, repo, lambda r: pytest.fail("model called"))
    worker.team_after_pfb(ctx, sid)
    job = team_job(repo, sid)
    tips.confirm(repo, sid, None, "声をかけてみる", None)      # the coach wrote them
    runner.execute_job(job, ctx)
    assert repo.get_job(job.id).status == JobStatus.done and backend.calls == []


def test_the_coach_can_ask_for_the_group_tips(settings, repo):
    _, sid, subj, _ = group(repo)
    assert worker.team_wanted(repo, sid)
    assert tips.queue_team(repo, sid) is not None
    assert tips.team_queued(repo, sid)
    assert tips.queue_team(repo, sid) is None                    # already asked
    _, sid2, _, _ = group(repo, label="B班", feedback=False)
    assert not worker.team_wanted(repo, sid2)                    # no texts yet
    assert tips.queue_team(repo, sid2) is None
