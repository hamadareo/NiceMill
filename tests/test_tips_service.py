"""次にためしてみること: what the coach sees and confirms (services.tips) and how an approval
carries the tips along."""
from pathlib import Path

import pytest
from tips_helpers import TIP, ctx_for, group, save, team_json, tips_json

from pfb_app.config import Settings
from pfb_app.domain.models import Stage
from pfb_app.pipeline import tips as worker
from pfb_app.services import review
from pfb_app.services import tips
from pfb_app.services.review import BusyError
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


@pytest.fixture
def made(settings, repo):
    """A group whose two people have feedback and the model's tips."""
    aid, sid, subj, pfb = group(repo)
    for pid in ("Pa", "Pz"):
        ctx, _ = ctx_for(settings, repo, lambda r, pid=pid: tips_json(pid))
        worker.run_person_tips(ctx, sid, subj[pid], pfb[pid])
    return aid, sid, subj, pfb


def test_view_shows_the_models_tips_unconfirmed(repo, made):
    _, sid, subj, _ = made
    v = tips.view(repo, sid, subj["Pa"])
    assert v.items == [TIP["Pa"]] and not v.confirmed and not v.stale and v.run_id
    assert tips.view(repo, sid, None).items == [] and tips.view(repo, sid, None).run is None
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []          # nothing is printed unconfirmed


def test_confirm_saves_the_lines_as_the_coachs_newest_run(repo, made):
    _, sid, subj, _ = made
    shown = tips.view(repo, sid, subj["Pa"])
    run = tips.confirm(repo, sid, subj["Pa"], "・" + TIP["Pa"] + "\n\n味方の声に応えてみる", shown.run_id)
    assert run.source == "teacher" and run.output["confirmed"] is True and run.model_id is None
    assert run.output["items"] == [{"text": TIP["Pa"], "evidence_refs": ["cam1-0001"]},
                                   {"text": "味方の声に応えてみる", "evidence_refs": []}]
    v = tips.view(repo, sid, subj["Pa"])
    assert v.confirmed and v.items == [TIP["Pa"], "味方の声に応えてみる"]
    assert tips.for_pdf(repo, sid, subj["Pa"]) == v.items
    assert repo.conn.execute("SELECT detail FROM audit_log WHERE action=?",
                             (tips.AUDIT,)).fetchone() is not None


def test_an_emptied_box_deletes_the_tips_for_good(repo, made, settings):
    _, sid, subj, pfb = made
    shown = tips.view(repo, sid, subj["Pa"])
    run = tips.confirm(repo, sid, subj["Pa"], "  \n", shown.run_id)
    assert run.output["tips_status"] == "none" and run.output["items"] == []
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []
    assert tips.view(repo, sid, subj["Pa"]).confirmed      # a decision, not a missing text


def test_nothing_shown_and_nothing_written_saves_nothing(repo):
    _, sid, subj, _ = group(repo)
    assert tips.confirm(repo, sid, subj["Pa"], "", None) is None
    assert repo.latest_run(sid, Stage.tips, subject_id=subj["Pa"]) is None
    run = tips.confirm(repo, sid, subj["Pa"], "自分で書いたヒントをためしてみる", None)
    assert run.output["items"] == [{"text": "自分で書いたヒントをためしてみる",
                                    "evidence_refs": []}]


def test_a_press_on_an_older_screen_is_refused(repo, made, settings):
    _, sid, subj, pfb = made
    shown = tips.view(repo, sid, subj["Pa"])
    tips.confirm(repo, sid, subj["Pa"], TIP["Pa"], shown.run_id)
    with pytest.raises(tips.TipsError, match="新しくなりました"):
        tips.confirm(repo, sid, subj["Pa"], "", shown.run_id)


def test_too_many_lines_and_a_busy_person_are_refused(repo, made):
    _, sid, subj, _ = made
    shown = tips.view(repo, sid, subj["Pa"])
    with pytest.raises(tips.TipsError, match="3個まで"):
        tips.confirm(repo, sid, subj["Pa"], "a\nb\nc\nd", shown.run_id)
    repo.enqueue(sid, Stage.pfb, "standard", None, subj["Pa"])
    with pytest.raises(BusyError):
        tips.confirm(repo, sid, subj["Pa"], TIP["Pa"], shown.run_id)
    # another person's text being written does not stop this person
    other = tips.view(repo, sid, subj["Pz"])
    assert tips.confirm(repo, sid, subj["Pz"], TIP["Pz"], other.run_id) is not None


def test_the_analysis_redone_makes_the_tips_outdated(repo, made):
    _, sid, subj, _ = made
    shown = tips.view(repo, sid, subj["Pa"])
    tips.confirm(repo, sid, subj["Pa"], TIP["Pa"], shown.run_id)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    repo.save_stage_run(con.model_copy(update={"id": None, "version": 0}))
    v = tips.view(repo, sid, subj["Pa"])
    assert v.stale and v.items == [] and not v.confirmed
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []


def test_wording_advises_in_plain_words(repo, made):
    _, sid, subj, _ = made
    found = tips.wording(repo, sid, ["山田花子さんにパスを出す", TIP["Pa"]])
    assert any("名前" in f for f in found) and any("終わり" in f for f in found)
    assert tips.wording(repo, sid, [TIP["Pa"]]) == []


# --- an approval carries the tips ---------------------------------------------------------------

def test_approving_confirms_the_models_tips(repo, made):
    _, sid, subj, _ = made
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == [TIP["Pa"]]
    assert tips.for_pdf(repo, sid, subj["Pz"]) == []       # not approved: untouched


def test_the_coachs_own_decision_is_never_overwritten_by_an_approval(repo, made):
    _, sid, subj, _ = made
    shown = tips.view(repo, sid, subj["Pa"])
    tips.confirm(repo, sid, subj["Pa"], "", shown.run_id)       # deleted
    review.approve(repo, subj["Pa"])
    assert tips.for_pdf(repo, sid, subj["Pa"]) == []
    assert tips.view(repo, sid, subj["Pa"]).items == []


def test_tips_with_a_wording_fault_wait_for_the_coach(repo, made, settings):
    _, sid, subj, pfb = made
    bad = save(repo, sid, Stage.tips, {"tips_status": "generated", "items": [
        {"text": "山田花子さんと声をかける", "evidence_refs": ["cam1-0001"]}]},
        subject_id=subj["Pa"], upstream=pfb["Pa"].upstream_run_ids)
    review.approve(repo, subj["Pa"])
    v = tips.view(repo, sid, subj["Pa"])
    assert v.run_id == bad.id and not v.confirmed and tips.for_pdf(repo, sid, subj["Pa"]) == []


def test_bulk_and_automatic_approval_confirm_clean_tips_too(repo, made):
    _, sid, subj, pfb = made
    result = review.bulk_approve(repo, {subj["Pa"]: pfb["Pa"].id, subj["Pz"]: pfb["Pz"].id})
    assert sorted(result.approved) == sorted(subj.values())
    assert tips.for_pdf(repo, sid, subj["Pa"]) == [TIP["Pa"]]
    assert tips.for_pdf(repo, sid, subj["Pz"]) == [TIP["Pz"]]


@pytest.mark.usefixtures("only_old_gates")
def test_the_workers_automatic_approval_does_not_confirm_the_tips(repo, made):
    aid, sid, subj, pfb = made
    review.set_auto_approve(repo, aid, True)
    assert review.auto_approve(repo, pfb["Pz"])
    assert tips.for_pdf(repo, sid, subj["Pz"]) == []     # a person confirms tips, a machine does not


def test_a_failure_in_the_tips_never_fails_an_approval(repo, made, monkeypatch):
    _, sid, subj, _ = made

    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(tips, "wording", boom)
    review.approve(repo, subj["Pa"])
    assert repo.get_subject(subj["Pa"])["approval_status"] == "approved"


# --- the group ---------------------------------------------------------------------------------

def test_group_tips_are_confirmed_by_the_coach_only(repo, made, settings):
    _, sid, subj, _ = made
    ctx, _ = ctx_for(settings, repo, lambda r: team_json())
    worker.run_team_tips(ctx, sid)
    review.approve(repo, subj["Pa"])
    assert tips.team_for_pdf(repo, sid) == []
    shown = tips.view(repo, sid, None)
    assert shown.items
    tips.confirm(repo, sid, None, "\n".join(shown.items), shown.run_id)
    assert tips.team_for_pdf(repo, sid) == shown.items


def test_team_title_follows_the_kind_of_work():
    assert tips.team_title("sports") == "チームでためしてみること"
    assert tips.team_title("school") == tips.team_title(None) == "みんなでためしてみること"
