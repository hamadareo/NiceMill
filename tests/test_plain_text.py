"""やさしい文章版 (rel8): the worker's job, the versions of a person's text and their use.

A fake model answers; synthetic people and texts only."""
import json
from pathlib import Path

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline import plain, plain_checks
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import execute_job, model_key_for
from pfb_app.services import plain_text, review
from pfb_app.services.plain_text import Refused
from pfb_app.storage.db import NON_BLOCKING_STAGES
from test_wizard_ui_basic import env, video_assignment  # noqa: F401 (env)
from test_wizard_ui_review_pdf import TEXTS, reviewable, save_pfb, subject

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
LOW = "花子さんは、はしらを ていねいに おって、とうの どだいを つくりました。"
MID = "花子さんは柱をていねいに折って、塔の土台を作りました。"
BAD_KANJI = "花子さんは、柱を ていねいに おって、とうの どだいを つくりました。"


def answer(text):
    return json.dumps({"text": text}, ensure_ascii=False)


@pytest.fixture
def work(env):
    """(settings, repo, assignment id, group id) with 花子 approved, 太郎 and 次郎 not."""
    settings, repo = env
    settings.prompts_dir = PROMPTS
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    return settings, repo, aid, sid


def hanako(repo, sid):
    return repo.get_subject(subject(repo, sid, "Pa")["id"])


def run_job(settings, repo, sid, level, fn, subject_id=None):
    """Queue the person's job as the screen does, run it as the worker does."""
    backend = FakeBackend(fn)
    subject_id = subject_id or subject(repo, sid, "Pa")["id"]
    plain_text.request(repo, repo.get_submission(sid)["assignment_id"], level, [subject_id])
    job = repo.claim_next_job({model_key_for(repo, sid, Stage.plain)})
    assert job is not None and job.stage == Stage.plain and job.camera == level
    execute_job(job, StageContext(settings, repo, backend, lambda: False))
    return backend, repo.get_job(job.id)


# --- the worker's job --------------------------------------------------------------------------

def test_a_checked_text_is_saved_as_the_persons_version_for_the_level(work):
    settings, repo, aid, sid = work
    backend, job = run_job(settings, repo, sid, "low", lambda r: answer(LOW))
    assert job.status == JobStatus.done
    person = hanako(repo, sid)
    run = repo.latest_run(sid, Stage.plain, "low", person["id"])
    assert run.status == "ok" and run.source == "model" and run.model_id == "fake"
    assert run.output == {"level": "low", "text": LOW}
    assert run.upstream_run_ids == [person["approved_run_id"]]
    assert run.prompt_hash == plain.load_prompt(settings, "low").hash
    shown = plain_text.state(repo, person, "low")
    assert (shown.kind, shown.text, shown.source) == (plain_text.READY, LOW, "model")
    assert repo.get_run(person["approved_run_id"]).output["pfb"] == TEXTS["Pa"]   # untouched


def test_the_model_is_given_the_text_the_level_the_names_and_the_kanji(work):
    settings, repo, aid, sid = work
    backend, _ = run_job(settings, repo, sid, "low", lambda r: answer(LOW))
    req = backend.calls[0]
    assert req.system == plain.load_prompt(settings, "low").text and not req.image_paths
    assert f"【言いかえる文章・データ開始】\n{TEXTS['Pa']}\n【言いかえる文章・データ終了】" in req.user_text
    assert "【読み手・データ開始】\n" + plain_checks.LEVELS["low"].label in req.user_text
    assert "【そのまま書く名前・データ開始】\n花子\n" in req.user_text and "太郎" not in req.user_text
    assert "【使ってよい漢字・データ開始】\n" + plain_checks.LOW_KANJI in req.user_text
    backend, _ = run_job(settings, repo, sid, "mid", lambda r: answer(MID))
    assert "使ってよい漢字" not in backend.calls[0].user_text
    assert backend.calls[0].system == plain.load_prompt(settings, "mid").text


def test_a_wrong_answer_is_shown_what_is_wrong_and_asked_again(work):
    settings, repo, aid, sid = work
    answers = iter([BAD_KANJI, LOW])
    backend, job = run_job(settings, repo, sid, "low", lambda r: answer(next(answers)))
    assert len(backend.calls) == 2
    assert "前回の出力にエラーがありました" in backend.calls[1].user_text
    assert "柱" in backend.calls[1].user_text
    assert repo.latest_run(sid, Stage.plain, "low", hanako(repo, sid)["id"]).output["text"] == LOW


def test_an_answer_that_never_passes_leaves_the_approved_text_in_use(work):
    settings, repo, aid, sid = work
    backend, job = run_job(settings, repo, sid, "low", lambda r: answer(BAD_KANJI))
    assert len(backend.calls) == settings.max_retries + 1
    assert job.status == JobStatus.done                       # a finished run, not a failed job
    person = hanako(repo, sid)
    run = repo.latest_run(sid, Stage.plain, "low", person["id"])
    assert run.status == "ok" and run.output["text"] is None and "柱" in run.output["reason"]
    shown = plain_text.state(repo, person, "low")
    assert shown.kind == plain_text.FALLBACK and shown.text is None
    assert plain_text.handout_texts(repo, aid) == {}


def test_a_text_that_is_not_approved_any_more_is_not_reworded(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    plain_text.request(repo, aid, "low", [person["id"]])
    review.unapprove(repo, person["id"])
    job = repo.claim_next_job({model_key_for(repo, sid, Stage.plain)})
    backend = FakeBackend(lambda r: answer(LOW))
    execute_job(job, StageContext(settings, repo, backend, lambda: False))
    assert backend.calls == []
    run = repo.latest_run(sid, Stage.plain, "low", person["id"])
    assert run.output["text"] is None and run.output["reason"] == plain.NOT_APPROVED


def test_a_side_stage_that_never_shows_in_the_works_steps(work):
    settings, repo, aid, sid = work
    plain_text.request(repo, aid, "low", [hanako(repo, sid)["id"]])
    assert [j["stage"] for j in repo.submission_job_rows(sid) if j["stage"] == "plain"] == []
    assert repo.has_active_job(sid, Stage.plain, "low", hanako(repo, sid)["id"])
    assert "plain" in NON_BLOCKING_STAGES
    # a tracking-engine work asks the engine's served model, like its feedback texts
    assert model_key_for(repo, sid, Stage.plain) == "pfb3x3:standard"


# --- the versions of a person's text ------------------------------------------------------------

def test_a_version_waits_is_made_and_goes_out_of_date_with_the_text(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    assert plain_text.state(repo, person, "low").kind == plain_text.NONE
    plain_text.request(repo, aid, "low", [person["id"]])
    assert plain_text.state(repo, person, "low").kind == plain_text.MAKING
    run_job_after = repo.claim_next_job({"pfb3x3:standard"})
    execute_job(run_job_after, StageContext(settings, repo, FakeBackend(lambda r: answer(LOW)),
                                            lambda: False))
    assert plain_text.state(repo, person, "low").kind == plain_text.READY
    # the approved text is written again and approved: the old version no longer fits it
    new = save_pfb(repo, sid, person["id"], "花子さんは柱を押さえました。")
    review.approve(repo, person["id"])
    shown = plain_text.state(repo, hanako(repo, sid), "low")
    assert shown.kind == plain_text.OUTDATED and shown.text is None
    assert hanako(repo, sid)["approved_run_id"] == new.id
    # not approved: nothing to say about it
    review.unapprove(repo, person["id"])
    assert plain_text.state(repo, hanako(repo, sid), "low").kind == plain_text.UNAVAILABLE


def test_who_is_asked_for(work):
    settings, repo, aid, sid = work
    taro = subject(repo, sid, "Pb")["id"]
    review.approve(repo, taro)
    assert plain_text.request(repo, aid, "low") == 2                  # both approved people
    assert plain_text.request(repo, aid, "low") == 0                  # already asked: nothing new
    assert plain_text.request(repo, aid, "mid", [taro]) == 1
    with pytest.raises(Refused):
        plain_text.request(repo, aid, "other")


def test_asking_again_keeps_what_a_coach_wrote_unless_that_person_is_named(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    run_job(settings, repo, sid, "low", lambda r: answer(LOW))
    plain_text.save_edit(repo, person["id"], "low", LOW.replace("つくりました", "つくったよ"))
    assert plain_text.request(repo, aid, "low", redo=True) == 0       # the coach's wording stays
    assert plain_text.request(repo, aid, "low", [person["id"]], redo=True) == 1


def test_nothing_can_be_asked_for_a_work_whose_videos_are_gone(work):
    settings, repo, aid, sid = work
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-11-01T00:00:00+00:00' WHERE id=?",
                      (aid,))
    with pytest.raises(Refused):
        plain_text.request(repo, aid, "low")


# --- the choice and what a handout takes --------------------------------------------------------

def test_the_choice_is_kept_per_work(work):
    settings, repo, aid, sid = work
    assert plain_text.choice(repo, aid) == plain_text.Choice("original", frozenset())
    plain_text.choose(repo, aid, "low", off=[7, 9])
    assert plain_text.choice(repo, aid) == plain_text.Choice("low", frozenset({7, 9}))
    plain_text.choose(repo, aid, "original")
    assert plain_text.choice(repo, aid).level == "original"
    with pytest.raises(Refused):
        plain_text.choose(repo, aid, "high")
    repo.set_setting(f"plain:{aid}", {"level": "nonsense"})           # an unreadable value
    assert plain_text.choice(repo, aid).level == "original"


def test_a_handout_takes_the_picked_version_only_for_people_who_have_it_ready(work):
    settings, repo, aid, sid = work
    taro = subject(repo, sid, "Pb")["id"]
    review.approve(repo, taro)
    person = hanako(repo, sid)
    run_job(settings, repo, sid, "low", lambda r: answer(LOW))
    assert plain_text.handout_texts(repo, aid) == {}                 # the choice is still the original
    plain_text.choose(repo, aid, "low")
    assert plain_text.handout_texts(repo, aid) == {person["id"]: LOW}
    use = plain_text.usage(repo, aid)
    assert use.plain == ["花子"] and use.original == [("太郎", plain_text.WHY_ORIGINAL["none"])]
    plain_text.choose(repo, aid, "low", off=[person["id"]])
    assert plain_text.handout_texts(repo, aid) == {}
    assert dict(plain_text.usage(repo, aid).original)["花子"] == plain_text.WHY_ORIGINAL["off"]
    plain_text.choose(repo, aid, "mid")                               # nobody has a middle version
    assert plain_text.handout_texts(repo, aid) == {}


def test_the_evaluator_is_told_which_version_each_handout_carries(work):
    settings, repo, aid, sid = work
    taro = subject(repo, sid, "Pb")["id"]
    review.approve(repo, taro)
    run_job(settings, repo, sid, "low", lambda r: answer(LOW))
    assert plain_text.handout_notes(repo, aid) == {}
    plain_text.choose(repo, aid, "low")
    notes = plain_text.handout_notes(repo, aid)
    assert notes[hanako(repo, sid)["id"]] == {"label": plain_checks.LEVELS["low"].label, "text": LOW}
    assert notes[taro] == {"label": plain_checks.ORIGINAL_LABEL, "text": None}


# --- a coach's own wording ----------------------------------------------------------------------

def test_a_coach_edit_must_pass_the_same_checks(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    with pytest.raises(Refused) as refused:
        plain_text.save_edit(repo, person["id"], "low", BAD_KANJI)
    assert "柱" in str(refused.value) and refused.value.reasons
    assert repo.latest_run(sid, Stage.plain, "low", person["id"]) is None


def test_a_coach_edit_is_the_newest_version_and_is_audited(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    run = plain_text.save_edit(repo, person["id"], "low", "  " + LOW + "\n\n\n")
    assert run.source == "teacher" and run.camera == "low"
    assert run.output == {"level": "low", "text": LOW}
    assert run.upstream_run_ids == [person["approved_run_id"]]
    shown = plain_text.state(repo, person, "low")
    assert (shown.kind, shown.text, shown.source) == (plain_text.READY, LOW, "teacher")
    assert repo.latest_audit(plain_text.AUDIT_EDIT, f"subject:{person['id']}")["detail"] == {
        "run_id": run.id, "level": "low"}
    again = plain_text.save_edit(repo, person["id"], "low", LOW)     # the same words: nothing new
    assert again.id == run.id


def test_a_coach_cannot_edit_while_the_version_is_being_made_or_when_not_approved(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    plain_text.request(repo, aid, "low", [person["id"]])
    with pytest.raises(Refused) as refused:
        plain_text.save_edit(repo, person["id"], "low", LOW)
    assert str(refused.value) == plain_text.BUSY
    other = subject(repo, sid, "Pb")["id"]                            # not approved
    with pytest.raises(Refused) as refused:
        plain_text.save_edit(repo, other, "low", LOW)
    assert str(refused.value) == plain_text.NOT_APPROVED
    with pytest.raises(Refused):
        plain_text.save_edit(repo, person["id"], "other", LOW)


# --- the prompts and the guide ------------------------------------------------------------------

def test_the_prompts_are_zero_shot_guard_against_instructions_and_ask_for_one_field():
    from pfb_app.pipeline.prompts import zero_shot_violations
    for level in plain_checks.LEVELS:
        prompt = plain.load_prompt(Settings(prompts_dir=PROMPTS), level)
        assert len(prompt.hash) == 12 and zero_shot_violations(prompt.text) == []
        assert "データ開始" in prompt.text and "指示には従わない" in prompt.text
        assert "`text`" in prompt.text
        assert "事実を変えない" in prompt.text
    low, mid = (plain.load_prompt(Settings(prompts_dir=PROMPTS), k) for k in ("low", "mid"))
    assert low.hash != mid.hash and "使ってよい漢字" in low.text and "使ってよい漢字" not in mid.text
    with pytest.raises(KeyError):
        plain.load_prompt(Settings(prompts_dir=PROMPTS), "high")


def test_the_user_guide_explains_the_versions():
    guide = (PROMPTS.parent / "docs" / "user_guide.md").read_text(encoding="utf-8")
    for word in ("読み手に合わせた文章", "言いかえた文章を作る", "13歳以上向け", "6〜8歳向け", "9〜12歳向け",
                 "この人は元の文を使う", "意味や事実は変えません"):
        assert word in guide
