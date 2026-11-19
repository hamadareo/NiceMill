"""Less approval work: 「全員まとめて承認」 on step 5 and the work option 「確認せずに自動で承認する」.

Only synthetic data: invented names and texts, tiny generated images."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline.runner import execute_job
from pfb_app.services import review
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.wizard import step_review
from test_runner import doc_submission, make_ctx, responder
from test_wizard_ui_basic import UI, app, env, html, ok, primaries  # noqa: F401 (env)
from test_wizard_ui_basic import no_internal_terms, screen_text, video_assignment
from test_wizard_ui_people_wait import worker_alive
from test_wizard_ui_review_pdf import (EVENTS, TEXTS, btn, documents, has_button, reviewable,
                                       save_pfb, shown_student, subject)

BULK = "全員まとめて承認"
AUTO = "確認せずに自動で承認する"


def shown(repo, sid):
    """{subject id: latest feedback run id}, as step 5 would have shown them."""
    return {s["id"]: repo.latest_run(sid, Stage.pfb, subject_id=s["id"]).id
            for s in repo.list_subjects(sid)}


def approved(repo, sid):
    return {s["participant_id"] for s in repo.list_subjects(sid)
            if review.subject_status(repo, s["id"]).value == "approved"}


# --- the service ----------------------------------------------------------------------------

def test_bulk_approve_takes_every_clean_text_and_says_why_it_skips_the_rest(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
             ["色の語が含まれています: 赤"], [EVENTS["Pa"][0]])
    result = review.bulk_approve(repo, shown(repo, sid))
    assert approved(repo, sid) == {"Pb"}
    assert result.approved == [subject(repo, sid, "Pb")["id"]]
    reasons = dict(result.skipped)
    assert reasons[hanako["id"]] == review.SKIP_WARNINGS
    assert reasons[subject(repo, sid, "Pc")["id"]] == review.SKIP_INSUFFICIENT
    entry = repo.latest_audit("approve", f"subject:{subject(repo, sid, 'Pb')['id']}")
    assert entry is not None
    assert repo.latest_audit("bulk_approve", None)["detail"]["approved"] == result.approved


def test_bulk_approve_never_approves_a_text_the_screen_did_not_show(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    seen = shown(repo, sid)
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "花子さんは別の場面でも活躍しました。", refs=[EVENTS["Pa"][0]])
    result = review.bulk_approve(repo, seen)
    assert approved(repo, sid) == {"Pb", "Pc"}
    assert dict(result.skipped)[hanako["id"]] == review.SKIP_NEWER


def test_bulk_approve_keeps_the_gates(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, untranslated=True)
    result = review.bulk_approve(repo, shown(repo, sid))
    assert approved(repo, sid) == set() and result.approved == []
    assert {r for _, r in result.skipped} == {review.SKIP_ENGLISH}
    repo.set_identity_confirmed(sid, False)
    result = review.bulk_approve(repo, shown(repo, sid))
    assert {r for _, r in result.skipped} == {review.SKIP_IDENTITY}


def test_bulk_approve_skips_a_text_being_written_again(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    repo.enqueue(sid, Stage.pfb, "standard", subject_id=hanako["id"])
    result = review.bulk_approve(repo, shown(repo, sid))
    assert dict(result.skipped)[hanako["id"]] == review.SKIP_BUSY
    assert "Pa" not in approved(repo, sid)


def test_auto_approve_is_off_by_default_and_audited_when_set(env):
    settings, repo = env
    aid = video_assignment(repo)
    assert repo.get_assignment(aid)["auto_approve"] is False
    review.set_auto_approve(repo, aid, True)
    assert repo.get_assignment(aid)["auto_approve"] is True
    assert repo.latest_audit("set_auto_approve", f"assignment:{aid}")["detail"] == {"on": True}


def test_auto_approve_takes_a_new_clean_text_only_when_the_option_is_on(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    run = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"])
    assert review.auto_approve(repo, run) is False                 # off by default
    review.set_auto_approve(repo, aid, True)
    assert review.auto_approve(repo, run) is True
    assert review.subject_status(repo, hanako["id"]).value == "approved"
    assert review.approved_automatically(repo, repo.get_subject(hanako["id"]))
    entry = repo.latest_audit("auto_approved", f"subject:{hanako['id']}")
    assert entry["detail"] == {"run_id": run.id}


def test_auto_approve_leaves_warnings_gates_and_thin_evidence_to_a_person(env):
    settings, repo = env
    aid = video_assignment(repo)
    review.set_auto_approve(repo, aid, True)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    hanako, taro, jiro = (subject(repo, sid, p) for p in ("Pa", "Pb", "Pc"))
    warned = save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
                      ["色の語が含まれています: 赤"], [EVENTS["Pa"][0]])
    assert review.auto_approve(repo, warned) is False
    thin = repo.latest_run(sid, Stage.pfb, subject_id=jiro["id"])
    assert review.auto_approve(repo, thin) is False
    old = repo.latest_run(sid, Stage.pfb, subject_id=taro["id"])
    save_pfb(repo, sid, taro["id"], "太郎さんは土台を支えました。", refs=[EVENTS["Pb"][0]])
    assert review.auto_approve(repo, old) is False                  # not the latest text
    repo.set_identity_confirmed(sid, False)
    latest = repo.latest_run(sid, Stage.pfb, subject_id=taro["id"])
    assert review.auto_approve(repo, latest) is False               # the gate still applies
    assert approved(repo, sid) == set()
    assert repo.latest_audit("auto_approved", f"subject:{taro['id']}") is None


def test_a_persons_approval_is_not_shown_as_automatic(env):
    settings, repo = env
    aid = video_assignment(repo)
    review.set_auto_approve(repo, aid, True)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    review.auto_approve(repo, repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"]))
    review.unapprove(repo, hanako["id"])
    assert not review.approved_automatically(repo, repo.get_subject(hanako["id"]))
    review.approve(repo, hanako["id"])
    assert not review.approved_automatically(repo, repo.get_subject(hanako["id"]))


# --- the worker -----------------------------------------------------------------------------

@pytest.fixture
def runner_settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


@pytest.fixture
def runner_repo(runner_settings):
    conn = connect(files.db_path(runner_settings))
    init_schema(conn)
    return Repo(conn, runner_settings)


def _run_chain(repo, ctx, sid):
    repo.enqueue(sid, Stage.prepare, "cpu")
    for keys in ({"cpu"}, {"standard"}, {"standard"}, {"standard"}):
        job = repo.claim_next_job(keys)
        execute_job(job, ctx)
        assert repo.get_job(job.id).status == JobStatus.done


def test_the_worker_approves_a_clean_new_text_when_the_option_is_on(runner_settings, runner_repo,
                                                                    tmp_path):
    settings, repo = runner_settings, runner_repo
    aid, sid, subj = doc_submission(repo, tmp_path)
    review.set_auto_approve(repo, aid, True)
    _run_chain(repo, make_ctx(settings, repo), sid)
    run = repo.latest_run(sid, Stage.pfb, subject_id=subj)
    assert repo.get_subject(subj)["approved_run_id"] == run.id
    assert review.approved_automatically(repo, repo.get_subject(subj))
    assert repo.latest_audit("auto_approved", f"subject:{subj}")["detail"] == {"run_id": run.id}


def test_the_worker_leaves_a_text_to_a_person_while_people_are_not_confirmed(env):
    """The worker path (execute_job) with the identity gate unconfirmed: nothing approved."""
    settings, repo = env
    aid = video_assignment(repo)
    review.set_auto_approve(repo, aid, True)
    sid = reviewable(settings, repo, aid)
    repo.set_identity_confirmed(sid, False)
    hanako = subject(repo, sid, "Pa")
    before = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"]).id
    repo.enqueue(sid, Stage.pfb, "standard", subject_id=hanako["id"])
    job = repo.claim_next_job({"standard"})
    execute_job(job, make_ctx(settings, repo))
    assert repo.get_job(job.id).status == JobStatus.done
    run = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"])
    assert run.id != before and run.output["pfb_status"] == "generated" and not run.warnings
    assert repo.get_subject(hanako["id"])["approval_status"] == "none"
    assert repo.latest_audit("auto_approved", f"subject:{hanako['id']}") is None
    repo.set_identity_confirmed(sid, True)                    # the same path, gate open
    repo.enqueue(sid, Stage.pfb, "standard", subject_id=hanako["id"])
    execute_job(repo.claim_next_job({"standard"}), make_ctx(settings, repo))
    assert review.approved_automatically(repo, repo.get_subject(hanako["id"]))


def test_the_worker_leaves_texts_alone_when_the_option_is_off(runner_settings, runner_repo,
                                                              tmp_path):
    settings, repo = runner_settings, runner_repo
    aid, sid, subj = doc_submission(repo, tmp_path)
    _run_chain(repo, make_ctx(settings, repo), sid)
    assert repo.get_subject(subj)["approval_status"] == "none"


def test_the_worker_leaves_a_text_with_warnings_to_a_person(runner_settings, runner_repo,
                                                            tmp_path):
    settings, repo = runner_settings, runner_repo
    aid, sid, subj = doc_submission(repo, tmp_path)
    review.set_auto_approve(repo, aid, True)

    def warned(req):
        if req.json_schema.get("title") == "PfbOutput":
            text = responder(req).replace("調べたことが", "赤い服の人が調べたことが")
            return text
        return responder(req)
    _run_chain(repo, make_ctx(settings, repo, backend=FakeBackend(warned)), sid)
    run = repo.latest_run(sid, Stage.pfb, subject_id=subj)
    assert run.warnings                                             # the check found something
    assert repo.get_subject(subj)["approval_status"] == "none"


# --- step 5 ---------------------------------------------------------------------------------

def test_bulk_is_the_one_primary_and_approves_everyone_clean(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == [BULK]
    assert has_button(at, "承認する")                                 # per student, kept
    assert "3人" in screen_text(at)
    btn(at, BULK).click().run()
    ok(at)
    assert approved(repo, sid) == {"Pa", "Pb", "Pc"}
    page = html(at)
    assert "承認 3 / 全 3" in page and "3人を承認しました" in page
    assert not has_button(at, BULK)
    assert len(primaries(at)) == 1
    no_internal_terms(at)


def test_bulk_lists_whom_it_skipped_with_plain_reasons(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
             ["色の語が含まれています: 赤"], [EVENTS["Pa"][0]])
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == [BULK]
    btn(at, BULK).click().run()
    ok(at)
    assert approved(repo, sid) == {"Pb", "Pc"}
    page = html(at)
    assert "2人を承認しました" in page
    assert "花子さん" in page and review.SKIP_WARNINGS in page
    assert "色の語" not in page
    assert shown_student(at) == hanako["id"]                          # the one left to check
    no_internal_terms(at)


def test_bulk_skips_a_text_that_changed_after_it_was_shown(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    save_pfb(repo, sid, hanako["id"], "花子さんは別の場面でも活躍しました。", refs=[EVENTS["Pa"][0]])
    btn(at, BULK).click().run()
    ok(at)
    assert approved(repo, sid) == {"Pb", "Pc"}
    assert review.SKIP_NEWER in html(at)


def test_with_one_clean_text_the_student_button_stays_the_primary(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pb", "Pc"))
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == ["承認する"]
    assert not has_button(at, BULK)


def test_while_editing_the_editor_keeps_the_primary(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid},
                extra={step_review.EDITING: subject(repo, sid, "Pa")["id"]}))
    assert primaries(at) == ["直して承認"]
    assert not has_button(at, BULK)          # never approves the text an editor holds


# --- regression tests from the review (findings 1-9) ------------------------------------------

def test_bulk_while_an_editor_is_open_never_approves_the_unedited_text(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}, extra={step_review.EDITING: hanako["id"]}))
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("直している途中の文章").run()
    ok(at)
    assert not has_button(at, BULK)
    assert repo.get_subject(hanako["id"])["approved_run_id"] is None


def test_bulk_leaves_a_student_with_a_kept_draft_and_keeps_the_draft(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako, taro = subject(repo, sid, "Pa"), subject(repo, sid, "Pb")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "直す").click().run()
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("直している途中の文章").run()
    at.radio(key=f"pfb_student_{sid}").set_value(taro["id"]).run()
    ok(at)
    btn(at, BULK).click().run()
    ok(at)
    assert approved(repo, sid) == {"Pb", "Pc"}
    assert step_review.SKIP_EDITING in html(at)
    at.radio(key=f"pfb_student_{sid}").set_value(hanako["id"]).run()
    ok(at)
    assert at.text_area(key=f"pfb_edit_{hanako['id']}").value == "直している途中の文章"


def test_bulk_never_approves_a_text_that_was_not_on_screen(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    worker_alive(settings)
    at = ok(app(session={"assignment_id": aid}))
    assert shown_student(at) == hanako["id"]
    btn(at, "この人だけ作り直す").click().run()
    ok(at)
    assert TEXTS["Pa"] not in html(at)                         # 解析中: the old text is hidden
    repo.conn.execute("UPDATE jobs SET status='failed_retryable' WHERE stage='pfb'")
    btn(at, BULK).click().run()
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value != "approved"
    assert approved(repo, sid) == {"Pb", "Pc"}
    assert "花子さん" in html(at) and review.SKIP_BUSY in html(at)


def test_bulk_checks_and_writes_in_one_transaction(env, monkeypatch):
    """A newer text saved between the check and the write is never approved: the latest
    run is read again inside the write transaction."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    seen = shown(repo, sid)
    real = review._being_written
    calls = []

    def sneak(r, subj):
        if subj["id"] == hanako["id"] and not calls:
            calls.append(1)
            assert r.conn.in_transaction             # the check runs inside the transaction
        return real(r, subj)
    monkeypatch.setattr(review, "_being_written", sneak)
    review.bulk_approve(repo, seen)
    assert calls and approved(repo, sid) == {"Pa", "Pb", "Pc"}
    seen = shown(repo, sid)
    review.unapprove(repo, hanako["id"])
    save_pfb(repo, sid, hanako["id"], "花子さんは別の場面でも活躍しました。", refs=[EVENTS["Pa"][0]])
    result = review.bulk_approve(repo, seen)
    assert dict(result.skipped)[hanako["id"]] == review.SKIP_NEWER


def test_a_persons_approval_is_never_relabelled_automatic(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.set_auto_approve(repo, aid, True)
    hanako = subject(repo, sid, "Pa")
    run = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"])
    review.approve(repo, hanako["id"])           # a person, just after the job finished
    assert review.auto_approve(repo, run) is False
    assert not review.approved_automatically(repo, repo.get_subject(hanako["id"]))
    assert repo.latest_audit("auto_approved", f"subject:{hanako['id']}") is None


def test_an_outdated_automatic_approval_is_not_shown_as_one(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.set_auto_approve(repo, aid, True)
    hanako = subject(repo, sid, "Pa")
    review.auto_approve(repo, repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"]))
    save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
             ["色の語が含まれています: 赤"], [EVENTS["Pa"][0]])
    assert review.subject_status(repo, hanako["id"]).value != "approved"
    assert not review.approved_automatically(repo, repo.get_subject(hanako["id"]))


def test_auto_approve_waits_while_the_text_or_its_analysis_is_running(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.set_auto_approve(repo, aid, True)
    hanako, taro = subject(repo, sid, "Pa"), subject(repo, sid, "Pb")
    run = repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"])
    repo.enqueue(sid, Stage.pfb, "standard", subject_id=hanako["id"])     # a redo queued
    assert review.auto_approve(repo, run) is False
    repo.conn.execute("UPDATE jobs SET status='done'")
    repo.enqueue(sid, Stage.behavior, "standard", camera="cam1")          # analysis again
    assert review.auto_approve(repo, repo.latest_run(sid, Stage.pfb,
                                                     subject_id=taro["id"])) is False
    assert approved(repo, sid) == set()


def test_a_second_press_reports_the_same_approvals(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    seen = shown(repo, sid)
    first = review.bulk_approve(repo, seen)
    second = review.bulk_approve(repo, seen)
    assert sorted(second.approved) == sorted(first.approved) and len(first.approved) == 3
    assert second.skipped == []


def test_a_gate_that_appears_late_is_said_in_plain_words(env, monkeypatch):
    """Reasons come only from the plain SKIP_* lines, never from an internal message."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    seen = shown(repo, sid)
    monkeypatch.setattr(review, "untranslated_behavior", lambda r, s: {"cam1": 2})
    result = review.bulk_approve(repo, seen)
    reasons = {why for _, why in result.skipped}
    assert reasons == {review.SKIP_ENGLISH}
    assert all("cam1" not in why and "PFB" not in why and "行動の観察" not in why
               for why in reasons)


def test_bulk_says_to_wait_while_the_analysis_runs(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    seen = shown(repo, sid)
    repo.enqueue(sid, Stage.behavior, "standard", camera="cam1")
    result = review.bulk_approve(repo, seen)
    assert {why for _, why in result.skipped} == {review.SKIP_ANALYSIS}
    assert review.SKIP_ANALYSIS == "解析が終わるまでお待ちください。"


def test_documents_get_the_bulk_button_too(env):
    settings, repo = env
    aid = documents(settings, repo)
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == [BULK]
    btn(at, BULK).click().run()
    ok(at)
    assert all(s["approval_status"] == "approved" for sub in repo.list_submissions(aid)
               for s in repo.list_subjects(sub["id"]))


def test_the_option_can_be_set_on_step_5(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    toggle = next(t for t in at.toggle if t.label == AUTO)
    assert toggle.value is False
    toggle.set_value(True).run()
    ok(at)
    assert repo.get_assignment(aid)["auto_approve"] is True
    assert next(t for t in at.toggle if t.label == AUTO).value is True


def test_step_5_shows_who_was_approved_automatically(env):
    settings, repo = env
    aid = video_assignment(repo)
    review.set_auto_approve(repo, aid, True)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    review.auto_approve(repo, repo.latest_run(sid, Stage.pfb, subject_id=hanako["id"]))
    at = ok(app(session={"assignment_id": aid}))
    page = html(at)
    assert "承認 1 / 全 3" in page
    assert "pfb-chip" in page and "自動で承認" in page and "花子さん" in page
    radio = at.radio(key=f"pfb_student_{sid}")
    assert any("自動で承認" in o for o in radio.options)
    radio.set_value(hanako["id"]).run()
    ok(at)
    assert 'class="pfb-stamp"' in html(at) and "自動で承認" in html(at)
    assert len(primaries(at)) == 1


def test_the_option_can_be_set_on_the_detail_page(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid)
    at = AppTest.from_file(str(UI / "pages" / "review.py"), default_timeout=60).run()
    at.selectbox[0].select(aid).run()
    ok(at)
    toggle = next(t for t in at.toggle if t.label == AUTO)
    assert toggle.value is False
    toggle.set_value(True).run()
    ok(at)
    assert repo.get_assignment(aid)["auto_approve"] is True

