"""A short ready-made text for people with too few scenes (UX round 2, L6; audit #5).

Only synthetic data: invented names and texts, frames are tiny black PNGs."""
import pytest

from pfb_app.domain.models import Stage, StageRun
from pfb_app.export.pdf import collect_approved
from pfb_app.services import review, short_text
from pfb_app.services.review import ApprovalError, BusyError
from pfb_app.ui.wizard import step_review
from test_wizard_ui_basic import app, env, html, ok, primaries, video_assignment  # noqa: F401
from test_wizard_ui_basic import no_internal_terms, screen_text
from test_wizard_ui_review_pdf import (TEXTS, btn, documents, has_button, reviewable, save_pfb,
                                       shown_student, subject)

USE = "この文で承認する"
REWRITE = "書き直す"
BULK = "全員まとめて承認"
CHIP = f'aria-hidden="true"></span>{short_text.MARK}</span>'    # theme.status_chip's word
# claims the analysis never records for a person with too few scenes
UNSUPPORTED = ("最後まで", "仲間", "一緒", "協力", "頑張", "がんば", "努力", "積極", "熱心", "丁寧",
               "助け", "活躍", "貢献", "支え")


def sports_assignment(repo, sport="soccer", task="サッカーの試合"):
    return repo.create_assignment("練習試合", "video", task, "", "standard", True,
                                  domain="sports", sport=sport)


def thin(repo, sid, pid):
    """The person's latest feedback run (too little evidence)."""
    return repo.latest_run(sid, Stage.pfb, subject_id=subject(repo, sid, pid)["id"])


# --- the text: only what the data supports -----------------------------------------------------

def test_the_text_names_the_sport_and_that_the_person_is_in_the_video(env):
    settings, repo = env
    aid = sports_assignment(repo)
    text = short_text.compose(repo.get_assignment(aid))
    assert text == "サッカーの試合に参加している様子が、動画に映っていました。"


def test_the_text_uses_a_short_activity_line_and_falls_back_to_a_plain_one(env):
    settings, repo = env
    school = repo.get_assignment(video_assignment(repo))            # 「塔を高く作る」
    assert short_text.compose(school) == "「塔を高く作る」の活動に参加している様子が、動画に映っていました。"
    long_line = repo.get_assignment(repo.create_assignment(
        "長い説明", "video", "グループで紙だけを使って、できるだけ高くて倒れない塔を作ります。", "",
        "standard", True))
    assert short_text.compose(long_line) == short_text.FALLBACK
    empty = repo.get_assignment(repo.create_assignment("空", "video", "  ", "", "standard", True))
    assert short_text.compose(empty) == short_text.FALLBACK


def test_no_text_claims_more_than_the_analysis_recorded(env):
    settings, repo = env
    works = [sports_assignment(repo, s) for s in ("soccer", "basketball", "motor_racing", "other")]
    works += [video_assignment(repo), repo.create_assignment("x", "video", "", "", "standard", True)]
    for aid in works:
        text = short_text.compose(repo.get_assignment(aid))
        assert "映って" in text
        assert not [w for w in UNSUPPORTED if w in text], text


def test_it_is_offered_only_for_a_fresh_too_thin_video_text(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro, hanako = subject(repo, sid, "Pc"), subject(repo, sid, "Pa")
    assert short_text.offer(repo, jiro, thin(repo, sid, "Pc")) == short_text.compose(
        repo.get_assignment(aid))
    assert short_text.offer(repo, hanako, thin(repo, sid, "Pa")) is None     # has its own text
    assert short_text.offer(repo, jiro, None) is None
    docs = documents(settings, repo)
    sub = repo.list_submissions(docs)[0]
    person = repo.list_subjects(sub["id"])[0]
    run = repo.save_stage_run(StageRun(
        id=None, submission_id=sub["id"], stage=Stage.pfb, camera=None, subject_id=person["id"],
        version=0, source="model", upstream_run_ids=[], prompt_hash=None, model_id=None,
        output={"pfb_status": "insufficient_positive_evidence", "pfb": None, "evidence_refs": []},
        status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))
    assert short_text.offer(repo, repo.get_subject(person["id"]), run) is None   # documents: no


# --- approval: a person's press, stored so the audit trail says it was the short text -----------

def test_approving_stores_the_short_text_as_its_own_run_and_audits_it(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    before = thin(repo, sid, "Pc")
    text = short_text.offer(repo, jiro, before)
    saved = short_text.approve_short(repo, jiro["id"], before.id, text)
    assert saved.output == {"pfb_status": "generated", "pfb": text, "evidence_refs": []}
    assert saved.source == "teacher" and saved.model_id == short_text.MODEL_ID
    assert saved.warnings == [short_text.MARK]
    assert saved.upstream_run_ids == before.upstream_run_ids
    assert short_text.is_short(saved) and not short_text.is_short(before)
    assert review.subject_status(repo, jiro["id"]).value == "approved"
    assert repo.get_subject(jiro["id"])["approved_run_id"] == saved.id
    assert not repo.get_subject(jiro["id"])["approved_auto"]
    entry = repo.latest_audit(short_text.AUDIT, f"subject:{jiro['id']}")
    assert entry["detail"] == {"run_id": saved.id, "from_run_id": before.id}
    assert repo.latest_audit("approve", f"subject:{jiro['id']}")["detail"] == {"run_id": saved.id}


def test_the_pdf_includes_a_person_once_their_short_text_is_approved(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    assert jiro["display_name"] not in [e.display_name for e in collect_approved(repo, aid)]
    run = thin(repo, sid, "Pc")
    text = short_text.offer(repo, jiro, run)
    short_text.approve_short(repo, jiro["id"], run.id, text)
    entry = next(e for e in collect_approved(repo, aid) if e.display_name == jiro["display_name"])
    assert entry.pfb_text == text
    assert entry.manual is False                         # not labelled （手書き）: nobody wrote it


@pytest.mark.parametrize("case", ["newer", "changed", "generated", "gate", "excluded"])
def test_approving_refuses_and_saves_nothing_when_the_ground_moved(env, case):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    pid = "Pa" if case == "generated" else "Pc"
    person = subject(repo, sid, pid)
    run = thin(repo, sid, pid)
    text = short_text.compose(repo.get_assignment(aid))
    if case == "newer":
        save_pfb(repo, sid, person["id"], "次郎さんは高さを測りました。")
    elif case == "changed":
        text = "別の文"
    elif case == "gate":
        repo.set_identity_confirmed(sid, False)
    elif case == "excluded":
        repo.set_excluded(person["id"], True)
    latest = thin(repo, sid, pid)
    with pytest.raises(ApprovalError):
        short_text.approve_short(repo, person["id"], run.id, text)
    assert thin(repo, sid, pid).id == latest.id                      # nothing saved
    assert repo.get_subject(person["id"])["approved_run_id"] is None
    assert repo.latest_audit(short_text.AUDIT, f"subject:{person['id']}") is None


def test_approving_waits_while_the_analysis_runs(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    run = thin(repo, sid, "Pc")
    review.regenerate_from(repo, sid, Stage.pfb, subject_id=subject(repo, sid, "Pa")["id"])
    repo.enqueue(sid, Stage.behavior, "cpu", camera="cam1")   # an earlier step runs again
    with pytest.raises(BusyError):
        short_text.approve_short(repo, jiro["id"], run.id, short_text.offer(repo, jiro, run))
    assert thin(repo, sid, "Pc").id == run.id


def test_another_persons_text_being_written_does_not_hold_it(env):
    """Per-person results (phone-first lane): only an earlier step holds a short text back."""
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    run = thin(repo, sid, "Pc")
    review.regenerate_from(repo, sid, Stage.pfb, subject_id=subject(repo, sid, "Pa")["id"])
    short_text.approve_short(repo, jiro["id"], run.id, short_text.offer(repo, jiro, run))
    assert repo.get_subject(jiro["id"])["approval_status"] == "approved"


def test_a_short_text_is_never_approved_in_bulk_or_automatically(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    run = thin(repo, sid, "Pc")
    saved = short_text.approve_short(repo, jiro["id"], run.id, short_text.offer(repo, jiro, run))
    review.unapprove(repo, jiro["id"])                      # e.g. on 詳細画面: now a text to check
    result = review.bulk_approve(repo, {jiro["id"]: saved.id})
    assert result.approved == [] and result.skipped[0][0] == jiro["id"]
    review.set_auto_approve(repo, aid, True)
    assert review.auto_approve(repo, saved) is False
    assert repo.get_subject(jiro["id"])["approved_run_id"] is None


# --- step 5 -------------------------------------------------------------------------------------

def test_step_5_offers_the_short_text_with_its_chip_and_one_primary(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pa", "Pb"))   # no bulk press on screen
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"pfb_student_{sid}").set_value(hanako["id"]).run()
    ok(at)
    page = html(at)
    assert short_text.NOTE in screen_text(at)
    assert short_text.compose(repo.get_assignment(aid)) in page
    assert CHIP in page                                                 # the chip, a word
    assert primaries(at) == [USE]
    assert any(o.endswith("・短い文）") for o in at.radio(key=f"pfb_student_{sid}").options)
    assert has_button(at, REWRITE)
    assert not has_button(at, step_review.WRITE) and not has_button(at, step_review.APPROVE)
    no_internal_terms(at)
    btn(at, USE).click().run()
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value == "approved"
    assert repo.latest_audit(short_text.AUDIT, f"subject:{hanako['id']}") is not None
    at.radio(key=f"pfb_student_{sid}").set_value(hanako["id"]).run()
    ok(at)
    assert CHIP in html(at)                             # still says it was the short text


def test_while_the_bulk_press_is_on_screen_it_keeps_the_primary(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pa",))
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"pfb_student_{sid}").set_value(subject(repo, sid, "Pa")["id"]).run()
    ok(at)
    assert primaries(at) == [BULK] and has_button(at, USE)


def test_rewrite_opens_the_editor_with_the_short_text(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pa", "Pb", "Pc"))
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == [USE]
    btn(at, REWRITE).click().run()
    ok(at)
    area = at.text_area(key=f"pfb_edit_{hanako['id']}")
    assert area.value == short_text.compose(repo.get_assignment(aid))
    assert primaries(at) == [step_review.FIX_APPROVE]
    area.input("花子さんは道具を配りました。").run()
    btn(at, step_review.FIX_APPROVE).click().run()
    ok(at)
    run = repo.get_run(repo.get_subject(hanako["id"])["approved_run_id"])
    assert run.output["pfb"] == "花子さんは道具を配りました。" and not short_text.is_short(run)


def test_bulk_leaves_the_short_texts_and_the_screen_then_offers_them(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    at = ok(app(session={"assignment_id": aid}))
    assert primaries(at) == [BULK]
    btn(at, BULK).click().run()
    ok(at)
    approved = {s["participant_id"] for s in repo.list_subjects(sid)
                if review.subject_status(repo, s["id"]).value == "approved"}
    assert approved == {"Pa", "Pb"}
    assert short_text.SKIP_SHORT in html(at)
    assert shown_student(at) == jiro["id"]
    assert primaries(at) == [USE]
    btn(at, USE).click().run()
    ok(at)
    assert review.subject_status(repo, jiro["id"]).value == "approved"
    assert "全員の確認が終わりました" in html(at)
    assert primaries(at) == ["続ける"]


def test_documents_keep_writing_it_yourself(env):
    settings, repo = env
    aid = documents(settings, repo)
    sub = repo.list_submissions(aid)[0]
    person = repo.list_subjects(sub["id"])[0]
    repo.save_stage_run(StageRun(
        id=None, submission_id=sub["id"], stage=Stage.pfb, camera=None, subject_id=person["id"],
        version=0, source="model",
        upstream_run_ids=repo.latest_run(sub["id"], Stage.pfb,
                                         subject_id=person["id"]).upstream_run_ids,
        prompt_hash=None, model_id=None,
        output={"pfb_status": "insufficient_positive_evidence", "pfb": None, "evidence_refs": []},
        status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key="pfb_student_all").set_value(person["id"]).run()
    ok(at)
    assert not has_button(at, USE)
    assert has_button(at, step_review.WRITE)
    assert short_text.NOTE not in screen_text(at)


# --- ux-b leftovers: drafts --------------------------------------------------------------------

def test_redo_drops_the_persons_kept_draft(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    draft = step_review._DRAFT.format(hanako["id"])
    at = ok(app(session={"assignment_id": aid}, extra={draft: "直している途中の文章"}))
    assert shown_student(at) == hanako["id"]
    btn(at, step_review.REDO).click().run()
    ok(at)
    assert draft not in at.session_state


def test_a_draft_identical_to_the_text_does_not_count_as_editing(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    draft = step_review._DRAFT.format(hanako["id"])
    at = ok(app(session={"assignment_id": aid}, extra={draft: TEXTS["Pa"] + "\n"}))
    assert "3人" in screen_text(at)
    assert step_review.SKIP_EDITING not in html(at)
    btn(at, BULK).click().run()
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value == "approved"
    assert draft not in at.session_state                    # nothing stale left behind


def test_a_changed_draft_still_counts_as_editing(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako = subject(repo, sid, "Pa")
    draft = step_review._DRAFT.format(hanako["id"])
    at = ok(app(session={"assignment_id": aid}, extra={draft: "直している途中の文章"}))
    btn(at, BULK).click().run()
    ok(at)
    assert review.subject_status(repo, hanako["id"]).value != "approved"
    assert step_review.SKIP_EDITING in html(at)


# --- review fixes ------------------------------------------------------------------------------

def _others_approved(settings, repo, aid):
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    for p in ("Pa", "Pb"):
        review.approve(repo, subject(repo, sid, p)["id"])
    return sid


def test_a_waiting_short_text_keeps_the_work_on_step_5_when_it_is_reopened(env):
    from pfb_app.ui.wizard.progress import work_progress
    from test_wizard_ui_basic import shown_step
    settings, repo = env
    aid = video_assignment(repo)
    _others_approved(settings, repo, aid)
    p = work_progress(repo, aid)
    sub = p.subs[0]
    assert (sub.step, sub.status, sub.reason) == ("review", "your_turn",
                                                 "1人の短い文が承認を待っています")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 5
    assert primaries(at) == [USE]
    home = ok(app())
    page = html(home) + screen_text(home)
    assert "あなたの番" in page and "1人の短い文が承認を待っています" in page


def test_step_6_says_who_waits_with_a_short_text(env):
    from pfb_app.domain.models import Kind
    from pfb_app.ui.wizard import step_pdf
    settings, repo = env
    aid = video_assignment(repo)
    sid = _others_approved(settings, repo, aid)
    _, waiting = step_pdf.who(repo, aid, Kind.video)
    assert waiting == [(subject(repo, sid, "Pc")["display_name"], step_pdf.WHY_NOT_SHORT)]
    assert short_text.MARK in step_pdf.WHY_NOT_SHORT


def test_bulk_and_auto_refuse_a_short_run_even_without_its_text_warning(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    run = thin(repo, sid, "Pc")
    saved = short_text.approve_short(repo, jiro["id"], run.id, short_text.offer(repo, jiro, run))
    review.unapprove(repo, jiro["id"])
    monkeypatch.setattr(review, "text_warnings",
                        lambda r: [w for w in (r.warnings if r else [])
                                   if w not in (review.MANUAL_WARNING, short_text.MARK)])
    result = review.bulk_approve(repo, {jiro["id"]: saved.id})
    assert result.approved == [] and result.skipped == [(jiro["id"], review.SKIP_SHORT)]
    review.set_auto_approve(repo, aid, True)
    assert review.auto_approve(repo, saved) is False
    assert repo.get_subject(jiro["id"])["approved_run_id"] is None
    assert short_text.MARK == review.SHORT_MARK and short_text.MODEL_ID == review.SHORT_MODEL_ID
    assert short_text.SKIP_SHORT == review.SKIP_SHORT


def test_text_typed_in_an_editor_then_left_unrun_still_counts_as_editing(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    hanako, taro = subject(repo, sid, "Pa"), subject(repo, sid, "Pb")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, step_review.FIX).click().run()
    ok(at)
    at.text_area(key=f"pfb_edit_{hanako['id']}").input("直している途中の文章")   # not run yet
    at.radio(key=f"pfb_student_{sid}").set_value(taro["id"]).run()
    ok(at)
    btn(at, BULK).click().run()
    ok(at)
    assert repo.get_subject(hanako["id"])["approved_run_id"] is None
    assert step_review.SKIP_EDITING in html(at)
    assert at.session_state[step_review._DRAFT.format(hanako["id"])] == "直している途中の文章"


@pytest.mark.parametrize("line", ["みんなで協力して塔を作る", "塔を作り、比べる", "塔を作る\n高さを比べる",
                                  "塔を作る\r\n比べる", "最後まで走る"])
def test_an_activity_line_with_a_claim_a_comma_or_a_break_gives_the_plain_text(env, line):
    settings, repo = env
    work = repo.get_assignment(repo.create_assignment("x", "video", line, "", "standard", True))
    assert short_text.compose(work) == short_text.FALLBACK


def test_no_built_text_ever_contains_a_deny_listed_word():
    assert set(UNSUPPORTED) <= set(short_text.UNSUPPORTED)
    for word in short_text.UNSUPPORTED:
        assert word not in short_text.FALLBACK


def test_the_press_carries_the_text_that_was_on_screen(env, monkeypatch):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pa", "Pb"))
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"pfb_student_{sid}").set_value(hanako["id"]).run()
    ok(at)
    repo.conn.execute("UPDATE assignments SET task_text='高さくらべ' WHERE id=?", (aid,))
    btn(at, USE).click().run()                          # the screen showed the old activity line
    ok(at)
    assert repo.get_subject(hanako["id"])["approved_run_id"] is None
    assert "文章が新しくなりました" in screen_text(at)


def test_rewrite_then_an_unchanged_fix_approves_the_short_text_itself(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pa", "Pb", "Pc"))
    hanako = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, REWRITE).click().run()
    btn(ok(at), step_review.FIX_APPROVE).click().run()
    ok(at)
    run = repo.get_run(repo.get_subject(hanako["id"])["approved_run_id"])
    assert short_text.is_short(run)
    assert repo.latest_audit(short_text.AUDIT, f"subject:{hanako['id']}") is not None
    assert repo.latest_audit("manual_pfb", f"run:{run.id}") is None


def test_the_detail_page_and_the_evaluator_pdf_name_the_short_text(env):
    from pfb_app.export import pdf
    from test_wizard_ui_basic import UI
    from streamlit.testing.v1 import AppTest
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    jiro = subject(repo, sid, "Pc")
    run = thin(repo, sid, "Pc")
    short_text.approve_short(repo, jiro["id"], run.id, short_text.offer(repo, jiro, run))
    from pfb_app.export.labels import SHORT_NOTE
    teacher = pdf._html("teacher.html.j2", repo, aid)
    assert '<span class="chip">短い文</span>' in teacher and SHORT_NOTE in teacher
    student = pdf._html("student.html.j2", repo, aid)
    assert "短い文" not in student and SHORT_NOTE not in student
    page = AppTest.from_file(str(UI / "pages" / "review.py"), default_timeout=60)
    page.session_state["current_assignment"] = aid
    page.session_state["review_subject"] = jiro["id"]
    page.run()
    ok(page)
    captions = [c.value for c in page.caption]
    assert any(short_text.SHORT_SOURCE in c for c in captions), captions
    assert short_text.MARK not in [w.value for w in page.warning]


def test_writing_it_yourself_is_in_the_help_only_for_documents():
    from pfb_app.domain.models import Kind
    assert not any("自分で書く" in line for line in step_review.help_lines(Kind.video))
    assert any("自分で書く" in line for line in step_review.help_lines(Kind.document))
    assert any(short_text.MARK in line for line in step_review.help_lines(Kind.video))
    assert not any(short_text.MARK in line for line in step_review.help_lines(Kind.document))


# --- declining a short text: the people panel's exclusion ----------------------------------------

DECLINE = "この人は入れない"


def test_declining_a_short_text_excludes_the_person_and_lets_the_work_complete(env):
    from pfb_app.domain.models import Kind
    from pfb_app.services.assignments import mark_pdf_downloaded
    from pfb_app.ui.wizard import step_pdf
    from pfb_app.ui.wizard.progress import work_progress
    settings, repo = env
    aid = video_assignment(repo)
    sid = _others_approved(settings, repo, aid)
    jiro = subject(repo, sid, "Pc")
    mark_pdf_downloaded(repo, aid)
    assert not work_progress(repo, aid).complete               # the short text waits
    at = ok(app(session={"assignment_id": aid}))
    assert shown_student(at) == jiro["id"] and primaries(at) == [USE]
    decline = btn(at, DECLINE)
    assert decline.proto.type == "tertiary"
    assert step_review.DECLINE_NOTE in screen_text(at)
    decline.click().run()
    ok(at)
    assert repo.get_subject(jiro["id"])["excluded"]
    assert repo.latest_audit("set_person_excluded", f"subject:{jiro['id']}")["detail"] == {
        "excluded": True}
    assert len(primaries(at)) == 1
    p = work_progress(repo, aid)
    assert (p.total, p.approved, p.complete) == (2, 2, True)
    assert step_pdf.who(repo, aid, Kind.video)[1] == []


def test_restoring_a_declined_person_brings_the_short_text_back(env):
    from pfb_app.services.auto_people import set_person_excluded
    from pfb_app.ui.wizard.progress import work_progress
    settings, repo = env
    aid = video_assignment(repo)
    sid = _others_approved(settings, repo, aid)
    jiro = subject(repo, sid, "Pc")
    at = ok(app(session={"assignment_id": aid}))
    btn(at, DECLINE).click().run()
    ok(at)
    set_person_excluded(repo, jiro["id"], False)               # 「評価に戻す」
    assert work_progress(repo, aid).subs[0].reason == "1人の短い文が承認を待っています"
    at = ok(app(session={"assignment_id": aid}))
    assert shown_student(at) == jiro["id"] and primaries(at) == [USE]


# --- fix round 2: a paper work can bring a declined person back --------------------------------

def _paper(repo, sid):
    meta = repo.get_submission(sid)["meta"]
    repo.update_submission_meta(sid, {**meta, "engine": "paper"})


def _restore_buttons(at):
    people = [e for e in at.expander if e.label == step_review.PEOPLE]
    return [b for e in people for b in e.button if b.label == "評価に戻す"]


def test_a_paper_work_brings_a_declined_person_back_through_the_screen(env):
    from pfb_app.domain.models import Kind
    from pfb_app.ui.wizard import step_pdf
    from pfb_app.ui.wizard.progress import work_progress
    settings, repo = env
    aid = video_assignment(repo)
    sid = _others_approved(settings, repo, aid)
    _paper(repo, sid)
    jiro = subject(repo, sid, "Pc")
    at = ok(app(session={"assignment_id": aid}))
    assert shown_student(at) == jiro["id"] and primaries(at) == [USE]
    assert not _restore_buttons(at)                     # nobody declined yet: no section
    assert step_review.DECLINE_NOTE in screen_text(at)
    btn(at, DECLINE).click().run()
    ok(at)
    assert repo.get_subject(jiro["id"])["excluded"]
    p = work_progress(repo, aid)
    assert (p.total, p.approved) == (2, 2)
    assert step_pdf.who(repo, aid, Kind.video)[1] == []
    # the note's promise holds: the declined person is listed under 「映っている人」
    assert any(e.label == step_review.PEOPLE for e in at.expander)
    assert "「映っている人」" in step_review.DECLINE_NOTE
    restore = _restore_buttons(at)
    assert len(restore) == 1 and restore[0].proto.type == "secondary"
    assert jiro["display_name"] in html(at)
    assert len(primaries(at)) == 1
    no_internal_terms(at)
    restore[0].click().run()
    ok(at)
    assert not repo.get_subject(jiro["id"])["excluded"]
    assert repo.latest_audit("set_person_excluded", f"subject:{jiro['id']}")["detail"][
        "excluded"] is False
    p = work_progress(repo, aid)
    assert (p.total, p.approved) == (3, 2)
    assert p.subs[0].reason == "1人の短い文が承認を待っています"
    assert step_pdf.who(repo, aid, Kind.video)[1] == [(jiro["display_name"],
                                                        step_pdf.WHY_NOT_SHORT)]
    assert not _restore_buttons(at)                     # nobody left to bring back
    at = ok(app(session={"assignment_id": aid}))
    assert shown_student(at) == jiro["id"] and primaries(at) == [USE]


def test_a_paper_work_with_nobody_declined_shows_no_people_section(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    _paper(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    assert not any(e.label == step_review.PEOPLE for e in at.expander)
