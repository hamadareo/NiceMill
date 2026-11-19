"""次にためしてみること on step 5 (確認して承認): AppTest on ui/app.py. Synthetic data only."""
from test_wizard_ui_basic import (app, env, no_internal_terms, ok, screen_text,  # noqa: F401 (env)
                                  video_assignment)
from test_wizard_ui_review_pdf import btn, has_button, reviewable, subject

from pfb_app.domain.models import Stage
from pfb_app.services import review, tips
from pfb_app.services.review import ApprovalError
from tips_helpers import save

TIP = {"Pa": "紙を折って柱を作る動きを、ほかの場面でも続けてみる",
       "Pb": "土台を押さえるときに、声をかけてみる",
       "Pc": "高さを測ったあと、次の手を仲間に伝えてみる"}
TEAM = "土台を支える動きを、みんなで意識してみる"


def tips_for(repo, sid, who, text=None, source="model"):
    s = subject(repo, sid, who)
    pfb = repo.latest_run(sid, Stage.pfb, subject_id=s["id"])
    return save(repo, sid, Stage.tips, {"tips_status": "generated", "pfb_run_id": pfb.id, "items": [
        {"text": text or TIP[who], "evidence_refs": pfb.output["evidence_refs"]}]},
        subject_id=s["id"], upstream=pfb.upstream_run_ids, source=source)


def work(env_, with_tips=("Pa", "Pb", "Pc")):
    settings, repo = env_
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    for who in with_tips:
        tips_for(repo, sid, who)
    return repo, aid, sid


def box(at, key_part):
    return next(t for t in at.text_area if key_part in (t.key or ""))


def test_the_tips_are_shown_under_the_feedback_and_open(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    pa = subject(repo, sid, "Pa")
    area = box(at, f"pfb_tips_box_{pa['id']}_")
    assert area.value == TIP["Pa"]
    page = screen_text(at)
    assert "次にためしてみること" in page
    assert "承認と一緒に確定します" in page
    no_internal_terms(at)


def test_approving_confirms_the_tips_on_screen_as_they_are(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "承認する").click().run()
    ok(at)
    pa = subject(repo, sid, "Pa")
    assert review.subject_status(repo, pa["id"]).value == "approved"
    assert tips.for_pdf(repo, sid, pa["id"]) == [TIP["Pa"]]
    assert tips.for_pdf(repo, sid, subject(repo, sid, "Pb")["id"]) == []


def test_changed_tips_are_confirmed_with_the_approval(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    pa = subject(repo, sid, "Pa")
    box(at, f"pfb_tips_box_{pa['id']}_").set_value("・柱を折る動きを続けてみる\n塔の高さを見てみる").run()
    ok(at)
    btn(at, "承認する").click().run()
    ok(at)
    assert tips.for_pdf(repo, sid, pa["id"]) == ["柱を折る動きを続けてみる", "塔の高さを見てみる"]
    assert repo.latest_run(sid, Stage.tips, subject_id=pa["id"]).source == "teacher"


def test_an_emptied_box_means_no_tips_in_the_pdf(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    pa = subject(repo, sid, "Pa")
    box(at, f"pfb_tips_box_{pa['id']}_").set_value("").run()
    btn(at, "承認する").click().run()
    ok(at)
    assert review.subject_status(repo, pa["id"]).value == "approved"
    assert tips.for_pdf(repo, sid, pa["id"]) == []
    assert tips.view(repo, sid, pa["id"]).confirmed and tips.view(repo, sid, pa["id"]).items == []


def test_four_lines_refuse_the_approval_and_say_why(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    pa = subject(repo, sid, "Pa")
    box(at, f"pfb_tips_box_{pa['id']}_").set_value("あ\nい\nう\nえ").run()
    btn(at, "承認する").click().run()
    ok(at)
    assert review.subject_status(repo, pa["id"]).value != "approved"
    assert "3個まで" in screen_text(at)


def test_confirm_button_and_the_drop_button_after_approval(env):
    repo, aid, sid = work(env)
    pa = subject(repo, sid, "Pa")
    review.approve(repo, pa["id"])
    at = ok(app(session={"assignment_id": aid}))
    at.radio(key=f"pfb_student_{sid}").set_value(pa["id"]).run()
    ok(at)
    assert "このヒントがPDFに載ります" in screen_text(at)
    area = box(at, f"pfb_tips_box_{pa['id']}_")
    area.set_value("声をかけながら動いてみる").run()
    btn(at, "ヒントを確定する").click().run()
    ok(at)
    assert tips.for_pdf(repo, sid, pa["id"]) == ["声をかけながら動いてみる"]
    btn(at, "ヒントを渡さない").click().run()
    ok(at)
    assert tips.for_pdf(repo, sid, pa["id"]) == []
    assert "ヒントはPDFに載りません" in screen_text(at)


def test_wording_faults_are_shown_in_plain_words_and_never_refuse(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    pa = subject(repo, sid, "Pa")
    box(at, f"pfb_tips_box_{pa['id']}_").set_value("花子さんは上手に折る").run()
    ok(at)
    page = screen_text(at)
    assert "人の名前が入っています" in page and "人を評価する言葉" in page
    btn(at, "ヒントを確定する").click().run()
    ok(at)
    assert tips.view(repo, sid, pa["id"]).confirmed


def test_a_person_without_tips_gets_a_closed_empty_section_to_write_in(env):
    repo, aid, sid = work(env, with_tips=("Pb", "Pc"))
    at = ok(app(session={"assignment_id": aid}))
    pa = subject(repo, sid, "Pa")
    area = box(at, f"pfb_tips_box_{pa['id']}_")
    assert area.value == ""
    assert "この参加者のヒントは作られませんでした" in screen_text(at)
    assert btn(at, "ヒントを確定する").disabled
    area.set_value("折る動きを続けてみる").run()
    btn(at, "ヒントを確定する").click().run()
    ok(at)
    assert tips.view(repo, sid, pa["id"]).items == ["折る動きを続けてみる"]


def test_outdated_tips_say_so_and_are_not_shown(env):
    repo, aid, sid = work(env)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    repo.save_stage_run(con.model_copy(update={"id": None, "version": 0}))   # analysis redone
    at = ok(app(session={"assignment_id": aid}))
    assert not any("pfb_tips_box" in (t.key or "") for t in at.text_area)


def test_group_tips_are_confirmed_by_their_own_button(env):
    repo, aid, sid = work(env)
    cons = repo.latest_run(sid, Stage.contribution, "cam1")
    save(repo, sid, Stage.tips, {"tips_status": "generated", "items": [
        {"text": TEAM, "evidence_refs": []}]}, upstream=[cons.id])
    at = ok(app(session={"assignment_id": aid}))
    area = box(at, f"pfb_tips_box_team{sid}_")
    assert area.value == TEAM
    assert "みんなでためしてみること" in screen_text(at)
    assert "このヒントで確定する" in screen_text(at) or has_button(at, "このヒントで確定する")
    assert tips.team_for_pdf(repo, sid) == []
    btn(at, "このヒントで確定する").click().run()
    ok(at)
    assert tips.team_for_pdf(repo, sid) == [TEAM]
    btn(at, "チームのヒントは渡さない").click().run()
    ok(at)
    assert tips.team_for_pdf(repo, sid) == []


def test_a_sports_work_calls_the_group_tips_team(env):
    settings, repo = env
    aid = repo.create_assignment("練習試合", "video", "5対5", "", "standard", True,
                                 domain="sports", sport="basketball")
    sid = reviewable(settings, repo, aid)
    cons = repo.latest_run(sid, Stage.contribution, "cam1")
    save(repo, sid, Stage.tips, {"tips_status": "generated", "items": [
        {"text": TEAM, "evidence_refs": []}]}, upstream=[cons.id])
    at = ok(app(session={"assignment_id": aid}))
    assert "チームでためしてみること" in screen_text(at)
    assert "みんなでためしてみること" not in screen_text(at)
    assert "この選手のヒントは作られませんでした" in screen_text(at)   # localised
