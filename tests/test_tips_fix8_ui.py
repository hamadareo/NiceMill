"""次にためしてみること on step 5, review fixes of rel8 (AppTest; synthetic data only)."""
from test_tips_ui import TEAM, TIP, box, tips_for, work
from test_wizard_ui_basic import (app, env, ok, screen_text)  # noqa: F401 (env)
from test_wizard_ui_review_pdf import btn, has_button, subject
from tips_helpers import save

from pfb_app.domain.models import Stage
from pfb_app.services import review, tips


def team_tips(repo, sid, text=TEAM):
    cons = repo.latest_run(sid, Stage.contribution, "cam1")
    pfbs = [repo.latest_run(sid, Stage.pfb, subject_id=s["id"]).id for s in repo.list_subjects(sid)]
    return save(repo, sid, Stage.tips, {"tips_status": "generated", "items": [
        {"text": text, "evidence_refs": []}]}, upstream=[cons.id, *pfbs])


# --- 「承認する」 confirms the box only when the coach changed it, and only clean wording --------

def test_approving_does_not_confirm_tips_with_a_wording_fault(env):
    repo, aid, sid = work(env, with_tips=("Pb", "Pc"))
    pa = subject(repo, sid, "Pa")
    tips_for(repo, sid, "Pa", text="柱を折る動きを続ける")           # no suggestion's ending
    at = ok(app(session={"assignment_id": aid}))
    btn(at, "承認する").click().run()
    ok(at)
    assert review.subject_status(repo, pa["id"]).value == "approved"
    assert tips.for_pdf(repo, sid, pa["id"]) == []
    assert not tips.view(repo, sid, pa["id"]).confirmed
    assert tips.view(repo, sid, pa["id"]).items == ["柱を折る動きを続ける"]
    at.radio(key=f"pfb_student_{sid}").set_value(pa["id"]).run()    # back to the approved person
    ok(at)
    assert "自動では確定しません" in screen_text(at)


def test_approving_does_not_confirm_a_changed_box_with_a_fault_either(env):
    repo, aid, sid = work(env)
    pa = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    box(at, f"pfb_tips_box_{pa['id']}_").set_value("花子さんは上手に折る").run()
    btn(at, "承認する").click().run()
    ok(at)
    assert review.subject_status(repo, pa["id"]).value == "approved"
    assert tips.for_pdf(repo, sid, pa["id"]) == []
    now = tips.view(repo, sid, pa["id"])                              # the coach's words are kept
    assert now.items == ["花子さんは上手に折る"] and not now.confirmed
    assert TIP["Pa"] not in tips.for_pdf(repo, sid, pa["id"])


def test_a_clean_box_the_coach_did_not_touch_is_left_to_the_approval(env):
    repo, aid, sid = work(env)
    pa = subject(repo, sid, "Pa")
    at = ok(app(session={"assignment_id": aid}))
    before = len(repo.run_history(sid, Stage.tips, None, pa["id"]))
    btn(at, "承認する").click().run()
    ok(at)
    assert tips.for_pdf(repo, sid, pa["id"]) == [TIP["Pa"]]
    assert len(repo.run_history(sid, Stage.tips, None, pa["id"])) == before + 1   # one confirmation


def test_a_hand_written_text_says_the_tips_are_not_confirmed_by_the_approval(env):
    repo, aid, sid = work(env)
    pa = subject(repo, sid, "Pa")
    review.write_manual_pfb(repo, pa["id"], "花子さんは最後まで柱を支えました。")
    at = ok(app(session={"assignment_id": aid}))
    assert "文章を書き直したため" in screen_text(at)
    assert "承認と一緒に確定します" not in screen_text(at)
    btn(at, "承認する").click().run()
    ok(at)
    assert tips.for_pdf(repo, sid, pa["id"]) == []
    at.radio(key=f"pfb_student_{sid}").set_value(pa["id"]).run()    # back to the approved person
    ok(at)
    assert "自動では確定しません" in screen_text(at)


# --- the group's tips: what they stand on, and asking for them ---------------------------------

def test_the_group_tips_say_that_unapproved_texts_are_material(env):
    repo, aid, sid = work(env)
    team_tips(repo, sid)
    at = ok(app(session={"assignment_id": aid}))
    assert "まだ承認していない人の文章も材料です" in screen_text(at)
    for who in ("Pa", "Pb", "Pc"):
        review.approve(repo, subject(repo, sid, who)["id"])
    at = ok(app(session={"assignment_id": aid}))
    assert "まだ承認していない人の文章も材料です" not in screen_text(at)


def test_group_tips_outdated_by_a_text_say_so_and_can_be_asked_again(env):
    repo, aid, sid = work(env)
    team_tips(repo, sid)
    review.write_manual_pfb(repo, subject(repo, sid, "Pb")["id"], "土台を支え続けました。")
    at = ok(app(session={"assignment_id": aid}))
    assert "古くなりました" in screen_text(at)
    assert not any(f"pfb_tips_box_team{sid}_" in (t.key or "") for t in at.text_area)
    btn(at, "チームのヒントを作り直す").click().run()
    ok(at)
    assert repo.has_active_job(sid, Stage.tips, None, None)
    assert "作っています" in screen_text(at)
    assert not has_button(at, "チームのヒントを作り直す")


def test_a_group_without_tips_can_ask_for_them(env):
    repo, aid, sid = work(env)
    at = ok(app(session={"assignment_id": aid}))
    assert has_button(at, "チームのヒントを作る")
    btn(at, "チームのヒントを作る").click().run()
    ok(at)
    assert repo.has_active_job(sid, Stage.tips, None, None)
    assert "作っています" in screen_text(at)


def test_no_button_while_a_text_is_still_being_written(env):
    repo, aid, sid = work(env)
    repo.enqueue(sid, Stage.pfb, "standard", None, subject(repo, sid, "Pa")["id"])
    at = ok(app(session={"assignment_id": aid}))
    assert not has_button(at, "チームのヒントを作る")
