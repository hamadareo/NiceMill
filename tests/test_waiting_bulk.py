"""確認待ちの人の一覧とまとめて承認 (services.pending_review and the home screen's panel).

Only synthetic data: invented names and texts, tiny generated images."""
import pytest

from pfb_app.domain.models import Stage
from pfb_app.services import deletion, pending_review, review
from pfb_app.ui.wizard import waiting_panel as panel
from test_wizard_ui_basic import app, env, html, ok, no_internal_terms, screen_text  # noqa: F401
from test_wizard_ui_basic import video_assignment
from test_wizard_ui_review_pdf import EVENTS, TEXTS, reviewable, save_pfb, subject


def approved(repo, sid):
    return {s["participant_id"] for s in repo.list_subjects(sid)
            if review.subject_status(repo, s["id"]).value == "approved"}


def key(repo, sid, pid):
    return panel.PICK.format(sid=subject(repo, sid, pid)["id"])


def button(at, label=None, key=None):
    return at.button(key=key) if key else next(b for b in at.button if b.label == label)


# --- the service ----------------------------------------------------------------------------

def test_waiting_lists_who_waits_and_who_may_go_together(env):
    settings, repo = env
    aid = video_assignment(repo, "県大会")
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
             ["色の語が含まれています: 赤"], [EVENTS["Pa"][0]])
    (work,) = pending_review.waiting(repo, [aid])
    assert work.assignment_id == aid and work.title == "県大会"
    by_name = {p.name: p for p in work.people}
    assert set(by_name) == {"花子", "太郎"}                      # 次郎: too little evidence
    assert by_name["太郎"].ready and by_name["太郎"].text == TEXTS["Pb"]
    assert by_name["太郎"].group == "A班" and by_name["太郎"].run_id is not None
    assert by_name["花子"].why == review.SKIP_WARNINGS and not by_name["花子"].ready
    assert [p.name for p in work.ready] == ["太郎"]
    assert pending_review.waiting(repo, []) == []


def test_waiting_leaves_out_the_approved_the_busy_and_works_being_deleted(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    repo.enqueue(sid, Stage.pfb, "standard", subject_id=subject(repo, sid, "Pb")["id"])
    (work,) = pending_review.waiting(repo, [aid])
    assert [p.name for p in work.people] == ["次郎"]
    assert pending_review.waiting(repo, [aid, 9999]) == [work]      # an unknown work is skipped
    repo.conn.execute("UPDATE jobs SET status='running' WHERE submission_id=?", (sid,))
    with pytest.raises(deletion.StillRunning):
        deletion.delete_work(repo, settings, aid, stop_analysis=True)
    assert pending_review.waiting(repo, [aid]) == []


def test_approve_goes_through_the_review_service_and_its_audit(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    (work,) = pending_review.waiting(repo, [aid])
    picked = {p.subject_id: p.run_id for p in work.ready if p.name in ("花子", "太郎")}
    result = pending_review.approve(repo, picked)
    assert approved(repo, sid) == {"Pa", "Pb"} and sorted(result.approved) == sorted(picked)
    hanako = subject(repo, sid, "Pa")["id"]
    assert repo.latest_audit("approve", f"subject:{hanako}")["detail"]["bulk"] is True
    assert repo.latest_audit("bulk_approve", None)["detail"]["approved"] == result.approved


def test_a_text_that_changed_after_it_was_listed_is_not_approved(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    (work,) = pending_review.waiting(repo, [aid])
    picked = {p.subject_id: p.run_id for p in work.ready}
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "花子さんは別の場面でも活躍しました。", refs=[EVENTS["Pa"][0]])
    result = pending_review.approve(repo, picked)
    assert approved(repo, sid) == {"Pb", "Pc"}
    assert dict(result.skipped) == {hanako["id"]: review.SKIP_NEWER}


def test_undo_is_explained_in_plain_words():
    text = pending_review.UNDO
    assert "直す" in text and "承認" in text and "PDF" in text
    for word in ("audit", "run", "job", "DB"):
        assert word not in text


# --- the home screen ------------------------------------------------------------------------

def open_panel(at):
    return button(at, key="pfb_wait_toggle").click().run()


def test_no_entry_while_nobody_is_at_the_check(env):
    settings, repo = env
    video_assignment(repo, "まだ何も入っていない作業")
    at = ok(app())
    assert not [b for b in at.button if b.key == "pfb_wait_toggle"]


def test_the_list_offers_boxes_only_to_people_who_may_go_together(env):
    settings, repo = env
    aid = video_assignment(repo, "県大会")
    sid = reviewable(settings, repo, aid, insufficient=("Pc",))
    hanako = subject(repo, sid, "Pa")
    save_pfb(repo, sid, hanako["id"], "赤い服の人が柱を作りました。",
             ["色の語が含まれています: 赤"], [EVENTS["Pa"][0]])
    at = ok(app())
    assert button(at, key="pfb_wait_toggle").label == "確認待ちの人をまとめて見る（1件の作業）"
    assert "太郎" not in screen_text(at)                       # closed: nothing listed
    at = ok(open_panel(at))
    page = html(at)
    assert TEXTS["Pb"] in page and "赤い服の人が柱を作りました。" in page
    assert review.SKIP_WARNINGS in page
    boxes = [c.key for c in at.checkbox if c.key and c.key.startswith("pfb_wait_pick_")]
    assert boxes == [key(repo, sid, "Pb")]                         # only the clean text
    assert not at.checkbox(key=boxes[0]).value                     # empty at first
    ask = button(at, key="pfb_wait_ask")
    assert ask.disabled and ask.label == "選んだ0人を承認する"
    assert approved(repo, sid) == set()
    no_internal_terms(at)


def pick(at, repo, sid, *pids):
    for pid in pids:
        at.checkbox(key=key(repo, sid, pid)).check()
    return ok(at.run())


def test_approving_takes_two_presses_and_says_how_to_undo(env):
    settings, repo = env
    aid = video_assignment(repo, "県大会")
    sid = reviewable(settings, repo, aid)
    at = ok(open_panel(ok(app())))
    at = pick(at, repo, sid, "Pa", "Pb")
    ask = button(at, key="pfb_wait_ask")
    assert not ask.disabled and ask.label == "選んだ2人を承認する"
    at = ok(ask.click().run())
    assert approved(repo, sid) == set()                            # the first press approves nothing
    page = html(at)
    assert "次の2人の文章を承認します。よろしいですか。" in page
    assert "県大会: 花子さん" in page and "県大会: 太郎さん" in page and "次郎" not in page
    assert [b.key for b in at.button if b.key in ("pfb_wait_go", "pfb_wait_cancel")] == [
        "pfb_wait_go", "pfb_wait_cancel"]
    at = ok(button(at, key="pfb_wait_go").click().run())
    assert approved(repo, sid) == {"Pa", "Pb"}
    page = html(at)
    assert "2人を承認しました。" in page and pending_review.UNDO in page
    assert repo.latest_audit("bulk_approve", None)["detail"]["approved"] == [
        subject(repo, sid, "Pa")["id"], subject(repo, sid, "Pb")["id"]]
    no_internal_terms(at)
    at = ok(at.run())                                              # the report is said once
    assert "2人を承認しました。" not in html(at)


def test_cancelling_approves_nothing(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = pick(ok(open_panel(ok(app()))), repo, sid, "Pa")
    at = ok(button(at, key="pfb_wait_ask").click().run())
    at = ok(button(at, key="pfb_wait_cancel").click().run())
    assert approved(repo, sid) == set()
    assert panel.ASK not in at.session_state and "文章を承認します" not in html(at)


def test_a_text_that_changed_between_the_presses_is_left_and_said(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = pick(ok(open_panel(ok(app()))), repo, sid, "Pa", "Pb")
    at = ok(button(at, key="pfb_wait_ask").click().run())
    save_pfb(repo, sid, subject(repo, sid, "Pa")["id"], "花子さんは別の場面でも活躍しました。",
             refs=[EVENTS["Pa"][0]])
    at = ok(button(at, key="pfb_wait_go").click().run())
    assert approved(repo, sid) == {"Pb"}
    page = html(at)
    assert "1人を承認しました。" in page and "花子さん（県大会）" not in page
    assert "花子さん（ペーパータワー）: " + review.SKIP_NEWER in page


def test_the_work_can_be_opened_from_the_list(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid)
    at = ok(open_panel(ok(app())))
    at = ok(button(at, key=f"pfb_wait_open_{aid}").click().run())
    assert at.session_state["wizard"]["assignment_id"] == aid


def test_several_works_are_listed_each_with_its_own_people(env):
    settings, repo = env
    first = video_assignment(repo, "第一試合")
    second = video_assignment(repo, "第二試合")
    a = reviewable(settings, repo, first, label="甲班")
    b = reviewable(settings, repo, second, label="乙班")
    at = ok(open_panel(ok(app())))
    assert button(at, key="pfb_wait_toggle").label == "とじる"
    page = html(at)
    assert "第一試合" in page and "第二試合" in page and "（甲班）" in page and "（乙班）" in page
    at = pick(at, repo, a, "Pa")
    at = pick(at, repo, b, "Pc")
    assert button(at, key="pfb_wait_ask").label == "選んだ2人を承認する"
    at = ok(button(at, key="pfb_wait_ask").click().run())
    at = ok(button(at, key="pfb_wait_go").click().run())
    assert approved(repo, a) == {"Pa"} and approved(repo, b) == {"Pc"}


def test_when_everybody_is_approved_the_entry_goes_but_the_report_stays_once(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    at = ok(open_panel(ok(app())))
    at = pick(at, repo, sid, "Pa", "Pb", "Pc")
    at = ok(button(at, key="pfb_wait_ask").click().run())
    at = ok(button(at, key="pfb_wait_go").click().run())
    assert approved(repo, sid) == {"Pa", "Pb", "Pc"}
    assert "3人を承認しました。" in html(at)
    assert not [b for b in at.button if b.key == "pfb_wait_toggle"]


def test_the_list_follows_what_was_searched_for(env):
    settings, repo = env
    first = video_assignment(repo, "県大会")
    second = video_assignment(repo, "朝の練習")
    reviewable(settings, repo, first, label="甲班")
    reviewable(settings, repo, second, label="乙班")
    at = ok(app())
    at = ok(at.text_input(key="pfb_find_text").input("朝の").run())
    assert button(at, key="pfb_wait_toggle").label == "確認待ちの人をまとめて見る（1件の作業）"
    at = ok(open_panel(at))
    page = html(at)
    assert '<p class="pfb-wait__work">朝の練習</p>' in page and "（乙班）" in page
    assert '<p class="pfb-wait__work">県大会</p>' not in page and "（甲班）" not in page
    assert "「県大会」の解析が終わりました" in page      # the band counts every work, found or not
