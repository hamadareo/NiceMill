"""Who may approve in bulk from the home screen: the real sign-in gate (AppTest).

A 見るだけの人 sees no way to approve; an 評価する人's approval is audited under their login, and a
session that stopped being valid between the two presses approves nothing."""
import pytest

from pfb_app.services import auth, review
from pfb_app.ui.wizard import waiting_panel as panel
from test_accounts_ui import PW, app as gated_app
from test_wizard_ui_basic import env, html, ok, video_assignment  # noqa: F401 (env)
from test_wizard_ui_review_pdf import reviewable, subject

pytestmark = pytest.mark.real_auth


def sign_in(env, role, login):
    settings, repo = env
    user = auth.create_user(repo, login, login, role, PW)
    return user, auth.start_session(repo, settings, user)


def approved(repo, sid):
    return {s["participant_id"] for s in repo.list_subjects(sid)
            if review.subject_status(repo, s["id"]).value == "approved"}


def test_a_viewer_is_offered_no_bulk_approval(env):
    settings, repo = env
    sid = reviewable(settings, repo, video_assignment(repo, "県大会"))
    sign_in(env, "admin", "kanri")
    _, cookie = sign_in(env, "viewer", "miru")
    at = ok(gated_app(token=cookie))
    assert "県大会" in html(at)
    assert not [b for b in at.button if b.key and b.key.startswith("pfb_wait_")]
    assert approved(repo, sid) == set()


def test_an_evaluators_bulk_approval_is_audited_under_their_login(env):
    settings, repo = env
    sid = reviewable(settings, repo, video_assignment(repo, "県大会"))
    sign_in(env, "admin", "kanri")
    _, cookie = sign_in(env, "evaluator", "tasuke")
    at = ok(gated_app(token=cookie))
    at = ok(at.button(key="pfb_wait_toggle").click().run())
    for pid in ("Pa", "Pb"):
        at.checkbox(key=panel.PICK.format(sid=subject(repo, sid, pid)["id"])).check()
    at = ok(at.run())
    at = ok(at.button(key="pfb_wait_ask").click().run())
    at = ok(at.button(key="pfb_wait_go").click().run())
    assert approved(repo, sid) == {"Pa", "Pb"}
    hanako = subject(repo, sid, "Pa")["id"]
    assert repo.latest_audit("approve", f"subject:{hanako}")["actor"] == "tasuke"
    assert repo.latest_audit("bulk_approve", None)["actor"] == "tasuke"


def test_a_session_that_ended_between_the_presses_approves_nothing(env):
    settings, repo = env
    sid = reviewable(settings, repo, video_assignment(repo, "県大会"))
    admin, _ = sign_in(env, "admin", "kanri")
    user, cookie = sign_in(env, "evaluator", "tasuke")
    at = ok(gated_app(token=cookie))
    at = ok(at.button(key="pfb_wait_toggle").click().run())
    at.checkbox(key=panel.PICK.format(sid=subject(repo, sid, "Pa")["id"])).check()
    at = ok(at.run())
    at = ok(at.button(key="pfb_wait_ask").click().run())
    auth.set_role(repo, user.id, "viewer")                  # demoted before the second press
    at.button(key="pfb_wait_go").click().run()
    assert approved(repo, sid) == set()
