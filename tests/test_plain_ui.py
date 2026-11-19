"""やさしい文章版 (rel8) on step 6: pick the reader, ask for the texts, read and edit them,
keep the approved text for someone. AppTest on ui/app.py; synthetic data only."""
import json

import pytest

from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline import plain_checks
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import execute_job
from pfb_app.services import plain_text, review
from pfb_app.ui import access
from pfb_app.ui.wizard import plain_view
from pfb_app.ui.wizard.progress import work_progress
from test_plain_text import BAD_KANJI, LOW, hanako, work  # noqa: F401 (work)
from test_wizard_ui_basic import app, env, no_internal_terms, ok, screen_text  # noqa: F401
from test_wizard_ui_review_pdf import TEXTS, subject


def step6(repo, aid):
    return ok(app(session={"assignment_id": aid, "step": len(work_progress(repo, aid).steps),
                           "submission_id": None}))


def worker_makes(settings, repo, text=LOW):
    """The worker runs every queued version job (a fake model answers `text`)."""
    ctx = StageContext(settings, repo, FakeBackend(lambda r: json.dumps({"text": text},
                                                                         ensure_ascii=False)),
                       lambda: False)
    while (job := repo.claim_next_job({"pfb3x3:standard"})) is not None:
        execute_job(job, ctx)


def radio(at, aid):
    return at.radio(key=f"pfb_plain_level_{aid}")


def buttons(at):
    return [b.label for b in at.button]


def test_the_choice_starts_as_the_approved_text_and_offers_nothing_else(work):
    settings, repo, aid, sid = work
    at = step6(repo, aid)
    r = radio(at, aid)
    assert r.value == "original"
    assert list(r.options) == [plain_checks.ORIGINAL_LABEL,
                               plain_checks.LEVELS["low"].label, plain_checks.LEVELS["mid"].label]
    assert plain_view.MAKE not in buttons(at)
    no_internal_terms(at)


def test_picking_a_reader_is_kept_and_offers_to_make_the_texts(work):
    settings, repo, aid, sid = work
    at = step6(repo, aid)
    at = ok(radio(at, aid).set_value("low").run())
    assert plain_text.choice(repo, aid).level == "low"
    assert plain_view.MAKE in buttons(at)
    assert "まだ言いかえた文章がありません" in screen_text(at)
    assert radio(step6(repo, aid), aid).value == "low"                 # still there on a new visit


def test_asking_queues_one_version_per_approved_person_and_shows_the_waiting(work):
    settings, repo, aid, sid = work
    review.approve(repo, subject(repo, sid, "Pb")["id"])
    at = ok(radio(step6(repo, aid), aid).set_value("low").run())
    at = ok(at.button(key=f"pfb_plain_make_{aid}").click().run())
    assert plain_text.making(repo, aid, "low") == 2
    assert "2人の言いかえを作っています" in screen_text(at)
    assert plain_view.MAKE not in buttons(at)


def test_a_made_text_is_shown_beside_the_approved_one_and_goes_into_the_pdf(work):
    settings, repo, aid, sid = work
    at = ok(radio(step6(repo, aid), aid).set_value("low").run())
    ok(at.button(key=f"pfb_plain_make_{aid}").click().run())
    worker_makes(settings, repo)
    at = step6(repo, aid)
    assert "1人の文章を言いかえました" in screen_text(at)
    person = hanako(repo, sid)
    area = next(t for t in at.text_area if t.key.startswith(f"pfb_plain_text_{person['id']}_low_"))
    assert area.value == LOW
    assert TEXTS["Pa"] in [t.value for t in at.get("text")]            # the approved text beside it
    assert "AIが作った文" in screen_text(at)
    assert plain_text.handout_texts(repo, aid) == {person["id"]: LOW}
    assert plain_view.REMAKE in buttons(at)


def test_a_coach_can_write_the_version_in_their_own_words(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    at = ok(radio(step6(repo, aid), aid).set_value("low").run())
    area = next(t for t in at.text_area if t.key.startswith(f"pfb_plain_text_{person['id']}_low_"))
    own = LOW.replace("つくりました", "つくったよ")
    at = ok(area.set_value(own).run())
    at = ok(at.button(key=f"pfb_plain_save_{person['id']}_low").click().run())
    shown = plain_text.state(repo, hanako(repo, sid), "low")
    assert (shown.kind, shown.text, shown.source) == (plain_text.READY, own, "teacher")
    assert plain_view.SAVED in [s.value for s in at.success]
    at = step6(repo, aid)
    assert "手で直した文" in screen_text(at)


def test_a_wording_that_changes_the_facts_is_refused_in_plain_words(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    at = ok(radio(step6(repo, aid), aid).set_value("low").run())
    area = next(t for t in at.text_area if t.key.startswith(f"pfb_plain_text_{person['id']}_low_"))
    at = ok(area.set_value(BAD_KANJI).run())
    at = ok(at.button(key=f"pfb_plain_save_{person['id']}_low").click().run())
    assert plain_text.state(repo, hanako(repo, sid), "low").kind == plain_text.NONE
    warned = " ".join(w.value for w in at.warning)
    assert "この文は使えません" in warned and "柱" in warned and "ひらがな" in warned
    no_internal_terms(at)


def test_keeping_the_approved_text_for_one_person(work):
    settings, repo, aid, sid = work
    person = hanako(repo, sid)
    at = ok(radio(step6(repo, aid), aid).set_value("low").run())
    at = ok(at.checkbox(key=f"pfb_plain_keep_{person['id']}").check().run())
    assert plain_text.choice(repo, aid) == plain_text.Choice("low", frozenset({person["id"]}))
    at = ok(at.checkbox(key=f"pfb_plain_keep_{person['id']}").uncheck().run())
    assert plain_text.choice(repo, aid).off == frozenset()


def test_a_person_whose_version_is_not_usable_is_told_why(work):
    settings, repo, aid, sid = work
    at = ok(radio(step6(repo, aid), aid).set_value("low").run())
    ok(at.button(key=f"pfb_plain_make_{aid}").click().run())
    worker_makes(settings, repo, text=BAD_KANJI)                       # never passes the checks
    at = step6(repo, aid)
    text = screen_text(at)
    assert "花子さん：意味を変えずに言いかえられなかったため、元の文を使います" in text
    assert plain_view.MAKE in buttons(at)                              # may ask again


def test_nothing_shows_for_a_session_that_may_not_edit(work, monkeypatch):
    settings, repo, aid, sid = work
    monkeypatch.setattr(access, "can_edit", lambda: False)
    at = step6(repo, aid)
    assert not [r for r in at.radio if (r.key or "").startswith("pfb_plain_")]


def test_nothing_shows_for_a_work_whose_videos_are_gone(work):
    settings, repo, aid, sid = work
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-11-01T00:00:00+00:00' WHERE id=?",
                      (aid,))
    at = step6(repo, aid)
    assert not [r for r in at.radio if (r.key or "").startswith("pfb_plain_")]
