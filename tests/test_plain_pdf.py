"""やさしい文章版 (rel8): the handouts take the picked version, the evaluator PDF keeps the
approved text and says which version each handout carries. Synthetic data only."""
import io
import json
import zipfile

import pytest
from pdfhelp import pdf_text

from pfb_app.export import pdf
from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline import plain_checks
from pfb_app.pipeline.common import StageContext
from pfb_app.pipeline.runner import execute_job
from pfb_app.services import plain_text, review
from test_plain_text import LOW, PROMPTS, hanako, work  # noqa: F401 (work)
from test_wizard_ui_basic import env  # noqa: F401 (the work fixture needs it)
from test_wizard_ui_review_pdf import TEXTS, save_pfb, subject


def squash(text: str) -> str:
    """A PDF's text with its line breaks and spaces out (it wraps where it likes)."""
    return "".join(text.split())


def made(settings, repo, sid, level, text):
    """The person's version, made by the worker (a fake model answers `text`)."""
    plain_text.request(repo, repo.get_submission(sid)["assignment_id"], level,
                       [hanako(repo, sid)["id"]])
    job = repo.claim_next_job({"pfb3x3:standard"})
    execute_job(job, StageContext(settings, repo,
                                  FakeBackend(lambda r: json.dumps({"text": text},
                                                                    ensure_ascii=False)),
                                  lambda: False))


def test_the_handout_shows_the_picked_version_and_the_evaluator_pdf_the_approved_text(work):
    settings, repo, aid, sid = work
    taro = subject(repo, sid, "Pb")["id"]
    review.approve(repo, taro)
    made(settings, repo, sid, "low", LOW)
    plain_text.choose(repo, aid, "low")
    handout = squash(pdf_text(pdf.render_student_pdf(repo, aid)))
    assert squash(LOW) in handout and squash(TEXTS["Pa"]) not in handout
    assert squash(TEXTS["Pb"]) in handout                         # no version of his: approved text
    evaluator = squash(pdf_text(pdf.render_teacher_pdf(repo, aid)))
    assert squash(TEXTS["Pa"]) in evaluator and squash(TEXTS["Pb"]) in evaluator
    label = plain_checks.LEVELS["low"].label
    assert f"参加者に渡すPDFの文章：{label}".replace(" ", "") in evaluator
    assert squash(LOW) in evaluator                               # what was handed out, on record
    assert f"参加者に渡すPDFの文章：{plain_checks.ORIGINAL_LABEL}" in evaluator


def test_nothing_changes_while_the_choice_is_the_approved_text(work):
    settings, repo, aid, sid = work
    made(settings, repo, sid, "low", LOW)
    handout = squash(pdf_text(pdf.render_student_pdf(repo, aid)))
    assert squash(TEXTS["Pa"]) in handout and squash(LOW) not in handout
    assert "参加者に渡すPDFの文章" not in pdf_text(pdf.render_teacher_pdf(repo, aid))


def test_the_one_person_zip_and_the_phone_page_take_it_too(work):
    settings, repo, aid, sid = work
    made(settings, repo, sid, "low", LOW)
    plain_text.choose(repo, aid, "low")
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))) as z:
        page = next(z.read(n).decode() for n in z.namelist() if n.endswith(".html"))
        one = next(z.read(n) for n in z.namelist() if n.endswith(".pdf"))
    assert LOW in page and TEXTS["Pa"] not in page
    assert squash(LOW) in squash(pdf_text(one))


def test_an_outdated_version_is_never_handed_out(work):
    settings, repo, aid, sid = work
    made(settings, repo, sid, "low", LOW)
    plain_text.choose(repo, aid, "low")
    save_pfb(repo, sid, hanako(repo, sid)["id"], "花子さんは柱を押さえました。")
    review.approve(repo, hanako(repo, sid)["id"])
    handout = squash(pdf_text(pdf.render_student_pdf(repo, aid)))
    assert squash("花子さんは柱を押さえました。") in handout and squash(LOW) not in handout


def test_a_person_switched_off_keeps_the_approved_text(work):
    settings, repo, aid, sid = work
    made(settings, repo, sid, "low", LOW)
    plain_text.choose(repo, aid, "low", off=[hanako(repo, sid)["id"]])
    handout = squash(pdf_text(pdf.render_student_pdf(repo, aid)))
    assert squash(TEXTS["Pa"]) in handout and squash(LOW) not in handout


def test_the_html_of_the_handouts_follows_the_choice(work):
    settings, repo, aid, sid = work
    made(settings, repo, sid, "low", LOW)
    plain_text.choose(repo, aid, "low")
    assert LOW in pdf._html("student.html.j2", repo, aid)
    assert TEXTS["Pa"] in pdf._html("teacher.html.j2", repo, aid)
