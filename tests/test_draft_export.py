"""The draft option of the PDFs: unapproved texts marked 未承認（下書き）, never mixed up with the
approved ones; off by default (nothing changes); the footage credit."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_export import add_video_subject, repo, text_of, video_assignment  # noqa: F401,E402

import pytest  # noqa: E402

from pfb_app.export import labels  # noqa: E402
from pfb_app.export.pdf import ExportError, _html, render_student_pdf, render_teacher_pdf  # noqa: E402

CREDIT = "Samsunspor TBF / CC BY 3.0 / 切り出し・枠の描画あり"


def _mixed(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "確定の人", pfb_text="確定した本文です。")
    add_video_subject(repo, aid, "保留の人", pfb_text="保留の本文です。", approved=False)
    return aid


def test_default_is_unchanged(repo):
    aid = _mixed(repo)
    t = text_of(render_student_pdf(repo, aid))
    assert "保留の人" not in t and "下書き" not in t and "出典" not in t
    assert "確かめたものです" in t
    teacher = _html("teacher.html.j2", repo, aid)
    assert "下書き" not in teacher and "出典" not in teacher


def test_draft_handouts_carry_no_status_band_only_a_neutral_line(repo):
    aid = _mixed(repo)
    pdf = text_of(render_student_pdf(repo, aid, drafts=True, credits=[CREDIT]))
    assert "保留の本文です" in pdf and CREDIT in pdf
    for word in ("承認", "未承認", "評価する人はまだ確認していません", "下書き"):
        assert word not in pdf, word
    assert pdf.count(labels.AI_LINE) == 1          # the draft person only (the other is approved)
    html = _html("student.html.j2", repo, aid, drafts=True, credits=[CREDIT])
    assert '<p class="draft-banner">' not in html and '<p class="draft-note">' not in html


def test_teacher_pdf_states_each_person_status(repo):
    aid = _mixed(repo)
    html = _html("teacher.html.j2", repo, aid, drafts=True, credits=[CREDIT])
    assert html.count('<section class="person-page">') == 2
    assert "未承認（下書き）" in html and ">承認<" in html     # the evaluator PDF keeps the states
    approved_page = next(p for p in html.split('<section class="person-page">')[1:]
                         if "確定の人" in p)
    assert '<p class="draft-banner">' not in approved_page
    assert "未承認（下書き）" in html.split('<section class="sheet-page">')[1]
    assert CREDIT in html
    assert "承認した出力ではありません" not in approved_page
    assert render_teacher_pdf(repo, aid, drafts=True, credits=[CREDIT])


def test_missing_credit_says_so_instead_of_blank(repo):
    aid = _mixed(repo)
    assert labels.CREDIT_MISSING in _html("student.html.j2", repo, aid, drafts=True)


def test_draft_option_alone_with_nobody_approved_still_works(repo):
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "保留だけ", approved=False)
    with pytest.raises(ExportError):
        render_student_pdf(repo, aid)                      # default: nothing to export
    assert "保留だけ" in text_of(render_student_pdf(repo, aid, drafts=True))


def test_jersey_label_three_states():
    from pfb_app.export import jersey
    # confirmed: a roster number, or a number name the coach gave
    assert jersey.label({"display_name": "選手3", "participant_id": "auto3", "roster": {"number": "8"}},
                        {}) == ("背番号 8", "人物ID 3", "confirmed")
    assert jersey.label({"display_name": "#8", "participant_id": "auto3", "roster": {}},
                        {"auto3": "coach"})[0::2] == ("背番号 8", "confirmed")
    # estimated: the AI's reading (name_source "number", or no source: never a fact)
    assert jersey.label({"display_name": "#8", "participant_id": "auto12", "roster": {}},
                        {"auto12": "number"}) == ("背番号 8番？（推定）", "人物ID 12", "estimated")
    assert jersey.label({"display_name": "#8（チーム1）", "participant_id": "auto1"}, {})[0] \
        == "背番号 8番？（推定）・チーム1"
    # unknown: the default name
    assert jersey.label({"display_name": "選手2", "participant_id": "auto2", "roster": {}},
                        {}) == ("背番号 不明", "人物ID 2", "unknown")


def test_draft_handouts_one_page_each_and_no_san(repo):
    from pfb_app.export.pdf import _html
    aid = video_assignment(repo)
    add_video_subject(repo, aid, "選手2", approved=False)
    add_video_subject(repo, aid, "#8", approved=False)
    add_video_subject(repo, aid, "選手5", pfb_text="確定した本文です。")
    from pdfhelp import pdf_page_count
    pdf = render_student_pdf(repo, aid, drafts=True, credits=[CREDIT])
    assert pdf_page_count(pdf) == 3                       # one page per person
    t = text_of(pdf)
    assert "背番号 不明" in t and "背番号 8番？（推定）" in t
    assert "さんへ" not in t and "さん" not in t and "承認" not in t.replace("確定した本文です。", "")
    assert t.count(CREDIT[:20]) >= 3
    # the default export keeps the names and 「さんへ」
    default = text_of(render_student_pdf(repo, aid))
    assert "選手5" in default and "さんへ" in default and "背番号" not in default
    teacher = text_of(render_teacher_pdf(repo, aid, drafts=True))
    assert "背番号 不明" in teacher and "さん" not in teacher


def test_brand_is_nicemill_in_every_pdf_and_pages_equal_people(repo):
    from pdfhelp import pdf_page_count
    aid = _mixed(repo)
    student = render_student_pdf(repo, aid, drafts=True, credits=[CREDIT])
    teacher = render_teacher_pdf(repo, aid, drafts=True, credits=[CREDIT])
    assert pdf_page_count(student) == 2                  # one page per person
    for pdf in (student, teacher):
        t = text_of(pdf)
        assert "NiceMill" in t and "ナイスミル" not in t
    assert text_of(render_student_pdf(repo, aid)).count("NiceMill") >= 1   # the default too
