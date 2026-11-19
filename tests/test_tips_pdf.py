"""次にためしてみること in the PDFs: only what the coach confirmed, per person and per group.
Synthetic works only (tests/pdf_synthetic.py)."""
import io
import zipfile

from pdf_synthetic import video_group, work
from pdfhelp import pdf_text
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import FORBIDDEN, env  # noqa: F401 (env)

from pfb_app.domain.models import Stage
from pfb_app.export import pdf
from pfb_app.export.sheets import build_report
from pfb_app.services import tips
from tips_helpers import save

ONE = "仲間の動きに合わせて声をかけてみる"
TWO = "パスのあとの動きを意識してみる"
TEAM = "全員でパスのあとに動き直してみる"


def made(repo, sid, who, items, source="model", confirmed=False):
    s = next(x for x in repo.list_subjects(sid) if x["participant_id"] == who)
    pfb = repo.latest_run(sid, Stage.pfb, subject_id=s["id"])
    out = {"tips_status": "generated", "items": [{"text": t, "evidence_refs": []}
                                                 for t in items]}
    if confirmed:
        out["confirmed"] = True
    return s["id"], save(repo, sid, Stage.tips, out, subject_id=s["id"],
                         upstream=pfb.upstream_run_ids, source=source)


def team(repo, sid, items, confirmed=True):
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    out = {"tips_status": "generated", "items": [{"text": t, "evidence_refs": []} for t in items]}
    if confirmed:
        out["confirmed"] = True
    return save(repo, sid, Stage.tips, out, upstream=[con.id],
                source="teacher" if confirmed else "model")


def group(env_, **kw):
    settings, repo = env_
    aid = work(repo, **kw)
    sid = video_group(settings, repo, aid, people=3, seconds=30)
    return repo, aid, sid


def test_confirmed_tips_are_in_both_pdfs_and_the_phone_page(env):
    repo, aid, sid = group(env)
    made(repo, sid, "Pa", [ONE, TWO], source="teacher", confirmed=True)
    team(repo, sid, [TEAM])
    for render in (pdf.render_student_pdf, pdf.render_teacher_pdf):
        text = pdf_text(render(repo, aid))
        assert "次にためしてみること" in text and ONE in text and TWO in text, render
        assert "チームでためしてみること" in text and TEAM in text
    parts = pdf.handout_parts(repo, aid)
    mine = next(p for p in parts if ONE in p[3])        # the phone page of the person
    assert "次にためしてみること" in mine[3] and TEAM in mine[3]
    pa = next(s for s in repo.list_subjects(sid) if s["participant_id"] == "Pa")["id"]
    assert [p[0] for p in parts if ONE in p[3]] == [pa]    # nobody else carries them
    assert all(TEAM in p[3] for p in parts)                # the group's reach every handout


def test_unconfirmed_deleted_and_outdated_tips_are_never_printed(env):
    repo, aid, sid = group(env)
    made(repo, sid, "Pa", [ONE])                                    # the model's own: unconfirmed
    pb, run = made(repo, sid, "Pb", [ONE], source="teacher", confirmed=True)
    tips.confirm(repo, sid, pb, "", run.id)                         # deleted by the coach
    pc, _ = made(repo, sid, "Pc", [TWO], source="teacher", confirmed=True)
    team(repo, sid, [TEAM], confirmed=False)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    for render in (pdf.render_student_pdf, pdf.render_teacher_pdf):
        text = pdf_text(render(repo, aid))
        assert ONE not in text and TEAM not in text and "次にためしてみること" in text   # only Pc
    # the analysis is redone: Pc's tips are outdated too (read on the report)
    repo.save_stage_run(con.model_copy(update={"id": None, "version": 0}))
    report = build_report(repo, aid, with_pictures=False)
    assert all(s.tips == [] for s in report.sheets) and report.team_tips == []


def test_nothing_changes_for_a_work_without_tips(env):
    repo, aid, sid = group(env)
    report = build_report(repo, aid, with_pictures=False)
    assert all(not s.tips and not s.team_tips for s in report.sheets) and not report.team_tips
    for render in (pdf.render_student_pdf, pdf.render_teacher_pdf):
        assert "ためしてみること" not in pdf_text(render(repo, aid))


def test_a_general_work_says_みんな_not_チーム(env):
    repo, aid, sid = group(env, domain="school", sport=None)
    made(repo, sid, "Pa", [ONE], source="teacher", confirmed=True)
    team(repo, sid, [TEAM])
    text = pdf_text(pdf.render_student_pdf(repo, aid))
    assert "みんなでためしてみること" in text and "チームでためしてみること" not in text


def test_two_groups_keep_their_own_tips_on_the_cover(env):
    settings, repo = env
    aid = work(repo)
    first = video_group(settings, repo, aid, people=2, seconds=30, label="A")
    second = video_group(settings, repo, aid, people=2, seconds=30, label="B")
    team(repo, first, ["一つ目の組のヒントを続けてみる"])
    team(repo, second, ["二つ目の組のヒントを続けてみる"])
    report = build_report(repo, aid, with_pictures=False)
    assert report.team_tips == [("A", ["一つ目の組のヒントを続けてみる"]),
                                ("B", ["二つ目の組のヒントを続けてみる"])]
    cover = pdf_text(pdf.render_teacher_pdf(repo, aid))
    assert "一つ目の組のヒントを続けてみる" in cover and "二つ目の組のヒントを続けてみる" in cover
    pages = {n: zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))).read(n).decode()
             for n in zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid))).namelist()
             if n.endswith(".html")}
    assert len(pages) == 4
    assert sum("一つ目の組のヒント" in p for p in pages.values()) == 2
    assert sum("二つ目の組のヒント" in p for p in pages.values()) == 2
    assert not any("一つ目の組のヒント" in p and "二つ目の組のヒント" in p for p in pages.values())


def test_no_school_or_internal_words_in_the_tips_blocks(env):
    repo, aid, sid = group(env)
    made(repo, sid, "Pa", [ONE], source="teacher", confirmed=True)
    team(repo, sid, [TEAM])
    for template in ("student.html.j2", "teacher.html.j2"):
        html = pdf._html(template, repo, aid)
        assert not [w for w in SCHOOL_WORDS if w in html]
        assert not [w for w in FORBIDDEN if w in html]


def test_erasing_a_person_removes_their_tips_and_the_groups_made_from_their_text(env):
    from pfb_app.services import erasure
    settings, repo = env
    repo, aid, sid = group(env)
    pa, _ = made(repo, sid, "Pa", [ONE], source="teacher", confirmed=True)
    pb, _ = made(repo, sid, "Pb", [TWO], source="teacher", confirmed=True)
    team(repo, sid, [TEAM])
    erasure.erase(repo, settings, pa)
    assert repo.run_history(sid, Stage.tips, subject_id=pa) == []
    assert repo.run_history(sid, Stage.tips) == []                  # the group's: made from theirs
    assert tips.for_pdf(repo, sid, pb) == [TWO]                    # the others' stay
