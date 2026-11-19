"""The optional 「成長のまとめ」 page of a person's handout (off by default): only for a person
whose record has two or more approved works up to this work; never a later work."""
import io
import zipfile

from pdfhelp import pdf_page_count, pdf_text

from growth_synthetic import add_work
from pfb_app.export import growth as record
from pfb_app.export import pdf
from pfb_app.services import growth
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import FORBIDDEN, env  # noqa: F401 (env)

TITLE = "成長のまとめ"


def season(repo):
    """山田 in three works (linked), 鈴木 only in the second."""
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    aid2, _, b = add_work(repo, "定例会2", {"山田": ["発言", "傾聴"], "鈴木": ["まとめ"]},
                          day="2026-09-08")
    _, _, c = add_work(repo, "定例会3", {"山田": ["傾聴"]}, day="2026-09-15")
    person = growth.link_manual(repo, a["山田"], None)
    for s in (b["山田"], c["山田"]):
        growth.link_manual(repo, s, person)
    return aid2, person


def test_off_by_default(env):
    _, repo = env
    aid, _ = season(repo)
    data = pdf.render_student_pdf(repo, aid)
    assert TITLE not in pdf_text(data)
    assert pdf_page_count(data) == 2          # 山田 and 鈴木, one page each


def test_the_page_follows_the_handout_of_a_person_with_two_works(env):
    _, repo = env
    aid, person = season(repo)
    data = pdf.render_student_pdf(repo, aid, growth=True)
    text = pdf_text(data)
    assert pdf_page_count(data) == 3          # 鈴木 has one work: no page for her
    assert text.count(TITLE) == 1
    assert "定例会1" in text and "定例会2" in text
    assert "定例会3" not in text              # a later work is never on an earlier handout
    for line in record.statements(record.for_person(repo, person, up_to=aid)):
        assert line[:12] in text.replace("\n", "")
    assert "まわりの人の話をよく聞いていました" in text
    found = [w for w in (*SCHOOL_WORDS, "ルーブリック", *FORBIDDEN) if w in text]
    assert not found, found


def test_the_zip_gets_the_page_too_when_asked(env):
    _, repo = env
    aid, _ = season(repo)
    with zipfile.ZipFile(io.BytesIO(pdf.render_student_zip(repo, aid, growth=True))) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".pdf"))
        texts = {n: pdf_text(z.read(n)) for n in names}
        page = next(n for n in z.namelist() if n.endswith(".html") and "山田" in n)
        assert TITLE in z.read(page).decode("utf-8")
    assert [TITLE in texts[n] for n in names] == [True, False]   # 山田, 鈴木 (name order)


def test_how_many_people_get_the_page(env):
    _, repo = env
    aid, _ = season(repo)
    assert record.handout_count(repo, aid) == 1


def test_counting_the_people_for_step_6_is_one_query(env):
    """Review MED4: step 6 counts on every run; it must not walk every work per person."""
    _, repo = env
    aid, _ = season(repo)
    for k in range(10):
        add_work(repo, f"ほかの作業{k}", {"山田": ["発言"], "鈴木": ["発言"]}, day=f"2026-08-{k + 1:02d}")
    seen = []
    repo.conn.set_trace_callback(seen.append)
    try:
        assert record.handout_count(repo, aid) == 1
    finally:
        repo.conn.set_trace_callback(None)
    assert len(seen) == 1, seen


def test_step_6_counts_only_the_people_it_lists_as_included(env):
    """Re-review (a): a text an update outdated is not in the PDF, so not in the count."""
    from growth_synthetic import save
    from pfb_app.domain.models import Kind, Stage
    from pfb_app.ui.wizard import step_pdf
    _, repo = env
    aid, _ = season(repo)
    sid = repo.list_submissions(aid)[0]["id"]
    save(repo, sid, Stage.contribution, {"judgments": []}, camera="cam1")   # the texts are outdated
    included, _ = step_pdf.who(repo, aid, Kind.video)
    assert included == []
    assert record.handout_count(repo, aid) == 1                  # the stored approval alone
    assert record.handout_count(repo, aid, [i for i, _ in included]) == 0
