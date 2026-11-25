"""成長の記録 screens (AppTest): the list with suggestions (accept, undo, decline), a person's
record, linking by hand, the team/group setting, and the entry points (home, step 5, step 6).
Synthetic data only."""
import pytest
from streamlit.testing.v1 import AppTest

from growth_synthetic import add_work
from pfb_app.export import growth as record
from pfb_app.services import growth
from test_neutral_wording import clean, everything
from test_wizard_ui_basic import FORBIDDEN, UI, app, env, ok, primaries  # noqa: F401 (env)

PAGE = UI / "pages" / "growth.py"


def page(query: dict | None = None) -> AppTest:
    at = AppTest.from_file(str(PAGE), default_timeout=60)
    for k, v in (query or {}).items():
        at.query_params[k] = str(v)
    return ok(at.run())


def button(at, label, key_part=""):
    return next(b for b in at.button if b.label == label and key_part in (b.key or ""))


def neutral(at, where):
    text = everything(at)
    clean(text, where)
    found = [w for w in FORBIDDEN if w in text]
    assert not found, f"{where}: {found}"


def two_works(repo):
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"], "参加者2": ["傾聴"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言", "傾聴"], "参加者2": ["傾聴"]},
                       day="2026-09-08")
    return a, b


def test_empty_list_says_what_to_do(env):
    at = page()
    assert "まだ記録がありません" in everything(at)
    assert primaries(at) == ["この人の記録を見る"]
    neutral(at, "growth empty")


def test_a_suggestion_is_accepted_then_undone(env):
    _, repo = env
    a, b = two_works(repo)
    at = page()
    assert primaries(at) == ["この人の記録を見る"]
    text = everything(at)
    assert "つながりの候補（1件）" in text and "同じ名前（山田）" in text
    assert "自動ではつなぎません" in text
    neutral(at, "growth list")
    at = ok(button(at, "同じ人としてつなぐ").click().run())
    person = repo.growth_person_of(a["山田"])
    assert person is not None and repo.growth_person_of(b["山田"]) == person
    assert "つなぎました" in everything(at)
    at = ok(button(at, "元に戻す").click().run())
    assert repo.growth_person_of(a["山田"]) is None and repo.growth_person_of(b["山田"]) is None
    assert "つながりの候補（1件）" in everything(at)


def test_a_suggestion_is_declined(env):
    _, repo = env
    two_works(repo)
    at = page()
    at = ok(button(at, "別の人").click().run())
    assert growth.suggestions(repo) == [] and repo.growth_links() == []
    assert "つながりの候補（" not in everything(at)


def test_open_a_person_from_the_list(env):
    _, repo = env
    a, b = two_works(repo)
    person = growth.link_manual(repo, a["山田"], None)
    growth.link_manual(repo, b["山田"], person)
    at = page()
    at.selectbox(key="pfb_growth_pick").set_value(f"p{person}")
    at = ok(button(at, "この人の記録を見る").click().run())
    assert at.query_params["person"] == [str(person)] or at.query_params["person"] == str(person)
    text = everything(at)
    assert "山田さんの成長の記録" in text
    for line in record.statements(record.for_person(repo, person)):
        assert line in text
    assert "定例会1" in text and "定例会2" in text
    assert "まわりの人の話をよく聞いていました" in text
    assert primaries(at) == ["つなぐ"]
    images = [el.proto.body for el in at.get("html") if "data:image/svg+xml" in el.proto.body]
    assert images and "発言" in images[0]                     # the grid, with its words in alt
    neutral(at, "growth person")


def test_one_work_says_so_and_manual_link_joins_anyone(env):
    _, repo = env
    a, b = two_works(repo)
    at = page({"subject": a["参加者2"]})
    assert record.ONE_WORK in everything(at)
    assert primaries(at) == ["つなぐ"]
    at.selectbox(key="pfb_growth_link_pick").set_value(b["参加者2"])
    at = ok(button(at, "つなぐ").click().run())
    person = repo.growth_person_of(a["参加者2"])
    assert person is not None and repo.growth_person_of(b["参加者2"]) == person
    assert {r["how"] for r in repo.growth_links(person)} == {"manual"}
    assert "作業2件の記録があります" in everything(at)


def test_unlink_a_work_from_a_record(env):
    _, repo = env
    a, b = two_works(repo)
    person = growth.link_manual(repo, a["山田"], None)
    growth.link_manual(repo, b["山田"], person)
    at = page({"person": person})
    at = ok(button(at, "この作業を外す", str(b["山田"])).click().run())
    assert repo.growth_person_of(b["山田"]) is None
    assert record.ONE_WORK in everything(at)


def test_only_approved_texts_are_shown(env):
    _, repo = env
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    _, _, b = add_work(repo, "定例会2", {"山田": ["発言"]}, day="2026-09-08")
    person = growth.link_manual(repo, a["山田"], None)
    growth.link_manual(repo, b["山田"], person)
    repo.set_approval(b["山田"], None)             # taken back after linking
    text = everything(page({"person": person}))
    assert "「定例会1」で" in text and "「定例会2」で" not in text
    assert record.NOT_APPROVED in text


def test_the_team_or_group_of_a_work_is_saved(env):
    _, repo = env
    aid, _, _ = add_work(repo, "練習1", {"#5": ["パス"]}, profile="sports")
    at = page()
    at.text_input(key=f"pfb_growth_group_{aid}").set_value("U12 A")
    at = ok(button(at, "チーム・グループを保存").click().run())
    assert repo.get_assignment(aid)["growth_group"] == "U12 A"


def test_home_and_sidebar_link_to_the_records(env):
    _, repo = env
    two_works(repo)
    at = ok(app())
    assert "成長の記録" in [link.proto.label for link in at.get("page_link")]
    assert "成長の記録" in [link.proto.label for link in at.sidebar.get("page_link")]


def test_an_approved_person_on_step_5_links_to_their_record(env):
    settings, repo = env
    from test_wizard_ui_basic import video_assignment
    from test_wizard_ui_review_pdf import reviewable, subject
    from pfb_app.services import review
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    pa = subject(repo, sid, "Pa")["id"]
    review.approve(repo, pa)
    at = ok(app(session={"assignment_id": aid, "step": 5, "submission_id": None},
                extra={f"pfb_student_{sid}": pa}))
    links = [link for link in at.get("page_link") if link.proto.label == "成長の記録を見る"]
    assert links and f"subject={pa}" in links[0].proto.page + str(links[0].proto)


def test_step_6_offers_the_page_off_by_default(env):
    settings, repo = env
    from test_wizard_ui_basic import video_assignment
    from test_wizard_ui_review_pdf import reviewable, subject
    from pfb_app.services import review
    hanako = []
    for title in ("練習1", "練習2"):
        aid = video_assignment(repo, title)
        sid = reviewable(settings, repo, aid)
        hanako.append(subject(repo, sid, "Pa")["id"])
        review.approve(repo, hanako[-1])
    person = growth.link_manual(repo, hanako[0], None)
    growth.link_manual(repo, hanako[1], person)
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    box = at.checkbox(key="pfb_pdf_growth")
    assert box.value is False and "成長のまとめ" in box.label
    assert "いま1人" in everything(at)
    assert primaries(at) == ["参加者に渡すPDFを受け取る"]
    neutral(at, "step 6 with the growth option")


def test_pdf_data_passes_the_choice(env):
    from pdfhelp import pdf_text
    from pfb_app.ui.wizard import step_pdf
    _, repo = env
    a, b = two_works(repo)
    person = growth.link_manual(repo, a["山田"], None)
    growth.link_manual(repo, b["山田"], person)
    aid = repo.get_submission(repo.get_subject(b["山田"])["submission_id"])["assignment_id"]
    assert "成長のまとめ" not in pdf_text(step_pdf.pdf_data(aid, "student")())
    assert "成長のまとめ" in pdf_text(step_pdf.pdf_data(aid, "student", growth=True)())


def test_only_people_with_an_approved_text_of_other_works_can_be_linked(env):
    """Review HIGH1/HIGH2: the choices hold approved people of other works only, and the
    screen says so (in words that also cover 「あとから訂正」 works, 2026-11-23)."""
    _, repo = env
    aid, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, day="2026-09-01")
    _, _, same = add_work(repo, "", {"鈴木": ["発言"]}, label="B", aid=aid)
    _, _, b = add_work(repo, "定例会2", {"佐藤": ["発言"], "田中": ["傾聴"]}, day="2026-09-08",
                       approve={"佐藤"})
    at = page({"subject": a["山田"]})
    assert growth.EITHER_ONLY in everything(at)
    assert at.selectbox(key="pfb_growth_link_pick").options == [
        "佐藤（定例会2・2026年9月8日）"]
    assert same["鈴木"] and b["田中"]


def test_a_person_without_an_approved_text_cannot_start_a_record(env):
    _, repo = env
    _, _, a = add_work(repo, "定例会1", {"山田": ["発言"]}, approve=False)
    add_work(repo, "定例会2", {"佐藤": ["発言"]}, day="2026-09-08")
    at = page({"subject": a["山田"]})
    assert growth.APPROVED_ONLY in everything(at)
    assert primaries(at) == ["つなぐ"] and button(at, "つなぐ").disabled


def test_the_undo_stays_on_the_screen_it_belongs_to(env):
    """Review LOW11: after an accept, another record's screen shows no 「元に戻す」."""
    _, repo = env
    a, b = two_works(repo)
    other = growth.link_manual(repo, a["参加者2"], None)
    at = page()
    at = ok(button(at, "同じ人としてつなぐ").click().run())
    assert any(x.label == "元に戻す" for x in at.button)
    at.selectbox(key="pfb_growth_pick").set_value(f"p{other}")
    at = ok(button(at, "この人の記録を見る").click().run())
    assert not any(x.label == "元に戻す" for x in at.button)


def test_a_refused_accept_says_why(env):
    _, repo = env
    a, b = two_works(repo)
    at = page()
    growth.link_manual(repo, a["山田"], None)      # meanwhile, in another tab
    growth.link_manual(repo, b["山田"], None)
    at = ok(button(at, "同じ人としてつなぐ").click().run())
    assert growth.OTHER_RECORD in everything(at)
    assert "つなぎました" not in everything(at)
