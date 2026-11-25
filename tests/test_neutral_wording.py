"""The app is for broad use, not only schools: no school words on any screen, PDF, download
name, prompt or the user guide (rule of 2026-10-22).

Renders the home, the drop zone, every wizard step and the 詳細画面 pages for a general work
and a sports work (plus a document work and a check card), and both PDFs, and asserts none of
the school words appear. Code comments and identifiers may keep English 'school'.

Only synthetic data: invented names and texts, generated tiny images and videos.
"""
import ast
import re
from pathlib import Path

from pdfhelp import pdf_font_names, pdf_page_count, pdf_text
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from pfb_app.domain import profiles
from pfb_app.domain.models import Stage
from pfb_app.domain.vocab import WORDS, localize, words
from pfb_app.export import pdf
from pfb_app.services import review
from pfb_app.ui.wizard.progress import work_progress
from test_content_check import detect_done
from test_wizard_check import ready_work
from test_wizard_ui_basic import (FORBIDDEN, UI, app, env, html, ok,  # noqa: F401 (env)
                                 screen_text, video_assignment)
from test_eta import REAL_RUNS, history_of, queued_work
from test_wizard_ui_people_wait import confirmed_and_analysing, group, save, track
from test_wizard_ui_review_pdf import reviewable, subject

ROOT = Path(__file__).resolve().parents[1]
DATA_URI = re.compile(r"data:image/[a-z+]+;base64,[A-Za-z0-9+/=]+")
SCHOOL_WORDS = ("生徒", "先生", "学生", "教員", "授業", "課題", "提出", "宿題", "学校",
                "教師", "児童", "担任", "クラス", "採点", "成績")
# strips these words from uploaded file names to guess a name: input parsing, never shown
NAME_STRIPPING = Path("preprocess") / "documents.py"
DETAIL = ("assignments", "submissions", "status", "review", "export")


def clean(text: str, where: str, extra=("ルーブリック",)) -> None:
    """No school words; on what people see, no 「ルーブリック」 either (it is 「評価の観点」)."""
    found = [w for w in (*SCHOOL_WORDS, *extra) if w in text]
    assert not found, f"{where}: {found}"


def everything(at: AppTest) -> str:
    """Screen text plus what screen_text leaves out: links, downloads, tabs, tables, metrics."""
    parts = [screen_text(at)]
    parts += [link.proto.label for link in at.get("page_link")]
    for b in at.get("download_button"):
        parts += [b.proto.label, b.proto.help]
    parts += [t.label for t in at.tabs]
    parts += [str(e.value) for e in at.get("text")]
    for frame in at.get("arrow_data_frame"):
        try:
            parts.append(frame.value.to_csv())
        except Exception:   # noqa: BLE001 - a frame that does not convert still has its proto
            parts.append(str(frame.proto))
    for kind in ("data_editor", "number_input", "text_area", "text_input", "multiselect"):
        for w in at.get(kind):
            parts.append(str(getattr(w, "proto", "")))
    return "\n".join(parts)


def doc_work(repo) -> int:
    """A document work with one approved and one waiting feedback."""
    aid = repo.create_assignment("探究レポート", "document", "調べてまとめる", "観点A", "standard",
                                 False)
    for name in ("山田太郎", "鈴木一郎"):
        sid = repo.create_submission(aid, f"{name}.pdf", {"source_path": "/x.pdf",
                                                         "original_filename": f"{name}.pdf"})
        sub = repo.add_subject(sid, None, name, None, {})
        prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x.pdf"})
        ev = save(repo, sid, Stage.evidence, {"items": [{
            "page": 1, "quote": "引用", "observed_content": "内容", "evidence_source": "text",
            "confidence": "high", "item_id": "p1-001", "quote_verified": False}]},
            upstream=[prep.id])
        con = save(repo, sid, Stage.contribution, {"judgments": [{
            "ref_id": "p1-001", "contribution_interpretation": "解釈",
            "is_positive_contribution": True, "rubric_criteria": ["観点A"]}]}, upstream=[ev.id])
        save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": "よく考えて書けていました。",
                                    "evidence_refs": ["p1-001"]}, subject_id=sub,
             upstream=[con.id])
        if name == "山田太郎":
            review.approve(repo, sub)
    return aid


@pytest.fixture
def works(env):
    """{name: assignment id} over every state worth a look, in both vocabularies."""
    settings, repo = env
    out = {}
    for domain in ("school", "sports"):
        sport = "basketball" if domain == "sports" else None
        aid = repo.create_assignment("ペーパータワー", "video", "塔を高く作る", "協力", "standard",
                                     True, domain=domain, sport=sport)
        sid = reviewable(settings, repo, aid)
        review.approve(repo, subject(repo, sid, "Pa")["id"])
        reviewable(settings, repo, aid, label="B班", insufficient=("Pc",))
        out[f"{domain}_review"] = aid
        # a work waiting for the people mapping (step 3's form)
        aid = repo.create_assignment("円陣", "video", "輪になって話す", "", "standard", True,
                                     domain=domain, sport=sport)
        track(settings, repo, group(repo, aid))
        out[f"{domain}_people"] = aid
        # a work with nothing yet (step 2's form)
        out[f"{domain}_empty"] = repo.create_assignment("空の作業", "video", "準備中", "",
                                                        "standard", True, domain=domain,
                                                        sport=sport)
    for profile in ("meeting", "presentation", "workshop", "learning"):   # activity kinds
        aid = repo.create_assignment("定例の場", "video", "週の振り返り", "観点", "standard", True,
                                     profile=profile)
        sid = reviewable(settings, repo, aid)
        review.approve(repo, subject(repo, sid, "Pa")["id"])
        out[f"{profile}_review"] = aid
    out["document"] = doc_work(repo)
    out["check_sport"], _ = ready_work(settings, repo)
    out["check_group"], sid = ready_work(settings, repo, scene="group_work")
    return out


def test_school_vocabulary_is_general():
    w = words("school")
    assert (w.work, w.people, w.giver, w.roster, w.time_anchor, w.rubric_label,
            w.person_prefix) == ("活動", "参加者", "評価者", "名簿", "開始から",
                                 "評価の観点（任意）", "参加者")
    for field in WORDS["school"].__dataclass_fields__:
        clean(getattr(WORDS["school"], field), f"words.{field}")
        clean(getattr(WORDS["sports"], field), f"sports words.{field}")


def test_localize_keeps_group_activity():
    assert localize("グループ活動を撮った動画", "sports") == "グループ活動を撮った動画"
    assert localize("活動を決める", "sports") == "試合・練習を決める"
    assert localize("承認した参加者", "sports") == "承認した選手"
    assert localize("評価者用（根拠つき）PDF", "sports") == "コーチ用（根拠つき）PDF"


def test_home_and_drop_zone(env, works):
    clean(everything(ok(app())), "home")
    clean(everything(ok(app(session={"assignment_id": None, "step": 1, "submission_id": None}))),
          "drop zone")


def test_queue_search_and_bulk_screens_use_neutral_words(env, works):
    """The waiting-time lines, the finished band, the search row, the list of people waiting for
    a look (open, with a person picked, and at its confirmation) and a search that finds nothing:
    every string on them passes the school-word and forbidden-word checks."""
    settings, repo = env
    history_of(repo, REAL_RUNS)
    ahead, _sid, _jid = queued_work(settings, repo, "先の作業")
    mine = video_assignment(repo, "解析中の作業")
    confirmed_and_analysing(settings, repo, mine)

    def check(at, where):
        text = DATA_URI.sub("", everything(at) + "\n" + html(at))
        clean(text, where)
        found = [w for w in FORBIDDEN if w in text]
        assert not found, f"{where}: {found}"
        return text

    at = ok(app())
    text = check(at, "home")
    assert "目安" in text and "の解析が終わりました" in text and "作業をさがす" in text
    at = ok(at.button(key="pfb_wait_toggle").click().run())
    assert "読んで確認した" in check(at, "waiting list")
    boxes = [c for c in at.checkbox if c.key and c.key.startswith("pfb_wait_pick_")]
    assert boxes
    at = ok(boxes[0].check().run())
    at = ok(at.button(key="pfb_wait_ask").click().run())
    assert "よろしいですか" in check(at, "bulk confirm")
    at = ok(at.button(key="pfb_wait_cancel").click().run())
    at = ok(at.text_input(key="pfb_find_text").input("バスケ").run())
    assert "条件を消す" in check(at, "search narrowed")
    at = ok(at.text_input(key="pfb_find_text").input("そんな作業はない").run())
    assert "条件に合う作業はありません" in check(at, "search finds nothing")
    check(ok(app(session={"assignment_id": mine, "step": 4, "submission_id": None})), "work screen")
    assert ahead


def test_home_delete_confirm_area(env, works):
    at = ok(app())
    for name in ("school_review", "sports_people", "document"):
        at = ok(at.button(key=f"pfb_del_{works[name]}").click().run())
        assert "削除しますか" in everything(at)
        clean(everything(at), f"home delete confirm ({name})")


@pytest.mark.parametrize("name", ["school_review", "sports_review", "school_people",
                                  "sports_people", "school_empty", "sports_empty", "document",
                                  "check_sport", "check_group", "meeting_review",
                                  "presentation_review", "workshop_review", "learning_review"])
def test_every_wizard_step(env, works, name):
    _, repo = env
    aid = works[name]
    steps = work_progress(repo, aid).steps
    for n in range(1, len(steps) + 1):
        clean("\n".join(steps), f"{name} stepper")
        at = ok(app(session={"assignment_id": aid, "step": n, "submission_id": None}))
        clean(everything(at), f"{name} step {n}")


@pytest.mark.parametrize("page", DETAIL)
@pytest.mark.parametrize("name", ["school_review", "sports_review", "document", "school_people"])
def test_detail_pages(env, works, page, name):
    at = AppTest.from_file(str(UI / "pages" / f"{page}.py"), default_timeout=60)
    at.session_state["current_assignment"] = works[name]
    clean(everything(ok(at.run())), f"{page} ({name})")


def test_empty_detail_pages(env):
    for page in DETAIL:
        at = ok(AppTest.from_file(str(UI / "pages" / f"{page}.py"), default_timeout=60).run())
        clean(everything(at), f"{page} (empty)")


@pytest.mark.parametrize("path", ["pages/guide.py", "pages/sample.py", "pages/quality.py",
                                  "pages/growth.py", "pages/dashboard.py", "pages/ask.py"])
def test_guide_sample_and_quality_pages(env, works, path):
    """はじめての方へ (it shows docs/user_guide.md), 見本の作業, この解析の確かさ and 成長の記録, with works
    in the database: no school words, no 「ルーブリック」 and none of the wizard's FORBIDDEN words."""
    at = app()
    at.switch_page(path)
    text = DATA_URI.sub("", everything(ok(at.run())))   # picture data is not text
    clean(text, path)
    found = [w for w in FORBIDDEN if w in text]
    assert not found, f"{path}: {found}"


def test_linecall_live_page(env, works):
    """判定支援's 「ライブ判定」 with sports works (their cameras are offered) and idle."""
    at = ok(AppTest.from_file(str(UI / "pages" / "linecall_live.py"), default_timeout=60).run())
    assert at.radio(key="lc_live_source").options[1:]   # the sports cameras are listed
    clean(everything(at), "linecall_live")


def test_linecall_review_page(env, works):
    """判定支援's 「試合後の判定材料」 with the sports works (the first is picked)."""
    at = ok(AppTest.from_file(str(UI / "pages" / "linecall_review.py"), default_timeout=60).run())
    assert at.selectbox(key="lc_work").options   # the sports works are offered
    clean(everything(at), "linecall_review")


@pytest.mark.parametrize("name", ["school_review", "sports_review", "document",
                                  "meeting_review", "presentation_review", "workshop_review",
                                  "learning_review"])
def test_both_pdfs_and_their_file_names(env, works, name, monkeypatch):
    _, repo = env
    names = []
    real = st.download_button

    def recording(*args, **kwargs):   # the file name stays on the server: record it here
        names.append(kwargs.get("file_name"))
        return real(*args, **kwargs)
    monkeypatch.setattr(st, "download_button", recording)
    aid = works[name]
    # a title of its own: AppTest's picker maps a label back to the first work with it
    repo.conn.execute("UPDATE assignments SET title=title || ? WHERE id=?", (str(aid), aid))
    for template, render in (("student.html.j2", pdf.render_student_pdf),
                             ("teacher.html.j2", pdf.render_teacher_pdf)):
        clean(pdf._html(template, repo, aid), f"{name} {template}")
        clean(pdf_text(render(repo, aid)), f"{name} {template} (pdf)")
    steps = work_progress(repo, aid).steps
    at = ok(app(session={"assignment_id": aid, "step": len(steps), "submission_id": None}))
    assert len(names) == 3 and all(names), names
    clean("\n".join(names), f"{name} download names")
    giver = words(repo.get_assignment(aid)["profile"]).giver   # 評価者, コーチ, 進行役 ...
    assert names[1].endswith(f"_{giver}用.pdf")
    assert names[2].endswith("ごと.zip")
    people = words(profiles.of_work(repo.get_assignment(aid))).people   # 参加者, 選手, 発表者 ...
    assert names[2].endswith(f"_{people}ごと.zip")
    wizard = sorted(names)
    # 詳細画面's PDF出力: each PDF is made on a click, then offered
    names.clear()
    page = AppTest.from_file(str(UI / "pages" / "export.py"), default_timeout=60)
    page.session_state["current_assignment"] = aid
    page = ok(page.run())
    for key in ("make-teacher", "make-student", "make-each"):
        page = ok(page.button(key=key).click().run())
        clean(everything(page), f"{name} export page")
    assert len(names) == 3 and all(names), names
    assert names[2].endswith("ごと.zip")
    assert sorted(names) == wizard            # the same file names as the wizard's
    clean("\n".join(names), f"{name} export download names")


def test_prompts_templates_and_user_guide():
    scanned = [*sorted((ROOT / "prompts").rglob("*.md")),
               *sorted((ROOT / "src" / "pfb_app" / "export" / "templates").iterdir()),
               ROOT / "docs" / "user_guide.md"]
    for path in scanned:   # the prompts may name the rubric section 「ルーブリック」
        clean(path.read_text(encoding="utf-8"), str(path.relative_to(ROOT)),
              extra=() if "prompts" in path.parts else ("ルーブリック",))
    assert not (ROOT / "docs" / "teacher_guide.md").exists()


def test_section_names_given_to_the_model():
    """The headings the pipeline puts around the work's text in the model's input."""
    from pfb_app.pipeline import document, video
    for module in (document, video):
        source = Path(module.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            if line.lstrip().startswith("#"):
                continue
            clean(line, f"{module.__name__}: {line.strip()}", extra=())


def _docstrings(tree: ast.AST) -> set[int]:
    return {id(node.body[0].value) for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)}


def test_no_school_words_in_any_string_of_the_app():
    """Every string literal in src/ (f-string parts too), docstrings aside: errors and flashes
    that no rendered screen above happens to show are covered as well."""
    src = ROOT / "src" / "pfb_app"
    found = []
    for path in sorted(src.rglob("*.py")):
        if path.relative_to(src) == NAME_STRIPPING:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docs):
                words_in = [w for w in SCHOOL_WORDS if w in node.value]
                if words_in:
                    found.append(f"{path.relative_to(src)}:{node.lineno} {words_in}")
    assert not found, found


def test_export_page_uses_sports_words_for_a_sports_work(env, works, monkeypatch):
    names = []
    real = st.download_button

    def recording(*args, **kwargs):
        names.append(kwargs.get("file_name"))
        return real(*args, **kwargs)
    monkeypatch.setattr(st, "download_button", recording)
    page = AppTest.from_file(str(UI / "pages" / "export.py"), default_timeout=60)
    page.session_state["current_assignment"] = works["sports_review"]
    page = ok(page.run())
    assert [b.label for b in page.button] == ["コーチ用PDFを作成", "選手用PDFを作成",
                                              "一人ずつのPDF（ZIPでまとめて）を作成"]
    for key in ("make-teacher", "make-student", "make-each"):
        page = ok(page.button(key=key).click().run())
    assert [n.rsplit("_", 1)[1] for n in names] == ["コーチ用.pdf", "選手用.pdf", "選手ごと.zip"]


def test_review_video_screens_are_neutral(env, tmp_path):
    """The review videos and the review player (cards, bands, the page and its data): the same
    word rules as every other screen (rule of 2026-10-22)."""
    from test_review_video import _numbered, all_screen_text
    from pfb_app.review_video import collect
    settings, repo, sid = _numbered(env)
    pkg = collect.build_package(repo, settings, sid)
    text = all_screen_text(pkg, pkg.people[0], tmp_path)
    clean(text, "review video / player")
