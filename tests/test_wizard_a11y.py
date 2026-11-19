"""Wizard accessibility and copy pass over every step screen (AppTest on ui/app.py).

Only synthetic data: invented names and texts, tiny black PNGs for sheets and frames.

Every screen state of both flows (video with the tracking and the paper method, documents) is
rendered once (module fixture); the checks then run over all of them:
- the 「この画面の使い方」 expander is there, with an 「うまくいかないとき」 line on steps 2–6;
- no internal term or English UI word in the rendered text (labels, captions, messages,
  aria-labels and titles too);
- every status chip carries its word;
- the guidance under each title is one sentence;
- exactly one primary button.
"""
import re
import sys

import pytest
import streamlit as st
import yaml
from streamlit import config as st_config

from pfb_app.config import Settings
from pfb_app.domain.models import Stage
from pfb_app.domain.vocab import mark_name
from pfb_app.services import review
from pfb_app.services.intake import register_documents
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.wizard import live, step_review, step_wait, theme
from test_wizard_ui_basic import (FORBIDDEN, PRIMARY_LABELS, UI, app, doc_assignment, html,
                                  pdf_bytes, primaries, register, video_assignment)
from test_wizard_ui_people_wait import confirmed_and_analysing, group, save, track, worker_alive
from test_wizard_ui_review_pdf import documents, reviewable, subject

MAX_MB = 512        # not the default: the caption must come from the server's limit
INTERNAL = re.compile(r"stage_run|\bjobs?\b|engine|\bP\d+\b|model_key|ジョブ|エンジン|PFB3x3",
                      re.I)
ENGLISH_UI = re.compile(r"\b(inbox|submit|upload|browse|drag and drop|limit|per file|"
                        r"press enter|running)\b", re.I)
HELP = "この画面の使い方"
TROUBLE = "うまくいかないとき"


def _rendered(at, skip_keys: tuple[str, ...] = ()) -> str:
    """What a teacher can read or hear: the HTML text and its aria-labels and titles, the
    widget labels, options, help texts and captions, and every message."""
    page = html(at)
    attrs = re.findall(r'(?:aria-label|title)="([^"]*)"', page)
    text = re.sub(r"<[^>]+>", " ", page)
    parts = [text, *attrs]
    for kind in ("markdown", "caption", "info", "warning", "error", "success", "expander"):
        parts += [getattr(e, "value", "") or getattr(e, "label", "") for e in at.get(kind)]
    parts += [b.label + (b.help or "") for b in at.button]
    parts += [b.proto.label + (b.proto.help or "") for b in at.get("download_button")]
    parts += [link.proto.label for link in at.get("page_link")]
    for w in [*at.radio, *at.selectbox]:
        if (w.key or "").startswith(skip_keys):
            continue
        parts += [w.label, w.help or "", *map(str, w.options)]
    for w in [*at.text_input, *at.text_area, *at.number_input, *at.checkbox]:
        parts += [w.label, w.help or ""]
    for up in at.get("file_uploader"):
        parts += [up.proto.label, up.proto.help or ""]
    return "\n".join(str(p) for p in parts)


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    """{name: AppTest} of every screen state, rendered once against a synthetic DB."""
    tmp = tmp_path_factory.mktemp("a11y")
    settings = Settings(data_dir=tmp / "data", inbox_dir=tmp / "inbox",
                        engine_dir=tmp / "engines" / "pfb3x3")
    cfg = tmp / "pfb.yaml"
    cfg.write_text(yaml.safe_dump({"data_dir": str(settings.data_dir),
                                   "inbox_dir": str(settings.inbox_dir),
                                   "engine_dir": str(settings.engine_dir)}), encoding="utf-8")
    main = sys.modules.get("__main__")
    out: dict = {}
    limit = st.get_option("server.maxUploadSize")
    st_config.set_option("server.maxUploadSize", MAX_MB)   # as start.sh's flag sets it
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PFB_CONFIG", str(cfg))
        conn = connect(files.db_path(settings))
        init_schema(conn)
        repo = Repo(conn, settings)
        worker_alive(settings)
        out["home, nothing yet"] = app()
        out["new: step 1"] = app(session={"assignment_id": None, "step": 1,
                                          "submission_id": None})
        empty = video_assignment(repo, "動画まだ")
        out["video: step 1 editable"] = app(session={"assignment_id": empty, "step": 1,
                                                     "submission_id": None})
        out["video: step 2 form"] = app(session={"assignment_id": empty})
        out["video: step 2 form, server folder"] = app(session={"assignment_id": empty},
                                                       extra={"v_source": "inbox"})
        started = video_assignment(repo, "追跡中")
        register(settings, repo, started)
        out["video: step 1 locked"] = app(session={"assignment_id": started, "step": 1,
                                                   "submission_id": None})
        out["video: step 2 summary"] = app(session={"assignment_id": started, "step": 2,
                                                    "submission_id": None})
        out["video: step 3 tracking"] = app(session={"assignment_id": started})
        mapping = video_assignment(repo, "つなぐ")
        msid = group(repo, mapping)
        track(settings, repo, msid)
        out["video: step 3 form"] = app(session={"assignment_id": mapping})
        paper = video_assignment(repo, "論文方式")
        psid = group(repo, paper, engine="paper")
        save(repo, psid, Stage.identity, {"participants": [
            {"participant_id": "Pa", "appearance": "赤い帽子", "initial_position": "左前",
             "identity_confidence": "high", "verification_points": ["帽子"]}]}, camera="cam1")
        out["paper: step 3"] = app(session={"assignment_id": paper})
        analysing = video_assignment(repo, "解析中")
        confirmed_and_analysing(settings, repo, analysing)
        out["video: step 4"] = app(session={"assignment_id": analysing})
        stopped = video_assignment(repo, "止まった")
        ssid = confirmed_and_analysing(settings, repo, stopped)
        repo.conn.execute("UPDATE jobs SET status='failed', error='VideoError: 読めません' "
                          "WHERE submission_id=?", (ssid,))
        out["video: step 4 stopped"] = app(session={"assignment_id": stopped, "step": 4,
                                                    "submission_id": None})
        checking = video_assignment(repo, "確認")
        csid = reviewable(settings, repo, checking, insufficient=("Pc",))
        out["video: step 5"] = app(session={"assignment_id": checking})
        out["video: step 5 editing"] = app(
            session={"assignment_id": checking},
            extra={step_review.EDITING: subject(repo, csid, "Pa")["id"]})
        review.approve(repo, subject(repo, csid, "Pa")["id"])
        out["video: step 6"] = app(session={"assignment_id": checking, "step": 6,
                                            "submission_id": None})
        for s in repo.list_subjects(csid):
            if s["approval_status"] != "approved" and s["participant_id"] != "Pc":
                review.approve(repo, s["id"])
        out["video: step 5 all done"] = app(session={"assignment_id": checking, "step": 5,
                                                     "submission_id": None})
        docs_new = doc_assignment(repo, "書類まだ")
        out["documents: step 2 form"] = app(session={"assignment_id": docs_new})
        docs_wait = doc_assignment(repo, "書類解析中")
        register_documents(repo, settings, docs_wait, [("山田太郎.pdf", pdf_bytes())], None)
        out["documents: step 2 summary"] = app(session={"assignment_id": docs_wait, "step": 2,
                                                        "submission_id": None})
        out["documents: step 3"] = app(session={"assignment_id": docs_wait})
        docs_review = documents(settings, repo)
        out["documents: step 4"] = app(session={"assignment_id": docs_review})
        first = repo.list_subjects(repo.list_submissions(docs_review)[0]["id"])[0]
        review.approve(repo, first["id"])
        out["documents: step 5"] = app(session={"assignment_id": docs_review, "step": 5,
                                                "submission_id": None})
        out["home, with works"] = app()
        out["_mapping_track"] = repo.latest_run(msid, Stage.track, "cam1")
        conn.close()
    st_config.set_option("server.maxUploadSize", limit)
    if main is not None:
        sys.modules["__main__"] = main    # see conftest.keep_main_module
    return out


def _each(screens):
    for name, at in screens.items():
        if not name.startswith("_"):
            yield name, at


def test_every_screen_renders_without_an_exception(screens):
    for name, at in _each(screens):
        assert not at.exception, (name, [e.value for e in at.exception])


# nothing to press while the analysis runs: only the secondary 「ホームへ戻る（解析は続きます）」
NO_PRIMARY = ("video: step 4", "documents: step 3")


def test_every_screen_has_one_primary_button(screens):
    for name, at in _each(screens):
        if name in NO_PRIMARY:
            assert primaries(at) == [], (name, primaries(at))
            continue
        assert len(primaries(at)) == 1, (name, primaries(at))
        assert primaries(at)[0] in PRIMARY_LABELS, (name, primaries(at))


def test_every_screen_has_the_help_expander(screens):
    for name, at in _each(screens):
        assert any(e.label == HELP for e in at.expander), name


def test_steps_2_to_6_say_what_to_do_when_something_goes_wrong(screens):
    for name, at in _each(screens):
        if name.startswith("home") or "step 1" in name:
            continue
        helps = [m.value for m in at.markdown if TROUBLE in m.value]
        assert helps, name


def test_no_internal_terms_or_english_words_on_any_screen(screens):
    for name, at in _each(screens):
        # persons are 「n番の人」 everywhere, the people form's selects too
        text = _rendered(at)
        assert not INTERNAL.search(text), (name, INTERNAL.search(text).group())
        assert not ENGLISH_UI.search(text), (name, ENGLISH_UI.search(text).group())
        found = [w for w in FORBIDDEN if w in text]   # incl. the round-2 old words
        assert not found, (name, found)


def test_the_people_form_uses_the_numbers_on_the_photo(screens):
    at, run = screens["video: step 3 form"], screens["_mapping_track"]
    labels = [w.label for w in at.selectbox if (w.key or "").startswith("pfb_map_")]
    assert labels == [mark_name(label) for label in run.output["labels"]]


def test_every_status_chip_has_its_word(screens):
    chips = 0
    for name, at in _each(screens):
        page = html(at)
        opened = page.count('<span class="pfb-chip ')
        worded = re.findall(r'<span class="pfb-chip pfb-chip--\w+"><span class="pfb-chip__mark" '
                            r'aria-hidden="true"></span>([^<]+)</span>', page)
        assert opened == len(worded), name
        assert all(w.strip() for w in worded), name
        chips += opened
    assert chips > 20         # the check saw the chips of the home, waiting and review screens


def test_the_guidance_under_each_title_is_one_sentence(screens):
    for name, at in _each(screens):
        guides = re.findall(r'<p class="pfb-card-guide">([^<]*)</p>', html(at))
        assert guides or name.startswith("home, with"), name
        for g in guides:
            assert g.endswith("。") and g.count("。") == 1, (name, g)


def _walk(node):
    yield node
    for child in getattr(node, "children", {}).values():
        yield from _walk(child)


def test_step_5_puts_the_buttons_right_under_the_text(screens):
    """The evidence comes after 「承認する」, so the button is in the first view."""
    for name in ("video: step 5", "documents: step 4"):
        order = []
        for node in _walk(screens[name].main):
            if node.type == "button":
                order.append(("button", node.label))
            elif node.type == "html":
                order.append(("html", node.proto.body))
        approve = order.index(("button", "承認する"))
        text = next(i for i, (k, v) in enumerate(order) if k == "html" and "pfb-text" in v)
        evidence = next(i for i, (k, v) in enumerate(order)
                        if k == "html" and "この文章の根拠" in v)
        assert text < approve < evidence, name


def test_waiting_screens_say_there_is_nothing_to_press(screens):
    at = screens["video: step 3 tracking"]
    assert all(b.disabled for b in at.button if b.proto.type == "primary")
    assert live.NOTHING_TO_PRESS in [c.value for c in at.caption]
    for name in NO_PRIMARY:     # the analysis goes on: the coach can go home meanwhile
        home = screens[name].button(key="pfb_wait_next")
        assert home.label == step_wait.HOME_WHILE_RUNNING and not home.disabled, name


def test_no_apologies_on_any_screen(screens):
    for name, at in _each(screens):
        text = _rendered(at)
        assert not re.search(r"すみません|申し訳|ごめん|失礼", text), name


def test_file_uploaders_say_the_servers_size_limit_in_japanese(screens):
    for name in ("new: step 1", "video: step 2 form", "documents: step 2 form"):
        at = screens[name]
        assert at.get("file_uploader"), name
        notes = [c.value for c in at.caption if "1ファイル最大" in c.value]
        assert notes and all(f"1ファイル最大 {MAX_MB} MB" in n for n in notes), (name, notes)


def test_the_focus_ring_and_the_reduced_motion_guard_are_in_the_theme():
    css = theme.theme_css()
    assert re.search(r":focus-visible\s*\{[^}]*outline:\s*3px solid", css)
    start = css.index("@media (prefers-reduced-motion: reduce)")
    reduce = css[start:css.index("\n}\n", start)]          # the whole media block
    assert re.search(r"animation-duration:\s*0\.01ms", reduce)
    assert re.search(r"transition-duration:\s*0\.01ms", reduce)
    assert "@media (prefers-reduced-motion: no-preference)" in css     # the stamp


def test_streamlits_english_chrome_is_hidden_or_in_japanese():
    css = theme.theme_css()
    # the uploader: the size line is ours (a caption), the button says ファイルを選ぶ
    assert re.search(r'\[data-testid="stFileUploaderDropzoneInstructions"\]\s*\{\s*display:\s*none',
                     css)
    assert 'content: "ファイルを選ぶ"' in css
    # "Press Enter to apply" and the like
    assert '[data-testid="InputInstructions"]' in css and "反映" in css
    # "Running..." / "Stop" in the corner
    assert '[data-testid="stStatusWidget"]' in css and 'content: "処理中' in css
    config = (UI.parents[2] / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert re.search(r'^\[client\]\s*\n(?:.*\n)*?toolbarMode\s*=\s*"minimal"', config, re.M)


def test_the_registration_form_does_not_submit_on_enter(screens):
    at = screens["video: step 2 form"]
    forms = at.get("form")
    assert forms and all(f.proto.form.enter_to_submit is False for f in forms)


def test_selectboxes_without_a_choice_have_a_japanese_placeholder(screens):
    assert any(w.index is None for w in screens["video: step 2 form, server folder"].selectbox)
    for name, at in _each(screens):
        for w in at.selectbox:
            if w.index is None:
                assert w.proto.placeholder and not re.search(r"[A-Za-z]", w.proto.placeholder), \
                    (name, w.key, w.proto.placeholder)


def test_the_user_guide_covers_both_flows_without_real_data():
    guide = (UI.parents[2] / "docs" / "user_guide.md").read_text(encoding="utf-8")
    for label in ("活動を決める", "動画を入れる", "人を確かめる（任意）", "参加者を確認", "解析を待つ",
                  "確認して承認", "PDFを受け取る", "書類を入れる", "管理用の画面", TROUBLE,
                  "はじめての方へ", "見本の作業", "この解析の確かさ"):
        assert label in guide, label
    for old in ("詳細画面", "人物の追跡", "根拠不足", "未割り当て"):   # replaced on screen
        assert old not in guide, old
    assert not re.search(r"!\[|\.png|\.jpg|\.mp4", guide)      # no screenshots, no files
    assert not INTERNAL.search(guide)

