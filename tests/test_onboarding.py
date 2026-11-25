"""Onboarding and honest quality (Phase 1 lane "onboarding"): the sample work, the in-app help,
「はじめての方へ」 and 「この解析の確かさ」 (AppTest on ui/app.py).

Only synthetic data: the fictional sample work, invented names and texts, generated frames."""
import json
import re
import xml.etree.ElementTree as ET

import pytest

from pfb_app.services import quality_stats, review, sample_work
from pfb_app.ui.wizard import help as guide
from pfb_app.ui.wizard import quality_view, sample_art, sample_view
from test_neutral_wording import SCHOOL_WORDS
from test_quality_stats import group, pfb
from test_wizard_ui_basic import (FORBIDDEN, app, env, html, ok, primaries,  # noqa: F401 (env)
                                  screen_text, video_assignment)
from test_wizard_ui_review_pdf import reviewable, subject

DATA_URI = re.compile(r'src="data:image/svg\+xml;base64,[A-Za-z0-9+/=]+"')


def text_of(at) -> str:
    """What the screen says (picture data taken out: base64 is not text)."""
    return DATA_URI.sub("", screen_text(at))


def plain(at) -> None:
    text = text_of(at)
    for word in (*FORBIDDEN, *SCHOOL_WORDS):
        assert word not in text, word


def links(at) -> list[str]:
    return [link.proto.label for link in at.get("page_link")]


def page(path: str, session: dict | None = None):
    at = app(session=session)
    at.switch_page(path)
    return ok(at.run())


# --- the sample work: fictional only -----------------------------------------------------------

def test_the_sample_work_is_fictional_and_holds_no_media_file():
    raw = sample_work.FIXTURE.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["synthetic"] is True
    assert not re.search(r"\.(mp4|mov|avi|mkv|jpe?g|png|gif|webp|heic)\b|https?://|/Users/|/home/",
                         raw, re.I)
    work = sample_work.load()
    assert {p.approval for p in work.people} == set(sample_work.APPROVALS)
    assert all(p.moments for p in work.people)
    for word in (*FORBIDDEN, *SCHOOL_WORDS):
        assert word not in raw, word


def test_a_sample_not_marked_synthetic_is_refused():
    data = json.loads(sample_work.FIXTURE.read_text(encoding="utf-8"))
    data["synthetic"] = False
    with pytest.raises(ValueError):
        sample_work.parse(data)


def test_the_sample_pictures_are_drawn_svg_without_any_embedded_image():
    work = sample_work.load()
    scenes = [work.overview, *(m.scene for p in work.people for m in p.moments)]
    for scene in scenes:
        doc = sample_art.scene_svg(work, scene, "見本の場面")
        root = ET.fromstring(doc)                    # well formed
        assert not root.findall(".//{http://www.w3.org/2000/svg}image")
        assert "href" not in doc and "<script" not in doc
    face = ET.fromstring(sample_art.player_svg("blue", 4, "見本の選手"))
    assert face.tag.endswith("svg")


def test_the_sample_shows_its_four_people_their_texts_and_moments():
    work = sample_work.load()
    page_html = sample_view.page_html(work)
    for person in work.people:
        assert person.text in page_html
        for m in person.moments:
            assert m.did in page_html and m.at in page_html
    assert page_html.count('alt="') >= sum(len(p.moments) for p in work.people)


# --- the sample on home -------------------------------------------------------------------------

def test_an_empty_home_shows_the_sample_work_with_one_primary(env):
    settings, repo = env
    at = ok(app())
    assert guide.SAMPLE_TITLE in html(at)
    assert primaries(at) == ["新しく始める"]
    assert sample_view.SEE in [b.label for b in at.button]
    assert guide.GUIDE_TITLE in links(at)
    plain(at)


def test_one_tap_removes_the_sample_and_the_guide_brings_it_back(env):
    settings, repo = env
    at = ok(app())
    next(b for b in at.button if b.label == sample_view.REMOVE).click().run()
    ok(at)
    assert not sample_work.is_shown(repo)
    assert sample_view.SEE not in [b.label for b in at.button]
    assert "見本を消しました" in text_of(at)
    at = page(guide.GUIDE_PAGE)
    assert primaries(at) == [guide_view_start()]
    next(b for b in at.button if b.label == "見本をもう一度出す").click().run()
    assert sample_work.is_shown(repo)


def guide_view_start() -> str:
    from pfb_app.ui.wizard import guide_view   # noqa: PLC0415
    return guide_view.START


def test_a_home_with_works_does_not_show_the_sample(env):
    settings, repo = env
    video_assignment(repo)
    at = ok(app())
    assert guide.SAMPLE_TITLE not in html(at)


# --- the sample page: read-only -----------------------------------------------------------------

def test_the_sample_page_is_read_only_and_leaves_no_trace(env):
    settings, repo = env
    at = page(guide.SAMPLE_PAGE)
    assert sample_work.load().title in html(at)
    assert primaries(at) == [sample_view.START]
    assert not (at.text_input or at.text_area or at.checkbox or at.selectbox or at.radio
                or at.number_input or at.toggle or at.download_button)
    assert {b.label for b in at.button} == {"ホームへ", sample_view.START, sample_view.REMOVE}
    plain(at)
    stats = quality_stats.measure(repo)
    assert stats.works == 0 and stats.people == 0      # never in the works tables or the numbers
    assert not repo.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]


def test_new_from_the_sample_opens_the_drop_zone(env):
    settings, repo = env
    at = page(guide.SAMPLE_PAGE)
    next(b for b in at.button if b.label == sample_view.START).click().run()
    ok(at)
    assert at.session_state["wizard"]["step"] == 1
    assert at.session_state["wizard"]["assignment_id"] is None
    assert "評価したいファイルを入れてください" in html(at)


def test_removing_the_sample_from_its_page_goes_home(env):
    settings, repo = env
    at = page(guide.SAMPLE_PAGE)
    next(b for b in at.button if b.label == sample_view.REMOVE).click().run()
    ok(at)
    assert not sample_work.is_shown(repo)
    assert guide.SAMPLE_TITLE not in html(at)       # home, without the card


# --- in-app help ---------------------------------------------------------------------------------

def test_every_step_key_has_a_short_plain_tip():
    for key in ("drop", "check", "assignment", "files", "people", "wait", "review", "pdf"):
        tip = guide.TIPS[key]
        assert 0 < len(tip) <= 90, (key, len(tip))
        for word in (*FORBIDDEN, *SCHOOL_WORDS):
            assert word not in tip, (key, word)


def test_the_drop_zone_has_the_filming_tip_and_the_guide_link(env):
    at = ok(app(session={"assignment_id": None, "step": 1, "submission_id": None}))
    assert "撮り方のこつ" in html(at)
    assert guide.GUIDE_TITLE in links(at)
    assert primaries(at) == ["入れて確かめる"]


def test_the_pdf_step_links_the_quality_page(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    review.approve(repo, subject(repo, sid, "Pa")["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert guide.QUALITY_TITLE in links(at) and guide.GUIDE_TITLE in links(at)
    assert guide.TIPS["pdf"] in html(at)
    assert len(primaries(at)) == 1
    plain(at)


def test_the_guide_page_has_the_flow_the_sample_and_the_user_guide(env):
    at = page(guide.GUIDE_PAGE)
    body = html(at)
    assert guide.GUIDE_TITLE in body and "動画の撮り方のこつ" in body
    assert primaries(at) == ["見本を見てみる"]
    expanders = [e.label for e in at.get("expander")]
    assert "どのくらい時間がかかりますか？" in expanders
    assert "ホーム" in expanders and "この解析の確かさ" in expanders     # docs/user_guide.md
    assert guide.QUALITY_TITLE in links(at)
    plain(at)


def test_the_guide_reads_the_user_guide_by_section():
    sections = dict(guide.guide_sections())
    assert "ホーム" in sections and "困ったときのまとめ" in sections
    assert all(body for body in sections.values())


def test_the_time_answer_uses_this_servers_own_median(env):
    from pfb_app.ui.wizard import guide_view   # noqa: PLC0415
    settings, repo = env
    assert guide_view.time_answer(quality_stats.measure(repo)) == guide_view.TIME_UNKNOWN
    stats = quality_stats.QualityStats(1, 1, dict.fromkeys(quality_stats.KINDS, 0),
                                       (600.0, 1800.0, 2400.0), 0, 0, 0, 0, "2026-10-01")
    assert "約30分" in guide_view.time_answer(stats) and "3グループ" in guide_view.time_answer(stats)


# --- 「この解析の確かさ」 -------------------------------------------------------------------------

def test_the_quality_page_says_when_nothing_is_measured_yet(env):
    at = page(guide.QUALITY_PAGE)
    body = html(at)
    assert quality_view.EMPTY in body
    assert "研究で確かめたこと" in body and "まだ確かめていないこと" in body
    assert not primaries(at)
    plain(at)


def test_the_quality_page_shows_this_servers_numbers(env):
    settings, repo = env
    _, sid, (a, b, c, d, e) = group(repo)
    for person in (a, b, c):
        repo.set_approval(person, pfb(repo, sid, person).id)
    repo.set_approval(d, pfb(repo, sid, d, source="teacher").id)
    at = page(guide.QUALITY_PAGE)
    body = DATA_URI.sub("", html(at))
    assert "下書きのまま承認" in body and "75" in body          # 3 of 4
    assert "承認した4人のうち" in body
    assert quality_view.FEW.format(n=4) in body               # rough with so few
    assert "承認した文章の内わけ" in html(at)
    plain(at)


def test_every_research_claim_names_its_source_and_no_unmeasured_claim_is_made():
    for claim in (*quality_view.VALIDATED, *quality_view.WEAK, *quality_view.NOT_YET):
        assert claim.source in (quality_view.RESEARCH, quality_view.DESIGN)
    said = "".join(c.text for c in quality_view.VALIDATED)
    for overclaim in ("正確", "専門家と同じ", "保証"):
        assert overclaim not in said
    assert "専門家" in "".join(c.text for c in quality_view.NOT_YET)


SPLIT = quality_stats.QualityStats(
    1, 12, {"as_is": 6, "edited": 2, "rewritten": 0, "short": 2, "auto": 2},
    (), 0, 0, 0, 0, "2026-10-01")


def test_the_split_bar_draws_one_segment_per_kind_with_gaps_auto_last():
    doc = quality_view.split_svg(SPLIT)
    root = ET.fromstring(doc)
    ns = "{http://www.w3.org/2000/svg}"
    fills = [r.get("fill") for r in root.iter(f"{ns}rect") if r.find(f"{ns}title") is not None]
    assert fills == [quality_view.RAMP[k] for k in ("as_is", "edited", "short", "auto")]
    widths = [float(r.get("width")) for r in root.iter(f"{ns}rect")
              if r.find(f"{ns}title") is not None]
    assert sum(widths) == pytest.approx(quality_view.BAR_W - 3 * quality_view.GAP)
    assert "50%" in doc and "下書きのまま承認 6人" in doc and "自動承認（人は未確認） 2人" in doc


def test_the_split_chart_alt_text_holds_every_value():
    alt = quality_view.split_alt(SPLIT)
    for label, n, pct in (("下書きのまま承認", 6, 50), ("直して承認", 2, 17),
                          ("自分で書いて承認", 0, 0), ("短い文で承認", 2, 17), ("自動承認（人は未確認）", 2, 17)):
        assert f"{label} {n}人（{pct}%）" in alt, label
    assert f'alt="{alt}"' in quality_view.split_html(SPLIT)


def test_the_chart_colours_are_theme_tokens():
    from pfb_app.ui.wizard.theme import TOKENS   # noqa: PLC0415
    assert set(quality_view.RAMP.values()) <= set(TOKENS.values())
    assert not re.search(r"#[0-9A-Fa-f]{6}\b|rgba?\(", quality_view.CSS.replace("#FFFFFF", ""))


def test_the_numbers_are_counted_at_most_once_a_minute_per_database(env, monkeypatch):
    settings, repo = env
    calls = []
    real = quality_stats.measure
    monkeypatch.setattr(quality_stats, "measure", lambda r: calls.append(1) or real(r))
    quality_view._measured.clear()
    first = quality_view.measured(repo)
    group(repo)
    assert quality_view.measured(repo) == first and len(calls) == 1
    quality_view._measured.clear()
    assert quality_view.measured(repo).works == 1 and len(calls) == 2


def test_the_sample_has_its_own_chip_and_does_not_promise_nothing_can_be_pressed():
    body = sample_view.page_html(sample_work.load())
    assert "pfb-chip--sample" in body and "押しても" not in body
    assert sample_view.VIEW_ONLY in body


def test_the_names_answer_is_about_video():
    from pfb_app.ui.wizard import guide_view   # noqa: PLC0415
    assert guide_view.NAMES.startswith("動画では")
    # sports works get number names (「#10」) automatically (services.auto_people.number_names)
    assert "自動ではつけません" not in guide_view.NAMES
    assert "「#10」" in guide_view.NAMES and "自動でつきます" in guide_view.NAMES


def test_the_accuracy_answer_tells_about_automatic_approval():
    from pfb_app.ui.wizard import guide_view   # noqa: PLC0415
    # the work's switch approves texts no person looked at: never claim every text is checked
    assert "機械チェックを通った文を自動承認する" in guide_view.ACCURACY
    assert "機械チェック済み・人は未確認" in guide_view.ACCURACY
    assert "保証しません" in guide_view.ACCURACY
    # whether they are handed out is the person's choice
    assert "確かめずに渡ります" not in guide_view.ACCURACY
    assert "人が確認しないまま自動承認されます" in guide_view.ACCURACY


def test_the_grounding_claim_leaves_out_short_and_handwritten_texts():
    grounded = next(c.text for c in quality_view.VALIDATED if "時刻つき" in c.text)
    assert "（短い文と手書きの文を除く）" in grounded


def test_the_quality_page_is_in_the_admin_section(env):
    at = ok(app())
    labels = links(at)
    assert guide.QUALITY_TITLE in labels and guide.GUIDE_TITLE in labels


def test_a_retry_on_the_status_page_is_recorded_for_the_first_try_count(env):
    from test_ui import populate, run_page   # noqa: PLC0415
    settings, repo = env
    ids = populate(settings, repo)
    at = run_page("status")
    at.button(key=f"retry-{ids['job']}").click().run()
    assert not at.exception
    entry = repo.latest_audit(quality_stats.RETRY_JOB, f"job:{ids['job']}")
    assert entry["detail"] == {"job_ids": [ids["job"]]}


def test_deleting_a_work_drops_it_from_the_cached_numbers_at_once(env):
    from pfb_app.services import deletion   # noqa: PLC0415
    settings, repo = env
    aid, _, _ = group(repo)
    quality_view._measured.clear()
    assert quality_view.measured(repo).works == 1
    deletion.delete_work(repo, settings, aid)
    assert quality_view.measured(repo).works == 0
