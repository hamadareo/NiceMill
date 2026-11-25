"""NiceMill theme (「A コートサイド」): tokens, self-hosted fonts, brand, stepper, chips, stamp."""
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import pfb_app.ui
from pfb_app.ui.wizard import theme

UI = Path(pfb_app.ui.__file__).parent
REPO = UI.parents[2]

EXPECTED_TOKENS = {
    "board": "#0F2D5C", "board-2": "#21477F", "paper": "#EDF1F6", "ink": "#13202F",
    "muted": "#4D5A6C", "line": "#C9D2DE", "wait": "#A87200", "sky": "#1F6FB2", "pen": "#0F2D5C",
    "chalk": "#F5F8FC", "chalk-2": "#C8D4E6", "tray": "#FFB000", "mist": "#E6ECF4",
    "chalk-3": "#A3B7D6", "chalk-4": "#86A0C8", "chalk-5": "#6A88B8",   # the chart on navy
    "ball": "#C8410C", "ball-2": "#A9370A", "amber": "#FFB000",
}
FACES = {("M PLUS 1", 400), ("M PLUS 1", 500), ("M PLUS 1", 700), ("M PLUS 1", 800),
         ("Dela Gothic One", 400), ("Barlow Condensed", 600), ("Barlow Condensed", 700)}
KINDS = ["your_turn", "processing", "waiting_gpu", "stopped", "approved", "insufficient", "done"]


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def no_internet(text: str) -> None:
    assert not re.search(r"https?:|//[a-z]", text, re.I), "the page must not reach the internet"


# --- tokens and CSS -------------------------------------------------------------------------

def test_tokens_are_exact():
    assert theme.TOKENS == EXPECTED_TOKENS
    css = theme.theme_css()
    for name, value in EXPECTED_TOKENS.items():
        assert f"--pfb-{name}: {value};" in css


def test_css_uses_only_the_vendored_fonts():
    css = theme.theme_css()
    no_internet(css)
    urls = re.findall(r"url\(([^)]*)\)", css)
    assert urls, "the @font-face rules must point at the vendored files"
    for url in urls:
        rel = url.strip("'\"")
        assert rel.startswith("app/static/")              # relative: served by the app itself
        assert (UI / rel.removeprefix("app/")).is_file(), rel
    assert css.count("@font-face") == len(FACES)
    for family, weight in FACES:
        assert re.search(rf'"{family}"; font-weight: {weight};', css), (family, weight)
    for old in ("Zen Kaku", "Klee One", "hanamaru"):
        assert old not in css, old


def test_css_has_focus_ring_and_reduced_motion():
    css = theme.theme_css()
    assert "<" not in css             # st.html's sanitiser would drop the whole sheet
    assert ":focus-visible" in css
    assert "prefers-reduced-motion" in css
    assert 'stBaseButton-primary' in css and 'stBaseButton-secondary' in css


def test_the_stamp_needs_no_file_and_moves_only_when_motion_is_welcome():
    css = theme.theme_css()
    assert "mask" not in css                       # CSS only: a ring, a line and the word
    rest = _strip_blocks(css, "@media (prefers-reduced-motion: no-preference)")
    assert re.search(r"pfb-stamp--animate[^{]*\{[^}]*animation", css)
    assert not re.search(r"pfb-stamp--animate[^{]*\{[^}]*animation", rest)


def test_fonts_are_vendored_with_their_license():
    fonts = UI / "static" / "fonts"
    licence = (fonts / "OFL.txt").read_text(encoding="utf-8")
    assert licence.count("SIL OPEN FONT LICENSE Version 1.1") == 3
    for family in ("M PLUS 1", "Dela Gothic One", "Barlow Condensed"):
        assert f"===== {family} =====" in licence, family
    assert {(f, w) for f, w, _ in theme.FONT_FACES} == FACES
    for name in theme.FONT_FILES:
        data = (fonts / name).read_bytes()
        assert data[:4] == b"wOF2", name
        assert len(data) < 1_200_000, name           # subset: the screens stay light
    # only the fonts in use are vendored
    assert {p.name for p in fonts.glob("*.woff2")} == set(theme.FONT_FILES)
    third = (REPO / "docs" / "THIRD_PARTY.md").read_text(encoding="utf-8")
    for family in ("M PLUS 1", "Dela Gothic One", "Barlow Condensed"):
        assert family in third, family


def test_the_japanese_fonts_cover_the_screens_words():
    from fontTools.ttLib import TTFont
    words = "NiceMillヒントホーム承認確かめて解析中あなたの番試合・練習のまとめ一人ずつ「」（）％0123456789"
    for name in ("MPLUS1-Regular.woff2", "MPLUS1-Bold.woff2", "DelaGothicOne-Regular.woff2"):
        cmap = TTFont(UI / "static" / "fonts" / name).getBestCmap()
        missing = [c for c in words if ord(c) not in cmap]
        assert not missing, (name, missing)


def test_text_colours_meet_contrast():
    t = theme.TOKENS
    for fg in ("ink", "muted", "board", "sky", "pen"):
        assert contrast(t[fg], t["paper"]) >= 4.5, fg
        assert contrast(t[fg], "#FFFFFF") >= 4.5, fg
    for bg in ("ball", "ball-2"):                           # the primary button's label
        assert contrast("#FFFFFF", t[bg]) >= 4.5, bg
    assert contrast("#FFFFFF", t["board"]) >= 4.5          # text on navy: band, sidebar
    assert contrast(t["amber"], t["board"]) >= 4.5         # the scoreboard numbers
    assert contrast(t["amber"], t["board-2"]) >= 4.5       # amber on the sidebar's current page
    for fg in ("#AFC0DA", "#C8D4E6", "#DCE4F0"):            # quiet text on navy (band, metric)
        assert contrast(fg, t["board"]) >= 4.5, fg
    assert contrast("#AFC0DA", t["board-2"]) >= 4.5
    for fg in ("chalk", "chalk-2"):                          # the check card's board
        assert contrast(t[fg], t["board"]) >= 4.5, fg
    for fg in ("ink", "board"):                              # a drop zone under the pointer
        assert contrast(t[fg], t["mist"]) >= 4.5, fg
    secondary = streamlit_theme()["secondaryBackgroundColor"]   # sidebar, inputs
    for fg in ("ink", "muted", "board"):
        assert contrast(t[fg], secondary) >= 4.5, fg
    for kind, (fg, bg, _mark) in theme.CHIP_COLOURS.items():
        assert contrast(fg, bg) >= 4.5, kind


def streamlit_config() -> dict:
    try:
        import tomllib
    except ImportError:                                     # Python 3.10: pytest's tomli
        import tomli as tomllib
    return tomllib.loads((REPO / ".streamlit" / "config.toml").read_text(encoding="utf-8"))


def streamlit_theme() -> dict:
    return streamlit_config()["theme"]


def test_streamlit_config_matches_the_theme():
    cfg = streamlit_config()
    assert cfg["server"]["enableStaticServing"] is True
    assert cfg["browser"]["gatherUsageStats"] is False      # kept from the live copy
    th = cfg["theme"]
    assert th["primaryColor"] == EXPECTED_TOKENS["ball"]
    assert th["backgroundColor"] == EXPECTED_TOKENS["paper"]
    assert th["textColor"] == EXPECTED_TOKENS["ink"]
    assert th["linkColor"] == EXPECTED_TOKENS["board"]
    assert th["font"].startswith("M PLUS 1") and th["headingFont"].startswith("Dela Gothic One")
    side = th["sidebar"]
    assert side["backgroundColor"] == EXPECTED_TOKENS["board"]
    assert side["secondaryBackgroundColor"] == EXPECTED_TOKENS["board-2"]
    assert contrast(side["textColor"], side["backgroundColor"]) >= 4.5
    no_internet(str(cfg))
    for face in th["fontFaces"]:
        assert (UI / face["url"].removeprefix("app/")).is_file()
    assert {(f["family"], f["weight"]) for f in th["fontFaces"]} == FACES


# --- components -----------------------------------------------------------------------------

LABELS = ["課題を決める", "動画を入れる", "人を確かめる（任意）", "解析を待つ", "確認して承認", "PDFを受け取る"]


def items(html: str) -> list[str]:
    return re.findall(r"<li\b.*?</li>", html, re.S)


def test_stepper_marks_done_and_current_steps():
    html = theme.stepper_html(LABELS, current=2, done={0, 1})
    steps = items(html)
    assert len(steps) == 6
    for label, step in zip(LABELS, steps):
        assert label in step
    assert html.count('aria-current="step"') == 1
    assert 'aria-current="step"' in steps[2]
    for i in (0, 1):
        assert "✓" in steps[i] and "済み" in steps[i]
    assert "いまここ" in steps[2] and "✓" not in steps[2]
    for i in (3, 4, 5):
        assert "これから" in steps[i] and "✓" not in steps[i]
    no_internet(html)


def test_stepper_current_step_can_also_be_done():
    """Going back to a finished step: it is the current one and still shows its check."""
    steps = items(theme.stepper_html(LABELS, current=0, done={0, 1, 2}))
    assert 'aria-current="step"' in steps[0] and "✓" in steps[0] and "いまここ" in steps[0]


def test_stepper_escapes_labels():
    html = theme.stepper_html(["<b>x</b>"], current=0, done=set())
    assert "<b>x</b>" not in html and "&lt;b&gt;" in html


@pytest.mark.parametrize("kind", KINDS)
def test_status_chip_always_includes_the_text(kind):
    html = theme.status_chip(kind, "あなたの番です")
    assert "あなたの番です" in html
    assert f"pfb-chip--{kind}" in html
    no_internet(html)


def test_status_chip_rejects_unknown_kind_and_escapes():
    with pytest.raises(ValueError):
        theme.status_chip("job_failed", "x")
    assert "<script>" not in theme.status_chip("done", "<script>")


def test_approval_stamp_is_labelled_and_offline():
    html = theme.approval_stamp()
    assert 'aria-label="承認済み"' in html and "承認" in html
    assert "<svg" not in html          # st.html's sanitiser drops inline SVG; the ring is CSS
    assert "pfb-stamp--animate" in html
    assert "pfb-stamp--animate" not in theme.approval_stamp(animate=False)
    no_internet(html)


# --- AppTest ------------------------------------------------------------------------------

def page_twice():
    from pfb_app.ui.wizard import theme as t
    t.inject_theme()
    t.inject_theme()
    t.stepper(["課題を決める", "書類を入れる"], current=1, done={0})
    with t.step_card("書類を入れてください", "提出された書類を選びます。"):
        import streamlit as st
        st.button("登録して解析を始める", type="primary")
    t.help_expander(["一行目", "二行目", "三行目"])


def test_inject_theme_once_per_run():
    at = AppTest.from_function(page_twice).run()
    assert not at.exception

    def css_count():
        return sum(":root" in el.proto.body and "--pfb-board" in el.proto.body
                   for el in at.get("html"))

    assert css_count() == 1
    at.run()                                       # a rerun injects it again, still once
    assert not at.exception
    assert css_count() == 1
    bodies = [el.proto.body for el in at.get("html")]
    assert any('aria-current="step"' in b for b in bodies)
    assert any("書類を入れてください" in b for b in bodies)
    assert at.button[0].label == "登録して解析を始める"
    assert at.expander[0].label == "この画面の使い方"
    assert [m.value for m in at.expander[0].markdown] == ["- 一行目\n- 二行目\n- 三行目"]


def test_bar_motion_only_without_reduced_motion():
    css = theme.theme_css()
    assert ".pfb-bar" in css and "::-webkit-progress-value" in css
    assert "::-webkit-progress-bar" in css and "::-moz-progress-bar" in css
    assert re.search(r"pfb-bar--moving[^{]*\{[^}]*animation", css)   # it does pulse somewhere
    # with every no-preference block cut out (brace matching), nothing animates the moving bar
    rest = _strip_blocks(css, "@media (prefers-reduced-motion: no-preference)")
    assert not re.search(r"pfb-bar--moving[^{]*\{[^}]*animation", rest)
    assert "repeating-linear-gradient" not in css


def _strip_blocks(css: str, head: str) -> str:
    """css without every block that starts with ``head``, nested braces included."""
    while (start := css.find(head)) >= 0:
        i, depth = css.index("{", start), 0
        while True:
            depth += {"{": 1, "}": -1}.get(css[i], 0)
            if depth == 0:
                break
            i += 1
        css = css[:start] + css[i + 1:]
    return css


def test_bars_have_a_forced_colors_rule_and_an_edge_on_the_paper():
    css = theme.theme_css()
    start = css.index("@media (forced-colors: active)")
    block = css[start:css.index("\n}\n", start)]
    assert "Highlight" in block and "CanvasText" in block
    edge = re.search(r"--pfb-bar-edge:\s*(#[0-9A-Fa-f]{6})", css).group(1)
    assert contrast(edge, EXPECTED_TOKENS["paper"]) >= 3


def test_bar_fills_keep_3_to_1_against_the_track_and_never_use_the_pen():
    css = theme.theme_css()
    track = re.search(r"--pfb-bar-track:\s*(#[0-9A-Fa-f]{6})", css).group(1)
    for name in ("board", "wait", "muted"):
        assert contrast(EXPECTED_TOKENS[name], track) >= 3, name
    bar_rules = re.findall(r"[^{}]*\.pfb-bar[^{]*\{[^}]*\}", css)
    assert bar_rules and not any("pfb-pen" in r for r in bar_rules)


def test_the_page_clears_the_fixed_header_and_captions_keep_their_contrast():
    css = theme.theme_css()
    # the top of 「ホームへ」 was hidden under the 60 px header at 2.5rem
    top = re.search(r'stMainBlockContainer"\] \{ padding-top: ([\d.]+)rem', css)
    assert top and float(top.group(1)) * 16 >= 64
    # Streamlit's 60 percent caption opacity gave about 3.9:1 on the paper: full muted ink
    rule = re.search(r'stMain"\] \[data-testid="stCaptionContainer"\] \{([^}]*)\}', css)
    assert rule and "opacity: 1" in rule.group(1) and "var(--pfb-muted)" in rule.group(1)
