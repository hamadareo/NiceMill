"""ナイスミル: the app's name, logo, tab title and icon, and the brand files they use."""
import ast
import importlib.util
import re
from pathlib import Path

import pfb_app.ui
from pfb_app.ui.wizard import theme
from test_wizard_ui_basic import env  # noqa: F401 (env)

UI = Path(pfb_app.ui.__file__).parent
REPO = UI.parents[2]
BRAND = UI / "static" / "brand"
FONTS = UI / "static" / "fonts"


def _brand_script():
    spec = importlib.util.spec_from_file_location("brand_assets", REPO / "scripts" / "brand_assets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_name_and_tagline():
    assert theme.APP_NAME == "ナイスミル"
    assert theme.TAGLINE == "ナイスプレーを、全員ぶん。"


def test_app_sets_the_title_icon_and_logo_before_the_navigation():
    source = (UI / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "st":
            calls.setdefault(node.func.attr, node)
    config, logo, nav = calls["set_page_config"], calls["logo"], calls["navigation"]
    kw = {k.arg: ast.unparse(k.value) for k in config.keywords}
    assert kw["page_title"] == "theme.APP_NAME"
    assert "theme.MARK_PNG" in kw["page_icon"]
    lkw = {k.arg: ast.unparse(k.value) for k in logo.keywords}
    assert lkw["size"] == "'large'" and "theme.MARK" in lkw["icon_image"]
    assert "theme.WORDMARK_ONDARK" in ast.unparse(logo.args[0])
    # the logo comes before the navigation and before the sign-in gate (every screen has it)
    assert logo.lineno < nav.lineno
    assert logo.lineno < source.splitlines().index(
        next(l for l in source.splitlines() if l.startswith("auth_ui.gate("))) + 1
    assert "フィードバック作成" not in source


def test_the_brand_files_exist_and_need_no_web_font():
    for name in (theme.WORDMARK, theme.WORDMARK_ONDARK, theme.MARK, theme.MARK_MONO,
                 "nicemiru_wordmark_mono.svg"):
        svg = (BRAND / name).read_text(encoding="utf-8")
        assert svg.startswith("<svg") and 'aria-label="ナイスミル"' in svg, name
        assert "<text" not in svg and "font-family" not in svg, name     # letters as outlines
        assert not re.search(r"https?:|href", svg.replace("http://www.w3.org/2000/svg", "")), name
    for name in (theme.WORDMARK, theme.WORDMARK_ONDARK):
        assert "<path d=" in (BRAND / name).read_text(encoding="utf-8")
    png = (BRAND / theme.MARK_PNG).read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    with Image.open(BRAND / theme.MARK_PNG) as img:
        assert img.size == (64, 64)


def test_the_vendored_marks_match_their_generator():
    mod = _brand_script()
    assert (BRAND / theme.MARK).read_text(encoding="utf-8") == mod.mark_svg()
    assert (BRAND / theme.MARK_MONO).read_text(encoding="utf-8") == mod.mark_svg(mono=True)
    assert mod.BALL == theme.TOKENS["ball"] and mod.COURT == theme.TOKENS["board"]
    assert mod.AMBER == theme.TOKENS["amber"]


def test_the_static_fonts_exist():
    for _family, _weight, name in theme.FONT_FACES:
        assert (FONTS / name).is_file(), name


def test_the_signed_out_screens_show_the_name():
    html = theme.brand_html()
    assert f'alt="{theme.APP_NAME}"' in html and theme.TAGLINE in html
    assert f'src="{theme.BRAND_URL}{theme.WORDMARK}"' in html
    assert theme.BRAND_URL.startswith("app/static/")           # served by the app itself
    from pfb_app.ui import auth_ui
    assert auth_ui.APP == theme.APP_NAME
    src = (UI / "auth_ui.py").read_text(encoding="utf-8")
    assert "theme.brand_html()" in src


def test_the_tab_titles_carry_the_name():
    from pfb_app.ui.wizard import notify
    assert notify.APP == theme.APP_NAME
    assert notify.home_title([]) == "ナイスミル"


def test_ops_mail_footer_uses_the_name():
    from pfb_app.ops import alerts
    assert alerts.FOOTER.startswith(theme.APP_NAME)


def test_the_pdfs_carry_the_mono_mark_and_name_in_the_footer():
    from pfb_app.export import pdf
    assert pdf.BRAND_NAME == theme.APP_NAME
    assert pdf.BRAND_MARK == BRAND / theme.MARK_MONO and pdf.BRAND_MARK.is_file()
    base = (REPO / "src" / "pfb_app" / "export" / "templates" / "_base.html.j2").read_text("utf-8")
    bottom_left = next(l for l in base.splitlines() if "@bottom-left" in l)
    assert "src(brand_mark)" in bottom_left and "brand_name" in bottom_left


def test_the_user_guide_uses_the_name():
    guide = (REPO / "docs" / "user_guide.md").read_text(encoding="utf-8")
    assert guide.startswith("# ナイスミル")
    assert "フィードバック作成アプリ" not in guide and "緑色" not in guide and "花丸" not in guide


def test_every_pdf_page_has_the_name_in_its_footer(env):
    """Rendered on a synthetic work: the handout and the evaluator PDF, every page."""
    import io

    import pypdfium2 as pdfium

    from pdf_synthetic import video_group, work
    from pfb_app.export import pdf

    _settings, repo = env
    aid = work(repo)
    video_group(_settings, repo, aid, people=2, seconds=30)
    for data in (pdf.render_student_pdf(repo, aid), pdf.render_teacher_pdf(repo, aid)):
        doc = pdfium.PdfDocument(io.BytesIO(data))
        texts = [doc[i].get_textpage().get_text_range() for i in range(len(doc))]
        assert texts and all("ナイスミル" in t for t in texts)
