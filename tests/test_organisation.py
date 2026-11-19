"""団体の設定: name, logo (validated, size-limited, stored in the data dir), accent preset."""
import io
from pathlib import Path

import pytest
from PIL import Image

from pfb_app.config import Settings
from pfb_app.services import organisation as org
from pfb_app.services.organisation import ACCENTS, OrgError


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts")


def png(w=64, h=32, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (20, 90, 160)).save(buf, fmt)
    return buf.getvalue()


SVG = (b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20">'
       b'<rect width="40" height="20" fill="#1c5cab"/></svg>')


def luminance(hex_colour: str) -> float:
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def test_defaults_without_a_file(settings):
    o = org.load(settings)
    assert o.name == "" and o.logo is None and o.accent == org.DEFAULT_ACCENT
    assert o.accent_hex == ACCENTS[org.DEFAULT_ACCENT][1]


def test_every_accent_preset_reads_as_text_on_white():
    for key, (label, hex_colour) in ACCENTS.items():
        contrast = (1.0 + 0.05) / (luminance(hex_colour) + 0.05)
        assert contrast >= 4.5, (key, contrast)
        assert label


def test_save_name_accent_and_png_logo(settings):
    o = org.save(settings, name="  みなと町クラブ  ", accent="green", logo=png(), logo_name="l.png")
    assert o.name == "みなと町クラブ" and o.accent == "green"
    assert o.logo is not None and o.logo.is_file() and o.logo.parent == org.org_dir(settings)
    again = org.load(settings)
    assert (again.name, again.accent, again.logo) == (o.name, o.accent, o.logo)
    # a later save without a logo keeps it; remove_logo drops it
    assert org.save(settings, name="x", accent="blue").logo == o.logo
    assert org.save(settings, name="x", accent="blue", remove_logo=True).logo is None
    assert not o.logo.exists()


def test_jpeg_is_stored_as_png_and_svg_as_svg(settings):
    o = org.save(settings, name="a", accent="blue", logo=png(fmt="JPEG"), logo_name="l.jpg")
    assert o.logo.suffix == ".png" and Image.open(o.logo).format == "PNG"
    o = org.save(settings, name="a", accent="blue", logo=SVG, logo_name="l.svg")
    assert o.logo.suffix == ".svg"


def test_no_free_colours(settings):
    with pytest.raises(OrgError):
        org.save(settings, name="a", accent="#ff0000")


NS = "xmlns='http://www.w3.org/2000/svg' xmlns:xlink='http://www.w3.org/1999/xlink'"
USE_BOMB = (f"<svg {NS}><defs><g id='a0'><rect width='1' height='1'/></g>"
            + "".join(f"<g id='a{i}'>" + f"<use href='#a{i - 1}'/>" * 10 + "</g>"
                      for i in range(1, 9)) + "</defs><use href='#a8'/></svg>")
# the review's probes (svgprobe.py) and the allowlist ruling's cases
SVG_BYPASSES = {
    "style @import": f"<svg {NS}><style>@import 'http://evil.example/x.css';</style></svg>",
    "css escaped url": f"<svg {NS}><rect style='fill: \\75 rl(http://evil.example/a)'/></svg>",
    "style attr image-set": f"<svg {NS}><rect style=\"fill:image-set('http://evil.example/a' 1x)\"/></svg>",
    "style attr url": f"<svg {NS}><rect style='fill:url(http://evil.example/a)'/></svg>",
    "style attr import": f"<svg {NS}><rect style='@import x'/></svg>",
    "nested data svg": f"<svg {NS}><image href='data:image/svg+xml;base64,PHN2Zz48L3N2Zz4='/></svg>",
    "use external": f"<svg {NS}><use xlink:href='http://evil.example/a.svg#x'/></svg>",
    "use local": f"<svg {NS}><defs><g id='a'/></defs><use href='#a'/></svg>",
    "use bomb": USE_BOMB,
    "huge viewBox": f"<svg {NS} viewBox='0 0 1e30 1e30' width='1e9' height='1e9'><rect width='1e30' height='1e30'/></svg>",
    "viewBox just too big": f"<svg {NS} viewBox='0 0 10001 20'/>",
    "doctype late": f"<!--{'x' * 3000}--><!DOCTYPE svg SYSTEM 'http://evil.example/a.dtd'><svg {NS}/>",
    "cdata": f"<svg {NS}><text><![CDATA[x]]></text></svg>",
    "processing instruction": f"<?xml-stylesheet href='http://evil.example/a.css'?><svg {NS}/>",
    "animate href": f"<svg {NS}><a href='#x'><set attributeName='href' to='javascript:alert(1)'/></a></svg>",
    "animate": f"<svg {NS}><rect><animate attributeName='x' to='5'/></rect></svg>",
    "animateTransform": f"<svg {NS}><rect><animateTransform attributeName='transform'/></rect></svg>",
    "a javascript": f"<svg {NS}><a href='javascript:alert(1)'><rect width='5' height='5'/></a></svg>",
    "feImage ext": f"<svg {NS}><filter id='f'><feImage href='file:///etc/passwd'/></filter></svg>",
    "xi:include": "<svg xmlns='http://www.w3.org/2000/svg' xmlns:xi='http://www.w3.org/2001/XInclude'><xi:include href='file:///etc/passwd' parse='text'/></svg>",
    "mixed-case Href": f"<svg {NS}><image HREF='http://evil.example/a.png'/></svg>",
    "src attr": f"<svg {NS}><image src='http://evil.example/a.png'/></svg>",
    "href on a path": f"<svg {NS}><path href='#x' d='M0 0'/></svg>",
    "unknown attribute": f"<svg {NS}><rect onclick2='x' data-x='1'/></svg>",
    "fill url outside": f"<svg {NS}><rect fill='url(http://evil.example/a#g)'/></svg>",
    "too many elements": f"<svg {NS}>" + "<rect/>" * 5001 + "</svg>",
    "foreign root": "<html xmlns='http://www.w3.org/1999/xhtml'/>",
}
SVG_BYPASS_BYTES = [
    ("utf16 entity", ("\ufeff<!DOCTYPE svg [<!ENTITY e 'x'>]><svg xmlns='http://www.w3.org/2000/svg'>"
                      "<text>&e;</text></svg>").encode("utf-16-le")),
    ("utf16 plain", "<svg xmlns='http://www.w3.org/2000/svg'/>".encode("utf-16")),
    ("not utf-8", b"<svg xmlns='http://www.w3.org/2000/svg'><text>\xff\xfe</text></svg>"),
    ("declared other encoding", b"<?xml version='1.0' encoding='latin-1'?><svg xmlns='http://www.w3.org/2000/svg'/>"),
]


def test_a_plain_drawing_with_gradient_and_text_is_kept():
    svg = ("<?xml version='1.0' encoding='UTF-8'?><!-- the club's mark -->"
           "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 40' width='120' height='40'>"
           "<title>みなと町</title><defs><linearGradient id='g' x1='0' x2='1'>"
           "<stop offset='0' stop-color='#1c5cab'/><stop offset='1' stop-color='#0d6670'/>"
           "</linearGradient></defs><g transform='translate(2 2)'>"
           "<rect width='36' height='36' rx='6' fill='url(#g)'/>"
           "<circle cx='18' cy='18' r='8' fill='#fff' stroke='#123' stroke-width='2'/>"
           "<path d='M0 0 L5 5' style='stroke:#000;stroke-width:1'/>"
           "<text x='44' y='26' font-size='16' font-family='sans-serif' fill='#23272e'>"
           "<tspan>みなと</tspan></text></g></svg>").encode("utf-8")
    data, ext = org.validate_logo(svg, "logo.svg")
    assert ext == ".svg" and data == svg


@pytest.mark.parametrize("data,name", [
    (b"not an image", "l.png"),
    (png(), "l.gif"),
    (png(5000, 10), "l.png"),
    (b"<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>", "l.svg"),
    (b"<svg xmlns='http://www.w3.org/2000/svg' onload='x()'/>", "l.svg"),
    (b"<svg xmlns='http://www.w3.org/2000/svg'><image href='http://example.com/a.png'/></svg>",
     "l.svg"),
    (b"<!DOCTYPE svg [<!ENTITY a 'b'>]><svg xmlns='http://www.w3.org/2000/svg'/>", "l.svg"),
    (b"<html/>", "l.svg"),
    *[(svg.encode("utf-8"), "l.svg") for svg in SVG_BYPASSES.values()],
    *[(raw, "l.svg") for _, raw in SVG_BYPASS_BYTES],
], ids=lambda v: v if isinstance(v, str) and len(v) < 20 else None)
def test_bad_logos_are_refused(settings, data, name):
    with pytest.raises(OrgError):
        org.save(settings, name="a", accent="blue", logo=data, logo_name=name)
    assert org.load(settings).logo is None


def test_logo_size_limit(settings):
    big = b"<svg xmlns='http://www.w3.org/2000/svg'>" + b" " * (org.MAX_LOGO_BYTES + 1) + b"</svg>"
    with pytest.raises(OrgError):
        org.save(settings, name="a", accent="blue", logo=big, logo_name="l.svg")


def test_name_length_limit(settings):
    with pytest.raises(OrgError):
        org.save(settings, name="長" * (org.MAX_NAME + 1), accent="blue")


def test_a_stored_svg_that_fails_the_check_is_not_used(settings):
    o = org.save(settings, name="a", accent="blue", logo=SVG, logo_name="l.svg")
    o.logo.write_text(USE_BOMB, encoding="utf-8")       # stored before the allowlist
    assert org.load(settings).logo is None
