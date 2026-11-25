"""Build the NiceMill brand files: the vendored web fonts and the logo files.

    uv run python scripts/brand_assets.py <dir with the source fonts>

The source fonts are the OFL TTFs from github.com/google/fonts (ofl/delagothicone,
ofl/mplus1, ofl/barlowcondensed):

    DelaGothicOne-Regular.ttf  MPLUS1[wght].ttf
    BarlowCondensed-SemiBold.ttf  BarlowCondensed-Bold.ttf

Writes, under src/pfb_app/ui/static:

- fonts/MPLUS1-{Regular,Medium,Bold,ExtraBold}.woff2: static instances of the variable
  font (400/500/700/800), hinting removed, subset to Latin, kana, CJK punctuation, full-width
  forms and the JIS X 0208 kanji (levels 1 and 2, with the NEC/IBM extensions of cp932) plus
  the few 常用漢字 outside it; other characters fall back to the system's Japanese font.
- fonts/DelaGothicOne-Regular.woff2: the same character set (headings).
- fonts/BarlowCondensed-{SemiBold,Bold}.woff2: Latin only (numbers).
- brand/nicemiru_mark.svg, nicemiru_mark_mono.svg: the mark (a ball-orange tile: a court's
  centre line and centre circle make an eye).
- brand/nicemiru_wordmark.svg, _ondark.svg, _mono.svg: the mark and 「NiceMill」, the
  letters as outlines (paths) from Dela Gothic One, so an <img> needs no web font.
- brand/nicemiru_mark_64.png, nicemiru_mark_192.png: the mark as a picture (tab icon).

The SVG texts come from functions here, so a test can keep the vendored files in step.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "src" / "pfb_app" / "ui" / "static"

BALL = "#C8410C"      # ball orange (theme TOKENS["ball"]): white on it 5.0:1
COURT = "#0F2D5C"     # court navy (TOKENS["board"])
AMBER = "#FFB000"     # shot-clock amber (TOKENS["amber"])

MPLUS_WEIGHTS = ((400, "Regular"), (500, "Medium"), (700, "Bold"), (800, "ExtraBold"))
WORD = "NiceMill"


# --- the mark -------------------------------------------------------------------------------

def mark_body(mono: bool = False) -> str:
    """The mark's shapes in a 32 x 32 box."""
    if mono:
        return ('<rect x="1.2" y="1.2" width="29.6" height="29.6" rx="6.5" fill="none" '
                'stroke="#000" stroke-width="2.4"/>'
                '<line x1="4" y1="16" x2="28" y2="16" stroke="#000" stroke-width="2.2"/>'
                '<circle cx="16" cy="16" r="8.4" fill="#fff" stroke="#000" stroke-width="2.4"/>'
                '<circle cx="16" cy="16" r="3.6" fill="#000"/>')
    return (f'<rect width="32" height="32" rx="7" fill="{BALL}"/>'
            '<line x1="3" y1="16" x2="29" y2="16" stroke="#fff" stroke-width="2.2"/>'
            f'<circle cx="16" cy="16" r="8.4" fill="{BALL}" stroke="#fff" stroke-width="2.4"/>'
            f'<circle cx="16" cy="16" r="3.8" fill="{COURT}"/>'
            f'<circle cx="17.4" cy="14.6" r="1.1" fill="{AMBER}"/>')


def mark_svg(mono: bool = False) -> str:
    """The colour mark at 32 px (st.logo's icon); the black-and-white one at 11 px (about
    8 pt: its size in the PDFs' footers, where an image in a page margin keeps its own size)."""
    px = 11 if mono else 32
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" '
            f'width="{px}" height="{px}" role="img" aria-label="NiceMill"><title>NiceMill</title>'
            f'{mark_body(mono)}</svg>\n')


# --- the wordmark: 「NiceMill」 in Dela Gothic One, as outlines -------------------------------

def word_path(font_path: Path, *, size: float = 20.5, x0: float = 49.0, baseline: float = 23.5,
              spacing: float = 0.5, skew_deg: float = -9.0) -> tuple[str, float]:
    """The letters as one SVG path (user units), slanted like the mockup's skewX(-9) about
    the top edge; and the x where the last letter ends."""
    from fontTools.pens.svgPathPen import SVGPathPen
    from fontTools.pens.transformPen import TransformPen
    from fontTools.ttLib import TTFont

    font = TTFont(str(font_path))
    cmap, glyphs = font.getBestCmap(), font.getGlyphSet()
    scale = size / font["head"].unitsPerEm
    k = math.tan(math.radians(skew_deg))
    pen = SVGPathPen(glyphs, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip("."))
    x = x0
    for ch in WORD:
        name = cmap[ord(ch)]
        # font units (y up) -> user units (y down), then the slant x' = x + k * y' (skewX
        # about y = 0, the top of the 32 px box, as translate(44 0) skewX(-9) in the mockup)
        sx, sy = scale, -scale
        tx, ty = x, baseline
        # x' = (sx*u + tx) + k * (sy*v + ty) ; y' = sy*v + ty
        glyphs[name].draw(TransformPen(pen, (sx, 0, k * sy, sy, tx + k * ty, ty)))
        x += glyphs[name].width * scale + spacing
    return pen.getCommands(), x - spacing


def wordmark_svg(path_d: str, *, ink: str, mono: bool = False, width: int = 156) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="0 0 {width} 32" width="{width}" height="32" role="img" '
            f'aria-label="NiceMill"><title>NiceMill</title>{mark_body(mono)}'
            f'<path d="{path_d}" fill="{ink}"/></svg>\n')


def mark_png(px: int):
    """The mark drawn with Pillow at 8x and scaled down (no SVG renderer needed)."""
    from PIL import Image, ImageDraw

    s = 8 * px / 32

    def u(v: float) -> float:
        return v * s

    img = Image.new("RGBA", (8 * px, 8 * px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, u(32) - 1, u(32) - 1), radius=u(7), fill=BALL)
    d.rectangle((u(3), u(16 - 1.1), u(29), u(16 + 1.1)), fill="#FFFFFF")
    d.ellipse((u(16 - 9.6), u(16 - 9.6), u(16 + 9.6), u(16 + 9.6)), fill="#FFFFFF")
    d.ellipse((u(16 - 7.2), u(16 - 7.2), u(16 + 7.2), u(16 + 7.2)), fill=BALL)
    d.ellipse((u(16 - 3.8), u(16 - 3.8), u(16 + 3.8), u(16 + 3.8)), fill=COURT)
    d.ellipse((u(17.4 - 1.1), u(14.6 - 1.1), u(17.4 + 1.1), u(14.6 + 1.1)), fill=AMBER)
    return img.resize((px, px), Image.LANCZOS)


# --- fonts ----------------------------------------------------------------------------------

def jp_charset() -> set[int]:
    """Latin, punctuation and symbols the screens use, kana, full-width forms and the
    JIS X 0208 kanji (every two-byte cp932 character), plus 常用漢字 outside it."""
    cps: set[int] = set()
    ranges = [(0x20, 0x7E), (0xA0, 0xFF), (0x2010, 0x2027), (0x2030, 0x203B), (0x2190, 0x2199),
              (0x2460, 0x2473), (0x25A0, 0x25CF), (0x2605, 0x2606), (0x2713, 0x2717),
              (0x3000, 0x303F), (0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF),
              (0xFF01, 0xFF9F), (0xFFE0, 0xFFE5)]
    for lo, hi in ranges:
        cps.update(range(lo, hi + 1))
    for lead in list(range(0x81, 0xA0)) + list(range(0xE0, 0xFD)):
        for trail in list(range(0x40, 0x7F)) + list(range(0x80, 0xFD)):
            try:
                ch = bytes((lead, trail)).decode("cp932")
            except UnicodeDecodeError:
                continue
            cps.add(ord(ch))
    cps.update(ord(c) for c in "𠮟剝頰塡啞嚙噓惡")
    return cps


LATIN = set(range(0x20, 0x7F)) | set(range(0xA0, 0x100)) | {0x2013, 0x2014, 0x2019, 0x201C,
                                                           0x201D, 0x2026, 0x2212}


def subset(font, cps: set[int], out: Path) -> None:
    from fontTools import subset as fts

    opts = fts.Options()
    opts.flavor = "woff2"
    opts.hinting = False
    opts.layout_features = ["*"]       # palt, kern, vert ... kept
    opts.name_IDs = ["*"]
    opts.name_languages = ["*"]
    opts.notdef_outline = True
    opts.drop_tables += ["DSIG"]
    sub = fts.Subsetter(opts)
    sub.populate(unicodes=sorted(cps))
    sub.subset(font)
    font.flavor = "woff2"
    font.save(str(out))


def build_fonts(src: Path) -> None:
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    fonts = STATIC / "fonts"
    fonts.mkdir(parents=True, exist_ok=True)
    jp = jp_charset()
    for weight, name in MPLUS_WEIGHTS:
        var = TTFont(str(src / "MPLUS1[wght].ttf"))
        inst = instancer.instantiateVariableFont(var, {"wght": weight}, updateFontNames=False)
        subset(inst, jp, fonts / f"MPLUS1-{name}.woff2")
    subset(TTFont(str(src / "DelaGothicOne-Regular.ttf")), jp, fonts / "DelaGothicOne-Regular.woff2")
    for name in ("SemiBold", "Bold"):
        subset(TTFont(str(src / f"BarlowCondensed-{name}.ttf")), LATIN,
               fonts / f"BarlowCondensed-{name}.woff2")


def build_logos(src: Path) -> None:
    brand = STATIC / "brand"
    brand.mkdir(parents=True, exist_ok=True)
    (brand / "nicemiru_mark.svg").write_text(mark_svg(), encoding="utf-8")
    (brand / "nicemiru_mark_mono.svg").write_text(mark_svg(mono=True), encoding="utf-8")
    d, end = word_path(src / "DelaGothicOne-Regular.ttf")
    width = math.ceil(end + 4)
    for suffix, ink, mono in (("", COURT, False), ("_ondark", "#FFFFFF", False),
                              ("_mono", "#000000", True)):
        (brand / f"nicemiru_wordmark{suffix}.svg").write_text(
            wordmark_svg(d, ink=ink, mono=mono, width=width), encoding="utf-8")
    for px in (64, 192):
        mark_png(px).save(brand / f"nicemiru_mark_{px}.png", optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    source = Path(sys.argv[1])
    build_fonts(source)
    build_logos(source)
    for f in sorted((STATIC / "fonts").glob("*.woff2")) + sorted((STATIC / "brand").iterdir()):
        print(f"{f.stat().st_size:>9}  {f.relative_to(STATIC)}")
