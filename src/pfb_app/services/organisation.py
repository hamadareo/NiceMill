"""団体の設定: the organisation's name, logo and accent colour, used by the PDFs and the
share pages (one set per app; stored in `<data_dir>/organisation/`).

- name: at most MAX_NAME characters (spaces trimmed);
- accent: one of the ACCENTS presets (each reads as text on white, contrast >= 4.5:1);
  free colours are refused. Data colours of the charts never change with it;
- logo: PNG, JPEG or SVG of at most MAX_LOGO_BYTES. Pictures are opened and saved again as
  PNG (no more than MAX_LOGO_PX a side; nothing but the pixels is kept). SVG is kept as it is
  only when it is plain drawing: no DOCTYPE/ENTITY, no script, no event attributes, no
  foreignObject, no link or url() to anything but the drawing itself or an inline image.
"""
from __future__ import annotations

import io
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

ACCENTS: dict[str, tuple[str, str]] = {
    "blue": ("青", "#1c5cab"),
    "green": ("緑", "#1d6b45"),
    "teal": ("青緑", "#0d6670"),
    "violet": ("紫", "#5b46b5"),
    "wine": ("えんじ", "#9b2f3c"),
    "ink": ("墨", "#353b45"),
}
DEFAULT_ACCENT = "blue"
MAX_NAME = 40
MAX_LOGO_BYTES = 512 * 1024
MAX_LOGO_PX = 2000
FILE = "organisation.json"
LOGO_STEM = "logo"
_SVG_NS = "{http://www.w3.org/2000/svg}"


class OrgError(ValueError):
    """A setting that cannot be saved (plain Japanese message)."""


@dataclass(frozen=True)
class Organisation:
    name: str = ""
    accent: str = DEFAULT_ACCENT
    logo: Path | None = None

    @property
    def accent_hex(self) -> str:
        return ACCENTS.get(self.accent, ACCENTS[DEFAULT_ACCENT])[1]


def org_dir(settings) -> Path:
    return Path(settings.data_dir) / "organisation"


def load(settings) -> Organisation:
    d = org_dir(settings)
    try:
        rec = json.loads((d / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Organisation()
    if not isinstance(rec, dict):
        return Organisation()
    logo = rec.get("logo")
    path = d / logo if isinstance(logo, str) and "/" not in logo and logo else None
    if path is not None and path.suffix == ".svg" and path.is_file():
        try:
            _svg(path.read_bytes())
        except (OrgError, OSError):
            path = None        # stored before the allowlist, or changed on disk: not used
    accent = rec.get("accent") if rec.get("accent") in ACCENTS else DEFAULT_ACCENT
    return Organisation(name=str(rec.get("name") or "")[:MAX_NAME], accent=accent,
                        logo=path if path is not None and path.is_file() else None)


_SVG_ELEMENTS = frozenset({
    "svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text",
    "tspan", "defs", "linearGradient", "radialGradient", "stop", "title", "desc"})
_GEOMETRY = frozenset({
    "x", "y", "x1", "y1", "x2", "y2", "cx", "cy", "r", "rx", "ry", "fx", "fy", "fr", "width",
    "height", "d", "points", "dx", "dy", "rotate", "pathLength", "textLength", "lengthAdjust"})
_SVG_ATTRS = _GEOMETRY | frozenset({
    "transform", "gradientTransform", "gradientUnits", "spreadMethod", "fill", "fill-opacity",
    "fill-rule", "opacity", "offset", "id", "viewBox", "preserveAspectRatio", "version",
    "text-anchor", "style"})
_SVG_PREFIXES = ("stroke", "font-", "stop-")
_LOCAL_PAINT = re.compile(r"^\s*url\(#[A-Za-z_][A-Za-z0-9_.-]*\)\s*(none|#[0-9A-Fa-f]{3,8})?\s*$")
_BAD_STYLE = ("url(", "@import", "\\", "image", "://", "expression", "var(")
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
_XML_DECL = re.compile(r"^<\?xml(?:\s+[^?]*)?\?>")
MAX_SVG_ELEMENTS = 5000
MAX_SVG_EXTENT = 10000.0
MAX_SVG_NUMBER = 1e6


def _refuse(why: str) -> OrgError:
    return OrgError(f"このSVGは使えません（{why}）。")


def _svg_text(data: bytes) -> str:
    """The logo as UTF-8 XML text: no other encoding, no NUL, no DOCTYPE/ENTITY/CDATA (any
    「<!」 but a comment) and no processing instruction but a UTF-8 XML declaration."""
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise _refuse("UTF-8のXMLではありません") from None
    if "\x00" in text:
        raise _refuse("UTF-8のXMLではありません")
    if re.search(r"<!(?!--)", text):
        raise _refuse("文書型やCDATAの宣言があります")
    decl = _XML_DECL.match(text.lstrip())
    if decl:
        enc = re.search(r"encoding\s*=\s*['\"]([^'\"]*)['\"]", decl.group(0))
        if enc and enc.group(1).lower().replace("_", "-") not in ("utf-8", "utf8"):
            raise _refuse("UTF-8のXMLではありません")
    rest = text.lstrip()[decl.end():] if decl else text
    if "<?" in rest:
        raise _refuse("処理の指定があります")
    return text


def _numbers(value: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(value)]


def _svg(data: bytes) -> bytes:
    """`data` when it is a plain drawing (allowlist; see the module doc), else OrgError."""
    text = _svg_text(data)
    try:
        parser = ET.XMLParser(encoding="utf-8")
        root = ET.fromstring(text.encode("utf-8"), parser=parser)
    except (ET.ParseError, ValueError):
        raise OrgError("SVGとして読めません。") from None
    if root.tag != f"{_SVG_NS}svg":
        raise OrgError("SVGとして読めません。")
    count = 0
    for el in root.iter():
        count += 1
        if count > MAX_SVG_ELEMENTS:
            raise _refuse("図形が多すぎます")
        if not isinstance(el.tag, str) or not el.tag.startswith(_SVG_NS) \
                or el.tag[len(_SVG_NS):] not in _SVG_ELEMENTS:
            raise _refuse("使えない要素があります")
        for name, value in el.attrib.items():
            if name.startswith("{") or not (name in _SVG_ATTRS or name.startswith(_SVG_PREFIXES)):
                raise _refuse("使えない属性があります")
            low = value.lower()
            if "href" in name.lower() or "javascript:" in low or "\\" in value:
                raise _refuse("外のファイルを読みます")
            if name == "style":
                if any(bad in low for bad in _BAD_STYLE):
                    raise _refuse("外のファイルを読みます")
            elif "url(" in low and not (name in ("fill", "stroke") and _LOCAL_PAINT.match(value)):
                raise _refuse("外のファイルを読みます")
            if name in _GEOMETRY and any(abs(n) > MAX_SVG_NUMBER for n in _numbers(value)):
                raise _refuse("大きすぎる数があります")
    view = root.get("viewBox")
    if view is not None:
        nums = _numbers(view)
        if len(nums) != 4 or any(abs(n) > MAX_SVG_EXTENT for n in nums):
            raise _refuse("表示の範囲が大きすぎます")
    for side in ("width", "height"):
        nums = _numbers(root.get(side) or "")
        if nums and abs(nums[0]) > MAX_SVG_EXTENT:
            raise _refuse("表示の範囲が大きすぎます")
    return data


def _picture(data: bytes) -> bytes:
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ("PNG", "JPEG"):
                raise OrgError("ロゴはPNG・JPEG・SVGのどれかにしてください。")
            if max(img.size) > MAX_LOGO_PX:
                raise OrgError(f"ロゴの大きさは、縦横{MAX_LOGO_PX}ピクセルまでにしてください。")
            img.load()
            clean = img.convert("RGBA")
    except OrgError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError):
        raise OrgError("画像として読めません。") from None
    buf = io.BytesIO()
    clean.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def validate_logo(data: bytes, filename: str) -> tuple[bytes, str]:
    """(the bytes to store, ".png" or ".svg"); OrgError when the file is not a usable logo."""
    if len(data) > MAX_LOGO_BYTES:
        raise OrgError(f"ロゴのファイルは{MAX_LOGO_BYTES // 1024}KBまでにしてください。")
    ext = Path(filename or "").suffix.lower()
    if ext == ".svg":
        return _svg(data), ".svg"
    if ext in (".png", ".jpg", ".jpeg"):
        return _picture(data), ".png"
    raise OrgError("ロゴはPNG・JPEG・SVGのどれかにしてください。")


def _write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save(settings, *, name: str, accent: str, logo: bytes | None = None,
         logo_name: str | None = None, remove_logo: bool = False) -> Organisation:
    """Validate and store the settings; a logo left out keeps the stored one."""
    name = " ".join((name or "").split())
    if len(name) > MAX_NAME:
        raise OrgError(f"団体の名前は{MAX_NAME}文字までにしてください。")
    if accent not in ACCENTS:
        raise OrgError("色は一覧から選んでください。")
    stored = None if logo is None else validate_logo(logo, logo_name or "")
    d = org_dir(settings)
    d.mkdir(parents=True, exist_ok=True)
    current = load(settings).logo
    logo_file = current.name if current is not None else None
    if stored is not None or remove_logo:
        for old in d.glob(f"{LOGO_STEM}.*"):
            old.unlink(missing_ok=True)
        logo_file = None
    if stored is not None:
        data, ext = stored
        logo_file = f"{LOGO_STEM}{ext}"
        _write(d / logo_file, data)
    _write(d / FILE, json.dumps({"name": name, "accent": accent, "logo": logo_file},
                                ensure_ascii=False).encode("utf-8"))
    return load(settings)
