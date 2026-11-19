"""PDF/Word preprocessing: Word -> PDF via LibreOffice, per-page text and page images."""
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c


class DocumentError(Exception):
    """書類を読めない・変換できない場合のエラー（メッセージは日本語）。"""


@dataclass(frozen=True)
class Page:
    number: int
    text: str
    image_path: Path | None

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "text": self.text,
            "image_path": str(self.image_path) if self.image_path is not None else None,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Page":
        img = d.get("image_path")
        return cls(
            number=int(d["number"]),
            text=d["text"],
            image_path=Path(img) if img else None,
        )


_SOFFICE_TIMEOUT_SEC = 120


def to_pdf(src: Path, out_dir: Path) -> Path:
    """Return a PDF for src. .pdf is returned as is; .docx/.doc are converted into out_dir."""
    src = Path(src)
    out_dir = Path(out_dir)
    ext = src.suffix.lower()
    if ext == ".pdf":
        return src
    if ext not in (".docx", ".doc"):
        raise DocumentError(
            f"対応していないファイル形式です（{src.suffix or '拡張子なし'}）。PDFまたはWord（.docx/.doc）を使ってください"
        )
    if not src.is_file():
        raise DocumentError(f"書類ファイルが見つかりません: {src.name}")
    soffice = shutil.which("soffice")
    if soffice is None:
        raise DocumentError("Wordを変換するLibreOffice（soffice）が見つかりません")
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="soffice_profile_")
    try:
        try:
            proc = subprocess.run(
                [
                    soffice,
                    f"-env:UserInstallation=file://{profile}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(out_dir),
                    str(src),
                ],
                capture_output=True,
                text=True,
                timeout=_SOFFICE_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            raise DocumentError(
                f"Wordの変換が{_SOFFICE_TIMEOUT_SEC}秒以内に終わりませんでした"
            ) from None
    finally:
        shutil.rmtree(profile, ignore_errors=True)
    pdf = out_dir / f"{src.stem}.pdf"
    if proc.returncode != 0 or not pdf.is_file():
        raise DocumentError("Wordの変換に失敗しました（ファイルが壊れている可能性があります）")
    return pdf


def _page_text(page) -> str:
    """Page text with \n line breaks (pdfium reports CR LF)."""
    textpage = page.get_textpage()
    try:
        return textpage.get_text_bounded().replace("\r\n", "\n").replace("\r", "\n")
    finally:
        textpage.close()


def _has_images(page) -> bool:
    return any(True for _ in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE]))


def _render_page(page, dest: Path, long_edge: int) -> None:
    width, height = page.get_size()
    scale = long_edge / max(width, height)
    bitmap = page.render(scale=scale, rotation=0)
    image = bitmap.to_pil().convert("RGB")
    if max(image.size) != long_edge:  # pdfium rounds up; keep the long edge exact
        k = long_edge / max(image.size)
        image = image.resize((max(1, round(image.width * k)), max(1, round(image.height * k))))
    image.save(str(dest))


# PDFium is not thread-safe: serialise every pdfium call (the worker runs a ThreadPoolExecutor).
_PDFIUM_LOCK = threading.Lock()


def extract_pages(pdf: Path, out_dir: Path, settings) -> list[Page]:
    """Per-page text; sparse or image-bearing pages are also rendered (pypdfium2)."""
    with _PDFIUM_LOCK:
        return _extract_pages(pdf, out_dir, settings)


def _extract_pages(pdf: Path, out_dir: Path, settings) -> list[Page]:
    pdf = Path(pdf)
    if not pdf.is_file():
        raise DocumentError(f"PDFファイルが見つかりません: {pdf.name}")
    try:
        doc = pdfium.PdfDocument(str(pdf))
    except pdfium.PdfiumError as e:
        if getattr(e, "err_code", None) == pdfium_c.FPDF_ERR_PASSWORD:
            raise DocumentError("パスワードで保護されたPDFです") from None
        raise DocumentError("PDFを読み込めません（破損している可能性があります）") from None
    except Exception:
        raise DocumentError("PDFを読み込めません（破損している可能性があります）") from None
    try:
        count = len(doc)
        if count == 0:
            raise DocumentError("PDFにページがありません")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pages: list[Page] = []
        rendered = 0
        try:
            for i in range(count):
                page = doc[i]
                try:
                    text = _page_text(page)
                    image_path = None
                    needs_image = len(text.strip()) < settings.low_text_chars or _has_images(page)
                    if needs_image and rendered < settings.doc_max_page_images:
                        image_path = out_dir / f"page_{i + 1:04d}.png"
                        _render_page(page, image_path, settings.doc_page_image_long_edge)
                        rendered += 1
                    pages.append(Page(number=i + 1, text=text, image_path=image_path))
                finally:
                    page.close()
        except DocumentError:
            raise
        except Exception:
            raise DocumentError("PDFを読み込めません（破損している可能性があります）") from None
        return pages
    finally:
        doc.close()


_BOILERPLATE = (
    "レポート", "課題", "提出", "宿題", "最終", "コピー", "scan", "report", "final",
    "assignment", "submission", "hw", "copy",
)
_SEPARATORS = "_-－ー・　 ()（）[]【】.,、"


def guess_student_name(filename: str, roster: list[str] | None) -> str:
    stem = Path(filename).stem
    if roster:
        hits = [n for n in roster if n and n in stem]
        if hits:
            return max(hits, key=len)
    rest = stem
    for w in _BOILERPLATE:
        rest = re.sub(re.escape(w), " ", rest, flags=re.IGNORECASE)
    rest = re.sub(r"[A-Za-z]*\d[A-Za-z0-9]*", " ", rest)  # student IDs / numbering
    rest = re.sub(f"[{re.escape(_SEPARATORS)}]+", " ", rest).strip()
    return rest or stem
