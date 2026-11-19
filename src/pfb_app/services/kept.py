"""The saved handouts of a work (保存版のPDF): PDFs are made on demand from the videos, so
before the worker removes a work's videos (services.retention) it saves what was handed out.

`data/assignments/<aid>/kept/`:
- `<subject id>.pdf` and `<subject id>.html`: each approved person's handout and phone page;
- `evaluator.pdf`: the evaluator PDF;
- `kept.json`: {"saved_at", "title", "people_word", "people": [{"subject_id", "name"}],
  "evaluator": "evaluator.pdf" | null}, people in the handout order.

Only names this module makes are ever read (`<digits>.pdf`, `evaluator.pdf`), so a changed
manifest can never point outside the folder. The folder is written beside it first and then
swapped in, so a crash never leaves half a set. The combined handout PDF is merged from the
per-person PDFs (pypdfium2) and the ZIP assembled from them when they are asked for.
"""
from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

FOLDER = "kept"
MANIFEST = "kept.json"
EVALUATOR = "evaluator.pdf"


@dataclass(frozen=True)
class Kept:
    saved_at: str
    title: str
    people_word: str
    people: list[dict] = field(default_factory=list)   # {"subject_id": int, "name": str}
    evaluator: str | None = None


def folder(settings, aid: int) -> Path:
    return files.assignment_dir(settings, aid, create=False) / FOLDER


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _write(path: Path, data: bytes) -> None:
    path.write_bytes(data)


def save(repo: Repo, aid: int) -> Kept:
    """Make and store the work's handouts now (ExportError: nothing approved)."""
    from pfb_app.export import pdf   # noqa: PLC0415 - WeasyPrint only when it is needed
    settings = repo.settings
    parts = pdf.handout_parts(repo, aid)
    evaluator = pdf.render_teacher_pdf(repo, aid)
    work = repo.get_assignment(aid)
    record = Kept(saved_at=_now(), title=work["title"], people_word=pdf.people_word(repo, aid),
                  people=[{"subject_id": sid, "name": name} for sid, name, _, _ in parts],
                  evaluator=EVALUATOR)
    files.assignment_dir(settings, aid)   # made if missing (a work without files yet)
    work_dir = work_folder(settings, aid)
    if work_dir is None:
        raise OSError("作業フォルダがリンクかデータフォルダの外にあるため、PDFを保存しません")
    target = work_dir / FOLDER
    new = target.with_name(f"{FOLDER}.new-{os.getpid()}")
    old = target.with_name(f"{FOLDER}.old-{os.getpid()}")
    shutil.rmtree(new, ignore_errors=True)
    new.mkdir()
    try:
        for sid, _, data, html in parts:
            _write(new / f"{sid}.pdf", data)
            _write(new / f"{sid}.html", html.encode("utf-8"))
        _write(new / EVALUATOR, evaluator)
        _write(new / MANIFEST, _manifest(record))
        if target.is_symlink():
            target.unlink()
        elif target.exists():
            os.replace(target, old)
        os.replace(new, target)
    finally:
        shutil.rmtree(new, ignore_errors=True)
        shutil.rmtree(old, ignore_errors=True)
    log.info("PDFを保存しました assignment=%s people=%d", aid, len(parts))
    return record


def _manifest(record: Kept) -> bytes:
    return json.dumps({"saved_at": record.saved_at, "title": record.title,
                       "people_word": record.people_word, "people": record.people,
                       "evaluator": record.evaluator}, ensure_ascii=False).encode("utf-8")


def work_folder(settings, aid: int) -> Path | None:
    """The work's folder when it is a real folder (not a link) directly inside
    data_dir/assignments, itself inside data_dir; else None."""
    root = files.assignment_dir(settings, aid, create=False)
    try:
        parent = (Path(settings.data_dir) / "assignments").resolve()
        if root.is_symlink() or not root.is_dir() or root.resolve().parent != parent \
                or not parent.is_relative_to(Path(settings.data_dir).resolve()):
            return None
        return root.resolve()
    except OSError:
        return None


def _real_folder(settings, aid: int) -> Path | None:
    """The kept folder when it is a real folder inside a real work folder (never a link)."""
    work = work_folder(settings, aid)
    if work is None:
        return None
    d = work / FOLDER
    if d.is_symlink() or not d.is_dir():
        return None
    return d


def load(settings, aid: int) -> Kept | None:
    """The saved set (people whose files are missing are left out), or None."""
    d = _real_folder(settings, aid)
    if d is None:
        return None
    try:
        rec = json.loads((d / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict):
        return None
    people = []
    for p in rec.get("people") or []:
        sid = p.get("subject_id") if isinstance(p, dict) else None
        if isinstance(sid, int) and not isinstance(sid, bool) and (d / f"{sid}.pdf").is_file():
            people.append({"subject_id": sid, "name": str(p.get("name") or "")})
    evaluator = EVALUATOR if rec.get("evaluator") == EVALUATOR and \
        (d / EVALUATOR).is_file() else None
    return Kept(saved_at=str(rec.get("saved_at") or ""), title=str(rec.get("title") or ""),
                people_word=str(rec.get("people_word") or "参加者"), people=people,
                evaluator=evaluator)


def current(repo: Repo, aid: int) -> Kept | None:
    """The saved set while it still holds exactly the work's approved people, under the names
    they have now, and no approval is newer than it (else None: the next download is made
    anew from the texts, without pictures once the videos are gone)."""
    record = load(repo.settings, aid)
    if record is None or not record.people:
        return None
    rows = repo.conn.execute(
        "SELECT s.id, s.approved_at, s.display_name FROM subjects s JOIN submissions b"
        " ON s.submission_id = b.id WHERE b.assignment_id=? AND s.approval_status='approved'",
        (aid,)).fetchall()
    if {r[0]: r[2] for r in rows} != {p["subject_id"]: p["name"] for p in record.people}:
        return None   # someone approved or taken back, or renamed (review MED 7)
    if any(r[1] and r[1] > record.saved_at for r in rows):
        return None
    return record


def _pdf(settings, aid: int, sid: int) -> bytes:
    return (folder(settings, aid) / f"{sid}.pdf").read_bytes()


def student_pdf(settings, aid: int, record: Kept) -> bytes:
    """The handouts one after the other (pypdfium2 copies the pages as they are)."""
    import pypdfium2 as pdfium   # noqa: PLC0415
    out = pdfium.PdfDocument.new()
    try:
        for p in record.people:
            src = pdfium.PdfDocument(_pdf(settings, aid, p["subject_id"]))
            try:
                out.import_pages(src)
            finally:
                src.close()
        buf = io.BytesIO()
        out.save(buf)
        return buf.getvalue()
    finally:
        out.close()


def student_zip(settings, aid: int, record: Kept) -> bytes:
    """`<番号>_<名前>.pdf` and `.html` per person, as the made-on-demand ZIP names them."""
    from pfb_app.export import labels   # noqa: PLC0415
    buf = io.BytesIO()
    total = len(record.people)
    d = folder(settings, aid)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, p in enumerate(record.people, 1):
            name = labels.handout_file_name(k, p["name"], total, fallback=record.people_word)
            z.writestr(name, _pdf(settings, aid, p["subject_id"]))
            page = d / f"{p['subject_id']}.html"
            if page.is_file():
                z.writestr(name[:-4] + ".html", page.read_bytes())
    return buf.getvalue()


def evaluator_pdf(settings, aid: int, record: Kept) -> bytes | None:
    if record.evaluator != EVALUATOR:
        return None
    try:
        return (folder(settings, aid) / EVALUATOR).read_bytes()
    except OSError:
        return None


def drop_person(settings, aid: int, subject_id: int) -> int:
    """Remove one person's saved files and the evaluator PDF (it lists everyone); returns how
    many files went."""
    d = _real_folder(settings, aid)
    if d is None:
        return 0
    record = load(settings, aid)
    gone = 0
    for name in (f"{int(subject_id)}.pdf", f"{int(subject_id)}.html", EVALUATOR):
        path = d / name
        if path.is_file() and not path.is_symlink():
            path.unlink()
            gone += 1
    if record is not None:
        left = [p for p in record.people if p["subject_id"] != subject_id]
        (d / MANIFEST).write_bytes(_manifest(Kept(record.saved_at, record.title,
                                                  record.people_word, left, None)))
    return gone
