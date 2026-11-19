"""Registration services: assignments, document uploads and video groups."""
import math
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.ops import disk
from pfb_app.pipeline.runner import enqueue_once
from pfb_app.preprocess.documents import DocumentError, extract_pages, guess_student_name, to_pdf
from pfb_app.preprocess.frames import VideoError, probe_video, validate_interval
from pfb_app.preprocess.transcript import parse_pasted_transcript
from pfb_app.services import chunked_upload, proxy
from pfb_app.storage import files
from pfb_app.storage.db import Repo

__all__ = ["ENGINES", "NO_PEOPLE_CAP", "IntakeReport", "ParticipantInput", "create_assignment_from_form",
           "list_inbox_videos", "register_documents", "register_video_group", "rename_subject",
           "save_uploaded_video"]

DOCUMENT_EXTENSIONS = (".pdf", ".docx", ".doc")
# a work whose videos were removed (services.retention) takes no new files: they could never
# be analysed there
CLEARED = ("この作業の動画は消してあります。新しい動画や書類は、新しい作業として始めてください。")
ENGINES = ("paper", "pfb3x3")  # submissions.meta["engine"]: the paper engine, or PFB3x3 tracking
NO_PEOPLE_CAP = 0              # register_video_group(max_people=...): track everyone found
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm")


@dataclass
class IntakeReport:
    created: list[tuple[int, str]] = field(default_factory=list)  # (submission_id, name)
    errors: list[tuple[str, str]] = field(default_factory=list)  # (filename, message)
    duplicate_names: list[str] = field(default_factory=list)


@dataclass
class ParticipantInput:
    participant_id: str
    display_name: str
    appearance: str
    initial_positions: dict[str, str]  # generated camera name (cam1..) -> position


def _basename(name: str) -> str:
    """The last path component of an uploaded name, whatever separator it used."""
    return name.replace("\\", "/").split("/")[-1].strip()


def _pdf_text(src: Path, work: Path, settings: Settings) -> str:
    pdf = to_pdf(src, work)
    return "\n".join(p.text for p in extract_pages(pdf, work / "pages", settings))


def create_assignment_from_form(repo: Repo, title: str, kind: str, task_text: str,
                                rubric_text: str, rubric_file: Path | None, model_key: str,
                                require_identity_approval: bool, settings: Settings | None = None
                                ) -> int:
    """Create an assignment; a rubric file is converted to text and appended to rubric_text."""
    rubric = rubric_text.strip()
    if rubric_file is not None:
        settings = settings or repo.settings or Settings()
        with tempfile.TemporaryDirectory() as tmp:
            extracted = _pdf_text(Path(rubric_file), Path(tmp), settings).strip()
        rubric = "\n\n".join(t for t in (rubric, extracted) if t)
    return repo.create_assignment(title.strip(), Kind(kind), task_text, rubric, model_key,
                                  require_identity_approval)


def _validate_document(filename: str, data: bytes, settings: Settings) -> None:
    ext = Path(filename).suffix.lower()
    if ext not in DOCUMENT_EXTENSIONS:
        raise DocumentError(
            f"対応していないファイル形式です（{ext or '拡張子なし'}）。PDFまたはWord（.docx/.doc）を使ってください")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"upload{ext}"
        src.write_bytes(data)
        pdf = to_pdf(src, Path(tmp) / "pdf")
        extract_pages(pdf, Path(tmp) / "pages", settings)


def register_documents(repo: Repo, settings: Settings, assignment_id: int,
                       files_: list[tuple[str, bytes]], roster: list[str] | None
                       ) -> IntakeReport:
    """One submission and one subject per readable file; unreadable files go to `errors`."""
    report = IntakeReport()
    if repo.get_assignment(assignment_id).get("media_removed_at"):
        report.errors.extend((_basename(name) or "無題", CLEARED) for name, _ in files_)
        return report
    try:   # before anything is written: never fill the disk
        disk.ensure_room(settings, sum(len(data) for _, data in files_))
    except disk.DiskLow as e:
        report.errors.extend((_basename(name) or "無題", str(e)) for name, _ in files_)
        return report
    seen = [s["display_name"] for sub in repo.list_submissions(assignment_id)
            for s in repo.list_subjects(sub["id"])]
    new_names: list[str] = []
    for raw_name, data in files_:
        filename = _basename(raw_name) or "無題"
        try:
            _validate_document(filename, data, settings)
        except DocumentError as e:
            report.errors.append((filename, str(e)))
            continue
        name = guess_student_name(filename, roster)
        sid = repo.create_submission(assignment_id, name, {})
        dest = files.submission_dir(settings, assignment_id, sid) / f"source{Path(filename).suffix.lower()}"
        dest.write_bytes(data)
        repo.update_submission_meta(sid, {"source_path": str(dest), "original_filename": filename})
        repo.add_subject(sid, None, name, None, {})
        enqueue_once(repo, sid, Stage.prepare)
        report.created.append((sid, name))
        new_names.append(name)
        seen.append(name)
    report.duplicate_names = list(dict.fromkeys(n for n in new_names if seen.count(n) > 1))
    return report


def list_inbox_videos(settings: Settings) -> list[Path]:
    inbox = settings.inbox_dir
    if not inbox.is_dir():
        return []
    return sorted(p for p in inbox.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _uploads_dir(settings: Settings) -> Path:
    return settings.data_dir / "uploads"


def save_uploaded_video(settings: Settings, filename: str, fileobj: BinaryIO) -> Path:
    """Copy a browser upload into data/uploads under a unique, safe name (streamed).
    disk.DiskLow (a VideoError) when it would leave less than disk_refuse_gb free."""
    disk.ensure_room(settings, disk.upload_size(fileobj))
    base = _basename(filename)
    ext = Path(base).suffix.lower()
    stem = re.sub(r"[^\w.-]+", "_", Path(base).stem, flags=re.UNICODE).strip("._") or "video"
    out_dir = _uploads_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{uuid.uuid4().hex[:8]}_{stem}{ext}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(fileobj, f)
    return dest


def register_video_group(repo: Repo, settings: Settings, assignment_id: int, label: str,
                         cameras: dict[str, Path], start_sec: float | None,
                         end_sec: float | None, transcript_text: str | None,
                         auto_transcribe: bool, participants: list[ParticipantInput],
                         engine: str = "paper", max_people: int | None = None,
                         hold_for_check: bool = False, clock_offset_min: float = 0.0) -> int:
    """Register one video group (several cameras of the same lesson).

    The keys of `cameras` are ignored: cameras are named cam1..camN in the order given, and
    `participants[].initial_positions` must use those names. Videos are referenced by path.
    `engine` ("paper" or "pfb3x3") is stored in meta["engine"]; it decides which pipeline the
    group's prepare jobs lead into. `max_people` (tracking engine only; meta["max_people"]):
    the tracker keeps at most this many people, the ones seen longest during the task, so
    onlookers in the background get no label. None (the default): the number of
    participants (no cap without participants); NO_PEOPLE_CAP (0): keep everyone it found.

    `hold_for_check`: the group waits for the content check (services/content_check):
    meta["check"] = "pending" and one `detect` job is enqueued instead of the cameras'
    `prepare` jobs, which the coach's confirm enqueues. `clock_offset_min` (meta, minutes, ≥ 0)
    is the game clock at the video's 0:00; it is stored with a hold or when it is not 0.
    """
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}; choices: {ENGINES}")
    if repo.get_assignment(assignment_id).get("media_removed_at"):
        raise VideoError(CLEARED)
    if max_people is not None and (engine != "pfb3x3" or int(max_people) < 0):
        raise ValueError("max_people is a number of people (0: no cap), for the tracking "
                         "engine only")
    offset = float(clock_offset_min or 0.0)
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("この動画の始まりの試合時間（分）は0以上にしてください")
    people_cap = None
    if engine == "pfb3x3":
        people_cap = (len(participants) or None) if max_people is None else (
            int(max_people) or None)
    paths = [Path(p) for p in cameras.values()]
    if not paths:
        raise VideoError("動画が選ばれていません")
    allowed = [settings.inbox_dir.resolve(), settings.data_dir.resolve()]
    for path in paths:
        real = path.resolve()  # follows symlinks
        if not any(real.is_relative_to(root) for root in allowed):
            raise VideoError("動画の場所が許可されていません"
                             f"（inboxフォルダまたはデータフォルダの中だけ使えます）: {path.name}")
    disk.ensure_room(settings)   # an inbox video is not copied, but its frames will be
    names = [f"cam{i}" for i in range(1, len(paths) + 1)]
    sid = None
    try:
        for p in participants:
            unknown = set(p.initial_positions) - set(names)
            if unknown:
                raise VideoError("登録されていないカメラが指定されています: "
                                 + ", ".join(sorted(unknown)))
        infos = []
        for cam, path in zip(names, paths):
            try:
                info = probe_video(path)
                infos.append((path, info))
                validate_interval(info, start_sec, end_sec, settings.frame_interval_sec)
            except VideoError as e:
                raise VideoError(f"{cam}（{path.name}）: {e}") from None
        meta = {"cameras": {cam: str(path) for cam, path in zip(names, paths)},
                "start_sec": None if start_sec is None else float(start_sec),
                "end_sec": None if end_sec is None else float(end_sec),
                "auto_transcribe": bool(auto_transcribe), "engine": engine}
        if engine == "pfb3x3":
            meta["max_people"] = people_cap   # None: no cap
        if hold_for_check:
            meta["check"] = "pending"
        if hold_for_check or offset:
            meta["clock_offset_min"] = offset
        sid = repo.create_submission(assignment_id, label.strip() or "無題", meta)
        for p in participants:
            repo.add_subject(sid, p.participant_id, p.display_name, p.appearance,
                             dict(p.initial_positions))
        if transcript_text and transcript_text.strip():
            segments = parse_pasted_transcript(transcript_text)
            repo.save_stage_run(StageRun(
                id=None, submission_id=sid, stage=Stage.transcribe, camera=None, subject_id=None,
                version=0, source="teacher", upstream_run_ids=[], prompt_hash=None,
                model_id=None, output={"segments": [s.to_dict() for s in segments]},
                status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))
        if hold_for_check:
            enqueue_once(repo, sid, Stage.detect)
        else:
            for cam in names:
                enqueue_once(repo, sid, Stage.prepare, camera=cam)
    except BaseException:
        if sid is not None:  # cascade removes subjects, runs and jobs
            repo.conn.execute("DELETE FROM submissions WHERE id=?", (sid,))
        uploads_root = _uploads_dir(settings).resolve()
        keep = chunked_upload.unclaimed_paths(settings)   # the coach's finished uploads
        for path in paths:  # only browser uploads saved into data/uploads; never other files
            real = path.resolve()
            if real.is_relative_to(uploads_root) and real not in keep:
                real.unlink(missing_ok=True)
        raise
    for path, info in infos:   # a 1080p proxy of a 4K video, made in the background
        proxy.start(settings, path, info.width, info.height)
    return sid


def rename_subject(repo: Repo, subject_id: int, display_name: str) -> None:
    name = display_name.strip()
    if not name:
        raise ValueError("名前が空です")
    repo.rename_subject(subject_id, name)
