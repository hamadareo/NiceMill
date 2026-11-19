import math
from pathlib import Path

import cv2
import numpy as np
from pdfhelp import make_pdf_bytes
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Kind, Stage
from pfb_app.pipeline.common import submission_engine
from pfb_app.preprocess.documents import DocumentError
from pfb_app.preprocess.frames import VideoError
from pfb_app.services.intake import (NO_PEOPLE_CAP, ParticipantInput, create_assignment_from_form,
                                     list_inbox_videos, register_documents,
                                     register_video_group, rename_subject,
                                     save_uploaded_video)
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def pdf_bytes(text="これは提出された書類の本文です。" * 5) -> bytes:
    return make_pdf_bytes([text])


def make_video(path: Path, seconds=20) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(seconds * 10):
        w.write(np.full((240, 320, 3), (i * 7) % 256, dtype=np.uint8))
    w.release()
    return path


@pytest.fixture
def doc_assignment(repo):
    return create_assignment_from_form(repo, "レポート", "document", "課題文", "", None,
                                       "standard", False)


@pytest.fixture
def video_assignment(repo):
    return create_assignment_from_form(repo, "授業", "video", "課題文", "", None,
                                       "standard", True)


def test_create_assignment_joins_rubric_file(repo, tmp_path):
    rub = tmp_path / "rubric.pdf"
    rub.write_bytes(pdf_bytes("ルーブリックの基準"))
    aid = create_assignment_from_form(repo, "t", "document", "課題", "手入力の基準", rub,
                                      "high", True)
    a = repo.get_assignment(aid)
    assert a["kind"] == "document" and a["model_key"] == "high"
    assert a["require_identity_approval"] is True
    assert a["rubric_text"].startswith("手入力の基準")
    assert "ルーブリックの基準" in a["rubric_text"]


def test_create_assignment_bad_rubric_file_raises(repo, tmp_path):
    bad = tmp_path / "rubric.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(DocumentError):
        create_assignment_from_form(repo, "t", "document", "課題", "", bad, "standard", False)
    assert repo.list_assignments() == []


def test_register_documents_reports_corrupt_file_and_continues(repo, settings, doc_assignment):
    report = register_documents(
        repo, settings, doc_assignment,
        [("山田太郎.pdf", pdf_bytes()), ("壊れ.pdf", b"garbage"), ("佐藤花子.pdf", pdf_bytes())],
        None)
    assert len(report.created) == 2
    assert [name for name, _ in report.errors] == ["壊れ.pdf"]
    assert report.errors[0][1]
    assert {n for _, n in report.created} == {"山田太郎", "佐藤花子"}
    subs = repo.list_submissions(doc_assignment)
    assert len(subs) == 2
    for sub in subs:
        assert len(repo.list_subjects(sub["id"])) == 1
        assert Path(sub["meta"]["source_path"]).is_file()
        assert set(sub["meta"]) == {"source_path", "original_filename"}


def test_register_documents_enqueues_prepare_on_cpu(repo, settings, doc_assignment):
    report = register_documents(repo, settings, doc_assignment, [("a.pdf", pdf_bytes())], None)
    sid = report.created[0][0]
    job = repo.conn.execute("SELECT * FROM jobs WHERE submission_id=?", (sid,)).fetchall()
    assert [(j["stage"], j["model_key"]) for j in job] == [("prepare", "cpu")]


def test_register_documents_safe_name_keeps_original(repo, settings, doc_assignment):
    report = register_documents(
        repo, settings, doc_assignment, [("../../etc/鈴木一郎.pdf", pdf_bytes())], None)
    sub = repo.get_submission(report.created[0][0])
    assert sub["meta"]["original_filename"] == "鈴木一郎.pdf"
    src = Path(sub["meta"]["source_path"]).resolve()
    assert files.submission_dir(settings, doc_assignment, sub["id"]).resolve() in src.parents
    assert report.created[0][1] == "鈴木一郎"


def test_register_documents_uses_roster(repo, settings, doc_assignment):
    report = register_documents(repo, settings, doc_assignment,
                                [("report_2024_田中.pdf", pdf_bytes())], ["田中", "田"])
    assert report.created[0][1] == "田中"


def test_register_documents_flags_duplicate_names(repo, settings, doc_assignment):
    report = register_documents(
        repo, settings, doc_assignment,
        [("山田太郎.pdf", pdf_bytes()), ("山田太郎 (1).pdf", pdf_bytes()),
         ("佐藤.pdf", pdf_bytes())], ["山田太郎", "佐藤"])
    assert report.duplicate_names == ["山田太郎"]
    assert len(report.created) == 3
    again = register_documents(repo, settings, doc_assignment, [("佐藤.pdf", pdf_bytes())],
                               ["佐藤"])
    assert again.duplicate_names == ["佐藤"]  # clashes with an earlier upload


def test_register_documents_rejects_unsupported_extension(repo, settings, doc_assignment):
    report = register_documents(repo, settings, doc_assignment, [("a.txt", b"hello")], None)
    assert report.created == [] and len(report.errors) == 1
    assert repo.list_submissions(doc_assignment) == []


def test_rename_subject(repo, settings, doc_assignment):
    report = register_documents(repo, settings, doc_assignment, [("a.pdf", pdf_bytes())], None)
    subject = repo.list_subjects(report.created[0][0])[0]
    rename_subject(repo, subject["id"], " 新しい名前 ")
    assert repo.get_subject(subject["id"])["display_name"] == "新しい名前"
    with pytest.raises(ValueError):
        rename_subject(repo, subject["id"], "  ")


def parts():
    return [ParticipantInput("Pa", "太郎", "赤い服", {"cam1": "左", "cam2": "右"}),
            ParticipantInput("Pb", "花子", "青い服", {"cam1": "右"})]


def test_register_video_group_writes_meta_and_jobs(repo, settings, video_assignment, tmp_path):
    v1 = make_video(settings.inbox_dir / "x.mp4")
    v2 = make_video(settings.inbox_dir / "y.mp4")
    sid = register_video_group(repo, settings, video_assignment, "A班",
                               {"ignored-name": v1, "また別": v2}, 0, 15,
                               None, True, parts())
    sub = repo.get_submission(sid)
    assert sub["label"] == "A班"
    assert sub["meta"] == {"cameras": {"cam1": str(v1), "cam2": str(v2)},
                           "start_sec": 0.0, "end_sec": 15.0, "auto_transcribe": True,
                           "engine": "paper"}
    subjects = repo.list_subjects(sid)
    assert [s["participant_id"] for s in subjects] == ["Pa", "Pb"]
    assert subjects[0]["initial_positions"] == {"cam1": "左", "cam2": "右"}
    jobs = repo.conn.execute("SELECT stage, camera, model_key FROM jobs ORDER BY id").fetchall()
    assert [tuple(j) for j in jobs] == [("prepare", "cam1", "cpu"), ("prepare", "cam2", "cpu")]
    assert repo.latest_run(sid, Stage.transcribe) is None


def test_register_video_group_stores_the_tracking_engine(repo, settings, video_assignment):
    v = make_video(settings.inbox_dir / "x.mp4")
    # appearance and positions are optional for the tracking engine
    sid = register_video_group(repo, settings, video_assignment, "A", {"c": v}, None, None,
                               None, False, [ParticipantInput("Pa", "太郎", "", {})],
                               engine="pfb3x3")
    assert repo.get_submission(sid)["meta"]["engine"] == "pfb3x3"
    assert repo.get_submission(sid)["meta"]["max_people"] == 1   # default: the participants
    assert submission_engine(repo.get_submission(sid)) == "pfb3x3"
    jobs = repo.conn.execute("SELECT stage, camera FROM jobs").fetchall()
    assert [tuple(j) for j in jobs] == [("prepare", "cam1")]
    two = [ParticipantInput("Pa", "太郎", "", {}), ParticipantInput("Pb", "花子", "", {})]
    for people, given, stored in ((two, None, 2), (two, 4, 4), (two, NO_PEOPLE_CAP, None),
                                  ([], None, None)):   # no participants: nothing to count
        capped = register_video_group(repo, settings, video_assignment, "B", {"c": v}, None,
                                      None, None, False, people, engine="pfb3x3",
                                      max_people=given)
        assert repo.get_submission(capped)["meta"]["max_people"] == stored
    paper = register_video_group(repo, settings, video_assignment, "C", {"c": v}, None, None,
                                 None, False, two)
    assert "max_people" not in repo.get_submission(paper)["meta"]


def test_register_video_group_refuses_an_unknown_engine(repo, settings, video_assignment):
    v = make_video(settings.inbox_dir / "x.mp4")
    with pytest.raises(ValueError):
        register_video_group(repo, settings, video_assignment, "A", {"c": v}, None, None,
                             None, False, [], engine="trackid")
    for engine, people in (("paper", 4), ("paper", 0), ("pfb3x3", -1)):   # tracking only
        with pytest.raises(ValueError):
            register_video_group(repo, settings, video_assignment, "A", {"c": v}, None, None,
                                 None, False, [], engine=engine, max_people=people)
    assert repo.list_submissions(video_assignment) == []


def test_register_video_group_pasted_transcript_saved(repo, settings, video_assignment, tmp_path):
    v = make_video(settings.inbox_dir / "x.mp4")
    sid = register_video_group(repo, settings, video_assignment, "A", {"c": v}, None, None,
                               "[00:05] こんにちは\n[00:10] はい", True,
                               [ParticipantInput("Pa", "太郎", "赤", {"cam1": "左"})])
    run = repo.latest_run(sid, Stage.transcribe)
    assert run.source == "teacher" and run.camera is None and run.status == "ok"
    segs = run.output["segments"]
    assert [s["text"] for s in segs] == ["こんにちは", "はい"]
    assert segs[-1]["end"] == math.inf
    stages = [r["stage"] for r in repo.conn.execute("SELECT stage FROM jobs")]
    assert stages == ["prepare"]


def test_register_video_rejects_bad_interval(repo, settings, video_assignment, tmp_path):
    v = make_video(settings.inbox_dir / "x.mp4")
    with pytest.raises(VideoError):
        register_video_group(repo, settings, video_assignment, "A", {"c": v}, 10, 5, None,
                             False, [])
    with pytest.raises(VideoError):
        register_video_group(repo, settings, video_assignment, "A", {"c": v}, 0, 999, None,
                             False, [])
    assert repo.list_submissions(video_assignment) == []
    assert repo.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_register_video_rejects_missing_or_corrupt_video(repo, settings, video_assignment,
                                                          tmp_path):
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    bad = settings.inbox_dir / "bad.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(VideoError):
        register_video_group(repo, settings, video_assignment, "A", {"c": bad}, None, None,
                             None, False, [])
    with pytest.raises(VideoError):
        register_video_group(repo, settings, video_assignment, "A", {}, None, None, None,
                             False, [])


def test_register_video_rejects_unknown_camera_in_positions(repo, settings, video_assignment,
                                                             tmp_path):
    v = make_video(settings.inbox_dir / "x.mp4")
    with pytest.raises(VideoError):
        register_video_group(repo, settings, video_assignment, "A", {"c": v}, None, None, None,
                             False, [ParticipantInput("Pa", "太郎", "", {"cam9": "左"})])


def test_inbox_video_not_copied(repo, settings, video_assignment):
    v = make_video(settings.inbox_dir / "classroom.mp4")
    assert list_inbox_videos(settings) == [v]
    sid = register_video_group(repo, settings, video_assignment, "A", {"c": v}, None, None,
                               None, False, [])
    assert repo.get_submission(sid)["meta"]["cameras"] == {"cam1": str(v)}
    copies = [p for p in settings.data_dir.rglob("*") if p.suffix == ".mp4"]
    assert copies == []


def test_save_uploaded_video_copies_into_data(settings, tmp_path):
    import io
    p = save_uploaded_video(settings, "../授業 1.mp4", io.BytesIO(b"abc"))
    assert p.read_bytes() == b"abc"
    assert settings.data_dir.resolve() in p.resolve().parents
    assert p.name.endswith(".mp4") and "/" not in p.name
    q = save_uploaded_video(settings, "授業 1.mp4", io.BytesIO(b"def"))
    assert q != p  # no overwrite


def test_list_inbox_videos_missing_dir(settings):
    assert list_inbox_videos(settings) == []


def test_register_video_rolls_back_on_any_failure_and_removes_upload(repo, settings,
                                                                     video_assignment,
                                                                     monkeypatch):
    import io
    import sqlite3
    up = save_uploaded_video(settings, "a.mp4", io.BytesIO(b""))
    make_video(up)
    inbox = make_video(settings.inbox_dir / "keep.mp4")

    def boom(*a, **k):
        raise sqlite3.IntegrityError("injected")
    monkeypatch.setattr(repo, "add_subject", boom)
    with pytest.raises(sqlite3.IntegrityError):
        register_video_group(repo, settings, video_assignment, "A", {"c": up, "d": inbox},
                             None, None, "[00:05] あ", False,
                             [ParticipantInput("Pa", "太郎", "", {})])
    assert repo.list_submissions(video_assignment) == []
    assert repo.conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    assert repo.conn.execute("SELECT COUNT(*) FROM stage_runs").fetchone()[0] == 0
    assert not up.exists()  # the copied upload is removed
    assert inbox.exists()  # inbox files are never deleted


def test_register_video_rejects_paths_outside_inbox_and_data(repo, settings, video_assignment,
                                                              tmp_path):
    outside = make_video(tmp_path / "elsewhere" / "x.mp4")
    with pytest.raises(VideoError, match="許可"):
        register_video_group(repo, settings, video_assignment, "A", {"c": outside}, None, None,
                             None, False, [])
    # a symlink inside the inbox that points outside is rejected too
    settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    link = settings.inbox_dir / "link.mp4"
    link.symlink_to(outside)
    with pytest.raises(VideoError, match="許可"):
        register_video_group(repo, settings, video_assignment, "A", {"c": link}, None, None,
                             None, False, [])
    assert outside.exists() and repo.list_submissions(video_assignment) == []


def test_register_video_rollback_only_removes_files_under_uploads(repo, settings,
                                                                  video_assignment, monkeypatch):
    import io
    import sqlite3
    up = save_uploaded_video(settings, "a.mp4", io.BytesIO(b""))
    make_video(up)
    other = make_video(settings.data_dir / "keep" / "other.mp4")  # in data/, not in uploads/

    def boom(*a, **k):
        raise sqlite3.IntegrityError("injected")
    monkeypatch.setattr(repo, "add_subject", boom)
    with pytest.raises(sqlite3.IntegrityError):
        register_video_group(repo, settings, video_assignment, "A", {"c": up, "d": other},
                             None, None, None, False, [ParticipantInput("Pa", "太郎", "", {})])
    assert not up.exists()
    assert other.exists()  # only uploads copied into data/uploads are removed
