"""「動画を自動で消す」 in the worker: what goes, what stays, never outside the work's own files,
never a work that is still analysed, and a stop half way is finished later.

Synthetic works only: generated videos and tiny made-up files."""
import os
from datetime import datetime, timedelta, timezone

import pytest

from data_helpers import hand_out
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.export import pdf, sheets
from pfb_app.pipeline.runner import enqueue_once
from pfb_app.services import kept, retention
from pfb_app.storage import files
from pdf_synthetic import video_group, work
from test_wizard_ui_basic import env  # noqa: F401 (env)

NOW = datetime(2026, 12, 1, 3, 0, tzinfo=timezone.utc)


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def outside(tmp_path):
    root = tmp_path / "outside"
    return {"dir": root, "folder_file": _write(root / "folder" / "keep.jpg", b"keep"),
            "linked": _write(root / "linked.jpg", b"linked")}


def finished(repo, aid, days_ago=40):
    """The work's handout was received `days_ago` days before NOW, after every approval."""
    hand_out(repo, aid, NOW - timedelta(days=days_ago))


def uploaded_work(settings, repo, outside, title="練習試合", upload=None):
    """A finished tracked work whose camera video is a browser upload in data/uploads, with
    frames, crops, the PDF picture cache, engine caches, and links pointing outside."""
    aid = work(repo, title=title)
    sid = video_group(settings, repo, aid, people=2, seconds=20, label=title)
    sub = repo.get_submission(sid)
    src = sub["meta"]["cameras"]["cam1"]
    if upload is None:
        upload = settings.data_dir / "uploads" / f"{abs(hash(title))}_{title}.mp4"
        upload.parent.mkdir(parents=True, exist_ok=True)
        os.replace(src, upload)
    meta = dict(sub["meta"], cameras={"cam1": str(upload)})
    repo.update_submission_meta(sid, meta)
    d = files.submission_dir(settings, aid, sid)
    _write(files.frames_dir(settings, aid, sid, "cam1") / "000001.jpg")
    _write(d / "people" / "cam1_P1_t1.jpg")
    _write(d / "handout" / "pictures" / "abc.jpg")
    _write(d / "pfb3x3" / "cache" / "det.npy")
    _write(d / "pfb3x3" / "frames" / "f1.jpg")
    _write(d / "pfb3x3" / "videos" / f"s{sid}_cam1.yaml", b"fps: 10\n")
    (d / "linked_folder").symlink_to(outside["folder_file"].parent)
    (d / "frames" / "cam1" / "linked.jpg").symlink_to(outside["linked"])
    finished(repo, aid)
    return aid, sid, upload


def media_files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in retention.MEDIA_SUFFIXES
                  and "kept" not in p.relative_to(root).parts)


def test_clear_media_removes_videos_and_heavy_files_and_keeps_the_rest(env, outside):
    settings, repo = env
    aid, sid, upload = uploaded_work(settings, repo, outside)
    subject = repo.list_subjects(sid)[0]
    text_before = repo.latest_run(sid, Stage.pfb, subject_id=subject["id"]).output["pfb"]
    moments_before = sheets.person_moments(repo, subject["id"])
    result = retention.clear_media(repo, settings, aid, NOW)
    assert result is not None and result.uploads == 1 and result.files >= 6
    assert not upload.exists()
    d = files.submission_dir(settings, aid, sid, create=False)
    work_dir = files.assignment_dir(settings, aid, create=False)
    assert media_files(work_dir) == []
    for gone in ("frames", "people", "handout", "pfb3x3/tracks", "pfb3x3/cache"):
        assert not (d / gone).exists(), gone
    assert (d / "pfb3x3" / "videos" / f"s{sid}_cam1.yaml").is_file()   # small records stay
    assert list((d / "pfb3x3" / "exp").rglob("behavior.engine.jsonl"))
    # never anything outside: the links lose only themselves
    assert outside["folder_file"].read_bytes() == b"keep"
    assert outside["linked"].read_bytes() == b"linked"
    assert (d / "linked_folder").is_symlink()
    # texts, approvals and the growth record's moments stay; the PDFs are saved
    assert repo.latest_run(sid, Stage.pfb, subject_id=subject["id"]).output["pfb"] == text_before
    assert repo.get_subject(subject["id"])["approval_status"] == "approved"
    after = sheets.person_moments(repo, subject["id"])
    assert [(m.ref_id, m.category) for m in after] == \
        [(m.ref_id, m.category) for m in moments_before]
    record = kept.current(repo, aid)
    assert record is not None and len(record.people) == 2
    assert repo.get_assignment(aid)["media_removed_at"] == NOW.isoformat(timespec="microseconds")
    assert repo.latest_audit(retention.DONE, f"assignment:{aid}")["detail"]["uploads"] == 1
    # nothing new is queued for it; the PDFs can still be made (without pictures)
    assert enqueue_once(repo, sid, Stage.pfb, subject_id=subject["id"]) is None
    assert pdf.render_student_pdf(repo, aid)[:4] == b"%PDF"
    # a second pass changes nothing
    assert retention.clear_media(repo, settings, aid, NOW + timedelta(hours=1)) is None


def test_an_upload_another_work_still_uses_is_kept(env, outside):
    settings, repo = env
    aid, _, upload = uploaded_work(settings, repo, outside, title="前半")
    other, _, _ = uploaded_work(settings, repo, outside, title="後半", upload=upload)
    repo.conn.execute("UPDATE assignments SET pdf_downloaded_at=NULL WHERE id=?", (other,))
    assert retention.clear_media(repo, settings, aid, NOW).uploads == 0
    assert upload.is_file()
    finished(repo, other)
    assert retention.clear_media(repo, settings, other, NOW).uploads == 1   # last user
    assert not upload.exists()


def test_a_work_with_an_active_step_is_left_alone(env, outside):
    settings, repo = env
    aid, sid, upload = uploaded_work(settings, repo, outside)
    for status in (JobStatus.queued, JobStatus.running, JobStatus.waiting_gpu):
        repo.conn.execute("DELETE FROM jobs")
        job = repo.enqueue(sid, Stage.linecall, "linecall", camera="cam1")
        repo.conn.execute("UPDATE jobs SET status=? WHERE id=?", (status.value, job))
        assert retention.clear_media(repo, settings, aid, NOW) is None
        assert upload.is_file() and repo.get_assignment(aid)["media_removed_at"] is None
    assert kept.load(settings, aid) is None


def test_nothing_goes_when_the_pdfs_cannot_be_saved(env, outside, monkeypatch):
    settings, repo = env
    aid, _, upload = uploaded_work(settings, repo, outside)

    def broken(repo, aid):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(kept, "save", broken)
    assert retention.clear_media(repo, settings, aid, NOW) is None
    assert upload.is_file() and repo.get_assignment(aid)["media_removed_at"] is None


def test_a_clearing_stopped_half_way_is_finished_by_the_next_pass(env, outside, monkeypatch):
    settings, repo = env
    aid, _, upload = uploaded_work(settings, repo, outside)
    retention.save(repo, 30, False)
    monkeypatch.setattr(retention, "_finish", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OSError):
        retention.clear_media(repo, settings, aid, NOW)
    assert repo.get_assignment(aid)["media_removed_at"] and upload.is_file()
    monkeypatch.undo()
    retention.sweep(repo, settings, NOW)
    assert not upload.exists()
    assert repo.latest_audit(retention.DONE, f"assignment:{aid}") is not None


def test_sweep_follows_the_setting(env, outside):
    settings, repo = env
    a, _, up_a = uploaded_work(settings, repo, outside, title="A")
    b, _, up_b = uploaded_work(settings, repo, outside, title="B")
    finished(repo, b, days_ago=10)
    assert retention.sweep(repo, settings, NOW) == {"cleared": [], "deleted": []}   # off
    assert up_a.is_file()
    retention.save(repo, 30, False)
    assert retention.sweep(repo, settings, NOW) == {"cleared": [a], "deleted": []}
    assert not up_a.exists() and up_b.is_file()
    retention.save(repo, 30, True)
    assert retention.sweep(repo, settings, NOW) == {"cleared": [], "deleted": [a]}
    with pytest.raises(KeyError):
        repo.get_assignment(a)
    assert not files.assignment_dir(settings, a, create=False).exists()
    assert repo.get_assignment(b)["title"] == "B"


def test_sweep_handles_at_most_a_few_works_per_pass(env, outside, monkeypatch):
    settings, repo = env
    monkeypatch.setattr(retention, "MAX_PER_PASS", 1)
    a, _, _ = uploaded_work(settings, repo, outside, title="A")
    b, _, _ = uploaded_work(settings, repo, outside, title="B")
    retention.save(repo, 30, False)
    assert retention.sweep(repo, settings, NOW)["cleared"] == [a]
    assert retention.sweep(repo, settings, NOW)["cleared"] == [b]


def test_a_symlinked_work_folder_is_never_followed(env, outside, tmp_path):
    settings, repo = env
    aid, sid, _ = uploaded_work(settings, repo, outside)
    work_dir = files.assignment_dir(settings, aid, create=False)
    moved = tmp_path / "elsewhere"
    os.replace(work_dir, moved)
    work_dir.symlink_to(moved)
    assert retention.clear_media(repo, settings, aid, NOW) is None   # nothing is kept there
    assert list(moved.rglob("*.jpg")) and not (moved / "kept").exists()
    assert repo.get_assignment(aid)["media_removed_at"] is None


def test_the_worker_runs_the_pass_hourly_and_survives_its_failure(env, monkeypatch):
    from pfb_app.worker import main
    settings, repo = env
    calls = []
    monkeypatch.setattr(retention, "sweep", lambda repo, settings: calls.append(1) or
                        {"cleared": [1], "deleted": []})
    clock = iter([0.0, 10.0, retention.SWEEP_SEC + 1]).__next__
    last = {"t": None}
    for _ in range(3):
        main._retention_pass(repo, settings, clock, last)
    assert len(calls) == 2

    def broken(repo, settings):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(retention, "sweep", broken)
    main._retention_pass(repo, settings, lambda: 0.0, {"t": None})   # logged, not raised


def test_one_worker_pass_clears_a_due_work(env, outside, monkeypatch):
    from test_worker import StubGuard
    from pfb_app.worker.main import worker_loop
    settings, repo = env
    aid, _, upload = uploaded_work(settings, repo, outside)
    hand_out(repo, aid, datetime.now(timezone.utc) - timedelta(days=40))
    retention.save(repo, 30, False)
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("must not spawn"),
                sleep=lambda s: None, once=True, on_start=lambda s: None)
    assert not upload.exists() and repo.get_assignment(aid)["media_removed_at"]


def test_an_approval_changed_while_the_pdfs_were_made_keeps_the_videos(env, outside,
                                                                      monkeypatch):
    settings, repo = env
    aid, sid, upload = uploaded_work(settings, repo, outside)
    real = kept.save

    def save_then_unapprove(repo, aid):
        record = real(repo, aid)
        repo.set_approval(repo.list_subjects(sid)[0]["id"], None)
        return record
    monkeypatch.setattr(kept, "save", save_then_unapprove)
    assert retention.clear_media(repo, settings, aid, NOW) is None
    assert upload.is_file() and repo.get_assignment(aid)["media_removed_at"] is None


def test_a_work_with_nobody_approved_any_more_is_never_due(env, outside):
    settings, repo = env
    aid, sid, upload = uploaded_work(settings, repo, outside)
    for s in repo.list_subjects(sid):
        repo.set_approval(s["id"], None)
    assert retention.due(repo, retention.Policy(30, False), NOW) == []
    assert retention.due(repo, retention.Policy(30, True), NOW) == [aid]   # すべて消す: goes


def test_a_failing_work_never_holds_the_others_up(env, outside, monkeypatch):
    settings, repo = env
    monkeypatch.setattr(retention, "MAX_PER_PASS", 1)
    a, _, _ = uploaded_work(settings, repo, outside, title="A")
    b, _, up_b = uploaded_work(settings, repo, outside, title="B")
    real = kept.save

    def save(repo, aid):
        if aid == a:
            raise RuntimeError("synthetic")
        return real(repo, aid)
    monkeypatch.setattr(kept, "save", save)
    retention.save(repo, 30, False)
    assert retention.sweep(repo, settings, NOW)["cleared"] == [b] and not up_b.exists()


def test_nothing_can_be_queued_or_retried_once_the_videos_are_gone(env, outside):
    from pfb_app.pipeline.runner import request_names
    settings, repo = env
    aid, sid, _ = uploaded_work(settings, repo, outside)
    job = repo.enqueue(sid, Stage.names, "cpu", camera="cam1")
    repo.conn.execute("UPDATE jobs SET status=? WHERE id=?", (JobStatus.failed.value, job))
    finished(repo, aid)        # the failed job is from before the handout
    assert retention.clear_media(repo, settings, aid, NOW) is not None
    assert not repo.retry_job(job)
    assert request_names(repo, sid) == []


def test_a_cleared_work_takes_no_new_videos_and_says_why(env, outside):
    from pfb_app.preprocess.frames import VideoError
    from pfb_app.services import content_check, intake
    from test_wizard_ui_basic import app, ok, primaries, screen_text
    settings, repo = env
    aid, sid, _ = uploaded_work(settings, repo, outside)
    retention.clear_media(repo, settings, aid, NOW)
    video = _write(settings.inbox_dir / "後半.mp4", b"v")
    with pytest.raises(VideoError, match="新しい作業として"):
        intake.register_video_group(repo, settings, aid, "後半", {"cam1": video}, 0.0, None,
                                    "", False, [], engine="pfb3x3")
    with pytest.raises(ValueError, match="新しい作業として"):
        content_check.add_videos_to_work(repo, settings, aid, [video])
    assert len(repo.list_submissions(aid)) == 1
    at = ok(app(session={"assignment_id": aid, "step": 2, "submission_id": None}))
    assert primaries(at) == ["新しい作業として始める"]
    assert "新しい作業として始めてください" in screen_text(at)


def test_a_cleared_document_work_takes_no_new_documents(env):
    from pfb_app.services import intake
    from pdf_synthetic import document_work
    settings, repo = env
    aid = document_work(repo)
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-01-01' WHERE id=?", (aid,))
    report = intake.register_documents(repo, settings, aid, [("b.pdf", b"%PDF")], None)
    assert not report.created and report.errors == [("b.pdf", intake.CLEARED)]


def test_a_work_whose_pdfs_cannot_be_saved_waits_a_day_and_says_so(env, outside, monkeypatch):
    """Review LOW 9: it neither shows 「まもなく消えます」 forever nor uses up every pass."""
    settings, repo = env
    now = datetime.now(timezone.utc)
    aid, _, upload = uploaded_work(settings, repo, outside)
    hand_out(repo, aid, now - timedelta(days=40))
    policy = retention.save(repo, 30, False)
    monkeypatch.setattr(kept, "save", lambda repo, aid: (_ for _ in ()).throw(RuntimeError()))
    assert retention.clear_media(repo, settings, aid, now) is None
    assert retention.due(repo, policy, now) == []
    assert retention.due(repo, policy, now + retention.FAIL_PAUSE + timedelta(hours=1)) == [aid]
    line = retention.card_line(repo, repo.get_assignment(aid), policy, now)
    assert line == "動画を自動で消せませんでした。続くときは、管理者に知らせてください"
    assert upload.is_file()


def test_a_work_with_nobody_approved_says_its_videos_stay(env, outside):
    settings, repo = env
    aid, sid, _ = uploaded_work(settings, repo, outside)
    for s in repo.list_subjects(sid):
        repo.set_approval(s["id"], None)
    line = retention.card_line(repo, repo.get_assignment(aid), retention.Policy(30, False), NOW)
    assert line == "承認した文章がないため、動画は自動では消しません"


def test_a_work_reopened_while_the_pdfs_were_made_keeps_its_videos(env, outside, monkeypatch):
    """Review R28: finished is checked again inside the flag's transaction."""
    settings, repo = env
    aid, _, upload = uploaded_work(settings, repo, outside)
    real = kept.save

    def save_then_add_a_group(repo, aid):
        record = real(repo, aid)
        repo.create_submission(aid, "後半", {"check": "pending"})
        return record
    monkeypatch.setattr(kept, "save", save_then_add_a_group)
    assert retention.clear_media(repo, settings, aid, NOW) is None
    assert upload.is_file() and repo.get_assignment(aid)["media_removed_at"] is None
