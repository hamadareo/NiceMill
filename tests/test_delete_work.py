"""Deleting a work from the home screen (user request 2026-10-22: 「ホーム画面で削除できるように」).

The service (services/deletion.py) removes the work's rows and only the app's own files under
data_dir: never a file outside it (the inbox source, the target of a symlink, the other name
of a hardlinked file). The home screen asks first, in a confirm area under the card.

Only synthetic data: tiny made-up files, no real video.
"""
import os
import re
import shutil

import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.pipeline import runner
from pfb_app.services import deletion
from pfb_app.storage import files
from pfb_app.ui.wizard import theme
from pfb_app.worker.main import worker_loop
from test_neutral_wording import SCHOOL_WORDS
from test_wizard_ui_basic import (FORBIDDEN, app, env, html, ok, primaries,  # noqa: F401 (env)
                                  screen_text, video_assignment)
from test_runner import doc_submission, make_ctx
from test_wizard_ui_people_wait import save
from test_worker import StubGuard


def _write(path, data=b"x"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def outside(tmp_path):
    """Files outside data_dir that a work may point at: none of them may ever go."""
    root = tmp_path / "outside"
    return {"dir": root,
            "inbox": _write(tmp_path / "inbox" / "試合.mp4", b"inbox"),
            "dataset": _write(root / "dataset" / "clip.mp4", b"dataset"),
            "linked": _write(root / "linked.mp4", b"linked"),
            "folder_file": _write(root / "folder" / "keep.txt", b"keep")}


def full_work(settings, repo, outside, title="消す作業"):
    """A video work with an upload copy, an inbox video, a symlink and a hardlink in uploads,
    frames, pages, engine outputs, a person, runs and a finished job."""
    up = settings.data_dir / "uploads"
    own = _write(up / "ab12cd34_own.mp4", b"own")
    sym = up / "ef56_sym.mp4"
    sym.symlink_to(outside["linked"])
    hard = up / "aa11_hard.mp4"
    os.link(outside["dataset"], hard)
    aid = video_assignment(repo, title)
    sid = repo.create_submission(aid, "A班", {"cameras": {
        "cam1": str(own), "cam2": str(outside["inbox"]), "cam3": str(sym), "cam4": str(hard)},
        "engine": "pfb3x3"})
    subj = repo.add_subject(sid, None, "Pa", None, {})
    prep = save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    save(repo, sid, Stage.pfb, {"pfb": "よくできました。"}, subject_id=subj, upstream=[prep.id])
    job = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    repo.claim_next_job({"cpu"})
    repo.finish_job(job, JobStatus.done)
    _write(files.frames_dir(settings, aid, sid, "cam1") / "000001.jpg")
    _write(files.pages_dir(settings, aid, sid) / "p1.png")
    _write(files.submission_dir(settings, aid, sid) / "pfb3x3" / "out" / "tracks.json")
    # a symlink to a folder outside, inside the work's own folder
    (files.submission_dir(settings, aid, sid) / "linked_folder").symlink_to(
        outside["folder_file"].parent)
    return aid, sid, {"own": own, "sym": sym, "hard": hard}


def count(repo, table):
    return repo.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


# --- the service ----------------------------------------------------------------------------

def test_delete_removes_rows_and_only_files_inside_data_dir(env, outside):
    settings, repo = env
    aid, sid, ups = full_work(settings, repo, outside)
    other = video_assignment(repo, "残す作業")
    keep_dir = _write(files.frames_dir(settings, other, 99, "cam1") / "f.jpg")
    result = deletion.delete_work(repo, settings, aid)
    assert result.title == "消す作業"
    for table in ("submissions", "subjects", "stage_runs", "jobs"):
        assert count(repo, table) == 0
    assert [a["id"] for a in repo.list_assignments()] == [other]
    assert not files.assignment_dir(settings, aid, create=False).exists()
    # the app's own copies in uploads are gone (links removed, not their targets)
    assert not ups["own"].exists()
    assert not ups["sym"].is_symlink() and not ups["hard"].exists()
    # nothing outside data_dir was touched
    assert outside["inbox"].read_bytes() == b"inbox"
    assert outside["linked"].read_bytes() == b"linked"
    assert outside["dataset"].read_bytes() == b"dataset"
    assert outside["folder_file"].read_bytes() == b"keep"
    assert keep_dir.exists()            # another work's files stay


def test_an_upload_another_work_still_uses_is_kept(env, outside):
    settings, repo = env
    aid, _, ups = full_work(settings, repo, outside)
    other = video_assignment(repo, "同じ動画の作業")
    repo.create_submission(other, "B班", {"cameras": {"cam1": str(ups["own"])}})
    deletion.delete_work(repo, settings, aid)
    assert ups["own"].read_bytes() == b"own"


def test_a_work_folder_that_is_a_symlink_loses_only_the_link(env, outside):
    settings, repo = env
    aid = video_assignment(repo)
    target = outside["dir"] / "elsewhere"
    _write(target / "submissions" / "1" / "f.jpg")
    adir = files.assignment_dir(settings, aid, create=False)
    adir.parent.mkdir(parents=True, exist_ok=True)
    adir.symlink_to(target, target_is_directory=True)
    deletion.delete_work(repo, settings, aid)
    assert not adir.is_symlink() and not adir.exists()
    assert (target / "submissions" / "1" / "f.jpg").exists()


def test_paths_outside_data_dir_are_never_removed(env, tmp_path):
    settings, _ = env
    victim = _write(tmp_path / "victim.mp4")
    assert deletion._own_entry(settings, victim) is None
    assert deletion._own_entry(settings, settings.data_dir / "uploads" / ".." / ".." /
                               "victim.mp4") is None
    assert deletion._own_entry(settings, settings.data_dir / "uploads") is None
    inside = settings.data_dir / "uploads" / "x.mp4"
    assert deletion._own_entry(settings, inside) == inside.parent.resolve() / inside.name


def test_the_audit_entry_has_counts_only(env, outside):
    settings, repo = env
    aid, _, _ = full_work(settings, repo, outside, title="山田さんの試合")
    deletion.delete_work(repo, settings, aid)
    entry = repo.latest_audit("delete_work", f"assignment:{aid}")
    assert entry is not None
    detail = entry["detail"]
    assert detail["groups"] == 1 and detail["people"] == 1 and detail["results"] == 2
    assert detail["uploads_removed"] == 3 and detail["jobs_cancelled"] == 0
    assert all(isinstance(v, int) for v in detail.values()), detail
    assert "山田" not in str(entry)


def test_an_unknown_work_raises_key_error(env):
    settings, repo = env
    with pytest.raises(KeyError):
        deletion.delete_work(repo, settings, 404)
    with pytest.raises(KeyError):
        deletion.deletion_plan(repo, 404)


def test_active_work_is_refused_unless_the_analysis_is_stopped(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = repo.create_submission(aid, "A班", {})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    w = repo.enqueue(sid, Stage.track, "track")
    repo.mark_waiting_gpu("track", "GPUに空きがありません")
    assert repo.get_job(w).status == JobStatus.waiting_gpu
    plan = deletion.deletion_plan(repo, aid)
    assert (plan.title, plan.groups, plan.pending, plan.running) == ("ペーパータワー", 1, 2, 0)
    assert plan.active
    with pytest.raises(deletion.WorkActive):
        deletion.delete_work(repo, settings, aid)
    assert count(repo, "jobs") == 2 and repo.get_assignment(aid)
    result = deletion.delete_work(repo, settings, aid, stop_analysis=True)
    assert result.jobs_cancelled == 2
    assert repo.list_assignments() == [] and count(repo, "jobs") == 0


def test_a_running_step_is_never_killed_the_work_goes_when_it_ends(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = repo.create_submission(aid, "A班", {"engine": "pfb3x3"})
    running = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    repo.claim_next_job({"cpu"})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam2")
    with pytest.raises(deletion.StillRunning) as e:
        deletion.delete_work(repo, settings, aid, stop_analysis=True)
    assert str(e.value) == "いまの処理が終わると削除されます。"
    assert e.value.cancelled == 1
    # the waiting step is cancelled, the running one is left to finish; the work is marked
    assert [j["id"] for j in repo.submission_job_rows(sid)] == [running]
    assert repo.get_job(running).status == JobStatus.running
    assert repo.get_submission(sid)["meta"] == {"engine": "pfb3x3", "delete_requested": True}
    assert repo.latest_audit("delete_requested", f"assignment:{aid}")["detail"] == {
        "jobs_cancelled": 1, "running": 1}
    assert deletion.pending_deletions(repo) == {aid}
    # nothing new is queued for it, and nothing is deleted while the step runs
    assert runner.enqueue_once(repo, sid, Stage.track, "cam1") is None
    assert runner.request_names(repo, sid) == []
    assert deletion.finish_requested_deletions(repo, settings) == []
    assert repo.get_assignment(aid)
    repo.finish_job(running, JobStatus.done)
    assert deletion.finish_requested_deletions(repo, settings) == [aid]
    assert repo.list_assignments() == [] and deletion.pending_deletions(repo) == set()


def test_the_request_survives_a_group_meta_rewrite(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = repo.create_submission(aid, "A班", {})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    job = repo.claim_next_job({"cpu"})
    with pytest.raises(deletion.StillRunning):
        deletion.delete_work(repo, settings, aid, stop_analysis=True)
    repo.update_submission_meta(sid, {"check": "ready"})   # a running step rewrote the meta
    assert deletion.pending_deletions(repo) == {aid}        # the audit entry still holds it
    repo.finish_job(job.id, JobStatus.done)
    assert deletion.finish_requested_deletions(repo, settings) == [aid]


def test_through_execute_job_the_finished_step_queues_nothing_and_the_work_goes(
        env, tmp_path, monkeypatch):
    settings, repo = env
    aid, sid, _ = doc_submission(repo, tmp_path)
    repo.enqueue(sid, Stage.prepare, "cpu")
    job = repo.claim_next_job({"cpu"})
    with pytest.raises(deletion.StillRunning):
        deletion.delete_work(repo, settings, aid, stop_analysis=True)
    queued = []
    real = repo.enqueue
    monkeypatch.setattr(repo, "enqueue", lambda *a, **k: queued.append(a) or real(*a, **k))
    runner.execute_job(job, make_ctx(settings, repo))
    assert queued == []                                  # no evidence step after prepare
    assert repo.list_assignments() == []
    assert not files.assignment_dir(settings, aid, create=False).exists()
    assert repo.latest_audit("delete_work", f"assignment:{aid}") is not None


def test_without_a_request_execute_job_still_queues_the_next_step(env, tmp_path):
    settings, repo = env
    aid, sid, _ = doc_submission(repo, tmp_path)
    repo.enqueue(sid, Stage.prepare, "cpu")
    runner.execute_job(repo.claim_next_job({"cpu"}), make_ctx(settings, repo))
    assert [(j["stage"], j["status"]) for j in repo.submission_job_rows(sid)] == [
        ("prepare", "done"), ("evidence", "queued")]
    assert repo.get_assignment(aid)


def test_the_worker_finishes_requested_deletions_and_sweeps_orphan_folders(env, outside):
    settings, repo = env
    aid = video_assignment(repo, "待っていた作業")
    sid = repo.create_submission(aid, "A班", {})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    job = repo.claim_next_job({"cpu"})
    with pytest.raises(deletion.StillRunning):
        deletion.delete_work(repo, settings, aid, stop_analysis=True)
    repo.finish_job(job.id, JobStatus.failed_retryable, "heartbeat timeout")   # e.g. a crash
    keep = video_assignment(repo, "残る作業")
    kept = _write(files.frames_dir(settings, keep, 1, "cam1") / "f.jpg")
    # deletions that stopped after their row went (the audit entry is written with it)
    orphan_id, linked_id, silent_id = (crashed(settings, repo, title, action) for title, action
                                       in (("落ちた作業", "delete_work"),
                                           ("リンクの作業", "delete_requested"),
                                           ("記録のない作業", None)))
    orphan = files.frames_dir(settings, orphan_id, 1, "cam1") / "f.jpg"
    linked = files.assignment_dir(settings, linked_id, create=False)
    shutil.rmtree(linked)                      # this one was a link to a folder outside
    linked.symlink_to(outside["folder_file"].parent, target_is_directory=True)
    worker_loop(settings, guard=StubGuard(), spawn=lambda key: pytest.fail("must not spawn"),
                sleep=lambda s: None, once=True, on_start=lambda s: None)
    assert [a["id"] for a in repo.list_assignments()] == [keep]
    assert kept.exists() and not orphan.exists()
    assert not files.assignment_dir(settings, orphan_id, create=False).exists()
    assert not linked.is_symlink() and outside["folder_file"].read_bytes() == b"keep"
    assert files.assignment_dir(settings, silent_id, create=False).exists()   # no record: kept


def crashed(settings, repo, title, action):
    """A work whose row was deleted (recorded as `action` in the audit log, or not at all)
    while its folder stayed."""
    aid = video_assignment(repo, title)
    _write(files.frames_dir(settings, aid, 1, "cam1") / "f.jpg")
    repo.conn.execute("DELETE FROM assignments WHERE id=?", (aid,))
    if action:
        repo.audit(action, f"assignment:{aid}", {})
    return aid


def folders(settings, ids):
    return {aid: _write(files.frames_dir(settings, aid, 1, "cam1") / "f.jpg") for aid in ids}


def test_a_fresh_or_empty_database_never_sweeps_existing_folders(tmp_path):
    from pfb_app.storage.db import Repo, connect, init_schema
    settings = Settings(data_dir=tmp_path / "data")
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    old = folders(settings, (1, 2, 3))
    assert deletion.sweep_orphan_dirs(repo, settings) == 0          # no work was ever made
    aid = video_assignment(repo, "一つ目")
    repo.conn.execute("DELETE FROM assignments WHERE id=?", (aid,))
    repo.audit("delete_work", f"assignment:{aid}", {})
    assert deletion.sweep_orphan_dirs(repo, settings) == 0          # no work left at all
    assert all(f.exists() for f in old.values())


def test_a_restored_older_backup_keeps_newer_folders(env):
    settings, repo = env
    first, second = video_assignment(repo, "残る作業"), video_assignment(repo, "消えた作業")
    repo.conn.execute("DELETE FROM assignments WHERE id=?", (second,))
    for aid in (second, second + 1, second + 2):   # the backup even names a newer id
        repo.audit("delete_work", f"assignment:{aid}", {})
    made = folders(settings, (second, second + 1, second + 2))
    odd = _write(settings.data_dir / "assignments" / "３" / "f.jpg")   # not ASCII digits
    assert deletion.sweep_orphan_dirs(repo, settings) == 1
    assert not made[second].exists()
    assert made[second + 1].exists() and made[second + 2].exists() and odd.exists()
    assert repo.get_assignment(first)


def test_a_symlinked_assignments_folder_is_never_swept(env, outside):
    settings, repo = env
    video_assignment(repo, "残る作業")
    gone = crashed(settings, repo, "消えた作業", "delete_work")
    real = settings.data_dir / "real_assignments"
    (settings.data_dir / "assignments").rename(real)
    (settings.data_dir / "assignments").symlink_to(real, target_is_directory=True)
    assert deletion.sweep_orphan_dirs(repo, settings) == 0
    assert (real / str(gone)).exists()


def test_call_log_rows_are_left_untouched(env, outside):
    """Rows with no owner (written outside a job, or before call_log_owner existed) cannot be
    traced to a work and stay (see services.deletion; owned rows: test_call_log_owner)."""
    settings, repo = env
    aid, _, _ = full_work(settings, repo, outside)
    repo.cache_put("h1", "m", "{}", 1, 1, 0.1, None)
    deletion.delete_work(repo, settings, aid)
    assert repo.cache_get("h1") == "{}"


def test_inbox_inside_uploads_and_nested_entries_are_never_removed(tmp_path):
    settings = Settings(data_dir=tmp_path / "data",
                        inbox_dir=tmp_path / "data" / "uploads" / "inbox")
    up = settings.data_dir / "uploads"
    in_inbox = _write(settings.inbox_dir / "試合.mp4", b"inbox")
    nested = _write(up / "sub" / "x.mp4")
    direct = _write(up / "x.mp4")
    assert deletion._own_entry(settings, in_inbox) is None
    assert deletion._own_entry(settings, nested) is None
    assert deletion._own_entry(settings, settings.inbox_dir) is None
    assert deletion._own_entry(settings, direct) == direct.parent.resolve() / direct.name
    inbox_is_uploads = Settings(data_dir=tmp_path / "data", inbox_dir=up)
    assert deletion._own_entry(inbox_is_uploads, direct) is None


def test_a_hardlink_of_an_inbox_file_inside_uploads_loses_only_the_app_name(env, tmp_path):
    _, repo = env
    settings = Settings(data_dir=tmp_path / "d2", inbox_dir=tmp_path / "d2" / "uploads" / "inbox")
    src = _write(settings.inbox_dir / "clip.mp4", b"inbox")
    hard = settings.data_dir / "uploads" / "aa_clip.mp4"
    os.link(src, hard)
    aid = video_assignment(repo)
    repo.create_submission(aid, "A班", {"cameras": {"cam1": str(src), "cam2": str(hard)}})
    deletion.delete_work(repo, settings, aid)
    assert src.read_bytes() == b"inbox"          # the inbox name stays
    assert not hard.exists()                     # the app's own name goes


# --- the home screen ------------------------------------------------------------------------

def _open_confirm(at, aid):
    return ok(at.button(key=f"pfb_del_{aid}").click().run())


def test_each_card_has_a_quiet_delete_and_home_keeps_one_primary(env):
    _, repo = env
    a = video_assignment(repo, "一つ目")
    b = video_assignment(repo, "二つ目")
    at = ok(app())
    for aid in (a, b):
        btn = at.button(key=f"pfb_del_{aid}")
        assert btn.label == "削除" and btn.proto.type == "tertiary"
    before = primaries(at)
    assert len(before) == 1
    at = _open_confirm(at, a)
    assert primaries(at) == before            # the confirm area adds no primary


def test_confirm_area_copy_and_delete(env):
    settings, repo = env
    aid = video_assignment(repo, "練習試合")
    keep = video_assignment(repo, "残る作業")
    at = _open_confirm(ok(app()), aid)
    page = screen_text(at)
    for text in ("「練習試合」を削除しますか？", "動画・書類・解析結果・PDF の記録",
                 "この操作は元に戻せません。", "削除してよいことを確認しました"):
        assert text in page, text
    assert '<p class="pfb-delete__undo">この操作は元に戻せません。</p>' in html(at)
    go = at.button(key=f"pfb_delete_go_{aid}")
    assert go.label == "削除する" and go.proto.disabled and go.proto.type == "secondary"
    assert go.proto.icon == ":material/delete:"
    assert at.button(key=f"pfb_delete_cancel_{aid}").label == "やめる"
    at = ok(at.checkbox(key=f"pfb_delete_ok_go_{aid}").check().run())
    assert not at.button(key=f"pfb_delete_go_{aid}").proto.disabled
    at = ok(at.button(key=f"pfb_delete_go_{aid}").click().run())
    assert any(s.value == "「練習試合」を削除しました。" for s in at.success)
    assert at.session_state["wizard"]["assignment_id"] is None      # still home
    assert [a["id"] for a in repo.list_assignments()] == [keep]
    assert "練習試合" not in html(at)


def test_cancel_closes_the_confirm_area_and_keeps_the_work(env):
    _, repo = env
    aid = video_assignment(repo, "練習試合")
    at = _open_confirm(ok(app()), aid)
    at = ok(at.button(key=f"pfb_delete_cancel_{aid}").click().run())
    assert "削除しますか" not in screen_text(at)
    assert repo.get_assignment(aid)


def test_an_active_work_offers_to_stop_and_delete(env):
    _, repo = env
    aid = video_assignment(repo, "解析中の作業")
    sid = repo.create_submission(aid, "A班", {})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    at = _open_confirm(ok(app()), aid)
    assert at.button(key=f"pfb_delete_stop_{aid}").label == "解析を止めて削除"
    at = ok(at.checkbox(key=f"pfb_delete_ok_stop_{aid}").check().run())
    at = ok(at.button(key=f"pfb_delete_stop_{aid}").click().run())
    assert any(s.value == "「解析中の作業」を削除しました。" for s in at.success)
    assert repo.list_assignments() == []


def test_a_plain_delete_never_silently_stops_analysis_started_meanwhile(env):
    _, repo = env
    aid = video_assignment(repo, "練習試合")
    at = _open_confirm(ok(app()), aid)
    at = ok(at.checkbox(key=f"pfb_delete_ok_go_{aid}").check().run())
    sid = repo.create_submission(aid, "A班", {})
    job = repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")   # started in another tab
    at = ok(at.button(key=f"pfb_delete_go_{aid}").click().run())
    assert repo.get_assignment(aid) and repo.get_job(job).status == JobStatus.queued
    stop = at.button(key=f"pfb_delete_stop_{aid}")       # shown now, unticked and unpressed
    assert stop.label == "解析を止めて削除" and stop.proto.disabled
    assert not at.checkbox(key=f"pfb_delete_ok_stop_{aid}").value


def test_a_running_step_marks_the_card_until_the_worker_deletes_it(env):
    settings, repo = env
    aid = video_assignment(repo, "解析中の作業")
    sid = repo.create_submission(aid, "A班", {})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    job = repo.claim_next_job({"cpu"})
    at = _open_confirm(ok(app()), aid)
    at = ok(at.checkbox(key=f"pfb_delete_ok_stop_{aid}").check().run())
    at = ok(at.button(key=f"pfb_delete_stop_{aid}").click().run())
    assert any("いまの処理が終わると削除されます。" in w.value for w in at.warning)
    assert repo.get_assignment(aid)
    assert "いまの処理が終わると削除されます" in html(at)
    assert not [b for b in at.button if b.key == f"pfb_del_{aid}"]
    assert "削除しますか" not in screen_text(at)
    repo.finish_job(job.id, JobStatus.done)
    deletion.finish_requested_deletions(repo, settings)     # the worker, after the step
    at = ok(at.run())
    assert "解析中の作業" not in html(at)


def test_a_work_deleted_meanwhile_is_a_plain_message(env):
    settings, repo = env
    aid = video_assignment(repo, "別の画面で消した作業")
    at = _open_confirm(ok(app()), aid)
    at = ok(at.checkbox(key=f"pfb_delete_ok_go_{aid}").check().run())
    deletion.delete_work(repo, settings, aid)          # e.g. from another tab
    at = ok(at.button(key=f"pfb_delete_go_{aid}").click().run())
    assert "削除しますか" not in screen_text(at)
    assert not at.exception


def test_the_tab_where_the_work_is_open_goes_home(env):
    settings, repo = env
    aid = video_assignment(repo, "開いている作業")
    at = ok(app(session={"assignment_id": aid, "step": None, "submission_id": None}))
    deletion.delete_work(repo, settings, aid)            # deleted from another tab's home
    at = ok(at.run())
    assert at.session_state["wizard"]["assignment_id"] is None
    assert any("見つかりませんでした" in s.value for s in at.success)


def test_the_confirm_area_has_no_school_or_internal_words(env):
    _, repo = env
    aid = video_assignment(repo, "活動")
    sid = repo.create_submission(aid, "A班", {})
    repo.enqueue(sid, Stage.prepare, "cpu", camera="cam1")
    at = _open_confirm(ok(app()), aid)
    text = screen_text(at)
    assert "削除しますか" in text
    for word in (*SCHOOL_WORDS, *FORBIDDEN):
        assert word not in text, word


def test_ball_orange_is_used_for_the_primary_action_only():
    """Orange is 「go」: only the primary button wears it, never the delete confirm."""
    css = theme.theme_css()
    ball = ("var(--pfb-ball", theme.TOKENS["ball"].lower(), theme.TOKENS["ball-2"].lower())
    rules = [(re.sub(r"/\*.*?\*/", "", sel, flags=re.S).strip(), body)
             for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css)]
    used = [sel for sel, body in rules if any(p in body.lower() for p in ball)]
    assert used
    for sel in used:
        assert re.search(r"stBaseButton-primary|stBaseLinkButton-primary|:root", sel), sel
    delete = [(sel, body) for sel, body in rules if "pfb_delete_" in sel or "pfb-delete" in sel]
    assert delete and not any(p in b.lower() for _, b in delete for p in ball)
    assert all("var(--pfb-ink)" in b for sel, b in delete if "pfb_delete_go_" in sel)


def test_a_work_waiting_for_deletion_offers_nothing_to_press(env):
    from test_wizard_ui_people_wait import confirmed_and_analysing, worker_alive
    settings, repo = env
    worker_alive(settings)
    aid = video_assignment(repo)
    sid = confirmed_and_analysing(settings, repo, aid)          # behavior queued
    repo.conn.execute("UPDATE jobs SET status='running' WHERE submission_id=?", (sid,))
    with pytest.raises(deletion.StillRunning):
        deletion.delete_work(repo, settings, aid, stop_analysis=True)
    assert deletion.is_pending(repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    assert deletion.RUNNING in [i.value for i in at.info]
    assert not [b for b in at.button if b.proto.type == "primary" and not b.disabled]
    home = ok(app())
    assert home.button(key=f"pfb_open_{aid}").disabled
    assert theme.L(deletion.RUNNING.rstrip("。")) in html(home)
