import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def repo(tmp_path, settings):
    conn = connect(tmp_path / "app.db")
    init_schema(conn)
    return Repo(conn, settings)


@pytest.fixture
def sid(repo):
    aid = repo.create_assignment("課題", Kind.video, "task", "rubric", "standard", False)
    return repo.create_submission(aid, "班A", {"k": "日本語"})


def make_run(sid, stage, upstream=(), camera=None, subject_id=None, output=None):
    return StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id,
        version=0, source="model", upstream_run_ids=list(upstream), prompt_hash="h",
        model_id="m", output=output, status="ok", error=None, warnings=[],
        duration_sec=1.0, created_at="")


def test_wal_and_foreign_keys(repo):
    assert repo.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert repo.conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_assignment_submission_subject_roundtrip(repo, sid):
    sub = repo.get_submission(sid)
    assert sub["meta"] == {"k": "日本語"} and sub["identity_confirmed"] is False
    repo.update_submission_meta(sid, {"x": 1})
    repo.set_identity_confirmed(sid, True)
    sub = repo.get_submission(sid)
    assert sub["meta"] == {"x": 1} and sub["identity_confirmed"] is True
    assert repo.get_assignment(sub["assignment_id"])["kind"] == "video"
    assert len(repo.list_assignments()) == 1
    subj = repo.add_subject(sid, "Pa", "田中", None, {"cam1": "左"})
    s = repo.list_subjects(sid)[0]
    assert s["initial_positions"] == {"cam1": "左"} and s["approval_status"] == "none"
    run = repo.save_stage_run(make_run(sid, Stage.behavior, subject_id=subj))
    repo.set_approval(subj, run.id)
    s = repo.list_subjects(sid)[0]
    assert s["approval_status"] == "approved" and s["approved_run_id"] == run.id and s["approved_at"]
    repo.set_approval(subj, None)
    assert repo.list_subjects(sid)[0]["approval_status"] == "none"


def test_stage_run_versions_increment(repo, sid):
    r1 = repo.save_stage_run(make_run(sid, Stage.behavior, output={"a": "日本語"}))
    r2 = repo.save_stage_run(make_run(sid, Stage.behavior))
    other = repo.save_stage_run(make_run(sid, Stage.behavior, camera="cam1"))
    assert (r1.version, r2.version, other.version) == (1, 2, 1)
    assert repo.latest_run(sid, Stage.behavior).id == r2.id
    assert repo.latest_run(sid, Stage.behavior, camera="cam1").id == other.id
    assert [r.version for r in repo.run_history(sid, Stage.behavior)] == [1, 2]
    assert repo.run_history(sid, Stage.behavior)[0].output == {"a": "日本語"}
    assert repo.latest_run(sid, Stage.pfb) is None


def test_is_stale_after_upstream_edit(repo, sid):
    b1 = repo.save_stage_run(make_run(sid, Stage.behavior))
    c = repo.save_stage_run(make_run(sid, Stage.contribution, upstream=[b1.id]))
    assert repo.is_stale(c) is False
    repo.save_stage_run(make_run(sid, Stage.behavior))
    assert repo.is_stale(c) is True


def test_claim_next_job_is_exclusive(repo, sid):
    jid = repo.enqueue(sid, Stage.behavior, "standard", camera="cam1")
    assert repo.has_active_jobs(sid) and repo.pending_model_keys() == {"standard"}
    assert repo.claim_next_job({"high"}) is None
    job = repo.claim_next_job({"standard"})
    assert job.id == jid and job.status == JobStatus.running and job.heartbeat_at
    assert repo.claim_next_job({"standard"}) is None
    repo.finish_job(jid, JobStatus.done)
    assert not repo.has_active_jobs(sid)


def test_claim_considers_waiting_gpu(repo, sid):
    jid = repo.enqueue(sid, Stage.behavior, "high")
    other = repo.enqueue(sid, Stage.behavior, "standard")
    assert repo.mark_waiting_gpu("high", "VRAM不足") == 1
    assert repo.get_job(jid).status == JobStatus.waiting_gpu
    assert repo.get_job(jid).error == "VRAM不足"
    assert repo.get_job(other).status == JobStatus.queued
    assert repo.pending_model_keys() == {"high", "standard"}
    job = repo.claim_next_job({"high"})
    assert job.id == jid and job.status == JobStatus.running and job.error is None


def test_finish_job_only_updates_running(repo, sid):
    jid = repo.enqueue(sid, Stage.behavior, "standard")
    assert repo.finish_job(jid, JobStatus.done) is False  # still queued
    assert repo.get_job(jid).status == JobStatus.queued
    repo.claim_next_job({"standard"})
    repo.conn.execute("UPDATE jobs SET heartbeat_at='2000-01-01T00:00:00+00:00'")
    assert repo.recover_stale_jobs(900) == 1
    assert repo.finish_job(jid, JobStatus.done) is False  # slow job lost its claim
    assert repo.get_job(jid).status == JobStatus.failed_retryable
    jid2 = repo.enqueue(sid, Stage.behavior, "standard")
    repo.claim_next_job({"standard"})
    assert repo.finish_job(jid2, JobStatus.done) is True
    assert repo.get_job(jid2).status == JobStatus.done


def test_has_active_job_matches_key(repo, sid):
    jid = repo.enqueue(sid, Stage.identity, "standard", camera="cam1")
    assert repo.has_active_job(sid, Stage.identity, camera="cam1")
    assert not repo.has_active_job(sid, Stage.identity, camera="cam2")
    assert not repo.has_active_job(sid, Stage.behavior, camera="cam1")
    repo.claim_next_job({"standard"})
    assert repo.has_active_job(sid, Stage.identity, camera="cam1")
    repo.finish_job(jid, JobStatus.done)
    assert not repo.has_active_job(sid, Stage.identity, camera="cam1")


def test_fail_pending_jobs_and_latest_audit(repo, sid):
    a = repo.enqueue(sid, Stage.behavior, "standard")
    b = repo.enqueue(sid, Stage.behavior, "standard")
    c = repo.enqueue(sid, Stage.behavior, "high")
    repo.mark_waiting_gpu("standard", "VRAM不足")
    running = repo.enqueue(sid, Stage.evidence, "standard")
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (running,))
    assert repo.fail_pending_jobs("standard", "モデルを読み込めませんでした: x") == 2
    assert {repo.get_job(i).status for i in (a, b)} == {JobStatus.failed_retryable}
    assert repo.get_job(a).error == "モデルを読み込めませんでした: x"
    assert repo.get_job(c).status == JobStatus.queued
    assert repo.get_job(running).status == JobStatus.running
    assert repo.latest_audit("model_load_failed", "standard") is None
    repo.audit("model_load_failed", "standard", {"error": "一回目"})
    repo.audit("model_load_failed", "standard", {"error": "二回目"})
    assert repo.latest_audit("model_load_failed", "standard")["detail"] == {"error": "二回目"}


def test_recover_stale_jobs(repo, sid):
    old = repo.enqueue(sid, Stage.behavior, "standard")
    fresh = repo.enqueue(sid, Stage.evidence, "standard")
    repo.claim_next_job({"standard"})
    repo.claim_next_job({"standard"})
    repo.conn.execute("UPDATE jobs SET heartbeat_at='2000-01-01T00:00:00+00:00' WHERE id=?", (old,))
    assert repo.recover_stale_jobs(900) == 1
    assert repo.get_job(old).status == JobStatus.failed_retryable
    assert repo.get_job(fresh).status == JobStatus.running
    repo.heartbeat(fresh)
    assert repo.recover_stale_jobs(900) == 0


def test_requeue_running(repo, sid):
    a = repo.enqueue(sid, Stage.behavior, "standard")
    b = repo.enqueue(sid, Stage.behavior, "high")
    repo.claim_next_job({"standard", "high"})
    repo.claim_next_job({"standard", "high"})
    repo.requeue_running("standard")
    assert repo.get_job(a).status == JobStatus.queued
    assert repo.get_job(b).status == JobStatus.running


def test_cache_roundtrip(repo):
    assert repo.cache_get("h1") is None
    repo.cache_put("h1", "m", "生の応答", 10, 5, 0.5, None)
    assert repo.cache_get("h1") == "生の応答"
    repo.cache_put("h2", "m", None, 10, 0, 0.5, "boom")
    assert repo.cache_get("h2") is None


def test_audit(repo):
    repo.audit("approve", "subject:1", {"名前": "田中"})
    assert repo.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 1


def test_delete_assignment_removes_rows_and_files(repo, settings, sid):
    aid = repo.get_submission(sid)["assignment_id"]
    subj = repo.add_subject(sid, None, "x", None, {})
    repo.save_stage_run(make_run(sid, Stage.behavior, subject_id=subj))
    job = repo.enqueue(sid, Stage.behavior, "standard")
    repo.claim_next_job({"standard"})
    repo.finish_job(job, JobStatus.done)  # deleting is refused while jobs are active
    fdir = files.frames_dir(settings, aid, sid, "cam1")
    (fdir / "f.jpg").write_bytes(b"x")
    files.pages_dir(settings, aid, sid)
    other = repo.create_assignment("他", Kind.document, "t", "r", "standard", True)
    odir = files.assignment_dir(settings, other)
    repo.delete_assignment(aid)
    assert not files.assignment_dir(settings, aid, create=False).exists()
    assert odir.exists()
    for table in ("submissions", "subjects", "stage_runs", "jobs"):
        assert repo.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0
    assert [a["id"] for a in repo.list_assignments()] == [other]


def test_file_layout(settings):
    p = files.submission_dir(settings, 3, 7)
    assert p == settings.data_dir / "assignments" / "3" / "submissions" / "7" and p.is_dir()
    assert files.frames_dir(settings, 3, 7, "cam1") == p / "frames" / "cam1"
    assert files.pages_dir(settings, 3, 7) == p / "pages"


def test_list_cameras_distinct_sorted_prepare_only(tmp_path):
    from pfb_app.domain.models import StageRun
    conn = connect(tmp_path / "c.db")
    init_schema(conn)
    repo = Repo(conn)
    aid = repo.create_assignment("t", Kind.video, "x", "y", "standard", False)
    sid = repo.create_submission(aid, "A", {})
    def mk(stage, cam):
        repo.save_stage_run(StageRun(
            id=None, submission_id=sid, stage=stage, camera=cam, subject_id=None, version=0,
            source="model", upstream_run_ids=[], prompt_hash=None, model_id=None, output=None,
            status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))
    for st, cam in [(Stage.prepare, "cam2"), (Stage.prepare, "cam1"), (Stage.prepare, "cam2"),
                    (Stage.behavior, "cam9"), (Stage.prepare, None)]:
        mk(st, cam)
    assert repo.list_cameras(sid) == ["cam1", "cam2"]


def test_list_jobs_newest_first_and_retry_failed_job(repo):
    aid = repo.create_assignment("t", Kind.document, "x", "", "standard", False)
    sid = repo.create_submission(aid, "s", {})
    first = repo.enqueue(sid, Stage.prepare, "cpu")
    repo.claim_next_job({"cpu"})
    repo.finish_job(first, JobStatus.failed_retryable, "boom")
    assert [j.id for j in repo.list_jobs()] == [first]
    assert repo.retry_job(first) is True
    job = repo.get_job(first)
    assert job.status == JobStatus.queued and job.error is None
    assert repo.retry_job(first) is False  # not failed any more


def test_retry_job_refuses_when_same_stage_is_active(repo):
    aid = repo.create_assignment("t", Kind.document, "x", "", "standard", False)
    sid = repo.create_submission(aid, "s", {})
    old = repo.enqueue(sid, Stage.prepare, "cpu")
    repo.claim_next_job({"cpu"})
    repo.finish_job(old, JobStatus.failed, "bad")
    repo.enqueue(sid, Stage.prepare, "cpu")
    assert repo.retry_job(old) is False
    assert repo.get_job(old).status == JobStatus.failed


def test_rename_subject(repo):
    aid = repo.create_assignment("t", Kind.document, "x", "", "standard", False)
    sid = repo.create_submission(aid, "s", {})
    sub = repo.add_subject(sid, None, "旧", None, {})
    repo.rename_subject(sub, "新")
    assert repo.get_subject(sub)["display_name"] == "新"


def test_delete_assignment_refused_while_jobs_active(repo):
    from pfb_app.storage.db import ActiveJobsError
    aid = repo.create_assignment("t", Kind.document, "x", "", "standard", False)
    sid = repo.create_submission(aid, "s", {})
    job = repo.enqueue(sid, Stage.prepare, "cpu")
    with pytest.raises(ActiveJobsError):
        repo.delete_assignment(aid)
    assert repo.get_assignment(aid)
    repo.claim_next_job({"cpu"})
    repo.finish_job(job, JobStatus.done)
    repo.delete_assignment(aid)
    assert repo.list_assignments() == []


def test_migration_adds_domain_sport_roster_and_is_idempotent(tmp_path, settings):
    from pfb_app.storage import db as dbm
    conn = connect(tmp_path / "old.db")
    conn.executescript(dbm.SCHEMA)   # the first-release schema: no domain/sport/roster
    conn.execute("INSERT INTO assignments (title, kind, task_text, rubric_text, model_key,"
                 " require_identity_approval, created_at) VALUES ('t','video','','','m',0,'x')")
    init_schema(conn)
    init_schema(conn)
    r = Repo(conn, settings)
    cols = {x[1] for x in conn.execute("PRAGMA table_info(assignments)")}
    assert {"domain", "sport"} <= cols
    a = r.list_assignments()[0]
    assert a["domain"] == "school" and a["sport"] is None
    sid = r.create_submission(1, "班", {})
    r.add_subject(sid, "P1", "A", None, {})
    assert r.list_subjects(sid)[0]["roster"] == {}


def test_update_assignment_check(repo):
    aid = repo.create_assignment("t", Kind.video, "a", "b", "standard", False)
    assert repo.get_assignment(aid)["domain"] == "school"
    repo.update_assignment_check(aid, title="試合", task_text="x", rubric_text="y",
                                 domain="sports", sport="soccer")
    a = repo.get_assignment(aid)
    assert (a["title"], a["task_text"], a["rubric_text"], a["domain"], a["sport"]) == (
        "試合", "x", "y", "sports", "soccer")
    bid = repo.create_assignment("u", Kind.video, "", "", "standard", False,
                                 domain="sports", sport="rugby")
    assert repo.get_assignment(bid)["sport"] == "rugby"


def test_delete_subjects_by_ids(repo, sid):
    for p in ("P1", "P2", "P3"):
        repo.add_subject(sid, p, p, None, {}, roster={"名前": p})
    assert repo.list_subjects(sid)[0]["roster"] == {"名前": "P1"}
    assert repo.delete_subjects(sid, ["P1", "P3"]) == 2
    assert [s["participant_id"] for s in repo.list_subjects(sid)] == ["P2"]
    assert repo.delete_subjects(sid, []) == 0
    assert repo.delete_subjects(sid) == 1


def test_stage_order():
    stages = list(Stage)
    assert stages[0] is Stage.detect
    assert stages[stages.index(Stage.track) + 1] is Stage.names


# --- joined transactions (Repo._tx(join=True)) ----------------------------------------------

def _titles(repo):
    return sorted(a["title"] for a in repo.list_assignments())


def _add(repo, title):
    repo.create_assignment(title, Kind.video, "t", "", "standard", True)


def test_nested_tx_without_join_raises(repo):
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        with repo._tx():
            _add(repo, "外")
            with repo._tx():
                _add(repo, "内")
    assert _titles(repo) == []                   # the outer transaction rolled back too
    assert not repo.conn.in_transaction


def test_a_joined_block_error_rolls_back_only_its_own_writes(repo):
    with repo._tx():
        _add(repo, "外")
        try:
            with repo._tx(join=True):
                _add(repo, "内")
                raise ValueError("inner")
        except ValueError:
            pass                                 # the caller decides: it goes on
        _add(repo, "外2")
    assert _titles(repo) == ["外", "外2"]
    assert not repo.conn.in_transaction


def test_an_outer_error_rolls_back_the_joined_writes_too(repo):
    with pytest.raises(RuntimeError):
        with repo._tx():
            _add(repo, "外")
            with repo._tx(join=True):
                _add(repo, "内")
            raise RuntimeError("outer")
    assert _titles(repo) == []


def test_join_at_the_top_is_a_plain_transaction(repo):
    with repo._tx(join=True):
        _add(repo, "単独")
    assert _titles(repo) == ["単独"] and not repo.conn.in_transaction


def test_a_failed_commit_leaves_no_open_transaction(repo, monkeypatch):
    import sqlite3

    class Conn:                                  # COMMIT fails once (e.g. the disk is full)
        def __init__(self, real):
            self.real = real

        def execute(self, sql, *args):
            if sql == "COMMIT":
                raise sqlite3.OperationalError("disk I/O error")
            return self.real.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(self.real, name)

    real = repo.conn
    repo.conn = Conn(real)
    with pytest.raises(sqlite3.OperationalError):
        with repo._tx():
            _add(repo, "失敗")
    repo.conn = real
    assert not real.in_transaction
    assert _titles(repo) == []


def test_jobs_progress_migration_is_guarded(tmp_path, settings):
    from pfb_app.storage import db as dbm
    conn = connect(tmp_path / "old.db")
    conn.executescript(dbm.SCHEMA)   # the first-release schema: no progress columns
    init_schema(conn)
    init_schema(conn)
    cols = {x[1] for x in conn.execute("PRAGMA table_info(jobs)")}
    assert {"progress", "progress_note"} <= cols
    r = Repo(conn, settings)
    aid = r.create_assignment("課題", Kind.video, "task", "rubric", "standard", False)
    sid = r.create_submission(aid, "班A", {})
    jid = r.enqueue(sid, Stage.behavior, "cpu", camera="cam1")
    job = r.get_job(jid)
    assert (job.progress, job.progress_note) == (None, None)
    r.set_job_progress(jid, 0.5, "1 / 2 区間")
    job = r.get_job(jid)
    assert (job.progress, job.progress_note) == (0.5, "1 / 2 区間")
    r.set_job_progress(jid, None, None)
    assert r.get_job(jid).progress is None


def _timed_run(repo, sid, stage, seconds, camera="cam1"):
    run = make_run(sid, stage, camera=camera, output={"x": 1})
    repo.save_stage_run(run.model_copy(update={"duration_sec": float(seconds)}))


def test_stage_durations_all_is_one_query_and_feeds_stage_durations(repo):
    from pfb_app.storage.db import clear_stage_durations_cache
    clear_stage_durations_cache()
    aid = repo.create_assignment("課題", Kind.video, "task", "rubric", "standard", False)
    sid = repo.create_submission(aid, "班A", {"engine": "pfb3x3"})
    for seconds in (10, 20, 30):
        _timed_run(repo, sid, Stage.track, seconds)
    _timed_run(repo, sid, Stage.behavior, 100)
    _timed_run(repo, sid, Stage.prepare, 0.5)   # no real work: left out
    paper = repo.create_submission(aid, "班B", {})
    _timed_run(repo, paper, Stage.track, 999)     # another method
    statements = []
    repo.conn.set_trace_callback(statements.append)
    try:
        assert repo.stage_durations_all("pfb3x3") == {"track": [30, 20, 10],
                                                      "behavior": [100]}
    finally:
        repo.conn.set_trace_callback(None)
    assert len(statements) == 1
    assert repo.stage_durations_all("pfb3x3", limit=2) == {"track": [30, 20],
                                                           "behavior": [100]}
    clear_stage_durations_cache()
    statements.clear()
    repo.conn.set_trace_callback(statements.append)
    try:
        assert repo.stage_durations(Stage.track, "pfb3x3") == [30, 20, 10]
        assert repo.stage_durations(Stage.behavior, "pfb3x3") == [100]
        assert repo.stage_durations(Stage.pfb, "pfb3x3") == []
    finally:
        repo.conn.set_trace_callback(None)
    assert len(statements) == 1, statements
    assert repo.stage_durations(Stage.track, "paper") == [999]


def test_set_job_progress_never_lower_and_track_rates(repo):
    aid = repo.create_assignment("課題", Kind.video, "task", "rubric", "standard", False)
    sid = repo.create_submission(aid, "班A", {"engine": "pfb3x3"})
    jid = repo.enqueue(sid, Stage.track, "cpu", camera="cam1")
    repo.set_job_progress(jid, 0.5, "a", never_lower=True)
    repo.set_job_progress(jid, 0.2, "b", never_lower=True)
    repo.set_job_progress(jid, None, "c", never_lower=True)
    assert (repo.get_job(jid).progress, repo.get_job(jid).progress_note) == (0.5, "a")
    repo.set_job_progress(jid, 0.6, "d", never_lower=True)
    assert repo.get_job(jid).progress == 0.6
    for seconds, analysed in ((120, 60.0), (30, None), (90, 30.0)):
        run = make_run(sid, Stage.track, camera="cam1",
                       output={"analysed_s": analysed} if analysed else {"x": 1})
        repo.save_stage_run(run.model_copy(update={"duration_sec": float(seconds)}))
    assert repo.track_rates() == [3.0, 2.0]


def test_a_retry_starts_its_progress_again_but_a_requeue_keeps_it(repo):
    aid = repo.create_assignment("活動", Kind.video, "task", "rubric", "standard", False)
    sid = repo.create_submission(aid, "班A", {"engine": "pfb3x3"})
    jid = repo.enqueue(sid, Stage.behavior, "cpu", camera="cam1")
    repo.claim_next_job({"cpu"})
    repo.set_job_progress(jid, 0.5, "20 / 40 区間", never_lower=True)
    repo.requeue_running("cpu")                       # a yield: the bar keeps its place
    assert (repo.get_job(jid).progress, repo.get_job(jid).progress_note) == (0.5, "20 / 40 区間")
    repo.claim_next_job({"cpu"})
    repo.recover_stale_jobs(-1)                        # a stale recovery keeps it too
    assert repo.get_job(jid).progress == 0.5
    repo.conn.execute("UPDATE jobs SET status='failed' WHERE id=?", (jid,))
    assert repo.retry_job(jid)                         # a deliberate retry is a redo
    assert (repo.get_job(jid).progress, repo.get_job(jid).progress_note) == (None, None)
