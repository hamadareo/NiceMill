"""判定支援 (line calls), the worker side: the `linecall` stage under its own GPU session key,
the coach's labels (DB + the engine's labels.json) and that a linecall job never holds a work
up. Fake engine only: no GPU, no image is opened."""
import ast
import json
from pathlib import Path

import pytest

from fakes.engine_env import alive, install_fake_engine
from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Kind, Stage
from pfb_app.engine import Pfb3x3Engine
from pfb_app.inference.gpu import GpuGuard, GpuSnapshot
from pfb_app.pipeline import runner
from pfb_app.pipeline.runner import LINECALL_KEY, execute_job, model_key_for
from pfb_app.pipeline.tracking import GpuUnavailable
from pfb_app.services import linecall, retry, review
from pfb_app.storage import files
from pfb_app.storage.db import NON_BLOCKING_STAGES, Repo, connect, init_schema
from pfb_app.ui.common import MODEL_LABELS, STAGE_LABELS
from pfb_app.ui.wizard import live
from pfb_app.ui.wizard.progress import work_progress
from pfb_app.ui.wizard.step_review import students_of
from pfb_app.worker import main as worker_main
from pfb_app.worker.session import EXIT_IDLE, EXIT_NEEDS_GPU, run_session
from test_pfb3x3_runner import make_ctx, make_video, save
from test_shared_server import Engine as SharedEngine
from test_shared_server import Clock, ready, spawner
from test_wizard_progress import group, track, tracked_to_review
from test_wizard_progress import save as wp_save
from pfb_app.worker.shared_server import SharedServer

TOTAL_GB = 97.9
STEPPED, CLEAR, OPEN = "A6-sideline_L-120", "P3-baseline-900", "B1-arc-2400"


@pytest.fixture
def fake(tmp_path, monkeypatch):
    return install_fake_engine(tmp_path, monkeypatch)


@pytest.fixture
def settings(tmp_path, fake):
    return Settings(data_dir=tmp_path / "data",
                    prompts_dir=Path(__file__).resolve().parents[1] / "prompts",
                    engine_dir=fake.engine_dir, engine_python=fake.engine_python,
                    gpu_poll_sec=7, session_idle_sec=0)


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


@pytest.fixture(autouse=True)
def pose(settings, monkeypatch):
    """The line-call pose weights in place (stand-in bytes, pinned)."""
    from test_pose_weights import pose_weights
    return pose_weights(settings.data_dir, monkeypatch)


class Guard(GpuGuard):
    def __init__(self, settings, free_gb=90.0):
        self.free_gb = free_gb
        self.can_start_calls: list[float] = []
        super().__init__(settings, query=lambda: GpuSnapshot(TOTAL_GB, TOTAL_GB - self.free_gb))

    def can_start(self, budget_gb):
        self.can_start_calls.append(budget_gb)
        return super().can_start(budget_gb)

    def should_yield(self):
        return False


def sports_work(repo, tmp_path, cameras=("cam1",)):
    aid = repo.create_assignment("練習試合", Kind.video, "3x3の試合", "", "standard", False,
                                 domain="sports", sport="basketball")
    paths = {c: str(make_video(tmp_path / f"{c}.mp4")) for c in cameras}
    sid = repo.create_submission(aid, "コートA", {"cameras": paths, "start_sec": 0.0,
                                                 "end_sec": None, "engine": "pfb3x3"})
    return aid, sid


def tracked(settings, repo, tmp_path, cam="cam1"):
    """A sports group whose camera went through the fake engine's track."""
    aid, sid = sports_work(repo, tmp_path, cameras=(cam,))
    save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
    jid = repo.enqueue(sid, Stage.track, "track", camera=cam)
    execute_job(repo.claim_next_job({"track"}), make_ctx(settings, repo))
    assert repo.get_job(jid).status == JobStatus.done
    return aid, sid


def run_linecall_job(settings, repo, sid, cam="cam1"):
    jid = linecall.request(repo, sid, cam)
    assert jid is not None
    job = repo.claim_next_job({LINECALL_KEY})
    assert job.id == jid
    execute_job(job, make_ctx(settings, repo))
    return repo.get_job(jid)


def labels_json(settings, repo, sid, cam="cam1"):
    path = linecall.episodes_path(settings, repo, sid, cam).with_name("labels.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# --- the stage ---

def test_model_key_and_dispatch(settings, repo, tmp_path, fake):
    _, sid = tracked(settings, repo, tmp_path)
    assert model_key_for(repo, sid, Stage.linecall, "cam1") == "linecall" == LINECALL_KEY
    assert linecall.job_state(repo, sid, "cam1") == "none"
    jid = linecall.request(repo, sid, "cam1")
    assert repo.get_job(jid).model_key == "linecall"
    assert linecall.job_state(repo, sid, "cam1") == "active"
    assert linecall.request(repo, sid, "cam1") is None          # one at a time per camera
    job = repo.claim_next_job({LINECALL_KEY})
    execute_job(job, make_ctx(settings, repo))
    assert repo.get_job(jid).status == JobStatus.done
    assert linecall.job_state(repo, sid, "cam1") == "done"
    run = repo.latest_run(sid, Stage.linecall, "cam1")
    track_run = repo.latest_run(sid, Stage.track, "cam1")
    assert run.status == "ok" and run.upstream_run_ids == [track_run.id]
    assert run.output["episodes"] == 4
    assert run.output["verdicts"] == {"stepped": 1, "clear": 1, "undecidable": 2}
    assert Path(run.output["episodes_json"]) == linecall.episodes_path(settings, repo, sid, "cam1")
    assert run.model_id.startswith("pfb3x3:linecall")
    call = [c for c in fake.calls() if c["argv"][0] == "linecall"][-1]["argv"]
    assert call[call.index("--video-id") + 1] == f"s{sid}_cam1"
    assert call[call.index("--device") + 1] == "cuda"
    # nothing follows it, and no server was started for it
    assert repo.conn.execute("SELECT COUNT(*) FROM jobs WHERE id > ?", (jid,)).fetchone()[0] == 0
    assert "serve" not in fake.commands()


def test_linecall_needs_a_tracked_camera(settings, repo, tmp_path, fake):
    _, sid = sports_work(repo, tmp_path)
    job = run_linecall_job(settings, repo, sid)
    assert job.status == JobStatus.failed
    assert linecall.job_state(repo, sid, "cam1") == "failed"
    assert "linecall" not in fake.commands()
    with pytest.raises(ValueError):
        linecall.request(repo, sid, "cam9")                   # not a camera of the group


def test_linecall_without_the_pose_weights_is_not_ready(settings, repo, tmp_path, fake, pose):
    """Never a download: without the pinned weights the engine is not called at all."""
    from pfb_app.pipeline.linecall import NOT_READY
    _, sid = tracked(settings, repo, tmp_path)
    pose.write_bytes(b"not the pinned weights")
    job = run_linecall_job(settings, repo, sid)
    assert job.status == JobStatus.failed
    assert repo.latest_run(sid, Stage.linecall, "cam1").error == NOT_READY
    assert "linecall" not in fake.commands()
    pose.unlink()
    job = run_linecall_job(settings, repo, sid)
    assert job.status == JobStatus.failed and "linecall" not in fake.commands()

def test_linecall_needs_gpu_requeues(settings, repo, tmp_path, monkeypatch):
    _, sid = tracked(settings, repo, tmp_path)
    jid = linecall.request(repo, sid, "cam1")
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "3")
    job = repo.claim_next_job({LINECALL_KEY})
    with pytest.raises(GpuUnavailable) as err:
        execute_job(job, make_ctx(settings, repo))
    assert "GPU" in err.value.reason
    assert repo.get_job(jid).status == JobStatus.queued
    assert linecall.job_state(repo, sid, "cam1") == "active"
    # the session: no server, the engine on the GPU; exit 3 -> waiting_gpu, EXIT_NEEDS_GPU
    code = run_session(None, LINECALL_KEY, settings=settings, guard=Guard(settings),
                       engine_factory=lambda s: Pfb3x3Engine(s, tick_sec=0.05, term_grace_sec=2),
                       sleep=lambda s: None, heartbeat_sec=60)
    assert code == EXIT_NEEDS_GPU
    assert repo.get_job(jid).status == JobStatus.waiting_gpu
    monkeypatch.delenv("FAKE_ENGINE_EXIT")
    code = run_session(None, LINECALL_KEY, settings=settings, guard=Guard(settings),
                       engine_factory=lambda s: Pfb3x3Engine(s, tick_sec=0.05, term_grace_sec=2),
                       sleep=lambda s: None, heartbeat_sec=60)
    assert code == EXIT_IDLE and repo.get_job(jid).status == JobStatus.done


# --- labels ---

def test_label_upsert_and_mirror(settings, repo, tmp_path, fake):
    _, sid = tracked(settings, repo, tmp_path)
    assert run_linecall_job(settings, repo, sid).status == JobStatus.done
    linecall.save_label(settings, repo, sid, "cam1", STEPPED, "correct", "stepped")
    linecall.save_label(settings, repo, sid, "cam1", STEPPED, "wrong", "stepped")   # last press wins
    assert repo.linecall_labels(sid, "cam1") == {STEPPED: "wrong"}
    assert repo.conn.execute("SELECT COUNT(*) FROM linecall_labels").fetchone()[0] == 1
    assert labels_json(settings, repo, sid) == {STEPPED: {"label": "wrong", "verdict": "stepped"}}
    # 判定不能 stays in the DB only; correct / wrong carry the verdict they confirmed
    linecall.save_label(settings, repo, sid, "cam1", CLEAR, "correct", "clear")
    linecall.save_label(settings, repo, sid, "cam1", OPEN, "undecidable", "undecidable")
    assert repo.linecall_labels(sid, "cam1") == {STEPPED: "wrong", CLEAR: "correct",
                                                 OPEN: "undecidable"}
    records = repo.linecall_label_records(sid, "cam1")
    assert records[OPEN] == {"label": "undecidable", "verdict": "undecidable"}
    mirrored = {STEPPED: {"label": "wrong", "verdict": "stepped"},
                CLEAR: {"label": "correct", "verdict": "clear"}}
    assert labels_json(settings, repo, sid) == mirrored
    linecall.save_label(settings, repo, sid, "cam1", CLEAR, "undecidable", "clear")   # changed again
    assert labels_json(settings, repo, sid) == {STEPPED: mirrored[STEPPED]}
    linecall.save_label(settings, repo, sid, "cam1", CLEAR, "correct", "clear")
    # a rerun of linecall (same episode ids) keeps both the rows and the file
    assert run_linecall_job(settings, repo, sid).status == JobStatus.done
    assert repo.linecall_labels(sid, "cam1") == {STEPPED: "wrong", CLEAR: "correct",
                                                 OPEN: "undecidable"}
    assert labels_json(settings, repo, sid) == mirrored
    lc = linecall.episodes_path(settings, repo, sid, "cam1").parent
    assert not list(lc.glob("labels.json.tmp*"))
    # bad input never reaches the DB or the file
    with pytest.raises(ValueError):
        linecall.save_label(settings, repo, sid, "cam1", STEPPED, "maybe", "stepped")
    with pytest.raises(ValueError):
        linecall.save_label(settings, repo, sid, "cam1", "X9-arc-1", "correct", "stepped")
    assert len(repo.linecall_labels(sid, "cam1")) == 3
    # labels are per camera and go with the group
    assert repo.linecall_labels(sid, "cam2") == {}
    repo.conn.execute("DELETE FROM submissions WHERE id=?", (sid,))
    assert repo.conn.execute("SELECT COUNT(*) FROM linecall_labels").fetchone()[0] == 0


def test_label_for_a_verdict_no_longer_shown_is_refused(settings, repo, tmp_path, fake):
    _, sid = tracked(settings, repo, tmp_path)
    assert run_linecall_job(settings, repo, sid).status == JobStatus.done
    linecall.save_label(settings, repo, sid, "cam1", STEPPED, "correct", "stepped")
    before = labels_json(settings, repo, sid)
    # a rerun flips the episode's verdict while the page still shows the old one
    assert run_linecall_job(settings, repo, sid).status == JobStatus.done
    path = linecall.episodes_path(settings, repo, sid, "cam1")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for ep in doc["episodes"]:
        if ep["id"] == STEPPED:
            ep["verdict"] = "clear"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        linecall.save_label(settings, repo, sid, "cam1", STEPPED, "wrong", "stepped")
    assert repo.linecall_label_records(sid, "cam1") == {
        STEPPED: {"label": "correct", "verdict": "stepped"}}
    assert labels_json(settings, repo, sid) == before
    # pressed again on the page that shows the new verdict
    linecall.save_label(settings, repo, sid, "cam1", STEPPED, "wrong", "clear")
    assert labels_json(settings, repo, sid) == {STEPPED: {"label": "wrong", "verdict": "clear"}}


def test_old_labels_table_gets_the_verdict_column(tmp_path):
    conn = connect(tmp_path / "old.db")
    conn.execute("CREATE TABLE linecall_labels (submission_id INTEGER NOT NULL, camera TEXT"
                 " NOT NULL, episode_id TEXT NOT NULL, label TEXT NOT NULL, labeled_at TEXT"
                 " NOT NULL, PRIMARY KEY (submission_id, camera, episode_id))")
    init_schema(conn)
    init_schema(conn)                                        # idempotent
    cols = {r[1] for r in conn.execute("PRAGMA table_info(linecall_labels)")}
    assert "verdict" in cols


def test_label_without_episodes_is_refused(settings, repo, tmp_path, fake):
    _, sid = tracked(settings, repo, tmp_path)
    with pytest.raises(ValueError):
        linecall.save_label(settings, repo, sid, "cam1", STEPPED, "correct", "stepped")
    assert repo.linecall_labels(sid, "cam1") == {}


# --- never blocking ---

def _view(repo, aid, sid):
    p = work_progress(repo, aid)
    sub = repo.get_submission(sid)
    return ((p.current, sorted(p.done), p.your_turn, p.stopped, p.reason,
             [(s.step, s.status, s.reason) for s in p.subs],
             p.analysis.percent if p.analysis else None),
            [s.state for s in students_of(repo, sub, Kind.video)],
            live.remaining_jobs(repo, aid, through=True), repo.has_blocking_jobs(sid),
            retry.open_failures(repo, sid))


@pytest.mark.parametrize("status", ["queued", "running", "waiting_gpu", "failed",
                                    "failed_retryable"])
def test_linecall_job_not_blocking(repo, status):
    aid = repo.create_assignment("練習試合", Kind.video, "3x3の試合", "", "standard", False,
                                 domain="sports", sport="basketball")
    sid, subjects = group(repo, aid)
    tracked_to_review(repo, sid, subjects)                  # its review step (step 5)
    before = _view(repo, aid, sid)
    assert before[0][0] == 5 and before[3] is False
    jid = repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status=?, error=? WHERE id=?",
                      (status, "EngineError: x" if status.startswith("failed") else None, jid))
    assert _view(repo, aid, sid) == before
    first = repo.get_subject(next(iter(subjects.values())))
    assert review._being_written(repo, first) is None
    assert Stage.linecall.value in NON_BLOCKING_STAGES


def test_failed_linecall_runs_and_jobs_leave_the_work_alone(repo):
    aid = repo.create_assignment("練習試合", Kind.video, "3x3の試合", "", "standard", False,
                                 domain="sports", sport="basketball")
    sid, subjects = group(repo, aid)
    tracked_to_review(repo, sid, subjects)
    before = _view(repo, aid, sid)
    wp_save(repo, sid, Stage.linecall, None, camera="cam1", status="failed")
    jid = repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='failed', error='判定材料を作れません' WHERE id=?",
                      (jid,))
    assert _view(repo, aid, sid) == before
    assert linecall.job_state(repo, sid, "cam1") == "failed"
    wp_save(repo, sid, Stage.linecall, {"episodes": 0}, camera="cam2", status="failed")
    assert linecall.job_state(repo, sid, "cam2") == "none"   # a failed run alone: nothing


def test_linecall_engine_error_is_a_retryable_failure(settings, repo, tmp_path, monkeypatch):
    _, sid = tracked(settings, repo, tmp_path)
    monkeypatch.setenv("FAKE_ENGINE_EXIT", "1")
    job = run_linecall_job(settings, repo, sid)
    assert job.status == JobStatus.failed_retryable
    assert linecall.job_state(repo, sid, "cam1") == "failed"
    assert repo.latest_run(sid, Stage.linecall, "cam1") is None
    assert linecall.request(repo, sid, "cam1") is not None   # pressed again: queued anew


def test_waiting_linecall_job_does_not_block_a_roster(repo, tmp_path):
    from pfb_app.services.roster import apply_roster, parse_roster_text
    _, sid = sports_work(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='waiting_gpu' WHERE id=?", (jid,))
    assert apply_roster(repo, sid, parse_roster_text("名前\nx\n"), "sports") == 1


def test_linecall_job_leaves_step_4_alone(repo):
    aid = repo.create_assignment("練習試合", Kind.video, "3x3の試合", "", "standard", False,
                                 domain="sports", sport="basketball")
    sid, _ = group(repo, aid)
    track(repo, sid)
    review.save_tracked_identity(repo, sid, "cam1", {"P1": "Pa", "P2": "Pb", "P3": "Pc",
                                                     "P4": "Pd"})
    review.confirm_identity(repo, sid)                       # queues behavior: step 4
    before = work_progress(repo, aid)
    jid = repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    repo.conn.execute("UPDATE jobs SET status='running' WHERE id=?", (jid,))
    after = work_progress(repo, aid)
    pct = lambda p: p.analysis.percent if p.analysis else None   # noqa: E731
    assert (after.current, after.reason, pct(after)) == (before.current, before.reason,
                                                         pct(before))
    assert after.current == 4
    assert repo.has_blocking_jobs(sid)                       # behavior still does


def test_stage_and_key_labels():
    assert STAGE_LABELS[Stage.linecall] == "判定材料づくり"
    assert MODEL_LABELS[LINECALL_KEY] == "判定材料づくり"


def test_new_modules_have_no_banned_words():
    from pfb_app.pipeline import linecall as stage_mod
    from test_linecall_view import RULING_WORDS
    from test_neutral_wording import SCHOOL_WORDS
    for module in (stage_mod, linecall):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        strings = [n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        for s in strings + [STAGE_LABELS[Stage.linecall]]:
            found = [w for w in (*SCHOOL_WORDS, "ルーブリック", *RULING_WORDS) if w in s]
            assert not found, (module.__name__, s, found)


# --- the worker ---

def test_session_order_has_linecall(settings):
    order = worker_main._session_order(settings)
    assert order.count("linecall") == 1 and order.index("linecall") == order.index("track") + 1
    # the engine refuses to start below GPU_RESERVE_MIB 32768 + LINECALL_NEED_MIB 6144 free
    # (38912 MiB = 38.0 GiB): budget + start margin never asks for less than that
    need = (32768 + 6144) / 1024
    assert worker_main.LINECALL_ENGINE_FREE_GB == need
    assert settings.linecall_budget_gb == 31
    assert worker_main._budget_gb(settings, "linecall") == 31
    assert worker_main._budget_gb(settings, "linecall") + settings.start_margin_gb >= need
    no_margin = settings.model_copy(update={"start_margin_gb": 0})
    assert worker_main._budget_gb(no_margin, "linecall") == need + 1     # 1 GB above it
    small = settings.model_copy(update={"linecall_budget_gb": 8})
    assert worker_main._budget_gb(small, "linecall") == need + 1 - settings.start_margin_gb


def test_linecall_waits_until_the_engine_would_fit(settings, repo, tmp_path):
    _, sid = sports_work(repo, tmp_path)
    jid = repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    calls = []
    guard = Guard(settings, free_gb=38.5)                   # the engine wants 38.0 + a margin
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {})
    assert calls == [] and guard.can_start_calls == [31]
    assert repo.get_job(jid).status == JobStatus.waiting_gpu
    guard.free_gb = 39.0
    worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {})
    assert calls == [("linecall", None)] and repo.get_job(jid).status == JobStatus.done


def _shared(settings, guard):
    return SharedServer(settings, guard, SharedEngine(settings), clock=Clock())


def test_a_warm_shared_server_makes_room_for_linecall(settings, repo, tmp_path, fake):
    _, sid = sports_work(repo, tmp_path)
    repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    guard, calls = Guard(settings), []
    shared = _shared(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        # with the warm server's 34 GB, 30 GB are free: too little for linecall (39)
        guard._query = lambda: GpuSnapshot(
            TOTAL_GB, TOTAL_GB - (30.0 if shared.handle is not None else 64.0))
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [("linecall", None)] and not alive(handle.pid)
    finally:
        shared.close()


def test_a_warm_shared_server_stays_when_linecall_fits_beside_it(settings, repo, tmp_path,
                                                                  fake):
    _, sid = sports_work(repo, tmp_path)
    repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    guard, calls = Guard(settings, free_gb=60.0), []
    shared = _shared(settings, guard)
    try:
        shared.request()
        handle = ready(shared)
        worker_main._gpu_pass(repo, settings, guard, spawner(repo, calls), lambda s: None, {},
                              shared=shared)
        assert calls == [("linecall", None)] and alive(handle.pid)
    finally:
        shared.close()


def test_a_crashed_linecall_session_has_its_engine_command_stopped(settings, repo, tmp_path):
    _, sid = sports_work(repo, tmp_path)
    repo.enqueue(sid, Stage.linecall, LINECALL_KEY, camera="cam1")
    cleaned = []
    worker_main._gpu_pass(repo, settings, Guard(settings), lambda key: 1, lambda s: None, {},
                          engine_cleanup=cleaned.append)
    assert cleaned == ["linecall"]
