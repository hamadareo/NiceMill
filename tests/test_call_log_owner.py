"""Model answers (call_log) are deleted with their work (audit: they used to survive it)."""
import pytest

from pfb_app.config import Settings
from pfb_app.domain.models import Stage
from pfb_app.pipeline import runner
from pfb_app.services import deletion
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from test_runner import make_ctx


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox")


@pytest.fixture
def repo(settings):
    conn = connect(files.db_path(settings))
    init_schema(conn)
    return Repo(conn, settings)


def work(repo, title):
    aid = repo.create_assignment(title, "video", "説明", "", "standard", False)
    return aid, repo.create_submission(aid, "グループ1", {"cameras": {}})


def put(repo, owner, h):
    repo.call_owner = owner
    try:
        repo.cache_put(h, "m", '{"ok": true}', 1, 1, 0.1, None)
    finally:
        repo.call_owner = None


def hashes(repo):
    return {r[0] for r in repo.conn.execute("SELECT request_hash FROM call_log")}


def test_deleting_a_work_deletes_its_answers_but_not_shared_or_untraced(settings, repo):
    a, sa = work(repo, "消す作業")
    b, sb = work(repo, "残る作業")
    put(repo, sa, "only-a")
    put(repo, sa, "shared")
    repo.call_owner = sb
    assert repo.cache_get("shared") == '{"ok": true}'   # a hit: b uses it too
    repo.call_owner = None
    put(repo, sb, "only-b")
    repo.cache_put("untraced", "m", "{}", 1, 1, 0.1, None)   # no job: no owner
    deletion.delete_work(repo, settings, a)
    assert hashes(repo) == {"shared", "only-b", "untraced"}
    assert repo.latest_audit("delete_work", f"assignment:{a}")["detail"]["calls_removed"] == 1
    deletion.delete_work(repo, settings, b)
    assert hashes(repo) == {"untraced"}
    assert deletion.untraced_calls(repo) == 1
    assert deletion.purge_untraced_calls(repo) == 1
    assert hashes(repo) == set()
    assert repo.latest_audit("purge_untraced_calls", None)["detail"] == {"removed": 1}


def test_a_hit_without_an_owner_records_nothing(repo):
    repo.cache_put("h", "m", "{}", 1, 1, 0.1, None)
    assert repo.cache_get("h") == "{}"
    assert repo.conn.execute("SELECT COUNT(*) FROM call_log_owner").fetchone()[0] == 0


def test_execute_job_names_the_group_while_it_runs(settings, repo, monkeypatch):
    _, sid = work(repo, "作業")
    jid = repo.enqueue(sid, Stage.pfb, "standard")   # the last stage: nothing follows
    seen = []

    def stage(job, ctx):
        seen.append(ctx.repo.call_owner)
        ctx.repo.cache_put("during", "m", "{}", 1, 1, 0.1, None)
        return []
    monkeypatch.setattr(runner, "_run_stage", stage)
    runner.execute_job(repo.claim_next_job({"standard"}), make_ctx(settings, repo))
    assert seen == [sid] and repo.call_owner is None
    assert repo.conn.execute("SELECT submission_id FROM call_log_owner WHERE request_hash="
                             "'during'").fetchone()[0] == sid
    assert jid


def test_many_answers_are_deleted_in_chunks(settings, repo):
    a, sa = work(repo, "大きな作業")
    for i in range(deletion._CHUNK * 2 + 7):
        put(repo, sa, f"h{i}")
    deletion.delete_work(repo, settings, a)
    assert hashes(repo) == set()
