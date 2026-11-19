"""Resumable (chunked) uploads for big phone videos (phone-first lane, audit #4).

The service stores chunks under data/uploads/.partial and moves a finished file into
data/uploads, where the drop zone's registration takes it like any upload. Tokens are signed
by the server (no login exists yet): only a token the app handed out may upload, and an upload
belongs to the token that started it."""
import json
import time

import pytest

from pfb_app.config import Settings
from pfb_app.services import chunked_upload as cu


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", inbox_dir=tmp_path / "inbox", max_upload_mb=1)


def token(settings, now=None):
    return cu.issue_token(settings, now=now)


# --- tokens -----------------------------------------------------------------------------------

def test_a_token_is_signed_and_expires(settings):
    t = token(settings)
    owner = cu.check_token(settings, t)
    assert owner and owner == cu.check_token(settings, t)
    assert cu.check_token(settings, t[:-2] + ("AA" if not t.endswith("AA") else "BB")) is None
    assert cu.check_token(settings, "") is None and cu.check_token(settings, "x.y.z") is None
    old = token(settings, now=time.time() - cu.TOKEN_TTL_SEC - 5)
    assert cu.check_token(settings, old) is None


def test_a_token_can_carry_its_signed_in_session(settings):
    t = cu.issue_token(settings, session_id=42)
    assert cu.check_token(settings, t) and cu.token_session(settings, t) == 42
    assert cu.token_session(settings, token(settings)) is None        # without a session
    v = t.split(".")
    forged = ".".join([v[0], v[1], "43", v[3]])                        # another session
    assert cu.check_token(settings, forged) is None and cu.token_session(settings, forged) is None
    same_owner = cu.issue_token(settings, session_id=7)
    assert cu.check_token(settings, same_owner) != cu.check_token(settings, t)


def test_the_secret_is_private_and_kept(settings):
    t = token(settings)
    path = cu.secret_path(settings)
    assert path.stat().st_mode & 0o777 == 0o600
    assert cu.check_token(settings, t) is not None        # read back, not made again


# --- create / append / finish -------------------------------------------------------------------

def test_an_upload_in_chunks_ends_in_data_uploads(settings):
    t = token(settings)
    up = cu.create(settings, t, "試合 後半.MOV", 10)
    assert up.offset == 0 and len(up.id) == 32
    assert cu.append(settings, t, up.id, 0, b"01234") == 5
    assert cu.status(settings, t, up.id).offset == 5
    assert cu.append(settings, t, up.id, 5, b"56789") == 10
    done = cu.status(settings, t, up.id)
    assert done.complete and done.path is not None
    assert done.path.parent == settings.data_dir / "uploads"
    assert done.path.read_bytes() == b"0123456789"
    assert done.path.suffix == ".mov" and "後半" in done.path.name and " " not in done.path.name
    assert not any((settings.data_dir / "uploads" / cu.PARTIAL).glob("*.part"))


def test_resume_needs_the_current_offset(settings):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 10)
    cu.append(settings, t, up.id, 0, b"0123")
    with pytest.raises(cu.OffsetMismatch) as e:
        cu.append(settings, t, up.id, 2, b"23456")     # a resent chunk: the client re-reads
    assert e.value.offset == 4
    assert cu.append(settings, t, up.id, 4, b"456789") == 10


def test_limits_types_and_size(settings):
    t = token(settings)
    with pytest.raises(cu.UploadRefused):
        cu.create(settings, t, "a.exe", 10)
    with pytest.raises(cu.UploadRefused):
        cu.create(settings, t, "a.mp4", 0)
    with pytest.raises(cu.UploadRefused):
        cu.create(settings, t, "a.mp4", 1024 * 1024 + 1)    # max_upload_mb = 1
    up = cu.create(settings, t, "a.mp4", 4)
    with pytest.raises(cu.UploadRefused):
        cu.append(settings, t, up.id, 0, b"12345")        # past the announced size
    with pytest.raises(cu.UploadRefused):
        cu.append(settings, t, up.id, 0, b"x" * (cu.CHUNK_MAX + 1))


GB = 1024 ** 3


def test_free_disk_space_keeps_the_refuse_level(settings, monkeypatch):
    t = token(settings)
    monkeypatch.setattr(cu, "free_bytes",
                        lambda path: int(cu.REFUSE_GB * GB) + cu.ADMIT_MARGIN_BYTES + 50)
    with pytest.raises(cu.UploadRefused) as e:
        cu.create(settings, t, "a.mp4", 100)               # would eat into the margin
    assert "空き" in str(e.value)
    cu.create(settings, t, "a.mp4", 40)


def test_open_uploads_reserve_their_remaining_bytes(settings, monkeypatch):
    """Two big uploads that each fit alone but not together: the second is refused."""
    big = Settings(data_dir=settings.data_dir, max_upload_mb=4096)
    t = token(big)
    monkeypatch.setattr(cu, "free_bytes", lambda path: int((cu.REFUSE_GB + 5) * GB))
    first = cu.create(big, t, "a.mp4", 3 * GB)
    with pytest.raises(cu.UploadRefused):
        cu.create(big, t, "b.mp4", 3 * GB)                 # 3 + 3 + margin > 5 GB above the level
    cu.append(big, t, first.id, 0, b"x" * 1024)            # sent bytes are no longer reserved
    cu.cancel(big, t, first.id)
    cu.create(big, t, "b.mp4", 3 * GB)                    # the first one freed its share


def test_append_rechecks_the_refuse_level(settings, monkeypatch):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 10)
    monkeypatch.setattr(cu, "free_bytes", lambda path: int(cu.REFUSE_GB * GB) - 1)
    with pytest.raises(cu.UploadRefused):
        cu.append(settings, t, up.id, 0, b"0123")


def test_ensure_room_matches_ops_signature(settings):
    class U:
        free = int(cu.REFUSE_GB * GB) + 10
    cu.ensure_room(settings, 10, usage=lambda path: U)
    with pytest.raises(cu.UploadRefused):
        cu.ensure_room(settings, 11, usage=lambda path: U)


def test_open_uploads_are_capped_per_owner_and_in_total(settings, monkeypatch):
    monkeypatch.setattr(cu, "MAX_OPEN_TOTAL", cu.MAX_OPEN_PER_OWNER + 2)
    a, b = token(settings), token(settings)
    for _ in range(cu.MAX_OPEN_PER_OWNER):
        cu.create(settings, a, "a.mp4", 4)
    with pytest.raises(cu.UploadRefused):
        cu.create(settings, a, "a.mp4", 4)                 # this token's own cap
    cu.create(settings, b, "b.mp4", 4)
    cu.create(settings, b, "b.mp4", 4)
    with pytest.raises(cu.UploadRefused):
        cu.create(settings, b, "b.mp4", 4)                 # the server's cap


def test_concurrent_creates_count_each_other(settings, monkeypatch):
    import threading
    big = Settings(data_dir=settings.data_dir, max_upload_mb=4096)
    monkeypatch.setattr(cu, "free_bytes", lambda path: int((cu.REFUSE_GB + 5) * GB))
    results = []

    def go():
        try:
            cu.create(big, token(big), "a.mp4", 3 * GB)
            results.append("ok")
        except cu.UploadRefused:
            results.append("refused")
    threads = [threading.Thread(target=go) for _ in range(6)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert sorted(results) == ["ok"] + ["refused"] * 5


def test_concurrent_first_tokens_share_one_whole_secret(settings):
    """The secret is made once: callers racing on a fresh data dir never read a half-written
    (empty) key, so every token they issue verifies (rel2: a flake of the test above)."""
    import threading
    keys, barrier = [], threading.Barrier(8)

    def go():
        barrier.wait()
        keys.append(cu._secret(settings))
    threads = [threading.Thread(target=go) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert len(keys) == 8 and len(set(keys)) == 1 and len(keys[0]) == 32
    assert cu.secret_path(settings).stat().st_mode & 0o777 == 0o600
    assert not list(cu.secret_path(settings).parent.glob("*.tmp"))


def test_unknown_ids_leave_no_lock_files(settings):
    t = token(settings)
    part = settings.data_dir / "uploads" / cu.PARTIAL
    for n in range(5):
        uid = f"{n:032x}"
        for call in (lambda: cu.cancel(settings, t, uid),
                     lambda: cu.append(settings, t, uid, 0, b"x"),
                     lambda: cu.claim(settings, t, uid)):
            with pytest.raises(cu.UnknownUpload):
                call()
    assert not list(part.glob("*.lock"))


def test_sweep_removes_stale_lock_files(settings):
    part = settings.data_dir / "uploads" / cu.PARTIAL
    part.mkdir(parents=True)
    (part / ("a" * 32 + ".lock")).touch()               # left by an older version
    assert cu.sweep(settings) == 0
    assert not list(part.glob("*.lock"))


def test_another_token_cannot_touch_it(settings):
    mine, theirs = token(settings), token(settings)
    up = cu.create(settings, mine, "a.mp4", 4)
    for call in (lambda: cu.status(settings, theirs, up.id),
                 lambda: cu.append(settings, theirs, up.id, 0, b"12"),
                 lambda: cu.cancel(settings, theirs, up.id),
                 lambda: cu.claim(settings, theirs, up.id)):
        with pytest.raises(cu.UnknownUpload):
            call()
    with pytest.raises(cu.BadToken):
        cu.create(settings, "forged.token.value", "a.mp4", 4)


def test_ids_are_never_paths(settings):
    t = token(settings)
    for bad in ("../../app.db", "a" * 31, "A" * 32, "../" + "0" * 29):
        with pytest.raises(cu.UnknownUpload):
            cu.status(settings, t, bad)


def test_claim_hands_over_a_finished_file_once(settings):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 2)
    with pytest.raises(cu.UploadRefused):
        cu.claim(settings, t, up.id)                      # not finished yet
    cu.append(settings, t, up.id, 0, b"ab")
    path = cu.claim(settings, t, up.id)
    assert path.read_bytes() == b"ab"
    with pytest.raises(cu.UnknownUpload):
        cu.claim(settings, t, up.id)


def test_cancel_removes_the_part(settings):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 4)
    cu.append(settings, t, up.id, 0, b"ab")
    cu.cancel(settings, t, up.id)
    left = (settings.data_dir / "uploads" / cu.PARTIAL).iterdir()
    assert not [p for p in left if not p.name.startswith(".")]   # .create.lock stays


def test_sweep_drops_old_parts_and_unclaimed_files(settings):
    t = token(settings)
    old = cu.create(settings, t, "old.mp4", 4)
    fresh = cu.create(settings, t, "new.mp4", 4)
    done = cu.create(settings, t, "done.mp4", 1)
    cu.append(settings, t, done.id, 0, b"x")
    finished = cu.status(settings, t, done.id).path
    later = time.time() + cu.KEEP_SEC + 10
    cu.touch(settings, fresh.id, later - 5)
    removed = cu.sweep(settings, now=later)
    assert removed == 2
    with pytest.raises(cu.UnknownUpload):
        cu.status(settings, t, old.id)
    assert cu.status(settings, t, fresh.id).offset == 0
    assert not finished.exists()


# --- review round 2 ----------------------------------------------------------------------------

def test_a_stalled_upload_stops_counting_and_is_swept(settings, monkeypatch):
    t = token(settings)
    stale = cu.create(settings, t, "a.mp4", 4)
    cu.touch(settings, stale.id, time.time() - cu.STALE_SEC - 5)
    for _ in range(cu.MAX_OPEN_PER_OWNER):                  # its place is free again
        cu.create(settings, t, "b.mp4", 4)
    assert cu.sweep(settings) == 1
    with pytest.raises(cu.UnknownUpload):
        cu.status(settings, t, stale.id)


def test_a_finished_file_is_named_by_its_upload(settings):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 2)
    cu.append(settings, t, up.id, 0, b"ab")
    assert cu.status(settings, t, up.id).path.name == f"{up.id}_a.mp4"


def test_done_is_written_before_the_rename(settings, monkeypatch):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 2)

    def crash(src, dst):
        raise OSError("power cut")
    monkeypatch.setattr(cu.os, "replace", crash)
    with pytest.raises(OSError):
        cu.append(settings, t, up.id, 0, b"ab")
    monkeypatch.undo()
    meta = json.loads((settings.data_dir / "uploads" / cu.PARTIAL / f"{up.id}.json").read_text())
    assert meta["done"].endswith(f"{up.id}_a.mp4")         # the sweep can find it
    later = time.time() + cu.KEEP_SEC + 10
    assert cu.sweep(settings, now=later) == 1
    assert not list((settings.data_dir / "uploads" / cu.PARTIAL).glob("*.part"))


def test_old_parts_without_an_upload_are_swept(settings):
    import os
    part = settings.data_dir / "uploads" / cu.PARTIAL
    part.mkdir(parents=True)
    orphan, young = part / ("b" * 32 + ".part"), part / ("c" * 32 + ".part")
    orphan.write_bytes(b"x")
    young.write_bytes(b"x")
    old = time.time() - cu.STALE_SEC - 10
    os.utime(orphan, (old, old))
    cu.sweep(settings)
    assert not orphan.exists() and young.exists()


def test_a_negative_offset_has_its_own_message(settings):
    t = token(settings)
    up = cu.create(settings, t, "a.mp4", 4)
    with pytest.raises(cu.UploadRefused) as e:
        cu.append(settings, t, up.id, -1, b"ab")
    assert str(e.value) == cu.NEGATIVE_OFFSET
