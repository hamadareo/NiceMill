"""SQLite storage: schema, versioned stage runs, stale detection and the job queue."""
import itertools
import json
import logging
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.domain import profiles
from pfb_app.domain.models import Job, JobStatus, Kind, Stage, StageRun
from pfb_app.storage import files

SCHEMA = """
CREATE TABLE IF NOT EXISTS assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    task_text TEXT NOT NULL,
    rubric_text TEXT NOT NULL,
    model_key TEXT NOT NULL,
    require_identity_approval INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS submissions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL REFERENCES assignments(id) ON DELETE CASCADE,
    label TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    identity_confirmed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subjects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    participant_id TEXT,
    display_name TEXT NOT NULL,
    appearance TEXT,
    initial_positions TEXT NOT NULL DEFAULT '{}',
    approval_status TEXT NOT NULL DEFAULT 'none' CHECK (approval_status IN ('none', 'approved')),
    approved_run_id INTEGER,
    approved_at TEXT
);
CREATE TABLE IF NOT EXISTS stage_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    camera TEXT,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    source TEXT NOT NULL,
    upstream_run_ids TEXT NOT NULL,
    prompt_hash TEXT,
    model_id TEXT,
    output TEXT,
    status TEXT NOT NULL,
    error TEXT,
    warnings TEXT NOT NULL,
    duration_sec REAL NOT NULL,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_stage_runs_key_version
    ON stage_runs (submission_id, stage, COALESCE(camera, ''), COALESCE(subject_id, 0), version);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    stage TEXT NOT NULL,
    camera TEXT,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    model_key TEXT NOT NULL,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    heartbeat_at TEXT,
    error TEXT,  -- waiting reason while status is waiting_gpu
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_status ON jobs (status, model_key);
CREATE TABLE IF NOT EXISTS call_log (
    request_hash TEXT PRIMARY KEY,
    model_id TEXT,
    raw TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    duration REAL,
    error TEXT,
    created_at TEXT NOT NULL
);
-- which groups a call_log row was made or reused for (ops lane, 2026-10-23): deleting a
-- work deletes the cache rows no remaining group uses (services.deletion). Rows written
-- before this table existed have no owner and stay (see deletion.py).
CREATE TABLE IF NOT EXISTS call_log_owner (
    request_hash TEXT NOT NULL,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    PRIMARY KEY (request_hash, submission_id)
);
CREATE INDEX IF NOT EXISTS ix_call_log_owner_sub ON call_log_owner (submission_id);
CREATE TABLE IF NOT EXISTS linecall_labels (
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    camera TEXT NOT NULL,
    episode_id TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('correct', 'wrong', 'undecidable')),
    verdict TEXT,  -- the episode's verdict when it was labelled (the engine skips other ones)
    labeled_at TEXT NOT NULL,
    PRIMARY KEY (submission_id, camera, episode_id)
);
-- 成長の記録 (growth lane, 2026-10-23): one real person across several works. A record
-- links subjects of different works; links are made by the user only (a suggestion is
-- accepted, or a person is linked by hand), never by themselves. A declined suggestion is a
-- pair of subjects (subject_a < subject_b) that is never offered again.
CREATE TABLE IF NOT EXISTS growth_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS growth_links (
    subject_id INTEGER PRIMARY KEY REFERENCES subjects(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES growth_people(id) ON DELETE CASCADE,
    how TEXT NOT NULL CHECK (how IN ('manual', 'suggested')),
    linked_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_growth_links_person ON growth_links (person_id);
CREATE TABLE IF NOT EXISTS growth_declined (
    subject_a INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    subject_b INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    declined_at TEXT NOT NULL,
    PRIMARY KEY (subject_a, subject_b)
);
-- 「データの扱い」 (data lane, 2026-10-23): app-wide settings as JSON values (the key
-- "retention": services.retention)
CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
-- 「あとから訂正」の作業 (services.corrections): a text's corrections, appended only
CREATE TABLE IF NOT EXISTS text_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    kind TEXT NOT NULL,
    run_id INTEGER,
    prev_run_id INTEGER,
    text TEXT,
    prev_text TEXT,
    reason TEXT,
    note TEXT,
    actor TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT,
    detail TEXT NOT NULL
);
-- accounts (2026-10-24, services.auth): one organisation per install for now; every new
-- table and the roots of the data carry org_id (all 1), so tenants need no data migration
CREATE TABLE IF NOT EXISTS orgs (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id INTEGER NOT NULL DEFAULT 1,
    login TEXT NOT NULL COLLATE NOCASE UNIQUE,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('admin', 'evaluator', 'viewer')),
    password_hash TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    password_changed_at TEXT NOT NULL,
    last_login_at TEXT
);
-- a signed-in browser: the cookie holds the id and a random part whose SHA-256 is kept here
CREATE TABLE IF NOT EXISTS auth_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id INTEGER NOT NULL DEFAULT 1,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    secret_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_auth_sessions_user ON auth_sessions (user_id);
-- one-time codes: an invitation (user_id NULL: a new account with `role`) or a password
-- reset of user_id; only the code's SHA-256 is kept
CREATE TABLE IF NOT EXISTS invites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_id INTEGER NOT NULL DEFAULT 1,
    code_hash TEXT NOT NULL UNIQUE,
    role TEXT NOT NULL CHECK (role IN ('admin', 'evaluator', 'viewer')),
    display_name TEXT NOT NULL DEFAULT '',
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    created_by INTEGER REFERENCES users(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    used_at TEXT,
    cancelled_at TEXT
);
-- failed sign-ins and code tries per key (login:<name>, ip:<address>, code:..., setup:*)
CREATE TABLE IF NOT EXISTS login_failures (
    key TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_login_failures ON login_failures (key, ts);
-- 解説動画 (explainer lane, 2026-10-24, services.explainer): the worker's video jobs, kept apart
-- from `jobs` so they never show in a work's steps, hold up an approval or an erasure, or move
-- a work's last activity. subject_id NULL: the group's 「追跡の様子」.
CREATE TABLE IF NOT EXISTS explainer_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    submission_id INTEGER NOT NULL REFERENCES submissions(id) ON DELETE CASCADE,
    subject_id INTEGER REFERENCES subjects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind IN ('person', 'tracking')),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed', 'cancelled')),
    stamp TEXT NOT NULL,
    session_id INTEGER,
    actor TEXT,
    progress REAL,
    note TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    heartbeat_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_explainer_jobs_key ON explainer_jobs (submission_id, kind, subject_id);
-- 自動承認 (2026-11-19, services.autoapprove): one row per text the app approved by its machine
-- checks without a person's look: when, under which policy (its version and thresholds), and the
-- result of every check. status: active (the approval stands), revoked (put back to waiting for
-- a person), confirmed (a person approved afterwards), replaced (a newer text was approved automatically). Added only: no existing table changes.
CREATE TABLE IF NOT EXISTS auto_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES stage_runs(id) ON DELETE CASCADE,
    approved_at TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy TEXT NOT NULL,
    checks TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked', 'confirmed', 'replaced')),
    ended_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_auto_approvals_subject ON auto_approvals (subject_id, run_id);
-- 動作の再確認 (2026-11-19, pipeline.claim_scan): the machine re-check of the action words of one recorded
-- scene (a behavior run's event): the verdict of pipeline.claim_check (shoot / dribble / pass / defend,
-- from a pose cue and two questions to a local vision model) or of pipeline.motion_check (run / walk /
-- move, from the tracks). One row per scene and method; the newest row counts. Added only.
CREATE TABLE IF NOT EXISTS claim_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES stage_runs(id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    method TEXT NOT NULL,
    verdict TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_claim_checks_run ON claim_checks (run_id, event_id);
"""

_ACTIVE = (JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value)
_CLAIMABLE = (JobStatus.queued.value, JobStatus.waiting_gpu.value)
# stage -> the stages of the same group it builds on (Repo.claim_next_job after_upstream)
UPSTREAM_STAGES = {Stage.contribution.value: (Stage.behavior.value,),
                   Stage.pfb.value: (Stage.behavior.value, Stage.contribution.value)}
_SAVEPOINTS = itertools.count(1)   # names of joined transactions (Repo._tx)
_log = logging.getLogger(__name__)
OWNER_TRIES = 3   # call_log_owner inserts (Repo._own_call)
ERASING_STALE_SEC = 3600   # an erasure's "erasing" stamp older than this was left by a stop
# stages outside the work's analysis (判定支援, やさしい文章版, チームのヒント): their jobs are left out of submission_job_rows,
# so they never show in the work's stepper, progress, remaining time or failures
SIDE_STAGES = (Stage.linecall.value, Stage.plain.value, Stage.tips.value)
# stages whose jobs never hold up a coach action (has_blocking_jobs)
NON_BLOCKING_STAGES = (Stage.names.value, *SIDE_STAGES)
LINECALL_LABELS = ("correct", "wrong", "undecidable")


SERVED_TIMED = ("contribution", "pfb")   # stage_durations_all prefers their served runs
# 成長の記録: subject `s` has an approved text now (the approval is its newest feedback run;
# staleness is review.subject_status's to judge) and is evaluated (a video's person has a
# participant id and is not excluded; `a` is its work)
_GROWTH_OK = ("s.approval_status = 'approved' AND s.approved_run_id = (SELECT r.id FROM"
              " stage_runs r WHERE r.submission_id = s.submission_id AND r.stage = 'pfb'"
              " AND r.subject_id = s.id ORDER BY r.version DESC LIMIT 1)")
# 「あとから訂正」 works: the newest feedback run is ok with a text, and the person's newest
# correction row is not an open report (services.corrections.usable_run, in one query)
_LIVE_OK = ("COALESCE((SELECT r.status = 'ok' AND TRIM(COALESCE(json_extract(r.output, '$.pfb'),"
            " '')) <> '' FROM stage_runs r WHERE r.submission_id = s.submission_id"
            " AND r.stage = 'pfb' AND r.subject_id = s.id ORDER BY r.version DESC LIMIT 1), 0)"
            " AND COALESCE((SELECT v.kind FROM text_versions v WHERE v.subject_id = s.id"
            " ORDER BY v.id DESC LIMIT 1), '') <> 'report'")
# the text the growth record uses: approved, or in a correct_later work the newest one
_GROWTH_USABLE = (f"(CASE WHEN a.review_mode = 'correct_later' THEN ({_LIVE_OK})"
                  f" ELSE ({_GROWTH_OK}) END)")
_GROWTH_EVALUATED = ("(a.kind <> 'video' OR (s.participant_id IS NOT NULL"
                     " AND COALESCE(s.excluded, 0) = 0))")


def _now(delta_sec: float = 0) -> str:
    t = datetime.now(timezone.utc) + timedelta(seconds=delta_sec)
    return t.isoformat(timespec="microseconds")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection in autocommit mode (transactions are explicit), WAL, FK on."""
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# Columns added after the first release: (table, column, declaration). init_schema adds each
# one only if it is missing, so it runs safely on the live DB and on every start.
MIGRATIONS = (
    ("assignments", "pdf_downloaded_at", "TEXT"),   # last student-PDF download (wizard)
    ("assignments", "domain", "TEXT DEFAULT 'school'"),
    ("assignments", "sport", "TEXT"),
    ("subjects", "roster", "TEXT NOT NULL DEFAULT '{}'"),
    ("subjects", "excluded", "INTEGER DEFAULT 0"),   # 1: not evaluated (この人は評価しない)
    ("jobs", "progress", "REAL"),          # a running job's own progress, 0..1 (NULL: unknown)
    ("jobs", "progress_note", "TEXT"),     # its plain short note, e.g. 「12 / 40 区間」
    ("assignments", "review_mode", "TEXT DEFAULT 'approve'"),   # approve | correct_later
    ("assignments", "auto_approve", "INTEGER DEFAULT 0"),   # 確認せずに自動で承認する (off)
    ("subjects", "approved_auto", "INTEGER DEFAULT 0"),     # 1: approved by the worker
    # the activity-type profile (domain/profiles.py); NULL (works from before it): read as
    # the domain's default, so the live rows are never rewritten
    ("assignments", "profile", "TEXT"),
    ("linecall_labels", "verdict", "TEXT"),   # the episode's verdict when it was labelled
    # 成長の記録: the work's 「チーム・グループ」 (NULL: none); people named by a number (#5)
    # are suggested as one person only between works with the same one
    ("assignments", "growth_group", "TEXT"),
    # 「データの扱い」: how consent was obtained (services.consent; NULL: not recorded), its
    # note, when the worker removed the work's videos (services.retention), and when a
    # person's data was removed on request (services.erasure)
    ("assignments", "consent", "TEXT"),
    ("assignments", "consent_note", "TEXT"),
    ("assignments", "media_removed_at", "TEXT"),
    ("subjects", "erased_at", "TEXT"),
    # accounts: the tenant of the data's roots (one organisation per install: 1), and who
    # did what (audit_log.actor: the login name at the time; NULL: the worker or the system)
    ("assignments", "org_id", "INTEGER NOT NULL DEFAULT 1"),
    ("growth_people", "org_id", "INTEGER NOT NULL DEFAULT 1"),
    ("audit_log", "org_id", "INTEGER NOT NULL DEFAULT 1"),
    ("audit_log", "actor_id", "INTEGER"),
    ("audit_log", "actor", "TEXT"),
    # 出典: the footage / material sources the work's PDFs print (one per line; NULL: none set)
    ("assignments", "credits", "TEXT"),
)


def init_schema(conn: sqlite3.Connection) -> None:
    existing = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='assignments'").fetchone()
    had_auto = conn.execute(   # both tables added in 2026-11 (auto_approvals, claim_checks)
        "SELECT 1 WHERE (SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        " AND name IN ('auto_approvals', 'claim_checks')) = 2").fetchone()
    copied = existing is None   # a new database needs no copy
    if existing is not None and had_auto is None:   # a new table on an existing database
        copied = True
        _pre_migration_copy(conn)
    conn.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        if column in {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}:
            continue
        if not copied:   # once, before the first column is added to an existing database
            copied = True
            _pre_migration_copy(conn)
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
        except sqlite3.OperationalError as e:   # another process added it just now
            if "duplicate column" not in str(e):
                raise
    if conn.execute("SELECT 1 FROM orgs LIMIT 1").fetchone() is None:
        conn.execute("INSERT OR IGNORE INTO orgs (id, name, created_at) VALUES (1, '', ?)",
                     (_now(),))


def _pre_migration_copy(conn: sqlite3.Connection) -> None:
    """An online copy of the database file beside it (ops.backup.pre_migration_copy)."""
    row = conn.execute("PRAGMA database_list").fetchone()
    if not row or not row[2]:   # an in-memory database
        return
    from pfb_app.ops import backup   # noqa: PLC0415 - ops imports storage
    backup.pre_migration_copy(Path(row[2]))


# stage_durations answers, per database and method: they feed a rough estimate only,
# so a few minutes old is fine, and the query (a join over all runs) is not repeated per tick.
# No index is added to the live database for it.
DURATIONS_TTL_SEC = 300
_DURATIONS: dict[tuple, tuple[float, dict[str, list[float]]]] = {}
_DURATIONS_LOCK = threading.Lock()
_clock = time.monotonic


def clear_stage_durations_cache() -> None:
    with _DURATIONS_LOCK:
        _DURATIONS.clear()


def _in(values) -> str:
    return ",".join("?" * len(values))


class ActiveJobsError(Exception):
    """The assignment still has queued, running or waiting jobs."""


class Repo:
    def __init__(self, conn: sqlite3.Connection, settings: Settings | None = None):
        """`settings` is needed only for delete_assignment to remove files."""
        self.conn = conn
        self.settings = settings
        # the group whose job is running on this connection (pipeline.runner sets it): the
        # call_log rows it writes or reuses are recorded as its own (call_log_owner)
        self.call_owner: int | None = None
        # who acts on this connection: (user id, login name), or None (the worker, the
        # system); ui.common.get_repo sets it from the signed-in person (services.auth)
        self.actor: tuple[int, str] | None = None

    @contextmanager
    def _tx(self, join: bool = False):
        """One write transaction (BEGIN IMMEDIATE … COMMIT; ROLLBACK on any error).

        Inside an open transaction this raises sqlite3.ProgrammingError, unless the caller
        asks to `join` it: the block then runs in a SAVEPOINT of the outer transaction, so an
        error rolls back only the block's own writes (the caller decides about the rest) and
        nothing is committed before the outer transaction commits."""
        if self.conn.in_transaction:
            if not join:
                raise sqlite3.ProgrammingError(
                    "a write transaction is already open; use _tx(join=True) to join it")
            name = f"pfb_join_{next(_SAVEPOINTS)}"
            self.conn.execute(f"SAVEPOINT {name}")
            try:
                yield
            except BaseException:
                self.conn.execute(f"ROLLBACK TO {name}")
                self.conn.execute(f"RELEASE {name}")
                raise
            self.conn.execute(f"RELEASE {name}")
            return
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        try:
            self.conn.execute("COMMIT")
        except BaseException:
            if self.conn.in_transaction:   # a failed COMMIT leaves the transaction open
                self.conn.execute("ROLLBACK")
            raise

    # --- assignments ---

    def create_assignment(self, title: str, kind: Kind, task_text: str, rubric_text: str,
                          model_key: str, require_identity_approval: bool,
                          domain: str = "school", sport: str | None = None,
                          profile: str | None = None, review_mode: str | None = None) -> int:
        """`profile` None: the domain's default (profiles.of_work reads it so). `review_mode`
        None: the column's default (approve); the screens pass services.corrections.NEW_WORK."""
        cur = self.conn.execute(
            "INSERT INTO assignments (title, kind, task_text, rubric_text, model_key,"
            " require_identity_approval, created_at, domain, sport, profile, review_mode)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,COALESCE(?, 'approve'))",
            (title, Kind(kind).value, task_text, rubric_text, model_key,
             int(require_identity_approval), _now(), domain, sport, profile, review_mode))
        return cur.lastrowid

    @staticmethod
    def _assignment(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["require_identity_approval"] = bool(d["require_identity_approval"])
        d["domain"] = d.get("domain") or "school"
        d.setdefault("sport", None)
        d["auto_approve"] = bool(d.get("auto_approve") or 0)
        d["review_mode"] = d.get("review_mode") or "approve"
        d.setdefault("credits", None)
        d["profile"] = profiles.of_work(d)
        d.setdefault("growth_group", None)
        for key in ("consent", "consent_note", "media_removed_at"):
            d.setdefault(key, None)
        return d

    def update_assignment_check(self, id: int, *, title: str, task_text: str, rubric_text: str,
                                domain: str, sport: str | None,
                                model_key: str | None = None,
                                profile: str | None = None) -> None:
        """The content check's confirm; `model_key` None keeps the work's; `profile` None is
        the domain's default."""
        self.conn.execute(
            "UPDATE assignments SET title=?, task_text=?, rubric_text=?, domain=?, sport=?,"
            " model_key=COALESCE(?, model_key), profile=? WHERE id=?",
            (title, task_text, rubric_text, domain, sport, model_key, profile, id))

    def rename_assignment(self, id: int, title: str) -> bool:
        """Change only the title; False when there is no such assignment."""
        cur = self.conn.execute("UPDATE assignments SET title=? WHERE id=?", (title, id))
        return cur.rowcount > 0

    def get_assignment(self, id: int) -> dict:
        row = self.conn.execute("SELECT * FROM assignments WHERE id=?", (id,)).fetchone()
        if row is None:
            raise KeyError(f"assignment {id} not found")
        return self._assignment(row)

    def list_assignments(self) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM assignments ORDER BY id").fetchall()
        return [self._assignment(r) for r in rows]

    def assignment_has_jobs(self, id: int) -> bool:
        """True once any job (in any state) exists for one of the assignment's submissions."""
        row = self.conn.execute(
            "SELECT 1 FROM jobs JOIN submissions ON jobs.submission_id = submissions.id"
            " WHERE submissions.assignment_id=? LIMIT 1", (id,)).fetchone()
        return row is not None

    def update_assignment_texts(self, id: int, task_text: str, rubric_text: str) -> bool:
        """Change the task and rubric text unless a job exists (checked in the same
        transaction). False when refused."""
        self.get_assignment(id)
        with self._tx():
            if self.assignment_has_jobs(id):
                return False
            self.conn.execute("UPDATE assignments SET task_text=?, rubric_text=? WHERE id=?",
                              (task_text, rubric_text, id))
        return True

    def set_review_mode(self, id: int, mode: str) -> None:
        """approve (the default) or correct_later (services.corrections)."""
        self.get_assignment(id)
        self.conn.execute("UPDATE assignments SET review_mode=? WHERE id=?", (mode, id))

    def set_credits(self, id: int, credits: str | None) -> None:
        """The work's 出典 (one source per line; None or blank: none set)."""
        self.get_assignment(id)
        text = (credits or "").strip() or None
        self.conn.execute("UPDATE assignments SET credits=? WHERE id=?", (text, id))

    def set_auto_approve(self, id: int, on: bool) -> None:
        """The work option 「確認せずに自動で承認する」 (services.review.auto_approve)."""
        self.get_assignment(id)
        self.conn.execute("UPDATE assignments SET auto_approve=? WHERE id=?", (int(on), id))

    def set_consent(self, id: int, consent: str | None, note: str | None) -> None:
        """The work's 同意の記録 (services.consent checks the values); blank clears the note.
        KeyError for an unknown work."""
        cur = self.conn.execute("UPDATE assignments SET consent=?, consent_note=? WHERE id=?",
                                (consent, (note or "").strip() or None, id))
        if cur.rowcount == 0:
            raise KeyError(f"assignment {id} not found")

    def set_pdf_downloaded(self, id: int) -> str:
        """Record now as the assignment's last student-PDF download; returns the time."""
        self.get_assignment(id)
        now = _now()
        self.conn.execute("UPDATE assignments SET pdf_downloaded_at=? WHERE id=?", (now, id))
        return now

    def stage_durations(self, stage: Stage, engine: str, limit: int = 50) -> list[float]:
        """How long the stage took (seconds) in its latest successful model runs of one
        method ("pfb3x3", "paper" or "document"), for rough remaining-time estimates. Runs a
        teacher saved and runs shorter than a second (no real work) are left out.

        Read with every other stage of the method (stage_durations_all), so the answers are
        kept for DURATIONS_TTL_SEC per database and method."""
        return self.stage_durations_all(engine, limit).get(Stage(stage).value, [])

    def stage_durations_all(self, engine: str, limit: int = 50) -> dict[str, list[float]]:
        """{stage: durations, newest first} of one method, as stage_durations, in one query;
        stages without a run are left out. Kept for DURATIONS_TTL_SEC per database, method
        and limit."""
        where = str(self.settings.data_dir) if self.settings is not None else None
        key = (where, engine, limit)
        now = _clock()
        if where is not None:
            with _DURATIONS_LOCK:
                hit = _DURATIONS.get(key)
            if hit is not None and now - hit[0] < DURATIONS_TTL_SEC:
                return {k: list(v) for k, v in hit[1].items()}
        durations = self._stage_durations_all(engine, limit)
        if where is not None:
            with _DURATIONS_LOCK:
                _DURATIONS[key] = (now, {k: list(v) for k, v in durations.items()})
        return durations

    def _stage_durations_all(self, engine: str, limit: int) -> dict[str, list[float]]:
        if engine == "document":
            method, args = "a.kind = 'document'", ()
        else:
            method = ("a.kind = 'video' AND (COALESCE(json_extract(s.meta, '$.engine'),"
                      " 'paper') = 'pfb3x3') = ?")
            args = (engine == "pfb3x3",)
        rows = self.conn.execute(
            "SELECT stage, served, duration_sec, id FROM (SELECT r.id, r.stage, r.duration_sec,"
            " COALESCE(r.model_id, '') LIKE 'served:%' AS served, ROW_NUMBER() OVER"
            " (PARTITION BY r.stage, COALESCE(r.model_id, '') LIKE 'served:%'"
            " ORDER BY r.id DESC) AS n FROM stage_runs r"
            " JOIN submissions s ON r.submission_id = s.id"
            " JOIN assignments a ON s.assignment_id = a.id WHERE r.source='model'"
            f" AND r.status='ok' AND r.duration_sec >= 1 AND {method})"
            " WHERE n <= ? ORDER BY stage, served, n", (*args, limit)).fetchall()
        split: dict[str, dict[bool, list[tuple[int, float]]]] = {}
        for stage, served, seconds, run_id in rows:
            split.setdefault(stage, {}).setdefault(bool(served), []).append((run_id, seconds))
        out: dict[str, list[float]] = {}
        for stage, parts in split.items():
            # the tracking engine's contribution / pfb on the served model (since the shared
            # server) take far less than the older in-process runs: those decide once any exist
            if engine == "pfb3x3" and stage in SERVED_TIMED and parts.get(True):
                runs = parts[True]
            else:
                runs = sorted([*parts.get(False, []), *parts.get(True, [])], reverse=True)
            out[stage] = [sec for _id, sec in runs[:limit]]   # newest first
        return out

    def submission_job_rows(self, submission_id: int) -> list[dict]:
        """Every job row of the submission's analysis (with created_at), oldest first; jobs
        of SIDE_STAGES (判定支援) are not part of it (latest_job_row reads them)."""
        rows = self.conn.execute(
            f"SELECT * FROM jobs WHERE submission_id=? AND stage NOT IN ({_in(SIDE_STAGES)})"
            " ORDER BY id", (submission_id, *SIDE_STAGES)).fetchall()
        return [dict(r) for r in rows]

    def latest_job_row(self, submission_id: int, stage: Stage,
                       camera: str | None = None) -> dict | None:
        """The newest job row of this stage key (subject-less), any status; None if none."""
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE submission_id=? AND stage=? AND camera IS ?"
            " AND subject_id IS NULL ORDER BY id DESC LIMIT 1",
            (submission_id, Stage(stage).value, camera)).fetchone()
        return dict(row) if row is not None else None

    # --- 判定支援: the coach's labels of line-call episodes ---

    def set_linecall_label(self, submission_id: int, camera: str, episode_id: str, label: str,
                           verdict: str | None = None) -> None:
        """Upsert one episode's label (correct / wrong / undecidable): the last press wins.
        `verdict`: the episode's verdict at this press (what a correct / wrong confirms)."""
        if label not in LINECALL_LABELS:
            raise ValueError(f"unknown line-call label: {label!r}")
        self.conn.execute(
            "INSERT INTO linecall_labels (submission_id, camera, episode_id, label, verdict,"
            " labeled_at) VALUES (?,?,?,?,?,?) ON CONFLICT (submission_id, camera, episode_id)"
            " DO UPDATE SET label=excluded.label, verdict=excluded.verdict,"
            " labeled_at=excluded.labeled_at",
            (submission_id, camera, episode_id, label, verdict, _now()))

    def linecall_labels(self, submission_id: int, camera: str) -> dict[str, str]:
        """episode id -> label of one camera."""
        return {eid: rec["label"]
                for eid, rec in self.linecall_label_records(submission_id, camera).items()}

    def linecall_label_records(self, submission_id: int, camera: str) -> dict[str, dict]:
        """episode id -> {"label", "verdict"} of one camera (the verdict it was given for)."""
        rows = self.conn.execute(
            "SELECT episode_id, label, verdict FROM linecall_labels"
            " WHERE submission_id=? AND camera=? ORDER BY episode_id",
            (submission_id, camera)).fetchall()
        return {r["episode_id"]: {"label": r["label"], "verdict": r["verdict"]} for r in rows}

    def delete_assignment(self, id: int) -> None:
        """Delete DB rows (cascade) and the assignment's files."""
        sids = [r["id"] for r in self.conn.execute(
            "SELECT id FROM submissions WHERE assignment_id=?", (id,))]
        if any(self.has_active_jobs(sid) for sid in sids):
            raise ActiveJobsError("実行中または待機中のジョブがあるため削除できません。完了するまでお待ちください。")
        with self._tx():
            self.conn.execute("DELETE FROM assignments WHERE id=?", (id,))
            self.prune_growth_people()   # 成長の記録 left without any work
        if self.settings is not None:
            for sid in sids:
                shutil.rmtree(files.submission_dir(self.settings, id, sid, create=False),
                              ignore_errors=True)
            shutil.rmtree(files.assignment_dir(self.settings, id, create=False),
                          ignore_errors=True)

    # --- submissions ---

    @staticmethod
    def _submission(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["meta"] = json.loads(d["meta"])
        d["identity_confirmed"] = bool(d["identity_confirmed"])
        return d

    def create_submission(self, assignment_id: int, label: str, meta: dict) -> int:
        cur = self.conn.execute(
            "INSERT INTO submissions (assignment_id, label, meta, created_at) VALUES (?,?,?,?)",
            (assignment_id, label, _json(meta), _now()))
        return cur.lastrowid

    def get_submission(self, id: int) -> dict:
        row = self.conn.execute("SELECT * FROM submissions WHERE id=?", (id,)).fetchone()
        if row is None:
            raise KeyError(f"submission {id} not found")
        return self._submission(row)

    def list_submissions(self, assignment_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM submissions WHERE assignment_id=? ORDER BY id", (assignment_id,)).fetchall()
        return [self._submission(r) for r in rows]

    def erasing(self, id: int) -> bool:
        """services.erasure is editing the group's files: its meta "erasing" stamp is younger
        than ERASING_STALE_SEC (an older one was left by a stop half way: logged, ignored)."""
        row = self.conn.execute("SELECT json_extract(meta, '$.erasing') FROM submissions"
                                " WHERE id=?", (id,)).fetchone()
        if row is None or not row[0]:
            return False
        try:
            at = datetime.fromisoformat(str(row[0]))
        except ValueError:
            return False
        if (datetime.now(timezone.utc) - at).total_seconds() < ERASING_STALE_SEC:
            return True
        _log.warning("消去の途中の印が古いまま残っています submission=%s（ワーカーが続きを"
                     "行います）", id)
        return False

    def update_submission_meta(self, id: int, meta: dict) -> None:
        self.conn.execute("UPDATE submissions SET meta=? WHERE id=?", (_json(meta), id))

    def set_identity_confirmed(self, id: int, confirmed: bool) -> None:
        self.conn.execute("UPDATE submissions SET identity_confirmed=? WHERE id=?",
                          (int(confirmed), id))

    # --- subjects ---

    @staticmethod
    def _subject(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["initial_positions"] = json.loads(d["initial_positions"])
        d["roster"] = json.loads(d.get("roster") or "{}")
        d["excluded"] = bool(d.get("excluded") or 0)
        d["approved_auto"] = bool(d.get("approved_auto") or 0)
        d.setdefault("erased_at", None)
        return d

    def add_subject(self, submission_id: int, participant_id: str | None, display_name: str,
                    appearance: str | None, initial_positions: dict[str, str],
                    roster: dict[str, str] | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO subjects (submission_id, participant_id, display_name, appearance,"
            " initial_positions, roster) VALUES (?,?,?,?,?,?)",
            (submission_id, participant_id, display_name, appearance, _json(initial_positions),
             _json(roster or {})))
        return cur.lastrowid

    def delete_subjects(self, submission_id: int, participant_ids: list[str] | None = None) -> int:
        """Delete the submission's subjects (all, or those with the given participant ids);
        returns how many were deleted.
        Their stage_runs and jobs go too (ON DELETE CASCADE on subject_id)."""
        if participant_ids is None:
            cur = self.conn.execute("DELETE FROM subjects WHERE submission_id=?", (submission_id,))
        elif not participant_ids:
            return 0
        else:
            cur = self.conn.execute(
                f"DELETE FROM subjects WHERE submission_id=? AND participant_id IN"
                f" ({_in(participant_ids)})", (submission_id, *participant_ids))
        return cur.rowcount

    def get_subject(self, id: int) -> dict:
        row = self.conn.execute("SELECT * FROM subjects WHERE id=?", (id,)).fetchone()
        if row is None:
            raise KeyError(f"subject {id} not found")
        return self._subject(row)

    def list_subjects(self, submission_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM subjects WHERE submission_id=? ORDER BY id", (submission_id,)).fetchall()
        return [self._subject(r) for r in rows]

    def rename_subject(self, subject_id: int, display_name: str) -> None:
        self.conn.execute("UPDATE subjects SET display_name=? WHERE id=?",
                          (display_name, subject_id))

    def set_excluded(self, subject_id: int, excluded: bool) -> None:
        """Mark a subject as not evaluated (True) or evaluated again (False); a person whose
        data was removed on request (erased_at) stays excluded."""
        self.conn.execute("UPDATE subjects SET excluded=? WHERE id=? AND (? = 1 OR erased_at"
                          " IS NULL)", (int(excluded), subject_id, int(excluded)))

    def set_approval(self, subject_id: int, run_id: int | None, auto: bool = False) -> None:
        """Approve the given run (`auto`: by the worker, not by a person), or clear approval
        when run_id is None."""
        if run_id is None:
            self.conn.execute(
                "UPDATE subjects SET approval_status='none', approved_run_id=NULL,"
                " approved_at=NULL, approved_auto=0 WHERE id=?", (subject_id,))
            self._end_auto_approvals(subject_id, "revoked")
        else:
            self.conn.execute(
                "UPDATE subjects SET approval_status='approved', approved_run_id=?,"
                " approved_at=?, approved_auto=? WHERE id=?",
                (run_id, _now(), int(auto), subject_id))
            if not auto:   # a person's approval after an automatic one: the person's stands
                self._end_auto_approvals(subject_id, "confirmed")

    # --- 自動承認 (services.autoapprove) ---

    def _end_auto_approvals(self, subject_id: int, status: str) -> None:
        self.conn.execute("UPDATE auto_approvals SET status=?, ended_at=? WHERE subject_id=?"
                          " AND status='active'", (status, _now(), subject_id))

    def add_auto_approval(self, subject_id: int, run_id: int, policy_version: str,
                          policy: dict, checks: list[dict]) -> int:
        """Keep what an automatic approval stood on (call it in the approving transaction)."""
        self._end_auto_approvals(subject_id, "replaced")
        cur = self.conn.execute(
            "INSERT INTO auto_approvals (subject_id, run_id, approved_at, policy_version,"
            " policy, checks) VALUES (?,?,?,?,?,?)",
            (subject_id, run_id, _now(), policy_version, _json(policy), _json(checks)))
        return cur.lastrowid

    def active_auto_approval(self, subject_id: int, run_id: int | None) -> dict | None:
        """The record of the standing automatic approval of this very text, else None."""
        row = self.conn.execute(
            "SELECT * FROM auto_approvals WHERE subject_id=? AND run_id=? AND status='active'"
            " ORDER BY id DESC LIMIT 1", (subject_id, run_id)).fetchone()
        return self._auto_row(row) if row else None

    def auto_approval_undone(self, subject_id: int, run_id: int) -> bool:
        """True when an automatic approval of this very text was put back by a person: it is
        never approved automatically again."""
        return self.conn.execute(
            "SELECT 1 FROM auto_approvals WHERE subject_id=? AND run_id=? AND status='revoked'",
            (subject_id, run_id)).fetchone() is not None

    def list_auto_approvals(self, subject_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT * FROM auto_approvals WHERE subject_id=? ORDER BY id",
                                 (subject_id,)).fetchall()
        return [self._auto_row(r) for r in rows]

    # --- 動作の再確認 (pipeline.claim_scan) ---

    def put_claim_check(self, run_id: int, event_id: str, method: str, verdict: dict) -> None:
        """Keep the re-check of one scene's action (a behavior run's event)."""
        self.conn.execute(
            "INSERT INTO claim_checks (run_id, event_id, method, verdict, created_at)"
            " VALUES (?,?,?,?,?)", (run_id, event_id, method, _json(verdict), _now()))

    def claim_checks(self, run_id: int) -> dict[str, dict[str, dict]]:
        """{event id: {method: verdict}} of a behavior run, the newest row of each method."""
        out: dict[str, dict[str, dict]] = {}
        for r in self.conn.execute("SELECT event_id, method, verdict FROM claim_checks"
                                   " WHERE run_id=? ORDER BY id", (run_id,)):
            out.setdefault(r["event_id"], {})[r["method"]] = json.loads(r["verdict"])
        return out

    @staticmethod
    def _auto_row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["policy"] = json.loads(d["policy"])
        d["checks"] = json.loads(d["checks"])
        return d

    # --- stage runs ---

    @staticmethod
    def _run(row: sqlite3.Row) -> StageRun:
        d = dict(row)
        d["upstream_run_ids"] = json.loads(d["upstream_run_ids"])
        d["warnings"] = json.loads(d["warnings"])
        d["output"] = json.loads(d["output"]) if d["output"] is not None else None
        return StageRun(**d)

    def save_stage_run(self, run: StageRun) -> StageRun:
        """Store as max(version)+1 for the run's key; id/version/created_at are assigned here."""
        created_at = _now()
        with self._tx(join=True):   # joins a caller's transaction (services.auto_people)
            version = self.conn.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM stage_runs WHERE submission_id=?"
                " AND stage=? AND camera IS ? AND subject_id IS ?",
                (run.submission_id, run.stage.value, run.camera, run.subject_id)).fetchone()[0]
            cur = self.conn.execute(
                "INSERT INTO stage_runs (submission_id, stage, camera, subject_id, version,"
                " source, upstream_run_ids, prompt_hash, model_id, output, status, error,"
                " warnings, duration_sec, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run.submission_id, run.stage.value, run.camera, run.subject_id, version,
                 run.source, _json(run.upstream_run_ids), run.prompt_hash, run.model_id,
                 _json(run.output) if run.output is not None else None, run.status, run.error,
                 _json(run.warnings), run.duration_sec, created_at))
            run_id = cur.lastrowid
            if run.stage in (Stage.identity, Stage.track):
                # Any new identity run (model or teacher) voids the teacher's confirmation, and
                # so does a new track run (tracking engine): its people must be mapped again.
                self.conn.execute("UPDATE submissions SET identity_confirmed=0 WHERE id=?",
                                  (run.submission_id,))
        return run.model_copy(update={"id": run_id, "version": version, "created_at": created_at})

    def get_run(self, run_id: int) -> StageRun | None:
        row = self.conn.execute("SELECT * FROM stage_runs WHERE id=?", (run_id,)).fetchone()
        return self._run(row) if row else None

    def run_history(self, submission_id: int, stage: Stage, camera: str | None = None,
                    subject_id: int | None = None) -> list[StageRun]:
        rows = self.conn.execute(
            "SELECT * FROM stage_runs WHERE submission_id=? AND stage=? AND camera IS ?"
            " AND subject_id IS ? ORDER BY version",
            (submission_id, Stage(stage).value, camera, subject_id)).fetchall()
        return [self._run(r) for r in rows]

    def latest_run(self, submission_id: int, stage: Stage, camera: str | None = None,
                   subject_id: int | None = None) -> StageRun | None:
        history = self.run_history(submission_id, stage, camera, subject_id)
        return history[-1] if history else None

    def list_cameras(self, submission_id: int) -> list[str]:
        """Distinct cameras that have a prepare run, sorted."""
        rows = self.conn.execute(
            "SELECT DISTINCT camera FROM stage_runs WHERE submission_id=? AND stage=?"
            " AND camera IS NOT NULL ORDER BY camera",
            (submission_id, Stage.prepare.value)).fetchall()
        return [r[0] for r in rows]

    def is_stale(self, run: StageRun) -> bool:
        """True if any upstream run is no longer the latest run for its key."""
        for up_id in run.upstream_run_ids:
            up = self.get_run(up_id)
            if up is None:
                return True
            latest = self.latest_run(up.submission_id, up.stage, up.camera, up.subject_id)
            if latest is None or latest.id != up_id:
                return True
        return False

    # --- jobs ---

    def enqueue(self, submission_id: int, stage: Stage, model_key: str,
                camera: str | None = None, subject_id: int | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO jobs (submission_id, stage, camera, subject_id, model_key, status,"
            " created_at) VALUES (?,?,?,?,?,?,?)",
            (submission_id, Stage(stage).value, camera, subject_id, model_key,
             JobStatus.queued.value, _now()))
        return cur.lastrowid

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        d = dict(row)
        d.pop("created_at")
        return Job(**d)

    def get_job(self, job_id: int) -> Job:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(f"job {job_id} not found")
        return self._job(row)

    def list_jobs(self, limit: int = 200) -> list[Job]:
        """Newest jobs first."""
        rows = self.conn.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._job(r) for r in rows]

    def retry_job(self, job_id: int) -> bool:
        """Put a failed job back in the queue. False if it is not failed or the same stage
        key already has an active job. A deliberate retry is a redo: its own progress starts
        again from nothing (a yield, a stale recovery or a GPU wait keep it: never_lower)."""
        with self._tx(join=True):   # the checks and the update at once (services.retention)
            return self._retry_job(job_id)

    def _retry_job(self, job_id: int) -> bool:
        job = self.get_job(job_id)
        if job.status not in (JobStatus.failed, JobStatus.failed_retryable):
            return False
        if self.conn.execute(   # the work's videos were removed (services.retention)
                "SELECT a.media_removed_at FROM assignments a JOIN submissions s"
                " ON s.assignment_id = a.id WHERE s.id=?", (job.submission_id,)).fetchone()[0]:
            return False
        if self.erasing(job.submission_id):   # services.erasure is editing its files
            return False
        if self.has_active_job(job.submission_id, job.stage, job.camera, job.subject_id):
            return False
        cur = self.conn.execute(
            "UPDATE jobs SET status=?, error=NULL, heartbeat_at=NULL, progress=NULL,"
            " progress_note=NULL, model_key=? WHERE id=? AND status IN (?,?)",
            (JobStatus.queued.value, self._retry_key(job), job_id, JobStatus.failed.value,
             JobStatus.failed_retryable.value))
        return cur.rowcount == 1

    def _retry_key(self, job: Job) -> str:
        """The key a retried job runs under now: a tracking-engine contribution / pfb job of
        an older release (queued under the plain model key, in-process) moves to its work's
        pfb3x3:<key> session, so one work never mixes two engines for these stages."""
        if job.stage not in (Stage.contribution, Stage.pfb) or ":" in job.model_key \
                or job.model_key in ("cpu", "whisper", "track", "detect"):
            return job.model_key
        meta = self.get_submission(job.submission_id)["meta"]
        return f"pfb3x3:{job.model_key}" if meta.get("engine") == "pfb3x3" else job.model_key

    def claim_next_job(self, model_keys: set[str], stage: Stage | None = None,
                       skip_stages=None, after_upstream: bool = False) -> Job | None:
        """Atomically move the oldest queued/waiting_gpu job for these models to running.
        `stage`: only when that oldest job is of this stage (None otherwise), so a job claimed
        alongside another never skips ahead of an older job of another stage.

        For a session that runs several jobs at once: `skip_stages` takes the oldest job of
        any other stage (one that runs as many engine commands as it may still claims the
        text-only jobs behind them); `after_upstream` leaves out a job while an older queued /
        waiting / running job of its own group is one it builds on (UPSTREAM_STAGES: a pfb
        never runs beside the contribution or behavior before it), which one job at a time
        gave by itself."""
        if not model_keys:
            return None
        keys = sorted(model_keys)
        skip = sorted(Stage(s).value for s in skip_stages or ())
        where = f" AND j.stage NOT IN ({_in(skip)})" if skip else ""
        params: list = [*_CLAIMABLE, *keys, *skip]
        if after_upstream:
            pairs = [(down, up) for down, ups in UPSTREAM_STAGES.items() for up in ups]
            where += (" AND NOT EXISTS (SELECT 1 FROM jobs k WHERE k.submission_id = j.submission_id"
                      f" AND k.id < j.id AND k.status IN ({_in(_ACTIVE)}) AND ("
                      + " OR ".join("(j.stage=? AND k.stage=?)" for _ in pairs) + "))")
            params += [*_ACTIVE, *(v for pair in pairs for v in pair)]
        with self._tx():
            row = self.conn.execute(
                f"SELECT j.id, j.stage FROM jobs j WHERE j.status IN ({_in(_CLAIMABLE)})"
                f" AND j.model_key IN ({_in(keys)}){where} ORDER BY j.id LIMIT 1",
                params).fetchone()
            if row is None or (stage is not None and row["stage"] != Stage(stage).value):
                return None
            self.conn.execute(
                "UPDATE jobs SET status=?, heartbeat_at=?, attempts=attempts+1, error=NULL"
                " WHERE id=?", (JobStatus.running.value, _now(), row["id"]))
        return self.get_job(row["id"])

    def set_job_progress(self, job_id: int, progress: float | None, note: str | None, *,
                         never_lower: bool = False) -> None:
        """A job's own progress (0..1, None: unknown) and its plain note, as the worker
        knows them. `never_lower` (the worker): a lower or unknown progress than the stored
        one leaves the row as it is (a requeued or reclaimed job keeps its bar; a real redo is
        a new job row)."""
        if not never_lower:
            self.conn.execute("UPDATE jobs SET progress=?, progress_note=? WHERE id=?",
                              (progress, note, job_id))
        elif progress is not None:
            self.conn.execute("UPDATE jobs SET progress=?, progress_note=? WHERE id=?"
                              " AND (progress IS NULL OR progress <= ?)",
                              (progress, note, job_id, progress))

    def track_rates(self, limit: int = 50) -> list[float]:
        """Seconds of tracking per analysed video second in the latest successful model track
        runs of the tracking method that recorded their analysed length, newest first."""
        rows = self.conn.execute(
            "SELECT r.duration_sec / json_extract(r.output, '$.analysed_s') FROM stage_runs r"
            " JOIN submissions s ON r.submission_id = s.id WHERE r.stage='track'"
            " AND r.source='model' AND r.status='ok' AND r.duration_sec >= 1"
            " AND json_extract(s.meta, '$.engine') = 'pfb3x3'"
            " AND json_extract(r.output, '$.analysed_s') > 0 ORDER BY r.id DESC LIMIT ?",
            (limit,)).fetchall()
        return [r[0] for r in rows]

    def heartbeat(self, job_id: int) -> None:
        self.conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (_now(), job_id))

    def finish_job(self, job_id: int, status: JobStatus, error: str | None = None) -> bool:
        """Set the outcome of a running job.

        Returns False and leaves the row unchanged if the job is no longer running
        (e.g. recover_stale_jobs already flipped a slow job).
        """
        cur = self.conn.execute("UPDATE jobs SET status=?, error=? WHERE id=? AND status=?",
                                (JobStatus(status).value, error, job_id, JobStatus.running.value))
        return cur.rowcount == 1

    def mark_waiting_gpu(self, model_key: str, reason: str) -> int:
        """Move this model's queued/waiting jobs to waiting_gpu and store the reason."""
        cur = self.conn.execute(
            f"UPDATE jobs SET status=?, error=? WHERE model_key=? AND status IN ({_in(_CLAIMABLE)})",
            (JobStatus.waiting_gpu.value, reason, model_key, *_CLAIMABLE))
        return cur.rowcount

    def count_jobs(self, model_key: str, status: JobStatus) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM jobs WHERE model_key=? AND status=?",
                                 (model_key, status.value)).fetchone()[0]

    def requeue_running(self, model_key: str) -> None:
        self.conn.execute("UPDATE jobs SET status=? WHERE status=? AND model_key=?",
                          (JobStatus.queued.value, JobStatus.running.value, model_key))

    def recover_stale_jobs(self, older_than_sec: float, *, model_key: str | None = None,
                           exclude_key: str | None = None) -> int:
        """Running jobs without a heartbeat for older_than_sec -> failed_retryable; only
        `model_key`'s, or all but `exclude_key`'s (a linecall child runs beside a session)."""
        sql = ("UPDATE jobs SET status=?, error=? WHERE status=? AND (heartbeat_at IS NULL"
               " OR heartbeat_at < ?)")
        args = [JobStatus.failed_retryable.value, "heartbeat timeout", JobStatus.running.value,
                _now(-older_than_sec)]
        if model_key is not None:
            sql += " AND model_key=?"
            args.append(model_key)
        if exclude_key is not None:
            sql += " AND model_key<>?"
            args.append(exclude_key)
        return self.conn.execute(sql, args).rowcount

    def has_active_jobs(self, submission_id: int) -> bool:
        row = self.conn.execute(
            f"SELECT 1 FROM jobs WHERE submission_id=? AND status IN ({_in(_ACTIVE)}) LIMIT 1",
            (submission_id, *_ACTIVE)).fetchone()
        return row is not None

    def active_job_statuses(self, submission_id: int) -> set[str]:
        """Statuses (queued/running/waiting_gpu) of the submission's active jobs."""
        rows = self.conn.execute(
            f"SELECT DISTINCT status FROM jobs WHERE submission_id=? AND status IN ({_in(_ACTIVE)})",
            (submission_id, *_ACTIVE)).fetchall()
        return {r[0] for r in rows}

    def blocking_job_statuses(self, submission_id: int) -> set[str]:
        """active_job_statuses without the stages that never hold up the coach
        (NON_BLOCKING_STAGES: names only stores proposal inputs)."""
        rows = self.conn.execute(
            f"SELECT DISTINCT status FROM jobs WHERE submission_id=? AND status IN ({_in(_ACTIVE)})"
            f" AND stage NOT IN ({_in(NON_BLOCKING_STAGES)})",
            (submission_id, *_ACTIVE, *NON_BLOCKING_STAGES)).fetchall()
        return {r[0] for r in rows}

    def has_blocking_jobs(self, submission_id: int) -> bool:
        """An active job that a coach action must wait for (every stage but names). Deleting
        a work and the worker-alive check still use has_active_jobs."""
        return bool(self.blocking_job_statuses(submission_id))

    def pending_job_keys(self, submission_id: int, stage: Stage,
                         camera: str | None = None) -> set[str]:
        """Model keys of this stage key's queued/waiting_gpu (not yet running) jobs."""
        rows = self.conn.execute(
            f"SELECT DISTINCT model_key FROM jobs WHERE submission_id=? AND stage=?"
            f" AND camera IS ? AND subject_id IS NULL AND status IN ({_in(_CLAIMABLE)})",
            (submission_id, Stage(stage).value, camera, *_CLAIMABLE)).fetchall()
        return {r[0] for r in rows}

    def cancel_pending_jobs(self, submission_id: int, stage: Stage,
                            camera: str | None = None, subject_id: int | None = None) -> int:
        """Delete this stage key's queued/waiting_gpu jobs (a running one is left alone);
        returns how many."""
        cur = self.conn.execute(
            f"DELETE FROM jobs WHERE submission_id=? AND stage=? AND camera IS ?"
            f" AND subject_id IS ? AND status IN ({_in(_CLAIMABLE)})",
            (submission_id, Stage(stage).value, camera, subject_id, *_CLAIMABLE))
        return cur.rowcount

    def fail_pending_jobs(self, model_key: str, error: str) -> int:
        """Mark this model's queued/waiting jobs failed_retryable with the error."""
        cur = self.conn.execute(
            f"UPDATE jobs SET status=?, error=? WHERE model_key=? AND status IN ({_in(_CLAIMABLE)})",
            (JobStatus.failed_retryable.value, error, model_key, *_CLAIMABLE))
        return cur.rowcount

    def has_active_job(self, submission_id: int, stage: Stage, camera: str | None = None,
                       subject_id: int | None = None) -> bool:
        """True if a queued/running/waiting_gpu job exists for exactly this stage key."""
        row = self.conn.execute(
            f"SELECT 1 FROM jobs WHERE submission_id=? AND stage=? AND camera IS ?"
            f" AND subject_id IS ? AND status IN ({_in(_ACTIVE)}) LIMIT 1",
            (submission_id, Stage(stage).value, camera, subject_id, *_ACTIVE)).fetchone()
        return row is not None

    def pending_model_keys(self) -> set[str]:
        rows = self.conn.execute(
            f"SELECT DISTINCT model_key FROM jobs WHERE status IN ({_in(_CLAIMABLE)})",
            _CLAIMABLE).fetchall()
        return {r["model_key"] for r in rows}

    # --- 成長の記録 (services.growth decides; these only store) ---

    def create_growth_person(self, name: str) -> int:
        cur = self.conn.execute("INSERT INTO growth_people (name, created_at) VALUES (?,?)",
                                (name, _now()))
        return cur.lastrowid

    def growth_person(self, person_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM growth_people WHERE id=?",
                                (person_id,)).fetchone()
        return dict(row) if row is not None else None

    def growth_people(self) -> list[dict]:
        """The records with at least one link, oldest first."""
        rows = self.conn.execute(
            "SELECT p.* FROM growth_people p WHERE EXISTS (SELECT 1 FROM growth_links l"
            " WHERE l.person_id = p.id) ORDER BY p.id").fetchall()
        return [dict(r) for r in rows]

    def rename_growth_person(self, person_id: int, name: str) -> None:
        self.conn.execute("UPDATE growth_people SET name=? WHERE id=?", (name, person_id))

    def growth_links(self, person_id: int | None = None) -> list[dict]:
        """{subject_id, person_id, how, linked_at} of one record (None: every record)."""
        if person_id is None:
            rows = self.conn.execute("SELECT * FROM growth_links ORDER BY subject_id")
        else:
            rows = self.conn.execute("SELECT * FROM growth_links WHERE person_id=?"
                                     " ORDER BY subject_id", (person_id,))
        return [dict(r) for r in rows.fetchall()]

    def growth_person_of(self, subject_id: int) -> int | None:
        row = self.conn.execute("SELECT person_id FROM growth_links WHERE subject_id=?",
                                (subject_id,)).fetchone()
        return row[0] if row is not None else None

    def _prune_growth(self, person_id: int | None) -> None:
        """Remove the record when it has no link left (a record just made is kept)."""
        if person_id is not None:
            self.conn.execute("DELETE FROM growth_people WHERE id=? AND NOT EXISTS (SELECT 1"
                              " FROM growth_links l WHERE l.person_id = growth_people.id)",
                              (person_id,))

    def link_growth(self, subject_id: int, person_id: int, how: str) -> None:
        """Link (or move) the subject to the record; a record left without links goes."""
        with self._tx(join=True):
            before = self.growth_person_of(subject_id)
            self.conn.execute(
                "INSERT INTO growth_links (subject_id, person_id, how, linked_at)"
                " VALUES (?,?,?,?) ON CONFLICT (subject_id) DO UPDATE SET"
                " person_id=excluded.person_id, how=excluded.how, linked_at=excluded.linked_at",
                (subject_id, person_id, how, _now()))
            if before != person_id:
                self._prune_growth(before)

    def unlink_growth(self, subject_id: int) -> None:
        with self._tx(join=True):
            before = self.growth_person_of(subject_id)
            self.conn.execute("DELETE FROM growth_links WHERE subject_id=?", (subject_id,))
            self._prune_growth(before)

    def decline_growth(self, subject_id: int, other_id: int) -> None:
        a, b = sorted((subject_id, other_id))
        self.conn.execute("INSERT OR IGNORE INTO growth_declined (subject_a, subject_b,"
                          " declined_at) VALUES (?,?,?)", (a, b, _now()))

    def growth_declined(self) -> set[tuple[int, int]]:
        return {(r[0], r[1]) for r in self.conn.execute(
            "SELECT subject_a, subject_b FROM growth_declined")}

    def prune_growth_people(self) -> int:
        """Remove every record left without a link (its people's works were deleted)."""
        return self.conn.execute("DELETE FROM growth_people WHERE NOT EXISTS (SELECT 1 FROM"
                                 " growth_links l WHERE l.person_id = growth_people.id)").rowcount

    def growth_member_rows(self, subject_ids=None) -> list[dict]:
        """Every person of every work (or only `subject_ids`) in one query, with their work,
        group, record and whether their newest feedback text is the approved one (`approved`;
        staleness is left to review.subject_status; in a 「あとから訂正」 work (`live`): the newest
        text is ok and not reported wrong)."""
        where, args = "", ()
        if subject_ids is not None:
            ids = sorted(set(subject_ids))
            if not ids:
                return []
            where, args = f" WHERE s.id IN ({_in(ids)})", tuple(ids)
        rows = self.conn.execute(
            "SELECT s.id AS subject_id, s.submission_id, s.participant_id, s.display_name,"
            f" s.excluded, {_GROWTH_USABLE} AS approved,"
            " COALESCE(a.review_mode, '') = 'correct_later' AS live,"
            " b.label AS group_label, b.meta, a.id AS assignment_id, a.title, a.created_at,"
            " a.kind, a.domain, a.sport, a.profile, a.growth_group,"
            " (SELECT COUNT(*) FROM submissions b2 WHERE b2.assignment_id = a.id) AS groups,"
            " l.person_id FROM subjects s JOIN submissions b ON s.submission_id = b.id"
            " JOIN assignments a ON b.assignment_id = a.id"
            " LEFT JOIN growth_links l ON l.subject_id = s.id" + where, args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["meta"] = json.loads(d["meta"] or "{}")
            d["approved"] = bool(d["approved"])
            d["live"] = bool(d["live"])
            d["excluded"] = bool(d["excluded"] or 0)
            out.append(d)
        return out

    def growth_handout_count(self, assignment_id: int, subject_ids=None) -> int:
        """People of this work with an approved text (only `subject_ids` when given: the
        people the PDF holds) whose record has approved texts in two or more works up to and
        including this work (date, then id), in one query (「あとから訂正」 works: the newest
        text not reported wrong, as growth_member_rows)."""
        only, extra = "", ()
        if subject_ids is not None:
            ids = sorted(set(subject_ids))
            if not ids:
                return 0
            only, extra = f" AND ok.id IN ({_in(ids)})", tuple(ids)
        row = self.conn.execute(
            "WITH ok AS (SELECT s.id, a.id AS aid, a.created_at FROM subjects s"
            " JOIN submissions b ON s.submission_id = b.id JOIN assignments a"
            f" ON b.assignment_id = a.id WHERE {_GROWTH_USABLE} AND {_GROWTH_EVALUATED}),"
            " w AS (SELECT id, COALESCE(created_at, '') AS at FROM assignments WHERE id = ?),"
            " here AS (SELECT DISTINCT l.person_id FROM growth_links l JOIN ok"
            f" ON ok.id = l.subject_id WHERE ok.aid = ?{only})"
            " SELECT COUNT(*) FROM (SELECT l.person_id FROM growth_links l"
            " JOIN ok ON ok.id = l.subject_id, w WHERE l.person_id IN (SELECT person_id FROM"
            " here) AND (COALESCE(ok.created_at, '') < w.at OR (COALESCE(ok.created_at, '')"
            " = w.at AND ok.aid <= w.id)) GROUP BY l.person_id"
            " HAVING COUNT(DISTINCT ok.aid) >= 2)",
            (assignment_id, assignment_id, *extra)).fetchone()
        return row[0]

    def set_growth_group(self, assignment_id: int, group: str | None) -> None:
        """The work's 「チーム・グループ」; blank clears it."""
        value = (group or "").strip() or None
        self.conn.execute("UPDATE assignments SET growth_group=? WHERE id=?",
                          (value, assignment_id))

    # --- call cache and audit ---

    def cache_get(self, request_hash: str) -> str | None:
        row = self.conn.execute(
            "SELECT raw FROM call_log WHERE request_hash=? AND error IS NULL",
            (request_hash,)).fetchone()
        if row is not None:
            self._own_call(request_hash)
        return row["raw"] if row else None

    def _own_call(self, request_hash: str) -> None:
        """Record the running group as a user of this answer, so deleting its work deletes it.
        A busy database is tried again (the connection already waits 30 s each time); if it
        still fails, the answer stays usable and the miss is logged."""
        if self.call_owner is None:
            return
        for attempt in range(OWNER_TRIES):
            try:
                self._insert_owner(request_hash)
                return
            except sqlite3.OperationalError:
                if attempt + 1 < OWNER_TRIES:
                    time.sleep(0.05 * (attempt + 1))
        _log.error("モデルの回答と作業のつながりを記録できませんでした request_hash=%s"
                   " submission=%s（作業を削除してもこの回答は残ります）", request_hash,
                   self.call_owner)

    def _insert_owner(self, request_hash: str) -> None:
        self.conn.execute("INSERT OR IGNORE INTO call_log_owner (request_hash, submission_id)"
                          " SELECT ?, id FROM submissions WHERE id=?",
                          (request_hash, self.call_owner))

    def cache_put(self, request_hash: str, model_id: str, raw: str | None, prompt_tokens: int,
                  completion_tokens: int, duration: float, error: str | None) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO call_log (request_hash, model_id, raw, prompt_tokens,"
            " completion_tokens, duration, error, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (request_hash, model_id, raw, prompt_tokens, completion_tokens, duration, error,
             _now()))
        self._own_call(request_hash)

    # --- app-wide settings (JSON values) ---

    def get_setting(self, key: str):
        """The stored value, or None when missing or unreadable."""
        row = self.conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except ValueError:
            return None

    def set_setting(self, key: str, value) -> None:
        """Store (replace) the value; audited with the value (settings hold no names)."""
        with self._tx(join=True):
            self.conn.execute(
                "INSERT INTO app_settings (key, value, updated_at) VALUES (?,?,?)"
                " ON CONFLICT (key) DO UPDATE SET value=excluded.value,"
                " updated_at=excluded.updated_at", (key, _json(value), _now()))
            self.audit("set_setting", key, value)

    # --- 解説動画 (services.explainer) ---

    def explainer_enqueue(self, submission_id: int, kind: str, subject_id: int | None,
                          stamp: str, session_id: int | None, actor: str | None
                          ) -> tuple[int, bool]:
        """(job id, True) for a new queued job, or (the active one's id, False) when one is
        already queued or running for the same group, kind and person."""
        with self._tx(join=True):
            row = self.conn.execute(
                "SELECT id FROM explainer_jobs WHERE submission_id=? AND kind=? AND subject_id IS ?"
                " AND status IN ('queued', 'running') ORDER BY id LIMIT 1",
                (submission_id, kind, subject_id)).fetchone()
            if row is not None:
                return row[0], False
            cur = self.conn.execute(
                "INSERT INTO explainer_jobs (submission_id, subject_id, kind, status, stamp,"
                " session_id, actor, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (submission_id, subject_id, kind, "queued", stamp, session_id, actor, _now()))
            return cur.lastrowid, True

    def explainer_job(self, job_id: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM explainer_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row is not None else None

    def explainer_latest(self, submission_id: int, kind: str,
                         subject_id: int | None) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM explainer_jobs WHERE submission_id=? AND kind=? AND subject_id IS ?"
            " ORDER BY id DESC LIMIT 1", (submission_id, kind, subject_id)).fetchone()
        return dict(row) if row is not None else None

    def explainer_claim(self) -> dict | None:
        """The oldest queued explainer job, now running (None: nothing queued)."""
        with self._tx():
            row = self.conn.execute("SELECT id FROM explainer_jobs WHERE status='queued'"
                                    " ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            now = _now()
            self.conn.execute("UPDATE explainer_jobs SET status='running', started_at=?,"
                              " heartbeat_at=?, progress=0 WHERE id=?", (now, now, row[0]))
        return self.explainer_job(row[0])

    def explainer_progress(self, job_id: int, progress: float | None, note: str | None) -> None:
        self.conn.execute("UPDATE explainer_jobs SET progress=?, note=?, heartbeat_at=?"
                          " WHERE id=? AND status='running'", (progress, note, _now(), job_id))

    def explainer_finish(self, job_id: int, status: str, error: str | None = None) -> bool:
        """End a queued or running job; False when it had ended already (or is gone)."""
        return self.conn.execute(
            "UPDATE explainer_jobs SET status=?, error=?, finished_at=? WHERE id=?"
            " AND status IN ('queued', 'running')", (status, error, _now(), job_id)).rowcount > 0

    def explainer_cancel(self, submission_ids, reason: str) -> int:
        """Cancel the queued and running explainer jobs of these groups."""
        ids = [int(i) for i in submission_ids]
        if not ids:
            return 0
        return self.conn.execute(
            f"UPDATE explainer_jobs SET status='cancelled', error=?, finished_at=?"
            f" WHERE submission_id IN ({_in(ids)}) AND status IN ('queued', 'running')",
            (reason, _now(), *ids)).rowcount

    def explainer_beat(self, job_id: int) -> None:
        self.conn.execute("UPDATE explainer_jobs SET heartbeat_at=? WHERE id=? AND"
                          " status='running'", (_now(), job_id))

    def explainer_fail_stale(self, older_than_sec: float, reason: str) -> int:
        """A running job whose heartbeat is older than this has stopped (a hung render)."""
        return self.conn.execute(
            "UPDATE explainer_jobs SET status='failed', error=?, finished_at=? WHERE"
            " status='running' AND COALESCE(heartbeat_at, started_at, created_at) < ?",
            (reason, _now(), _now(-older_than_sec))).rowcount

    def explainer_fail_running(self, reason: str) -> int:
        """At worker start: a job left running by a stopped worker has ended."""
        return self.conn.execute(
            "UPDATE explainer_jobs SET status='failed', error=?, finished_at=?"
            " WHERE status='running'", (reason, _now())).rowcount

    def latest_audit(self, action: str, target: str | None) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM audit_log WHERE action=? AND target IS ? ORDER BY id DESC LIMIT 1",
            (action, target)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["detail"] = json.loads(d["detail"])
        return d

    def count_audit_since(self, action: str, target: str, since_action: str) -> int:
        """Entries `action` for `target` after its latest `since_action` entry (or ever)."""
        return self.conn.execute(
            "SELECT COUNT(*) FROM audit_log WHERE action=? AND target=? AND id > COALESCE("
            "(SELECT MAX(id) FROM audit_log WHERE action=? AND target=?), 0)",
            (action, target, since_action, target)).fetchone()[0]

    def audit(self, action: str, target: str | None, detail: dict) -> None:
        actor_id, actor = self.actor or (None, None)
        self.conn.execute("INSERT INTO audit_log (ts, action, target, detail, actor_id, actor)"
                          " VALUES (?,?,?,?,?,?)",
                          (_now(), action, target, _json(detail), actor_id, actor))
