"""Where a piece of work (one assignment and its submissions) stands, derived from the DB alone.

Nothing here is stored: every call re-reads jobs, runs, confirmations and approvals, so a
browser reload or a second tab always sees the same step. Each submission gets its own step
and status; the work's current step is the earliest step any submission still needs.

- Work steps (`WorkProgress.keys`): "assignment", "files", "people" (video only), "wait",
  "review", "pdf".
- A submission's step (`SubProgress.step`) is one of `SUB_STEPS`: "assignment" (a video
  group held for its content check: 「動画の中身を確かめています」 while the check runs, then
  the teacher's turn to confirm the card), "people" (video only), "wait", "review", "pdf".
  A submission always has its files, so it is never at "files".
- Statuses follow the theme's status chips: "your_turn", "processing", "waiting_gpu",
  "stopped", "done". Reasons are plain Japanese for teachers. A stopped group the teacher can
  move on with one press carries `SubProgress.action` ("retry": a failed job, put back in
  the queue by services.retry (「もう一度ためす」) until it has failed GIVE_UP_AFTER retries,
  then the reason says to tell the administrator; "reselect": nobody to evaluate; "rerun":
  no behavior recorded for anyone evaluated). Nobody tracked on any camera stops with no
  action (NO_ONE_TRACKED: redo it on the admin pages, 管理用の画面).
- A work started from the drop zone (a group with meta["check"]) calls step 1 「動画の中身」
  and step 2 「動画（n本）」, and step 2 counts as done: the videos are in.
- Tracking method: step 3 (人を確かめる（任意）) never blocks when every camera has its
  automatic people (an identity run with source "auto"); only a camera with no current
  mapping at all waits for the coach there.

- How far the analysis has got: each SubProgress and the WorkProgress carry an `analysis`
  (ui/wizard/analysis.py: stages, percentages, the review line), derived from the same rows.

One call reads the assignment's rows once (`_Snapshot`), and a complete work whose rows have
not changed since it was last derived is answered from a small cache. A screen builds one
snapshot per page run (and the live parts one per tick) with `snapshot` and passes it to every
reader, `work_progress` included.
"""
import copy
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pfb_app.domain import profiles
from pfb_app.domain.vocab import words
from pfb_app.domain.models import JobStatus, Kind, Stage, StageRun, SubjectStatus
from pfb_app.pipeline.auto_people import is_evaluated
from pfb_app.pipeline.common import submission_engine, usable
from pfb_app.pipeline.tracked_identity import mapping_is_current
from pfb_app.services import retry, review
from pfb_app.storage.db import NON_BLOCKING_STAGES, SIDE_STAGES, Repo
from pfb_app.ui.wizard.analysis import (PIPELINES, Analysis, stage_weights, submission_analysis,
                                        target_counts, weight_counts, work_analysis)

# step 3 of the tracking method never blocks: the people are chosen automatically
TRACKING_STEPS = ["活動を決める", "動画を入れる", "人を確かめる（任意）", "解析を待つ",
                  "確認して承認", "PDFを受け取る"]
PAPER_STEPS = ["活動を決める", "動画を入れる", "参加者を確認", "解析を待つ", "確認して承認",
               "PDFを受け取る"]
DOCUMENT_STEPS = ["活動を決める", "書類を入れる", "解析を待つ", "確認して承認", "PDFを受け取る"]
VIDEO_KEYS = ["assignment", "files", "people", "wait", "review", "pdf"]
DOCUMENT_KEYS = ["assignment", "files", "wait", "review", "pdf"]
SUB_STEPS = ("assignment", "people", "wait", "review", "pdf")
# meta["check"] of a video group still held for its content check (not confirmed)
CHECK_HELD = ("pending", "ready")
CHECKING = "動画の中身を確かめています"
CHECKED = "確かめた内容を見て、決定してください"
# the first two steps of a work started from the drop zone (a group with meta["check"])
CHECK_FIRST = "動画の中身"
CHECK_VIDEOS = "動画（{n}本）"
# Nothing running but the next result not there yet: shown as processing for this long after
# the submission's last activity (a finished job enqueues its follow-ups just after).
GRACE_SEC = 30
# stopped with nothing left to run: the reason, and the action (SubProgress.action) that
# moves the work on again (tracking method)
NOBODY = "評価する人が見つかりませんでした。"
# every camera's tracking found nobody: choosing the people again cannot help (no action)
NO_ONE_TRACKED = "動画に人が見つかりませんでした。管理用の画面でやり直してください"
NO_EVENTS = "人の動きが記録できませんでした。"
NOT_STARTED = "映っている人は選びましたが、解析が始まりませんでした。"
RESELECT = "映っている人を自動で選び直す"   # action "reselect"
RERUN = "もう一度解析する"                 # action "rerun"
RETRY = "もう一度ためす"                   # action "retry" (services.retry)
GIVE_UP = "何度ためしても止まります。管理者に知らせてください。"   # after GIVE_UP_AFTER retries
OUT_OF_MEMORY = "ほかの解析と重なって止まりました。もう一度ためすと、たいてい進みます。"

_ACTIVE = {JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value}
_SIDE_MARKS = ",".join("?" * len(SIDE_STAGES))   # 判定支援 jobs are not the work's analysis
_ORDER = {s.value: n for n, s in enumerate(Stage)}   # pipeline order
AUTO_SOURCE = "auto"   # the identity run of the automatic people (services.auto_people)
_PEOPLE_STAGES = {Stage.prepare.value, Stage.transcribe.value, Stage.track.value,
                  Stage.identity.value}

# what a stage is called, and what it is doing, in plain words
_NOUN = {"detect": "動画の中身の確認", "prepare": "動画の読み込み", "transcribe": "文字起こし",
         "track": "人を見つける", "identity": "参加者の特定", "behavior": "したことの書き出し",
         "evidence": "場面の書き出し", "contribution": "よかった点を選ぶ", "pfb": "文章づくり",
         "names": "名前の候補づくり", "linecall": "判定材料づくり"}
_DOING = {"detect": CHECKING, "prepare": "動画を読み込んでいます",
          "transcribe": "音声を文字にしています", "track": "人を見つけています",
          "identity": "参加者を特定しています", "behavior": "したことを書き出しています",
          "evidence": "場面を書き出しています", "contribution": "よかった点を選んでいます",
          "pfb": "文章を書いています", "names": "名前の候補を作っています",
          "linecall": "判定材料を作っています"}
_DOC_NOUN = {"prepare": "書類の読み込み"}
_DOC_DOING = {"prepare": "書類を読み込んでいます"}
# a feedback text built on a contribution that is itself outdated: making the text again (or
# writing it by hand) would only give another outdated text, so 管理用の画面 has to redo that first
REDO_CONTRIBUTION = ("解析の途中の結果が直されたため、"
                     "管理用の画面で『貢献の判定』から作り直してください")

_CAMERA = re.compile(r"(?<![A-Za-z0-9])cam(\d+)(?![A-Za-z0-9])")
_EXC_PREFIX = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception)\s*:\s*")
_JAPANESE = re.compile(r"[぀-ヿ㐀-鿿]")
# pipeline texts that name old screens, internal steps or server paths, in plain words
# (mapped here so the pipeline's own texts, still read on 管理用の画面, stay as they are)
_PLAIN = (
    (re.compile(r"^解析サーバーが繰り返し停止しました"),            # runner.SERVER_DOWN_MESSAGE
     "解析サーバーが止まりました。もう一度ためすと、たいてい進みます。"),
    (re.compile(r"先に人物の追跡を実行してください"),               # engine/pfb3x3.py (+ a path)
     "先に人を見つける必要があります。"),
    (re.compile(r"^人物の追跡結果がありません"),                    # pipeline/tracking.py, naming.py
     "人を見つけた結果がまだありません。"),
)
# Service texts (services/review.py, pipeline/tracked_identity.py) still say 人物の追跡 because
# the admin pages and their tests use them; step 3 shows them through `plain_service`.
_SERVICE_WORDS = (
    ("先に人物の追跡を実行してください", "先に人を見つけてください"),
    ("人物の追跡が実行中または待機中です", "人を見つけている途中です（順番待ちを含みます）"),
    ("人物の追跡がやり直されています", "人を見つけ直しています"),
    ("人物の追跡結果", "人を見つけた結果"),
    ("人物の追跡からやり直して", "人を見つけるところからやり直して"),
    ("人物の追跡", "人を見つける"),
)
MODEL_NOT_LOADED = "モデルを読み込めませんでした"                    # worker/main.py


@dataclass
class SubProgress:
    submission_id: int
    label: str
    engine: str   # "pfb3x3" (tracking), "paper" or "document"
    step: str     # one of SUB_STEPS
    status: str   # your_turn / processing / waiting_gpu / stopped / done
    reason: str
    action: str | None = None   # stopped: "retry" (RETRY), "reselect" (RESELECT) or "rerun" (RERUN)
    analysis: Analysis | None = None   # how far its analysis has got (analysis.py)

    @property
    def progress_fraction(self) -> float:
        return self.analysis.fraction if self.analysis is not None else 0.0


@dataclass
class WorkProgress:
    assignment_id: int
    title: str
    kind: str
    steps: list[str]
    current: int          # 1-based step number: the earliest step any submission needs
    done: set[int]
    your_turn: bool
    stopped: bool
    reason: str
    subs: list[SubProgress]
    approved: int
    total: int
    complete: bool
    keys: list[str] = field(default_factory=list)   # step key of each step
    furthest: int = 0              # the furthest step any submission has reached
    turn_step: int | None = None   # the lowest step where it is the teacher's turn
    domain: str = "school"
    analysis: Analysis | None = None   # the groups' analyses together (analysis.py)
    wording: str = "general"   # the work's profile: the key of its screen words (vocab.WORDS)

    @property
    def progress_fraction(self) -> float:
        return self.analysis.fraction if self.analysis is not None else 0.0

    def number(self, key: str) -> int:
        """The 1-based step number of a step key."""
        return self.keys.index(key) + 1

    def reachable(self, step: int) -> bool:
        """Steps up to the furthest one any submission has reached (and done ones) can be
        opened; the PDF step as soon as one student is approved."""
        if not 1 <= step <= len(self.keys):
            return False
        if step <= max(self.current, self.furthest) or step in self.done:
            return True
        return self.keys[step - 1] == "pdf" and self.approved > 0


# --- a per-call, read-only view of one assignment ---

class _Snapshot(Repo):
    """The assignment's submissions, subjects, runs and jobs, read once (in one read
    transaction) and served from memory; each run is decoded at most once and each run's
    staleness is checked once. Only the readers the derivation uses (with
    review.subject_status and review.untranslated_behavior) are served from memory; other
    ids fall back to the database. Never written through."""

    def __init__(self, repo: Repo, assignment: dict):
        super().__init__(repo.conn, repo.settings)
        aid = assignment["id"]
        own_tx = not self.conn.in_transaction
        if own_tx:
            self.conn.execute("BEGIN")
        try:
            assignment = super().get_assignment(aid)   # re-read in the snapshot's transaction
            subs = super().list_submissions(aid)
            subjects = [self._subject(r) for r in self.conn.execute(
                "SELECT t.* FROM subjects t JOIN submissions s ON t.submission_id = s.id"
                " WHERE s.assignment_id=? ORDER BY t.id", (aid,))]
            runs = self.conn.execute(
                "SELECT r.* FROM stage_runs r JOIN submissions s ON r.submission_id = s.id"
                " WHERE s.assignment_id=? ORDER BY r.version, r.id", (aid,)).fetchall()
            jobs = [dict(r) for r in self.conn.execute(   # as submission_job_rows
                "SELECT j.* FROM jobs j JOIN submissions s ON j.submission_id = s.id"
                f" WHERE s.assignment_id=? AND j.stage NOT IN ({_SIDE_MARKS})"
                " ORDER BY j.id", (aid, *SIDE_STAGES))]
            # read in the same transaction as the data, so a cache key made from it matches it
            self.signature = (_signature(self, assignment)
                              if assignment.get("pdf_downloaded_at") else None)
        finally:
            if own_tx:
                self.conn.execute("COMMIT")
        self.assignment = assignment
        self.submissions = subs
        self._subs = {s["id"]: s for s in subs}
        self._subjects = {s["id"]: s for s in subjects}
        self._subjects_of = defaultdict(list)
        for s in subjects:
            self._subjects_of[s["submission_id"]].append(s)
        self._rows = {r["id"]: r for r in runs}
        self._decoded: dict[int, StageRun] = {}
        self._stale: dict[int, bool] = {}
        self._keys = defaultdict(list)   # (sid, stage, camera, subject_id) -> ids by version
        self._cameras = defaultdict(set)
        self._last = defaultdict(str)    # sid -> time of its latest activity (ISO, UTC)
        for r in runs:
            sid = r["submission_id"]
            self._keys[(sid, r["stage"], r["camera"], r["subject_id"])].append(r["id"])
            if r["stage"] == Stage.prepare.value and r["camera"] is not None:
                self._cameras[sid].add(r["camera"])
            self._last[sid] = max(self._last[sid], r["created_at"])
        self._jobs = defaultdict(list)
        for j in jobs:
            sid = j["submission_id"]
            self._jobs[sid].append(j)
            self._last[sid] = max(self._last[sid], j["created_at"], j["heartbeat_at"] or "")

    def get_assignment(self, id: int) -> dict:
        return self.assignment if id == self.assignment["id"] else super().get_assignment(id)

    def list_submissions(self, assignment_id: int) -> list[dict]:
        if assignment_id == self.assignment["id"]:
            return list(self.submissions)
        return super().list_submissions(assignment_id)

    def get_submission(self, id: int) -> dict:
        return self._subs[id] if id in self._subs else super().get_submission(id)

    def get_subject(self, id: int) -> dict:
        return self._subjects[id] if id in self._subjects else super().get_subject(id)

    def list_subjects(self, submission_id: int) -> list[dict]:
        if submission_id in self._subs:
            return list(self._subjects_of[submission_id])
        return super().list_subjects(submission_id)

    def get_run(self, run_id: int) -> StageRun | None:
        if run_id not in self._rows:
            return super().get_run(run_id)
        if run_id not in self._decoded:
            self._decoded[run_id] = self._run(self._rows[run_id])
        return self._decoded[run_id]

    def run_history(self, submission_id, stage, camera=None, subject_id=None):
        if submission_id not in self._subs:
            return super().run_history(submission_id, stage, camera, subject_id)
        ids = self._keys.get((submission_id, Stage(stage).value, camera, subject_id), [])
        return [self.get_run(i) for i in ids]

    def latest_run(self, submission_id, stage, camera=None, subject_id=None):
        if submission_id not in self._subs:
            return super().latest_run(submission_id, stage, camera, subject_id)
        ids = self._keys.get((submission_id, Stage(stage).value, camera, subject_id))
        return self.get_run(ids[-1]) if ids else None

    def list_cameras(self, submission_id: int) -> list[str]:
        if submission_id not in self._subs:
            return super().list_cameras(submission_id)
        return sorted(self._cameras[submission_id])

    def is_stale(self, run: StageRun) -> bool:
        if run.id not in self._rows:
            return super().is_stale(run)
        if run.id not in self._stale:
            self._stale[run.id] = super().is_stale(run)   # through the cached readers
        return self._stale[run.id]

    def submission_job_rows(self, submission_id: int) -> list[dict]:
        if submission_id not in self._subs:
            return super().submission_job_rows(submission_id)
        return list(self._jobs[submission_id])

    def active_job_statuses(self, submission_id: int) -> set[str]:
        if submission_id not in self._subs:
            return super().active_job_statuses(submission_id)
        return {j["status"] for j in self._jobs[submission_id] if j["status"] in _ACTIVE}

    def has_active_jobs(self, submission_id: int) -> bool:
        return bool(self.active_job_statuses(submission_id))

    def blocking_job_statuses(self, submission_id: int) -> set[str]:
        if submission_id not in self._subs:
            return super().blocking_job_statuses(submission_id)
        return {j["status"] for j in self._jobs[submission_id]
                if j["status"] in _ACTIVE and j["stage"] not in NON_BLOCKING_STAGES}

    def has_blocking_jobs(self, submission_id: int) -> bool:
        return bool(self.blocking_job_statuses(submission_id))

    def last_activity(self, submission_id: int) -> str:
        return self._last[submission_id]


def snapshot(repo: Repo, assignment_id: int) -> _Snapshot:
    """One read of the work's rows for a page run or a tick; KeyError if it was deleted."""
    if isinstance(repo, _Snapshot) and repo.assignment["id"] == assignment_id:
        return repo
    return _Snapshot(repo, repo.get_assignment(assignment_id))


def contribution_redo_needed(repo: Repo, run: StageRun) -> bool:
    """True when a feedback text cannot be made fresh again from 確認して承認: a contribution it
    was built from, or the latest one of that camera (which a new text would be built from),
    is outdated itself, failed or gone. A plain newer contribution is not enough: making the
    text again then gives a fresh one."""
    for up_id in run.upstream_run_ids:
        up = repo.get_run(up_id)
        if up is None or Stage(up.stage) != Stage.contribution:
            continue
        latest = repo.latest_run(run.submission_id, Stage.contribution, up.camera)
        if not usable(latest) or review.is_stale_chain(repo, latest):
            return True
    return False


# --- reasons ---

def camera_label(text: str) -> str:
    """cam1 -> カメラ1 wherever it appears."""
    return _CAMERA.sub(r"カメラ\1", text or "")


def plain_service(text: str) -> str:
    """A service's message with 人物の追跡 put in the screens' words (人を見つける)."""
    for old, new in _SERVICE_WORDS:
        text = text.replace(old, new)
    return text


def plain_error(error: str | None) -> str:
    """A teacher-facing reason from a job's error: its first line in plain Japanese."""
    lines = [line.strip() for line in (error or "").splitlines() if line.strip()]
    if not lines:
        return "原因がわかりません"
    first = lines[0]
    if "out of memory" in first.lower() or "OutOfMemory" in first:
        return OUT_OF_MEMORY
    if first.startswith("heartbeat timeout"):
        return "処理が途中で応答しなくなりました"
    first = _EXC_PREFIX.sub("", first, count=1)
    for pattern, plain in _PLAIN:
        if pattern.search(first):
            return plain
    if first.startswith(MODEL_NOT_LOADED):   # worker/main.py: keep the first sentence only
        return MODEL_NOT_LOADED + "。"
    if not _JAPANESE.search(first):
        return "予期しないエラーが起きました"
    return camera_label(first)


def _where(camera: str | None) -> str:
    return f"（{camera_label(camera)}）" if camera else ""


def _what(noun: str, camera: str | None) -> str:
    """「（人を見つける・カメラ1）」: which part of the analysis, after a sentence."""
    return f"（{noun}・{camera_label(camera)}）" if camera else f"（{noun}）"


def _noun(stage: str, kind: Kind) -> str:
    return (_DOC_NOUN if kind == Kind.document else {}).get(stage, _NOUN[stage])


def _doing(stage: str, kind: Kind) -> str:
    return (_DOC_DOING if kind == Kind.document else {}).get(stage, _DOING[stage])


def job_line(stage: str, status: str, camera: str | None, kind: Kind | str) -> str:
    """What one active job of a stage is doing, in plain words: 「人を見つけています（カメラ1）」
    while it runs, else 「順番を待っています（人を見つける・カメラ1）」 (queued, or waiting for
    the GPU: then it also says that it starts by itself)."""
    kind = Kind(kind)
    if status == JobStatus.running.value:
        return _doing(stage, kind) + _where(camera)
    if status == JobStatus.waiting_gpu.value:
        return (f"順番を待っています（{_noun(stage, kind)}）。"
                "ほかの解析が終わると自動で始まります")
    return "順番を待っています" + _what(_noun(stage, kind), camera)


# --- one submission ---

@dataclass
class _Sub:
    progress: SubProgress
    approved: int = 0
    total: int = 0
    last_approved_at: str = ""
    settled: bool = False    # every student approved or with too little evidence
    summable: int = 0        # students waiting for approval, when the reason says only that
    reached: str | None = None   # a later step it can already be opened at (see _submission)


def _make(sub: dict, engine: str, step: str, status: str, reason: str,
          action: str | None = None, **kw) -> _Sub:
    return _Sub(SubProgress(sub["id"], sub["label"], engine, step, status, reason, action),
                **kw)


def _job_key(row: dict) -> tuple[int, str]:
    return _ORDER[row["stage"]], row["camera"] or ""


def _step_of(stage: str, kind: Kind) -> str:
    return "people" if kind == Kind.video and stage in _PEOPLE_STAGES else "wait"


def _processing(sub: dict, engine: str, kind: Kind, step: str, active: list[dict]) -> _Sub:
    running = sorted((r for r in active if r["status"] == JobStatus.running.value), key=_job_key)
    waiting = sorted((r for r in active if r["status"] == JobStatus.waiting_gpu.value),
                     key=_job_key)
    if running:
        row, status = running[0], "processing"
    elif waiting:
        row, status = waiting[0], "waiting_gpu"
    else:
        row, status = sorted(active, key=_job_key)[0], "processing"
    reason = job_line(row["stage"], row["status"], row["camera"], kind)
    if row["stage"] == Stage.pfb.value:
        n = sum(1 for r in active if r["stage"] == Stage.pfb.value)
        if n > 1:
            reason += f"（{n}人分）"
    return _make(sub, engine, step, status, reason)


class _Ctx:
    """What one submission's derivation needs besides its own rows."""

    def __init__(self, repo: _Snapshot, assignment: dict, now: datetime):
        self.repo, self.assignment, self.now = repo, assignment, now
        self.kind = Kind(assignment["kind"])

    def stuck(self, sub: dict, engine: str, step: str, reason: str,
              action: str | None = None, **kw) -> _Sub:
        """Nothing is running yet the next result is missing: processing within the grace
        period after the last activity (follow-ups are about to be queued), else stopped
        (with `action`, if any)."""
        last = self.repo.last_activity(sub["id"])
        if last and (self.now - datetime.fromisoformat(last)).total_seconds() < GRACE_SEC:
            return _make(sub, engine, step, "processing", "次の処理を準備しています", **kw)
        return _make(sub, engine, step, "stopped", reason, action, **kw)


def _tracking_people(ctx: _Ctx, sub: dict, engine: str) -> _Sub:
    """Tracking engine, nothing running before the mapping, not confirmed yet. Step 3 waits
    for the coach only when a camera has no current mapping (its automatic selection failed)
    or the coach has saved a mapping of their own; when every camera has its automatic people,
    the work is past step 3 (they are confirmed automatically) and stops with 「選び直す」 if
    that never happens or nobody is evaluated."""
    repo, sid = ctx.repo, sub["id"]
    cameras = list(sub["meta"].get("cameras") or {})
    if not cameras:
        return _make(sub, engine, "people", "stopped",
                     "カメラの動画が登録されていません。管理用の画面で確かめてください")
    need, people, auto = [], 0, 0
    for cam in cameras:
        track = repo.latest_run(sid, Stage.track, cam)
        if not usable(track):
            return ctx.stuck(sub, engine, "people", f"人を見つけた結果がありません{_where(cam)}。"
                             "管理用の画面でやり直してください")
        if mapping_is_current(repo, sid, cam):   # not before a retrack or a roster change
            auto += repo.latest_run(sid, Stage.identity, cam).source == AUTO_SOURCE
            continue
        need.append(cam)
        people += len(track.output.get("labels") or [])
    if need:
        where = ("（" + "、".join(camera_label(c) for c in need) + "）") if len(cameras) > 1 else ""
        return _make(sub, engine, "people", "your_turn",
                     f"{people}人の名前を選ぶのを待っています{where}")
    if auto == len(cameras):
        if not any(is_evaluated(s) for s in repo.list_subjects(sid)):
            return _nobody(ctx, sub, engine)
        return ctx.stuck(sub, engine, "wait", NOT_STARTED, "reselect")
    return _make(sub, engine, "people", "your_turn", "選んだ名前の決定を待っています")


def _nobody_tracked(repo: _Snapshot, sub: dict) -> bool:
    """Every registered camera's latest track run is usable and has no labels."""
    cameras = list(sub["meta"].get("cameras") or {})
    if not cameras:
        return False
    for cam in cameras:
        track = repo.latest_run(sub["id"], Stage.track, cam)
        if not usable(track) or (track.output or {}).get("labels"):
            return False
    return True


def _nobody(ctx: _Ctx, sub: dict, engine: str, **kw) -> _Sub:
    """Tracking engine, nobody evaluated: 「選び直す」, unless nobody was tracked at all (then
    only 管理用の画面 can help: the tracking has to be redone)."""
    if _nobody_tracked(ctx.repo, sub):
        return ctx.stuck(sub, engine, "wait", NO_ONE_TRACKED, **kw)
    return ctx.stuck(sub, engine, "wait", NOBODY, "reselect", **kw)


def _no_events(ctx: _Ctx, sub: dict, subjects: list[dict]) -> bool:
    """Every registered camera's behavior run is usable and current, and no event of any of
    them belongs to an evaluated person (`subjects`)."""
    repo, sid = ctx.repo, sub["id"]
    cameras = list(sub["meta"].get("cameras") or {})
    pids = {s["participant_id"] for s in subjects}
    if not cameras:
        return False
    for cam in cameras:
        run = repo.latest_run(sid, Stage.behavior, cam)
        if not usable(run) or review.is_stale_chain(repo, run):
            return False
        events = (run.output or {}).get("events") or []
        if any(isinstance(e, dict) and e.get("participant") in pids for e in events):
            return False
    return True


def _paper_identity_missing(ctx: _Ctx, sub: dict, engine: str) -> _Sub | None:
    for cam in sub["meta"].get("cameras") or {}:
        if not usable(ctx.repo.latest_run(sub["id"], Stage.identity, cam)):
            return ctx.stuck(sub, engine, "people",
                             f"参加者の特定結果がありません{_where(cam)}。"
                             "管理用の画面で確かめてください")
    return None


def _students(ctx: _Ctx, sub: dict) -> tuple[list[dict], dict, dict]:
    """(the students, {id: status}, the counts _Sub carries: approved, total and the time of
    the latest approval)."""
    repo, kind = ctx.repo, ctx.kind
    subjects = [s for s in repo.list_subjects(sub["id"])
                if kind == Kind.document or is_evaluated(s)]
    statuses = {s["id"]: review.subject_status(repo, s["id"]) for s in subjects}
    approved = [i for i, st in statuses.items() if st == SubjectStatus.approved]
    counts = dict(approved=len(approved), total=len(subjects),
                  last_approved_at=max((s["approved_at"] or "" for s in subjects
                                        if s["id"] in approved), default=""))
    return subjects, statuses, counts


def _short_waiting(repo: Repo, subjects: list[dict], statuses: dict) -> int:
    """How many 根拠不足 people have a short text waiting for 「この文で承認する」 (step 5,
    services.short_text): they are the coach's turn, not settled (UX round 2, L6)."""
    from pfb_app.services import short_text   # noqa: PLC0415 (kept local: one self-contained hunk)
    return sum(1 for s in subjects if statuses[s["id"]] == SubjectStatus.insufficient
               and short_text.offer(repo, s, repo.latest_run(s["submission_id"], Stage.pfb,
                                                             subject_id=s["id"])) is not None)


def _results(ctx: _Ctx, sub: dict, engine: str) -> _Sub:
    """Nothing is running: the students' feedback decides (review, settled or stuck)."""
    repo, kind = ctx.repo, ctx.kind
    subjects, statuses, counts = _students(ctx, sub)
    names = {s["id"]: s["display_name"] for s in subjects}
    failed = [i for i, st in statuses.items() if st == SubjectStatus.failed]
    if failed:
        return _make(sub, engine, "wait", "stopped",
                     f"{names[failed[0]]}さんの文章を作れませんでした。管理用の画面で直してください",
                     **counts)
    if engine == "pfb3x3" and subjects and _no_events(ctx, sub, subjects) and all(
            st == SubjectStatus.insufficient for st in statuses.values()):
        # nobody's behavior was recorded, so every feedback says too little: record again
        return ctx.stuck(sub, engine, "wait", NO_EVENTS, "rerun", **counts)
    if any(st == SubjectStatus.pending for st in statuses.values()) or not subjects:
        if engine == "pfb3x3":   # (confirmed) nothing will come of waiting
            if not subjects:
                return _nobody(ctx, sub, engine, **counts)
            if _no_events(ctx, sub, subjects):
                return ctx.stuck(sub, engine, "wait", NO_EVENTS, "rerun", **counts)
        return ctx.stuck(sub, engine, "wait",
                         "解析が最後まで進んでいません。管理用の画面で確かめてください",
                         **counts)
    in_review = [i for i, st in statuses.items() if st == SubjectStatus.review]
    if in_review:
        n = len(in_review)
        runs = [repo.latest_run(sub["id"], Stage.pfb, subject_id=i) for i in in_review]
        stale = [r for r in runs if review.is_stale_chain(repo, r)]
        if kind == Kind.video and review.untranslated_behavior(repo, sub["id"]):
            reason, summable = ("したことの書き出しの一部が英語のままです。"
                                "管理用の画面で直してから承認してください"), 0
        elif any(contribution_redo_needed(repo, r) for r in stale):
            reason, summable = REDO_CONTRIBUTION, 0
        elif stale:
            reason, summable = f"{n}人の承認を待っています（内容が古くなった人がいます）", 0
        else:
            reason, summable = f"{n}人の承認を待っています", n
        return _make(sub, engine, "review", "your_turn", reason, summable=summable, **counts)
    short = _short_waiting(repo, subjects, statuses)
    if short:
        return _make(sub, engine, "review", "your_turn", f"{short}人の短い文が承認を待っています",
                     **counts)
    return _make(sub, engine, "pdf", "your_turn", "参加者に渡すPDFを受け取れます", settled=True,
                 **counts)


# some texts are ready while the others are still written (per-person results, audit #4)
READY_WHILE_WRITING = "{n}人を確かめられます（ほかの{m}人の文章を作っています）"


def _ready_while_writing(ctx: _Ctx, sub: dict, engine: str, active: list[dict]) -> _Sub | None:
    """Only feedback texts are being written: when someone's text is already there to check
    (current, generated or offered as a short text, not approved, its own job not active),
    the group is the coach's turn on step 5 while the rest are written. None when nobody is
    ready yet, or when records left in English hold every approval (the analysis then shows
    as before and step 5 can still be opened)."""
    from pfb_app.services import short_text   # noqa: PLC0415 (kept local: one self-contained hunk)
    repo = ctx.repo
    if ctx.kind == Kind.video and review.untranslated_behavior(repo, sub["id"]):
        return None
    writing = {r["subject_id"] for r in active if r["stage"] == Stage.pfb.value}
    subjects = [s for s in repo.list_subjects(sub["id"])
                if ctx.kind == Kind.document or is_evaluated(s)]
    ready = 0
    for s in subjects:
        if s["id"] in writing:
            continue
        run = repo.latest_run(sub["id"], Stage.pfb, subject_id=s["id"])
        if not usable(run) or review.is_stale_chain(repo, run):
            continue
        if s["approval_status"] == "approved" and s["approved_run_id"] == run.id:
            continue
        if (run.output or {}).get("pfb_status") == "generated" \
                or short_text.offer(repo, s, run) is not None:
            ready += 1
    if not ready:
        return None
    _, _, counts = _students(ctx, sub)
    others = sum(1 for s in subjects if s["id"] in writing)
    return _make(sub, engine, "review", "your_turn",
                 READY_WHILE_WRITING.format(n=ready, m=others), summable=0, **counts)


def is_held(sub: dict) -> bool:
    """A video group whose content check is not confirmed yet (step 1, the check card)."""
    return (sub.get("meta") or {}).get("check") in CHECK_HELD


def held_submissions(repo: Repo, assignment_id: int) -> list[dict]:
    """The work's groups held for their content check, oldest first (the first one is
    confirmed first)."""
    return sorted((s for s in repo.list_submissions(assignment_id) if is_held(s)),
                  key=lambda s: s["id"])


def _submission(ctx: _Ctx, sub: dict) -> _Sub:
    repo, kind, sid = ctx.repo, ctx.kind, sub["id"]
    engine = "document" if kind == Kind.document else submission_engine(sub)
    rows = repo.submission_job_rows(sid)
    latest: dict[tuple, dict] = {}
    for row in rows:
        latest[(row["stage"], row["camera"], row["subject_id"])] = row
    if kind == Kind.video and is_held(sub):
        # "checking" comes from an active detect job, never from the meta alone
        checking = any(r["stage"] == Stage.detect.value and r["status"] in _ACTIVE
                       for r in latest.values())
        if checking:
            return _make(sub, engine, "assignment", "processing", CHECKING)
        return _make(sub, engine, "assignment", "your_turn", CHECKED)
    active = [r for r in latest.values()   # names jobs never hold the work up
              if r["status"] in _ACTIVE and r["stage"] not in NON_BLOCKING_STAGES]
    failures = retry.open_failures(repo, sid, rows)
    if failures:
        row = failures[0]
        what = _what(_noun(row["stage"], kind), row["camera"])
        # give up only when every open failure has failed GIVE_UP_AFTER retries (a camera
        # that failed for the first time was never retried); retry_work has no such guard
        # (a stale tab can still re-queue a given-up job: acceptable)
        if min(retry.times_retried(repo, ctx.assignment["id"], [r["id"]])
               for r in failures) >= retry.GIVE_UP_AFTER:
            reason, action = GIVE_UP + what, None
        else:
            reason, action = plain_error(row["error"]) + what, "retry"
        if row["stage"] == Stage.pfb.value:   # the other students can still be checked
            _, _, counts = _students(ctx, sub)
            return _make(sub, engine, "review", "stopped", reason, action, **counts)
        return _make(sub, engine, _step_of(row["stage"], kind), "stopped", reason, action)
    front = [r for r in active if r["stage"] in _PEOPLE_STAGES]
    if kind == Kind.video and front:   # reading the videos, tracking or identifying people
        return _processing(sub, engine, kind, "people", front)
    if kind == Kind.video and not sub["identity_confirmed"]:
        if engine == "pfb3x3":
            return _tracking_people(ctx, sub, engine)
        missing = _paper_identity_missing(ctx, sub, engine)
        if missing is not None:
            return missing
        # with the gate the analysis waits for the confirmation; without it, it goes on and
        # the confirmation (needed before any approval) is asked for when it has finished
        if ctx.assignment["require_identity_approval"] or not active:
            return _make(sub, engine, "people", "your_turn", "参加者の特定の確認を待っています")
    if active:
        if all(r["stage"] == Stage.pfb.value for r in active):
            partial = _ready_while_writing(ctx, sub, engine, active)
            if partial is not None:   # some texts are there: check them while the rest come
                return partial
        busy = _processing(sub, engine, kind, "wait", active)
        if all(r["stage"] == Stage.pfb.value for r in active):
            busy.reached = "review"   # only feedback is being written: the others can be checked
            _, _, counts = _students(ctx, sub)    # and the approvals so far still count
            busy.approved, busy.total = counts["approved"], counts["total"]
            busy.last_approved_at = counts["last_approved_at"]
        return busy
    return _results(ctx, sub, engine)


# --- the work ---

def step_labels(kind: str, engine: str, domain: str) -> list[str]:
    """The step names of a work in its words: `domain` is a domain value or a profile key
    (step 1 reads 試合・練習を決める for sports, 会議を決める for a meeting, …)."""
    if kind == Kind.document.value:
        labels = list(DOCUMENT_STEPS)
    elif engine == "paper":
        labels = list(PAPER_STEPS)
    else:
        labels = list(TRACKING_STEPS)
    if kind == Kind.video.value:
        labels[0] = f"{words(domain).work}を決める"
        if engine == "paper":
            labels[2] = f"{words(domain).people}を確認"
    return labels


def _steps(kind: Kind, subs: list[_Sub], wording: str = "general") -> list[str]:
    paper = bool(subs) and all(s.progress.engine == "paper" for s in subs)
    return step_labels(kind.value, "paper" if paper else "tracking", wording)


def _work_reason(subs: list[_Sub], keys: list[str]) -> str:
    """The home line: what blocks (stopped), else what waits for the teacher (earliest step),
    else what is being processed. Plain approval waits add up; a wait with a note (stale or
    untranslated records) is named instead of being summed away."""
    def prefixed(chosen: list[_Sub]) -> str:
        first = chosen[0].progress
        text = f"{first.label}: {first.reason}" if len(subs) > 1 else first.reason
        return text + (f"（ほか{len(chosen) - 1}件）" if len(chosen) > 1 else "")

    for status in ("stopped", "your_turn", "processing", "waiting_gpu"):
        chosen = [s for s in subs if s.progress.status == status]
        if not chosen:
            continue
        first_step = min(keys.index(s.progress.step) for s in chosen)
        chosen = [s for s in chosen if keys.index(s.progress.step) == first_step]
        if status == "your_turn" and chosen[0].progress.step == "review":
            noted = [s for s in chosen if not s.summable]
            if noted:
                return prefixed(noted + [s for s in chosen if s.summable])
            if len(chosen) > 1:
                return f"{sum(s.summable for s in chosen)}人の承認を待っています"
        if status == "your_turn" and all(s.settled for s in chosen):
            return chosen[0].progress.reason
        return prefixed(chosen)
    return "完了しました"


# complete works by (database file, assignment id): (row signature, progress)
_COMPLETE: dict[tuple[str, int], tuple[tuple, WorkProgress]] = {}


def _signature(repo: Repo, assignment: dict) -> tuple | None:
    """Everything the derivation of a work reads, condensed into one query's answer: runs are
    append-only (count and last id), jobs and approvals change in place (their states are
    listed). None for an in-memory database."""
    path = next((r[2] for r in repo.conn.execute("PRAGMA database_list") if r[1] == "main"), "")
    if not path:
        return None
    join = "JOIN submissions s ON x.submission_id = s.id WHERE s.assignment_id = :a"
    row = repo.conn.execute(f"""SELECT
        (SELECT COUNT(*) || ':' || COALESCE(MAX(id), 0) || ':' || COALESCE(group_concat(
                id || '=' || identity_confirmed || '=' || label || '=' || meta, '|'), '')
           FROM submissions WHERE assignment_id = :a),
        (SELECT COUNT(*) || ':' || COALESCE(MAX(x.id), 0) FROM stage_runs x {join}),
        (SELECT COUNT(*) || ':' || COALESCE(MAX(x.id), 0) || ':' || COALESCE(group_concat(
                CASE WHEN x.status != 'done' THEN x.id || '=' || x.status END, '|'), '')
           FROM jobs x {join}),
        (SELECT COUNT(*) || ':' || COALESCE(group_concat(x.id || '=' || x.approval_status
                || '=' || COALESCE(x.approved_run_id, '') || '=' || COALESCE(x.approved_at, '')
                || '=' || COALESCE(x.participant_id, '') || '=' || x.excluded
                || '=' || x.display_name, '|'), '')
           FROM subjects x {join})""", {"a": assignment["id"]}).fetchone()
    return (path, tuple(sorted(assignment.items())), tuple(row))


def work_progress(repo: Repo, assignment_id: int, now: datetime | None = None) -> WorkProgress:
    """The work's progress; `now` (UTC) is only for the grace period and defaults to now.
    `repo` may be the page run's `snapshot` of this work, which is then read, not re-read.

    A complete work is answered from the cache while its rows are unchanged (only works with a
    PDF download can be complete)."""
    assignment = repo.get_assignment(assignment_id)
    given = repo if isinstance(repo, _Snapshot) and repo.assignment["id"] == assignment_id else None
    if given is not None:
        signature = given.signature      # None (no PDF yet, or none read): not cached
    else:
        signature = _signature(repo, assignment) if assignment.get("pdf_downloaded_at") else None
    if signature is not None:
        hit = _COMPLETE.get((signature[0], assignment_id))
        if hit is not None and hit[0] == signature:
            return copy.deepcopy(hit[1])
    snap = given if given is not None else _Snapshot(repo, assignment)
    progress = _derive(snap, assignment, now or datetime.now(timezone.utc))
    # the key is the signature read with the data (never one read before or after it)
    stored = snap.signature
    if stored is not None and progress.complete:
        _COMPLETE[(stored[0], assignment_id)] = (stored, copy.deepcopy(progress))
    return progress


def _derive(repo: _Snapshot, assignment: dict, now: datetime) -> WorkProgress:
    ctx = _Ctx(repo, assignment, now)
    kind = ctx.kind
    keys = list(DOCUMENT_KEYS if kind == Kind.document else VIDEO_KEYS)
    subs = [_submission(ctx, s) for s in repo.submissions]
    domain = assignment.get("domain") or "school"
    wording = profiles.of_work(assignment)
    base = dict(assignment_id=assignment["id"], title=assignment["title"], kind=kind.value,
                steps=_steps(kind, subs, wording), keys=keys, domain=domain, wording=wording)
    if not subs:
        what = "書類" if kind == Kind.document else "動画"
        files_step = keys.index("files") + 1
        return WorkProgress(**base, current=files_step, done={1}, your_turn=True,
                            stopped=False, reason=f"{what}を入れてください。", subs=[],
                            approved=0, total=0, complete=False, furthest=files_step,
                            turn_step=files_step)
    approved = sum(s.approved for s in subs)
    total = sum(s.total for s in subs)
    settled = all(s.settled for s in subs)
    if settled and approved == 0:   # only students with too little evidence: write by hand
        for s in subs:
            s.settled = False
            s.progress.step, s.progress.status = "review", "your_turn"
            s.progress.reason = "映っている場面が少ない参加者がいます。確かめて承認できます"
    downloaded_at = assignment.get("pdf_downloaded_at") or ""
    last_approval = max(s.last_approved_at for s in subs)
    complete = (settled and approved > 0 and bool(downloaded_at)
                and downloaded_at >= last_approval)
    for s in subs:
        if complete:
            s.progress.status, s.progress.reason = "done", "完了しました"
        elif (s.settled and s.approved > 0 and downloaded_at
              and downloaded_at >= s.last_approved_at):   # its PDF is out; others still run
            s.progress.status, s.progress.reason = "done", "PDFを受け取りました"
    analysis = _analyse(ctx, subs)
    numbers = [keys.index(s.progress.step) + 1 for s in subs]
    current = min(numbers)
    turns = [n for n, s in zip(numbers, subs) if s.progress.status == "your_turn"]
    done = set(range(1, current)) | ({current} if complete else set())
    if kind == Kind.video and any("check" in (s.get("meta") or {}) for s in repo.submissions):
        # started from the drop zone: the videos are in (step 2 is done) from the start
        videos = sum(len((s.get("meta") or {}).get("cameras") or {}) for s in repo.submissions)
        base["steps"][:2] = [CHECK_FIRST, CHECK_VIDEOS.format(n=videos)]
        done.add(keys.index("files") + 1)
    return WorkProgress(
        **base, current=current, done=done, your_turn=bool(turns),
        stopped=any(s.progress.status == "stopped" for s in subs),
        reason=_work_reason(subs, keys), subs=[s.progress for s in subs],
        approved=approved, total=total, complete=complete,
        furthest=max(numbers + [keys.index(s.reached) + 1 for s in subs if s.reached]),
        turn_step=min(turns) if turns else None, analysis=analysis)


def _approvals(ctx: _Ctx, sub: dict) -> tuple[int, int]:
    """(approved, evaluated) people of a group, whatever its step: approved as
    review.subject_status says (the approved text is its latest one and is current)."""
    repo, sid = ctx.repo, sub["id"]
    subjects = [s for s in repo.list_subjects(sid)
                if ctx.kind == Kind.document or is_evaluated(s)]
    approved = 0
    for s in subjects:
        if s["approval_status"] != "approved":
            continue
        run = repo.latest_run(sid, Stage.pfb, subject_id=s["id"])
        if (run is not None and run.id == s["approved_run_id"]
                and not review.is_stale_chain(repo, run)):
            approved += 1
    return approved, len(subjects)


def _analyse(ctx: _Ctx, subs: list[_Sub]):
    """Each group's analysis (set on its SubProgress) and the work's; the durations of each
    method are read once (one cached query), the weights made per group (its targets)."""
    durations: dict[str, dict[str, list[float]]] = {}
    parts, approved, total = [], 0, 0
    for s, raw in zip(subs, ctx.repo.submissions):
        p = s.progress
        if p.engine not in durations:
            durations[p.engine] = ctx.repo.stage_durations_all(p.engine)
        weights = stage_weights(durations[p.engine], PIPELINES[p.engine],
                                weight_counts(target_counts(ctx.repo, raw, ctx.kind), p.engine))
        done, n = _approvals(ctx, raw)
        p.analysis = submission_analysis(ctx.repo, raw, ctx.kind, p.engine, weights,
                                         status=p.status, reason=p.reason, approved=done,
                                         total=n, step=p.step)
        parts.append((p.label, p.analysis))
        approved, total = approved + done, total + n
    return work_analysis(parts, approved, total)


def _rank(p: WorkProgress) -> int:
    if p.your_turn:
        return 0
    if p.stopped:
        return 1
    return 3 if p.complete else 2


class View:
    """What one page run, or one tick of a live part, reads of a work: one `snapshot` of its
    rows (`repo`) and its progress, derived from that snapshot when first asked for (or the
    router's own on a page run). Read-only: writes go through the real Repo."""

    def __init__(self, repo: _Snapshot, progress: WorkProgress | None = None):
        self.repo = repo
        self._progress = progress

    @classmethod
    def load(cls, repo: Repo, assignment_id: int) -> "View":
        return cls(snapshot(repo, assignment_id))

    @property
    def assignment_id(self) -> int:
        return self.repo.assignment["id"]

    @property
    def progress(self) -> WorkProgress:
        if self._progress is None:
            self._progress = work_progress(self.repo, self.assignment_id)
        return self._progress


def all_work(repo: Repo, now: datetime | None = None) -> list[WorkProgress]:
    """Every work for the home screen: the teacher's turn first, then stopped, then
    processing, then complete; newest first within each group."""
    works = [work_progress(repo, a["id"], now) for a in repo.list_assignments()]
    return sorted(works, key=lambda p: (_rank(p), -p.assignment_id))
