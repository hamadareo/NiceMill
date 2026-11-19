"""Shared UI helpers: settings, a per-run repo, labels, assignment picker, flash messages, the
registration glue shared by the detail pages and the wizard (rubric file, roster CSV,
participants table, uploads, and the calls into the intake services), the people-sheet
mapping form data of the tracking method (確認・承認 page and the wizard's step 3), and the
evidence a feedback text cites (the wizard's step 5)."""
import csv
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from pfb_app.config import Settings, load_settings
from pfb_app.domain import vocab
from pfb_app.domain.models import JobStatus, Stage, StageRun, SubjectStatus
from pfb_app.pipeline.common import usable
from pfb_app.pipeline.name_fusion import Proposal, reason_text
from pfb_app.pipeline.tracked_identity import OUTSIDE, mapping_is_current, student_names
from pfb_app.preprocess.documents import DocumentError
from pfb_app.preprocess.frames import VideoError
from pfb_app.services import review
from pfb_app.services.intake import (IntakeReport, ParticipantInput, create_assignment_from_form,
                                     register_documents, register_video_group,
                                     save_uploaded_video)
from pfb_app.services.review import AUTO, OPP, EvidenceFrame
from pfb_app.storage import files
from pfb_app.storage.db import Repo, connect, init_schema
from pfb_app.ui.wizard.progress import camera_label

KIND_LABELS = {"document": "書類", "video": "動画"}
MODEL_LABELS = {"standard": "標準", "high": "高精度", "whisper": "文字起こし", "cpu": "CPU",
                "track": "人物の追跡", "detect": "内容の確認", "names:cpu": "名前の候補づくり",
                "pfb3x3:standard": "PFB3x3（標準）",
                "pfb3x3:high": "PFB3x3（高精度）", "linecall": "判定材料づくり"}
SUBJECT_STATUS_LABELS = {
    SubjectStatus.pending: "未処理", SubjectStatus.processing: "処理中",
    SubjectStatus.waiting_gpu: "GPU待ち", SubjectStatus.review: "確認待ち",
    SubjectStatus.approved: "承認済み", SubjectStatus.insufficient: "根拠不足",
    SubjectStatus.failed: "失敗",
}
JOB_STATUS_LABELS = {
    JobStatus.queued: "待機中", JobStatus.running: "実行中", JobStatus.waiting_gpu: "GPU待ち",
    JobStatus.done: "完了", JobStatus.failed: "失敗", JobStatus.failed_retryable: "失敗（再試行可）",
}
STAGE_LABELS = {
    Stage.detect: "内容の確認", Stage.names: "名前の候補づくり",   # as the wizard says
    Stage.prepare: "前処理", Stage.transcribe: "文字起こし", Stage.track: "人物の追跡",
    Stage.identity: "人物の特定",
    Stage.behavior: "行動の観察", Stage.evidence: "根拠の抽出", Stage.contribution: "貢献の判定",
    Stage.pfb: "PFB", Stage.linecall: "判定材料づくり", Stage.plain: "読み手に合わせた文章づくり",
    Stage.tips: "チームのヒントづくり",
}


def validation_text(error) -> str:
    """The Japanese messages of a pydantic ValidationError raised by the review services."""
    return "\n".join(str(e.get("ctx", {}).get("error", e["msg"])) for e in error.errors())


def get_settings() -> Settings:
    """Settings from the YAML named by PFB_CONFIG, or the defaults."""
    path = os.environ.get("PFB_CONFIG")
    return load_settings(Path(path)) if path else Settings()


def get_repo() -> Repo:
    """A fresh connection per script run: Streamlit reruns scripts in different threads.
    Its audit entries name the signed-in person. Inside a page run, a callback or a fragment
    rerun, the session is re-checked (ui.access): without a valid session of a 管理者 or
    評価する人 (a 見るだけの人, signed out, revoked, demoted, expired) the connection reads only."""
    from pfb_app.ui import access   # noqa: PLC0415
    settings = get_settings()
    conn = connect(files.db_path(settings))
    init_schema(conn)
    repo = Repo(conn, settings)
    repo.actor = access.actor()
    if access.in_page_run() and not access.can_edit():
        access.read_only_guard(conn)
    return repo


def upload_limit_mb() -> int:
    """The server's own size limit per file (server.maxUploadSize, start.sh sets it)."""
    return int(st.get_option("server.maxUploadSize"))


def upload_note(what: str) -> str:
    """The line under a file uploader, in place of Streamlit's English one (hidden by the
    theme): what can be chosen and the server's own size limit (server.maxUploadSize), so it
    is always what the server accepts."""
    return f"{what}を選べます（1ファイル最大 {upload_limit_mb()} MB）。"


def too_large_text(limit_mb: int) -> str:
    """Said in place of Streamlit's English 「File must be … or smaller.」 (or 「… files are
    not allowed.」): the browser refuses such a file before sending it, and the page cannot
    tell which of the two it was."""
    return (f"このファイルは入れられません。1ファイル{limit_mb} MBまでの、"
            "入れられる種類のファイルを選んでください。")


UPLOAD_FAILED = "送れませんでした。通信を確かめて、ファイルを押すともう一度送ります。"
# a failed send whose chip cannot be pressed (Streamlit 1.65 offers no retry there)
UPLOAD_FAILED_AGAIN = "送れませんでした。通信を確かめて、もう一度ファイルを入れてください。"

# A chip of a file in error is marked by the script below (data-pfb-send): "refused" when the
# browser refused it before sending (Streamlit's 「File must be … or smaller.」 and the like),
# "failed" when sending it failed. A clickable chip (role="button", Streamlit's retry) is a
# failed send too. Without the script, any other chip in error counts as refused.
_REFUSED = ('[data-testid="stFileChip"][aria-invalid="true"]:not([role="button"])'
            ':not([data-pfb-send="failed"])')
_FAILED = '[data-testid="stFileChip"][aria-invalid="true"][role="button"]'
_FAILED_AGAIN = '[data-testid="stFileChip"][data-pfb-send="failed"]:not([role="button"])'
_LINE = (" display: block; margin: 0.5rem 0 0; padding: 0.4rem 0.75rem;"
         " border-left: 4px solid var(--pfb-ink); background: var(--pfb-mist);"
         " color: var(--pfb-ink); font-weight: 700; font-size: 0.9375rem; }")
_HIDDEN = (" position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0;"
           " overflow: hidden; clip: rect(0 0 0 0); clip-path: inset(50%);"
           " white-space: nowrap; border: 0; }")
_WATCH_JS = """(() => {
  if (window.__pfbSendWatch) { window.__pfbSendWatch.texts = %(texts)s; return; }
  const watch = window.__pfbSendWatch = {texts: %(texts)s};
  const REFUSED = /must be|not allowed|too small|only one file/i;
  let said = document.getElementById("pfb-send-said");
  if (!said) {
    said = document.createElement("p");
    said.id = "pfb-send-said";
    said.className = "pfb-sr-only";
    said.style.cssText = %(hidden)s;   // hidden on every screen, not only where files go in
    said.setAttribute("aria-live", "polite");
    document.body.appendChild(said);
  }
  const mark = () => {
    document.querySelectorAll('[data-testid="stFileChip"][aria-invalid="true"]:not([data-pfb-send])')
      .forEach((chip) => {
        const alert = chip.querySelector('[role="alert"]');
        const text = alert ? alert.textContent : "";
        const retry = chip.getAttribute("role") === "button";
        const kind = !retry && REFUSED.test(text) ? "refused" : "failed";
        chip.setAttribute("data-pfb-send", kind);
        said.textContent = watch.texts[retry ? "retry" : kind];
        if (alert) alert.setAttribute("aria-hidden", "true");   // said in Japanese instead
      });
    // no chip in error (e.g. a screen without a file field): nothing left to read out
    if (said.textContent && !document.querySelector(
        '[data-testid="stFileChip"][aria-invalid="true"]')) said.textContent = "";
  };
  new MutationObserver(mark).observe(document.body, {subtree: true, childList: true,
    attributes: true, attributeFilter: ["aria-invalid", "role"]});
  mark();
})();"""


def upload_refused_style() -> str:
    """A <style> for st.html: the Japanese line under the files in place of Streamlit's
    English one. Streamlit gives Python no word of these files, so this is CSS (with the
    marks of upload_watch_script):
    - a file the browser refused before sending (over the size limit, or a type not
      accepted): too_large_text;
    - a file whose sending failed: UPLOAD_FAILED.
    The chip's own role="alert" line is only made invisible (not display: none), so a
    screen reader still has it; the script says the Japanese line in an aria-live line
    and only then hides the English one from screen readers."""
    refused = too_large_text(upload_limit_mb()).replace('"', "")
    return ("<style>"
            '[data-testid="stFileChip"][aria-invalid="true"] [role="alert"],'
            f" .pfb-sr-only {{{_HIDDEN}"
            f'[data-testid="stFileChips"]:has({_REFUSED})::after {{ content: "{refused}";'
            f"{_LINE}"
            f'[data-testid="stFileChips"]:has({_FAILED})'
            "::before"
            f' {{ content: "{UPLOAD_FAILED}"; order: 99;{_LINE}'
            f'[data-testid="stFileChips"]:has({_FAILED_AGAIN})::before'
            f' {{ content: "{UPLOAD_FAILED_AGAIN}"; order: 99;{_LINE}'
            "</style>")


def upload_watch_script() -> str:
    """The <script> (st.html with unsafe_allow_javascript) that marks chips in error as
    refused or failed and says the matching Japanese line in an aria-live line."""
    import json   # noqa: PLC0415
    texts = json.dumps({"refused": too_large_text(upload_limit_mb()), "retry": UPLOAD_FAILED,
                        "failed": UPLOAD_FAILED_AGAIN}, ensure_ascii=False)
    # the aria-live line's hiding goes inline: the .pfb-sr-only rule is on upload screens only
    hidden = json.dumps(_HIDDEN.strip().removesuffix("}").strip())
    return f"<script>{_WATCH_JS % {'texts': texts, 'hidden': hidden}}</script>"


def flash(message: str, kind: str = "success", raw: bool = False) -> None:
    """Keep a message across the next rerun; `kind` "error" or "warning" shows it as such.

    The message is shown in the open work's vocabulary (static words only). A message built
    from user-entered text (names, labels) passes `raw=True` and localises its static parts
    itself, so the user's own words are never rewritten."""
    st.session_state["_flash"] = (message, kind, raw)


def show_flash(domain: str | None = None) -> None:
    """`domain` None: the open work's (set by the wizard); the legacy pages pass "school"."""
    item = st.session_state.pop("_flash", None)
    if not item:
        return
    message, kind, raw = (*item, False)[:3] if isinstance(item, tuple) else (item, "success", False)
    if not raw:
        message = vocab.localize(message, domain or st.session_state.get("pfb_domain"))
    {"error": st.error, "warning": st.warning}.get(kind, st.success)(message)


def select_assignment(repo: Repo, kind: str | None = None) -> dict | None:
    """Assignment picker; the choice is remembered across pages."""
    items = [a for a in repo.list_assignments() if kind is None or a["kind"] == kind]
    if not items:
        st.info("活動がありません。先に「活動」ページで作成してください。")
        return None
    ids = [a["id"] for a in items]
    last = st.session_state.get("current_assignment")
    chosen = st.selectbox(
        "活動", ids, index=ids.index(last) if last in ids else 0,
        format_func=lambda i: next(
            f"{a['title']}（{KIND_LABELS[a['kind']]}）" for a in items if a["id"] == i))
    st.session_state["current_assignment"] = chosen
    return next(a for a in items if a["id"] == chosen)


def read_roster(upload) -> list[str]:
    """Names from a roster CSV upload (a 「name」 column); ValueError without one."""
    text = upload.getvalue().decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows or "name" not in rows[0]:
        raise ValueError("名簿のCSVには「name」列が必要です。")
    return [r["name"].strip() for r in rows if r["name"] and r["name"].strip()]


PARTICIPANT_COLUMNS = ("参加者ID", "表示名", "外見")


def position_column(camera: int) -> str:
    return f"初期位置（カメラ{camera}）"


def participants_table(count: int) -> list[dict]:
    """The empty participants table of a video group with `count` cameras."""
    row = {c: "" for c in PARTICIPANT_COLUMNS}
    row.update({position_column(i): "" for i in range(1, count + 1)})
    return [row]


def participants_from_table(table: list[dict], count: int) -> list[ParticipantInput]:
    """The participants of an edited table; blank rows are skipped. ValueError (a message for
    the teacher) when a row has only one of ID and name, or an ID repeats."""
    participants, seen = [], set()
    for row in table:
        pid, name = (row.get("参加者ID") or "").strip(), (row.get("表示名") or "").strip()
        if not pid and not name:
            continue
        if not pid or not name:
            raise ValueError("参加者IDと表示名は両方入力してください。")
        if pid in seen:
            raise ValueError(f"参加者IDが重複しています: {pid}")
        seen.add(pid)
        positions = {f"cam{i}": (row.get(position_column(i)) or "").strip()
                     for i in range(1, count + 1)}
        participants.append(ParticipantInput(pid, name, (row.get("外見") or "").strip(),
                                             {c: v for c, v in positions.items() if v}))
    return participants


# --- registration glue (both the detail pages and the wizard call these) --------------------

log = logging.getLogger(__name__)

UNEXPECTED = "登録できませんでした。もう一度試しても直らないときは、管理者に知らせてください。"


@dataclass
class IntakeFailed(Exception):
    """A registration that did not happen. `message` is for the teacher; `detail` is the raw
    cause (an exception text) for whoever fixes it, or None when the message says it all."""
    message: str
    detail: str | None = None

    def __str__(self) -> str:
        return self.message


def show_intake_error(err: IntakeFailed) -> None:
    """The plain message, and the raw cause folded away."""
    st.error(err.message)
    if err.detail:
        with st.expander("くわしい情報（管理者向け）"):
            st.code(err.detail, language=None)


def engine_ready(settings: Settings) -> bool:
    """The tracking method is set up (scripts/setup_pfb3x3_engine.sh wrote ENGINE_COMMIT).
    Only a file check: engine commands run in the worker, never in the UI process."""
    return (settings.engine_dir / "ENGINE_COMMIT").is_file()


def create_assignment_from_inputs(repo: Repo, settings: Settings, title: str, kind: str,
                                  task_text: str, rubric_text: str, rubric_upload,
                                  model_key: str, require_identity: bool) -> int:
    """Create an assignment from the form values (the rubric file is an upload or None;
    callers check for an empty title or task in their own screen's words). Raises
    IntakeFailed when the rubric file is unreadable."""
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = None
            if rubric_upload is not None:
                path = Path(tmp) / f"rubric{Path(rubric_upload.name).suffix.lower()}"
                path.write_bytes(rubric_upload.getvalue())
            return create_assignment_from_form(repo, title, kind, task_text, rubric_text, path,
                                               model_key, require_identity and kind == "video",
                                               settings)
    except DocumentError as e:
        raise IntakeFailed(f"評価の観点のファイルを読めません: {e}") from e


def register_documents_from_inputs(repo: Repo, settings: Settings, assignment_id: int,
                                   uploads, roster_upload) -> IntakeReport:
    """Register uploaded documents (with an optional roster CSV upload). Raises IntakeFailed
    when the roster cannot be read; unreadable documents are listed in the report."""
    roster = None
    if roster_upload is not None:
        try:
            roster = read_roster(roster_upload)
        except UnicodeDecodeError as e:          # a ValueError too: caught first
            raise IntakeFailed("名簿を読めません。UTF-8で保存したCSVを使ってください。",
                               str(e)) from e
        except ValueError as e:
            raise IntakeFailed(f"名簿を読めません: {e}") from e
    return register_documents(repo, settings, assignment_id,
                              [(f.name, f.getvalue()) for f in uploads], roster)


def register_group_from_picks(repo: Repo, settings: Settings, assignment_id: int, label: str,
                              picks: list, from_inbox: bool, start_sec: float,
                              end_sec: float | None, transcript: str, auto_transcribe: bool,
                              table: list[dict], engine: str, max_people: int | None) -> int:
    """Register one video group from the form: `picks` are inbox paths or browser uploads,
    one per camera, and `table` is the edited participants table. Uploads are saved first
    and removed again when the registration fails. Raises IntakeFailed (nothing registered)."""
    if any(p is None for p in picks):
        raise IntakeFailed("すべてのカメラの動画を選んでください。")
    try:
        participants = participants_from_table(table, len(picks))
    except ValueError as e:
        raise IntakeFailed(str(e)) from e
    saved, cameras = [], {}
    try:
        for i, pick in enumerate(picks, 1):
            if from_inbox:
                cameras[f"cam{i}"] = pick
            else:
                cameras[f"cam{i}"] = path = save_uploaded_video(settings, pick.name, pick)
                saved.append(path)
        return register_video_group(repo, settings, assignment_id, label, cameras, start_sec,
                                    end_sec or None, transcript, auto_transcribe, participants,
                                    engine=engine,
                                    max_people=None if max_people is None else int(max_people))
    except VideoError as e:
        for p in saved:  # uploads copied for a registration that failed
            p.unlink(missing_ok=True)
        raise IntakeFailed(str(e)) from e
    except Exception as e:  # noqa: BLE001 - the service already rolled back; keep the page alive
        for p in saved:
            p.unlink(missing_ok=True)
        log.exception("video group registration failed")
        raise IntakeFailed(UNEXPECTED, f"{type(e).__name__}: {e}") from e


# --- the tracking method's people mapping (確認・承認 page and wizard step 3) -----------------

NEW_PERSON = {"sports": "新しい選手", "school": "新しい参加者"}   # an AUTO option with no plan
PLACE_REASON = reason_text("place", "")   # 「初期位置から推定」: the initial-position rule

@dataclass
class MappingForm:
    """What the mapping form of one camera shows: the latest track run's people sheet and
    labels, the people, and the values to start from.

    `saved` is the stored mapping ({label: participant_id or OUTSIDE}) when it was made for
    this track run and after the last roster replacement (`current`), else empty.

    `prefill` per label, first that applies: `saved` (all labels); the fused `proposals`
    (shown as 推定 with `reasons`); the initial-position rule's suggestion (reason
    「初期位置から推定」); the team defaults:
    - the 「other」 cluster (referees, onlookers): OUTSIDE;
    - the other team (with a team answer): OUTSIDE, or OPP when `evaluate_opponents`;
    - the own team (every label without a team answer): AUTO when the work has no people of
      its own (no roster), else unassigned (missing), or AUTO (「選手（番号なし）n」) when
      `evaluate_opponents`.
    `suggested`: nothing is saved and something was proposed (a proposal or the rule)."""
    track: StageRun
    identity: StageRun | None
    current: bool
    saved: dict[str, str]
    prefill: dict[str, str]
    suggested: bool
    labels: list[str]
    students: dict[str, str]     # {participant_id: display_name}
    options: list[str]           # participant ids, the AUTO / OPP options of `prefill`, OUTSIDE
    proposals: dict[str, Proposal] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    teams: dict[str, str] = field(default_factory=dict)      # {label: T1 | T2 | other}
    team_reason: str = ""
    strips: dict[str, Path] = field(default_factory=dict)    # {T1 | T2: sample strip}
    crops: dict[str, Path] = field(default_factory=dict)     # {label: crop}
    own_team: str | None = None
    evaluate_opponents: bool = False
    names_pending: bool = False   # a names job for this camera is queued or running
    auto_names: dict[str, str] = field(default_factory=dict)  # names_for(prefill)
    domain: str = "school"
    planner: review.NamePlanner | None = None

    @property
    def sheet(self) -> Path | None:
        raw = self.track.output.get("sheet")
        return Path(raw) if raw and Path(raw).is_file() else None

    def name(self, option: str) -> str:
        """A participant's name; an AUTO / OPP option as the default name it gets when the
        prefill is kept (names_for gives the names for other choices)."""
        if option in self.students:
            return self.students[option]
        if option in self.auto_names:
            return self.auto_names[option]
        if review.is_auto_option(option):
            return NEW_PERSON["sports" if self.domain == "sports" else "school"]
        return option

    def options_for(self, label: str) -> list[str]:
        """One person's choices: the people, the person's own AUTO option (and OPP option
        when opponents are evaluated), OUTSIDE."""
        extra = [AUTO + label] + ([OPP + label] if self.evaluate_opponents else [])
        return list(self.students) + extra + [OUTSIDE]

    def names_for(self, chosen: dict[str, str | None]) -> dict[str, str]:
        """{AUTO / OPP option: name} for the choices on screen: the chosen options numbered in
        label order exactly as saving them names them (materialise_choices), then every other
        option of options_for numbered after them."""
        if self.planner is None:
            return {}
        extra = [o for label in self.labels for o in self.options_for(label)]
        return self.planner.names_for(chosen, extra)

    def index(self, label: str) -> int | None:
        value = self.prefill.get(label)
        return self.options.index(value) if value in self.options else None


def _team_default(label: str, teams: dict[str, str], own: set[str] | None, roster: bool,
                  evaluate: bool) -> str | None:
    if teams.get(label) == "other":
        return OUTSIDE
    if own is not None and label not in own:
        return OPP + label if evaluate else OUTSIDE
    if not roster or evaluate:
        return AUTO + label
    return None


def mapping_form(repo: Repo, submission_id: int, camera: str) -> MappingForm | None:
    """The mapping form of a camera; None while it has no usable track run. Opening it fuses
    the names stage's cues (CPU only; nothing is saved)."""
    track = repo.latest_run(submission_id, Stage.track, camera)
    if not usable(track):
        return None
    identity = repo.latest_run(submission_id, Stage.identity, camera)
    submission = repo.get_submission(submission_id)
    meta = submission["meta"] or {}
    # a retrack, or a roster replaced after the mapping was saved, makes it stale (apply_roster)
    current = mapping_is_current(repo, submission_id, camera)
    saved = ({p["label"]: p["participant_id"] or OUTSIDE
              for p in identity.output["participants"]} if current else {})
    defaults = dict(track.output.get("default_names") or {})
    subjects = repo.list_subjects(submission_id)
    students = student_names(subjects)
    labels = list(track.output.get("labels") or [])
    view = review.names_view(repo, submission_id, camera, track, fuse=not saved)
    evaluate = bool(meta.get(review.EVALUATE_OPPONENTS))
    reasons: dict[str, str] = {}
    if saved:
        prefill = saved
    else:
        roster = bool(review.roster_subjects(subjects, meta))
        prefill = {}
        for label in labels:
            if label in view.proposals:
                prefill[label] = view.proposals[label].row
                reasons[label] = view.reasons[label]
            elif label in defaults:
                prefill[label] = defaults[label]
                reasons[label] = PLACE_REASON
            elif (value := _team_default(label, view.teams, view.own_labels, roster,
                                         evaluate)) is not None:
                prefill[label] = value
    shown = sorted((v for v in prefill.values() if review.is_auto_option(v)),
                   key=review.option_sort_key)
    domain = repo.get_assignment(submission["assignment_id"]).get("domain") or "school"
    planner = review.name_planner(domain, subjects, meta)
    form = MappingForm(track=track, identity=identity, current=current, saved=saved,
                       prefill=prefill,
                       suggested=not saved and bool(view.proposals or defaults),
                       labels=labels, students=students,
                       options=list(students) + shown + [OUTSIDE],
                       proposals={} if saved else view.proposals, reasons=reasons,
                       teams=view.teams, team_reason=view.team_reason, strips=view.strips,
                       crops=view.crops, own_team=view.own_team, evaluate_opponents=evaluate,
                       names_pending=view.pending, domain=domain, planner=planner)
    form.auto_names = form.names_for(prefill)
    return form


def mapping_from_choices(chosen: dict[str, str | None]) -> tuple[dict[str, str | None],
                                                                  list[str]]:
    """({label: participant_id, or None for 対象外}, the labels nobody was chosen for)."""
    unset = [label for label, value in chosen.items() if value is None]
    return {label: (None if v == OUTSIDE else v) for label, v in chosen.items()}, unset


# --- the evidence a feedback text cites (wizard step 5) --------------------------------------

MAX_FRAMES = 3


@dataclass
class CitedEvidence:
    """One piece of evidence a PFB cites: when (「01:05」, 「カメラ2 01:05」 or 「3ページ」),
    what the student did or wrote, why it counted, and up to MAX_FRAMES frame images that
    exist (videos only)."""
    ref_id: str
    when: str
    what: str
    why: str
    frames: list[EvidenceFrame]


def cited_evidence(repo: Repo, pfb_run: StageRun | None,
                   max_frames: int = MAX_FRAMES) -> list[CitedEvidence]:
    """The evidence `pfb_run` cites (its evidence_refs), in the order cited, looked up through
    its contribution runs (the judgment: why) and their behavior or evidence runs (what).
    `when` is the video time, or for sports the game clock 「試合時間 MM:SS」."""
    refs = list(((pfb_run.output or {}).get("evidence_refs") or []) if pfb_run else [])
    if not refs:
        return []
    several = len(repo.list_cameras(pfb_run.submission_id)) > 1
    sub = repo.get_submission(pfb_run.submission_id)
    # a game clock only when the coach gave the clip's start time (pfb3x3_bridge.clock_is_known)
    sports = (repo.get_assignment(sub["assignment_id"])["domain"] == "sports"
              and sub["meta"].get("clock_offset_min") is not None)
    found: dict[str, CitedEvidence] = {}
    for con_id in pfb_run.upstream_run_ids:
        con = repo.get_run(con_id)
        if con is None or con.stage != Stage.contribution or not con.output:
            continue
        source = repo.get_run(con.upstream_run_ids[0]) if con.upstream_run_ids else None
        if source is None or not source.output:
            continue
        why = {j["ref_id"]: j.get("contribution_interpretation", "")
               for j in con.output.get("judgments", [])}
        for e in source.output.get("events", []):
            if e["event_id"] in refs and e["event_id"] not in found:
                when = (f"試合時間 {e['task_elapsed']}" if sports and e.get("task_elapsed")
                        else e["video_time"])
                if several:
                    when = f"{camera_label(e['camera'])} {when}"
                shots = [f for f in review.frames_for(repo, source, e) if f.path.is_file()]
                found[e["event_id"]] = CitedEvidence(e["event_id"], when, e["observed_behavior"],
                                                     why.get(e["event_id"], ""),
                                                     shots[:max_frames])
        for i in source.output.get("items", []):
            if i["item_id"] in refs and i["item_id"] not in found:
                found[i["item_id"]] = CitedEvidence(
                    i["item_id"], f"{i['page']}ページ", i.get("quote") or i["observed_content"],
                    why.get(i["item_id"], ""), [])
    return [found[r] for r in refs if r in found]
