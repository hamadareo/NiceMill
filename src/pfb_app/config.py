"""Application settings with defaults that work without any YAML file."""
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, field_validator

_STAGES = ("prepare", "transcribe", "track", "identity", "behavior", "evidence", "contribution", "pfb")


class ModelSpec(BaseModel):
    hf_id: str
    budget_gb: float
    max_model_len: int


def _default_models() -> dict[str, ModelSpec]:
    return {
        "standard": ModelSpec(hf_id="Qwen/Qwen3-VL-8B-Instruct", budget_gb=24, max_model_len=32768),
        "high": ModelSpec(hf_id="Qwen/Qwen3-VL-32B-Instruct-FP8", budget_gb=52, max_model_len=32768),
    }


class Settings(BaseModel):
    model_config = ConfigDict(validate_default=True)

    backend: Literal["transformers", "vllm"] = "transformers"
    host: str = "127.0.0.1"
    port: int = 8501
    data_dir: Path = Path("~/pfb_app/data")
    inbox_dir: Path = Path("~/pfb_app/inbox")
    prompts_dir: Path = Path("~/pfb_app/prompts")
    models: dict[str, ModelSpec] = _default_models()
    whisper_model: str = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    whisper_budget_gb: float = 6
    start_margin_gb: float = 8
    yield_threshold_gb: float = 6
    gpu_poll_sec: int = 180
    session_idle_sec: int = 300
    stale_job_sec: int = 900
    frame_interval_sec: int = 5
    frame_size: tuple[int, int] = (854, 480)
    frames_per_chunk: int = 5
    identity_sample_points: int = 3
    doc_page_image_long_edge: int = 1280
    doc_max_page_images: int = 20
    low_text_chars: int = 50
    max_retries: int = 2
    # kept for old settings files; the server's limit is start.sh's --server.maxUploadSize,
    # and the uploader captions read that (st.get_option), not this
    max_upload_mb: int = 4096
    max_tokens: dict[str, int] = {stage: 4096 for stage in _STAGES}
    # PFB3x3 tracking engine: a frozen copy with its own venv (scripts/setup_pfb3x3_engine.sh).
    engine_dir: Path = Path("~/pfb_app/engines/pfb3x3")
    engine_python: Path | None = None   # None: <engine_dir>/.venv/bin/python
    tracker_python: Path | None = None  # None: the engine venv (it has ultralytics)
    whisper_python: Path | None = None  # None: pfb_app's own .venv python (has faster-whisper)
    track_budget_gb: float = 4
    track_cpu_fallback: bool = False    # True: track on the CPU once the GPU stayed full for
    track_cpu_after_sec: int = 20 * 60  # this long; False: track waits for the GPU
    # pfb3x3:<key> sessions: the engine's vLLM server gets budget/total of the GPU
    # (8B needs ~0.33 x 97.9 GB, 32B ~0.58 x 97.9 GB, plus headroom).
    pfb3x3_budget_gb: dict[str, float] = {"standard": 34, "high": 60}
    pfb3x3_condition: Literal["clip_som", "static_som", "static_ref"] = "clip_som"
    # the worker's shared server may take more than its budget when the GPU has room: up to
    # this many GB per key (None: exactly the budget, as before), leaving track_budget_gb +
    # start_margin_gb free beside it (worker.shared_server). The GPU guard still yields.
    pfb3x3_max_budget_gb: dict[str, float] = {}
    # a pfb3x3 session runs this many jobs at once (pfb, contribution, behavior, names; 1: one
    # at a time, pfb in parallel as before), at most served_engine_jobs engine commands
    served_parallel_jobs: int = 8
    served_engine_jobs: int = 2
    # requests of one contribution / pfb / translation batch sent to the server at once
    served_concurrency: int = 8
    # the engine's behavior requests in flight (None: the engine's default, 16)
    pfb3x3_behavior_concurrency: int | None = None
    # frames per YOLO call in the engine's tracker (PFB_TRACK_BATCH; None: the engine's 8)
    track_batch: int | None = None
    pfb3x3_port: int = 8200
    # sports works: read every person's shirt / car number (the names stage's numbers cue, on
    # the GPU) and name the automatic people 「#10」 / 「10号車」, whatever the cue gate says
    # (the gate still decides whether numbers propose roster rows). False: the gate alone.
    number_names: bool = True
    # games of people: the numbers cue also reads each person's number every this many
    # seconds of their track (the number timeline): moments when their box was on another
    # player are left out of their evidence (pipeline.identity_check). 0: off.
    number_timeline_s: float = 6.0
    # detect (the content check of a dropped video): frames sampled per camera, and the served
    # model of its scene verdict (its GPU budget is pfb3x3_budget_gb[detect_model_key]; without
    # room the check runs detector-only on the CPU and the coach picks the scene).
    detect_samples: int = 8
    detect_model_key: str = "standard"
    detect_timeout_sec: int = 600   # each probe / scene call and the server start of a check
    engine_timeout_sec: int = 6 * 3600
    # --- operations (pfb_app.ops) ---
    # free space where data_dir lives: below disk_warn_gb every screen warns; below
    # disk_refuse_gb (after the incoming file) new uploads are refused and the worker starts
    # no new step until room is made (deleting works frees it)
    disk_warn_gb: float = 50
    disk_refuse_gb: float = 15
    # nightly SQLite online backup of data/app.db plus a data-dir manifest (ops.backup);
    # None: <data_dir>/../backups (beside data_dir, never inside it)
    backup_dir: Path | None = None
    backup_keep: int = 14            # newest backups kept (older ones are removed)
    # beyond those: the newest backup of each of this many further weeks is kept too (0: off).
    # A backup is therefore at most backup_keep + 7 * backup_keep_weekly days old (ops.backup).
    backup_keep_weekly: int = 4
    # rotating log files (ops.logs); None: <data_dir>/logs
    log_dir: Path | None = None
    log_max_mb: int = 10
    log_keep: int = 5
    # optional operator alert (ops.alerts): a file holding a Discord-style webhook URL (mode
    # 600, outside git). None: alerts go to the log only. The URL itself is never a setting.
    alert_webhook_file: Path | None = None
    alert_repeat_sec: int = 3600     # the same alert is sent at most once per this long
    # one automatic re-queue of a step that stopped for a passing reason (services.auto_retry)
    auto_retry: bool = True
    # GPU memory the engine's own device checks (linecall, live: pfb3x3.engine.gpu_device)
    # always leave to other GPU users, passed as PFB_GPU_RESERVE_MIB (the engine never takes
    # less than 4 GB). 32 is the engine's own default (GPU_RESERVE_MIB 32768).
    engine_gpu_reserve_gb: float = 32
    # 判定支援 (line calls): the engine's linecall job (a GPU session of its own) and live demo
    # linecall_budget_gb + start_margin_gb is the free GPU memory the worker waits for: 31 + 8 =
    # 39 GB covers the engine's own check (engine_gpu_reserve_gb 32 + LINECALL_NEED_MIB 6144 =
    # 38.0 GB free, else exit 3), plus 1 GB so a start is not refused by a small change
    # in between. The worker never asks for less than that (worker.main._budget_gb floors it).
    linecall_budget_gb: float = 31
    linecall_demo_video: Path | None = None   # the live demo's video (read only)
    tid_python: Path | None = None   # None: ~/tid_tracker/venv/bin/python (the TID venv)
    # accounts (services.auth): how long a sign-in lasts in a browser (days, absolute)
    session_days: int = 7
    # open mode (ruling 2026-11-13: 「ログインなしで使えるように」): false opens the app
    # without a sign-in — ui.auth_ui.resolve answers an open administrator (admin of role),
    # so every screen and upload works, and the audit names (0, "open"). The sign-in
    # screens, the auth tables and `pfb-admin` stay as they are: true returns the gate.
    auth_enabled: bool = True
    # ききたいことを聞く (services.ask): the local served model that answers a coach question
    # from one work's records. None: none is set up (the screen says so). Must name this
    # machine (localhost); ask_model None takes the first model the server lists.
    ask_server: str | None = None
    ask_model: str | None = None
    ask_timeout_sec: int = 120

    @field_validator("data_dir", "inbox_dir", "prompts_dir", "engine_dir")
    @classmethod
    def _expand_user(cls, v: Path) -> Path:
        return v.expanduser()

    @field_validator("engine_python", "tracker_python", "whisper_python", "tid_python",
                     "linecall_demo_video", "backup_dir", "log_dir", "alert_webhook_file")
    @classmethod
    def _expand_user_optional(cls, v: Path | None) -> Path | None:
        return None if v is None else v.expanduser()


def load_settings(path: Path | None = None) -> Settings:
    """Load settings; values in the YAML file override the defaults."""
    if path is None:
        return Settings()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Settings(**data)
