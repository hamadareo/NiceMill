"""Domain enums, model-output schemas, stored forms and record forms."""
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator


class Kind(str, Enum):
    video = "video"
    document = "document"


class Stage(str, Enum):
    detect = "detect"  # content check of a video: scene, sport, people count
    prepare = "prepare"
    transcribe = "transcribe"
    track = "track"  # tracking engine (PFB3x3) only: person tracking per camera
    names = "names"  # roster names for the tracked people
    identity = "identity"
    behavior = "behavior"
    evidence = "evidence"
    contribution = "contribution"
    pfb = "pfb"
    linecall = "linecall"  # 判定支援: line-call evidence per camera (outside the work's analysis)
    # やさしい文章版 (rel8): an approved text reworded for a younger reader, per person; the
    # `camera` column of its jobs and runs holds the reader level ("low", "mid"). A side stage.
    plain = "plain"
    tips = "tips"  # 次にためしてみること: a person's (subject_id) or the group's (none) practice tips


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    waiting_gpu = "waiting_gpu"
    done = "done"
    failed = "failed"
    failed_retryable = "failed_retryable"


class SubjectStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    waiting_gpu = "waiting_gpu"
    review = "review"
    approved = "approved"
    insufficient = "insufficient"
    failed = "failed"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- model output schemas ---

class IdentityRecord(_Strict):
    participant_id: str
    appearance: str
    initial_position: str
    verification_points: list[str]
    identity_confidence: Confidence
    verified: bool
    notes: str


class IdentityOutput(_Strict):
    participants: list[IdentityRecord]


class BehaviorEventDraft(_Strict):
    video_time: str
    participant: str
    observed_behavior: str
    utterance: str = ""
    speaker: str = "unknown"
    evidence_source: Literal["video", "audio", "video+audio"]
    confidence: Confidence


class BehaviorOutput(_Strict):
    events: list[BehaviorEventDraft]


class EvidenceDraft(_Strict):
    page: int
    quote: str
    observed_content: str
    evidence_source: Literal["text", "figure"]
    confidence: Confidence


class EvidenceOutput(_Strict):
    items: list[EvidenceDraft]


class Judgment(_Strict):
    ref_id: str
    contribution_interpretation: str
    is_positive_contribution: bool
    rubric_criteria: list[str] = []


class ContributionOutput(_Strict):
    judgments: list[Judgment]


class PfbOutput(_Strict):
    pfb_status: Literal["generated", "insufficient_positive_evidence"]
    pfb: str | None
    evidence_refs: list[str]

    @model_validator(mode="after")
    def _check_consistency(self) -> "PfbOutput":
        if self.pfb_status == "generated":
            if self.pfb is None or not self.pfb.strip():
                raise ValueError("pfb must be non-empty when pfb_status is 'generated'")
        else:
            if self.pfb is not None:
                raise ValueError("pfb must be None when evidence is insufficient")
            if self.evidence_refs:
                raise ValueError("evidence_refs must be empty when evidence is insufficient")
        return self


class TipDraft(_Strict):
    text: str
    evidence_refs: list[str]


class TipsOutput(_Strict):
    """The model's practice tips for one person (video works): each tip stands on events the
    person's feedback already cites."""
    tips_status: Literal["generated", "insufficient_evidence"]
    tips: list[TipDraft]

    @model_validator(mode="after")
    def _check_consistency(self) -> "TipsOutput":
        if self.tips_status == "generated":
            if not self.tips:
                raise ValueError("tips must not be empty when tips_status is 'generated'")
        elif self.tips:
            raise ValueError("tips must be empty when the evidence is insufficient")
        return self


class TeamTipDraft(_Strict):
    text: str
    people: list[int]   # the numbers (1, 2, ...) of the people's texts the tip stands on


class TeamTipsOutput(_Strict):
    """The model's practice tips for the whole group."""
    tips_status: Literal["generated", "insufficient_evidence"]
    tips: list[TeamTipDraft]

    @model_validator(mode="after")
    def _check_consistency(self) -> "TeamTipsOutput":
        if self.tips_status == "generated":
            if not self.tips:
                raise ValueError("tips must not be empty when tips_status is 'generated'")
        elif self.tips:
            raise ValueError("tips must be empty when the evidence is insufficient")
        return self


# --- stored forms ---

class BehaviorEvent(BehaviorEventDraft):
    event_id: str  # paper: f"{camera}-{n:04d}"; tracking engine: f"{camera}-{window:04d}-{n:02d}"
    camera: str
    task_elapsed: str


class BehaviorRunOutput(_Strict):
    """A stored behavior run. Tracking engine only, keyed by event_id:
    - `frames`: the overlay images the engine sent to the model for the event's window;
    - `source_text`: the engine's own (English) observation the Japanese text was made from;
    - `provenance`: what made the run (engine commit, the engine's prompt hashes and
      settings_sha from its run_meta.json, the translate prompt hash, whether the observations
      were translated and by which model); the run's prompt_hash is a hash over its parts.
    Paper runs have none. Not a model-output schema: the paper model answers BehaviorOutput."""
    events: list[BehaviorEvent]
    frames: dict[str, list[str]] | None = None
    source_text: dict[str, str] | None = None
    provenance: dict[str, Any] | None = None

    def stored(self) -> dict:
        """The JSON form, without the optional maps that are None (paper runs unchanged)."""
        return self.model_dump(mode="json", exclude_none=True)


class TranslationItem(_Strict):
    ref_id: str
    text: str


class TranslationOutput(_Strict):
    """The served model's answer when the engine's English observations are translated."""
    translations: list[TranslationItem]


class EvidenceItem(EvidenceDraft):
    item_id: str  # f"p{page}-{n:03d}"
    quote_verified: bool


# --- record forms ---

class StageRun(BaseModel):
    id: int | None
    submission_id: int
    stage: Stage
    camera: str | None
    subject_id: int | None
    version: int
    source: Literal["model", "teacher", "auto"]
    upstream_run_ids: list[int]
    prompt_hash: str | None
    model_id: str | None
    output: dict | None
    status: Literal["ok", "failed"]
    error: str | None
    warnings: list[str]
    duration_sec: float
    created_at: str


class Job(BaseModel):
    id: int
    submission_id: int
    stage: Stage
    camera: str | None
    subject_id: int | None
    model_key: str  # standard / high / whisper / cpu / track / pfb3x3:<standard|high>
    status: JobStatus
    attempts: int
    heartbeat_at: str | None
    error: str | None
    progress: float | None = None        # a running job's own progress, 0..1 (None: unknown)
    progress_note: str | None = None     # its plain short note, e.g. 「12 / 40 区間」
