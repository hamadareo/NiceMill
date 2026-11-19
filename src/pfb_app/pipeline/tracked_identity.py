"""Tracking engine (PFB3x3) identity: the teacher's mapping of tracked people to students.

The track stage numbers the people of one camera (P1..Pn). The teacher maps every label to a
registered student or to 「対象外」 (an onlooker or the teacher, participant_id None). The
mapping is stored as a camera-scoped identity run with source "teacher" whose upstream is the
track run it was made for. The paper engine's IdentityRecord/IdentityOutput are unchanged.
"""
from collections import Counter

from pydantic import Field

from pfb_app.domain.models import Stage, _Strict
from pfb_app.pipeline.common import usable

OUTSIDE = "対象外"   # what the UI shows for participant_id None
LABEL_PATTERN = r"^P\d+$"
# A mapping made for an older track run (the camera was tracked again since).
RETRACKED = "人物の追跡がやり直されています。最新の人物一覧で対応づけを保存してください。"
# A mapping saved before the roster was replaced (services.roster.apply_roster).
ROSTER_RESET = "名簿を入れ直したため、もう一度選んでください。"


def not_current_message(camera: str | None) -> str:
    """Why a camera's mapping cannot be used: it is not tied to the latest track run."""
    return (f"人物と参加者の対応づけが最新の人物の追跡結果に基づいていません（{camera}）。"
            "最新の人物一覧で対応づけを保存し直してください。")


def roster_reset_message(camera: str | None) -> str:
    """Why a camera's mapping cannot be confirmed: the roster was replaced after it was saved."""
    return f"{ROSTER_RESET[:-1]}（{camera}）。"


def roster_floor(meta: dict | None) -> int:
    """The highest stage run id when the roster was last replaced (meta roster_run_floor);
    a mapping run with an id at or below it was saved for the old roster."""
    return int((meta or {}).get("roster_run_floor") or 0)


def roster_was_reset(meta: dict | None, identity) -> bool:
    """The mapping run was saved before the roster was last replaced."""
    return identity is not None and identity.id <= roster_floor(meta)


def mapping_is_current(repo, submission_id: int, camera: str) -> bool:
    """The camera's latest mapping can be used: it is usable, made for the latest usable track
    run, and saved after the roster was last replaced."""
    track = repo.latest_run(submission_id, Stage.track, camera)
    identity = repo.latest_run(submission_id, Stage.identity, camera)
    if not usable(track) or not usable(identity):
        return False
    meta = repo.get_submission(submission_id)["meta"]
    return track.id in mapped_track_ids(identity) and not roster_was_reset(meta, identity)


def student_names(subjects: list[dict]) -> dict[str, str]:
    """{participant_id: display_name} of a submission's students, in registration order."""
    return {s["participant_id"]: s["display_name"] for s in subjects if s["participant_id"]}


class TrackedIdentityRecord(_Strict):
    participant_id: str | None   # None = 対象外
    label: str = Field(pattern=LABEL_PATTERN)
    display_name: str | None
    note: str = ""


class TrackedIdentityOutput(_Strict):
    participants: list[TrackedIdentityRecord]
    labels: list[str]
    track_run_id: int


def mapping_errors(out: TrackedIdentityOutput, track_labels: list[str],
                   students: dict[str, str]) -> list[str]:
    """Japanese errors of a mapping against the track run's labels and the submission's
    students ({participant_id: display_name}). Students without a label are allowed (they
    get no behavior events and so end as insufficient)."""
    labels = [p.label for p in out.participants]
    known = set(track_labels)
    errors = [f"人物の追跡結果にない番号です: {x}" for x in _ordered(set(labels) - known)]
    errors += [f"人物の割り当てがありません: {x}" for x in _ordered(known - set(labels))]
    errors += [f"同じ人物が複数回あります: {x}" for x in _duplicates(labels)]
    pids = [p.participant_id for p in out.participants if p.participant_id is not None]
    errors += [f"このグループに登録されていない参加者です: {x}" for x in _ordered(set(pids) - set(students))]
    errors += [f"同じ参加者が複数の人物に割り当てられています: {students.get(x, x)}"
               for x in _duplicates(pids)]
    if sorted(out.labels) != sorted(track_labels):
        errors.append("人物の番号の一覧が人物の追跡結果と一致しません。")
    return errors


def mapped_track_ids(identity) -> set[int]:
    """The track runs a (teacher) mapping run was made for: its track_run_id and its upstream."""
    ids = set(identity.upstream_run_ids)
    if isinstance(identity.output, dict) and identity.output.get("track_run_id") is not None:
        ids.add(identity.output["track_run_id"])
    return ids


def label_mapping(identity_output: dict) -> dict[str, str | None]:
    """{P-label: participant_id, or None for 対象外} of a stored mapping run's output."""
    return {p["label"]: p["participant_id"] for p in identity_output["participants"]}


def unmapped_students(out: TrackedIdentityOutput, students: dict[str, str]) -> list[str]:
    """Display names of the students no label was mapped to, in registration order."""
    mapped = {p.participant_id for p in out.participants}
    return [name for pid, name in students.items() if pid not in mapped]


def _label_key(label: str):
    return (0, int(label[1:])) if label[1:].isdigit() else (1, label)


def _ordered(items) -> list[str]:
    return sorted(items, key=lambda x: _label_key(x) if isinstance(x, str) else (2, str(x)))


def _duplicates(items: list[str]) -> list[str]:
    return _ordered(i for i, n in Counter(items).items() if n > 1)
