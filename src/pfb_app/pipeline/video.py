"""4-stage video pipeline: identity -> behavior -> contribution -> pfb. Each stage saves one StageRun."""
import logging
import re
import time
import unicodedata

from pfb_app.domain import profiles
from pfb_app.domain.models import (BehaviorEvent, BehaviorOutput, ContributionOutput,
                                   IdentityOutput, Judgment, Kind, PfbOutput, Stage, StageRun)
from pfb_app.domain.vocab import SPORT_LABELS
from pfb_app.inference.backend import ChatRequest
from pfb_app.inference.structured import call_structured, call_structured_batch
from pfb_app.pipeline import checks, identity_check
from pfb_app.pipeline.common import (StageContext, assignment_of, fail_run, save_run,
                                     submission_engine, usable)
from pfb_app.pipeline.pfb3x3_bridge import clock_is_known
from pfb_app.pipeline.prompts import build_user_text, load_prompt
from pfb_app.preprocess.frames import Frame, chunk, sample_points
from pfb_app.preprocess.transcript import Segment, slice_transcript

log = logging.getLogger(__name__)

CONTRIBUTION_CHUNK = 60
# The tracking engine observes far more events (e.g. 510 in 408 s of a paper-tower activity video): 60
# judgments of them overran max_tokens (4096) and were cut off mid-JSON, so its chunks are
# smaller. The paper engine keeps CONTRIBUTION_CHUNK.
TRACKED_CONTRIBUTION_CHUNK = 20
NO_TRANSCRIPT = "（この時間帯の文字起こしはありません）"


def _frames_of(prepare: StageRun) -> list[Frame]:
    return [Frame.from_dict(d) for d in prepare.output["frames"]]


def _frame_lines(frames: list[Frame]) -> str:
    return "\n".join(f"フレーム{n}: 時刻 {f.video_time}（活動の経過 {f.task_elapsed}）"
                     for n, f in enumerate(frames, 1))


def _subjects(ctx: StageContext, submission_id: int) -> list[dict]:
    return [s for s in ctx.repo.list_subjects(submission_id) if s["participant_id"]]


def _game_minute(clock: str) -> int | None:
    """The nearest whole minute of a game clock MM:SS (also an older H:MM:SS)."""
    parts = clock.split(":")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    sec = 0
    for p in parts:
        sec = sec * 60 + int(p)
    return int(sec / 60 + 0.5)


FOLD_GAP_S = 10   # the same action recorded again within this many seconds is one run
RUN_NOTE = "\n備考: 同じ行動が続けて記録されたため、1件にまとめています"


def _seconds(clock: str) -> int | None:
    parts = clock.split(":")
    if not parts or not all(p.isdigit() for p in parts):
        return None
    sec = 0
    for p in parts:
        sec = sec * 60 + int(p)
    return sec


def _action_key(text: str) -> str:
    """The recorded action without spacing, width or closing punctuation."""
    return re.sub(r"[\s。、.,，．]+", "", unicodedata.normalize("NFKC", text))


def _fold_runs(events: list[BehaviorEvent]) -> list[tuple[BehaviorEvent, int]]:
    """(event, how many) with runs of the same recorded action merged: the same person's same
    action on the same camera again within FOLD_GAP_S seconds of the run's last record is one
    event (the first of the run; a camera records it again every second). The order of first
    appearance is kept."""
    out: list[list] = []
    last: dict[tuple, list] = {}
    for e in events:
        sec = _seconds(e.video_time)
        key = (e.camera, e.participant, _action_key(e.observed_behavior))
        run = last.get(key)
        if run is not None and sec is not None and run[2] is not None \
                and 0 <= sec - run[2] <= FOLD_GAP_S:
            run[1] += 1
            run[2] = sec
            continue
        run = [e, 1, sec]
        out.append(run)
        last[key] = run
    return [(e, n) for e, n, _ in out]


def _event_minute(e: BehaviorEvent, clock_known: bool) -> int | None:
    """The nearest minute of the event on the clock its line shows: the game clock, or the
    video's time when the game clock is not known."""
    return _game_minute(e.task_elapsed if clock_known else e.video_time)


def _event_lines(events: list[BehaviorEvent], domain: str = "school",
                 clock_known: bool = True) -> str:
    """School: the video time. Sports: the game clock (task_elapsed) and its nearest minute,
    「試合時間: 65:30（66分ごろ）」; without a known game clock (`clock_known` False, see
    `pfb3x3_bridge.clock_is_known`) the video's own time, 「動画の時間: 13:07（約13分ごろ）」
    (「1分未満」 under half a minute), so a clip from the middle of a game is never placed at
    the game's start. Only the tracking engine's bridge puts the coach's clock offset into
    task_elapsed; the paper (VLM) behaviour path never applies it, which is fine because
    sports always uses the tracking engine."""
    def when(e: BehaviorEvent) -> str:
        if domain != "sports":
            return f"時刻: {e.video_time}"
        if not clock_known:
            minute = _game_minute(e.video_time)
            return f"動画の時間: {e.video_time}" + (
                "" if minute is None else "（1分未満）" if minute == 0 else f"（約{minute}分ごろ）")
        minute = _game_minute(e.task_elapsed)
        return f"試合時間: {e.task_elapsed}" + ("" if minute is None else f"（{minute}分ごろ）")
    return "\n\n".join(
        f"ref_id: {e.event_id}\n{when(e)}\n行動: {e.observed_behavior}\n"
        f"発話: {e.utterance or 'なし'}" for e in events)


def _rubric_label(domain: str, profile: str | None = None) -> str:
    """The rubric's data section: sports calls it the coach's 見てほしい観点, a profile with its
    own prompts 評価の観点 (same JSON)."""
    if profile is not None and profile not in profiles.UNCHANGED:
        return "評価の観点"
    return "見てほしい観点" if domain == "sports" else "ルーブリック"


def _work_sections(assignment: dict) -> list[tuple[str, str]]:
    """The data sections that describe the work: general, the task text; sports, the sport and
    the activity line (the work's task text); a profile with its own prompts (会議, 発表, 研修,
    学習活動), the task text, the kind of activity and the categories to look for."""
    profile = profiles.of_work(assignment)
    if profile not in profiles.UNCHANGED:
        return [("活動の内容", assignment["task_text"]),
                ("活動の種類", profiles.label(profile)),
                ("見る観点", "\n".join(profiles.categories(profile)))]
    if assignment.get("domain") != "sports":
        return [("活動の内容", assignment["task_text"])]
    sport = assignment.get("sport")
    return [("競技", SPORT_LABELS.get(sport or "", sport or SPORT_LABELS["other"])),
            ("試合・練習の内容", assignment["task_text"])]


def run_identity(ctx: StageContext, submission_id: int, camera: str) -> StageRun:
    started = time.monotonic()
    prepare = ctx.repo.latest_run(submission_id, Stage.prepare, camera)
    if not usable(prepare):
        return fail_run(ctx, submission_id, Stage.identity,
                        f"フレームの準備結果がありません（{camera}）。先に動画の準備を実行してください。",
                        camera=camera, started=started)
    subjects = _subjects(ctx, submission_id)
    points = sample_points(_frames_of(prepare), ctx.settings.identity_sample_points)
    prompt = load_prompt(ctx.settings, Kind.video, Stage.identity)
    meta = "\n\n".join(
        f"参加者ID: {s['participant_id']}\n表示名: {s['display_name']}\n"
        f"服装などの特徴: {s['appearance'] or '未入力'}\n"
        f"初期位置（{camera}）: {s['initial_positions'].get(camera, '未入力')}"
        for s in subjects)
    req = ChatRequest(
        system=prompt.text,
        user_text=build_user_text([("識別メタデータ", meta), ("カメラ", camera),
                                   ("フレーム", _frame_lines(points))]),
        image_paths=[f.path for f in points], json_schema=IdentityOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens[Stage.identity.value])
    expected = {s["participant_id"] for s in subjects}

    def validate(out: IdentityOutput) -> list[str]:
        got = [p.participant_id for p in out.participants]
        errors = [f"参加者が欠落しています: {i}" for i in sorted(expected - set(got))]
        errors += [f"メタデータにない参加者IDです: {i}" for i in sorted(set(got) - expected)]
        errors += [f"参加者が重複しています: {i}" for i in sorted({g for g in got if got.count(g) > 1})]
        return errors

    result = call_structured(ctx.backend, ctx.repo, req, IdentityOutput,
                             max_retries=ctx.settings.max_retries,
                             yield_check=ctx.yield_check, validate=validate)
    common = dict(camera=camera, upstream=[prepare.id], prompt_hash=prompt.hash,
                  model_id=ctx.backend.model_id, started=started)
    if result.parsed is None:
        return fail_run(ctx, submission_id, Stage.identity, "\n".join(result.errors), **common)
    return save_run(ctx, submission_id, Stage.identity, output=result.parsed.model_dump(),
                    **common)


def run_behavior(ctx: StageContext, submission_id: int, camera: str) -> StageRun:
    started = time.monotonic()
    prepare = ctx.repo.latest_run(submission_id, Stage.prepare, camera)
    identity = ctx.repo.latest_run(submission_id, Stage.identity, camera)
    if not usable(prepare):
        return fail_run(ctx, submission_id, Stage.behavior,
                        f"フレームの準備結果がありません（{camera}）。先に動画の準備を実行してください。",
                        camera=camera, started=started)
    if not usable(identity):
        return fail_run(ctx, submission_id, Stage.behavior,
                        f"参加者の特定結果がありません（{camera}）。先に参加者の特定を実行してください。",
                        camera=camera, started=started)
    transcribe = ctx.repo.latest_run(submission_id, Stage.transcribe, None)
    segments = ([Segment.from_dict(d) for d in transcribe.output["segments"]]
                if usable(transcribe) else [])
    upstream = [identity.id, prepare.id] + ([transcribe.id] if usable(transcribe) else [])
    warnings = (["文字起こしが失敗しているため、文字起こしなしで処理しました"]
                if transcribe is not None and not usable(transcribe) else [])
    prompt = load_prompt(ctx.settings, Kind.video, Stage.behavior)
    common = dict(camera=camera, upstream=upstream, prompt_hash=prompt.hash,
                  model_id=ctx.backend.model_id, started=started, warnings=warnings)

    participants = IdentityOutput.model_validate(identity.output).participants
    allowed = {p.participant_id for p in participants}
    identity_text = "\n\n".join(
        f"参加者ID: {p.participant_id}\n外見: {p.appearance}\n初期位置: {p.initial_position}"
        for p in participants)
    chunks = chunk(_frames_of(prepare), ctx.settings.frames_per_chunk)
    reqs, validators = [], []
    for frames in chunks:
        start = frames[0].video_sec
        end = frames[-1].video_sec + ctx.settings.frame_interval_sec
        transcript = slice_transcript(segments, start, end) or NO_TRANSCRIPT
        reqs.append(ChatRequest(
            system=prompt.text,
            user_text=build_user_text([("カメラ", camera), ("参加者identity", identity_text),
                                       ("フレーム", _frame_lines(frames)),
                                       ("文字起こし", transcript)]),
            image_paths=[f.path for f in frames], json_schema=BehaviorOutput.model_json_schema(),
            max_tokens=ctx.settings.max_tokens[Stage.behavior.value]))
        validators.append(_behavior_validator(allowed, frames))

    results = call_structured_batch(ctx.backend, ctx.repo, reqs, BehaviorOutput,
                                    max_retries=ctx.settings.max_retries,
                                    yield_check=ctx.yield_check, validate=validators)
    errors = [f"チャンク{n}: {e}" for n, r in enumerate(results, 1) if r.parsed is None
              for e in r.errors]
    if errors:
        return fail_run(ctx, submission_id, Stage.behavior, "\n".join(errors), **common)

    placed = []  # (frame, draft) in chunk order; sort is stable
    for frames, r in zip(chunks, results):
        by_time = {f.video_time: f for f in frames}
        placed += [(by_time[d.video_time], d) for d in r.parsed.events]
    placed.sort(key=lambda fd: fd[0].video_sec)
    events = [BehaviorEvent(**d.model_dump(), event_id=f"{camera}-{n:04d}", camera=camera,
                            task_elapsed=f.task_elapsed).model_dump()
              for n, (f, d) in enumerate(placed, 1)]
    return save_run(ctx, submission_id, Stage.behavior, output={"events": events}, **common)


def _behavior_validator(allowed: set[str], frames: list[Frame]):
    times = {f.video_time for f in frames}

    def validate(out: BehaviorOutput) -> list[str]:
        errors = checks.unknown_participant_errors(out.events, allowed)
        errors += [f"このチャンクのフレームにない時刻です: {e.video_time}"
                   for e in out.events if e.video_time not in times]
        return errors

    return validate


def _events_of(run: StageRun) -> list[BehaviorEvent]:
    return [BehaviorEvent.model_validate(d) for d in run.output["events"]]


def contribution_chunk(submission: dict) -> int:
    """Events per contribution request for this submission's engine."""
    return (TRACKED_CONTRIBUTION_CHUNK if submission_engine(submission) == "pfb3x3"
            else CONTRIBUTION_CHUNK)


def run_contribution(ctx: StageContext, submission_id: int, camera: str) -> StageRun:
    started = time.monotonic()
    behavior = ctx.repo.latest_run(submission_id, Stage.behavior, camera)
    if not usable(behavior):
        return fail_run(ctx, submission_id, Stage.contribution,
                        f"行動記録がありません（{camera}）。先に行動の記録を実行してください。",
                        camera=camera, started=started)
    events = _events_of(behavior)
    # moments while the person's tracked box was on another player (an ID swap the names
    # stage's number timeline shows) are not the person's: they are left out
    try:
        other = identity_check.foreign_events(ctx.repo, behavior)
    except Exception:   # noqa: BLE001 - the check is a filter; without it nothing is dropped
        log.exception("番号の読み取りで人の入れ替わりを確かめられませんでした (%s)", camera)
        other = set()
    events = [e for e in events if e.event_id not in other]
    warnings = ([f"別の人に入れ替わっていた場面を除きました（{len(other)}件）。"] if other else [])
    assignment = assignment_of(ctx, submission_id)
    domain = assignment.get("domain") or "school"
    profile = profiles.of_work(assignment)
    prompt = load_prompt(ctx.settings, Kind.video, Stage.contribution, domain, profile)
    common = dict(camera=camera, upstream=[behavior.id], prompt_hash=prompt.hash, started=started,
                  warnings=warnings)
    if not events:
        return save_run(ctx, submission_id, Stage.contribution, output={"judgments": []},
                        **common)
    submission = ctx.repo.get_submission(submission_id)
    known = clock_is_known(submission["meta"], domain)
    size = contribution_chunk(submission)
    groups = [events[i:i + size] for i in range(0, len(events), size)]
    reqs = [ChatRequest(
        system=prompt.text,
        user_text=build_user_text([*_work_sections(assignment),
                                   (_rubric_label(domain, profile), assignment["rubric_text"]),
                                   ("行動イベント", _event_lines(g, domain, known))]),
        image_paths=[], json_schema=ContributionOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens[Stage.contribution.value]) for g in groups]
    validators = [
        (lambda ids: lambda out: checks.judgment_alignment_errors(ids, out.judgments))(
            [e.event_id for e in g]) for g in groups]
    results = call_structured_batch(ctx.backend, ctx.repo, reqs, ContributionOutput,
                                    max_retries=ctx.settings.max_retries,
                                    yield_check=ctx.yield_check, validate=validators)
    common["model_id"] = ctx.backend.model_id
    errors = [f"チャンク{n}: {e}" for n, r in enumerate(results, 1) if r.parsed is None
              for e in r.errors]
    judgments = [j for r in results if r.parsed for j in r.parsed.judgments]
    if not errors:  # whole-run alignment, after the per-chunk checks passed
        errors = checks.judgment_alignment_errors([e.event_id for e in events], judgments)
    if errors:
        return fail_run(ctx, submission_id, Stage.contribution, "\n".join(errors), **common)
    order = {e.event_id: n for n, e in enumerate(events)}
    judgments.sort(key=lambda j: order[j.ref_id])
    return save_run(ctx, submission_id, Stage.contribution,
                    output={"judgments": [j.model_dump() for j in judgments]}, **common)


def run_pfb(ctx: StageContext, submission_id: int, subject_id: int) -> StageRun:
    started = time.monotonic()
    subject = next((s for s in ctx.repo.list_subjects(submission_id) if s["id"] == subject_id),
                   None)
    fail = lambda msg, **kw: fail_run(ctx, submission_id, Stage.pfb, msg, subject_id=subject_id,
                                      started=started, **kw)
    if subject is None or not subject["participant_id"]:
        return fail("対象の参加者が見つかりません。")
    pid = subject["participant_id"]
    cameras = ctx.repo.list_cameras(submission_id)
    if not cameras:
        return fail("カメラのフレーム準備結果がありません。先に動画の準備を実行してください。")
    contributions, true_events = [], []
    for cam in cameras:
        contribution = ctx.repo.latest_run(submission_id, Stage.contribution, cam)
        if not usable(contribution):
            return fail(f"貢献判定の結果がありません（{cam}）。先に貢献判定を実行してください。")
        contributions.append(contribution)
        behavior = (ctx.repo.get_run(contribution.upstream_run_ids[0])
                    if contribution.upstream_run_ids else None)
        if not usable(behavior):
            return fail(f"行動記録がありません（{cam}）。先に行動の記録を実行してください。",
                        upstream=[c.id for c in contributions])
        by_id = {e.event_id: e for e in _events_of(behavior)}
        judgments = [Judgment.model_validate(d) for d in contribution.output["judgments"]]
        missing = sorted({j.ref_id for j in judgments} - set(by_id))
        if missing:
            return fail("貢献判定が存在しない行動イベントを参照しています: " + ", ".join(missing),
                        upstream=[c.id for c in contributions])
        true_events += [by_id[j.ref_id] for j in judgments
                        if j.is_positive_contribution and by_id[j.ref_id].participant == pid]
    assignment = assignment_of(ctx, submission_id)
    domain = assignment.get("domain") or "school"
    profile = profiles.of_work(assignment)
    prompt = load_prompt(ctx.settings, Kind.video, Stage.pfb, domain, profile)
    common = dict(subject_id=subject_id, upstream=[c.id for c in contributions],
                  prompt_hash=prompt.hash, started=started)
    if not true_events:
        out = PfbOutput(pfb_status="insufficient_positive_evidence", pfb=None, evidence_refs=[])
        return save_run(ctx, submission_id, Stage.pfb, output=out.model_dump(), **common)

    interpretations = {j["ref_id"]: j["contribution_interpretation"]
                       for c in contributions for j in c.output["judgments"]}
    criteria = {j["ref_id"]: j.get("rubric_criteria") or []
                for c in contributions for j in c.output["judgments"]}

    def category_line(e: BehaviorEvent) -> str:
        """A profile with its own prompts: the event's category (当てはまる観点)."""
        if profile in profiles.UNCHANGED:
            return ""
        return f"\n当てはまる観点: {profiles.category_of(profile, criteria[e.event_id]) or 'なし'}"

    known = clock_is_known(ctx.repo.get_submission(submission_id)["meta"], domain)
    sports = domain == "sports"
    # sports: a camera records the same action again every second; a run is one event here
    items = _fold_runs(true_events) if sports else [(e, 1) for e in true_events]
    body = "\n\n".join(
        _event_lines([e], domain, known) + (RUN_NOTE if n > 1 else "")
        + f"\n貢献の解釈: {interpretations[e.event_id]}" + category_line(e) for e, n in items)
    req = ChatRequest(
        system=prompt.text,
        user_text=build_user_text([*_work_sections(assignment),
                                   ("貢献ありと判定された行動イベント", body)]),
        image_paths=[], json_schema=PfbOutput.model_json_schema(),
        max_tokens=ctx.settings.max_tokens[Stage.pfb.value])
    true_ids = {e.event_id for e in true_events}
    by_id = {e.event_id: e for e in true_events}
    soft: list[tuple[PfbOutput, list[str]]] = []   # the last answer whose only faults are time words

    def validate(out: PfbOutput) -> list[str]:
        errors = checks.evidence_ref_errors(out, true_ids)
        if errors or not sports or not out.pfb:
            return errors
        minutes = {m for r in out.evidence_refs
                   if (m := _event_minute(by_id[r], known)) is not None}
        problems = checks.time_errors(out.pfb, minutes, known)
        soft[:] = [(out, problems)] if problems else []
        return problems

    result = call_structured(
        ctx.backend, ctx.repo, req, PfbOutput,
        max_retries=ctx.settings.max_retries, yield_check=ctx.yield_check, validate=validate)
    common["model_id"] = ctx.backend.model_id
    pfb, time_warnings = result.parsed, []
    if pfb is None and soft:   # asked again and still faulty in its time words only: kept, flagged
        pfb = soft[0][0]
        time_warnings = [f"時間の書き方を確かめてください: {p}" for p in soft[0][1]]
    if pfb is None:
        return fail_run(ctx, submission_id, Stage.pfb, "\n".join(result.errors), **common)
    if sports and pfb.pfb:
        text = checks.normalize_time_words(pfb.pfb, known)
        refs = checks.trim_refs(pfb.evidence_refs, text, {
            r: _event_minute(by_id[r], known) for r in pfb.evidence_refs if r in by_id})
        pfb = pfb.model_copy(update={"pfb": text, "evidence_refs": refs})
    warnings = list(time_warnings)
    if pfb.pfb:
        internal_ids = set(interpretations) | {
            s["participant_id"] for s in ctx.repo.list_subjects(submission_id)
            if s["participant_id"]}
        tracked = submission_engine(ctx.repo.get_submission(submission_id)) == "pfb3x3"
        warnings += checks.pfb_text_warnings(pfb.pfb, internal_ids, set(cameras), tracked, known)
    return save_run(ctx, submission_id, Stage.pfb, output=pfb.model_dump(), warnings=warnings,
                    **common)
