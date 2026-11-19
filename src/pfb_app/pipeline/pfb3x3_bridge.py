"""PFB3x3 behavior rows -> pfb_app behavior events (tracking engine).

The engine writes one row per window (`behavior.engine.jsonl`), each with events naming a
tracked person by P-label and a frame clock ("m:ss" since the task start), and an
`observed_action` that is one short English sentence using P-labels ("P1 hands a paper strip
to P3"). The teacher's mapping turns the labels into students, so contribution, PFB, review
and export work on the events exactly as they do for the paper engine.

1. `bridge_behavior` (pure): drops rows whose status is not ok (error, skipped), events the
   engine excluded (`excluded_reason`), events of labels mapped to 対象外 or not in the
   mapping, events without an observation, and (defensive; the engine excludes them itself)
   events whose clock cannot be read. In the observation, the event's own label becomes
   「本人」 and every other P-label 「別の参加者」; the engine's original sentence is kept in
   `source_text[event_id]`.
2. `translate_events`: the session's served model turns each (label-free) observation into
   Japanese. If that fails, the label-free English stays and the run gets a warning.

`event_id` is f"{camera}-{window_index:04d}-{n:02d}", n being the event's 1-based position in
its engine row, so an id always points back at the engine's own event.
"""
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.domain.models import BehaviorEvent, BehaviorRunOutput, Kind, TranslationOutput
from pfb_app.engine import EngineServerDown
from pfb_app.inference.backend import ChatRequest, InferenceBackend
from pfb_app.inference.served import ServerUnreachable
from pfb_app.inference.structured import YieldRequested, call_structured_batch
from pfb_app.pipeline.checks import P_LABEL_RE
from pfb_app.pipeline.prompts import Prompt, build_user_text, load_prompt_file
from pfb_app.preprocess.frames import format_time
from pfb_app.storage.db import Repo

CONFIDENCES = ("high", "medium", "low")
DEFAULT_CONFIDENCE = "medium"
SELF, OTHER = "本人", "別の参加者"
TRANSLATE_CHUNK = 30   # observations per translation request
RETRY_CHUNK = 1        # ... when retrying the ones a previous run could not translate
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]")
RUN_META = "run_meta.json"   # the engine's traceability record, next to its rows
_CLOCK = re.compile(r"^(?:(\d+):)?(\d+):(\d{1,2}(?:\.\d+)?)$")
_WINDOW_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*_w(\d+)$")


class BridgeError(Exception):
    """The engine's rows cannot be read (Japanese message for the run's error)."""


def clock_seconds(clock) -> float | None:
    """'m:ss' (also 'mm:ss' and 'h:mm:ss') -> seconds; None when it cannot be read."""
    m = _CLOCK.match(str(clock if clock is not None else "").strip())
    if m is None:
        return None
    h, mins, sec = m.groups()
    return int(h or 0) * 3600 + int(mins) * 60 + float(sec)


def _window_id(row: dict) -> str:
    wid = str(row.get("window_id") or row.get("id") or "")
    if _WINDOW_ID.match(wid) is None:
        raise BridgeError(f"PFB3x3の行動の行に、読み取れない区間IDがあります: {wid!r}")
    return wid


def window_index(row: dict) -> int:
    """The row's window index; rows without one are numbered by their id (`..._w0003`)."""
    wid = _window_id(row)
    value = row.get("window_index")
    if value is None:
        return int(_WINDOW_ID.match(wid).group(1))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BridgeError(f"PFB3x3の行動の行に、読み取れない区間番号があります: {wid} {value!r}")
    return value


def window_frames(overlay_dir: Path, window_id: str) -> list[str]:
    """The overlay images of one window (`<overlay_dir>/<window_id>/<k>.jpg`, k = 1..n), in
    frame order: exactly the JPEGs the engine sent to the model. Only files that resolve
    (symlinks included) inside `overlay_dir` are returned."""
    root = Path(overlay_dir).resolve()
    d = Path(overlay_dir) / window_id
    if not d.is_dir():
        return []
    shots = []
    for p in d.glob("*.jpg"):
        real = p.resolve()
        if p.stem.isdigit() and real.is_file() and real.is_relative_to(root):
            shots.append((int(p.stem), str(real)))
    return [path for _, path in sorted(shots)]


def usable_row(row: dict) -> bool:
    return row.get("status", "ok") == "ok" and not row.get("error")


def without_labels(text: str, own_label: str) -> str:
    """The observation with the event's own P-label as 「本人」 and any other as 「別の参加者」."""
    norm = unicodedata.normalize("NFKC", text)
    own = own_label.upper()
    return P_LABEL_RE.sub(lambda m: SELF if m.group(0).upper() == own else OTHER, norm)


def game_clock_offset_s(meta: dict, domain: str | None) -> float | None:
    """Sports: the game clock at the analysed part's start, in seconds. The coach's
    clock_offset_min is the game clock at the file's 0:00, so start_sec adds to it.
    School, or sports without a clock_offset_min (the coach left it blank: a clip from the
    middle of a game, whose game time is not known; a typed 0 is a clock): None
    (task_elapsed stays the time from the analysed start)."""
    if domain != "sports" or meta.get("clock_offset_min") is None:
        return None
    return 60 * float(meta["clock_offset_min"]) + float(meta.get("start_sec") or 0)


def clock_is_known(meta: dict, domain: str) -> bool:
    """Whether a sports work's events sit on a real game clock: only a clock_offset_min the
    coach gave (a typed 0 included) makes task_elapsed a game clock; a clip whose start the
    coach left blank has none (the same rule as export.sheets). Not sports: True (unused)."""
    return domain != "sports" or meta.get("clock_offset_min") is not None


def format_game_clock(sec: float) -> str:
    """The game clock in total minutes, MM:SS (65:30, never 1:05:30)."""
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def bridge_behavior(engine_rows: list[dict], mapping: dict[str, str | None], camera: str,
                    start_s: float, transcript_used: bool, overlay_dir: Path,
                    warnings: list[str] | None = None,
                    clock_offset_s: float | None = None) -> dict:
    """The behavior run output {"events", "frames", "source_text"} of one camera, before
    translation. `mapping` is the teacher's {P-label: participant_id or None (対象外)};
    `start_s` is the task start in the video. Notes for the teacher (failed windows, labels
    that are not in the mapping) are appended to `warnings`. `clock_offset_s` (sports, from
    game_clock_offset_s) is the game clock at the analysed part's start: task_elapsed is then
    the game clock in total minutes; video_time never changes. None (school): task_elapsed is
    the time from the task start. BridgeError: unreadable rows."""
    by_window: dict[int, dict] = {}
    for row in engine_rows:   # one current row per window; a repeated window keeps the last
        if not isinstance(row, dict):
            raise BridgeError("PFB3x3の行動の行の形式が正しくありません。")
        by_window[window_index(row)] = row
    source = "video+audio" if transcript_used else "video"
    placed: list[tuple[float, int, int, BehaviorEvent, str]] = []
    frames: dict[str, list[str]] = {}
    failed = unreadable = 0
    unmapped: Counter[str] = Counter()
    for w, row in sorted(by_window.items()):
        if not usable_row(row):
            failed += row.get("status") != "skipped"
            continue
        shots = None
        for n, e in enumerate(row.get("events") or [], 1):
            if not isinstance(e, dict) or e.get("excluded_reason"):
                continue
            label = str(e.get("label") or "").strip().upper()
            if label not in mapping:
                unmapped[label or "（なし）"] += 1
                continue
            pid = mapping[label]
            action = str(e.get("observed_action") or "").strip()
            if pid is None or not action:   # 対象外 (the teacher, an onlooker) / nothing seen
                continue
            sec = clock_seconds(e.get("clock"))
            if sec is None:
                unreadable += 1
                continue
            confidence = str(e.get("confidence") or "").strip().lower()
            event = BehaviorEvent(
                event_id=f"{camera}-{w:04d}-{n:02d}", camera=camera, participant=pid,
                video_time=format_time(start_s + sec),
                task_elapsed=(format_time(sec) if clock_offset_s is None
                              else format_game_clock(clock_offset_s + sec)),
                observed_behavior=without_labels(action, label), utterance="",
                speaker="unknown", evidence_source=source,
                confidence=confidence if confidence in CONFIDENCES else DEFAULT_CONFIDENCE)
            placed.append((sec, w, n, event, action))
            if shots is None:
                shots = window_frames(overlay_dir, _window_id(row))
            if shots:
                frames[event.event_id] = list(shots)
    placed.sort(key=lambda t: t[:3])
    if warnings is not None:
        if failed:
            warnings.append(f"PFB3x3が観察できなかった区間があります（{failed}/{len(by_window)}区間）。"
                            "その区間の行動は記録されていません。")
        if unmapped:
            warnings.append("対応づけにない人物の行動を除外しました: " + "、".join(
                f"{label}（{k}件）" for label, k in sorted(unmapped.items())))
        if unreadable:
            warnings.append(f"時刻を読み取れない行動を除外しました（{unreadable}件）")
    out = BehaviorRunOutput(events=[t[3] for t in placed], frames=frames,
                            source_text={t[3].event_id: t[4] for t in placed})
    return out.stored()


def translation_prompt(settings: Settings) -> Prompt:
    return load_prompt_file(settings, Kind.video, "translate")


def engine_run_meta(rows_path: Path) -> dict | None:
    """The engine's run_meta.json next to its behavior.engine.jsonl; None when it is missing
    or unreadable."""
    try:
        rec = json.loads((Path(rows_path).parent / RUN_META).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return rec if isinstance(rec, dict) else None


def provenance_parts(engine_commit: str | None, run_meta: dict | None,
                     translate_hash: str) -> dict:
    """What a tracking-engine behavior run depends on (None where run_meta has nothing)."""
    meta = run_meta or {}
    prompts = meta.get("prompts")
    return {"engine_commit": engine_commit,
            "engine_prompts": dict(prompts) if isinstance(prompts, dict) else None,
            "settings_sha": meta.get("settings_sha"),
            "translate_prompt": translate_hash}


def provenance_hash(parts: dict) -> str:
    """The behavior run's prompt_hash: sha256 over the provenance parts, 12 hex digits."""
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def translate_events(output: dict, backend: InferenceBackend, repo: Repo, settings: Settings,
                     yield_check, warnings: list[str], stats: dict | None = None,
                     retry_ids: set[str] | frozenset = frozenset()) -> dict:
    """`output` with every observation translated to Japanese by `backend` (batched, cached,
    validated, retried). Chunks that still fail keep their label-free English (warning).
    A yield request is passed on; any other failure of the backend is a warning.
    `stats` (if given) gets {"total", "translated"}: events, and events now in Japanese.

    `retry_ids`: events whose translation failed (stayed English) in the run being
    regenerated. They are asked again one per request and never from call_log, so a cached
    (deterministic) failure is not simply repeated; the other events use the cache."""
    events = output["events"]
    if stats is not None:
        stats.update(total=len(events), translated=0)
    if not events:
        return output
    prompt = translation_prompt(settings)
    again = [e for e in events if e["event_id"] in retry_ids]
    rest = [e for e in events if e["event_id"] not in retry_ids]
    batches = [([rest[i:i + TRANSLATE_CHUNK] for i in range(0, len(rest), TRANSLATE_CHUNK)],
                False),
               ([again[i:i + RETRY_CHUNK] for i in range(0, len(again), RETRY_CHUNK)], True)]
    texts: dict[str, str] = {}
    untranslated, first_error = 0, None
    for groups, fresh in batches:
        if not groups:
            continue
        reqs = [ChatRequest(
            system=prompt.text,
            user_text=build_user_text([("行動の記録", "\n\n".join(
                f"ref_id: {e['event_id']}\n記録: {e['observed_behavior']}" for e in g))]),
            image_paths=[], json_schema=TranslationOutput.model_json_schema(),
            max_tokens=settings.max_tokens.get("behavior", 4096)) for g in groups]
        validators = [(lambda ids: lambda out: _translation_errors(ids, out))(
            [e["event_id"] for e in g]) for g in groups]
        try:
            results = call_structured_batch(backend, repo, reqs, TranslationOutput,
                                            max_retries=settings.max_retries,
                                            yield_check=yield_check, validate=validators,
                                            fresh=fresh)
        except YieldRequested:
            raise
        except ServerUnreachable as e:   # the session's server is gone: requeue, new session
            raise EngineServerDown(f"PFB3x3のサーバーに接続できません: {e}") from e
        except Exception as e:   # the served model answered nonsense, or a stand-in failed
            untranslated += sum(len(g) for g in groups)
            first_error = first_error or f"{type(e).__name__}: {e}"
            continue
        for g, r in zip(groups, results):
            if r.parsed is None:
                untranslated += len(g)
                first_error = first_error or (r.errors[-1] if r.errors else "不明なエラー")
                continue
            texts.update({t.ref_id: t.text.strip() for t in r.parsed.translations})
    if untranslated:
        warnings.append("行動の記録を日本語に訳せなかったため、英語のまま記録しました"
                        f"（{untranslated}件）: {first_error}")
    if stats is not None:
        stats["translated"] = len(events) - untranslated
    return dict(output, events=[dict(e, observed_behavior=texts.get(e["event_id"],
                                                                   e["observed_behavior"]))
                                for e in events])


def still_english(text: str) -> bool:
    """True when an observation has no Japanese in it besides the label stand-ins 本人 /
    別の参加者 (its translation did not happen)."""
    rest = (text or "").replace(SELF, " ").replace(OTHER, " ")
    return _JAPANESE.search(rest) is None


def english_event_ids(output: dict | None) -> set[str]:
    """Events of a behavior run output whose observation is still English."""
    return {e["event_id"] for e in (output or {}).get("events") or []
            if still_english(e.get("observed_behavior", ""))}


def _translation_errors(ids: list[str], out: TranslationOutput) -> list[str]:
    got = [t.ref_id for t in out.translations]
    errors = [f"訳が欠落しています: {i}" for i in ids if i not in got]
    errors += [f"余分な訳があります: {i}" for i in sorted(set(got) - set(ids))]
    errors += [f"訳が重複しています: {i}" for i, k in Counter(got).items() if k > 1]
    for t in out.translations:
        if not t.text.strip():
            errors.append(f"訳が空です: {t.ref_id}")
        elif P_LABEL_RE.search(unicodedata.normalize("NFKC", t.text)):
            errors.append(f"訳に人物の番号が残っています: {t.ref_id}")
    return errors
