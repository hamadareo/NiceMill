"""Pure view helpers of the 判定支援 pages: copy, names, episode rows, agreement, the court
mini-map SVG, the stream.jsonl tail and the live state. No Streamlit calls, so all of it is
tested directly.

Engine inputs (PFB3x3 feature/linecall):
- ``<tracks>/<id>/linecall/episodes.json``: ``{status, video_id, source, fps, dir, episodes}``;
  each episode has id, label (``P7`` engine tracks, ``A6``/``B3`` TrackID jerseys), line,
  verdict (stepped | clear | undecidable), clock (game clock or None), video_s, crop/overlay/
  clip paths relative to ``linecall/``.
- ``<tracks>/<id>/linecall/geometry.json`` (engine 58f8341): ``H`` (image px -> court cm),
  ``H_frames``, ``lens``, ``h_unc_group``; ``frames.jsonl`` (one row per episode, frame and
  foot, with the T11 fields ``hi_raw``, ``frame_would_step``, ``ground_gap_px``): only
  ``near_court`` (the player's court point) of a 踏んだ episode's evidence frame is read
  (court_points). An episode's ``evidence`` is ``[{side, method, d, unc}]``.
- ``live/<session>/stream.jsonl``, one JSON record per line: ``start``; ``positions``
  (players ``{id, x, y, in_play, near_line?, near_cm?}`` in court cm); ``alert`` (event
  touch | cross, alert_id, player, line, side, distance_cm, needs_check), ``alert_end``;
  ``dwell`` / ``dwell_end`` (dwell_id, label ライン付近に滞在); ``stats`` (latency
  ``{p50_ms, ...}``); ``end``.

Court frame (cm, ``pfb3x3.linecall.court``): x = 0 the left sideline, y = 0 the baseline under
the basket (far from the camera), y = 1105 the top line (near the camera). The mini-map keeps
the camera's view: the far baseline at the top, so court y grows down the SVG like image y.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import stat
from html import escape
from pathlib import Path

from dataclasses import dataclass

from pfb_app.domain.vocab import mark_name
from pfb_app.ui.wizard.progress import camera_label
from pfb_app.ui.wizard.theme import TOKENS

# -- copy (plan Global Constraints, exact) --------------------------------------------------

NOTE = "ここに出るのは判定材料です。最後に決めるのは審判とコーチです。"
VERDICT_PREFIX = "判定材料"
VERDICT_LABELS = {"stepped": "踏んだ", "clear": "踏んでいない", "undecidable": "判定不能"}
LABEL_BUTTONS = (("correct", "正しい"), ("wrong", "違う"), ("undecidable", "判定不能"))
LINE_LABELS = {"sideline_L": "サイドライン（左）", "sideline_R": "サイドライン（右）",
               "baseline": "エンドライン", "top_line": "反対側のライン", "arc": "2ポイントライン"}
SIDE_LABELS = {"in": "内側", "out": "外側", "inside": "内側", "outside": "外側"}
NO_EPISODES = "判定材料はまだありません"
BUILD = "判定材料を作る"
BUILDING = "判定材料を作っています"
BUILD_FAILED = "判定材料を作れませんでした。処理状況の画面を確認してください。"
WATCH_CLIP = "動画で見る"
LIVE_START = "デモを開始"
LIVE_STOP = "停止"
LIVE_RUNNING = "ライブ判定を実行中"
LIVE_STOPPED = "停止中"
NEEDS_CHECK = "要確認"
# A 踏んだ at a line toward the camera is shown as 要確認 (toward_camera): the engine's ground
# check sees nothing there (PFB3x3 T11 review: a foot reaching toward the camera IS the box
# bottom), so it is never shown as a plain 踏んだ.
# the live page: alerts are distances to the lines, sidelines included
LIVE_SIDELINE_NOTE = "サイドラインまでの距離は、まだ正解データでの確認が済んでいません。"
NEAR_LINE_NOTE = ("カメラに近い側の線では、上げた足が線にかかって見えることがあります。"
                  "踏んだかどうかは、写真と動画で確かめてください。")
# Sideline verdicts are not validated until a GT check of the sideline has side_agreement
# n > 0 (PFB3x3 T11: n = 0 on IMG_0105/0104/0109). Set to True only then.
SIDELINE_VALIDATED = False
SIDELINE_NOTE = "サイドラインの判定は、まだ正解データでの確認が済んでいません。"
LIVE_NO_GPU = "解析用の機械が混んでいるため開始できません。少し待ってからもう一度押してください。"
LIVE_NO_DEMO = "デモ用の動画がまだ用意されていません。使うときは、管理者に用意を頼んでください。"
LIVE_STOPPING = "停止しています…"
LIVE_FAILED = ("ライブ判定が途中で止まりました。もう一度押しても止まるときは、"
               "管理者に連絡してください。")
LIVE_START_FAILED = "ライブ判定を始められませんでした。管理者に連絡してください。"
LIVE_VIDEO_MISSING = "選んだ動画が見つかりません。動画の置き場所を確認してください。"
LIVE_DEMO_LABEL = "デモ動画"
LIVE_SOURCE_LABEL = "使う動画"
LIVE_NO_ALERTS = "まだアラートはありません。"
LIVE_DROPS = "飛ばした映像 {pct}%"
LIVE_DROPS_NOTE = ("映像の一部を飛ばしています。"
                   "アラートが出ないことは、線を踏んでいないことを意味しません。")
# ... when other work kept the machine busy while it dropped (inference.gpu.machine_busy)
LIVE_DROPS_NOTE_BUSY = "混んでいるため" + LIVE_DROPS_NOTE
LIVE_LOAD_CHECK_SEC = 10.0   # the live page asks machine_busy at most this often, while dropping
LIVE_TICK_SEC = 1.0          # the live page redraws this often
LIVE_ALERTS_SHOWN = 10       # newest first
LIVE_OUTCOME_SHOWN_SEC = 600   # a needs-GPU / error banner is shown this long after the end
LIVE_RUNNING_VIDEO = "実行中の動画: {label}"
LIVE_RUNNING_ELSEWHERE = "実行中の動画はこの一覧にありません。"
AGREEMENT_MIN_N = 50
AGREEMENT_COUNT = "コーチの確認: {n}件（あと{m}件で一致率を表示）"
AGREEMENT_RATE = "コーチの確認との一致率: 踏んだ（要確認を含む） {stepped} / 踏んでいない {clear}"
AGREEMENT_CLASS = "{p}%（{n}件）"           # a class with at least min_n confirmations
AGREEMENT_CLASS_WAIT = "—（あと{m}件）"     # a class below min_n
_LINE_FALLBACK = "ライン"
_DASH = "—"


def coach_label_text(label: str | None) -> str | None:
    """'correct' -> 「コーチの確認: 正しい」; None for no (or an unknown) label."""
    word = dict(LABEL_BUTTONS).get(label or "")
    return f"コーチの確認: {word}" if word else None


def drop_texts(state: "LiveState", busy: bool = False) -> tuple[str | None, str | None]:
    """(「飛ばした映像 3%」, the drops note when any frame was dropped: LIVE_DROPS_NOTE_BUSY
    (「混んでいるため…」) when other work kept the machine busy meanwhile, else the neutral
    LIVE_DROPS_NOTE); None, None before a stats / end record with frame counts."""
    rate = state.drop_rate
    if rate is None:
        return None, None
    pct = rate * 100
    shown = f"{pct:.0f}" if pct == 0 or pct >= 1 else f"{pct:.1f}"
    note = (LIVE_DROPS_NOTE_BUSY if busy else LIVE_DROPS_NOTE) if state.dropped > 0 else None
    return LIVE_DROPS.format(pct=shown), note


def latency_text(latency_s: float | None) -> str | None:
    if latency_s is None or not math.isfinite(latency_s):
        return None
    return f"遅れ 約{latency_s:.1f}秒"


# -- names ----------------------------------------------------------------------------------

def player_name(label: str | int) -> str:
    """'P7' / 7 / '7' -> 「7番の人」; 'A6' -> 「Aチーム #6」; anything else as given."""
    if isinstance(label, int) and not isinstance(label, bool):
        return mark_name(f"P{label}")
    s = str(label).strip() if label is not None else ""
    if re.fullmatch(r"[Pp]\d+", s):
        return mark_name(s)
    if re.fullmatch(r"\d+", s):
        return mark_name(f"P{s}")
    if re.fullmatch(r"[AB]\d+", s):
        return f"{s[0]}チーム #{s[1:]}"
    return s or _DASH


def line_name(line: str | None) -> str:
    return LINE_LABELS.get(line or "", _LINE_FALLBACK)


# -- episodes -------------------------------------------------------------------------------

def load_episodes(path: Path) -> list[dict] | None:
    """The episodes of an episodes.json; None when it is missing, unreadable, torn (the engine
    writes it atomically, but a copy may be half-written) or has no ``"episodes"`` list."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):   # UnicodeDecodeError and JSONDecodeError are ValueErrors
        return None
    eps = doc.get("episodes") if isinstance(doc, dict) else None
    if not isinstance(eps, list):
        return None
    return [e for e in eps if isinstance(e, dict)]


def _mss(seconds) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return _DASH
    if not math.isfinite(s) or s < 0:
        return _DASH
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def episode_row(ep: dict, geom=None, points: dict | None = None) -> dict:
    """One table row: 時間 (game clock, else m:ss into the video), 選手, 線, 判定材料 (the
    shown verdict, shown_verdict: a 踏んだ at a line toward the camera, or with no geometry,
    is 要確認)."""
    clock = ep.get("clock")
    return {"時間": clock if isinstance(clock, str) and clock else _mss(ep.get("video_s")),
            "選手": player_name(ep.get("label")),
            "線": line_name(ep.get("line")),
            "判定材料": SHOWN_LABELS[shown_verdict(ep, geom, points)]}


# -- lines toward the camera ------------------------------------------------------------------

LINE_GROUP = {"sideline_L": "sideline", "sideline_R": "sideline", "baseline": "baseline",
              "top_line": "top_line", "arc": "arc"}   # pfb3x3.linecall.court.GROUP
DOWN_PX = 5.0       # the image-down step that stands for "toward the camera"
MIN_INWARD = 0.1    # a line is away from the camera only when that step moves at least this
#                     share of its length away from the line on the player's side
# A player this close to a line (cm; at least the geometry's H uncertainty of that line group)
# may be on either side of it: the line counts as toward the camera (final review MED 4). The
# floor covers geometries without an uncertainty (TID meta): their line offsets reach ~13 cm.
SIDE_BAND_MIN_CM = 15.0
SHOWN_LABELS = {**VERDICT_LABELS, "check": "踏んだ（要確認）"}


@dataclass(frozen=True)
class CameraGeometry:
    """geometry.json of one camera: H (image px -> court cm), H_frames and the H uncertainty
    per line group (h_unc_group, else h_unc_cm)."""
    H: object
    H_frames: object
    h_unc_group: dict
    h_unc_cm: float

    def band_cm(self, line: str) -> float:
        v = _num(self.h_unc_group.get(LINE_GROUP.get(line, line)))
        return max(v if v is not None else (self.h_unc_cm or 0.0), SIDE_BAND_MIN_CM)


def load_geometry(lc_dir: Path) -> CameraGeometry | None:
    """The camera's linecall/geometry.json as a CameraGeometry; None when it is missing,
    unreadable or its H is not usable (export.tracks.load_geometry_file)."""
    from pfb_app.export.tracks import load_geometry_file   # noqa: PLC0415 - numpy
    f = Path(lc_dir) / "geometry.json"
    g = load_geometry_file(f)
    if g is None:
        return None
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    group = doc.get("h_unc_group") if isinstance(doc.get("h_unc_group"), dict) else {}
    return CameraGeometry(g.H, g.H_frames, {str(k): v for k, v in group.items()},
                          _num(doc.get("h_unc_cm")) or 0.0)


def _signed_cm(line: str, x: float, y: float) -> float:
    """Signed cm to the line (+ in court / inside the arc), as pfb3x3.linecall.court."""
    if line == "sideline_L":
        return x
    if line == "sideline_R":
        return CW - x
    if line == "baseline":
        return y
    if line == "top_line":
        return CH - y
    bx, by = BASKET            # the arc: straight parts up to ARC_Y, then the circle
    dy_seg = y - min(max(y, 0.0), ARC_Y)
    dist = min(math.hypot(x - ARC_X, dy_seg), math.hypot(x - (CW - ARC_X), dy_seg))
    phi = math.atan2(y - by, x - bx)
    alpha = math.atan2(ARC_Y - by, bx - ARC_X)
    if alpha <= phi <= math.pi - alpha:
        dist = min(dist, abs(math.hypot(x - bx, y - by) - R3))
    inside = (math.hypot(x - bx, y - by) < R3) if y > ARC_Y else (ARC_X < x < CW - ARC_X)
    return dist if inside else -dist


def _point(pt) -> tuple[float, float] | None:
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        x, y = _num(pt[0]), _num(pt[1])
        if x is not None and y is not None:
            return x, y
    return None


_POINTS_CACHE: dict[tuple, dict[str, tuple[float, float]]] = {}
_EPISODE_RE = re.compile(rb'"episode":\s*"((?:[^"\\]|\\.)*)"')
_FRAME_RE = re.compile(rb'"frame":\s*(\d+)')


def court_points(lc_dir: Path, want: dict[str, int]) -> dict[str, tuple[float, float]]:
    """{episode id: the player's court point (near_court) at frame want[id]} from the camera's
    linecall/frames.jsonl (engine 58f8341; tens of MB, so only the wanted episodes' lines are
    parsed, and the answer is kept per file size / mtime). {} when the file is missing or
    unreadable; an episode without a finite near_court there is left out."""
    if not want:
        return {}
    f = Path(lc_dir) / "frames.jsonl"
    try:
        st = f.stat()
    except OSError:
        return {}
    key = (str(f), st.st_size, st.st_mtime_ns, tuple(sorted(want.items())))
    if key in _POINTS_CACHE:
        return _POINTS_CACHE[key]
    out: dict[str, tuple[float, float]] = {}
    try:
        with open(f, "rb") as fh:
            for line in fh:
                m = _EPISODE_RE.search(line)
                if m is None:
                    continue
                try:
                    eid = json.loads(b'"' + m.group(1) + b'"')
                except ValueError:
                    continue
                if eid not in want or eid in out:
                    continue
                fm = _FRAME_RE.search(line)
                if fm is None or int(fm.group(1)) != want[eid]:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                pt = _point(row.get("near_court")) if isinstance(row, dict) else None
                if pt is not None and row.get("frame") == want[eid]:
                    out[eid] = pt
    except OSError:
        return {}
    if len(_POINTS_CACHE) > 16:
        _POINTS_CACHE.clear()
    _POINTS_CACHE[key] = out
    return out


def toward_camera(ep: dict, geom, point=None) -> bool | None:
    """Whether the episode's line lies toward the camera from the player: True / False, None
    when it cannot be told (no geometry, an unknown line, no court point, a point that does not
    map).

    The engine gives no near/far flag, so it is derived from geometry.json: the court point
    (`point`, the player's near_court at the evidence frame, court_points) is mapped into the
    image with H's inverse, moved DOWN_PX image-down (toward the camera on the floor; the side a
    raised foot reaching toward the camera extends to) and mapped back. A player within the
    line's band (CameraGeometry.band_cm: the H uncertainty, at least SIDE_BAND_MIN_CM) may be
    on either side: toward. Otherwise the line is away from the camera only when that step
    moves away from the line on the player's side (the sign of the signed distance, outline
    lines and the arc alike) by at least MIN_INWARD of the step. H_frames' entry of the
    evidence frame is used when there is one; the lens is left out (DOWN_PX is a few pixels)."""
    import numpy as np   # noqa: PLC0415 - only with a geometry
    line = ep.get("line")
    pt = _point(point)
    if geom is None or line not in LINE_GROUP or pt is None:
        return None
    H = geom.H
    frame = _int_id(ep.get("frame"))
    if geom.H_frames is not None and frame is not None and frame >= 1:
        Hf = np.asarray(geom.H_frames[min(frame, len(geom.H_frames)) - 1], dtype=float)
        if Hf.shape == (3, 3) and np.all(np.isfinite(Hf)) and Hf[2, 2] != 0:
            H = Hf
    try:
        q = np.linalg.solve(H, np.array([pt[0], pt[1], 1.0]))
        if not np.all(np.isfinite(q)) or abs(q[2]) < 1e-12:
            return None
        p2 = H @ np.array([q[0] / q[2], q[1] / q[2] + DOWN_PX, 1.0])
    except (np.linalg.LinAlgError, ValueError):
        return None
    if not np.all(np.isfinite(p2)) or abs(p2[2]) < 1e-12:
        return None
    x2, y2 = float(p2[0] / p2[2]), float(p2[1] / p2[2])
    step = math.hypot(x2 - pt[0], y2 - pt[1])
    if not math.isfinite(step) or step <= 0:
        return None
    d0, d1 = _signed_cm(line, *pt), _signed_cm(line, x2, y2)
    band = geom.band_cm(line) if hasattr(geom, "band_cm") else SIDE_BAND_MIN_CM
    if abs(d0) <= band:
        return True
    away = (d1 - d0) * (1.0 if d0 > 0 else -1.0)
    return bool(away < MIN_INWARD * step)


def shown_verdict(ep: dict, geom, points: dict | None = None) -> str:
    """The verdict the pages show: "stepped" / "clear" / "undecidable", or "check" (要確認)
    for a 踏んだ at a line toward the camera, or whose geometry is unknown. `points`: the
    court_points of the camera. The engine's verdict stays what labels are stored for."""
    verdict = ep.get("verdict")
    if verdict not in VERDICT_LABELS:
        return "undecidable"
    if verdict == "stepped" and toward_camera(
            ep, geom, (points or {}).get(str(ep.get("id")))) is not False:
        return "check"
    return verdict


def stepped_frames(episodes: list[dict]) -> dict[str, int]:
    """{episode id: evidence frame} of the 踏んだ episodes (what court_points looks up)."""
    out = {}
    for ep in episodes:
        frame = _int_id(ep.get("frame"))
        if ep.get("verdict") == "stepped" and ep.get("id") is not None and frame is not None:
            out[str(ep["id"])] = frame
    return out


def episode_notes(ep: dict, shown: str) -> list[str]:
    """The notes under an episode: why a 踏んだ is 要確認, and that sideline verdicts are not
    validated yet (SIDELINE_VALIDATED)."""
    notes = []
    if shown == "check":
        notes.append(NEAR_LINE_NOTE)
    if not SIDELINE_VALIDATED and shown in ("stepped", "clear", "check") \
            and str(ep.get("line") or "").startswith("sideline"):
        notes.append(SIDELINE_NOTE)
    return notes


def _label_of(value, verdict: str | None) -> str | None:
    """The coach label, or None when there is none or it was given for another verdict (a
    stored record {"label", "verdict"} whose episode changed on a rerun)."""
    if isinstance(value, dict):
        if value.get("verdict") not in (None, verdict):
            return None
        value = value.get("label")
    return value if isinstance(value, str) else None


def agreement(episodes: list[dict], labels: dict[str, str],
              min_n: int = AGREEMENT_MIN_N) -> str:
    """The coach agreement line. Confirmed = 正しい + 違う on 踏んだ / 踏んでいない episodes
    still in the list, for their current verdict (判定不能 episodes never count). Until one
    class has `min_n` the line counts them; then each class shows its agreement, or a dash
    with what is still missing while it is below `min_n`."""
    verdict_of = {str(e.get("id")): e.get("verdict") for e in episodes
                  if isinstance(e, dict) and e.get("id") is not None}
    tally = {v: {"correct": 0, "wrong": 0} for v in ("stepped", "clear")}
    for ep_id, value in labels.items():
        v = verdict_of.get(str(ep_id))
        label = _label_of(value, v)
        if v in tally and label in ("correct", "wrong"):
            tally[v][label] += 1
    counts = {v: c["correct"] + c["wrong"] for v, c in tally.items()}
    if max(counts.values()) < min_n:
        return AGREEMENT_COUNT.format(n=sum(counts.values()),
                                      m=min_n - max(counts.values()))

    def cell(v: str) -> str:
        n = counts[v]
        if n < min_n:
            return AGREEMENT_CLASS_WAIT.format(m=min_n - n)
        return AGREEMENT_CLASS.format(p=int(100 * tally[v]["correct"] / n + 0.5), n=n)

    return AGREEMENT_RATE.format(stepped=cell("stepped"), clear=cell("clear"))


# -- the 「試合後の判定材料」 page ----------------------------------------------------------------

REVIEW_TITLE = "試合後の判定材料"
REVIEW_GUIDE = ("線の近くの場面を1件ずつ見て、写真と動画で確かめたら"
                "「正しい」「違う」「判定不能」のどれかを押してください。")
NO_SPORTS_WORK = "判定支援を使える試合・練習がありません。"
NEEDS_TRACK = "判定材料を作るには、先にこのカメラで人を見つける処理を終えてください。"
NOT_QUEUED = "この試合・練習は削除の手続き中のため、判定材料を作れません。"
MEDIA_REMOVED = "この試合・練習の動画は消してあるため、判定材料を作れません。"
ERASING = ("このグループの人のデータを消している途中のため、いまは判定材料を作れません。"
           "少し待ってから、もう一度開いてください。")
BLOCKED_TEXTS = {"deleting": NOT_QUEUED, "media_removed": MEDIA_REMOVED, "erasing": ERASING}
# the pose weights are missing or not the pinned ones (pipeline.linecall.NOT_READY)
NOT_READY = "判定材料を作る準備ができていません。管理者に連絡してください。"
STALE_VERDICT = "判定材料が新しくなりました。もう一度確かめてください。"
NO_PHOTO = "写真なし"
NO_CLIP = "動画なし"
CROP_CAPTION = "足もと"
OVERLAY_CAPTION = "線を重ねた写真"
PAGE_SIZE = 20
NEXT_PAGE = "次の20件"
PREV_PAGE = "前の20件"
PAGE_RANGE = "{a}〜{b}件目（全{n}件）"
PICK_GROUP = "グループ"
PICK_CAMERA = "カメラ"
# The verdict chips (theme.status_chip kinds): each verdict has its own look, and the words
# are always there.
VERDICT_CHIPS = {"stepped": "your_turn", "clear": "done", "undecidable": "insufficient",
                 "check": "guess"}   # 要確認: to be checked
# pfb3x3.linecall.verdict.REASONS, in plain words
REASON_LABELS = {
    "occluded": "足が隠れています",
    "no_box": "一部の場面で人が見つかりません",
    "no_pose": "足の位置が分かりません",
    "low_conf": "足の位置がはっきりしません",
    "line_not_found": "線が見つかりません",
    "line_unstable": "線の位置が場面ごとにずれています",
    "margin": "線に近すぎて決められません",
}
REASON_TEXT = "判定不能の理由: {reason}"
# The per-foot gap to the painted line (distance_text)
NO_MEASURE = "測れませんでした"
ON_LINE = "線の上"
UNDER_1CM = "{side}に1 cm未満"
GAP_CM = "{side}に{n} cm"
UNC_TEXT = "（誤差 ±{n} cm）"
SIDE_WORDS = {"court": "線の内側（コート側）", "goal": "線の内側（ゴール側）", "out": "線の外側"}
_FEET = (("L", "左足"), ("R", "右足"))
# The engine's lines (pfb3x3.linecall.court): the outline's d is the signed distance to the
# court-surface edge (+ = in the court), the line itself 5 cm wide outside it; the arc's d is
# signed to the stripe's centre line (+ = the basket's side), the stripe RIDGE_HALF_CM each way.
OUTLINE_LINES = ("sideline_L", "sideline_R", "baseline", "top_line")
OUTLINE_LINE_CM = 5.0
ARC_HALF_CM = 3.4
# Evidence files the page opens: pictures as photos, mp4 as clips, and nothing huge.
EVIDENCE_SUFFIXES = {"photo": (".jpg", ".jpeg", ".png"), "clip": (".mp4",)}
EVIDENCE_MAX_BYTES = 256 * 1024 * 1024


def evidence_path(base: Path, rel, kind: str) -> Path | None:
    """A crop / overlay ("photo") or clip ("clip") of an episode: `rel` is relative to the
    camera's linecall/ dir. The resolved file, or None when there is none, it is not a file,
    its type is not one of EVIDENCE_SUFFIXES[kind], or the path (or a link on it) leads out
    of `base`."""
    if not isinstance(rel, str) or not rel or "\x00" in rel or Path(rel).is_absolute():
        return None
    if Path(rel).suffix.lower() not in EVIDENCE_SUFFIXES[kind]:
        return None
    try:
        root = Path(base).resolve(strict=True)
        full = (root / rel).resolve(strict=True)
        if not full.is_relative_to(root) or not full.is_file():
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    if full.suffix.lower() not in EVIDENCE_SUFFIXES[kind]:   # a link to another type
        return None
    return full


def evidence_file(base: Path, rel, kind: str) -> bytes | None:
    """The bytes of an episode's photo or clip (see evidence_path); None when it cannot be
    used or read. The page hands the bytes to Streamlit, so the file is opened once, here."""
    full = evidence_path(base, rel, kind)
    if full is None:
        return None
    try:
        with open(full, "rb") as fh:
            info = os.fstat(fh.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > EVIDENCE_MAX_BYTES:
                return None
            return fh.read(EVIDENCE_MAX_BYTES + 1)[:EVIDENCE_MAX_BYTES]
    except OSError:
        return None


def _gap_text(line: str, d) -> str:
    """One foot: 「線の上」, 「線の外側に1 cm未満」, 「線の内側（コート側）に12 cm」 ..."""
    if line == "arc":
        gap, side = abs(d) - ARC_HALF_CM, "goal" if d > 0 else "out"
    else:
        gap, side = (d, "court") if d > 0 else (-d - OUTLINE_LINE_CM, "out")
    if gap <= 0:
        return ON_LINE
    if gap < 0.5:
        return UNDER_1CM.format(side=SIDE_WORDS[side])
    return GAP_CM.format(side=SIDE_WORDS[side], n=int(gap + 0.5))


def distance_text(ep: dict) -> str | None:
    """Each foot's gap to the painted line at the evidence frame, with its uncertainty:
    「左足: 線の上（誤差 ±4 cm）　右足: 線の内側（コート側）に12 cm（誤差 ±4 cm）」. A foot
    without a measurement 「測れませんでした」; None when neither foot (or the line) is known.
    A single uncertainty is used for both feet."""
    line, dist, unc = ep.get("line"), ep.get("distances"), ep.get("uncertainty_cm")
    if line not in (*OUTLINE_LINES, "arc") or not isinstance(dist, dict):
        return None
    feet = {s: _num(dist.get(s)) for s, _ in _FEET}
    if all(d is None for d in feet.values()):
        return None
    parts = []
    for s, word in _FEET:
        d = feet[s]
        if d is None:
            parts.append(f"{word}: {NO_MEASURE}")
            continue
        u = _num(unc.get(s)) if isinstance(unc, dict) else _num(unc)
        tail = UNC_TEXT.format(n=math.ceil(u)) if u is not None else ""
        parts.append(f"{word}: {_gap_text(line, d)}{tail}")
    return "　".join(parts)


def watch_signature(settings, repo, submission_id: int, camera: str) -> tuple:
    """What the post-match page redraws on: the camera's linecall job state and whether the
    worker seems stopped while that job waits."""
    from pfb_app.services import linecall   # noqa: PLC0415 - only this helper reads the DB
    from pfb_app.ui.wizard.live import worker_stopped   # noqa: PLC0415 - a Streamlit page module
    state = linecall.job_state(repo, submission_id, camera)
    return state, worker_stopped(settings, state == "active")


def reason_text(ep: dict) -> str | None:
    """「判定不能の理由: …」 for a 判定不能 episode with a known reason, else None."""
    if ep.get("verdict") != "undecidable":
        return None
    reason = REASON_LABELS.get(ep.get("reason") or "")
    return REASON_TEXT.format(reason=reason) if reason else None


def listed(episodes: list[dict]) -> list[dict]:
    """The episodes the page lists: those with an id, the first of each id (labels and the
    page's buttons are keyed by it)."""
    seen, out = set(), []
    for ep in episodes:
        eid = ep.get("id")
        if isinstance(eid, (str, int)) and not isinstance(eid, bool) and str(eid) \
                and str(eid) not in seen:
            seen.add(str(eid))
            out.append(ep)
    return out


def page_slice(items: list, offset: int, size: int = PAGE_SIZE) -> tuple[int, list]:
    """(start, items of the page): `offset` rounded down to a page start and held to the last
    page when the list got shorter."""
    if not items:
        return 0, []
    last = (len(items) - 1) // size * size
    start = min(max(0, int(offset)) // size * size, last)
    return start, items[start:start + size]


def page_range_text(start: int, shown: int, total: int) -> str:
    return PAGE_RANGE.format(a=start + 1, b=start + shown, n=total)


# -- court mini-map -------------------------------------------------------------------------

# The engine's court model (pfb3x3/linecall/court.py, also pfb_report court_geometry.json).
CW, CH = 1505.0, 1105.0
BASKET = (752.5, 134.0)
R3 = 674.5
ARC_X = 97.5
ARC_Y = BASKET[1] + math.sqrt(R3 ** 2 - (BASKET[0] - ARC_X) ** 2)   # 295.0132
# The key as fitted by TrackID's autocourt on this court (tid3/autocourt.py KEY_W, KEY_L:
# outer edges of its lines); drawn for orientation only, it is not a line-call line.
KEY_W, KEY_L = 500.0, 555.0
# Court cm shown beyond the outline, so a player just outside a line is not drawn on it.
PAD_CM = 60.0
_S = min(CW / (CW + 2 * PAD_CM), CH / (CH + 2 * PAD_CM))
_OX, _OY = (CW - CW * _S) / 2, (CH - CH * _S) / 2
DOT_CM, RING_CM, NUM_CM = 36.0, 58.0, 40.0


def to_svg(x: float, y: float) -> tuple[float, float]:
    """Court cm -> SVG user units in the 1505 x 1105 viewBox (far baseline at the top)."""
    return round(_OX + x * _S, 1), round(_OY + y * _S, 1)


def _pt(x: float, y: float) -> str:
    sx, sy = to_svg(x, y)
    return f"{sx:g} {sy:g}"


def _court_svg() -> str:
    t = TOKENS
    x0, y0 = to_svg(0, 0)
    x1, y1 = to_svg(CW, CH)
    kx0, _ = to_svg(BASKET[0] - KEY_W / 2, 0)
    kx1, ky1 = to_svg(BASKET[0] + KEY_W / 2, KEY_L)
    bx, by = to_svg(*BASKET)
    r = round(R3 * _S, 1)
    stroke = f'stroke="{t["chalk"]}" stroke-width="{round(8 * _S, 1):g}"'
    line = f'fill="none" {stroke}'
    arc = (f"M {_pt(ARC_X, 0)} L {_pt(ARC_X, ARC_Y)} A {r:g} {r:g} 0 0 0 "
           f"{_pt(CW - ARC_X, ARC_Y)} L {_pt(CW - ARC_X, 0)}")
    return (f'<rect class="surround" x="0" y="0" width="{CW:g}" height="{CH:g}" '
            f'fill="{t["board-2"]}"/>'
            f'<rect class="court" x="{x0:g}" y="{y0:g}" width="{round(x1 - x0, 1):g}" '
            f'height="{round(y1 - y0, 1):g}" fill="{t["board"]}" {stroke}/>'
            f'<rect class="key" x="{kx0:g}" y="{y0:g}" width="{round(kx1 - kx0, 1):g}" '
            f'height="{round(ky1 - y0, 1):g}" {line}/>'
            f'<path class="arc" d="{arc}" {line}/>'
            f'<circle class="basket" cx="{bx:g}" cy="{by:g}" r="{round(23 * _S, 1):g}" {line}/>')


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _int_id(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def minimap_svg(players: list[dict], flagged: set[int], uid: str = "lc-minimap") -> str:
    """The court seen from the camera, one numbered dot per player.

    A flagged player (an open alert) has the warning colour AND a ring around the dot; a
    player not in play has a dashed outline. Rows without finite x/y are skipped; positions
    far off the court are held at the edge of the drawn margin."""
    t = TOKENS
    uid = re.sub(r"[^A-Za-z0-9_-]", "-", uid) or "lc-minimap"
    flagged_ids = {i for i in (_int_id(f) for f in flagged) if i is not None}
    dots = []
    r, ring, fs = (round(v * _S, 1) for v in (DOT_CM, RING_CM, NUM_CM))
    for p in players:
        if not isinstance(p, dict):
            continue
        x, y = _num(p.get("x")), _num(p.get("y"))
        if x is None or y is None:
            continue
        x = min(max(x, -PAD_CM), CW + PAD_CM)
        y = min(max(y, -PAD_CM), CH + PAD_CM)
        cx, cy = to_svg(x, y)
        pid = escape(str(p.get("id")), quote=True)
        is_flagged = _int_id(p.get("id")) in flagged_ids
        in_play = p.get("in_play", True) is not False
        if is_flagged:
            dots.append(f'<circle class="flag-ring" cx="{cx:g}" cy="{cy:g}" r="{ring:g}" '
                        f'fill="none" stroke="{t["chalk"]}" stroke-width="{round(9 * _S, 1):g}"/>')
            style, text = (f'fill="{t["wait"]}" stroke="{t["chalk"]}" '
                           f'stroke-width="{round(4 * _S, 1):g}"'), t["ink"]
        elif in_play:
            style, text = f'fill="{t["chalk"]}" stroke="{t["ink"]}" stroke-width="2"', t["ink"]
        else:
            style, text = (f'fill="{t["board"]}" stroke="{t["chalk-2"]}" stroke-width="4" '
                           f'stroke-dasharray="8 6"'), t["chalk-2"]
        dots.append(f'<circle class="player" data-id="{pid}" cx="{cx:g}" cy="{cy:g}" '
                    f'r="{r:g}" {style}/>'
                    f'<text class="num" x="{cx:g}" y="{cy:g}" fill="{text}" '
                    f'font-size="{fs:g}" font-weight="700" text-anchor="middle" '
                    f'dominant-baseline="central">{pid}</text>')
    desc = minimap_alt(players, flagged)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1505 1105" role="img" '
            f'aria-labelledby="{uid}-title {uid}-desc" width="100%" '
            f'style="display:block;max-width:640px;height:auto" '
            f'font-family="sans-serif">'
            f'<title id="{uid}-title">コート上の選手の位置</title>'
            f'<desc id="{uid}-desc">{escape(desc)}</desc>'
            f'{_court_svg()}{"".join(dots)}</svg>')


def _drawn(players: list[dict]) -> list[dict]:
    return [p for p in players if isinstance(p, dict)
            and _num(p.get("x")) is not None and _num(p.get("y")) is not None]


def minimap_alt(players: list[dict], flagged: set[int]) -> str:
    """The map in words (the SVG's desc, and the alt text of the image)."""
    drawn = _drawn(players)
    flagged_ids = {i for i in (_int_id(f) for f in flagged) if i is not None}
    named = [player_name(p.get("id")) for p in drawn if _int_id(p.get("id")) in flagged_ids]
    return (f"上が奥のエンドライン、下が手前のライン。選手 {len(drawn)}人。"
            + (f"線の近くでアラート中: {'、'.join(named)}。" if named
               else "アラート中の選手はいません。"))


def minimap_data_uri(players: list[dict], flagged: set[int], uid: str = "lc-minimap") -> str:
    """The map as a data URI for st.image / <img> (st.html drops inline SVG). Web fonts do
    not load inside an image SVG, so the map uses the generic sans-serif family."""
    svg = minimap_svg(players, flagged, uid=uid)
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


# -- live stream ----------------------------------------------------------------------------

class StreamTail:
    """Reads the records appended to a stream.jsonl since the last call.

    Keeps a byte offset and the trailing partial line (the engine may be mid-write), so no
    record is dropped or mis-parsed. Starts over when the file is replaced (another inode) or
    shrinks. Complete lines that are not a JSON object are skipped."""

    def __init__(self, path: Path, max_bytes: int = 4 << 20):
        self.path = Path(path)
        self.max_bytes = int(max_bytes)
        self._offset = 0
        self._buf = b""
        self._file: tuple[int, int] | None = None

    def read_new(self) -> list[dict]:
        try:
            with self.path.open("rb") as f:
                st = os.fstat(f.fileno())
                key = (st.st_dev, st.st_ino)
                if self._file is not None and (key != self._file or st.st_size < self._offset):
                    self._offset, self._buf = 0, b""
                self._file = key
                if st.st_size <= self._offset:
                    return []
                f.seek(self._offset)
                data = f.read(self.max_bytes)
        except OSError:
            return []
        self._offset += len(data)
        *lines, self._buf = (self._buf + data).split(b"\n")
        out = []
        for raw in lines:
            if not raw.strip():
                continue
            try:
                rec = json.loads(raw.decode("utf-8"))
            except ValueError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out


class LiveState:
    """What the live page shows, folded from stream records.

    - ``players``: the rows of the latest ``positions`` record;
    - ``alerts``: the last ``max_alerts`` alert / dwell ids, newest (first seen) first, each
      ``{id, kind, record (latest alert/dwell record), needs_check, ended, end}``;
    - ``latency_s``: the latest ``stats`` latency p50;
    - ``dropped`` / ``drop_rate``: frames the engine skipped because it was behind (dropped /
      frames_in of the latest ``stats`` or ``end`` record; None before one);
    - ``ended``: an ``end`` record was seen. A ``start`` record starts over."""

    def __init__(self, max_alerts: int = 20):
        self.max_alerts = int(max_alerts)
        self._reset()

    def _reset(self) -> None:
        self.players: list[dict] = []
        self.alerts: list[dict] = []
        self.latency_s: float | None = None
        self.dropped = 0
        self.drop_rate: float | None = None
        self.ended = False
        self._by_id: dict[str, dict] = {}

    @property
    def flagged(self) -> set[int]:
        """Players with an open alert (none once the stream has ended)."""
        if self.ended:
            return set()
        return {i for a in self.alerts if a["kind"] == "alert" and not a["ended"]
                for i in [_int_id(a["record"].get("player"))] if i is not None}

    def _open(self, key: str, kind: str, rec: dict) -> None:
        entry = self._by_id.get(key)
        needs_check = bool(rec.get("needs_check")) if kind == "alert" else False
        if entry is not None:
            entry["record"], entry["needs_check"] = rec, needs_check
            return
        entry = {"id": key, "kind": kind, "record": rec, "needs_check": needs_check,
                 "ended": False, "end": None}
        self._by_id[key] = entry
        self.alerts.insert(0, entry)
        for old in self.alerts[self.max_alerts:]:
            self._by_id.pop(old["id"], None)
        del self.alerts[self.max_alerts:]

    def _close(self, key, rec: dict) -> None:
        entry = self._by_id.get(key) if isinstance(key, str) else None
        if entry is None:
            return
        entry["ended"], entry["end"] = True, rec
        if entry["kind"] == "alert" and "needs_check" in rec:
            entry["needs_check"] = bool(rec["needs_check"])

    def apply(self, records: list[dict]) -> None:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            kind = rec.get("type")
            if kind == "start":
                self._reset()
            elif kind == "positions":
                rows = rec.get("players")
                self.players = [r for r in rows if isinstance(r, dict)] \
                    if isinstance(rows, list) else []
            elif kind in ("alert", "dwell"):
                key = rec.get("alert_id" if kind == "alert" else "dwell_id")
                if isinstance(key, str):
                    self._open(key, kind, rec)
            elif kind == "alert_end":
                self._close(rec.get("alert_id"), rec)
            elif kind == "dwell_end":
                self._close(rec.get("dwell_id"), rec)
            elif kind == "stats":
                lat = rec.get("latency")
                p50 = _num(lat.get("p50_ms")) if isinstance(lat, dict) else None
                if p50 is not None:
                    self.latency_s = p50 / 1000.0
                self._frames(rec)
            elif kind == "end":
                self.ended = True
                self._frames(rec)

    def _frames(self, rec: dict) -> None:
        frames_in, dropped = _num(rec.get("frames_in")), _num(rec.get("dropped"))
        if frames_in is None or dropped is None or frames_in <= 0 or dropped < 0:
            return
        self.dropped = int(dropped)
        self.drop_rate = min(dropped / frames_in, 1.0)


def alert_text(a: dict) -> str:
    """「7番の人　サイドライン（左）まで 8 cm（内側）」 for an alert, 「… 付近に滞在」 for a
    dwell. Takes a LiveState entry or a raw stream record."""
    rec = a.get("record", a) if isinstance(a.get("record"), dict) else a
    who, line = player_name(rec.get("player")), line_name(rec.get("line"))
    if a.get("kind", rec.get("type")) == "dwell":
        return f"{who}　{line}付近に滞在"
    d = _num(rec.get("distance_cm"))
    dist = str(int(abs(d) + 0.5)) if d is not None else _DASH
    side = SIDE_LABELS.get(rec.get("side"))
    return f"{who}　{line}まで {dist} cm" + (f"（{side}）" if side else "")


# -- live sources ---------------------------------------------------------------------------

DEMO_SOURCE = "demo"
_TID_VIDEO = re.compile(r"IMG_\d+")


@dataclass(frozen=True)
class LiveSource:
    """One choice of the live page: the settings demo or a camera of a sports group."""
    key: str                  # "demo" or "s<submission>_<camera>"
    label: str
    path: Path | None         # None: the demo is not configured
    video_id: str             # the live run's registry id
    tid_video: str | None = None   # --tid-video (the court from TrackID's meta)


def tid_video_for(path: Path | None) -> str | None:
    """The TrackID video of a demo file: IMG_0105.MOV, or T10's 1920 proxy
    IMG_0105_1920.mp4, -> "IMG_0105"; None for any other name."""
    if path is None:
        return None
    m = _TID_VIDEO.match(Path(path).name)
    return m.group(0) if m else None


def section_shown(repo, settings) -> bool:
    """The sidebar shows 判定支援 only with something to judge: a sports video work, or the
    live demo's video."""
    return settings.linecall_demo_video is not None or any(
        work.get("domain") == "sports" and work.get("kind") == "video"
        for work in repo.list_assignments())


def live_sources(repo, settings) -> list[LiveSource]:
    """The demo (always first, also when not configured) and every camera of the sports
    works' video groups, in the works' and groups' order."""
    demo = settings.linecall_demo_video
    out = [LiveSource(DEMO_SOURCE, LIVE_DEMO_LABEL, demo, "linecall_demo", tid_video_for(demo))]
    for work in repo.list_assignments():
        if work.get("domain") != "sports" or work.get("kind") != "video":
            continue
        for sub in repo.list_submissions(work["id"]):
            cameras = (sub.get("meta") or {}).get("cameras")
            if not isinstance(cameras, dict):
                continue
            for cam, path in cameras.items():
                out.append(LiveSource(f"s{sub['id']}_{cam}",
                                      f"{work['title']}（{sub['label']}）{camera_label(cam)}",
                                      Path(path), f"live_s{sub['id']}_{cam}"))
    return out
