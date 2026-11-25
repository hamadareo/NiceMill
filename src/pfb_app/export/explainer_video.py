"""The 解説動画 files: drawing and encoding (services.explainer plans them and runs the job).

A person's video (`render_person`): an intro card (name, one line of the approved text, what
the boxes mean), one part per cited moment and an end card. A moment's part is BEFORE_S
before to AFTER_S after the moment, read from the playable source (services.moment_clip's
rule: a small non-HEVC original or the ready 1080p proxy, never a 4K/HEVC original), laid out
on a 1280x720 canvas:

    +----------------------------+-----------+
    | the video (896x504)        | AIに見せた |
    | the person: white + blue   | 画像       |
    | others: thin grey, no name | (the still |
    | dashed: AIに見せた範囲      |  sent now) |
    +----------------------------+-----------+
    | when · category / what they did / why it counted (216 px) |
    +----------------------------------------------------------+

The dashed rectangle is drawn only for a region that covers less than 90 % of the frame and
that at least two recorded stills confirm, better than moved or resized regions
(export.explainer_ai.check_region), and only while the time is inside the window those stills
are from; the cards and the panel mention the dashed line only when one is drawn. A
moment without a time is a caption card. The 「追跡の様子」 video (`render_tracking`): the
analysed part with every evaluated person's box and name; everyone else thin grey, nameless.

Encoding: PyAV, libx264 (CRF 24, veryfast, yuv420p, +faststart, 2 threads), silent, at the
source's rate up to 30 fps; text in Noto Sans CJK (the PDFs' font), never below 29 px (4 %
of 720).
"""
from __future__ import annotations

import functools
import logging
import math
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pfb_app.export import explainer_ai as ai
from pfb_app.export.explainer_ai import AiInput
from pfb_app.export.tracks import Box, BoxIndex

log = logging.getLogger(__name__)

VERSION = 1                     # bump when the drawing changes: older files are made again
W, H = 1280, 720
VIDEO_W, VIDEO_H = 896, 504
PANEL_W = W - VIDEO_W
BAR_H = H - VIDEO_H
TRACK_BAR_H = 72
BEFORE_S, AFTER_S = 4.0, 3.0
INTRO_S, END_S, CARD_S = 4.0, 3.5, 4.0
MAX_FPS = 30
CRF, PRESET, THREADS = "24", "veryfast", 2
TEXT_MIN = 29                   # px: 4 % of the height
SMALL, BODY, HEAD, NAME = 30, 32, 36, 64
BG = (20, 24, 31)
PANEL = (32, 38, 48)
INK = (245, 247, 250)
MUTED = (190, 198, 210)
ACCENT = (109, 167, 236)        # RAMP 300, readable on BG
BLUE = (42, 120, 214)           # export.pictures.OUTLINE
GREY = (150, 156, 166)
DASH = (255, 214, 64)
FONT_FILES = {
    False: ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"),
    True: ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
           "/usr/share/fonts/noto-cjk/NotoSansCJK-Bold.ttc",
           "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc"),
}
WHY_HEAD = "評価のポイント"
AI_STILLS = "AIに見せた画像"
AI_REGION = "AIに見せた範囲"
AI_NOTE = ("AIの注目の可視化は", "研究中です")
NO_STILLS = ("この場面の画像の", "記録がありません")
NO_RECT = "点線は出していません"
RECT_RULE = "点線は、AIに見せた範囲を記録の画像で確かめられた場面だけに出しています。"


class RenderError(Exception):
    """The video could not be made (a plain Japanese message)."""


class Cancelled(Exception):
    """The job was stopped while the video was being made (nothing is kept)."""


# --- the plan (services.explainer fills it) ---------------------------------------------------

@dataclass
class Source:
    """A camera's playable file and what the tracks say about it."""
    video: Path
    fps: float                       # the tracks' (original) frame rate
    size: tuple[int, int]            # the tracks' (original) frame size
    boxes: BoxIndex
    start_s: float = 0.0             # the analysed start, in the video
    last_frame: int | None = None    # the last analysed frame (the engine's windows end there)


@dataclass
class Part:
    when: str
    category: str | None
    what: str
    why: str
    source: Source | None = None
    at_s: float | None = None        # seconds in the video
    mine: list[str] = field(default_factory=list)   # the person's track labels
    stills: list[str] = field(default_factory=list)  # the behavior run's frames[event_id]
    verified: bool = False           # the dashed region is drawn (set while rendering)


@dataclass
class PersonPlan:
    name: str
    title: str
    date: str
    summary: str
    others: str                      # 「ほかの選手」
    parts: list[Part]
    live: str | None = None          # 「あとから訂正」: 確認状況 and 文の版 (the end card says so)


@dataclass
class TrackingPlan:
    title: str
    group: str
    source: Source
    names: dict[str, str]            # track label -> display name (evaluated people only)
    from_s: float
    to_s: float
    tick: Callable[[float], str]     # seconds since the analysed start -> 「開始から 01:23」


# --- text ------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _fc_files() -> dict[bool, str | None]:
    found: dict[bool, str | None] = {}
    for bold, style in ((False, "Regular"), (True, "Bold")):
        try:
            out = subprocess.run(["fc-match", f"Noto Sans CJK JP:style={style}",
                                  "--format=%{file}"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out = ""
        found[bold] = out if out and "cjk" in out.lower() and Path(out).is_file() else None
    return found


@functools.lru_cache(maxsize=32)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Noto Sans CJK JP (as the PDFs); RenderError when no CJK font is installed."""
    size = max(int(size), TEXT_MIN)
    for path in (*FONT_FILES[bold], _fc_files().get(bold)):
        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, size, index=0)
            except OSError:
                continue
    raise RenderError("日本語の文字（Noto Sans CJK）が見つからないため、動画を作れません。")


def wrap(text: str, f: ImageFont.FreeTypeFont, width: int, lines: int) -> list[str]:
    """`text` in at most `lines` lines of `width` px (the last one ends with … when cut)."""
    out: list[str] = []
    for para in (text or "").split("\n"):
        cur = ""
        for ch in para:
            if f.getlength(cur + ch) > width and cur:
                out.append(cur)
                cur = ch
            else:
                cur += ch
        out.append(cur)
    out = [s for s in out if s.strip()] or [""]
    if len(out) > lines:
        last = out[lines - 1]
        while last and f.getlength(last + "…") > width:
            last = last[:-1]
        out = out[:lines - 1] + [last + "…"]
    return out


def _text(draw: ImageDraw.ImageDraw, xy, lines: list[str], f, fill, gap: int = 8) -> int:
    """Draw lines from (x, y); returns the y below them."""
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def _tag(text: str, fill, size: int = SMALL, ink=(255, 255, 255)) -> np.ndarray:
    """A name tag: text on a filled rounded box, as an RGB array."""
    f = font(size, True)
    w = int(f.getlength(text)) + 16
    img = Image.new("RGB", (w, size + 14), fill)
    ImageDraw.Draw(img).text((8, 3), text, font=f, fill=tuple(ink))
    return np.asarray(img)


def _paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    h, w = img.shape[:2]
    H_, W_ = canvas.shape[:2]
    x, y = int(min(max(x, 0), W_ - w)), int(min(max(y, 0), H_ - h))
    canvas[y:y + h, x:x + w] = img[:H_ - y, :W_ - x]


# --- cards -----------------------------------------------------------------------------------

def _card() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    return img, ImageDraw.Draw(img)


def legend_lines(plan: PersonPlan, dashed: bool) -> list[str]:
    """What the boxes mean (the intro card): the dashed line only when one is drawn."""
    return [f"青い枠：{plan.name}さん　灰色の枠：{plan.others}",
            f"点線：{AI_REGION}　右の画像：{AI_STILLS}" if dashed else f"右の画像：{AI_STILLS}"]


LIVE_END = ("この動画は、AIが作成した文章と、そのもとになった場面の記録から作りました。",
            "人の確認を経ていない場合があります。誤りがあれば訂正できます。")


def end_lines(dashed: bool, live: str | None = None) -> list[str]:
    note = f"{AI_NOTE[0]}{AI_NOTE[1]}"
    lines = ([*LIVE_END, live] if live else
             ["この動画は、承認した文章と、そのもとになった場面の記録から作りました。"]) + [
             f"点線と右の画像は、AIに渡した範囲と画像です（{note}）。" if dashed
             else f"右の画像は、AIに渡した画像です（{note}）。"]
    return lines + [RECT_RULE] if dashed else lines


def intro_card(plan: PersonPlan, dashed: bool = False) -> np.ndarray:
    img, d = _card()
    x, width = 96, W - 192
    y = _text(d, (x, 72), wrap(f"{plan.title}　{plan.date}".strip(), font(SMALL), width, 1),
              font(SMALL), MUTED)
    y = _text(d, (x, y + 4), ["解説動画"], font(SMALL, True), ACCENT)
    y = _text(d, (x, y + 8), wrap(f"{plan.name}さん", font(NAME, True), width, 1),
              font(NAME, True), INK)
    y = _text(d, (x, y + 16), wrap(plan.summary, font(HEAD), width, 2), font(HEAD), INK)
    y = _text(d, (x, y + 16), [f"よかった場面 {len(plan.parts)}つ"], font(BODY, True), ACCENT)
    legend = legend_lines(plan, dashed)
    _text(d, (x, max(y + 24, H - 150)), [wrap(s, font(SMALL), width, 1)[0] for s in legend],
          font(SMALL), MUTED)
    return np.asarray(img)


def end_card(plan: PersonPlan, dashed: bool = False) -> np.ndarray:
    img, d = _card()
    x, width = 96, W - 192
    y = _text(d, (x, 180), wrap(f"{plan.name}さんのよかった場面は以上です", font(HEAD + 4, True),
                                width, 2), font(HEAD + 4, True), INK)
    for s in end_lines(dashed, plan.live):
        y = _text(d, (x, y + 20), wrap(s, font(SMALL), width, 2), font(SMALL), MUTED)
    return np.asarray(img)


def text_card(part: Part) -> np.ndarray:
    """A moment whose time (or video) is unknown: its caption, without pictures."""
    img, d = _card()
    x, width = 96, W - 192
    y = _text(d, (x, 120), wrap(_head_line(part), font(HEAD, True), width, 1),
              font(HEAD, True), ACCENT)
    y = _text(d, (x, y + 16), wrap(part.what, font(HEAD), width, 3), font(HEAD), INK)
    y = _text(d, (x, y + 16), wrap(caption_lines(part)[2], font(BODY), width, 3),
              font(BODY), INK)
    _text(d, (x, y + 24), ["この場面の動画はありません。"], font(SMALL), MUTED)
    return np.asarray(img)


def _head_line(part: Part) -> str:
    return f"{part.when}　{part.category}" if part.category else part.when


def caption_lines(part: Part) -> tuple[str, str, str]:
    """(when and category, what they did, why it counted): the caption's three parts."""
    return _head_line(part), part.what, f"{WHY_HEAD}：{part.why}"


def caption_bar(part: Part) -> np.ndarray:
    img = Image.new("RGB", (W, BAR_H), BG)
    d = ImageDraw.Draw(img)
    x, width = 32, W - 64
    head, what, why = caption_lines(part)
    y = _text(d, (x, 12), wrap(head, font(HEAD, True), width, 1), font(HEAD, True), ACCENT,
              gap=6)
    y = _text(d, (x, y), wrap(what, font(BODY), width, 2), font(BODY), INK, gap=4)
    _text(d, (x, y + 2), wrap(why, font(SMALL), width, 2), font(SMALL), MUTED, gap=4)
    return np.asarray(img)


def side_panel(info: AiInput, k: int | None, verified: bool) -> np.ndarray:
    img = Image.new("RGB", (PANEL_W, VIDEO_H), PANEL)
    d = ImageDraw.Draw(img)
    x, width = 12, PANEL_W - 24
    y = _text(d, (x, 10), [AI_STILLS], font(SMALL, True), INK)
    if k is None:
        y = _text(d, (x, y + 8), list(NO_STILLS), font(SMALL), MUTED)
    else:
        try:
            still = Image.open(info.stills[k]).convert("RGB")
            still.thumbnail((width, 205), Image.LANCZOS)
            img.paste(still, (x, y + 4))
            y += still.height + 12
        except OSError:
            y = _text(d, (x, y + 8), list(NO_STILLS), font(SMALL), MUTED)
        y = _text(d, (x, y), [f"{k + 1} / {len(info.stills)} 枚め"], font(SMALL), INK)
        if verified:
            y = _text(d, (x, y), [f"点線＝{AI_REGION}"], font(SMALL), DASH)
        else:
            y = _text(d, (x, y), [NO_RECT], font(SMALL), MUTED)
    _text(d, (x, VIDEO_H - 2 * (SMALL + 6) - 10), list(AI_NOTE), font(SMALL), MUTED, gap=6)
    return np.asarray(img)


# --- frames ----------------------------------------------------------------------------------

def fit(frame_w: int, frame_h: int, box_w: int, box_h: int) -> tuple[float, int, int, int, int]:
    """(scale, x, y, w, h) of a frame letterboxed into box_w x box_h."""
    k = min(box_w / frame_w, box_h / frame_h)
    w, h = max(2, round(frame_w * k)), max(2, round(frame_h * k))
    return k, (box_w - w) // 2, (box_h - h) // 2, w, h


def _rect(img: np.ndarray, b: tuple[float, float, float, float], colour, width: int,
          line=cv2.LINE_AA) -> None:
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    cv2.rectangle(img, (x0, y0), (x1, y1), colour, width, line)


def _dashed(img: np.ndarray, b: tuple[float, float, float, float], colour, width: int = 3,
            dash: int = 14) -> None:
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                               ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        length = math.hypot(bx - ax, by - ay)
        n = max(1, int(length // dash))
        for i in range(0, n, 2):
            s, e = i / n, min((i + 1) / n, 1.0)
            cv2.line(img, (round(ax + (bx - ax) * s), round(ay + (by - ay) * s)),
                     (round(ax + (bx - ax) * e), round(ay + (by - ay) * e)), (0, 0, 0),
                     width + 2, cv2.LINE_AA)
            cv2.line(img, (round(ax + (bx - ax) * s), round(ay + (by - ay) * s)),
                     (round(ax + (bx - ax) * e), round(ay + (by - ay) * e)), colour, width,
                     cv2.LINE_AA)


def draw_boxes(img: np.ndarray, boxes: list[tuple[str, Box]], scale: tuple[float, float],
               offset: tuple[int, int], highlight: set[str],
               tags: dict[str, np.ndarray]) -> None:
    """On an RGB frame: thin grey boxes for everyone not in `highlight`, then the highlighted
    ones in white + blue (as the PDF pictures) with their tag above (no tag for others)."""
    sx, sy = scale
    ox, oy = offset

    def at(b: Box):
        return (ox + b.x * sx, oy + b.y * sy, ox + (b.x + b.w) * sx, oy + (b.y + b.h) * sy)
    for label, b in boxes:
        if label not in highlight:
            _rect(img, at(b), GREY, 2, cv2.LINE_8)
    for label, b in boxes:
        if label in highlight:
            r = at(b)
            _rect(img, r, (255, 255, 255), 6)
            _rect(img, r, BLUE, 3)
            tag = tags.get(label)
            if tag is not None:
                _paste(img, tag, int(r[0]), int(r[1]) - tag.shape[0] - 4)


class _Reader:
    """Sequential frames of a video by time (OpenCV: the display rotation applied)."""

    def __init__(self, path: Path):
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RenderError("動画を読めないため、解説動画を作れません。")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.pos: int | None = None
        self.last: np.ndarray | None = None

    @property
    def duration(self) -> float:
        return self.count / self.fps if self.count else 0.0

    def at(self, t: float) -> np.ndarray | None:
        """The frame shown at `t` seconds (BGR), reading on from the last one when close."""
        want = max(0, int(round(t * self.fps)))
        if self.pos is not None and want == self.pos - 1 and self.last is not None:
            return self.last
        if self.pos is None or not 0 <= want - self.pos <= round(4 * self.fps):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, want)
            self.pos = want
        while self.pos < want:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        if not ok:
            return None
        self.pos += 1
        self.last = frame
        return frame

    def close(self) -> None:
        self.cap.release()


def _check_shape(reader_size: tuple[int, int], meta_size: tuple[int, int]) -> None:
    rw, rh = reader_size
    mw, mh = meta_size
    if not (rw and rh and mw and mh) or abs(rw / rh - mw / mh) > 0.03 * (mw / mh):
        raise RenderError("動画の向きや大きさが追跡の記録と合わないため、枠を正しく描けません。")


# --- encoding --------------------------------------------------------------------------------

class Encoder:
    def __init__(self, path: Path, fps: float):
        import av   # noqa: PLC0415
        self.rate = Fraction(fps).limit_denominator(1001)
        self.out = av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"})
        self.stream = self.out.add_stream("libx264", rate=self.rate, options={
            "crf": CRF, "preset": PRESET, "g": str(max(1, round(float(self.rate) * 2)))})
        self.stream.width, self.stream.height = W, H
        self.stream.pix_fmt = "yuv420p"
        self.stream.codec_context.thread_count = THREADS
        self.time_base = Fraction(1, 1) / self.rate
        self.stream.codec_context.time_base = self.time_base
        self.frames = 0

    def write(self, rgb: np.ndarray, times: int = 1) -> None:
        import av   # noqa: PLC0415
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame = frame.reformat(format="yuv420p")
        for _ in range(times):
            frame.pts = self.frames
            frame.time_base = self.time_base
            for packet in self.stream.encode(frame):
                self.out.mux(packet)
            self.frames += 1

    def close(self) -> None:
        for packet in self.stream.encode():
            self.out.mux(packet)
        self.out.close()

    def abort(self) -> None:
        try:
            self.out.close()
        except Exception:   # noqa: BLE001 - the part file is removed by the caller
            pass

    @property
    def seconds(self) -> float:
        return self.frames / float(self.rate)


def out_fps(sources: list[Source]) -> float:
    for s in sources:
        cap = cv2.VideoCapture(str(s.video))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
        finally:
            cap.release()
        if fps and fps > 0:
            return fps if fps <= MAX_FPS + 0.5 else fps / math.ceil(fps / MAX_FPS)
    return float(MAX_FPS)


# --- a person's video ------------------------------------------------------------------------

Progress = Callable[[float, str], None]


def _noop(_f: float, _n: str) -> None:
    pass


def render_person(plan: PersonPlan, target: Path, *, progress: Progress = _noop,
                  stop: Callable[[], bool] = lambda: False) -> dict:
    """Write the person's video to `target`; returns {seconds, verified (parts with a drawn
    region), parts}. RenderError / Cancelled: the caller removes `target`.

    First every part's AI input is read and its region checked (explainer_ai.check_region),
    so the cards mention the dashed line only when one is drawn (each preparation is preceded
    by a stop() check and a progress note: they beat the job's heartbeat)."""
    fps = out_fps([p.source for p in plan.parts if p.source is not None])
    prepared: list[_Prepared | None] = []
    for n, p in enumerate(plan.parts, 1):
        if p.source is None or p.at_s is None:
            prepared.append(None)
            continue
        # before each part's preparation (frames read, the region checked): stop() beats the
        # job's heartbeat and honours a cancel, so a long preparation neither looks hung to
        # the worker's supervisor (explainer.HUNG_SEC) nor ignores a stop
        if stop():
            raise Cancelled()
        progress(0.0, f"場面を確かめています {n} / {len(plan.parts)}")
        prepared.append(_prepare(p))
    dashed = any(p.verified for p in plan.parts)
    enc = Encoder(target, fps)
    try:
        enc.write(intro_card(plan, dashed), round(INTRO_S * fps))
        total = len(plan.parts) or 1
        for n, (part, prep) in enumerate(zip(plan.parts, prepared), 1):
            if stop():
                raise Cancelled()
            progress((n - 1) / total, f"場面 {n} / {len(plan.parts)}")
            if prep is None or not _clip(enc, part, prep, fps, stop, f"{plan.name}さん"):
                enc.write(text_card(part), round(CARD_S * fps))
        enc.write(end_card(plan, dashed), round(END_S * fps))
        enc.close()
    except BaseException:
        enc.abort()
        raise
    progress(1.0, "仕上げ")
    return {"seconds": round(enc.seconds, 1), "parts": len(plan.parts),
            "verified": sum(1 for p in plan.parts if p.verified)}


@dataclass
class _Prepared:
    info: AiInput
    region_px: tuple[float, float, float, float] | None
    scale: tuple[float, float]
    size: tuple[int, int]             # the decoded frame


def _prepare(part: Part) -> _Prepared | None:
    """The part's AI input and whether its region may be drawn (sets part.verified); None
    when the video has no frame for it."""
    src = part.source
    reader = _Reader(src.video)
    try:
        first = reader.at(0.0)
        if first is None:
            return None
        fh, fw = first.shape[:2]
        _check_shape((fw, fh), src.size)
        if reader.duration and part.at_s > reader.duration:
            return None
        info = ai.ai_input(part.stills, src.boxes, src.fps, src.size, src.start_s,
                           src.last_frame)
        sx, sy = fw / src.size[0], fh / src.size[1]
        found = ai.check_region(info, lambda f: reader.at((f - 1) / src.fps), src.size,
                                (sx, sy), round(part.at_s * src.fps) + 1)
        part.verified = found.drawn
        region_px = None
        if found.drawn:
            x0, y0, x1, y1 = info.region
            region_px = (x0 * sx, y0 * sy, x1 * sx, y1 * sy)
        return _Prepared(info, region_px, (sx, sy), (fw, fh))
    finally:
        reader.close()


def _clip(enc: Encoder, part: Part, prep: _Prepared, fps: float, stop, tag: str) -> bool:
    """The moment's part on `enc`; False when the video has no frame there."""
    src, info = part.source, prep.info
    reader = _Reader(src.video)
    try:
        fw, fh = prep.size
        sx, sy = prep.scale
        end = reader.duration or (part.at_s + AFTER_S)
        t0, t1 = max(part.at_s - BEFORE_S, 0.0), min(part.at_s + AFTER_S, end)
        k_scale, ox, oy, vw, vh = fit(fw, fh, VIDEO_W, VIDEO_H)
        bar = caption_bar(part)
        panels: dict[int | None, np.ndarray] = {}
        mine = set(part.mine)
        tags = {lb: _tag(tag, BLUE) for lb in mine}
        n = max(1, int(round((t1 - t0) * fps)))
        written = 0
        for i in range(n):
            if i % 30 == 0 and stop():
                raise Cancelled()
            t = t0 + i / fps
            frame = reader.at(t)
            if frame is None:
                break
            vf = round(t * src.fps) + 1                      # the tracks' frame number
            canvas = np.empty((H, W, 3), dtype=np.uint8)
            canvas[:] = BG
            small = cv2.cvtColor(cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB)
            draw_boxes(small, ai.boxes_at(src.boxes, vf), (sx * k_scale, sy * k_scale), (0, 0),
                       mine, tags)
            if prep.region_px is not None and info.frames[0] <= vf <= info.frames[-1]:
                x0, y0, x1, y1 = prep.region_px
                _dashed(small, (x0 * k_scale, y0 * k_scale, x1 * k_scale, y1 * k_scale), DASH)
            # where the moment is in this part: a thin bar under the picture
            cv2.rectangle(small, (0, vh - 8), (vw, vh), (60, 66, 76), -1)
            cv2.rectangle(small, (0, vh - 8), (int(vw * (t - t0) / max(t1 - t0, 1e-6)), vh),
                          ACCENT, -1)
            mx = int(vw * (part.at_s - t0) / max(t1 - t0, 1e-6))
            cv2.rectangle(small, (mx - 3, vh - 18), (mx + 3, vh), DASH, -1)
            canvas[oy:oy + vh, ox:ox + vw] = small
            k = info.still_at(vf)
            if k not in panels:
                panels[k] = side_panel(info, k, part.verified)
            canvas[:VIDEO_H, VIDEO_W:] = panels[k]
            canvas[VIDEO_H:] = bar
            enc.write(canvas)
            written += 1
        return written > 0
    finally:
        reader.close()


# --- 追跡の様子 ---------------------------------------------------------------------------------

def render_tracking(plan: TrackingPlan, target: Path, *, progress: Progress = _noop,
                    stop: Callable[[], bool] = lambda: False) -> dict:
    src = plan.source
    fps = out_fps([src])
    enc = Encoder(target, fps)
    reader = _Reader(src.video)
    try:
        first = reader.at(plan.from_s)
        if first is None:
            raise RenderError("動画のこの部分を読めないため、追跡の様子を作れません。")
        fh, fw = first.shape[:2]
        _check_shape((fw, fh), src.size)
        enc.write(_tracking_card(plan), round(INTRO_S * fps))
        sx, sy = fw / src.size[0], fh / src.size[1]
        k_scale, ox, oy, vw, vh = fit(fw, fh, W, H - TRACK_BAR_H)
        tags = {lb: _tag(name, BLUE) for lb, name in plan.names.items()}
        named = set(plan.names)
        to_s = min(plan.to_s, reader.duration or plan.to_s)
        n = max(1, int(round((to_s - plan.from_s) * fps)))
        bars: dict[int, np.ndarray] = {}
        for i in range(n):
            if i % 30 == 0:
                if stop():
                    raise Cancelled()
                progress(i / n, f"{int(i / fps)}秒 / {int(n / fps)}秒")
            t = plan.from_s + i / fps
            frame = reader.at(t)
            if frame is None:
                break
            vf = round(t * src.fps) + 1
            canvas = np.empty((H, W, 3), dtype=np.uint8)
            canvas[:] = BG
            small = cv2.cvtColor(cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA),
                                 cv2.COLOR_BGR2RGB)
            draw_boxes(small, ai.boxes_at(src.boxes, vf), (sx * k_scale, sy * k_scale), (0, 0),
                       named, tags)
            canvas[oy:oy + vh, ox:ox + vw] = small
            sec = int(t - src.start_s)
            if sec not in bars:
                bars[sec] = _tracking_bar(plan, t - src.start_s)
            canvas[H - TRACK_BAR_H:] = bars[sec]
            enc.write(canvas)
        enc.close()
    except BaseException:
        enc.abort()
        raise
    finally:
        reader.close()
    progress(1.0, "仕上げ")
    return {"seconds": round(enc.seconds, 1), "people": len(plan.names)}


def _tracking_card(plan: TrackingPlan) -> np.ndarray:
    img, d = _card()
    x, width = 96, W - 192
    y = _text(d, (x, 120), wrap(f"{plan.title}　{plan.group}", font(SMALL), width, 1),
              font(SMALL), MUTED)
    y = _text(d, (x, y + 8), ["追跡の様子"], font(NAME, True), INK)
    lines = ["解析で、だれがどこにいたかを記録した様子です。",
             "青い枠と名前：評価した人　灰色の枠：ほかの人（名前は出しません）",
             f"{AI_NOTE[0]}{AI_NOTE[1]}。"]
    for s in lines:
        y = _text(d, (x, y + 16), wrap(s, font(BODY), width, 2), font(BODY), INK)
    return np.asarray(img)


def _tracking_bar(plan: TrackingPlan, t: float) -> np.ndarray:
    img = Image.new("RGB", (W, TRACK_BAR_H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 16), f"追跡の様子　{plan.tick(max(t, 0.0))}", font=font(SMALL, True), fill=INK)
    legend = "青い枠：評価した人　灰色：ほかの人"
    f = font(SMALL)
    d.text((W - 24 - f.getlength(legend), 16), legend, font=f, fill=MUTED)
    return np.asarray(img)
