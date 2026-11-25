"""A person's highlight video (`render_highlight`): the cited moments cut from the original
match video, joined, with the person's box and name, the generated sentences as subtitles, the
time, and a plain note that this is the AI's draft.

Only ordinary editing: frames of the playable file (services.moment_clip's source rule) are read
as they are, a box / text / bar is drawn on them, and PyAV encodes them (as the 解説動画 does,
export.explainer_video.Encoder). Nothing is generated or repainted by a model.

    +--[ 動画の時刻 03:05 · 選手3                        AIの判定・要確認 ]--+
    |  the video (the person: white + blue box and name; others: thin grey) |
    |  ▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬▬ (where we are in this scene, ▮ = the moment)       |
    +-----------------------------------------------------------------------+
    | 試合時間 12:30 · パス                                      (solid band) |
    | AIの記録　<what, as generated>                                          |
    | 評価のポイント　<why, as generated>                                      |
    | AIが作成した文です。違っていれば、あとから直せます。                        |
    +-----------------------------------------------------------------------+

A moment the preliminary identity check flagged (Moment.suspect) is drawn with an orange box
and tag and an orange line in the band: 「別の人の動きかもしれません」. Scenes that overlap
(or nearly touch) on the same camera are one continuous scene; the sentences change at each moment.
The sentences are the generated ones, unchanged; one that does not fit in three lines ends with …
(the review player shows it whole).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pfb_app.export import explainer_ai as ai
from pfb_app.export import explainer_video as ev
from pfb_app.export.explainer_video import Cancelled, RenderError
from pfb_app.review_video import holder, model
from pfb_app.review_video.model import (BAND_NOTE, BAND_NOTE_VERIFIED, CHECK_NOTE, CHECKED,
                                        DRAFT_NOTICE, FIX_NOTE, GATE_NOTE, KIND_CHANGE, KIND_GOOD,
                                        NO_MARK_NOTE, NUMBER_NOTE, NUMBER_NOTE_UNVERIFIED,
                                        SUSPECT_NOTE, Moment, Package, Person)

log = logging.getLogger(__name__)

BEFORE_S, AFTER_S, JOIN_GAP_S = 4.0, 3.0, 0.5
INTRO_S, END_S = 4.5, 5.0
HUD_H = 46
TL_H = 30                      # the whole-match timeline strip under the picture
ORANGE = (255, 150, 30)
MAX_TEXT_LINES = 3
PAD = 24
LINE = ev.BODY + 8

Progress = Callable[[float, str], None]


def _noop(_f: float, _n: str) -> None:
    pass


@dataclass
class Segment:
    camera: str
    t0: float
    t1: float
    moments: list[Moment] = field(default_factory=list)


def segments(moments: list[Moment], length_of: Callable[[str], float | None]) -> list[Segment]:
    """Scenes: BEFORE_S before to AFTER_S after each moment, scenes of one camera that overlap
    or are closer than JOIN_GAP_S joined. `length_of(camera)`: the video's length (None: unknown)."""
    out: list[Segment] = []
    for m in sorted(moments, key=lambda m: (m.camera or "", m.t_s)):
        end = length_of(m.camera)
        t0, t1 = max(m.t_s - BEFORE_S, 0.0), m.t_s + AFTER_S
        if end:
            t1 = min(t1, end)
            if t0 >= end:
                continue
        cur = out[-1] if out else None
        if cur is not None and cur.camera == m.camera and t0 <= cur.t1 + JOIN_GAP_S:
            cur.t1 = max(cur.t1, t1)
            cur.moments.append(m)
        else:
            out.append(Segment(m.camera, t0, t1, [m]))
    out.sort(key=lambda s: s.t0)
    return out


def active(seg: Segment, t: float) -> Moment:
    """The moment whose sentences show at time t: the latest one that is within 1.5 s of
    starting (the first one before any)."""
    best = seg.moments[0]
    for m in seg.moments:
        if m.t_s - 1.5 <= t:
            best = m
    return best


def clock(t: float) -> str:
    t = max(int(t), 0)
    return f"{t // 60:02d}:{t % 60:02d}"


# --- the band, the HUD ---------------------------------------------------------------------------

def main_text(m: Moment) -> str:
    """The one sentence a scene shows: the generated reason it counted (「よかった所」), else the
    generated observation. The other is in the review player (くわしく)."""
    return ("要確認：" if m.weak else "") + (m.why or m.what)


def caption_lines(m: Moment) -> tuple[str, list[str]]:
    head = "　".join(s for s in (m.when, m.category) if s)
    return head, ev.wrap(main_text(m), ev.font(ev.BODY), ev.W - 2 * PAD, MAX_TEXT_LINES)


def band_height(moments: list[Moment]) -> int:
    n = 0
    suspect = False
    for m in moments:
        _, lines = caption_lines(m)
        n = max(n, len(lines))
        suspect = suspect or m.suspect
    return 12 + 40 + n * LINE + (LINE if suspect else 0) + LINE + 8


def caption_band(m: Moment, h: int, verified: bool = False) -> np.ndarray:
    img = Image.new("RGB", (ev.W, h), ev.PANEL)
    d = ImageDraw.Draw(img)
    head, lines = caption_lines(m)
    y = 10
    d.text((PAD, y), head, font=ev.font(ev.HEAD, True), fill=ev.ACCENT)
    y += 44
    y = ev._text(d, (PAD, y), lines, ev.font(ev.BODY), ev.INK, gap=8)
    if m.suspect:
        y = ev._text(d, (PAD, y), [SUSPECT_NOTE], ev.font(ev.SMALL, True), ORANGE, gap=8)
    ev._text(d, (PAD, h - LINE - 4), [BAND_NOTE_VERIFIED if verified else BAND_NOTE],
             ev.font(ev.SMALL), ev.MUTED, gap=8)
    return np.asarray(img)


def hud(width: int, left: str, right: str, right_fill) -> np.ndarray:
    img = Image.new("RGB", (width, HUD_H), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((PAD, 5), left, font=ev.font(ev.SMALL, True), fill=(255, 255, 255))
    f = ev.font(ev.SMALL, True)
    d.text((width - PAD - f.getlength(right), 5), right, font=f, fill=right_fill)
    return np.asarray(img)


def _blend(dst: np.ndarray, top: np.ndarray, alpha: float) -> None:
    h = top.shape[0]
    dst[:h] = (dst[:h] * (1 - alpha) + top * alpha).astype(np.uint8)


# --- cards ---------------------------------------------------------------------------------------

def intro_card(pkg: Package, person: Person, scenes: int, suspects: int) -> np.ndarray:
    img, d = ev._card()
    y = 34
    d.text((PAD * 2, y), person.shown(), font=ev.font(ev.NAME, True), fill=ev.INK)
    y += 80
    small = [f"NiceMill上の名前：{person.name}"
             + (f"　追跡ID：{person.internal}" if person.internal else "")]
    y = ev._text(d, (PAD * 2, y), small, ev.font(ev.SMALL), ev.MUTED, gap=6)
    if person.label_note:
        y = ev._text(d, (PAD * 2, y), [person.label_note], ev.font(ev.SMALL, True), ORANGE, gap=6)
    y += 8
    title = pkg.title if model.date_in_title(pkg.title, pkg.date) else f"{pkg.title}　{pkg.date}"
    y = ev._text(d, (PAD * 2, y), [f"{title}　振り返り用のハイライト"], ev.font(ev.HEAD),
                 ev.ACCENT, gap=8)
    lines = [f"AIの記録をもとに、元の動画から {scenes} 場面を切り出して並べました。",
             f"枠の色はチームごと（{model.TEAM_WORDS[person.team_letter]}）。この人の枠：{person.shown()}" + ("（点線：番号は未検証）" if person.dashed else "")
             + (f"　灰色の枠：{pkg.others}" if SHOW_OTHERS else "")]
    if suspects:
        lines.append(f"点線の枠：別の人かもしれない場面（{suspects} 場面）")
    if pkg.holders:
        lines.append("赤い丸：ボールを持っていそうな人（AIの予想。外れたり、出ない時間のほうが長かったりします）")
    weak_n = sum(1 for m in person.highlights() if m.weak)
    y = ev._text(d, (PAD * 2, y + 6), lines, ev.font(ev.BODY), ev.INK, gap=8)
    y = ev._text(d, (PAD * 2, y + 8), [DRAFT_NOTICE] + (
        ["文は、AIが作ったものです。", f"「要確認」の字幕（{weak_n} 場面）は、断定しない形に直してあります。"] if weak_n
        else ["文は、AIが作ったものをそのまま字幕にしています。"]),
                 ev.font(ev.BODY, True), ev.ACCENT, gap=8)
    small_lines: list[str] = []
    f = ev.font(ev.SMALL)
    wide = ev.W - 4 * PAD
    if not person.verified:
        small_lines += ev.wrap(GATE_NOTE, f, wide, 2)
    small_lines += ev.wrap(NO_MARK_NOTE, f, wide, 2) + [CHECKED[person.checked]]
    ev._text(d, (PAD * 2, y + 8), small_lines, f, ev.MUTED, gap=6)
    return np.asarray(img)


def end_card(pkg: Package, person: Person) -> np.ndarray:
    img, d = ev._card()
    y = 36
    y = ev._text(d, (PAD * 2, y), [DRAFT_NOTICE], ev.font(ev.HEAD, True), ev.ACCENT, gap=8)
    body = ["この動画は、元の映像を切り出して、枠と文字を重ねたものです。",
            "AIが映像を作り替えたものではありません。",
            "人物の取り違えや、文の間違いがないか、元の映像で確かめてください。"]
    if person.status != "approved":
        body.append("このときの文は、まだ確認が済んでいません。")
    y = ev._text(d, (PAD * 2, y + 8), body, ev.font(ev.BODY), ev.INK, gap=8)
    f = ev.font(ev.SMALL)
    wide = ev.W - 4 * PAD
    small = [NUMBER_NOTE if person.verified else NUMBER_NOTE_UNVERIFIED, FIX_NOTE]
    if not person.verified:
        small += ev.wrap(GATE_NOTE, f, wide, 2)
    small += ev.wrap(NO_MARK_NOTE, f, wide, 2)
    y = ev._text(d, (PAD * 2, y + 10), small, f, ev.MUTED, gap=6)
    credit = [c for c in pkg.credit_lines if c]
    if credit:
        wrapped: list[str] = []
        for c in credit:
            wrapped += ev.wrap(c, f, wide, 3)
        ev._text(d, (PAD * 2, y + 12), wrapped, f, ev.MUTED, gap=6)
    return np.asarray(img)


# --- frames --------------------------------------------------------------------------------------

SHOW_OTHERS = False     # thin grey boxes on everyone else (referees, bench and onlookers get one too:
                        # off by default; the CLI's --others turns it on)


def _draw_boxes(img: np.ndarray, boxes, mine: set[str], scale: tuple[float, float],
                colour, tag: np.ndarray, dashed: bool = False) -> None:
    sx, sy = scale

    def at(b):
        return (b.x * sx, b.y * sy, (b.x + b.w) * sx, (b.y + b.h) * sy)
    for label, b in boxes:
        if label not in mine and SHOW_OTHERS:
            ev._rect(img, at(b), ev.GREY, 2, cv2.LINE_8)
    for label, b in boxes:
        if label in mine:
            r = at(b)
            if dashed:      # a number the precision gate has not vouched for: a dashed box
                ev._dashed(img, r, colour, 4, dash=10)
            else:
                ev._rect(img, r, (255, 255, 255), 6)
                ev._rect(img, r, colour, 3)
            ev._paste(img, tag, int(r[0]), int(r[1]) - tag.shape[0] - 4)


COURT_YELLOW = (240, 228, 66)    # the registered court's outline (RGB)
BALL_RED = (224, 32, 32)      # red is for the ball holder only (RGB)


def _ball_ring(img: np.ndarray, boxes, holder: str, scale: tuple[float, float]) -> None:
    """A red ellipse lying on the floor under the ball holder (white rim first), if their box is here."""
    for label, b in boxes:
        if label == holder:
            cx, cy = (b.x + b.w / 2) * scale[0], (b.y + b.h) * scale[1]
            rx = max(16, int(b.w * scale[0] * 0.75))
            ry = max(5, int(rx * 0.32))
            cv2.ellipse(img, (int(cx), int(cy)), (rx, ry), 0, 0, 360, (255, 255, 255), 7, cv2.LINE_AA)
            cv2.ellipse(img, (int(cx), int(cy)), (rx, ry), 0, 0, 360, BALL_RED, 4, cv2.LINE_AA)


def strip_x(t: float, total: float) -> int:
    return int(PAD + (ev.W - 2 * PAD) * min(max(t / max(total, 1e-6), 0.0), 1.0))


def timeline_strip(person: Person, camera: str, total: float) -> np.ndarray:
    """The whole match as one thin strip (RGB, TL_H x W): the person's recorded actions as small
    dots, the よかった所 as yellow diamonds, the places flagged as 「別の人かも」 as orange
    triangles (shape and colour, never colour alone). Everyone's own strip: no other person."""
    img = np.empty((TL_H, ev.W, 3), dtype=np.uint8)
    img[:] = (40, 46, 58)
    cv2.line(img, (PAD, TL_H // 2), (ev.W - PAD, TL_H // 2), (90, 98, 112), 2)
    mid = TL_H // 2
    marks = [m for m in person.everything() if m.camera == camera and m.t_s is not None]
    order = {"event": 0, "good": 1, "change": 2}
    for m in sorted(marks, key=lambda m: order.get(m.kind, 0)):
        x = strip_x(m.t_s, total)
        if m.kind == KIND_GOOD:       # always a diamond (cited: white rim); orange when flagged
            pts = np.array([[x, mid - 11], [x + 9, mid], [x, mid + 11], [x - 9, mid]], np.int32)
            cv2.fillPoly(img, [pts], ORANGE if (m.suspect or m.weak) else ev.DASH, cv2.LINE_AA)
            if m.cited:
                cv2.polylines(img, [pts], True, (255, 255, 255), 2, cv2.LINE_AA)
        elif m.kind == KIND_CHANGE or m.suspect:
            pts = np.array([[x, mid - 9], [x - 8, mid + 8], [x + 8, mid + 8]], np.int32)
            cv2.fillPoly(img, [pts], ORANGE, cv2.LINE_AA)
        else:
            cv2.circle(img, (x, mid), 4, (150, 170, 200), -1, cv2.LINE_AA)
    return img


def compose(frame_bgr: np.ndarray, boxes, mine: set[str], src_scale: tuple[float, float],
            seg_len: float, t_in: float, m: Moment, t: float, person: Person, band: np.ndarray,
            video_h: int, strip: np.ndarray | None = None, total: float = 0.0,
            holder: str | None = None, court=None) -> np.ndarray:
    """One output frame (RGB, W x H)."""
    canvas = np.empty((ev.H, ev.W, 3), dtype=np.uint8)
    canvas[:] = ev.BG
    fh, fw = frame_bgr.shape[:2]
    pic_h = video_h - (TL_H if strip is not None else 0)
    k, ox, oy, vw, vh = ev.fit(fw, fh, ev.W, pic_h)
    small = cv2.cvtColor(cv2.resize(frame_bgr, (vw, vh), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2RGB)
    near = abs(t - m.t_s) <= max(BEFORE_S, AFTER_S)
    flagged = m.suspect and near
    pc = model.team_color(person.team_letter)
    colour = pc["rgb"]                       # the person's own colour; a flag is a dotted box + words
    tag = ev._tag((f"{person.tag()}（別の人かも）" if flagged else person.tag()), colour,
                  ink=pc["ink_rgb"])
    _draw_boxes(small, boxes, mine, (src_scale[0] * k, src_scale[1] * k), colour, tag,
                person.dashed or flagged)
    if court is not None:
        pts = np.array([[x * src_scale[0] * k, y * src_scale[1] * k] for x, y in court], np.int32)
        ov = small.copy()
        cv2.polylines(ov, [pts], True, COURT_YELLOW, 6, cv2.LINE_AA)
        small[:] = cv2.addWeighted(ov, 0.65, small, 0.35, 0)
    if holder is not None:
        _ball_ring(small, boxes, holder, (src_scale[0] * k, src_scale[1] * k))
    cv2.rectangle(small, (0, vh - 8), (vw, vh), (60, 66, 76), -1)
    cv2.rectangle(small, (0, vh - 8), (int(vw * t_in / max(seg_len, 1e-6)), vh), ev.ACCENT, -1)
    mx = int(vw * (m.t_s - (t - t_in)) / max(seg_len, 1e-6))
    if 0 <= mx <= vw:
        cv2.rectangle(small, (max(mx - 3, 0), vh - 18), (min(mx + 3, vw), vh), ev.DASH, -1)
    canvas[oy:oy + vh, ox:ox + vw] = small
    if strip is not None:
        row = strip.copy()
        x = strip_x(t, total)
        cv2.line(row, (x, 0), (x, TL_H), (255, 255, 255), 2)
        canvas[pic_h:video_h] = row
    left = f"動画の時刻 {clock(t)}　{person.tag()}" + (f"（{person.label_note}）"
                                                    if person.label_note else "")
    _blend(canvas, hud(ev.W, left,
                      f"{CHECK_NOTE}・別の人かも" if flagged else CHECK_NOTE,
                      ORANGE if flagged else (255, 255, 255)), 0.72)
    canvas[video_h:] = band
    return canvas


# --- the video -----------------------------------------------------------------------------------

def render_highlight(pkg: Package, person: Person, target: Path, *, progress: Progress = _noop,
                     stop: Callable[[], bool] = lambda: False) -> dict:
    """Write `person`'s highlight video to `target`; {seconds, scenes, moments, suspects}.
    RenderError / Cancelled: the caller removes `target`."""
    chosen = [m for m in person.highlights() if m.camera in pkg.sources]
    if not chosen:
        raise RenderError("この人には、動画に入れられる場面がありません。")
    readers: dict[str, ev._Reader] = {}
    try:
        for cam in {m.camera for m in chosen}:
            readers[cam] = ev._Reader(pkg.sources[cam].video)
        scenes = segments(chosen, lambda c: readers[c].duration or None)
        if not scenes:
            raise RenderError("この人には、動画に入れられる場面がありません。")
        fps = ev.out_fps([pkg.sources[c] for c in readers])
        suspects = sum(1 for m in chosen if m.suspect)
        enc = ev.Encoder(target, fps)
        written = 0
        try:
            enc.write(intro_card(pkg, person, len(scenes), suspects), round(INTRO_S * fps))
            for n, seg in enumerate(scenes, 1):
                if stop():
                    raise Cancelled()
                progress((n - 1) / len(scenes), f"場面 {n} / {len(scenes)}")
                written += _scene(enc, pkg, person, seg, readers[seg.camera], fps, stop)
            if written == 0:
                raise RenderError("動画から場面の画像を読めませんでした。")
            enc.write(end_card(pkg, person), round(END_S * fps))
            enc.close()
        except BaseException:
            enc.abort()
            raise
    finally:
        for r in readers.values():
            r.close()
    progress(1.0, "仕上げ")
    return {"seconds": round(enc.seconds, 1), "scenes": len(scenes), "moments": len(chosen),
            "suspects": suspects}


def _scene(enc: ev.Encoder, pkg: Package, person: Person, seg: Segment, reader: ev._Reader,
           fps: float, stop) -> int:
    src = pkg.sources[seg.camera]
    first = reader.at(0.0)
    if first is None:
        return 0
    fh, fw = first.shape[:2]
    ev._check_shape((fw, fh), src.size)
    sx, sy = fw / src.size[0], fh / src.size[1]
    mine = set(person.labels.get(seg.camera) or [])
    h_band = band_height(seg.moments)
    video_h = ev.H - h_band
    bands = {m.ref_id: caption_band(m, h_band, person.verified) for m in seg.moments}
    length = max(seg.t1 - seg.t0, 1e-6)
    total = reader.duration or max([seg.t1] + [m.t_s for m in person.everything()
                                               if m.camera == seg.camera])
    strip = timeline_strip(person, seg.camera, total)
    n = max(1, int(round(length * fps)))
    written = 0
    for i in range(n):
        if i % 30 == 0 and stop():
            raise Cancelled()
        t = seg.t0 + i / fps
        frame = reader.at(t)
        if frame is None:
            break
        m = active(seg, t)
        vf = round(t * src.fps) + 1            # the tracks' 1-based frame number
        boxes = ai.boxes_at(src.boxes, vf)
        enc.write(compose(frame, boxes, mine, (sx, sy), length, t - seg.t0, m, t, person,
                          bands[m.ref_id], video_h, strip, total,
                          holder=holder.holder_at(pkg.holders, vf),
                          court=model.court_at(pkg.court, vf)))
        written += 1
    return written
