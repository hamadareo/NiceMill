"""「場面ごとの確認」 for the demo booklet (デモ版): which of a person's scenes are one player.

The identity gate (export/identity_gate.py) judges a whole label; it passes few people because the
number reads are sparse. The demo booklet keeps every person and judges each SCENE (a recorded
good moment the person's text cites), from data the app already holds:

  number      the number timeline reads the person's dominant number at the scene's time
              (identity_check.event_states = SAME, which allows `reach` seconds around a read);
  continuity  the person's box is present in at least `MIN_COVER` of the frames within
              +-`WINDOW_S` seconds of the scene and does not jump (moves of more than `JUMP`
              body heights between two frames);
  unique      in that window no other person's box covers the same body (IoU > `IOU_DUP`) and
              no other person reads the same dominant number there.

A scene counts only when all three hold. Nothing is computed from the video here. These are
cues, not proof: an ID switch inside a box track that looks continuous is not detectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WINDOW_S = 1.0
MIN_COVER = 0.9
JUMP = 0.5          # body heights per frame
IOU_DUP = 0.5
MIN_CONFIRMED = 2   # scenes for a person page (may be lowered to 1: shown as 推定)


@dataclass
class Scene:
    ref_id: str
    frame: int
    number: bool
    continuity: bool
    unique: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def confirmed(self) -> bool:
        return self.number and self.continuity and self.unique


@dataclass
class PersonScenes:
    scenes: list[Scene]
    dominant: str | None
    reads: int
    numbered: int
    reach_s: float
    labels: list = field(default_factory=list, repr=False)
    series: tuple = field(default=None, repr=False)       # (frames, boxes) of the person
    others: dict = field(default_factory=dict, repr=False)  # the other labels' series
    other_reads: list = field(default_factory=list, repr=False)
    own_segments: list = field(default_factory=list, repr=False)   # (first, last, number)

    @property
    def confirmed(self) -> list[Scene]:
        return [s for s in self.scenes if s.confirmed]

    @property
    def n_all(self) -> int:
        return len(self.scenes)

    @property
    def n_ok(self) -> int:
        return len(self.confirmed)


def _series(cam, labels: list[str]):
    """One person's boxes by frame (frames sorted, unique): (frames, [x, y, w, h] rows)."""
    fr, bx = [], []
    for lb in labels:
        f, b = cam.boxes.boxes(lb)
        fr += [int(v) for v in f]
        bx += [tuple(map(float, r)) for r in b]
    if not fr:
        return np.zeros(0, int), np.zeros((0, 4))
    o = np.argsort(fr, kind="stable")
    fr = np.array(fr)[o]
    bx = np.array(bx)[o]
    keep = np.concatenate([[True], np.diff(fr) > 0])
    return fr[keep], bx[keep]


def _iou(a, b) -> float:
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _move(a, b) -> float:
    """Distance of the box centres in body heights (of b)."""
    d = (((b[0] + b[2] / 2) - (a[0] + a[2] / 2)) ** 2 + ((b[1] + b[3] / 2) - (a[1] + a[3] / 2)) ** 2) ** 0.5
    return d / max(b[3], 1.0)


def continuity_ok(fr: np.ndarray, bx: np.ndarray, frame: int, fps: float) -> bool:
    """The window around `frame`: boxes in >= MIN_COVER of its frames, no jump."""
    w = max(1, round(WINDOW_S * fps))
    a, b = frame - w, frame + w
    sel = (fr >= a) & (fr <= b)
    if sel.sum() < MIN_COVER * (b - a + 1):
        return False
    f, x = fr[sel], bx[sel]
    return all(_move(x[i - 1], x[i]) / max(int(f[i] - f[i - 1]), 1) <= JUMP for i in range(1, len(f)))


def unique_ok(a: int, b: int, mine: tuple, others: dict, other_reads: list) -> bool:
    """No other person's box overlaps this person's in frames a..b; nobody else reads the same
    number in them (`other_reads`: [(first, last)] frames where another label reads our number)."""
    for o_fr, o_bx in others.values():
        if not len(o_fr):
            continue
        lo, hi = np.searchsorted(o_fr, a), np.searchsorted(o_fr, b, side="right")
        for k in range(lo, hi):
            j = np.searchsorted(mine[0], o_fr[k])
            if j < len(mine[0]) and mine[0][j] == o_fr[k] and _iou(mine[1][j], o_bx[k]) > IOU_DUP:
                return False
    return not any(s <= b and e >= a for s, e in other_reads)


def check_person(repo, sub, cam, subject: dict, sheet, chain, others_cache: dict | None = None
                 ) -> PersonScenes:
    """Judge every scene of `sheet` (its moments that the text cites)."""
    from pfb_app.export import identity_gate   # noqa: PLC0415
    from pfb_app.services import auto_people   # noqa: PLC0415
    pid = subject.get("participant_id") or ""
    verdict = identity_gate.assess(repo, sub, subject, chain)
    fps = float(cam.fps or 30.0)
    mine_labels = cam.labels_of(pid)
    mine = _series(cam, mine_labels)
    others = others_cache if others_cache is not None else {}
    for lb in cam.boxes.labels:
        if lb not in others:
            others[lb] = _series(cam, [lb])
    others_mine = {lb: v for lb, v in others.items() if lb not in mine_labels}
    doms = verdict.facts.get("dominant") or []
    dominant = doms[0] if len(doms) == 1 else None
    other_reads: list[tuple[int, int]] = []
    reach_s = 0.0
    for _con, source in chain:
        if not source.camera:
            continue
        run = auto_people._number_run(repo, sub.sid, source.camera, cam.track)
        reads = ((run.output or {}).get("cues") or {}).get("number") if run else None
        for lb, r in (reads or {}).items():
            if isinstance(r, dict):
                reach_s = max(reach_s, float(r.get("timeline_s") or 0.0) * 1.5)
            if lb in mine_labels or not isinstance(r, dict) or dominant is None:
                continue
            for seg in r.get("segments") or []:
                if str(seg[2]) == dominant:
                    other_reads.append((int(seg[0]), int(seg[1])))
    scenes = []
    for m in sorted((m for m in sheet.moments), key=lambda m: (m.t_s or 0.0, m.ref_id)):
        if m.t_s is None:
            continue
        frame = round((sub.start_s + m.t_s) * fps) + 1
        num = m.ref_id in verdict.same_refs
        has = bool(len(mine[0]))
        cont = has and continuity_ok(mine[0], mine[1], frame, fps)
        w = max(1, round(WINDOW_S * fps))
        uniq = has and unique_ok(frame - w, frame + w, mine, others_mine, other_reads)
        why = []
        if not num:
            why.append("背番号が読み取られた時刻の近くではない")
        if not cont:
            why.append("枠が途切れる、または飛ぶ")
        if not uniq:
            why.append("ほかの人の枠・背番号と重なる")
        scenes.append(Scene(m.ref_id, frame, num, bool(cont), bool(uniq), why))
    f = verdict.facts
    own = []
    for _con, source in chain:
        if not source.camera:
            continue
        run = auto_people._number_run(repo, sub.sid, source.camera, cam.track)
        reads = ((run.output or {}).get("cues") or {}).get("number") if run else None
        for lb in mine_labels:
            for seg in ((reads or {}).get(lb) or {}).get("segments") or []:
                own.append((int(seg[0]), int(seg[1]), str(seg[2])))
    return PersonScenes(scenes, dominant, f.get("reads", 0), f.get("numbered", 0), max(reach_s, 1.0),
                        mine_labels, mine, others_mine, other_reads, own)


def unbroken_runs(cam, labels: list[str]) -> list[tuple[int, int]]:
    """Maximal runs of consecutive frames with a box of the person and no jump between them."""
    fr, bx = _series(cam, labels)
    if not len(fr):
        return []
    runs, s0 = [], 0
    for i in range(1, len(fr) + 1):
        brk = i == len(fr) or fr[i] - fr[i - 1] != 1 or _move(bx[i - 1], bx[i]) > JUMP
        if brk:
            runs.append((int(fr[s0]), int(fr[i - 1])))
            s0 = i
    return runs


STRIP_FRAMES = 6
STRIP_STEP = 24                       # frames between two pictures of a strip (0.8 s at 30 fps)
WINDOW_FRAMES = STRIP_STEP * (STRIP_FRAMES - 1)    # 120: the strip spans 121 frames


def windows(ps: PersonScenes, runs: list[tuple[int, int]], *, max_n: int = 2
            ) -> list[dict]:
    """Up to `max_n` non-overlapping windows of WINDOW_FRAMES+1 consecutive frames inside one
    unbroken run of the person's box, where no other person's box covers the body, nobody else
    reads the same number and no other number is read for the person. Those that hold a
    confirmed scene, then those that overlap the dominant number's read, come first."""
    scene_frames = [s.frame for s in ps.confirmed]
    dom = [(s, e) for s, e, n in ps.own_segments if n == ps.dominant and ps.dominant is not None]
    cands = {}
    for ra, rb in runs:
        if rb - ra < WINDOW_FRAMES:
            continue
        anchors = [f for f in scene_frames if ra <= f <= rb]
        anchors += [(s + e) // 2 for s, e in dom if not (e < ra or s > rb)]
        anchors.append((ra + rb) // 2)
        for f in anchors:
            a = min(max(f - WINDOW_FRAMES // 2, ra), rb - WINDOW_FRAMES)
            b = a + WINDOW_FRAMES
            if (a, b) in cands:
                continue
            if not unique_ok(a, b, ps.series, ps.others, ps.other_reads):
                continue
            if any(s <= b and e >= a and n != ps.dominant for s, e, n in ps.own_segments):
                continue
            cands[(a, b)] = {
                "a": a, "b": b, "run": (ra, rb),
                "scene": any(a <= x <= b for x in scene_frames),
                "number": any(s <= b and e >= a for s, e in dom)}
    out: list[dict] = []
    for c in sorted(cands.values(), key=lambda c: (not c["scene"], not c["number"],
                                                  -(c["run"][1] - c["run"][0]), c["a"])):
        if all(c["b"] < o["a"] or c["a"] > o["b"] for o in out):
            out.append(c)
        if len(out) >= max_n:
            break
    return out
