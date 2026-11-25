"""The review player: one local HTML file (+ the video next to it) for watching a whole match
with each person's moments marked.

`build_player(pkg, out_dir)` writes `review_player.html` (all script and style inline, the data
embedded; it makes no network request and loads nothing from outside) and puts the playable video
next to it (`video.<ext>`: a hard link, else a copy). Open the HTML in a browser: the video plays
on the left with the boxes drawn on it (a switch turns them off), the right side lists each
person's moments (the generated sentences, as generated), a click jumps to the moment
(PREROLL_S seconds before it, the same lead-in as the highlight video, so the action can be seen), the moment now playing is highlighted,
and the speed can be changed. Keys: space = play/pause, left/right = 5 s, N / P = next /
previous moment of the chosen person.

One video per player (the camera with most moments); moments of other cameras are not listed
(the HTML says how many). Boxes are the tracks of the listed people only, thinned to about
TRACK_FPS per second.
"""
from __future__ import annotations

import json
import math
import os
import logging
import shutil
from pathlib import Path

import numpy as np

from pfb_app.export.extra_boxes import player_rows
from pfb_app.review_video import calm, court_est, court_reg, everyone
from pfb_app.review_video.model import (ACT_NOTE, COURT_EST_NOTE, TEAM_ADJUST_NOTE, TEAM_COLOUR_NOTE, ink_for, EVAL_AI_NOTE, EVAL_NONE_NOTE, EVAL_SHORT_NOTE, CLAIM_NOTE, EVENT_NOTE, EVENT_WEAK_NOTE, EXTRA_NOTE, BALL_NOTE, COURT_NOTE, CHECK_NOTE, NONPLAYER_SET_APART, TEAM_WORDS, team_color, date_in_title, CHECKED, DRAFT_NOTICE, FIX_NOTE, GATE_NOTE,
                                        KIND_GOOD, NO_MARK_NOTE, NUMBER_NOTE,
                                        NUMBER_NOTE_UNVERIFIED, SUSPECT_NOTE, Package)

log = logging.getLogger(__name__)
MAX_BUBBLES = 2                      # speech bubbles over heads at once (the chosen person's first): the match must stay visible
PREROLL_S = 4.0                      # = the highlight video's lead-in (edit.BEFORE_S)
WINDOW_BEFORE_S, WINDOW_AFTER_S = 4.0, 3.0   # = the highlight video's window (edit.BEFORE_S/AFTER_S)
TRACK_FPS = 10
HTML_NAME = "review_player.html"


def _mark(m) -> dict:
    # a weakened moment never carries the sentence it was weakened from: `what` is the generated
    # observation (it states the same action), so it is withheld here and in the page's data
    return {"id": m.ref_id, "t": round(m.t_s, 2), "when": m.when, "what": "" if m.weak else m.what, "why": m.why,
            "category": m.category, "cited": m.cited, "suspect": m.suspect, "kind": m.kind,
            "note": m.note, "weak": m.weak}


def _track_rows(src, label: str, step: int) -> list[list[int]]:
    """One track thinned to every `step`-th frame: [frame, x, y, w, h] in video pixels."""
    frames, boxes = src.boxes.boxes(label)
    return [[int(f), round(float(b[0])), round(float(b[1])), round(float(b[2])), round(float(b[3]))]
            for f, b in zip(frames, boxes) if int(f) % step == 0]


def player_data(pkg: Package, video_name: str) -> dict:
    """The JSON the page embeds (also what the tests look at)."""
    counts: dict[str, int] = {}
    for p in pkg.people:
        for m in p.everything():
            if m.t_s is not None and m.camera:
                counts[m.camera] = counts.get(m.camera, 0) + 1
    cams = [c for c in sorted(counts, key=lambda c: -counts[c]) if c in pkg.sources]
    if not cams:
        raise ValueError("no camera with a playable video and moments")
    cam = cams[0]
    src = pkg.sources[cam]
    step = max(1, int(math.ceil(src.fps / TRACK_FPS)))
    people, tracks, skipped = [], {}, 0
    latest = 0.0
    for p in pkg.people:
        labels = list(p.labels.get(cam) or [])
        good, other = [], []
        for m in sorted(p.everything(), key=lambda m: (m.t_s if m.t_s is not None else 1e18,
                                                       m.ref_id)):
            if m.camera != cam or m.t_s is None:
                skipped += 1
                continue
            latest = max(latest, m.t_s)
            (good if m.kind == KIND_GOOD else other).append(_mark(m))
        if not good and not other:
            continue
        weak_any = any(m["weak"] for m in good)   # the generated paragraph may restate a sentence the action check weakened
        people.append({"id": p.subject_id, "name": p.name, "label": p.shown(), "tag": p.tag(),
                       "dashed": p.dashed, "checked": p.checked, "checked_note": CHECKED[p.checked],
                       "label_note": p.label_note, "number": p.number, "team": p.team,
                       "internal": p.internal, "nonplayer": p.nonplayer, "labels": labels,
                       "color": team_color(p.team_letter)["hex"], "ink": team_color(p.team_letter)["ink"],
                       "tm": p.team_letter, "team_word": TEAM_WORDS[p.team_letter], "excluded": p.excluded,
                       "np_score": p.np_score, "np_apart": NONPLAYER_SET_APART, "status": p.status,
                       "text": None if (weak_any or p.eval_text) else p.text,
                       "text_withheld": bool(weak_any and p.text and not p.eval_text),
                       "eval": ({"text": p.eval_text, "source": p.eval_source, "hedged": p.eval_hedged,
                                 "note": EVAL_SHORT_NOTE if p.eval_source == "short" else EVAL_AI_NOTE}
                                if p.eval_text else None),
                       "eval_why": p.eval_why,
                       "moments": good, "events": other})
    for lb in src.boxes.labels:       # every labelled box of the camera (the listed people's and all the others')
        tracks[lb] = _track_rows(src, lb, step)
    ex = pkg.extras.get(cam)
    extras = player_rows(ex, step) if ex is not None and len(ex) else None
    every = _everyone(pkg, cam, src, step, people, tracks, extras)
    style, others, court_e = None, {}, None
    if every:
        tracks = every["tracks"]
        extras = every["extras"] or extras
        style, others = every["style"], every["others"]
        if style:
            for q in people:
                if q["tm"] in ("A", "B"):
                    q["color"] = style[q["tm"]]["hex"]
                    q["ink"] = ink_for(q["color"])
                    q["rim"] = style[q["tm"]]["rim"]
                else:
                    q["color"], q["ink"], q["rim"] = "#8E9196", ink_for("#8E9196"), "#000000"
        court_e = every.get("court")
    court_out = None
    if every and pkg.court and pkg.court.get("frames"):
        tracks, extras, court_out = _flag_outside(pkg.court["frames"], src, step, people, tracks, extras)
    calm_rows, extra_tid = {}, None
    if every:
        calm_rows, extra_tid = _calm(src.fps or 30.0, step, tracks, extras)
    duration = max(latest + 10.0, (src.last_frame or 0) / src.fps if src.fps else 0.0)
    return {"title": pkg.title, "date": "" if date_in_title(pkg.title, pkg.date) else pkg.date, "video": video_name, "fps": src.fps,
            "size": list(src.size), "step": step, "preroll": PREROLL_S, "duration": duration,
            "video_s": round(src.last_frame / src.fps, 1) if src.last_frame and src.fps else 0,
            "window": [WINDOW_BEFORE_S, WINDOW_AFTER_S], "max_bubbles": MAX_BUBBLES, "people": people, "tracks": tracks,
            "extras": extras, "extra_note": EXTRA_NOTE if extras else None,
            "every": bool(every), "team_style": _style_json(style), "others_team": others,
            "act_words": everyone.ACT_WORDS, "act_short": everyone.ACT_SHORT, "act_note": ACT_NOTE if every else None,
            "court_est": ({"samples": court_e["samples"], "note": court_e["note"], "text": COURT_EST_NOTE,
                           "far_share": court_e["far_share"], "near_share": court_e["near_share"]} if court_e else None), "others": pkg.others, "banner": DRAFT_NOTICE, "check_note": CHECK_NOTE,
            "suspect_note": SUSPECT_NOTE, "claim_note": CLAIM_NOTE, "event_note": EVENT_NOTE, "event_weak_note": EVENT_WEAK_NOTE,
            "fix_note": FIX_NOTE, "eval_on": any(p.eval_text or p.eval_why for p in pkg.people), "eval_none_note": EVAL_NONE_NOTE,
            "number_verified": pkg.number_verified,
            "number_note": NUMBER_NOTE if pkg.number_verified else NUMBER_NOTE_UNVERIFIED,
            "gate_note": "" if pkg.number_verified else GATE_NOTE, "no_mark_note": NO_MARK_NOTE,
            "credit": list(pkg.credit_lines), "notes": list(pkg.notes),
            "skipped_other_camera": skipped, "camera": cam,
            "court": ({"frames": pkg.court["frames"], "note": pkg.court["note"], "text": pkg.court.get("text") or COURT_NOTE} if pkg.court else None),
            "court_out": court_out, "calm": calm_rows, "extra_tid": extra_tid,
            "ball": ({"segs": pkg.holders, "n": pkg.ball_frames, "note": BALL_NOTE}
                     if pkg.ball_frames else None)}


def _flag_outside(court_rows: list, src, step: int, people: list, tracks: dict, extras):
    """Append the "outside the court" flag (0/1) as the LAST column of every track / unnamed row.
    Named (listed) people are never flagged: the person who is evaluated stays a player whatever the
    picture says. Returns (tracks, extras, summary for the page)."""
    fps = src.fps or 30.0
    vh = float(src.size[1])
    listed = {lb for q in people for lb in q["labels"]}
    n_rows = n_flag = 0
    new_t = {}
    for lb, rows in tracks.items():
        fl = [0] * len(rows) if lb in listed else court_reg.flag_rows(rows, court_rows, fps, vh, step)
        new_t[lb] = [r + [f] for r, f in zip(rows, fl)]
        if lb not in listed:
            n_rows += len(rows)
            n_flag += sum(fl)
    new_e = None
    if extras:
        ex = [list(r) for r in extras]
        tid = everyone.link_rows(ex, step)
        flags = [0] * len(ex)
        by: dict[int, list] = {}
        for i, t in enumerate(tid):
            by.setdefault(t, []).append(i)
        for idx in by.values():
            idx.sort(key=lambda i: ex[i][0])
            fl = court_reg.flag_rows([ex[i] for i in idx], court_rows, fps, vh, step)
            for i, f in zip(idx, fl):
                flags[i] = f
        new_e = [r + [f] for r, f in zip(ex, flags)]
        n_rows += len(ex)
        n_flag += sum(flags)
    return new_t, (new_e if new_e is not None else extras), {
        "margin_m": court_reg.OUT_MARGIN_M, "min_s": court_reg.OUT_MIN_S, "stay_s": court_reg.IN_STAY_S,
        "note": court_reg.OUT_NOTE, "rows": n_rows, "flagged": n_flag, "n": n_flag}


def _calm(fps: float, step: int, tracks: dict, extras):
    """A steady anchor (over the head) and a steady activity word for everyone (calm.py):
    ({key: [[frame, ax, ay, act], ...]}, the key of each unnamed row's person). Keys: a track's label, or
    "e<N>" for unnamed person N (the same linking as the page uses)."""
    out = {}
    for lb, rows in tracks.items():
        rows = sorted(rows)
        if rows:
            out[lb] = calm.calm_entity(rows, [r[5] for r in rows], fps)
    tid = None
    if extras:
        tid = everyone.link_rows([list(r[:5]) for r in extras], step)
        by: dict[int, list] = {}
        for i, t in enumerate(tid):
            by.setdefault(t, []).append(i)
        for t, idx in by.items():
            idx.sort(key=lambda i: extras[i][0])
            rows = [extras[i] for i in idx]
            out["e%d" % t] = calm.calm_entity(rows, [r[6] for r in rows], fps)
    return out, tid


def _style_json(style):
    if not style:
        return None
    out = {k: dict(style[k], ink=ink_for(style[k]["hex"])) for k in ("A", "B")}
    out["adjusted"] = style["adjusted"]
    out["note"] = TEAM_COLOUR_NOTE
    out["adjust_note"] = (TEAM_ADJUST_NOTE.format(name=style["B"]["name"]) if style["adjusted"] else "")
    return out


def _everyone(pkg: Package, cam: str, src, step: int, people: list, tracks: dict, extras):
    """Team colours of the kits, everyone's team and what they seem to be doing, the court's sidelines
    (cached on the package: the player data is made twice). None when the picture cannot be read."""
    key = (cam, len(extras or []), len(tracks), len(people))
    if pkg.every is not None and pkg.every.get("key") == key:
        return pkg.every["res"]
    try:
        res = everyone.analyse(src.video, src.fps, step, people, tracks, extras, pkg.holders, pkg.pose_path)
        if res is not None:
            res["court"] = None
            if not pkg.court and pkg.court_est:
                feet = [[r[0], r[1] + r[3] / 2, r[2] + r[4]] for lb, rows in res["tracks"].items()
                        for r in rows if _plays(people, lb, res["others"])]
                feet += [[r[0], r[1] + r[3] / 2, r[2] + r[4]] for r in res["extras"] if r[5]]
                fa = np.array(feet, float).reshape(-1, 3)
                est = court_est.estimate(src.video, src.fps or 30.0, src.last_frame, feet=fa)
                res["court"] = court_est.validate(est, fa, src.fps or 30.0)
    except Exception:   # noqa: BLE001 - a picture that cannot be read: the plain player, nothing invented
        log.exception("everyone analysis failed")
        res = None
    pkg.every = {"key": key, "res": res}
    return res


def _plays(people: list, label: str, others: dict) -> bool:
    for q in people:
        if label in q["labels"]:
            return not q["excluded"]
    return others.get(label, 0) in (1, 2)


def _json_for_script(data: dict) -> str:
    s = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # no raw "<" at all: neither "</script>" nor "<!--" can end or confuse the data block
    return s.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _place_video(src: Path, dst: Path) -> None:
    """A plain copy (never a hard link: deleting the original must delete the data), readable
    by the owner only."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copyfile(src, dst)
    os.chmod(dst, 0o600)


def build_player(pkg: Package, out_dir: Path) -> dict:
    """Write the player into `out_dir`; {html, video, people, moments, bytes}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    probe = player_data(pkg, "video")
    src_path = pkg.sources[probe["camera"]].video
    video_name = "video" + (src_path.suffix.lower() or ".mp4")
    data = player_data(pkg, video_name)
    _place_video(src_path, out_dir / video_name)
    html = TEMPLATE.replace("__TITLE__", _esc(f"振り返りプレーヤー　{pkg.title}")) \
                   .replace("__DATA__", _json_for_script(data))
    target = out_dir / HTML_NAME
    tmp = out_dir / (HTML_NAME + ".part")
    tmp.write_text(html, encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, target)
    return {"html": str(target), "video": str(out_dir / video_name),
            "people": len(data["people"]),
            "moments": sum(len(p["moments"]) for p in data["people"]),
            "bytes": target.stat().st_size}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<title>__TITLE__</title>
<style>
:root{--bg:#fff;--ink:#1a1f1d;--muted:#4a5651;--brand:#1d4d3f;--soft:#eef3f0;--line:#cfdad4;--warn:#b45309;--warnfill:#e8730c;--dot:#6b7b75}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.7 "Noto Sans CJK JP","Hiragino Sans","Yu Gothic",sans-serif}
#top{display:flex;align-items:center;gap:12px;padding:8px 20px;background:var(--brand);color:#fff;flex-wrap:wrap}
#top .ttl{font-weight:700;flex:1;min-width:200px}
#info{background:#fff;color:var(--brand);border:0;border-radius:999px;padding:4px 14px;font:inherit;font-size:14px;font-weight:700;cursor:pointer}
#panel{display:none;background:var(--soft);border-bottom:1px solid var(--line);padding:16px 20px;font-size:15px}
#panel.open{display:block}
#panel p{margin:0 0 8px;max-width:900px}
#panel h2{font-size:15px;margin:12px 0 4px;color:var(--brand)}
#panel .warn{color:var(--warn);font-weight:700}
main{max-width:1180px;margin:0 auto;padding:20px;position:relative}
#stage{position:relative;background:#000;line-height:0;border-radius:8px;overflow:hidden}
#stage video{width:100%;max-height:64vh;display:block;background:#000;object-fit:contain}
#stage canvas{position:absolute;left:0;top:0;pointer-events:none}
#stage:fullscreen{display:flex;align-items:center;border-radius:0}
#stage:fullscreen video{max-height:100vh;height:100vh}
#stage:-webkit-full-screen{display:flex;align-items:center;border-radius:0}
#stage:-webkit-full-screen video{max-height:100vh;height:100vh}
#cm{position:absolute;left:0;top:0;pointer-events:none;overflow:hidden;line-height:1.5}
.cmt{position:absolute;max-height:4.4em;overflow:hidden;max-width:58%;background:rgba(0,0,0,.55);color:#fff;border-radius:6px;padding:3px 8px;font-size:13px;line-height:1.35;white-space:pre-wrap;opacity:0;transition:opacity .25s;border-left:3px solid #2a78d6}
.cmt.in{opacity:1}
.cmt.warn{border-left-color:#ff961e}
.bl{position:absolute;max-width:44%;background:rgba(255,255,255,.95);color:#1a1f1d;border:2px solid var(--pc);border-radius:8px;padding:2px 7px 2px 3px;line-height:1.3;opacity:0;transition:opacity .25s;display:flex;gap:6px;align-items:flex-start;box-shadow:0 1px 4px rgba(0,0,0,.35)}
.bl.in{opacity:1}
.bl.flag{border-style:dotted}
.bl .cn{flex:none;background:var(--pc);color:var(--pi);font-weight:700;border-radius:5px;padding:0 6px;font-size:.92em;white-space:nowrap}
.bl .s{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;word-break:normal;overflow-wrap:anywhere}
.bl .fl{font-weight:700;margin-right:4px}
.bl:after{content:"";position:absolute;left:var(--tx,20px);bottom:-9px;margin-left:-7px;border:7px solid transparent;border-top:8px solid var(--pc);border-bottom:0}
.bl.dn:after{bottom:auto;top:-9px;border-top:0;border-bottom:8px solid var(--pc)}
.cmt .cn{background:var(--pc);color:var(--pi);font-weight:700;border-radius:4px;padding:0 5px;margin-right:6px;font-size:.9em}
.cmt{border-left-color:var(--pc,#2a78d6)}
.chip .dot{display:inline-block;width:11px;height:11px;border-radius:50%;background:var(--pc);border:1px solid #1a1f1d;margin-right:6px;vertical-align:-1px}
.chip[aria-pressed=true]{background:var(--pc);border-color:#1a1f1d;color:var(--pi)}
.chip[aria-pressed=true] .dot{background:var(--pi);border-color:var(--pi)}
.nm{border-left:8px solid var(--pc);outline:1px solid var(--line);outline-offset:-1px}
.cmt .s{display:inline}
@media (prefers-reduced-motion:reduce){.cmt,.bl{transition:none}}
#tools{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin:12px 0 4px;font-size:14px}
#tools .grp{display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center}
.tg span{white-space:nowrap}
#opts{margin:10px 0 0;border:1px solid var(--line);border-radius:8px;padding:2px 12px;background:#fff}
#opts>summary{cursor:pointer;color:var(--brand);font-weight:700;padding:6px 0;font-size:15px}
#opts[open]>summary{border-bottom:1px solid var(--line);margin-bottom:6px}
#feat1{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:10px 0 0;font-size:14px}
#vfail{display:none;margin:8px 0;padding:8px 12px;border:2px solid var(--warn);border-radius:8px;color:var(--warn);font-weight:700;font-size:15px}#vfail.on{display:block}
button,select{background:#fff;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:4px 12px;font:inherit;font-size:14px;cursor:pointer}
button:focus-visible,select:focus-visible,input:focus-visible+span,.chip:focus-visible,.m:focus-visible,summary:focus-visible,#info:focus-visible{outline:3px solid var(--brand);outline-offset:2px}
.tg{display:inline-flex;align-items:center;gap:4px;cursor:pointer}
.tg input{position:absolute;opacity:0;width:1px;height:1px}
.tg span{display:inline-flex;align-items:center;gap:4px;border:1px solid var(--line);border-radius:999px;padding:3px 12px;background:#fff}
.tg input:checked+span{background:var(--soft);border-color:var(--brand);font-weight:700}
.tg input:checked+span:before{content:"✓";font-weight:700;color:var(--brand)}
#optstate{font-weight:400;color:var(--muted);font-size:14px;margin-left:8px}
#live .clip{display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
#live{display:none;margin-top:6px;padding:5px 12px;background:var(--soft);border:1px solid var(--line);border-radius:8px;font-size:14px;line-height:1.5;min-height:34px}
#live.on{display:block}#live .tag{display:inline-block;background:var(--brand);color:#fff;border-radius:999px;padding:0 10px;margin-right:8px;font-size:14px;font-weight:700}
#feat{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin:8px 0 0;font-size:14px}#feat .lbl{color:var(--muted)}
#helpbox{display:none;margin:6px 0;padding:8px 12px;background:var(--soft);border:1px solid var(--line);border-radius:8px;font-size:14px}#helpbox.open{display:block}
#legend{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;margin:10px 0 0;font-size:14px;color:var(--muted)}
#legend .sw{display:inline-block;width:22px;height:12px;border:2px solid #fff;outline:1px solid #1a1f1d;margin-right:6px;vertical-align:-1px}
#legend .ln{display:inline-block;width:26px;height:0;border-top:4px solid #E02020;box-shadow:0 0 0 1px #1a1f1d;margin-right:8px;vertical-align:3px}
#legend .ov{display:inline-block;width:24px;height:10px;border:3px solid #E02020;outline:2px solid #fff;border-radius:50%;background:rgba(224,32,32,.28);margin-right:7px;vertical-align:-3px}
#legend .note{flex-basis:100%;font-size:14px}
#chips{margin:14px 0 8px}
.tgrp{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 8px}
.tgh{flex:none;min-width:116px;background:var(--pc);color:var(--pi);border:2px solid #1a1f1d;border-radius:8px;padding:3px 12px;font-weight:700;font-size:15px}
.tgh .n{font-weight:400;font-size:14px;margin-left:4px}
#evalbox{display:none;margin:8px 0 6px;border:1px solid var(--line);outline:1px solid var(--line);border-left:8px solid var(--pc,var(--brand));border-radius:8px;padding:10px 16px;background:#fff}
#evalbox .who{font-size:16px;font-weight:700;color:var(--brand)}
#evalbox .tmw{display:inline-block;margin-left:10px;background:var(--pc);color:var(--pi);border-radius:999px;padding:0 10px;font-size:14px;font-weight:700}
#evalbox .etx{font-size:18px;line-height:1.7;margin:4px 0;white-space:pre-wrap}
#evalbox .etx.none{color:var(--muted);font-size:16px}
#evalbox .enote{font-size:14px;color:var(--muted)}
#evalbox .enote.w{color:var(--warn);font-weight:700}
.chip{border:1px solid var(--line);background:#fff;border-radius:999px;padding:4px 14px;font-size:15px}
.chip[aria-pressed=true]{font-weight:700}
.chip .n{font-size:14px;opacity:.85}
#allon{margin-left:auto}
#tlwrap{border:1px solid var(--line);border-radius:8px;background:#fafcfb;padding:4px 8px 8px}
#tl{display:grid;grid-template-columns:150px minmax(0,1fr);max-height:260px;overflow-y:auto}
#tlnames{border-right:1px solid var(--line)}
.nm{height:34px;display:flex;align-items:center;padding:0 6px;overflow:hidden;white-space:nowrap;font-size:14px;border-bottom:1px solid var(--line)}
.nm span{overflow:hidden;text-overflow:ellipsis}
#tlscroll{overflow-x:auto;overflow-y:hidden}
#tlinner{position:relative;min-width:100%}
.row{height:34px;position:relative;border-bottom:1px solid var(--line)}
.row.sel{background:var(--soft)}
.m:before{content:"";position:absolute;left:-11px;right:-11px;top:-11px;bottom:-11px}   /* the mark looks 22px, the touch area is 44px */
.m{position:absolute;top:50%;width:22px;height:22px;margin:-11px 0 0 -11px;padding:0;border:0;background:none;border-radius:4px;display:flex;align-items:center;justify-content:center;cursor:pointer}
.m svg{display:block;pointer-events:none}
.m.g{z-index:3}.m.c{z-index:2}.m.e{z-index:1}
.m.now{outline:2px solid var(--ink)}
#head{position:absolute;top:0;bottom:0;width:2px;margin-left:-1px;background:var(--ink);z-index:5;pointer-events:none}
#tick{position:relative;height:20px;border-bottom:1px solid var(--line);font-size:14px;color:var(--muted)}
#tick span{position:absolute;top:0;transform:translateX(-50%)}
#card{margin-top:16px;border:1px solid var(--line);border-left:6px solid var(--brand);border-radius:8px;padding:14px 20px;min-height:110px}
#card .who{font-size:16px;color:var(--brand);font-weight:700}
#card .main{font-size:20px;line-height:1.6;margin:4px 0;white-space:pre-wrap}
#card .flag{color:var(--warn);font-weight:700;font-size:16px}
#card .more{font-size:16px;color:var(--muted);margin-top:8px;border-top:1px solid var(--line);padding-top:8px}
#card .more div{margin:2px 0}
#card .hint{color:var(--muted);font-size:16px}
#morebtn{margin-top:6px}
#list{margin-top:16px;font-size:15px}
#list summary{cursor:pointer;color:var(--brand);font-weight:700}
.mom{padding:6px 10px;border-top:1px solid var(--line);cursor:pointer}
.mom:hover{background:var(--soft)}
.mom.act{background:var(--soft);border-left:4px solid var(--brand)}
.mom .t{font-weight:700;color:var(--brand)}
.tip{position:fixed;z-index:20;max-width:340px;background:#1a1f1d;color:#fff;border-radius:8px;padding:8px 12px;font-size:14px;line-height:1.5;display:none;pointer-events:none}
.tip .h{font-weight:700}.tip .w{color:#ffb877;font-weight:700}
#hints{display:none;align-items:center;gap:6px 16px;flex-wrap:wrap;margin:0 0 10px;padding:6px 40px 6px 14px;background:var(--brand);color:#fff;border-radius:10px;font-size:15px;position:relative}
#hints.on{display:flex}
#hints b{font-weight:700}
#hints button{position:absolute;right:4px;top:50%;transform:translateY(-50%);background:none;border:0;color:#fff;font-size:22px;padding:0 10px;min-width:44px;min-height:44px}
#vmix{display:none;margin:8px 0;padding:8px 12px;border:2px solid var(--warn);border-radius:8px;color:var(--warn);font-weight:700;font-size:15px}#vmix.on{display:block}
@media (forced-colors:active){.chip[aria-pressed=true],.tg input:checked+span{outline:3px solid Highlight}.m svg,.chip .dot,#legend .sw{forced-color-adjust:none}}
@media print{*{-webkit-print-color-adjust:exact;print-color-adjust:exact}
 #top{background:none;color:#000;border-bottom:2px solid #000}#info{border:1px solid #000;background:none;color:#000}
 #panel{display:block;background:none}#panelx,#hints,#feat1,#opts,#helpbox,#allon{display:none!important}
 #stage video{max-height:none}#tl{max-height:none;overflow:visible}main{max-width:none}}
@media (pointer:coarse),(max-width:600px){.row,.nm{height:44px}}
@media (pointer:coarse){.chip{min-height:44px}}
@media (max-width:600px){.tgrp{flex-wrap:nowrap;overflow-x:auto;padding-bottom:6px}.chip,#allon{flex:none;white-space:nowrap}#allon{margin-left:0}   /* 15 people would fill the phone screen */
 #hints{line-height:1.45;gap:2px 10px;padding:6px 44px 6px 12px}}   /* last, so it wins over the row heights above */
</style>
</head>
<body>
<div id="top"><span class="ttl" id="title"></span><button id="info" type="button" aria-expanded="false" aria-controls="panel">AIの下書き・背番号は未検証 ⓘ</button></div>
<div id="panel" role="region" aria-label="注意とくわしい説明"></div>
<main>
<div id="hints" role="note"><b>はじめに</b><span>① ここで再生します（映像）</span><span>② ここで場面を選びます（下のタイムライン）</span><span>③ ここに文が出ます（その下のカード）</span><button id="hintx" type="button" aria-label="ヒントを閉じる">×</button></div>
<div id="stage"><video id="v" controls preload="metadata" playsinline></video><canvas id="c"></canvas><div id="cm" aria-live="polite" aria-relevant="additions"></div></div>
<div id="live" aria-live="off"></div>
<div id="vmix" role="alert">この映像は、プレーヤーを作ったときの映像と違うようです（縦横の比や長さが合いません）。枠や場面の位置は合いません。作り直すか、元の映像に戻してください。</div>
<div id="vfail" role="alert">映像を読み込めませんでした。このHTMLと同じ場所に video ファイルがあるか、ブラウザが形式に対応しているか確認してください。</div>
<div id="feat1"><button id="prevm" type="button">◀ 前の見せ場</button><button id="nextm" type="button">次の見せ場 ▶</button><button id="f-help" type="button" aria-expanded="false">ショートカット ?</button></div>
<div id="helpbox" role="region" aria-label="ショートカット一覧"><b>ショートカット</b>　空白：再生/停止　←→：前後のイベント　Shift+←→：5秒　N / P：次 / 前の見せ場　C：吹き出しの入/切　F：全画面　?：この一覧</div>
<details id="opts"><summary>表示の設定・見どころ機能<span id="optstate"></span></summary>
<div id="feat"><span class="lbl">見どころ機能：</span>
<label class="tg"><input id="f-auto" type="checkbox"><span>見せ場だけ連続再生</span></label>
<label class="tg"><input id="f-slow" type="checkbox"><span>見せ場は0.5倍速</span></label>
<label class="tg"><input id="f-spot" type="checkbox"><span>スポットライト</span></label>
<label class="tg"><input id="f-trail" type="checkbox"><span>足跡</span></label>
<label class="tg"><input id="f-live" type="checkbox"><span>実況メモ</span></label>
<button id="f-shot" type="button">画面を保存</button></div>
<div id="tools">
<div class="grp"><label class="tg"><input id="k-good" type="checkbox" checked><span><svg width="14" height="14" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 1 19 10 10 19 1 10Z" fill="#1d4d3f"/></svg>よかった所</span></label>
<label class="tg"><input id="k-event" type="checkbox" checked><span><svg width="14" height="14" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="5" fill="#6b7b75"/></svg>行動</span></label>
<label class="tg"><input id="k-flag" type="checkbox" checked><span><svg width="14" height="14" viewBox="0 0 20 20" aria-hidden="true"><path d="M10 2 18 17H2Z" fill="#e8730c"/></svg>要確認</span></label></div>
<div class="grp"><button id="zout" type="button" title="タイムラインを縮小" aria-label="縮小">－</button><button id="zin" type="button" title="タイムラインを拡大" aria-label="拡大">＋</button><button id="zall" type="button">全体</button></div>
<div class="grp"><label>速さ <select id="speed"><option>0.5</option><option>0.75</option><option selected>1</option><option>1.25</option><option>1.5</option><option>2</option></select></label>
<label class="tg"><input id="boxes" type="checkbox" checked><span>選んだ人の枠</span></label>
<label class="tg"><input id="allbox" type="checkbox"><span>全員の枠</span></label>
<label class="tg" id="tags-wrap"><input id="tags-on" type="checkbox" checked><span>全員の吹き出し（番号・いまの様子）</span></label>
<label class="tg" id="act-wrap"><input id="act-all" type="checkbox"><span>全員の様子を常に表示</span></label>
<label class="tg" id="court-wrap" style="display:none"><input id="court-on" type="checkbox" checked><span id="court-lbl">コートの枠（推定）</span></label>
<label class="tg" id="ball-wrap" style="display:none"><input id="ball-on" type="checkbox" checked><span>ボール保持（推定）</span></label>
<label class="tg" id="np-wrap" style="display:none"><input id="np-on" type="checkbox"><span>選手以外も表示</span></label>
<label class="tg"><input id="cm-on" type="checkbox" checked><span>コメント(C)</span></label>
<label class="tg"><input id="cm-sel" type="checkbox" checked><span>選んだ人のコメントだけ</span></label>
<button id="fs" type="button">全画面</button></div>
</div>
</details>
<div id="legend" aria-label="枠の色の凡例"></div>
<div id="chips"></div>
<div id="evalbox" aria-live="polite"></div>
<div id="tlwrap"><div id="tl"><div id="tlnames"><div style="height:20px"></div></div><div id="tlscroll"><div id="tlinner"><div id="tick"></div><div id="head"></div></div></div></div></div>
<div id="card" aria-live="polite"></div>
<details id="list"><summary>場面の一覧を開く</summary><div id="listbody"></div></details>
</main>
<div class="tip" id="tip" role="tooltip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
"use strict";
var D=JSON.parse(document.getElementById("data").textContent);
var v=document.getElementById("v"),cv=document.getElementById("c"),ctx=cv.getContext("2d");
var cm=document.getElementById("cm"),cmEls={};var card=document.getElementById("card"),tip=document.getElementById("tip"),chips=document.getElementById("chips");
var tlnames=document.getElementById("tlnames"),tlinner=document.getElementById("tlinner"),tlscroll=document.getElementById("tlscroll"),headEl=document.getElementById("head"),tickEl=document.getElementById("tick");
var showNP=false,slowOn=false,slowUser=false,lastSet=null;
function isOut(q){return q.excluded&&!showNP}
function players(){return D.people.filter(function(q){return !isOut(q)})}
var sel=players().length?players()[0].id:null, zoom=1, allRows=false, more=false, kinds={good:true,event:true,flag:true}, lastKey=null;
function esc(s){return String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function clock(t){t=Math.max(0,Math.floor(t));var m=Math.floor(t/60),s=t%60;return (m<10?"0":"")+m+":"+(s<10?"0":"")+s}
function dur(){return (isFinite(v.duration)&&v.duration>0)?v.duration:D.duration}
function person(id){for(var i=0;i<D.people.length;i++)if(D.people[i].id===id)return D.people[i];return null}
function mainText(m){return m.why||m.what||""}
function flagged(m){return !!(m.suspect||m.weak||m.kind==="change")}   // 要確認: always a symbol + words, never colour alone
var STY=D.team_style||null;
function tw(q){return (STY&&q.tm&&STY[q.tm])?STY[q.tm].name+'のチーム':q.team_word}
function rimFor(hex){var h=String(hex||'#000').replace('#','');var c=[0,2,4].map(function(i){var v=parseInt(h.substr(i,2),16)/255;return v<=0.04045?v/12.92:Math.pow((v+0.055)/1.055,2.4)});return (0.2126*c[0]+0.7152*c[1]+0.0722*c[2])>0.30?'#000000':'#FFFFFF'}
function pname(p){return p.tag+(p.label_note?"（"+p.label_note+"）":"")}
document.getElementById("title").textContent="振り返りプレーヤー　"+D.title+(D.date?"　"+D.date:"");
v.addEventListener("error",function(){document.getElementById("vfail").classList.add("on")});
v.src=D.video;

// --- the info panel: every note, in one place
(function(){var h='<h2 style="margin-top:0">この画面でできること</h2><p>試合の映像を見ながら、AIが気づいた場面（◆）にとべます。人物の名前のボタンを選び、「次の見せ場」を押してください。</p>'
  +'<h2>どこまで信じてよいか</h2><p>枠と文は、AIが映像から作った下書きです。人物の取り違えや文の間違いが混ざります。「▲要確認」の場面は、特に映像で確かめてください。背番号の「？」は、AIが読んだだけで、確かめていない印です。</p>'
  +'<p><b>'+esc(D.banner)+'（'+esc(D.check_note)+'）</b></p><p>'+esc(D.number_note)+esc(D.fix_note)+'</p>';
 if(D.gate_note)h+='<p class="warn">'+esc(D.gate_note)+'</p>';
 h+='<p>'+esc(D.no_mark_note)+'</p>';
 h+='<h2>人物の照合</h2>'+D.people.map(function(p){return '<div>'+esc(p.tag)+'：'+esc(p.checked_note)+'</div>'}).join("");
 h+='<h2>つかいかた</h2><p>タイムラインの印をクリックすると、その'+D.preroll+'秒前から再生します（ハイライト動画と同じ前4秒・後3秒）。←→：前後のイベント、Shift+←→：5秒、N/P：選んだ人の次/前のよかった所、空白：再生/停止。</p>';
 var foot=[];if(D.credit&&D.credit.length)foot=foot.concat(D.credit);
 if(D.notes&&D.notes.length)foot=foot.concat(D.notes);
 if(D.skipped_other_camera)foot.push("別のカメラのイベント "+D.skipped_other_camera+" 件は、このプレーヤーには入れていません。");
 if(D.ball)foot.push(D.ball.note);
 if(D.court)foot.push(D.court.text);
 if(D.court_est)foot.push(D.court_est.text);
 if(D.court_out)foot.push(D.court_out.note);
 if(D.team_style){foot.push(D.team_style.note);if(D.team_style.adjust_note)foot.push(D.team_style.adjust_note)}
 if(D.act_note)foot.push(D.act_note);
 if(D.extra_note)foot.push(D.extra_note);
 if(D.people.some(function(q){return q.moments.concat(q.events).some(function(m){return m.weak})}))foot.push(D.claim_note);
 foot.push("枠や文は、AIの記録をもとに重ねたものです。人物の取り違えや文の間違いがないか、映像で確かめてください。映像はこのパソコンの中だけで再生され、外には送られません。");
 h+='<h2>出典・そのほか</h2>'+foot.map(function(s){return "<p>"+esc(s)+"</p>"}).join("");
 h+='<p><button id="panelx" type="button">閉じる</button></p>';
 document.getElementById("panel").innerHTML=h;
 function toggle(open){var p=document.getElementById("panel");p.classList.toggle("open",open);document.getElementById("info").setAttribute("aria-expanded",open?"true":"false")}
 document.getElementById("info").addEventListener("click",function(){toggle(!document.getElementById("panel").classList.contains("open"))});
 document.getElementById("panelx").addEventListener("click",function(){toggle(false)});
})();

var EXI=null;var LP={};D.people.forEach(function(q){q.labels.forEach(function(l){LP[l]=q})});
var NP={};D.people.forEach(function(q){if(q.nonplayer)q.labels.forEach(function(l){NP[l]=1})});
var T={};Object.keys(D.tracks).forEach(function(k){var rows=D.tracks[k];T[k]={f:rows.map(function(r){return r[0]}),r:rows}});
function boxAt(label,frame){var t=T[label];if(!t||!t.f.length)return null;var f=t.f,lo=0,hi=f.length-1;
 while(lo<hi){var mid=(lo+hi)>>1;if(f[mid]<frame)lo=mid+1;else hi=mid}
 var best=lo;if(lo>0&&Math.abs(f[lo-1]-frame)<=Math.abs(f[lo]-frame))best=lo-1;
 return Math.abs(f[best]-frame)<=D.step?t.r[best]:null}
function vis(m){return (m.kind==="good"&&kinds.good)||(m.kind==="event"&&kinds.event)||((m.suspect||m.kind==="change")&&kinds.flag)}
function marksOf(p){return p.moments.concat(p.events).filter(vis)}
function shownPeople(){return allRows?players():D.people.filter(function(p){return p.id===sel})}
function activeMark(p,t){if(!p)return null;var best=null,bd=1e9;
 marksOf(p).forEach(function(m){if(t>=m.t-D.window[0]&&t<=m.t+D.window[1]){var d=Math.abs(t-m.t)-(m.kind==="good"?0.5:0);if(d<bd){bd=d;best=m}}});return best}

function buildLegend(){var lg=document.getElementById("legend"),seen={},h='<span>枠の色（映像から推定した服の色）：</span>';
 function sw(hex,rim,lbl){return '<span><span class="sw" style="background:'+hex+';border-color:'+rim+'"></span>'+esc(lbl)+'</span>'}
 players().forEach(function(q){var k=tw(q);if(!seen[k]){seen[k]=1;h+=sw(q.color,q.rim||rimFor(q.color),k)}});
 if(D.every)h+='<span><span class="sw" style="background:#8E9196;border-style:dotted;border-color:#000"></span>灰色の点線：選手以外かも（「選手以外も表示」で出ます）</span>';
 if(D.every)h+='<span>頭上の札：番号は全員に。「いまの様子」の文は、選んだ人・ボールを持っていそうな人・見せ場の場面の人だけ（「全員の様子を常に表示」で全員）。</span>';
 if(D.ball)h+='<span><span class="ov"></span>赤い楕円：ボールを持っていそうな人（AIの予想・外れます）</span>';
 if(D.court_out&&D.court_out.n>0)h+='<span><span class="sw" style="background:#8E9196;border-style:dotted;border-color:#000"></span>コート外（控え・観客かも）：外周の外に'+D.court_out.min_s+'秒以上いる人（既定では枠なし。「選手以外も表示」で灰色の点線）</span>';
 if(D.court||D.court_est)h+='<span><span class="ln"></span>赤い線：コートの枠（推定）'+(D.court&&D.court.note?'（'+esc(D.court.note)+'）':'')+'</span>';
 h+='<span><b style="color:#b45309">▲要確認</b>：あてにならないかもしれない場面</span>';
 if(STY){h+='<span class="note">'+esc(STY.note)+(STY.adjust_note?' '+esc(STY.adjust_note):'')+'</span>'}
 lg.innerHTML=h}
// --- people chips
var TEAMS=[["A",STY?STY.A.name+"のチーム":"チームA"],["B",STY?STY.B.name+"のチーム":"チームB"],[null,"チーム不明"]];   // a fixed order: grouping only, nobody is ranked
function buildChips(){chips.innerHTML="";
 var ps=players().slice().sort(function(a,b){return a.id-b.id});
 TEAMS.forEach(function(tm){var g=ps.filter(function(p){return (p.tm||null)===tm[0]});if(!g.length)return;
  var row=document.createElement("div");row.className="tgrp";row.setAttribute("role","group");row.setAttribute("aria-label",tm[1]);
  var h=document.createElement("span");h.className="tgh";h.style.setProperty("--pc",g[0].color);h.style.setProperty("--pi",g[0].ink);h.innerHTML=esc(tm[1])+'<span class="n">'+g.length+'人</span>';row.appendChild(h);
  g.forEach(function(p){var b=document.createElement("button");b.type="button";b.className="chip";b.dataset.id=p.id;b.style.setProperty("--pc",p.color);b.style.setProperty("--pi",p.ink);
   b.setAttribute("aria-pressed",p.id===sel?"true":"false");
   var dot='<span class="dot"></span>';
   b.innerHTML=dot+esc(p.tag)+(p.label_note?' <span class="n">・重複</span>':'');
   if(p.label_note&&p.label_note.indexOf("重複")<0)b.innerHTML=dot+esc(p.tag)+' <span class="n">・要確認</span>';
   if(p.excluded)b.innerHTML+=' <span class="n">・選手以外（推定）</span>';else if(p.nonplayer)b.innerHTML+=' <span class="n">・選手以外かも</span>';
   b.title=pname(p)+"　NiceMill上の名前："+p.name+(p.nonplayer?"　"+p.nonplayer:"");
   b.addEventListener("click",function(){select(p.id,true)});row.appendChild(b)});
  chips.appendChild(row)});
 var r2=document.createElement("div");r2.className="tgrp";
 var a=document.createElement("button");a.type="button";a.id="allon";a.textContent=allRows?"選んだ人だけ":"全員を見る";
 a.addEventListener("click",function(){allRows=!allRows;a.textContent=allRows?"選んだ人だけ":"全員を見る";buildTl();draw()});r2.appendChild(a);chips.appendChild(r2)}
function buildEval(){var box=document.getElementById("evalbox");if(!D.eval_on){box.style.display="none";return}
 var p=person(sel);box.innerHTML="";if(!p){box.style.display="none";return}
 box.style.display="block";box.style.setProperty("--pc",p.color);box.style.setProperty("--pi",p.ink);
 var h='<div class="who">評価文　'+esc(pname(p))+'<span class="tmw">'+esc(tw(p))+'</span></div>';
 if(p.eval){h+='<div class="etx">'+esc(p.eval.text)+'</div><div class="enote">'+esc(p.eval.note)+(p.eval.hedged?'　「要確認」の形にした文があります。':'')+'</div>'}
 else{h+='<div class="etx none">'+esc(D.eval_none_note)+'</div><div class="enote">'+esc(p.eval_why||"")+'</div>'}
 if(p.nonplayer&&!p.excluded)h+='<div class="enote w">選手以外かも：'+esc(p.nonplayer)+'</div>';
 box.innerHTML=h}
function select(id){sel=id;[].forEach.call(chips.querySelectorAll(".chip"),function(c){c.setAttribute("aria-pressed",Number(c.dataset.id)===id?"true":"false")});
 var pc=chips.querySelector('.chip[aria-pressed="true"]'),gr=pc&&pc.parentNode;if(gr&&gr.scrollWidth>gr.clientWidth)gr.scrollLeft=Math.max(0,pc.getBoundingClientRect().left-gr.getBoundingClientRect().left+gr.scrollLeft-(gr.clientWidth-pc.offsetWidth)/2);
 buildEval();buildTl();buildList();lastKey=null;draw()}

// --- timeline
function shape(kind,cited,flag,col){col=col||"#1d4d3f";var s='<svg width="22" height="22" viewBox="0 0 20 20" aria-hidden="true">';
 if(kind==="good")s+='<path d="M10 1 19 10 10 19 1 10Z" fill="'+col+'" stroke="'+(flag?"#1a1f1d":"#fff")+'" stroke-width="'+(flag?2:(cited?2.4:1))+'"'+(flag?' stroke-dasharray="3 2"':'')+'/>';
 else if(kind==="event")s+='<circle cx="10" cy="10" r="4.5" fill="'+col+'" stroke="#1a1f1d" stroke-width="1"/>';
 else s+='<path d="M10 2 18 17H2Z" fill="'+col+'" stroke="#1a1f1d" stroke-width="1.6"/><path d="M10 7v5M10 13.5v1.5" stroke="#fff" stroke-width="1.8"/>';
 return s+'</svg>'}
function kindOf(m){return m.kind==="good"?"good":(m.kind==="change"||m.suspect?"warn":"event")}
function allMarks(){var out=[];shownPeople().forEach(function(p){marksOf(p).forEach(function(m){out.push({p:p,m:m})})});out.sort(function(a,b){return a.m.t-b.m.t});return out}
function buildTl(){tlnames.innerHTML='<div style="height:20px"></div>';
 [].slice.call(tlinner.querySelectorAll(".row")).forEach(function(n){n.remove()});
 var d=dur();
 shownPeople().forEach(function(p){
  var nm=document.createElement("div");nm.className="nm";nm.dataset.id=p.id;nm.style.setProperty("--pc",p.color);var sp=document.createElement("span");
  sp.textContent=p.tag;sp.title=pname(p)+"　NiceMill上の名前："+p.name;nm.appendChild(sp);
  nm.addEventListener("click",function(){select(p.id,true)});tlnames.appendChild(nm);
  var row=document.createElement("div");row.className="row"+(p.id===sel?" sel":"");row.dataset.id=p.id;
  marksOf(p).forEach(function(m){var b=document.createElement("button");b.type="button";var k=kindOf(m);
   b.className="m "+(k==="good"?"g":(k==="event"?"e":"c"));b.dataset.key=p.id+"|"+m.id;b.dataset.t=m.t;b.dataset.pid=p.id;
   b.style.left=(100*m.t/d)+"%";b.innerHTML=shape(k,m.cited,m.suspect||m.weak,p.color);
   b.setAttribute("aria-label",p.tag+"　"+clock(m.t)+"　"+(m.kind==="change"?m.note:mainText(m)+(m.suspect?"　要確認：人物が別人かもしれません":"")));
   b.addEventListener("mouseenter",function(){showTip(b,p,m)});b.addEventListener("focus",function(){showTip(b,p,m)});
   b.addEventListener("mouseleave",hideTip);b.addEventListener("blur",hideTip);
   b.addEventListener("click",function(ev){ev.stopPropagation();if(p.id!==sel)select(p.id);showTip(b,p,m);seek(m.t-D.preroll)});
   row.appendChild(b)});
  tlinner.insertBefore(row,headEl)});
 nowKey=null;ticks();place()}
function ticks(){tickEl.innerHTML="";var d=dur(),w=tlinner.clientWidth||600;var step=[5,10,15,30,60,120,300,600,1800].filter(function(s){return w*s/d>=70})[0]||1800;
 for(var t=0;t<=d;t+=step){var s=document.createElement("span");s.style.left=(100*t/d)+"%";s.textContent=clock(t);tickEl.appendChild(s)}}
function place(){tlinner.style.width=(100*zoom)+"%";headEl.style.left=(100*(v.currentTime||0)/dur())+"%"}
function setZoom(z){var c=(v.currentTime||0)/dur();zoom=Math.max(1,Math.min(32,z));place();ticks();
 var w=tlinner.clientWidth;tlscroll.scrollLeft=Math.max(0,c*w-tlscroll.clientWidth/2)}
document.getElementById("zin").addEventListener("click",function(){setZoom(zoom*2)});
document.getElementById("zout").addEventListener("click",function(){setZoom(zoom/2)});
document.getElementById("zall").addEventListener("click",function(){setZoom(1)});
tlscroll.addEventListener("wheel",function(ev){if(ev.ctrlKey){ev.preventDefault();setZoom(ev.deltaY<0?zoom*2:zoom/2)}},{passive:false});
["good","event","flag"].forEach(function(k){document.getElementById("k-"+k).addEventListener("change",function(e){kinds[k]=e.target.checked;buildTl();buildList();draw()})});
tlinner.addEventListener("click",function(ev){if(ev.target.closest&&ev.target.closest(".m"))return;var r=tlinner.getBoundingClientRect();seek(dur()*(ev.clientX-r.left)/r.width)});
function showTip(b,p,m){var h='<div class="h">'+esc(pname(p))+'　'+esc(clock(m.t))+'</div>';
 if(m.kind==="change")h+='<div class="w">'+esc(m.note)+'</div>';
 else{h+='<div>'+esc(mainText(m))+'</div>';if(m.suspect)h+='<div class="w">▲ '+esc(D.suspect_note)+'</div>';if(m.weak)h+='<div class="w">▲ 要確認：動作は断定していません</div>'}
 tip.innerHTML=h;tip.style.display="block";var r=b.getBoundingClientRect();var tw=tip.offsetWidth,th=tip.offsetHeight;
 var x=Math.min(Math.max(8,r.left+r.width/2-tw/2),window.innerWidth-tw-8),y=r.top-th-8;if(y<8)y=r.bottom+8;tip.style.left=x+"px";tip.style.top=y+"px"}
function hideTip(){tip.style.display="none"}

// --- the one scene card
function npNote(p){if(p.excluded)return '<div class="flag">'+esc(p.np_apart)+'　'+esc(p.nonplayer)+'　<button type="button" id="restore">選手に戻す</button></div>';
 return p.nonplayer?'<div class="flag">選手以外かも：'+esc(p.nonplayer)+'　（選手の可能性もあります。確かめてください）</div>':''}
function drawCard(p,am){
 if(!p){card.innerHTML='<div class="hint">場面がありません。</div>';return}
 if(!am){card.innerHTML='<div class="who">'+esc(pname(p))+'</div>'+npNote(p)+'<div class="hint">いまは、記録された場面の前後ではありません。タイムラインの印を押すと、その場面が出ます。</div>';return}
 var h='<div class="who">'+esc(pname(p))+'　'+esc(clock(am.t))+(am.when?'　'+esc(am.when):'')+'</div>';
 h+=npNote(p);
 if(am.kind==="change"){h+='<div class="flag">'+esc(am.note)+'</div>'}
 else{h+='<div class="main">'+esc(mainText(am))+'</div>';if(am.kind==="event"&&!am.weak)h+='<div class="hint" style="font-size:14px">'+esc(D.event_note)+'</div>';if(am.suspect)h+='<div class="flag">▲ '+esc(D.suspect_note)+'</div>';if(am.weak)h+='<div class="flag">▲ 要確認　'+esc(am.kind==="event"?D.event_weak_note:D.claim_note)+'</div>'}
 h+='<button id="morebtn" type="button" aria-expanded="'+(more?"true":"false")+'">'+(more?"閉じる":"くわしく")+'</button>';
 if(more){h+='<div class="more">';
  if(am.kind!=="change"){if(am.what&&am.what!==mainText(am)&&!am.weak)h+='<div>したこと：'+esc(am.what)+'</div>';
   h+='<div>'+(am.kind==="good"?(am.cited?"よかった所（文の根拠）":"よかった所"):"行動の記録")+(am.category?'　'+esc(am.category):'')+'</div>'}
  h+='<div>NiceMill上の名前：'+esc(p.name)+(p.internal?'　追跡ID：'+esc(p.internal):'')+(p.status==="approved"?'':'　（文は下書き）')+'</div><div>'+esc(p.checked_note)+'</div></div>'}
 card.innerHTML=h;var mb=document.getElementById("morebtn");mb.addEventListener("click",function(){more=!more;lastKey=null;draw()})}

// --- the closed list
function buildList(){var box=document.getElementById("listbody");box.innerHTML="";var p=person(sel);if(!p)return;
 if(p.text_withheld){var wd=document.createElement("div");wd.style.cssText="font-weight:700;margin-top:6px;color:#b45309";wd.textContent="▲要確認：動作の確認で弱めた場面があるため、AIの下書きの文（全文）はここに出しません。NiceMillの確認画面で読み、映像で確かめてください。";box.appendChild(wd)}
 if(p.text){var hd=document.createElement("div");hd.style.cssText="font-weight:700;margin-top:6px";hd.textContent="AIの下書きの文（そのまま）";box.appendChild(hd);var tx=document.createElement("div");tx.style.cssText="color:#4a5651;white-space:pre-wrap;margin:2px 0 6px";tx.textContent=p.text;box.appendChild(tx)}
 p.moments.filter(vis).forEach(function(m){var e=document.createElement("div");e.className="mom";e.tabIndex=0;e.dataset.key=p.id+"|"+m.id;e.setAttribute("role","button");
  e.innerHTML='<span class="t">'+esc(clock(m.t))+'</span>　'+esc(mainText(m))+(flagged(m)?'　<b style="color:#b45309">▲要確認</b>':'');
  function go(){seek(m.t-D.preroll)}
  e.addEventListener("click",go);e.addEventListener("keydown",function(ev){if(ev.key==="Enter"||ev.key===" "){ev.preventDefault();go()}});
  box.appendChild(e)})}

function seek(t){if(isFinite(v.duration))t=Math.min(Math.max(0,t),v.duration);else t=Math.max(0,t);v.currentTime=t;draw()}
function moments(){var p=person(sel);return p?p.moments.filter(vis):[]}
function step(dir){var t=v.currentTime,ms=moments();if(!ms.length)return;var tgt=null;
 if(dir>0){for(var i=0;i<ms.length;i++)if(ms[i].t-D.preroll>t+0.5){tgt=ms[i];break}}
 else{for(var j=ms.length-1;j>=0;j--)if(ms[j].t-D.preroll<t-0.5){tgt=ms[j];break}}
 if(tgt)seek(tgt.t-D.preroll)}
function stepEvent(dir){var t=v.currentTime,ms=allMarks(),tgt=null;
 if(dir>0){for(var i=0;i<ms.length;i++)if(ms[i].m.t-D.preroll>t+0.3){tgt=ms[i];break}}
 else{for(var j=ms.length-1;j>=0;j--)if(ms[j].m.t-D.preroll<t-0.3){tgt=ms[j];break}}
 if(tgt){if(tgt.p.id!==sel)select(tgt.p.id);seek(tgt.m.t-D.preroll);var key=tgt.p.id+"|"+tgt.m.id;var b=tlinner.querySelector('.m[data-key="'+(window.CSS&&CSS.escape?CSS.escape(key):key.replace(/["\\]/g,''))+'"]');if(b){showTip(b,tgt.p,tgt.m);keepVisible(tgt.m.t)}}}
function keepVisible(t){var w=tlinner.clientWidth,x=w*t/dur();if(x<tlscroll.scrollLeft+20||x>tlscroll.scrollLeft+tlscroll.clientWidth-20)tlscroll.scrollLeft=Math.max(0,x-tlscroll.clientWidth/2)}
function setRate(r){lastSet=r;v.playbackRate=r}
document.getElementById("speed").addEventListener("change",function(e){slowOn=false;slowUser=true;setRate(Number(e.target.value));draw()});   // the person's own choice wins until this moment is over
v.addEventListener("ratechange",function(){var sp=document.getElementById("speed"),r=v.playbackRate;if(lastSet!==null&&Math.abs(r-lastSet)<1e-6)return;   // not ours: the browser's own speed menu
 if(slowOn){slowOn=false;slowUser=true}lastSet=r;if([].some.call(sp.options,function(o){return Number(o.value)===r}))sp.value=String(r)});
document.getElementById("boxes").addEventListener("change",draw);document.getElementById("allbox").addEventListener("change",draw);
if(D.court||D.court_est){document.getElementById("court-wrap").style.display="";document.getElementById("court-on").addEventListener("change",draw);if(D.court)document.getElementById("court-lbl").textContent=(D.court.note==="登録"?"コートの枠（登録）":"コートの枠（登録・追跡は推定）")}
if(D.every){document.getElementById("allbox").checked=true}else{document.getElementById("tags-wrap").style.display="none";document.getElementById("act-wrap").style.display="none"}
document.getElementById("tags-on").addEventListener("change",draw);document.getElementById("act-all").addEventListener("change",draw);
if(D.ball){document.getElementById("ball-wrap").style.display="";document.getElementById("ball-on").addEventListener("change",draw)}
(function(){var any=D.people.some(function(q){return q.excluded})||!!(D.court_out&&D.court_out.n>0);if(!any)return;var w=document.getElementById("np-wrap");w.style.display="";
 document.getElementById("np-on").addEventListener("change",function(e){showNP=e.target.checked;if(!showNP&&person(sel)&&isOut(person(sel))){var f=players()[0];sel=f?f.id:null}buildChips();buildEval();buildTl();buildList();lastKey=null;draw()});
 card.addEventListener("click",function(ev){if(ev.target&&ev.target.id==="restore"){var q=person(sel);if(q){q.excluded=false;buildChips();buildEval();buildLegend();buildTl();buildList();lastKey=null;draw()}}})})();
document.addEventListener("keydown",function(ev){if(ev.ctrlKey||ev.metaKey||ev.altKey)return;var tg=ev.target&&ev.target.tagName;var cbx=tg==="INPUT"&&ev.target.type==="checkbox";
 if(tg==="SELECT"||tg==="SUMMARY"||(tg==="INPUT"&&!cbx)||tg==="TEXTAREA")return;
 if(cbx&&(ev.key===" "||ev.key==="Enter"))return;
 if(ev.target&&ev.target.classList&&(ev.target.classList.contains("mom")||ev.target.classList.contains("chip")||tg==="BUTTON")&&(ev.key===" "||ev.key==="Enter"))return;
 if(ev.key===" "){ev.preventDefault();v.paused?v.play():v.pause()}
 else if(ev.key==="ArrowLeft"){ev.preventDefault();ev.shiftKey?seek(v.currentTime-5):stepEvent(-1)}
 else if(ev.key==="ArrowRight"){ev.preventDefault();ev.shiftKey?seek(v.currentTime+5):stepEvent(1)}
 else if(ev.key==="c"||ev.key==="C"){var cb=document.getElementById("cm-on");cb.checked=!cb.checked;draw()}
 else if(ev.key==="n"||ev.key==="N"){step(1)}else if(ev.key==="p"||ev.key==="P"){step(-1)}
 else if(ev.key==="f"||ev.key==="F"){document.getElementById("fs").click()}
 else if(ev.key==="?"){document.getElementById("f-help").click()}
 else if(ev.key==="Escape"){hideTip()}});

// --- live comments over the video
var reduce=window.matchMedia&&window.matchMedia("(prefers-reduced-motion: reduce)").matches;
function activeComments(t,cw){var out=[],only=document.getElementById("cm-sel").checked;
 players().forEach(function(p){if(only&&p.id!==sel)return;
  p.moments.forEach(function(m){if(m.kind!=="good"||!kinds.good)return;
   var a=m.t-D.window[0],b=m.t+D.window[1];if(t>=a&&t<=b)out.push({p:p,m:m,d:Math.abs(t-m.t)+(p.id===sel?0:100)})})});
 out.sort(function(x,y){return x.d-y.d});var seen={},one=[];out.forEach(function(a){if(!seen[a.p.id]){seen[a.p.id]=1;one.push(a)}});   // one bubble per person: the nearest moment
 return one.slice(0,D.max_bubbles)}   // never more than two at once: the match must stay visible
function headBox(p,frame,sx,sy,L){var b=null;for(var i=0;i<p.labels.length&&!b;i++)b=boxAt(p.labels[i],frame);
 if(!b)return null;var x=b[1]*sx,y=b[2]*sy,w=b[3]*sx,h=b[4]*sy;
 var vis=Math.max(0,Math.min(x+w,L.cw)-Math.max(x,0))*Math.max(0,Math.min(y+h,L.ch)-Math.max(y,0));
 if(w<=0||h<=0||vis<0.5*w*h)return null;return {x:x,y:y,w:w,h:h}}
function updateComments(L,t,frame){
 var on=document.getElementById("cm-on").checked;var list=on?activeComments(t,L.cw):[];var keep={};
 var sx=L.cw/D.size[0],sy=L.ch/D.size[1],placed=[],capY=Math.max(56,Math.min(64,L.ch*0.18)),capUsed=0,fs=Math.max(12,Math.min(15,L.cw/75));
 list.forEach(function(a){var key=a.p.id+"|"+a.m.id;keep[key]=1;var hb=headBox(a.p,frame,sx,sy,L),mode=hb?"bl":"cap";
  var e=cmEls[key];if(e&&e._mode!==mode){e.remove();delete cmEls[key];e=null}
  var flag=a.m.suspect||a.m.weak;
  if(!e){e=document.createElement("div");e._mode=mode;e.title="AI下書き"+(flag?"・要確認":"")+"　"+mainText(a.m);
   e.style.setProperty("--pc",a.p.color);e.style.setProperty("--pi",a.p.ink);
   var chip='<span class="cn">'+esc(a.p.tag)+'</span>';
   var said=mainText(a.m).indexOf("要確認")>=0;var txt=(flag?'<span class="fl">▲'+(said?'':'要確認')+'</span>':'')+esc(mainText(a.m));
   if(mode==="bl"){e.className="bl"+(flag?" flag":"");e.innerHTML=chip+'<span class="s">'+txt+'</span>'}
   else{e.className="cmt"+(flag?" warn":"");e.innerHTML=chip+'<span class="s">'+txt+'</span>'}
   cm.appendChild(e);cmEls[key]=e;
   if(reduce)e.classList.add("in");else requestAnimationFrame(function(){e.classList.add("in")})}
  e.classList.remove("out");e.style.fontSize=fs+"px";
  var w=e.offsetWidth,h=e.offsetHeight,x,y;
  if(mode==="bl"){var cx=hb.x+hb.w/2;x=Math.max(2,Math.min(cx-w/2,L.cw-w-2));y=hb.y-(a.p.id===sel&&document.getElementById("boxes").checked?26:0)-h-10;e.classList.remove("dn");
   if(y<2){y=Math.min(L.ch-h-2,hb.y+hb.h+12);e.classList.add("dn")}
   for(var n=0;n<placed.length;n++){var q=placed[n];if(x<q.x+q.w&&q.x<x+w&&y<q.y+q.h&&q.y<y+h){y=e.classList.contains("dn")?q.y+q.h+4:q.y-h-4;n=-1;if(y<2){y=2;break}}}
   e.style.setProperty("--tx",Math.max(12,Math.min(w-12,cx-x))+"px")}
  else{x=8;y=L.ch-h-capY-capUsed;capUsed+=h+4}
  if(e._x==null||reduce||!isFinite(e._x)||Math.abs(e._x-x)>L.cw*0.3){e._x=x;e._y=y}else{e._x+=(x-e._x)*0.45;e._y+=(y-e._y)*0.45}
  e.style.left=e._x+"px";e.style.top=e._y+"px";if(mode==="bl")placed.push({x:e._x,y:e._y,w:w,h:h})});
 Object.keys(cmEls).forEach(function(k){if(keep[k])return;var e=cmEls[k];delete cmEls[k];
  if(reduce){e.remove()}else{e.classList.remove("in");setTimeout(function(){e.remove()},300)}});
 var sp={};list.forEach(function(a){sp[a.p.id]=1});return sp}
document.getElementById("cm-on").addEventListener("change",draw);document.getElementById("cm-sel").addEventListener("change",draw);
(function(){var st=document.getElementById("stage"),de=document.documentElement,btn=document.getElementById("fs");
 var req=st.requestFullscreen||st.webkitRequestFullscreen,out=document.exitFullscreen||document.webkitExitFullscreen;
 if(!(de.requestFullscreen||de.webkitRequestFullscreen))btn.style.display="none";   // an iPhone cannot show an element full screen: no button that does nothing
 btn.addEventListener("click",function(){if(document.fullscreenElement||document.webkitFullscreenElement){out.call(document)}else if(req){var r=req.call(st);if(r&&r.catch)r.catch(function(){})}})})();
["fullscreenchange","webkitfullscreenchange"].forEach(function(n){document.addEventListener(n,function(){setTimeout(draw,50)})});

// --- one-look features (each off unless it is the live memo; all need only the data on the page)
var featMark=null;
function nowMoment(t){var best=null,bd=1e9;players().forEach(function(q){q.moments.forEach(function(m){if(m.kind!=="good")return;
  if(t>=m.t-D.window[0]&&t<=m.t+D.window[1]){var d=Math.abs(t-m.t);if(d<bd){bd=d;best={p:q,m:m}}}})});return best}
function updateLive(t,frame){var el=document.getElementById("live");var on=document.getElementById("f-live").checked;el.classList.toggle("on",on);if(!on)return;
 var cnt={},n=0;players().forEach(function(q){var seen=false;for(var i=0;i<q.labels.length&&!seen;i++)seen=!!boxAt(q.labels[i],frame);if(seen){n++;cnt[tw(q)]=(cnt[tw(q)]||0)+1}});
 var h='<span class="tag">実況メモ・AI下書き</span>';
 if(!n){h+='この時刻の人物の記録がありません'}
 else{h+='いま画面にいる選べる人：'+n+'人（'+Object.keys(cnt).map(function(k){return k+' '+cnt[k]+'人'}).join('・')+'）';
  var nm=nowMoment(t);if(nm)h+='　記録された場面：'+esc(nm.p.tag)+'「'+esc(mainText(nm.m))+'」'+(flagged(nm.m)?'（▲要確認）':'');
  if(D.ball&&document.getElementById("ball-on").checked){for(var si=0;si<D.ball.segs.length;si++){var sg=D.ball.segs[si];if(frame>=sg[0]&&frame<=sg[1]){var hp=D.people.filter(function(q){return q.labels.indexOf(sg[2])>=0&&!isOut(q)})[0];if(hp)h+='　ボール保持（推定）：'+esc(hp.tag);break}}}}
 if(el.dataset.h!==h){el.dataset.h=h;el.innerHTML=h.replace(/^(<span class="tag">[^<]*<\/span>)(.*)$/,'$1<span class="clip" title="">$2</span>');var cl=el.querySelector(".clip");if(cl)cl.title=cl.textContent}}
function drawFeatures(L,t,frame,sx,sy){var p=person(sel);
 if(document.getElementById("f-spot").checked&&p){var b=null;for(var i=0;i<p.labels.length&&!b;i++)b=boxAt(p.labels[i],frame);
  if(b){ctx.save();ctx.fillStyle="rgba(0,0,0,.55)";ctx.beginPath();ctx.rect(0,0,L.cw,L.ch);
   var cx=(b[1]+b[3]/2)*sx,cy=(b[2]+b[4]/2)*sy,rx=Math.max(40,b[3]*sx*1.1),ry=Math.max(60,b[4]*sy*0.8);ctx.ellipse(cx,cy,rx,ry,0,0,6.2832,true);ctx.fill("evenodd");ctx.restore()}}
 if(document.getElementById("f-trail").checked&&p){var pts=[];for(var d=0;d<=D.fps*3;d+=Math.max(1,Math.round(D.step))){var bb=null;for(var j=0;j<p.labels.length&&!bb;j++)bb=boxAt(p.labels[j],frame-d);if(bb)pts.push([(bb[1]+bb[3]/2)*sx,(bb[2]+bb[4])*sy])}
  if(pts.length>1){ctx.save();ctx.lineCap="round";for(var k=1;k<pts.length;k++){ctx.globalAlpha=1-k/pts.length;ctx.lineWidth=7;ctx.strokeStyle="#fff";ctx.beginPath();ctx.moveTo(pts[k-1][0],pts[k-1][1]);ctx.lineTo(pts[k][0],pts[k][1]);ctx.stroke();
   ctx.lineWidth=4;ctx.strokeStyle=p.color;ctx.beginPath();ctx.moveTo(pts[k-1][0],pts[k-1][1]);ctx.lineTo(pts[k][0],pts[k][1]);ctx.stroke()}ctx.restore()}}
 updateLive(t,frame);
 // slow motion inside a moment, and the chain of moments
 var am2=p?activeMark(p,t):null,inGood=!!(am2&&am2.kind==="good");if(!inGood)slowUser=false;
 var slow=document.getElementById("f-slow").checked&&inGood&&!slowUser;
 if(slow!==slowOn){slowOn=slow;var base=Number(document.getElementById("speed").value)||1;setRate(slowOn?Math.min(base,0.5):base)}
 if(document.getElementById("f-auto").checked&&p&&!v.paused){var ms=p.moments.filter(function(m){return m.kind==="good"}).sort(function(a,b){return a.t-b.t});
  var cur=null;ms.forEach(function(m){if(t>=m.t-D.window[0]-0.3&&t<=m.t+D.window[1])cur=m});
  if(!cur){var nx=ms.filter(function(m){return m.t-D.preroll>t+0.2})[0];if(nx)seek(nx.t-D.preroll);else v.pause()}}}
document.getElementById("nextm").addEventListener("click",function(){step(1)});document.getElementById("prevm").addEventListener("click",function(){step(-1)});
["f-auto","f-slow","f-spot","f-trail","f-live"].forEach(function(id){document.getElementById(id).addEventListener("change",draw)});
document.getElementById("f-auto").addEventListener("change",function(e){if(e.target.checked){var p=person(sel),ms=p?p.moments.filter(function(m){return m.kind==="good"}).sort(function(a,b){return a.t-b.t}):[];if(ms.length){seek(ms[0].t-D.preroll);v.play()}}});
document.getElementById("f-help").addEventListener("click",function(){var o=document.getElementById("helpbox").classList.toggle("open");this.setAttribute("aria-expanded",o?"true":"false")});
function shotCanvas(withVideo){var vw=v.videoWidth||D.size[0],vh=v.videoHeight||D.size[1],bh=Math.max(48,Math.round(vh*0.09)),c=document.createElement("canvas");c.width=vw;c.height=vh+bh;var x=c.getContext("2d");
 x.fillStyle="#222";x.fillRect(0,0,vw,vh+bh);if(withVideo)x.drawImage(v,0,0,vw,vh);
 x.drawImage(cv,0,0,vw,vh);
 var p=person(sel),am=p?activeMark(p,v.currentTime):null,fs=Math.round(bh*0.3);x.fillStyle="#fff";x.fillRect(0,vh,vw,bh);x.fillStyle="#1a1f1d";x.font="bold "+fs+"px sans-serif";x.textBaseline="top";
 var line1="AIの下書き・要確認　"+clock(v.currentTime)+(p?"　"+pname(p):"")+(withVideo?"":"　（このブラウザの制限で映像は入っていません。枠と文だけです）");
 var line2=am?(flagged(am)?"▲要確認　":"")+(am.kind==="change"?am.note:mainText(am)):"";
 x.fillText(line1,8,vh+4,vw-16);x.font=fs+"px sans-serif";x.fillText(line2,8,vh+4+fs*1.3,vw-16);return c}
document.getElementById("f-shot").addEventListener("click",function(){
 var url,withVideo=true;
 try{url=shotCanvas(true).toDataURL("image/png")}catch(e){withVideo=false;url=shotCanvas(false).toDataURL("image/png")}   // some browsers will not export a local file's video frame
 var a=document.createElement("a");a.href=url;a.download="review_"+clock(v.currentTime).replace(":","-")+(withVideo?"":"_boxes")+".png";a.click();
 if(!withVideo)alert("このブラウザでは、ローカルのファイルの映像の1コマを保存できません。枠と文だけを保存しました。映像ごと残すには、ブラウザやOSの画面キャプチャを使ってください。")});
// --- drawing
function layout(){var w=v.clientWidth,h=v.clientHeight;var dpr=window.devicePixelRatio||1;
 var vw=v.videoWidth||D.size[0],vh=v.videoHeight||D.size[1];var k=Math.min(w/vw,h/vh);var cw=vw*k,ch=vh*k;
 var ox=(w-cw)/2,oy=(h-ch)/2;
 cv.style.left=ox+"px";cv.style.top=oy+"px";cv.style.width=cw+"px";cv.style.height=ch+"px";
 cm.style.left=ox+"px";cm.style.top=oy+"px";cm.style.width=cw+"px";cm.style.height=ch+"px";
 if(cv.width!==Math.round(cw*dpr)||cv.height!==Math.round(ch*dpr)){cv.width=Math.round(cw*dpr);cv.height=Math.round(ch*dpr)}
 return {cw:cw,ch:ch,dpr:dpr}}
var settle=0,nowKey=null;
function optState(){var on=[["f-live","実況メモ"],["f-auto","連続再生"],["f-slow","0.5倍速"],["f-spot","スポットライト"],["f-trail","足跡"],["np-on","選手以外も表示"],["act-all","全員の様子を常に表示"]],off=[["cm-on","コメント"],["boxes","選んだ人の枠"],["allbox","全員の枠"],["tags-on","全員の吹き出し"]],a=[],b=[];
 on.forEach(function(x){var e=document.getElementById(x[0]);if(e&&e.checked)a.push(x[1])});off.forEach(function(x){var e=document.getElementById(x[0]);if(e&&!e.checked)b.push(x[1])});
 var s=(a.length?"　オン："+a.join("・"):"")+(b.length?"　オフ："+b.join("・"):"");var el=document.getElementById("optstate");if(el.textContent!==s)el.textContent=s}
function draw(){settle=20;optState();render()}
var statLast={framed:0,tagged:0,compact:0,skipped:0,visible:0};
function exAt(frame){if(!D.extras)return [];if(!EXI){EXI={};D.extras.forEach(function(r){(EXI[r[0]]=EXI[r[0]]||[]).push(r)})}return EXI[Math.round(frame/D.step)*D.step]||[]}
function tmColor(tm){return tm===1&&STY?STY.A.hex:(tm===2&&STY?STY.B.hex:(tm===1?"#0072B2":(tm===2?"#CC79A7":"#8E9196")))}
function everyoneAt(frame){var out=[];
 players().forEach(function(q){q.labels.forEach(function(l){var b=boxAt(l,frame);if(!b)return;out.push({b:b,q:q,l:l,key:l,tm:q.tm==="A"?1:(q.tm==="B"?2:0),act:b[5]|0,sel:q.id===sel,np:!!q.excluded})})});
 Object.keys(T).forEach(function(l){if(LP[l])return;var b=boxAt(l,frame);if(!b)return;var tm=(D.others_team&&D.others_team[l])||0,o=!!b[6];out.push({b:b,q:null,l:l,key:l,tm:tm,act:b[5]|0,sel:false,np:!tm||!!NP[l]||o,co:o&&!!tm&&!NP[l]})});
 exAt(frame).forEach(function(r){var tm=r[5]|0,o=!!r[7];out.push({b:r,q:null,key:extraKey(r),tm:tm,act:r[6]|0,sel:false,np:!tm||o,co:o&&!!tm})});
 return out.filter(function(e){return !e.np||showNP}).sort(function(a,b){return (b.np-a.np)||((a.b[2]+a.b[4])-(b.b[2]+b.b[4]))})}
function drawEveryone(es,sx,sy){es.forEach(function(e){var b=e.b,x=b[1]*sx,y=b[2]*sy,w=b[3]*sx,h=b[4]*sy;
  var col=e.np?"#8E9196":(e.q?e.q.color:tmColor(e.tm)),rim=rimFor(col),dashed=e.np||(e.q&&e.q.dashed)||(!e.q&&!e.np&&false);
  ctx.setLineDash(e.np?[3,4]:(e.q&&e.q.dashed?[8,5]:[]));
  ctx.lineWidth=e.np?3.6:6.2;ctx.strokeStyle=rim;ctx.strokeRect(x,y,w,h);
  ctx.lineWidth=e.np?1.6:3.6;ctx.strokeStyle=col;ctx.strokeRect(x,y,w,h);ctx.setLineDash([]);
  if(e.co){ctx.font="bold 11px sans-serif";var s="コート外かも",tw3=ctx.measureText(s).width+8,ty3=Math.max(0,y-15);ctx.fillStyle="rgba(60,66,76,.88)";ctx.fillRect(x,ty3,tw3,14);ctx.fillStyle="#fff";ctx.fillText(s,x+4,ty3+11)}})}
var tw_cache={};
function twid(str,fs){var k=fs+str;if(tw_cache[k]==null)tw_cache[k]=ctx.measureText(str).width;return tw_cache[k]}
function rrect(x,y,w,h,r){ctx.beginPath();ctx.moveTo(x+r,y);ctx.arcTo(x+w,y,x+w,y+h,r);ctx.arcTo(x+w,y+h,x,y+h,r);ctx.arcTo(x,y+h,x,y,r);ctx.arcTo(x,y,x+w,y,r);ctx.closePath()}
var CAL=D.calm||{},CALI={},ETID=null;
function calmAt(key,frame){var s=CALI[key];if(!s){var rows=CAL[key];if(!rows||!rows.length)return null;s=CALI[key]={f:rows.map(function(r){return r[0]}),r:rows}}
 var f=s.f,lo=0,hi=f.length-1;while(lo<hi){var m=(lo+hi)>>1;if(f[m]<frame)lo=m+1;else hi=m}
 var b=s.r[lo],a=lo>0?s.r[lo-1]:null;
 if(a&&b&&f[lo]>=frame&&f[lo]-f[lo-1]<=2*D.step){var u=(frame-f[lo-1])/Math.max(1,f[lo]-f[lo-1]);return {ax:a[1]+(b[1]-a[1])*u,ay:a[2]+(b[2]-a[2])*u,act:u<.5?a[3]:b[3]}}
 var best=b,bf=f[lo];if(a&&Math.abs(f[lo-1]-frame)<Math.abs(bf-frame)){best=a;bf=f[lo-1]}
 return Math.abs(bf-frame)<=D.step?{ax:best[1],ay:best[2],act:best[3]}:null}
function extraKey(r){if(!D.extra_tid)return null;if(!ETID){ETID=new Map();D.extras.forEach(function(x,i){ETID.set(x,D.extra_tid[i])})}var t=ETID.get(r);return t==null?null:"e"+t}
function holderLbl(frame){if(!D.ball)return null;for(var i=0;i<D.ball.segs.length;i++){var g=D.ball.segs[i];if(frame>=g[0]&&frame<=g[1])return g[2]}return null}
function ease(cur,tgt,db){var d=tgt-cur;if(Math.abs(d)<=db)return cur;var m=(d-(d>0?db:-db))*0.2;return cur+Math.max(-12,Math.min(12,m))}
var TS={},tsT=-1,SENT_RETRY=2.5,FADE_S=0.2,HOLD_GHOST=0.6;
function drawChip(x,y,w,h,ls,col,rim,sel,tx,ay,fs,lh,alpha){
 ctx.globalAlpha=alpha;
 ctx.fillStyle="rgba(255,255,255,.95)";rrect(x,y,w,h,5);ctx.fill();
 ctx.lineWidth=sel?5:3.6;ctx.strokeStyle=rim;ctx.stroke();ctx.lineWidth=sel?3:2;ctx.strokeStyle=col;ctx.stroke();
 var apex=Math.max(y+h+4,Math.min(ay-2,y+h+14));
 ctx.fillStyle=rim;ctx.beginPath();ctx.moveTo(tx-6,y+h);ctx.lineTo(tx+6,y+h);ctx.lineTo(tx,apex+2);ctx.closePath();ctx.fill();
 ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(tx-3.5,y+h);ctx.lineTo(tx+3.5,y+h);ctx.lineTo(tx,apex-1);ctx.closePath();ctx.fill();
 ctx.fillStyle="#1a1f1d";ls.forEach(function(z,zi){ctx.font=(zi?"":"bold ")+fs+"px sans-serif";ctx.fillText(z,x+(w>60?8:5),y+(h>lh*ls.length+5?4:2)+lh*(zi+1)-3)});
 ctx.globalAlpha=1}
function drawTags(L,es,sx,sy,sp){
 var t=v.currentTime||0,frame=Math.round(t*D.fps)+1,jump=(Math.abs(t-tsT)>0.5||t<tsT-0.01),dt=jump?0:Math.min(0.1,Math.max(0,t-tsT));
 if(jump)TS={};tsT=t;
 var phone=L.cw<520,fs=Math.max(phone?10:11,Math.min(34,Math.round(L.ch*0.026))),lh=fs+(phone?3:5),placed=[],st={framed:es.length,tagged:0,compact:0,skipped:0,visible:es.length},selBox=document.getElementById("boxes").checked;
 var showAll=document.getElementById("act-all").checked,ballOn=D.ball&&document.getElementById("ball-on").checked,hl=ballOn?holderLbl(frame):null,live={},seen={};
 D.people.forEach(function(q){if(activeMark(q,t))live[q.id]=1});
 ctx.textBaseline="alphabetic";
 var list=[];
 es.forEach(function(e){if(e.np)return;if(e.q&&sp[e.q.id]){st.tagged++;return}
  var key=e.key||(e.q?"q"+e.q.id:null),c=e.key?calmAt(e.key,frame):null;
  var ax=(c?c.ax:(e.b[1]+e.b[3]/2))*sx,ay=(c?c.ay:e.b[2])*sy,act=c?c.act:(e.act|0);
  var word=(D.act_words&&D.act_words[act])||"";
  var want=!!word&&(phone?e.sel:(showAll||e.sel||(hl&&e.l===hl)||(e.q&&live[e.q.id])));
  list.push({e:e,key:key||("x"+Math.round(e.b[1])),ax:ax,ay:ay,word:word,want:want})});
 list.sort(function(a,b){return (b.e.sel-a.e.sel)||(b.want-a.want)||(b.e.b[4]-a.e.b[4])});
 function free(x,y,w,h){for(var n=0;n<placed.length;n++){var q=placed[n];if(x<q.x+q.w&&q.x<x+w&&y<q.y+q.h&&q.y<y+h)return false}return true}
 list.forEach(function(it){var e=it.e,key=it.key;seen[key]=1;
  var s=TS[key];if(!s){s=TS[key]={a:jump||v.paused?1:0,x:null,y:null,k:0,di:0,noSent:-1,sent:null,failSince:null,last:null,seen:t}}
  s.seen=t;
  if(it.want){if(!(s.wantUntil>t))s.wantStart=t;s.wantUntil=Math.max(s.wantStart+2.5,t+1.5)}   // a sentence shows at least 2.5 s and rides over gaps shorter than 1.5 s
  var wantNow=it.want||s.wantUntil>t;
  var num=e.q?e.q.tag:"番号不明",col=e.q?e.q.color:tmColor(e.tm),rim=rimFor(col),top=it.ay-(e.sel&&selBox?26:0),done=false;
  var modes=(wantNow&&it.word&&s.noSent<=t)?[2,1]:[1];
  for(var mi=0;mi<modes.length&&!done;mi++){var md=modes[mi],ls=md===2?[num,it.word]:[num];
   ctx.font="bold "+fs+"px sans-serif";var w=twid(num,fs);if(md===2){ctx.font=fs+"px sans-serif";w=Math.max(w,twid(it.word,"p"+fs))}w+=phone?10:16;var h=ls.length*lh+(phone?4:8);
   var cands=[[s.k,s.di]];for(var k=0;k<4;k++)for(var di=0;di<3;di++)if(k!==s.k||di!==s.di)cands.push([k,di]);
   var put=null;
   for(var ci=0;ci<cands.length&&!put;ci++){var kk=cands[ci][0],dd=cands[ci][1],ty=top-h-8-kk*(h+3);if(ty<1)continue;
    var tx=Math.max(2,Math.min(it.ax-w/2+[0,0.62*w,-0.62*w][dd],L.cw-w-2)),ex=tx,ey=ty;
    if(s.x!==null){ex=ease(s.x,tx,6);ey=ease(s.y,ty,4)}
    var X=Math.round(ex),Y=Math.round(ey);
    if(free(tx,ty,w,h)&&free(X,Y,w,h))put={k:kk,di:dd,x:ex,y:ey,X:X,Y:Y,w:w,h:h}}
   if(!put&&md===2&&s.mode===2&&s.last){      // the sentence does not fit just now: keep it where it is for a moment
    if(s.failSince==null)s.failSince=t;
    if(t-s.failSince<0.6||t-(s.sentStart||0)<2.5){var g=s.last;put={k:s.k,di:s.di,x:s.x,y:s.y,X:g.X,Y:g.Y,w:g.w,h:g.h,held:1}}}
   if(!put)continue;
   if(md===2&&s.mode!==2)s.sentStart=t;
   s.k=put.k;s.di=put.di;s.x=put.x;s.y=put.y;s.mode=md;if(!put.held)s.failSince=null;
   placed.push({x:put.X,y:put.Y,w:put.w,h:put.h});
   s.a=Math.min(1,s.a+(dt?dt/FADE_S:(v.paused?1:0)));
   var tail=Math.max(put.X+9,Math.min(it.ax,put.X+put.w-9));
   drawChip(put.X,put.Y,put.w,put.h,ls,col,rim,e.sel,tail,it.ay,fs,lh,s.a);
   s.last={X:put.X,Y:put.Y,w:put.w,h:put.h,ls:ls,col:col,rim:rim,sel:e.sel,tail:tail,ay:it.ay};
   if(window.__tl)window.__tl.push({t:t,x:put.X,y:put.Y,w:put.w,h:put.h,txt:ls.join("|"),fs:fs,ax:(e.b[1]+e.b[3]/2)*sx,ay:e.b[2]*sy-(e.sel&&selBox?26:0),sax:it.ax,say:top,sel:e.sel?1:0,q:e.q?e.q.id:-1});
   st.tagged++;
   if(md===1&&wantNow&&modes.length===2){s.noSent=t+SENT_RETRY;st.compact++}   // a chip only: the sentence did not fit; not tried again for a while
   done=true}
  if(!done)st.skipped++});
 Object.keys(TS).forEach(function(k){var s=TS[k];if(seen[k])return;var gap=t-s.seen;
  if(gap>1||gap<0){delete TS[k];return}
  if(s.last&&gap<=HOLD_GHOST){var g=s.last;drawChip(g.X,g.Y,g.w,g.h,g.ls,g.col,g.rim,g.sel,g.tail,g.ay,fs,lh,s.a*(1-gap/HOLD_GHOST))}});
 statLast=st}
function courtAt(frame){var S=D.court.frames,lo=0,hi=S.length;while(lo<hi){var m=(lo+hi)>>1;if(S[m][0]<frame)lo=m+1;else hi=m}
 var a=lo>0?S[lo-1]:null,b=lo<S.length?S[lo]:null,i,o=[];
 if(a&&b&&b[0]-a[0]<=31){var u=(frame-a[0])/Math.max(1,b[0]-a[0]);for(i=1;i<9;i++)o.push(a[i]+(b[i]-a[i])*u);return o}
 var da=a?frame-a[0]:1e9,db=b?b[0]-frame:1e9;
 if(da<=db&&da<=20)return a.slice(1);if(db<da&&db<=20)return b.slice(1);return null}
function lineAt(k,frame){var S=D.court_est.samples,lo=0,hi=S.length;while(lo<hi){var m=(lo+hi)>>1;if(S[m][0]<frame)lo=m+1;else hi=m}
 var a=lo>0?S[lo-1]:null,b=lo<S.length?S[lo]:null,la=a&&a[k],lb=b&&b[k];
 if(la&&lb&&b[0]-a[0]<=45){var u=(frame-a[0])/Math.max(1,b[0]-a[0]);return la.map(function(v,j){return v+(lb[j]-v)*u})}
 var da=a?frame-a[0]:1e9,db=b?b[0]-frame:1e9;
 if(la&&da<=db&&da<=20)return la;if(lb&&db<da&&db<=20)return lb;return null}
function render(){var L=layout();ctx.setTransform(L.dpr,0,0,L.dpr,0,0);ctx.clearRect(0,0,L.cw,L.ch);
 var t=v.currentTime||0, frame=Math.round(t*D.fps)+1;
 var p=person(sel), am=activeMark(p,t);
 var sx=L.cw/D.size[0],sy=L.ch/D.size[1];
 if(D.every&&document.getElementById("allbox").checked){var es=everyoneAt(frame);drawEveryone(es,sx,sy)}else{var es=[]}
 if(document.getElementById("boxes").checked){var mine={};if(p)p.labels.forEach(function(l){mine[l]=1});
  if(!D.every&&document.getElementById("allbox").checked){      // old pictures without the analysis: everyone else faint, in grey
   Object.keys(T).forEach(function(l){if(mine[l])return;var b=boxAt(l,frame);if(!b)return;var np=NP[l],q=LP[l];
    ctx.globalAlpha=.6;ctx.lineWidth=2;ctx.strokeStyle=q?q.color:"rgba(150,156,166,.9)";ctx.setLineDash(np?[3,4]:[]);
    ctx.strokeRect(b[1]*sx,b[2]*sy,b[3]*sx,b[4]*sy);ctx.setLineDash([]);ctx.globalAlpha=1;
    if(np){ctx.font="12px sans-serif";var s="選手以外かも",tw2=ctx.measureText(s).width+8,ty2=Math.max(0,b[2]*sy-16);
     ctx.fillStyle="rgba(60,66,76,.85)";ctx.fillRect(b[1]*sx,ty2,tw2,15);ctx.fillStyle="#fff";ctx.fillText(s,b[1]*sx+4,ty2+12)}})}
  if(!D.every&&document.getElementById("allbox").checked&&D.extras){
   var fr=exAt(frame);ctx.globalAlpha=.7;ctx.lineWidth=1.5;ctx.strokeStyle="rgba(190,194,200,.95)";
   fr.forEach(function(r){ctx.strokeRect(r[1]*sx,r[2]*sy,r[3]*sx,r[4]*sy)});ctx.globalAlpha=1}
  var warn=am&&(am.suspect||am.kind==="change");var col=p?p.color:"#2a78d6";
  if(p)p.labels.forEach(function(l){var b=boxAt(l,frame);if(!b)return;var x=b[1]*sx,y=b[2]*sy,w=b[3]*sx,h=b[4]*sy;
   var rim=rimFor(p.color);
   if(p.dashed||warn){ctx.setLineDash(warn&&!p.dashed?[2,4]:[8,5]);ctx.lineWidth=7;ctx.strokeStyle=rim;ctx.strokeRect(x,y,w,h);ctx.lineWidth=4.5;ctx.strokeStyle=col;ctx.strokeRect(x,y,w,h);ctx.setLineDash([])}
   else{ctx.lineWidth=9;ctx.strokeStyle=rim;ctx.strokeRect(x,y,w,h);ctx.lineWidth=5;ctx.strokeStyle=col;ctx.strokeRect(x,y,w,h)}
   var label=p.tag+(warn?"　▲別の人かも":"");ctx.font="bold 15px sans-serif";var tw=ctx.measureText(label).width+12;
   var ty=Math.max(0,y-24);ctx.fillStyle=col;ctx.fillRect(x,ty,tw,22);ctx.strokeStyle=rim;ctx.lineWidth=1.5;ctx.strokeRect(x,ty,tw,22);ctx.fillStyle=p.ink;ctx.fillText(label,x+6,ty+16)})}
 var clab=null;
 if(document.getElementById("court-on").checked){
  ctx.save();ctx.lineJoin="round";ctx.lineCap="round";
  if(D.court){var cr=courtAt(frame);
   if(cr){ctx.beginPath();for(var q=0;q<4;q++){var cx2=cr[2*q]*sx,cy2=cr[2*q+1]*sy;if(q)ctx.lineTo(cx2,cy2);else ctx.moveTo(cx2,cy2)}ctx.closePath();
    ctx.lineWidth=8;ctx.strokeStyle="rgba(0,0,0,.55)";ctx.stroke();ctx.lineWidth=4.5;ctx.strokeStyle="#E02020";ctx.stroke()}}
  else if(D.court_est){var first=null;["1","2"].forEach(function(kk){var ln=lineAt(Number(kk),frame);if(!ln)return;
    ctx.beginPath();ctx.moveTo(ln[0]*sx,ln[1]*sy);ctx.lineTo(ln[2]*sx,ln[3]*sy);ctx.lineWidth=8;ctx.strokeStyle="rgba(0,0,0,.55)";ctx.stroke();ctx.lineWidth=4.5;ctx.strokeStyle="#E02020";ctx.stroke();if(!first)first=ln});
   if(first)clab=first}
  ctx.restore()}
 if(D.ball&&document.getElementById("ball-on").checked){var hs=null;
  for(var si=0;si<D.ball.segs.length;si++){var sg=D.ball.segs[si];if(frame>=sg[0]&&frame<=sg[1]){hs=sg;break}}
  if(hs&&LP[hs[2]]&&isOut(LP[hs[2]]))hs=null;
  if(hs){var hb=boxAt(hs[2],frame);if(hb){var fx=(hb[1]+hb[3]/2)*sx,fy=(hb[2]+hb[4])*sy,rx=Math.max(16,hb[3]*sx*0.75),ry=rx*0.32;
   ctx.beginPath();ctx.ellipse(fx,fy,rx,ry,0,0,6.2832);ctx.fillStyle="rgba(224,32,32,.28)";ctx.fill();
   ctx.lineWidth=8;ctx.strokeStyle="#fff";ctx.stroke();
   ctx.lineWidth=4;ctx.strokeStyle="#E02020";ctx.stroke()}}}
 drawFeatures(L,t,frame,L.cw/D.size[0],L.ch/D.size[1]);
 var sp=updateComments(L,t,frame);
 if(D.every&&es.length&&document.getElementById("tags-on").checked)drawTags(L,es,sx,sy,sp);else statLast={framed:es.length,tagged:0,compact:0,skipped:0,visible:es.length};
 if(clab){ctx.save();ctx.font="bold 12px sans-serif";ctx.textBaseline="alphabetic";var lt="コートの枠（推定）",lw=ctx.measureText(lt).width+10,lx=Math.max(2,Math.min(clab[0]*sx+6,L.cw-lw-2)),ly=Math.max(2,Math.min(clab[1]*sy+8,L.ch-20));
  ctx.fillStyle="#E02020";ctx.fillRect(lx,ly,lw,17);ctx.strokeStyle="#fff";ctx.lineWidth=1.5;ctx.strokeRect(lx,ly,lw,17);ctx.fillStyle="#fff";ctx.fillText(lt,lx+5,ly+13);ctx.restore()}
 var key=am&&p?(p.id+"|"+am.id+"|"+more):("none|"+(p?p.id:"")+"|"+more);
 if(key!==lastKey){lastKey=key;drawCard(p,am);
  [].forEach.call(document.querySelectorAll(".mom"),function(e){e.classList.toggle("act",!!am&&e.dataset.key===p.id+"|"+am.id)})}
 var nk=Math.round(t*2)+"|"+(p?p.id:"");if(nk!==nowKey){nowKey=nk;optState();[].forEach.call(tlinner.querySelectorAll(".m"),function(e){e.classList.toggle("now",Math.abs(Number(e.dataset.t)-t)<=1.5)})}
 headEl.style.left=(100*t/dur())+"%";
 if(zoom>1&&!v.paused)keepVisible(t)}
function loop(){if(!v.paused||settle>0){if(settle>0)settle--;render()}requestAnimationFrame(loop)}
function checkVideo(){var w=v.videoWidth,h=v.videoHeight,d=v.duration,bad=false;   // a different video put next to this page: the boxes would be on the wrong picture
 if(w&&h&&Math.abs((w/h)/(D.size[0]/D.size[1])-1)>0.03)bad=true;
 if(isFinite(d)&&D.video_s&&d<D.video_s-1.5)bad=true;
 document.getElementById("vmix").classList.toggle("on",bad)}
v.addEventListener("loadedmetadata",checkVideo);
["loadedmetadata","durationchange"].forEach(function(n){v.addEventListener(n,function(){buildTl();draw()})});
["timeupdate","seeked","pause","play","resize"].forEach(function(n){v.addEventListener(n,draw)});
window.addEventListener("resize",function(){ticks();draw()});

// --- three first-time hints (remembered where the browser allows it)
function tipsSeen(){try{return window.localStorage.getItem("rv_tips_done")==="1"}catch(e){return false}}
function tipsDone(){try{window.localStorage.setItem("rv_tips_done","1")}catch(e){}}
function hints(){if(tipsSeen())return;var h=document.getElementById("hints");h.classList.add("on");   // one strip above the picture (in the page flow: it never covers the picture)
 function close(){tipsDone();h.classList.remove("on")}
 v.addEventListener("play",close,{once:true});document.getElementById("hintx").addEventListener("click",close)}
buildLegend();buildChips();buildEval();buildTl();buildList();draw();requestAnimationFrame(loop);hints();
window.__review={stats:function(){return statLast},seek:seek,select:select,step:step,stepEvent:stepEvent,setZoom:setZoom,state:function(){return {sel:sel,t:v.currentTime,act:lastKey,rate:v.playbackRate,boxes:document.getElementById("boxes").checked,allbox:document.getElementById("allbox").checked,zoom:zoom,comments:Object.keys(cmEls).length,all:allRows,more:more}}};
})();
</script>
</body>
</html>
"""
