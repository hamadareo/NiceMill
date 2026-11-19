"""Synthetic works for the PDF tests (and the sample PDFs): generated videos with coloured
blocks as people, a MOT boxes.txt, engine rows next to overlay frames, a made-up court
homography, invented names and texts. No real video, frame or feedback is ever read."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from pfb_app.domain.models import Kind, Stage, StageRun
from pfb_app.pipeline.pfb3x3_bridge import format_game_clock
from pfb_app.preprocess.frames import format_time
from pfb_app.services import review, short_text
from pfb_app.services.review import confirm_identity, save_tracked_identity
from pfb_app.storage import files

FPS = 10
W, H = 640, 360
NAMES = ["山田花子", "鈴木一郎", "#7", "#10", "佐藤みなみ", "田中大地", "高橋さくら", "伊藤蓮",
         "渡辺結衣", "中村陽太", "小林葵", "加藤悠真", "吉田凛", "山本湊", "松本楓"]
SPORTS_TYPES = ["pass", "shot_or_score", "defense", "transition", "communication",
                "effort_recovery", "positioning", "assist"]
GENERAL_TYPES = ["speak_or_discuss", "hand_over", "build_or_assemble", "observe",
                 "adjust_or_fix", "point_or_gesture"]
SAYING = {"pass": "本人が別の参加者に素早くパスを出した",
          "shot_or_score": "本人がゴール下でシュートを決めた",
          "defense": "本人が相手の前に立って守った",
          "transition": "本人がボールを取った後すぐに前へ走った",
          "communication": "本人が別の参加者に声をかけて位置を伝えた",
          "effort_recovery": "本人が転がったボールを最後まで追いかけた",
          "positioning": "本人が空いた場所に動いてボールを待った",
          "assist": "本人が別の参加者の得点につながるパスを出した",
          "speak_or_discuss": "本人が自分の考えをみんなに話した",
          "hand_over": "本人が別の参加者に紙を手わたした",
          "build_or_assemble": "本人が紙を折って柱を作った",
          "observe": "本人が別の参加者の説明をよく見て聞いた",
          "adjust_or_fix": "本人がかたむいた塔をまっすぐに直した",
          "point_or_gesture": "本人が指さしで置く場所を伝えた"}
TEXT = ("{name}さんは、まわりをよく見て動いていました。\n"
        "とくに、仲間が困っているときにすぐに近づいて手をかしていたのがよかったです。"
        "次の{work}でも、その動きを続けてください。")


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def box_of(k: int, frame: int) -> tuple[int, int, int, int]:
    """Person k's box in 1-based `frame`: a block walking left and right in its own lane."""
    lane = 70 + (k % 4) * 60
    span = W - 120
    base = 30 + (k * 97) % span
    x = base + int(40 * np.sin(frame / (12.0 + k)))
    return max(0, min(W - 34, x)), lane, 30, 72


def off_screen(k: int, frame: int, n_frames: int) -> bool:
    """Each person leaves the picture once, for a while."""
    start = (k * 37) % max(n_frames - 60, 1) + 20
    return start <= frame < start + 40 + (k % 3) * 20


def make_video(path: Path, people: int, seconds: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    n = seconds * FPS
    for f in range(1, n + 1):
        img = np.full((H, W, 3), (70, 92, 60), dtype=np.uint8)      # a dull green field
        cv2.rectangle(img, (20, 40), (W - 20, H - 20), (210, 210, 210), 2)
        for k in range(people):
            if off_screen(k, f, n):
                continue
            x, y, bw, bh = box_of(k, f)
            hue = (k * 37) % 180
            colour = cv2.cvtColor(np.uint8([[[hue, 160, 230]]]), cv2.COLOR_HSV2BGR)[0, 0]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), tuple(int(c) for c in colour), -1)
            cv2.circle(img, (x + bw // 2, y - 8), 8, (190, 200, 230), -1)
        w.write(img)
    w.release()
    return path


def write_tracks(track_dir: Path, people: int, seconds: int, geometry: bool) -> Path:
    track_dir.mkdir(parents=True, exist_ok=True)
    n = seconds * FPS
    rows = []
    for f in range(1, n + 1):
        for k in range(people):
            if not off_screen(k, f, n):
                x, y, bw, bh = box_of(k, f)
                rows.append(f"{f},{k + 1},{x},{y},{bw},{bh},0.9")
    (track_dir / "boxes.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (track_dir / "meta.json").write_text(json.dumps(
        {"fps": FPS, "n_frames": n, "width": W, "height": H}), encoding="utf-8")
    if geometry:   # image px -> court cm: the whole frame is the half court
        (track_dir / "linecall").mkdir(exist_ok=True)
        (track_dir / "linecall" / "geometry.json").write_text(json.dumps(
            {"H": [[1505 / W, 0, 0], [0, 1105 / H, 0], [0, 0, 1]], "lens": None,
             "source": "autocourt"}), encoding="utf-8")
    return track_dir / "boxes.txt"


def work(repo, *, domain="sports", sport="basketball", title="練習試合", kind=Kind.video) -> int:
    return repo.create_assignment(title, kind, "3人ずつで試合をする。" if domain == "sports"
                                  else "紙で塔を高く作る。", "", "standard", True,
                                  domain=domain, sport=sport if domain == "sports" else None)


def tiny_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((24, 32, 3), 128, dtype=np.uint8))
    return path


def video_group(settings, repo, aid, *, people=4, seconds=60, geometry=True, teams=False,
                short=(), unapproved=(), events_each=4, clock_offset_min=None,
                label="A", cameras=("cam1",), start_sec=0.0) -> int:
    """A tracked video group with `people` evaluated people, each with `events_each` positive
    moments (all cited, all seen by cam1); people in `short` get a short text, those in
    `unapproved` stay waiting. Every camera films the same scene (the same boxes). The
    analysed part starts at `start_sec`. Returns the submission id."""
    assignment = repo.get_assignment(aid)
    sports = assignment["domain"] == "sports"
    video = make_video(settings.inbox_dir / f"{label}_cam1.mp4", people, seconds)
    meta = {"cameras": {c: str(video) for c in cameras}, "start_sec": float(start_sec),
            "end_sec": None, "auto_transcribe": False, "engine": "pfb3x3"}
    if clock_offset_min is not None:
        meta["clock_offset_min"] = clock_offset_min
    sid = repo.create_submission(aid, label, meta)
    pids = [f"P{chr(97 + k)}" for k in range(people)]
    names = {pid: NAMES[k % len(NAMES)] + ("" if k < len(NAMES) else str(k)) for k, pid in
             enumerate(pids)}
    for pid in pids:
        repo.add_subject(sid, pid, names[pid], "外見", {})
    sub_dir = files.submission_dir(settings, aid, sid)
    labels_ = [f"P{k + 1}" for k in range(people)]
    tracks = {}
    for cam in cameras:
        vid = f"s{sid}_{cam}"
        boxes = write_tracks(sub_dir / "pfb3x3" / "tracks" / vid, people, seconds, geometry)
        prep = save(repo, sid, Stage.prepare, {"frames": []}, camera=cam)
        tracks[cam] = (prep, save(repo, sid, Stage.track, {
            "people": people, "labels": labels_,
            "sheet": str(tiny_png(sub_dir / f"sheet_{cam}.png")),
            "default_names": {}, "default_source": None, "boxes": str(boxes), "video_id": vid,
            "analysed_s": float(seconds - start_sec)}, camera=cam, upstream=[prep.id]))
        if teams:
            save(repo, sid, Stage.names, {"track_run_id": tracks[cam][1].id, "teams": {
                lb: ("T1" if k % 2 == 0 else "T2") for k, lb in enumerate(labels_)}},
                 camera=cam, upstream=[tracks[cam][1].id])
        save_tracked_identity(repo, sid, cam, dict(zip(labels_, pids)))
    confirm_identity(repo, sid)
    repo.conn.execute("UPDATE jobs SET status='done' WHERE submission_id=?", (sid,))
    prep, track = tracks["cam1"]
    vid = f"s{sid}_cam1"
    # engine rows next to overlay frames, as the engine writes them
    exp = sub_dir / "pfb3x3" / "exp" / "qwen3vl-8b" / "clip_som" / vid
    types = SPORTS_TYPES if sports else GENERAL_TYPES
    rows, events, frames, source_text = [], [], {}, {}
    w = 0
    length = seconds - start_sec
    for k, pid in enumerate(pids):
        for j in range(events_each):
            w += 1
            t = 4 + (k * 3 + j * (length - 10) / max(events_each, 1)) % (length - 8)
            while off_screen(k, round((start_sec + t) * FPS) + 1, seconds * FPS):
                t = (t + 3) % (length - 2)
            kind = types[(k + j) % len(types)]
            action = f"P{k + 1} does {kind} near P{(k + 1) % people + 1}"
            wid = f"{vid}_w{w:04d}"
            rows.append({"window_id": wid, "window_index": w, "status": "ok", "events": [
                {"label": f"P{k + 1}", "clock": f"{int(t) // 60}:{t % 60:04.1f}",
                 "observed_action": action, "action_type": kind, "confidence": "high"}]})
            eid = f"cam1-{w:04d}-01"
            shot = tiny_png(exp / "overlays" / wid / "1.jpg")
            frames[eid] = [str(shot)]
            source_text[eid] = action
            events.append({
                "event_id": eid, "camera": "cam1", "participant": pid,
                "video_time": format_time(start_sec + t),
                "task_elapsed": (format_game_clock(60 * clock_offset_min + start_sec + t)
                                 if sports and clock_offset_min is not None else format_time(t)),
                "observed_behavior": SAYING[kind], "utterance": "", "speaker": "unknown",
                "evidence_source": "video", "confidence": "high"})
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "behavior.engine.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    ident = repo.latest_run(sid, Stage.identity, "cam1")
    beh = save(repo, sid, Stage.behavior, {"events": events, "frames": frames,
                                           "source_text": source_text},
               camera="cam1", upstream=[ident.id, prep.id, track.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [
        {"ref_id": e["event_id"], "contribution_interpretation": "仲間の動きを助けた",
         "is_positive_contribution": True, "rubric_criteria": []} for e in events]},
        camera="cam1", upstream=[beh.id])
    word = "試合" if sports else "活動"
    for s in repo.list_subjects(sid):
        pid = s["participant_id"]
        mine = [e["event_id"] for e in events if e["participant"] == pid]
        if pid in short:
            run = save(repo, sid, Stage.pfb, {"pfb_status": "insufficient_positive_evidence",
                                              "pfb": None, "evidence_refs": []},
                       subject_id=s["id"], upstream=[con.id])
            offered = short_text.offer(repo, repo.get_subject(s["id"]), run)
            short_text.approve_short(repo, s["id"], run.id, offered)
            continue
        save(repo, sid, Stage.pfb, {"pfb_status": "generated",
                                    "pfb": TEXT.format(name=s["display_name"], work=word),
                                    "evidence_refs": mine[:4]}, subject_id=s["id"],
             upstream=[con.id])
        if pid not in unapproved:
            review.approve(repo, s["id"])
    return sid


def document_work(repo) -> int:
    aid = repo.create_assignment("防災マップ", Kind.document, "地域の防災を調べてまとめる", "",
                                 "standard", False)
    sid = repo.create_submission(aid, "A", {"source_path": "/x.pdf", "original_filename": "a.pdf"})
    subj = repo.add_subject(sid, None, "山田花子", None, {})
    prep = save(repo, sid, Stage.prepare, {"pages": [], "pdf_path": "/x.pdf"})
    items = [{"page": 3, "quote": "避難経路を家族で確認する", "observed_content": "c",
              "evidence_source": "text", "confidence": "high", "item_id": "p3-001",
              "quote_verified": True},
             {"page": 4, "quote": "", "observed_content": "避難所の地図を手で描いた",
              "evidence_source": "figure", "confidence": "high", "item_id": "p4-001",
              "quote_verified": False}]
    ev = save(repo, sid, Stage.evidence, {"items": items}, upstream=[prep.id])
    con = save(repo, sid, Stage.contribution, {"judgments": [
        {"ref_id": i["item_id"], "contribution_interpretation": "調べた内容を自分の言葉にした",
         "is_positive_contribution": True, "rubric_criteria": []} for i in items]},
        upstream=[ev.id])
    save(repo, sid, Stage.pfb, {"pfb_status": "generated",
                                "pfb": "避難の道を家族と確かめる大切さを、自分の言葉でまとめていました。",
                                "evidence_refs": ["p3-001", "p4-001"]}, subject_id=subj,
         upstream=[con.id])
    review.approve(repo, subj)
    return aid
