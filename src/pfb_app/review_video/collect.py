"""The app's database -> a Package (what the review outputs are made from).

Read only: the same sheets the handout uses (export.sheets.group_sheets: the person's
positive moments, which of them the text cites, their time and sentences), the playable file
(services.explainer._source: a small non-HEVC original or the ready proxy, never a 4K/HEVC
original) and the tracks. The text may still be a draft: nothing here needs an approval,
because these videos are for the evaluator to read before handing anything out.

A moment is `suspect` when the preliminary identity check (the names stage's number timeline,
sheets.foreign) read another number on the person's box at that time. No timeline, no check,
no flag.
"""
from __future__ import annotations

import logging

from pfb_app.config import Settings
from pfb_app.domain.models import Stage
from pfb_app.export import explainer_video as ev
from pfb_app.export import sheets
from pfb_app.export import labels
from pfb_app.pipeline.pfb3x3_bridge import clock_seconds
from pfb_app.review_video import labelling
from pfb_app.review_video.model import (CHANGE_NOTE, KIND_CHANGE, KIND_EVENT, NONPLAYER_WHY_CHECK,
                                        NONPLAYER_WHY_IMAGE, NONPLAYER_WHY_OTHER, NO_CUES_NOTE, Moment, Package, Person)
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)


def _draft_text(repo: Repo, sid: int, subject_id: int) -> str | None:
    run = repo.latest_run(sid, Stage.pfb, subject_id=subject_id)
    text = ((run.output or {}).get("pfb") if run else None)
    return text if isinstance(text, str) and text.strip() else None


def build_package(repo: Repo, settings: Settings, submission_id: int, *,
                  subject_ids: list[int] | None = None,
                  credit_lines: list[str] | None = None,
                  ball_file: str | None = None,
                  claim_cues: str | None = None,
                  eval_texts: str | None = None,
                  pose_file: str | None = None,
                  estimate_court: bool = True,
                  court_reg: str | None = None) -> Package:
    from pfb_app.services import explainer   # noqa: PLC0415 - imports the pipeline

    submission = repo.get_submission(submission_id)
    work = repo.get_assignment(submission["assignment_id"])
    sub, found = sheets.group_sheets(repo, work["id"], submission_id)
    sources: dict[str, ev.Source] = {}
    extras: dict = {}
    missing: dict[str, str] = {}
    everyone: list[tuple[Person, bool]] = []
    gates: list[bool] = []
    notes: list[str] = []
    for sheet in found:
        selected = subject_ids is None or sheet.subject_id in subject_ids
        subject = repo.get_subject(sheet.subject_id)
        pid = subject.get("participant_id")
        person = Person(sheet.subject_id, sheet.name, {},
                        "approved" if sheet.approved else "draft",
                        sheet.text or _draft_text(repo, submission_id, sheet.subject_id))
        if pid:
            for c in sub.cameras():
                if c.labels_of(pid):
                    person.labels[c.name] = c.labels_of(pid)
        # every evaluated person is read for a number, selected or not: a duplicate or a
        # conflict is judged among all of them, then the selection is made
        _numbers(repo, sub, person, pid, gates)
        everyone.append((person, selected))
        if not selected:
            continue
        untimed = 0
        for m in sheet.moments:
            if m.t_s is None or m._source is None:
                untimed += 1
                continue
            cam = sub.camera(m._source.camera, sheets._by_stage(repo, m._source, Stage.track),
                             sheets._by_stage(repo, m._source, Stage.identity))
            if cam is None:
                untimed += 1
                continue
            if cam.name not in sources and cam.name not in missing:
                try:
                    src = explainer._source(settings, sub, cam)
                except ev.RenderError as e:
                    src, missing[cam.name] = None, str(e)
                if src is not None:
                    sources[cam.name] = src
                    extras[cam.name] = _extras(cam)
                elif cam.name not in missing:
                    missing[cam.name] = "動画か追跡の記録が見つかりません。"
            if cam.name not in sources:
                untimed += 1
                continue
            person.labels.setdefault(cam.name, cam.labels_of(pid) if pid else [])
            person.moments.append(Moment(
                m.ref_id, cam.name, sub.start_s + m.t_s, m.when, m.what, m.why,
                m.category.label if m.category else None, m.cited,
                suspect=m.ref_id in sub.foreign(m._source)))
        good_ids = {m.ref_id for m in person.moments}
        _events(repo, sub, work, sheet, subject, person, good_ids, sources, missing)
        if untimed:
            notes.append(f"{person.name}: 時刻や動画のない場面 {untimed} 件は入れていません。")
    verified = bool(gates) and all(gates)
    ordered = [p for p, _ in everyone]
    labelling.assign(ordered, verified)
    chosen = {id(p) for p, sel in everyone if sel}
    people = [p for p in ordered if id(p) in chosen and (p.moments or p.events)]
    _apart(people, sources)
    holders, nball = _holders(ball_file, people, sources)
    from pfb_app.review_video import claims   # noqa: PLC0415
    # the action check always runs: without cues the good moments stay as they were (unchecked), but
    # an unchecked action sentence of the other recorded actions is never stated as a fact
    r = claims.apply(people, claims.load_cues(claim_cues) if claim_cues else {})
    if claim_cues:
        notes.append("動作の確認: %d 場面を確認、%d 場面の文を断定しない形にした。" % (r["checked"], r["weakened"]))
    else:
        notes.append(NO_CUES_NOTE)
    if eval_texts:
        e = claims.apply_eval_texts(people, claims.load_cues(eval_texts))
        notes.append("評価文: AIの文 %d 人、短い文 %d 人、付かなかった %d 人。確かめられない動作・結果の文を断定しない形にした／外した: %d 件。"
                     % (e["ai"], e["short"], len(e["none"]), e["hedged"]))
    known = {sheet.subject_id for sheet in found}
    for sid in subject_ids or ():
        if sid not in known:
            notes.append(f"指定された人（ID {sid}）は、この解析の評価対象にいません。")
    court = _court(sub, sources)
    if court is None and court_reg:
        court = _court_registered(court_reg, sources, notes)
    court_est = bool(court is None and estimate_court)
    for cam, why in missing.items():
        notes.append(f"{cam}: {why}")
    return Package(work["title"], sheets.local_date(work.get("created_at")),
                   f"ほかの{labels.words_for(work).people}", people, sources,
                   list(credit_lines or []), notes, verified, holders, extras, court, nball,
                   court_est=court_est, pose_path=pose_file)


def _events(repo, sub, work, sheet, subject, person: Person, good_ids: set[str], sources,
            missing) -> None:
    """The person's other recorded actions (the behavior run's events that are not a counted
    よかった所): when, what, and the judgment's words if there is one."""
    from pfb_app.services import review   # noqa: PLC0415
    pid = subject.get("participant_id")
    run = (repo.get_run(subject["approved_run_id"]) if sheet.approved
           else repo.latest_run(sub.sid, Stage.pfb, subject_id=sheet.subject_id))
    if run is None or not pid:
        return
    for con, source in sheets._chain(repo, sub, run):
        out = source.output or {}
        events = out.get("events") or []
        if not events:
            continue
        cam = sub.camera(source.camera, sheets._by_stage(repo, source, Stage.track),
                         sheets._by_stage(repo, source, Stage.identity))
        if cam is None or cam.name not in sources:
            continue
        types = sub.action_types(source, cam)
        judged = {j.get("ref_id"): j for j in (con.output or {}).get("judgments") or []}
        foreign = sub.foreign(source)
        person.labels.setdefault(cam.name, cam.labels_of(pid))
        for e in events:
            ref = e.get("event_id")
            if e.get("participant") != pid or ref in good_ids:
                continue
            j = judged.get(ref) or {}
            if j.get("is_positive_contribution"):
                continue                       # a counted moment without a time: not here
            kind, clock = types.get(ref, (None, None))
            t = clock if clock is not None else (
                None if clock_seconds(e.get("video_time")) is None
                else clock_seconds(e.get("video_time")) - sub.start_s)
            if t is None:
                continue
            try:
                cat = sheets._category(work, kind, j)
            except Exception:   # noqa: BLE001 - a category is optional
                cat = None
            person.events.append(Moment(
                ref, cam.name, sub.start_s + t,
                sheets.when_text(t, sub.clock_offset_s, e.get("video_time"), None),
                e.get("observed_behavior") or "", j.get("contribution_interpretation") or "",
                cat.label if cat else None, False, suspect=ref in foreign, kind=KIND_EVENT))


def _numbers(repo, sub, person: Person, pid, gates: list[bool]) -> None:
    """The person's read jersey number, team and tracker id, and the places where the number
    read on their track changes (a mark: 「人物が別人かも」). Nothing is guessed: the number is
    the one the app names people by (services.auto_people._read_number)."""
    from pfb_app.services import auto_people   # noqa: PLC0415
    numbers: dict[str, int] = {}
    meta = (repo.get_submission(sub.sid).get("meta") or {}).get("participation") or {}
    teams: dict[str, int] = {}
    others = labelled = unsure = 0   # tracks in neither team / with a team verdict / marked 要確認
    first = None
    for cam_name, labs in person.labels.items():
        cam = sub.camera(cam_name)
        if cam is None or cam.track is None or not labs:
            continue
        run = auto_people._number_run(repo, sub.sid, cam_name, cam.track)
        reads = ((run.output or {}).get("cues") or {}).get("number") if run else None
        if run is not None:      # the precision gate the engine applied to the number cue
            gates.append(bool((((run.output or {}).get("gate") or {}).get("number") or {})
                              .get("enabled")))
        fps = cam.fps or 0
        for lb in labs:
            first = first or lb
            if cam.teams.get(lb) in model_teams():
                teams[cam.teams[lb]] = teams.get(cam.teams[lb], 0) + 1
            if cam.teams.get(lb):
                labelled += 1
                others += cam.teams[lb] == "other"
            if lb in ((meta.get(cam_name) or {}).get("check") or {}):
                unsure += 1
            read = (reads or {}).get(lb)
            n = auto_people._read_number(read)
            if n is not None:
                numbers[n] = numbers.get(n, 0) + 1
            if isinstance(read, dict) and read.get("segments"):
                person.checked = True          # a number timeline: the identity check can run
            if fps and isinstance(read, dict):
                segs = []
                for seg in read.get("segments") or []:
                    try:
                        segs.append((int(seg[0]), int(seg[1]), str(seg[2])))
                    except (TypeError, ValueError, IndexError):
                        continue
                segs.sort()
                prev = None
                for f0, _f1, num in segs:
                    if num in ("None", ""):
                        continue
                    if prev is not None and prev != num:
                        t = (f0 - 1) / fps
                        person.events.append(Moment(
                            f"chg-{lb}-{f0}", cam_name, t, "", "", "", None, False, suspect=True,
                            kind=KIND_CHANGE, note=CHANGE_NOTE.format(a=prev, b=num),
                            nums=(prev, num)))
                    prev = num
    person.internal = (first or "") + ("ほか" if sum(len(v) for v in person.labels.values()) > 1 else "")
    if len(numbers) == 1:
        person.number = next(iter(numbers))
    elif len(numbers) > 1:
        person.conflict = True          # different numbers on one person's tracks: unknown
    if teams:
        person.team = max(teams, key=teams.get)
    # only the app's own cues, and only as a hint (nobody is removed on it): the colour cue put
    # every track in neither team ("other": officials, onlookers), or the participation rule was
    # unsure that the person plays on the court (要確認). A person with a team colour never gets it.
    if not teams and labelled and others == labelled:
        person.nonplayer = NONPLAYER_WHY_OTHER
    elif not teams and unsure:
        person.nonplayer = NONPLAYER_WHY_CHECK


def _extras(cam):
    """Unnamed detections of one camera (read only; empty when the files are missing)."""
    from pathlib import Path   # noqa: PLC0415
    from pfb_app.export.extra_boxes import load_extra_boxes   # noqa: PLC0415
    out = (cam.track.output or {}) if getattr(cam, "track", None) else {}
    path = out.get("boxes")
    return load_extra_boxes(Path(path).parent) if path else None


def _court(sub, sources):
    """The registered court (the track stage's / line-call geometry: image px -> court cm) as an
    outline in the picture, sampled every 0.5 s. Only when the submission has one and the camera
    has no lens correction to undo; nothing is estimated here."""
    import numpy as np   # noqa: PLC0415
    from pfb_app.export.tracks import COURT_H, COURT_W   # noqa: PLC0415
    for c in sub.cameras():
        g = getattr(c, "geometry", None)
        src = sources.get(c.name)
        if g is None or src is None or g.lens:
            continue
        corners = np.array([[0, 0, 1], [COURT_W, 0, 1], [COURT_W, COURT_H, 1], [0, COURT_H, 1]], float).T
        step = max(1, int(round((src.fps or 30) / 2)))
        last = int(src.last_frame or 0)
        rows = []
        for f in range(1, max(last, 2) + 1, step):
            H = g.H_frames[f - 1] if g.H_frames is not None and f - 1 < len(g.H_frames) else g.H
            try:
                p = np.linalg.inv(np.asarray(H, float)) @ corners
            except np.linalg.LinAlgError:
                continue
            if not np.all(np.isfinite(p)) or np.any(np.abs(p[2]) < 1e-9):
                continue
            xy = (p[:2] / p[2]).T
            rows.append([f] + [round(float(v), 1) for v in xy.reshape(-1)])
        if rows:
            return {"frames": rows, "note": "登録"}
    return None


def _court_registered(path, sources, notes):
    """A court registered by a person on one picture (court_reg.py): the outline followed over the video.
    None (and a note) when the file or the camera movement cannot be used: no line is drawn then."""
    from pfb_app.review_video import court_reg   # noqa: PLC0415
    try:
        reg = court_reg.load_registration(path)
    except (OSError, ValueError) as e:
        notes.append(f"コートの登録ファイルを使えません（{e}）。コートの線は出していません。")
        return None
    for cam, src in sources.items():
        try:
            res = court_reg.register(src.video, src.fps or 30.0, src.last_frame, reg)
        except Exception:   # noqa: BLE001 - a picture that cannot be followed: no line
            log.exception("court registration failed")
            res = None
        if res:
            st = res["stats"]
            notes.append("コートの登録: %d 時刻のうち %d 時刻で線を出しています（追跡が確かでない時刻は出していません）。" % (st["samples"], st["rows"]))
            return res
    notes.append("コートの登録: カメラの動きを追えなかったため、線は出していません。")
    return None


def _holders(ball_file, people, sources):
    """The ball holder segments (an estimate) from an external ball-detection tool (not included) file: ([[f0, f1, label]], n)."""
    if not ball_file or not people:
        return [], 0
    import json   # noqa: PLC0415
    from pfb_app.review_video import holder   # noqa: PLC0415
    try:
        data = json.load(open(ball_file, encoding="utf8"))
    except (OSError, ValueError):
        log.warning("ball file unreadable: %s", ball_file)
        return [], 0
    cam = next((c for c in sources if any(c in p.labels for p in people)), None)
    if cam is None:
        return [], 0
    src = sources[cam]
    labels = {lb for p in people if not p.excluded for lb in p.labels.get(cam, [])}
    cache = {lb: src.boxes.boxes(lb) for lb in labels}

    def boxes_at(f):
        out = {}
        for lb, (frames, boxes) in cache.items():
            i = int(frames.searchsorted(f)) if hasattr(frames, "searchsorted") else None
            if i is not None and i < len(frames) and int(frames[i]) == f:
                out[lb] = tuple(float(v) for v in boxes[i][:4])
        return out
    ball = {int(k): v for k, v in (data.get("ball") or {}).items()}
    return holder.segments(ball, boxes_at), len(ball)


def _apart(people, sources) -> None:
    """Colour groups from the picture (nonplayer.py) -> each person's team letter, and the
    probable non-players (referee, coach...): a HIGH score sets the person apart, a MID score
    only marks them. The app's own T1/T2 verdict (names stage) wins over the picture."""
    from pfb_app.review_video import nonplayer   # noqa: PLC0415
    try:
        res = nonplayer.assess(people, sources)
    except Exception:   # noqa: BLE001 - a picture that cannot be read: no judgement, nobody is touched
        log.exception("non-player estimate failed")
        res = {}
    for p in people:
        r = res.get(p.subject_id) or {}
        if p.team in ("T1", "T2"):
            p.team_letter = {"T1": "A", "T2": "B"}[p.team]
            continue                  # a team colour from the names stage: a player
        p.team_letter = r.get("team")
        if r:
            p.np_score = r["score"]
        if r.get("level") == "high":
            p.excluded = True
            p.nonplayer = p.nonplayer or NONPLAYER_WHY_IMAGE
        elif r.get("level") == "mid" and not p.nonplayer:
            p.nonplayer = NONPLAYER_WHY_IMAGE


def model_teams():
    from pfb_app.review_video.model import TEAM_LETTERS   # noqa: PLC0415
    return TEAM_LETTERS
