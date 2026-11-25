"""Re-check the action words of the scenes a feedback text cites, and keep the result (claim_checks).

After a new feedback text of a work whose automatic approval is on (`runner._claim_scan`), every
scene the text cites is read for the actions its recorded sentence says:

* movement words (走る・歩く・移動する・位置につく): `motion_check`, from the tracks. No model.
* shoot / dribble / pass / defend: `claim_check`: a pose measurement of the person around the scene
  (engine venv, an external pose tool (not included)) and two questions to a LOCAL vision model about the
  eight frames around the scene (the person's box drawn). Off unless `claim_check_enabled`.

The verdicts go to the table `claim_checks` (one row per scene and action). Nothing here changes a
text or an approval: `services.autoapprove` reads the rows (an action that is not confirmed keeps
the text for a person). A scene that cannot be checked (no video, no model, an error) gets no row,
which is the same as "not confirmed".
"""
from __future__ import annotations

import base64
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from pfb_app.config import Settings
from pfb_app.domain.models import PfbOutput, Stage, StageRun
from pfb_app.pipeline import claim_check as cc
from pfb_app.pipeline import motion_check as mc
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

METHOD_CLAIM = "claim"
METHOD_MOTION = "motion"
CLAIM_ACTIONS = ("shoot", "dribble", "pass", "defend")   # asked of the model; "run" is the tracks'
TILES = tuple(range(-4, 4))          # the picture sheet: one frame a second, -4 s .. +3 s
CUE_TIMES = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
TILE_W, TILE_H = 560, 315
_PROBE = {"shoot": "シュート", "dribble": "ドリブル", "pass": "パス", "defend": "ブロック"}


@dataclass
class Scene:
    event_id: str
    behavior_run_id: int
    text: str                         # the recorded sentence of the scene
    actions: set[str]
    camera: str
    at_s: float                       # seconds in the playable video
    labels: list[str]                 # the person's track labels on that camera
    source: object                    # export.explainer_video.Source


def key(action: str, method: str) -> str:
    return f"{method}:{action}"


def scenes_of(repo: Repo, settings: Settings, run: StageRun) -> list[Scene]:
    """The cited scenes of a feedback run that state an action, with what is needed to look at
    them (the video, the person's tracks). Scenes without a time or a video are left out."""
    from pfb_app.review_video import claims, collect   # noqa: PLC0415 - they import services
    from pfb_app.services import autoapprove   # noqa: PLC0415

    ctx = autoapprove._context(repo, run)   # noqa: SLF001
    pkg = collect.build_package(repo, settings, run.submission_id, subject_ids=[run.subject_id],
                                estimate_court=False)
    person = next((p for p in pkg.people if p.subject_id == run.subject_id), None)
    if person is None:
        return []
    moments = {m.ref_id: m for m in list(person.moments) + list(person.events)}
    out: list[Scene] = []
    for ref in ctx.refs:
        if ref not in ctx.events or ref not in moments:
            continue
        behavior, ev = ctx.events[ref]
        m = moments[ref]
        text = ev.get("observed_behavior") or m.what or ""
        acts = claims.sentence_actions(text) | mc.motion_actions(text)
        src = pkg.sources.get(m.camera or "")
        if not acts or src is None or m.t_s is None:
            continue
        out.append(Scene(ref, behavior.id, text, acts, m.camera, m.t_s,
                         list(person.labels.get(m.camera, [])), src))
    return out


# --- the movement check (tracks) -----------------------------------------------------------------

def motion_verdicts(scene: Scene) -> dict[str, dict]:
    want = scene.actions & set(mc.ACTIONS)
    if not want:
        return {}
    src = scene.source
    vf = round(scene.at_s * src.fps) + 1        # the tracks' frame number
    s = mc.stats_at(src.boxes, scene.labels, vf, src.fps)
    return {a: mc.verdict(a, s) for a in want}


# --- the picture sheet and the pose cue ----------------------------------------------------------

def _frames(scene: Scene, times: list[float]):
    """[(t, frame or None, box in the frame's pixels or None)] for the times around the scene."""
    from pfb_app.export import explainer_video as ev   # noqa: PLC0415
    src = scene.source
    reader = ev._Reader(src.video)   # noqa: SLF001
    out = []
    try:
        for dt in times:
            t = max(scene.at_s + dt, 0.0)
            frame = reader.at(t)
            box = None
            if frame is not None:
                fh, fw = frame.shape[:2]
                near = src.boxes.nearest(scene.labels, round(t * src.fps) + 1, 15)
                if near is not None:
                    b = near[1]
                    sx, sy = fw / src.size[0], fh / src.size[1]
                    box = [b.x * sx, b.y * sy, b.w * sx, b.h * sy]
            out.append((dt, frame, box))
    finally:
        reader.close()
    return out


def tiles(scene: Scene) -> list:
    """The eight crops (one frame a second) centred on the person, the person's box in yellow.
    [] when the video has no frame for one of them."""
    out, centre = [], None
    for dt, frame, box in _frames(scene, [float(d) for d in TILES]):
        if frame is None:
            return []
        frame = frame.copy()
        fh, fw = frame.shape[:2]
        if box is not None:
            x, y, w, h = (int(v) for v in box)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            centre = (x + w // 2, y + h // 2)
        cx, cy = centre if centre is not None else (fw // 2, fh // 2)
        cw, ch = min(TILE_W, fw), min(TILE_H, fh)
        x0, y0 = int(min(max(cx - cw // 2, 0), fw - cw)), int(min(max(cy - ch // 2, 0), fh - ch))
        c = frame[y0:y0 + ch, x0:x0 + cw].copy()
        cv2.putText(c, "%+d s" % dt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if dt == 0 else (255, 255, 255), 2)
        out.append(c)
    return out


def pose_cues(settings: Settings, scenes: list[Scene]) -> dict[str, cc.Cues]:
    """Pose cues of the scenes, measured by the engine venv's python in a child process. A failure
    gives {} (the shot cannot be confirmed without its arm cue)."""
    py = settings.tracker_python or settings.engine_python \
        or Path(settings.engine_dir) / ".venv" / "bin" / "python"
    weights = settings.claim_check_pose_weights or Path(settings.data_dir) / "models" / "yolo11x-pose.pt"
    # the pose script is not in the public repository (it imports AGPL-3.0 ultralytics); without
    # it no pose cue is measured ({} below), so a shot is never confirmed
    script = Path(__file__).resolve().parents[3] / "scripts" / "claim_cues_scene.py"
    if not scenes or not Path(py).exists() or not Path(weights).exists() or not script.exists():
        log.warning("姿勢の手がかりを測れません（実行環境か重みが見つかりません）")
        return {}
    with tempfile.TemporaryDirectory(prefix="claimcues_") as tmp:
        index: dict[str, list] = {}
        for sc in scenes:
            rows = []
            for i, (_dt, frame, box) in enumerate(_frames(sc, list(CUE_TIMES))):
                if frame is None or box is None:
                    continue
                f = Path(tmp) / f"{abs(hash(sc.event_id)) % 10**9}_{i}.jpg"
                cv2.imwrite(str(f), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                rows.append({"file": str(f), "box": box})
            index[sc.event_id] = rows
        (Path(tmp) / "index.json").write_text(json.dumps(index), encoding="utf8")
        try:
            subprocess.run([str(py), str(script), str(Path(tmp) / "index.json"),
                            str(Path(tmp) / "out.json"), str(weights)],
                           check=True, capture_output=True, timeout=1800)
            data = json.loads((Path(tmp) / "out.json").read_text(encoding="utf8"))
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            log.warning("姿勢の手がかりの測定に失敗しました: %s", e)
            return {}
    return {k: cc.Cues(v.get("wrist_above_head"), int(v.get("pose_frames") or 0))
            for k, v in data.items()}


# --- the question to the local vision model -------------------------------------------------------

def make_ask(settings: Settings):
    """`ask(images, prompt) -> reply text` against the local OpenAI-compatible server of the settings
    (inference.served: refuses a server that is not on this machine; the pictures are PFB data)."""
    from pfb_app.inference import served   # noqa: PLC0415
    backend = served.ServedChatBackend(settings.claim_check_vlm_server.rstrip("/"),
                                       settings.claim_check_vlm_model,
                                       timeout_sec=settings.claim_check_timeout_sec)

    def ask(images, prompt: str) -> str:
        content = []
        for im in images:
            ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                content.append({"type": "image_url", "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(buf).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": settings.claim_check_vlm_model, "temperature": 0, "max_tokens": 3000,
                "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": False}}
        return backend._post(body)["choices"][0]["message"]["content"] or ""   # noqa: SLF001
    return ask


# --- one feedback text ----------------------------------------------------------------------------

@dataclass
class Report:
    scenes: int = 0
    motion: int = 0
    asked: int = 0
    skipped: list[str] = field(default_factory=list)


def scan_run(repo: Repo, settings: Settings, run: StageRun, *, ask=None, cues=None,
             vlm: bool | None = None, force: bool = False) -> Report:
    """Re-check the cited scenes of a feedback run and keep the verdicts. `ask` (the model call) and
    `cues` (a function scenes -> {event id: Cues}) are injected by tests; by default they are the
    local server and the engine venv. `vlm` False: movement words only."""
    rep = Report()
    if run.stage != Stage.pfb or run.subject_id is None or not run.output:
        return rep
    PfbOutput.model_validate(run.output)
    scenes = scenes_of(repo, settings, run)[:max(1, settings.claim_check_max_scenes)]
    rep.scenes = len(scenes)
    use_vlm = settings.claim_check_enabled if vlm is None else vlm
    done: dict[int, dict] = {}
    todo: list[tuple[Scene, set[str]]] = []
    for sc in scenes:
        have = done.setdefault(sc.behavior_run_id, repo.claim_checks(sc.behavior_run_id))
        known = have.get(sc.event_id, {})
        for a, v in motion_verdicts(sc).items():
            if force or key(a, METHOD_MOTION) not in known:
                repo.put_claim_check(sc.behavior_run_id, sc.event_id, key(a, METHOD_MOTION), v)
                rep.motion += 1
        asked = {a for a in sc.actions if a in CLAIM_ACTIONS}
        if asked and use_vlm:
            need = {a for a in asked if force or key(a, METHOD_CLAIM) not in known}
            if need:
                todo.append((sc, need))
    if not todo:
        return rep
    ask = ask or make_ask(settings)
    measured = (cues or (lambda ss: pose_cues(settings, ss)))([sc for sc, _ in todo])
    for sc, need in todo:
        try:
            images = tiles(sc)
            if not images:
                rep.skipped.append(sc.event_id)
                continue
            for a in sorted(need):
                votes = cc.ask_votes(a, images, ask)
                cue = measured.get(sc.event_id)
                d = cc.check(_PROBE[a], cue, votes).as_dict()   # one action of the sentence at a time
                d.update({"action": a, "method": METHOD_CLAIM, "version": "claim/1",
                          "votes": {"pos": votes.pos, "ctl": votes.ctl} if votes else None,
                          "cues": ({"wrist_above_head": cue.wrist_above_head,
                                    "pose_frames": cue.pose_frames} if cue else None)})
                repo.put_claim_check(sc.behavior_run_id, sc.event_id, key(a, METHOD_CLAIM), d)
                rep.asked += 1
        except Exception as e:   # noqa: BLE001 - a scene that cannot be checked gets no row
            log.warning("場面の動作を再確認できませんでした %s: %s", sc.event_id, e)
            rep.skipped.append(sc.event_id)
    return rep
