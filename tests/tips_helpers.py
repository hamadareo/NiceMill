"""A reviewable video group with real-looking feedback and its cited events, for the practice
tips tests (synthetic names and texts only)."""
import json

from pfb_app.domain.models import Stage, StageRun
from pfb_app.inference.fake import FakeBackend
from pfb_app.pipeline.common import StageContext

PA_EVENTS = ["ドリブルでボールを運び、仲間にパスを出す", "リバウンドを取ってすぐ前へ走り出す"]
PZ_EVENTS = ["ディフェンスで声をかけて、仲間を守る位置へ動く", "ボールを持った仲間に近づいて受け取る"]
FEEDBACK = {"Pa": "ドリブルでボールを運んでパスを出し、仲間の攻めを助けていました。",
            "Pz": "守りで声をかけ、仲間を守る位置に動いていました。"}
TIP = {"Pa": "ドリブルのあと、周りの動きを見てパスを出してみる",
       "Pz": "声をかけながら、仲間を守る位置に動くことを続けてみる"}


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=(), source="model",
         status="ok"):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source=source, upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status=status, error=None if status == "ok" else "x", warnings=[],
        duration_sec=0.0, created_at=""))


def group(repo, label="A班", domain="school", sport=None, feedback=True):
    """(assignment id, submission id, {pid: subject id}, {pid: pfb run}) with two people, one
    camera, their events, positive judgments and generated feedback (model source)."""
    aid = repo.create_assignment("ミニゲーム", "video", "ボールを回して攻める", "協力", "standard",
                                 False, domain=domain, sport=sport)
    sid = repo.create_submission(aid, label, {})
    subjects = {"Pa": repo.add_subject(sid, "Pa", "山田花子", "", {}),
                "Pz": repo.add_subject(sid, "Pz", "鈴木一郎", "", {})}
    repo.set_identity_confirmed(sid, True)
    save(repo, sid, Stage.prepare, {"frames": []}, camera="cam1")
    events, n = [], 0
    for pid, texts in (("Pa", PA_EVENTS), ("Pz", PZ_EVENTS)):
        for t in texts:
            n += 1
            events.append({"video_time": f"00:{10 * n:02d}", "participant": pid,
                           "observed_behavior": t, "utterance": "", "speaker": "unknown",
                           "evidence_source": "video", "confidence": "high",
                           "event_id": f"cam1-{n:04d}", "camera": "cam1",
                           "task_elapsed": f"00:{10 * n:02d}"})
    behavior = save(repo, sid, Stage.behavior, {"events": events}, camera="cam1")
    contribution = save(repo, sid, Stage.contribution, {"judgments": [
        {"ref_id": e["event_id"], "contribution_interpretation": "チームの攻めや守りにつながった",
         "is_positive_contribution": True, "rubric_criteria": []} for e in events]},
        camera="cam1", upstream=[behavior.id])
    runs = {}
    if not feedback:
        return aid, sid, subjects, runs
    for pid, refs in (("Pa", ["cam1-0001", "cam1-0002"]), ("Pz", ["cam1-0003", "cam1-0004"])):
        runs[pid] = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": FEEDBACK[pid],
                                                "evidence_refs": refs},
                         subject_id=subjects[pid], upstream=[contribution.id])
    return aid, sid, subjects, runs


def tips_json(pid="Pa", refs=None, text=None):
    refs = refs or {"Pa": ["cam1-0001"], "Pz": ["cam1-0003"]}[pid]
    return json.dumps({"tips_status": "generated", "tips": [
        {"text": text or TIP[pid], "evidence_refs": refs}]}, ensure_ascii=False)


def team_json(text="ボールを持った仲間を見て、声をかけながら動いてみる", people=(1, 2)):
    return json.dumps({"tips_status": "generated", "tips": [
        {"text": text, "people": list(people)}]}, ensure_ascii=False)


def ctx_for(settings, repo, fn, yield_check=lambda: False):
    backend = FakeBackend(fn)
    return StageContext(settings, repo, backend, yield_check), backend
