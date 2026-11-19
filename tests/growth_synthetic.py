"""Synthetic works for the 成長の記録 tests: invented names, made-up observations and texts, no
video file at all (the camera path does not exist). Each good moment is tagged with one of the
profile's categories through the judgment's 評価の観点 (rubric_criteria)."""
from __future__ import annotations

from pfb_app.domain.models import Stage, StageRun


def save(repo, sid, stage, output, camera=None, subject_id=None, upstream=()):
    return repo.save_stage_run(StageRun(
        id=None, submission_id=sid, stage=stage, camera=camera, subject_id=subject_id, version=0,
        source="model", upstream_run_ids=list(upstream), prompt_hash=None, model_id=None,
        output=output, status="ok", error=None, warnings=[], duration_sec=0.0, created_at=""))


def text_of(name: str, title: str) -> str:
    return f"{name}さんは「{title}」で、まわりの人の話をよく聞いていました。最後まで声をかけ続けたのもよかったです。"


def add_work(repo, title: str, people: dict, *, profile: str = "meeting", day: str = "2026-09-01",
             group: str | None = None, label: str = "A", approve=True, meta: dict | None = None,
             aid: int | None = None) -> tuple[int, int, dict[str, int]]:
    """A video work (or one more group of `aid`) whose people have the good moments given:
    `people` {name: [category, ...]} (one moment per entry). `approve`: True, False or the set
    of names to approve. Returns (assignment id, submission id, {name: subject id})."""
    if aid is None:
        domain = "sports" if profile == "sports" else "school"
        aid = repo.create_assignment(title, "video", "活動の内容", "", "standard", False,
                                     domain=domain, sport="basketball" if domain == "sports"
                                     else None, profile=profile)
        repo.conn.execute("UPDATE assignments SET created_at=? WHERE id=?",
                          (f"{day}T10:00:00+09:00", aid))
        if group:
            repo.set_growth_group(aid, group)
    sid = repo.create_submission(aid, label, {"cameras": {"cam1": "/nonexistent/cam1.mp4"},
                                              **(meta or {})})
    subjects, events, judgments, first = {}, [], [], {}
    for k, (name, cats) in enumerate(people.items(), 1):
        pid = f"P{k}"
        subjects[name] = repo.add_subject(sid, pid, name, None, {})
        for n, cat in enumerate(cats, 1):
            eid = f"cam1-{k:04d}-{n:02d}"
            first.setdefault(name, eid)
            events.append({"event_id": eid, "participant": pid, "video_time": f"00:{n:02d}",
                           "camera": "cam1", "observed_behavior": f"本人が{cat}の場面で動いた"})
            judgments.append({"ref_id": eid, "contribution_interpretation": "場が進んだ",
                              "is_positive_contribution": True, "rubric_criteria": [cat]})
    beh = save(repo, sid, Stage.behavior, {"events": events}, camera="cam1")
    con = save(repo, sid, Stage.contribution, {"judgments": judgments}, camera="cam1",
               upstream=[beh.id])
    for name, subject_id in subjects.items():
        run = save(repo, sid, Stage.pfb, {"pfb_status": "generated", "pfb": text_of(name, title),
                                          "evidence_refs": [first[name]] if name in first else []},
                   subject_id=subject_id, upstream=[con.id])
        if approve is True or (approve and name in approve):
            repo.set_approval(subject_id, run.id)
    return aid, sid, subjects
