"""解説動画 (services.explainer): who can have one, the request, the worker's job, the stamp, the
session check, the caps, and that erasure / retention / deletion take the videos with them.
Synthetic works only (tests/pdf_synthetic.py)."""
import json

import pytest

from pdf_synthetic import save, video_group, work
from pfb_app.domain.models import Stage
from pfb_app.export import explainer_video as ev
from pfb_app.export import sheets
from pfb_app.export.views import person as person_view
from pfb_app.services import auth, erasure, explainer, retention, review
from test_wizard_ui_basic import env  # noqa: F401 (env)


@pytest.fixture(autouse=True)
def one_moment_each(request, monkeypatch):
    """One moment per video keeps the encodes short, except where the count is the point."""
    if "every_moment" not in request.node.name:
        monkeypatch.setattr(explainer, "MAX_MOMENTS", 1)
        monkeypatch.setattr(explainer, "TRACKING_MAX_S", 4.0)


def _group(env, **kw):
    settings, repo = env
    aid = work(repo)
    sid = video_group(settings, repo, aid, people=kw.pop("people", 3), seconds=40, **kw)
    subjects = repo.list_subjects(sid)
    return settings, repo, aid, sid, subjects


def _make(repo, settings, sid, subject_id, session_id=None):
    job = explainer.request(repo, sid, subject_id, session_id=session_id, actor="coach")
    assert explainer.run_next(repo, settings)
    return repo.explainer_job(job)


def test_who_can_have_a_video(env):
    settings, repo, aid, sid, subs = _group(env, people=4, short=("Pb",), unapproved=("Pc",))
    a, b, c, d = subs
    assert explainer.person_reason(repo, a["id"]) is None
    assert explainer.person_reason(repo, b["id"]) == explainer.NO_MOMENTS     # short text
    assert explainer.person_reason(repo, c["id"]) == explainer.NOT_APPROVED
    repo.set_excluded(d["id"], True)
    assert explainer.person_reason(repo, d["id"]) == explainer.NOT_EVALUATED
    with pytest.raises(explainer.Refused, match="承認"):
        explainer.request(repo, sid, c["id"], session_id=None, actor=None)
    assert explainer.tracking_reason(repo, sid) is None
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-10-24' WHERE id=?", (aid,))
    assert explainer.person_reason(repo, a["id"]) == explainer.MEDIA_REMOVED
    assert explainer.tracking_reason(repo, sid) == explainer.MEDIA_REMOVED


def _pdf_order(repo, aid, subject_id):
    report = sheets.build_report(repo, aid)
    sheet = next(s for s in report.approved if s.subject_id == subject_id)
    shown = person_view(report, sheet, "handout")
    return [p["m"] for p in shown["pictures"]] + [p["m"] for p in shown["moments_text"]]


def test_the_moments_are_the_handouts_in_its_order_every_moment(env):
    settings, repo, aid, sid, subs = _group(env)
    for s in subs:
        pdf = _pdf_order(repo, aid, s["id"])
        assert [m.ref_id for m in explainer.handout_moments(repo, s["id"])] == \
            [m.ref_id for m in pdf]
        plan = explainer.person_plan(repo, settings, s["id"])
        assert [p.when for p in plan.parts] == [m.when for m in pdf]
        assert all("本人" not in p.what for p in plan.parts)          # the handout's words
        assert all(p.source is not None and p.mine for p in plan.parts)


def test_a_failed_picture_keeps_the_pdfs_order(env, monkeypatch):
    from pfb_app.export import pictures
    settings, repo, aid, sid, subs = _group(env)
    real = pictures.make_pictures

    def one_fails(requests, cache_dir):
        made = real(requests, cache_dir)
        first = min(made, key=lambda r: r.frame)            # the earliest picture fails
        return {r: p for r, p in made.items() if r != first}
    monkeypatch.setattr(pictures, "make_pictures", one_fails)
    s = subs[0]
    pdf = _pdf_order(repo, aid, s["id"])
    got = explainer.handout_moments(repo, s["id"])
    assert [m.ref_id for m in got] == [m.ref_id for m in pdf]
    assert pdf[-1].picture is None                          # it moved to the end in the PDF


def test_the_why_is_in_the_handouts_words_under_its_heading(env):
    import json as _json
    settings, repo, aid, sid, subs = _group(env)
    con = repo.latest_run(sid, Stage.contribution, "cam1")
    out = dict(con.output)
    for j in out["judgments"]:
        j["contribution_interpretation"] = "本人が別の参加者の動きを助けた"
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?",
                      (_json.dumps(out, ensure_ascii=False), con.id))
    plan = explainer.person_plan(repo, settings, subs[0]["id"])
    assert plan.parts and all(p.why == "あなたがほかの選手の動きを助けた" for p in plan.parts)
    head, what, why = ev.caption_lines(plan.parts[0])
    assert why.startswith("評価のポイント：") and "本人" not in why + what


def test_the_last_analysed_frame_is_the_engines(env):
    settings, repo, aid, sid, subs = _group(env)
    meta = repo.get_submission(sid)["meta"]
    meta["end_sec"] = 30.0
    repo.update_submission_meta(sid, meta)
    plan = explainer.person_plan(repo, settings, subs[0]["id"])
    assert plan.parts[0].source.last_frame == round(30.0 * 10) + 1


def test_a_video_is_made_cached_and_shown_until_something_changes_every_moment(env):
    settings, repo, aid, sid, subs = _group(env)
    a = subs[0]
    assert explainer.status(repo, settings, sid, a["id"]).state == "none"
    job = explainer.request(repo, sid, a["id"], session_id=None, actor="coach")
    assert explainer.request(repo, sid, a["id"], session_id=None, actor="coach") == job
    assert explainer.status(repo, settings, sid, a["id"]).state == "queued"
    assert explainer.run_next(repo, settings)
    assert repo.explainer_job(job)["status"] == "done"
    shown = explainer.status(repo, settings, sid, a["id"])
    assert shown.state == "done" and shown.path.is_file()
    assert shown.path.parent == explainer.folder(settings, aid, sid)
    assert shown.path.parent.parent.name == "handout"
    meta = json.loads(shown.path.with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["moments"] == 4 and meta["stamp"] == explainer.person_stamp(repo, a["id"])
    expect = ev.INTRO_S + ev.END_S + 4 * (ev.BEFORE_S + ev.AFTER_S)
    assert abs(meta["seconds"] - expect) < 1.5
    # a new name: the file is outdated (never shown with the old name)
    repo.rename_subject(a["id"], "山田花子2")
    assert explainer.status(repo, settings, sid, a["id"]).state == "stale"
    repo.rename_subject(a["id"], a["display_name"])
    assert explainer.status(repo, settings, sid, a["id"]).state == "done"
    # the video file changed: outdated
    import os
    video = repo.get_submission(sid)["meta"]["cameras"]["cam1"]
    os.utime(video, ns=(1, 1))
    assert explainer.status(repo, settings, sid, a["id"]).state == "stale"
    # a new track run (re-tracked): never shown as current
    track = repo.latest_run(sid, Stage.track, "cam1")
    save(repo, sid, Stage.track, track.output, camera="cam1", upstream=track.upstream_run_ids)
    assert explainer.status(repo, settings, sid, a["id"]).state in ("stale", "unavailable")
    # the approval taken back: nothing to show
    review.unapprove(repo, a["id"])
    assert explainer.status(repo, settings, sid, a["id"]).state == "unavailable"


def test_another_persons_approval_does_not_outdate_a_video(env):
    settings, repo, aid, sid, subs = _group(env, unapproved=("Pc",))
    a, _, c = subs
    _make(repo, settings, sid, a["id"])
    review.approve(repo, c["id"])
    assert explainer.status(repo, settings, sid, a["id"]).state == "done"


def test_the_tracking_video_names_only_evaluated_people(env):
    settings, repo, aid, sid, subs = _group(env)
    repo.set_excluded(subs[1]["id"], True)
    plan = explainer.tracking_plan(repo, settings, sid)
    assert sorted(plan.names.values()) == sorted([subs[0]["display_name"],
                                                  subs[2]["display_name"]])
    assert plan.to_s - plan.from_s <= explainer.TRACKING_MAX_S
    assert plan.tick(5.0).startswith("開始から")
    job = _make(repo, settings, sid, None)
    assert job["status"] == "done"
    shown = explainer.status(repo, settings, sid, None)
    assert shown.state == "done" and shown.path.name == "tracking.mp4"
    repo.rename_subject(subs[0]["id"], "べつの名前")       # a name on screen changed
    assert explainer.status(repo, settings, sid, None).state == "stale"


_LOGINS = iter(range(1, 10 ** 6))


def _session(repo, settings, role="evaluator"):
    user = auth.create_user(repo, f"user{next(_LOGINS)}", "テスト", role, "long-enough-pass-1")
    cookie = auth.start_session(repo, settings, user)
    return user, int(cookie.split(".")[1])


def test_a_revoked_or_demoted_requester_gets_nothing_written(env):
    settings, repo, aid, sid, subs = _group(env)
    a, b = subs[0], subs[1]
    user, session = _session(repo, settings)
    job = explainer.request(repo, sid, a["id"], session_id=session, actor=user.login)
    auth.revoke_session(repo, session)
    assert explainer.run_next(repo, settings)
    row = repo.explainer_job(job)
    assert (row["status"], row["error"]) == ("cancelled", explainer.SESSION_GONE)
    assert not explainer.folder(settings, aid, sid).exists() or \
        not list(explainer.folder(settings, aid, sid).glob("*.mp4"))
    user2, session2 = _session(repo, settings, "admin")
    job = explainer.request(repo, sid, b["id"], session_id=session2, actor=user2.login)
    auth.create_user(repo, "spare", "予備", "admin", "long-enough-pass-2")   # keep an admin
    auth.set_role(repo, user2.id, "viewer")
    assert explainer.run_next(repo, settings)
    assert repo.explainer_job(job)["status"] == "cancelled"
    assert not explainer.file_of(settings, aid, sid, b["id"])[0].exists()
    _, ok = _session(repo, settings, "admin")
    assert _make(repo, settings, sid, b["id"], ok)["status"] == "done"


def test_revoked_while_rendering_publishes_nothing(env, monkeypatch):
    settings, repo, aid, sid, subs = _group(env)
    user, session = _session(repo, settings)
    real = ev.render_person

    def render(plan, target, **kw):
        out = real(plan, target, **kw)
        auth.revoke_session(repo, session)
        return out
    monkeypatch.setattr(ev, "render_person", render)
    job = explainer.request(repo, sid, subs[0]["id"], session_id=session, actor=user.login)
    explainer.run_next(repo, settings)
    assert repo.explainer_job(job)["status"] == "cancelled"
    folder = explainer.folder(settings, aid, sid)
    assert not list(folder.glob("*.mp4")) and not list(folder.glob(".*part*"))


def test_changed_while_rendering_is_not_published(env, monkeypatch):
    settings, repo, aid, sid, subs = _group(env)
    real = ev.render_person

    def render(plan, target, **kw):
        out = real(plan, target, **kw)
        repo.rename_subject(subs[0]["id"], "べつの名前")
        return out
    monkeypatch.setattr(ev, "render_person", render)
    job = _make(repo, settings, sid, subs[0]["id"])
    assert (job["status"], job["error"]) == ("failed", explainer.CHANGED)
    assert not list(explainer.folder(settings, aid, sid).glob("*.mp4"))


def test_low_disk_refuses_and_a_render_error_is_plain(env, monkeypatch):
    from pfb_app.ops import disk
    settings, repo, aid, sid, subs = _group(env)

    def low(*a, **k):
        raise disk.DiskLow("low")
    monkeypatch.setattr(disk, "ensure_room", low)
    job = _make(repo, settings, sid, subs[0]["id"])
    assert (job["status"], job["error"]) == ("failed", explainer.LOW_DISK)
    shown = explainer.status(repo, settings, sid, subs[0]["id"])
    assert (shown.state, shown.reason) == ("failed", explainer.LOW_DISK)
    monkeypatch.undo()

    def broken(*a, **k):
        raise ev.RenderError("読めません")
    monkeypatch.setattr(ev, "render_person", broken)
    job = _make(repo, settings, sid, subs[0]["id"])
    assert (job["status"], job["error"]) == ("failed", "読めません")
    assert not list(explainer.folder(settings, aid, sid).glob(".*"))


def test_the_group_folder_is_capped_and_loses_people_without_a_video(env, monkeypatch):
    settings, repo, aid, sid, subs = _group(env)
    a, b, c = subs
    _make(repo, settings, sid, a["id"])
    first = explainer.file_of(settings, aid, sid, a["id"])[0]
    monkeypatch.setattr(explainer, "MAX_GROUP_BYTES", first.stat().st_size + 10)
    _make(repo, settings, sid, b["id"])
    assert not first.exists() and explainer.file_of(settings, aid, sid, b["id"])[0].exists()
    monkeypatch.setattr(explainer, "MAX_GROUP_BYTES", 10 ** 9)
    repo.set_excluded(b["id"], True)
    _make(repo, settings, sid, c["id"])
    assert not explainer.file_of(settings, aid, sid, b["id"])[0].exists()


def test_erasure_takes_every_video_of_the_group_and_cancels_its_jobs(env):
    settings, repo, aid, sid, subs = _group(env)
    a, b, c = subs
    _make(repo, settings, sid, a["id"])
    _make(repo, settings, sid, None)
    queued = explainer.request(repo, sid, c["id"], session_id=None, actor=None)
    erasure.erase(repo, settings, b["id"])
    assert not explainer.folder(settings, aid, sid).exists()
    assert repo.explainer_job(queued)["status"] == "cancelled"
    assert explainer.person_reason(repo, b["id"]) == explainer.NOT_EVALUATED


def test_retention_removes_the_videos_and_cancels_the_jobs(env):
    settings, repo, aid, sid, subs = _group(env)
    _make(repo, settings, sid, subs[0]["id"])
    queued = explainer.request(repo, sid, subs[1]["id"], session_id=None, actor=None)
    folder = explainer.folder(settings, aid, sid)
    assert folder.exists()
    repo.conn.execute("UPDATE assignments SET media_removed_at='2026-10-24' WHERE id=?", (aid,))
    retention._finish(repo, settings, aid)
    assert not folder.exists() and not folder.parent.exists()
    assert repo.explainer_job(queued)["status"] == "cancelled"
    assert "handout" in retention.HEAVY_DIRS


def test_a_deleted_work_mid_render_leaves_nothing(env, monkeypatch):
    from pfb_app.services import deletion
    settings, repo, aid, sid, subs = _group(env)
    real = ev.render_person

    def render(plan, target, **kw):
        out = real(plan, target, **kw)
        deletion.delete_work(repo, settings, aid)
        return out
    monkeypatch.setattr(ev, "render_person", render)
    job = explainer.request(repo, sid, subs[0]["id"], session_id=None, actor=None)
    explainer.run_next(repo, settings)
    assert repo.explainer_job(job) is None
    from pfb_app.storage import files
    assert not files.assignment_dir(settings, aid, create=False).exists()


def test_a_worker_start_fails_a_half_made_video(env):
    settings, repo, aid, sid, subs = _group(env)
    job = explainer.request(repo, sid, subs[0]["id"], session_id=None, actor=None)
    repo.explainer_claim()
    assert explainer.recover(repo) == 1
    assert explainer.status(repo, settings, sid, subs[0]["id"]).reason == explainer.STOPPED
    assert repo.explainer_job(job)["status"] == "failed"


def test_a_job_without_a_heartbeat_fails_so_it_can_be_made_again(env):
    settings, repo, aid, sid, subs = _group(env)
    job = explainer.request(repo, sid, subs[0]["id"], session_id=None, actor=None)
    repo.explainer_claim()
    assert explainer.fail_stale(repo) == 0                     # it just started
    repo.conn.execute("UPDATE explainer_jobs SET heartbeat_at='2026-01-01T00:00:00+00:00'"
                      " WHERE id=?", (job,))
    assert explainer.fail_stale(repo) == 1
    shown = explainer.status(repo, settings, sid, subs[0]["id"])
    assert (shown.state, shown.reason) == ("failed", explainer.HUNG)
    again = explainer.request(repo, sid, subs[0]["id"], session_id=None, actor=None)
    assert again != job and repo.explainer_job(again)["status"] == "queued"


def test_the_sweep_removes_videos_nobody_may_see_any_more(env):
    import os
    settings, repo, aid, sid, subs = _group(env)
    a, b, c = subs
    for s in (a, b, c):
        _make(repo, settings, sid, s["id"])
    _make(repo, settings, sid, None)
    folder = explainer.folder(settings, aid, sid)
    part = folder / ".person_9.abc.part.mp4"
    part.write_bytes(b"x")
    os.utime(part, (1, 1))                                      # left by a stop long ago
    review.unapprove(repo, a["id"])
    repo.set_excluded(b["id"], True)                            # the tracking names b: outdated
    explainer.sweep(repo, settings)
    left = sorted(p.name for p in folder.glob("*.mp4"))
    assert left == [f"person_{c['id']}.mp4"]
    assert not part.exists()
    assert explainer.status(repo, settings, sid, c["id"]).state == "done"


def _hevc(monkeypatch):
    """Every video reads as a 4K HEVC one (a player needs its proxy)."""
    from pfb_app.services import moment_clip
    explainer._probe.cache_clear()
    monkeypatch.setattr(moment_clip, "probe",
                        lambda p: moment_clip.Probe(3840, 2160, "hevc", 40.0))


def test_a_video_that_needs_a_proxy_waits_for_it_truthfully(env, monkeypatch):
    from pfb_app.services import proxy
    settings, repo, aid, sid, subs = _group(env)
    a = subs[0]
    _hevc(monkeypatch)
    video = repo.get_submission(sid)["meta"]["cameras"]["cam1"]
    why = explainer.person_reason(repo, a["id"])
    assert why == explainer.PREPARING.format(n=0)
    shown = explainer.status(repo, settings, sid, a["id"])
    assert (shown.state, shown.progress) == ("preparing", 0.0)
    with pytest.raises(explainer.Refused):
        explainer.request(repo, sid, a["id"], session_id=None, actor=None)
    _, meta_path, _ = proxy.paths(settings, video)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({**proxy._fingerprint(__import__("pathlib").Path(video)),
                                     "complete": False, "pending": True}))
    proxy.progress_path(settings, video).write_text("0.42")
    assert explainer.person_reason(repo, a["id"]) == explainer.PREPARING.format(n=42)
    assert explainer.tracking_reason(repo, sid) == explainer.PREPARING.format(n=42)
    meta_path.write_text(json.dumps({**proxy._fingerprint(__import__("pathlib").Path(video)),
                                     "complete": False, "failed": True}))
    assert explainer.person_reason(repo, a["id"]) == explainer.CONVERT_FAILED
    assert explainer.status(repo, settings, sid, a["id"]).state == "unavailable"


def test_the_worker_starts_the_proxies_approved_groups_need_one_at_a_time(env, monkeypatch):
    from pfb_app.services import proxy
    settings, repo, aid, sid, subs = _group(env)
    other = video_group(settings, repo, aid, people=2, seconds=20, label="B",
                        unapproved=("Pa", "Pb"))
    _hevc(monkeypatch)
    started = []
    monkeypatch.setattr(proxy, "start", lambda s, src, w, h, codec=None:
                        started.append((str(src), codec)) or True)
    monkeypatch.setattr(proxy, "busy", lambda s: False)
    explainer.ensure_sources(repo, settings)
    video = repo.get_submission(sid)["meta"]["cameras"]["cam1"]
    assert started == [(video, "hevc")]                    # group B has nobody approved yet
    assert repo.get_submission(other)["meta"]["cameras"]["cam1"] != video
    monkeypatch.setattr(proxy, "busy", lambda s: True)     # one is being made: wait
    started.clear()
    explainer.ensure_sources(repo, settings)
    assert started == []
