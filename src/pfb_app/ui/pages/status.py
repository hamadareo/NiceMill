"""処理状況：運用の状態（ワーカー・待ち・空き・バックアップ・止まったもの）、GPU、人を見つける方法（エンジンが報告したとき）、ジョブ（進み具合つき）、失敗したジョブの再試行。"""
import math

import streamlit as st

from pfb_app.domain.models import JobStatus, Stage
from pfb_app.engine import LOG_PREFIX
from pfb_app.inference.gpu import GpuGuard, GpuQueryError, query_nvidia_smi
from pfb_app.pipeline.common import usable
from pfb_app.pipeline.tracking import tracker_text
from pfb_app.services import review
from pfb_app.services.quality_stats import RETRY_JOB
from pfb_app.ui import ops_panel
from pfb_app.ui.common import (JOB_STATUS_LABELS, MODEL_LABELS, STAGE_LABELS, flash, get_repo,
                               get_settings, show_flash)
from pfb_app.ui.wizard.progress import camera_label
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

repo = get_repo()
settings = get_settings()
st.title("処理状況")
show_flash("school")

ops_panel.render(repo, settings)   # 運用の状態: worker, queue, disk, backup, stops (ops lane)

st.subheader("GPU")
try:
    snapshot = query_nvidia_smi()
except (GpuQueryError, OSError, ValueError):
    st.info("GPU情報を取得できません")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("全体", f"{snapshot.total_gb:.1f} GB")
    c2.metric("使用中", f"{snapshot.used_gb:.1f} GB")
    c3.metric("空き", f"{snapshot.free_gb:.1f} GB")
    guard = GpuGuard(settings, query=lambda: snapshot)
    for key, spec in settings.models.items():
        ok, reason = guard.can_start(spec.budget_gb)
        st.write(f"{MODEL_LABELS.get(key, key)}モデル: " + ("開始できます" if ok else f"開始待ち（{reason}）"))
    # tracking engine (PFB3x3): person tracking and the engine's own vLLM server
    engine_keys = [("track", settings.track_budget_gb)] + [
        (f"pfb3x3:{k}", settings.pfb3x3_budget_gb[k]) for k in settings.models
        if k in settings.pfb3x3_budget_gb]
    for key, budget in engine_keys:
        ok, reason = guard.can_start(budget)
        st.write(f"{MODEL_LABELS.get(key, key)}: " + ("開始できます" if ok else f"開始待ち（{reason}）"))


def tracker_lines() -> list[str]:
    """「作業／グループ（カメラ1）: 標準の追跡」 per camera whose latest usable track run says
    which tracker found the people (engine 58f8341); cameras without that say nothing."""
    out = []
    for a in repo.list_assignments():
        for sub in repo.list_submissions(a["id"]):
            cameras = (sub.get("meta") or {}).get("cameras")
            for cam in (cameras if isinstance(cameras, dict) else {}):
                run = repo.latest_run(sub["id"], Stage.track, cam)
                text = tracker_text(run.output) if usable(run) else None
                if text:
                    out.append(f"{a['title']}／{sub['label']}（{camera_label(cam)}）: {text}")
    return out


trackers = tracker_lines()
if trackers:
    st.subheader("人を見つける方法")
    for line in trackers:
        st.caption(line)

untranslated = [(a, sub, cams) for a in repo.list_assignments()
                for sub in repo.list_submissions(a["id"])
                if (cams := review.untranslated_behavior(repo, sub["id"]))]
if untranslated:
    st.subheader(review.UNTRANSLATED)
    for a, sub, cams in untranslated:
        st.warning(f"{a['title']}／{sub['label']}: " + review.untranslated_message(cams))


def short_error(error: str | None) -> str:
    """The first line of a job's error, plus the engine's log file when it names one
    (the full stderr tail stays in the job and in that log)."""
    lines = [line.strip() for line in (error or "").splitlines() if line.strip()]
    if not lines:
        return ""
    log_line = next((line for line in lines[1:] if line.startswith(LOG_PREFIX)), None)
    return lines[0] + (f"（{log_line}）" if log_line else "")


def progress_text(progress: float | None, note: str | None) -> str:
    """A job's own progress as the worker last wrote it: 「30%（12 / 40 区間）」 (read with the
    job rows: no query of its own)."""
    pct = f"{math.floor(round(progress * 100, 6))}%" if progress is not None else ""
    if pct and note:
        return f"{pct}（{note}）"
    return pct or note or ""


st.subheader("ジョブ")
jobs = repo.list_jobs()
if not jobs:
    st.info("ジョブはありません。")
else:
    show_all = st.toggle("すべての作業を表示", key="status_show_all",
                         help="切ると、まだ終わっていない作業のジョブだけを表示します。")
    if not show_all:
        # open work = a group with any job that has not finished (queued, running, waiting, failed)
        open_work = {j.submission_id for j in jobs if j.status != JobStatus.done}
        jobs = [j for j in jobs if j.submission_id in open_work]
        if not jobs:
            st.info("いま動いている作業・止まっている作業はありません。"
                    "終わった作業も見るには「すべての作業を表示」をオンにしてください。")
            st.stop()          # the jobs table is the last section of this page
    labels = {s["id"]: s["label"] for a in repo.list_assignments() for s in repo.list_submissions(a["id"])}
    st.dataframe([{
        "ID": j.id, "グループ・書類": labels.get(j.submission_id, j.submission_id),
        "段階": STAGE_LABELS[j.stage], "処理": MODEL_LABELS.get(j.model_key, j.model_key),
        "カメラ": j.camera or "",
        "状態": JOB_STATUS_LABELS[j.status], "進み具合": progress_text(j.progress, j.progress_note),
        "待機・失敗の理由": short_error(j.error), "試行回数": j.attempts} for j in jobs],
        hide_index=True)
    failed = [j for j in jobs if j.status in (JobStatus.failed, JobStatus.failed_retryable)]
    for j in failed:
        cols = st.columns([6, 1])
        cols[0].write(f"ジョブ{j.id}（{STAGE_LABELS[j.stage]}）: {short_error(j.error) or '理由不明'}")
        if cols[1].button("再試行", key=f"retry-{j.id}"):
            if repo.retry_job(j.id):
                # 「この解析の確かさ」: a step put back here is not a first try
                repo.audit(RETRY_JOB, f"job:{j.id}", {"job_ids": [j.id]})
                flash(f"ジョブ{j.id}を再試行します。")
                st.rerun()
            else:
                st.error("同じ処理がすでに実行中・待機中のため、再試行できません。")
