"""「運用の状態」: the top section of the admin status page (処理状況, under 管理用の画面).

Worker alive, the queue, free disk space, the last backup, recent stops (with whether the
automatic retry already ran), recorded alerts, log files and versions, from one read-only
snapshot (ops.health). GPU memory is the page's own GPU section right below."""
from datetime import datetime

import streamlit as st

from pfb_app.config import Settings
from pfb_app.domain.models import JobStatus, Stage
from pfb_app.ops import backup, disk, guard, health, logs
from pfb_app.storage.db import Repo
from pfb_app.ui.common import JOB_STATUS_LABELS, STAGE_LABELS

TITLE = "運用の状態"
WORKER_DOWN = ("解析を進めるワーカーが動いていません。サーバーで "
               "`systemctl --user restart pfb-worker`（systemd を使っていないときは "
               "`scripts/start.sh`）を実行してください。")
WORKER_UNCLEAN = "前回のワーカーは正しく終了していません（停電・強制終了など）。"
GUARD_LINE = "見張り（{ago}に確認）: {summary}"
GUARD_STALE = ("見張りの確認が{ago}から更新されていません。サーバーで "
               "`systemctl --user status pfb-guard.timer` を確かめてください。")
BACKUP_NONE = "バックアップはまだありません。"
BACKUP_FAILED = "前回のバックアップに失敗しました: {error}"
BACKUP_OLD = "最後のバックアップから{hours:.0f}時間たっています。夜間のバックアップが動いているか確かめてください。"
BACKUP_STALE_HOURS = 36


def backup_privacy_note(settings: Settings) -> str:
    """Deleting a work does not reach into the nightly backups (they age out)."""
    n, weeks = settings.backup_keep, settings.backup_keep_weekly
    longest = backup.max_age_days(settings)
    later = (f"、そのあとは週に1回分ずつ{weeks}週分" if weeks > 0 else "")
    return (f"削除した作業も、夜間のバックアップには最大{longest}日間残ります"
            f"（毎晩のもの{n}日分{later}を残し、それより古いものは自動で消えます）。"
            "データベースを戻す前に作られたコピー（app.db.before-restore-…）は、"
            "管理者が消すまで残ります。"
            "記録が始まる前の作業のモデルの回答は、整理の手順を実行するまで残ります。")


def _size(n: int) -> str:
    for unit, size in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= size:
            return f"{n / size:.1f} {unit}"
    return f"{n} B"


def _local(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso).astimezone().strftime("%m/%d %H:%M")
    except ValueError:
        return iso


def _ago(hours: float | None) -> str:
    if hours is None:
        return "—"
    if hours < 1:
        return f"{max(1, round(hours * 60))}分前"
    if hours < 48:
        return f"{hours:.0f}時間前"
    return f"{hours / 24:.0f}日前"


def _stage(value: str) -> str:
    try:
        return STAGE_LABELS[Stage(value)]
    except (KeyError, ValueError):
        return value


def _status(value: str) -> str:
    try:
        return JOB_STATUS_LABELS[JobStatus(value)]
    except (KeyError, ValueError):
        return value


def render(repo: Repo, settings: Settings, snap: health.Health | None = None) -> health.Health:
    snap = snap or health.snapshot(repo, settings)
    st.subheader(TITLE)
    q = snap.queue
    waiting = q.get("queued", 0) + q.get("waiting_gpu", 0)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ワーカー", "動いています" if snap.worker_alive else "止まっています",
              help=(f"最後の応答: {snap.worker_beat_age_sec:.0f}秒前"
                    if snap.worker_beat_age_sec is not None else "応答の記録がありません"))
    c2.metric("待ち／実行中", f"{waiting}件／{q.get('running', 0)}件")
    c3.metric("保存場所の空き", f"{snap.disk.free_gb:.0f} GB" if snap.disk.level != "unknown"
              else "不明", help=f"全体 {snap.disk.total_gb:.0f} GB（{snap.disk.path}）")
    last = snap.backup_last
    c4.metric("最後のバックアップ",
              "失敗" if last and not last.get("ok") else _ago(snap.backup_age_hours),
              help=f"保存しているバックアップ: {snap.backup_count}件"
                   f"（{_size(snap.backup_bytes)}）")
    watch = guard.read_status(settings)
    if watch is not None:
        status, age = watch
        if age > guard.STATUS_FRESH_SEC:
            st.warning(GUARD_STALE.format(ago=_ago(age / 3600)))
        else:
            st.caption(GUARD_LINE.format(summary=status.get("summary", ""), ago=_ago(age / 3600)))
    if not snap.worker_alive:
        st.error(WORKER_DOWN + (WORKER_UNCLEAN if snap.worker_unclean else ""))
    if snap.disk.level == "refuse":
        st.error(snap.disk.message)
    elif snap.disk.level == "warn":
        st.warning(snap.disk.message)
    if last is None:
        st.info(BACKUP_NONE)
    elif not last.get("ok"):
        st.error(BACKUP_FAILED.format(error=last.get("error") or "理由不明"))
    elif (snap.backup_age_hours or 0) > BACKUP_STALE_HOURS:
        st.warning(BACKUP_OLD.format(hours=snap.backup_age_hours))

    with st.expander(f"最近止まったもの（{len(snap.failures)}件）"):
        if not snap.failures:
            st.write("止まったものはありません。")
        else:
            st.table([{"ID": f.job_id, "段階": _stage(f.stage), "状態": _status(f.status),
                           "理由": f.error, "最後の記録": _local(f.at),
                           "自動の再試行": "済み" if f.auto_retried else "—"}
                      for f in snap.failures])
    with st.expander(f"お知らせの記録（{len(snap.alerts)}件）"):
        st.caption("お知らせの送り先: " + ("設定済み" if snap.alert_configured
                                       else "未設定（記録だけ残します）"))
        for a in snap.alerts:
            when = datetime.fromtimestamp(a["at"]).strftime("%m/%d %H:%M")
            st.write(f"{when}　{a.get('title', '')}" + ("（送信済み）" if a.get("sent") else ""))
    with st.expander("記録ファイルと版"):
        st.write(f"記録ファイルの場所: `{snap.log_dir}`")
        st.write(f"アプリの版: `{snap.app_commit or '不明'}`　解析部品の版: "
                 f"`{snap.engine_commit or '不明'}`　起動の仕方: {health.service_hint()}")
        if snap.backup_newest:
            st.write(f"最新のバックアップ: `{snap.backup_newest}`　"
                     f"バックアップの大きさ: {_size(snap.backup_bytes)}")
        st.caption(backup_privacy_note(settings))
        if snap.untraced_calls:
            st.caption(f"作業とのつながりが記録されていないモデルの回答: {snap.untraced_calls}件"
                       "（この記録が始まる前のもの。作業を削除しても残ります）")
        if snap.log_errors:
            st.code("\n".join(snap.log_errors), language=None)
    return snap


def app_banner(settings: Settings, disk_check=disk.disk_state) -> None:
    """Every page run (ui/app.py): pfb_app's messages go to ui.log (once per process), and
    a low disk shows its plain sentence on top of every screen (a warning below
    disk_warn_gb; an error below disk_refuse_gb, when new files are refused)."""
    try:
        logs.setup_ui(settings)
        state = disk_check(settings)
    except Exception:   # noqa: BLE001 - never breaks a screen
        return
    if state.level == "refuse":
        st.error(state.message, icon=":material/hard_drive:")
    elif state.level == "warn":
        st.warning(state.message, icon=":material/hard_drive:")
