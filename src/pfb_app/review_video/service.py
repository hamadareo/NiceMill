"""The calls a screen, the worker or the CLI makes. Read only on the database; files go to the
folder the caller names (never into the app's data, never into Git)."""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

from pfb_app.config import Settings
from pfb_app.export.explainer_video import Cancelled, RenderError
from pfb_app.review_video import edit, player
from pfb_app.review_video.collect import build_package
from pfb_app.review_video.model import Package
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

CC_BY_3_0 = "CC BY 3.0 (https://creativecommons.org/licenses/by/3.0/)"


def safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_#" else "_" for c in name)[:40] or "person"


def make_person_video(pkg: Package, subject_id: int, target: Path, *,
                      progress: Callable[[float, str], None] = edit._noop,
                      stop: Callable[[], bool] = lambda: False) -> dict:
    """Render one person's highlight video to `target`, atomically (a part file next to it,
    removed on any failure)."""
    person = pkg.person(subject_id)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}.", suffix=".part.mp4")
    os.close(fd)
    part = Path(name)
    try:
        made = edit.render_highlight(pkg, person, part, progress=progress, stop=stop)
        os.replace(part, target)
        os.chmod(target, 0o600)                  # the owner only (the video of people)
        made["file"] = str(target)
        made["bytes"] = target.stat().st_size
        return made
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def make_all(repo: Repo, settings: Settings, submission_id: int, out_dir: Path, *,
             subject_ids: list[int] | None = None, credit_lines: list[str] | None = None,
             with_player: bool = True, ball_file: str | None = None, claim_cues: str | None = None,
             eval_texts: str | None = None, pose_file: str | None = None,
             estimate_court: bool = True, court_reg: str | None = None) -> dict:
    """Every listed person's highlight video + the player, in `out_dir`."""
    out_dir = Path(out_dir)
    # the package is built (read only) before anything is written: a wrong id leaves no empty folder behind
    pkg = build_package(repo, settings, submission_id, subject_ids=subject_ids,
                        credit_lines=credit_lines, ball_file=ball_file, claim_cues=claim_cues,
                        eval_texts=eval_texts, pose_file=pose_file, estimate_court=estimate_court, court_reg=court_reg)
    result: dict = {"videos": [], "notes": list(pkg.notes), "people": len(pkg.people)}
    if not pkg.people:      # nobody to make anything for: no folder is created
        return result
    out_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(out_dir, 0o700)
    for n, p in enumerate(pkg.people, 1):
        if p.excluded:
            result["videos"].append({"subject_id": p.subject_id, "name": p.name, "label": p.shown(),
                                     "skipped": "選手以外（推定）のため作っていません（プレーヤーでは「選手以外も表示」で見られます）。"})
            continue
        if not p.highlights():
            result["videos"].append({"subject_id": p.subject_id, "name": p.name,
                                     "label": p.shown(), "skipped": "よかった場面の記録がありません。"})
            continue
        file = out_dir / f"{n:02d}_{safe_name(p.name)}.mp4"
        try:
            made = make_person_video(pkg, p.subject_id, file)
        except RenderError as e:
            result["videos"].append({"subject_id": p.subject_id, "name": p.name,
                                     "error": str(e)})
            continue
        made.update(subject_id=p.subject_id, name=p.name, label=p.shown())
        result["videos"].append(made)
    if with_player and pkg.people:
        result["player"] = player.build_player(pkg, out_dir / "player")
    made = {Path(v["file"]).name for v in result["videos"] if v.get("file")}
    old = sorted(f.name for f in out_dir.glob("*.mp4") if f.name not in made and not f.name.startswith("."))
    if old:      # an earlier run's files stay as they were (a folder is never cleaned up here)
        result["notes"].append("出力先に、前回までの動画が残っています（今回は作り直していません）: " + "、".join(old))
    if not with_player and (out_dir / "player" / player.HTML_NAME).exists():
        result["notes"].append("出力先の player/ は前回のままです（今回は作り直していません）。")
    return result


__all__ = ["build_package", "make_person_video", "make_all", "Cancelled", "RenderError"]


def open_readonly(settings: Settings) -> Repo:
    """The database opened read only (SQLite mode=ro): no WAL switch, no schema step, no
    migration, no write of any kind (a write would raise). For the CLI and any batch run."""
    import sqlite3   # noqa: PLC0415
    from pfb_app.storage import files   # noqa: PLC0415
    path = files.db_path(settings)
    if not Path(path).is_file():
        raise FileNotFoundError(f"database not found: {path}")
    conn = sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True, timeout=30,
                           isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return Repo(conn, settings)
