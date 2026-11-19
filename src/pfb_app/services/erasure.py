"""「この人のデータを消す」: a deletion request for one person.

A person is the chosen one plus everyone linked to the same 成長の記録 (other works). Nothing
changes while any of their groups has a queued, running or GPU-waiting step (`Busy`).

What goes, for each of them:
- a video's person: the person row stays as an anonymous placeholder (ERASED_NAME, no
  appearance, roster or positions, excluded, `erased_at`), so the people selection never
  makes the same tracked person again (auto_people.auto_identity) and nobody can bring them
  back (`is_evaluated`, Repo.set_excluded). Their feedback texts (every version) and jobs go;
  in the group's shared analysis their scenes go (the behavior events of their participant
  id, the judgments citing them, the label's entries of the names runs); their face crops,
  the group's PDF picture cache, their saved handout files and the work's saved evaluator PDF
  (services.kept) go; the group's 解説動画 (services.explainer: every person's and the
  追跡の様子, which names everyone) go and its queued or running video jobs are cancelled;
- a document's person is the document: its group goes as a whole (rows, the document copy
  and page images);
- the group's cached model answers that no other group uses (call_log_owner, as
  services.deletion does for a work), the growth links, the declined suggestions, and a
  record left without links.

Their name also goes from the group's identity mappings and run warnings. Known limit: while
the video is kept, a fresh tracking of it (new track run, new labels) may find them again as
a new person; a re-selection on the same track keeps their placeholder.

Not removed: the video (everyone is in it) until the work or its videos are deleted, PDFs
already handed out, the nightly backups until they rotate out. Files are removed only inside
the work's own folder (path-checked; links lose only themselves). The audit log records
ids and counts, never a name.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pfb_app.domain.models import Stage
from pfb_app.services import kept
from pfb_app.storage import files
from pfb_app.storage.db import Repo

log = logging.getLogger(__name__)

ERASED_NAME = "消した人"
TITLE = "この人のデータを消す"
BUSY = "この人の作業はまだ解析中です。解析が終わってから消せます。"
_ACTIVE = ("queued", "running", "waiting_gpu")


class Busy(Exception):
    def __init__(self, message: str = BUSY):
        super().__init__(message)


@dataclass(frozen=True)
class Item:
    subject_id: int
    submission_id: int
    assignment_id: int
    title: str
    name: str
    kind: str          # "video" | "document"
    texts: int         # feedback text versions
    approved: bool


@dataclass(frozen=True)
class Plan:
    items: list[Item] = field(default_factory=list)
    busy: bool = False
    linked: bool = False   # linked in 成長の記録
    engine_skipped: list[str] = field(default_factory=list)   # titles (engine_targets)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _group(repo: Repo, subject_id: int) -> list[int]:
    """The subject and everyone linked to the same 成長の記録, by work then id."""
    person = repo.growth_person_of(subject_id)
    ids = {subject_id} | ({l["subject_id"] for l in repo.growth_links(person)}
                         if person is not None else set())
    rows = repo.conn.execute(
        f"SELECT s.id FROM subjects s JOIN submissions b ON s.submission_id = b.id"
        f" WHERE s.id IN ({','.join('?' * len(ids))}) AND s.erased_at IS NULL"
        f" ORDER BY b.assignment_id, s.id", sorted(ids)).fetchall()
    return [r[0] for r in rows]


def _busy(repo: Repo, submission_ids) -> bool:
    ids = sorted(set(submission_ids))
    if not ids:
        return False
    return repo.conn.execute(
        f"SELECT 1 FROM jobs WHERE submission_id IN ({','.join('?' * len(ids))})"
        f" AND status IN ({','.join('?' * len(_ACTIVE))}) LIMIT 1",
        (*ids, *_ACTIVE)).fetchone() is not None


def _item(repo: Repo, subject_id: int) -> Item:
    s = repo.get_subject(subject_id)
    sub = repo.get_submission(s["submission_id"])
    work = repo.get_assignment(sub["assignment_id"])
    texts = repo.conn.execute("SELECT COUNT(*) FROM stage_runs WHERE subject_id=? AND stage=?",
                              (subject_id, Stage.pfb.value)).fetchone()[0]
    return Item(subject_id, sub["id"], work["id"], work["title"], s["display_name"],
                work["kind"], texts, s["approval_status"] == "approved")


def plan(repo: Repo, subject_id: int) -> Plan:
    """What `erase` would remove (KeyError: no such person)."""
    repo.get_subject(subject_id)
    items = [_item(repo, sid) for sid in _group(repo, subject_id)]
    skipped = []
    for i in items:
        pid = repo.get_subject(i.subject_id)["participant_id"]
        if i.kind != "document" and pid and engine_targets(
                repo, i.assignment_id, i.submission_id,
                _labels_of(repo, i.submission_id, pid))[1]:
            skipped.append(i.title)
    return Plan(items, _busy(repo, (i.submission_id for i in items)),
                repo.growth_person_of(subject_id) is not None, list(dict.fromkeys(skipped)))


def what_goes(p: Plan) -> list[str]:
    """The confirm area's lines, in plain words."""
    lines = []
    for i in p.items:
        what = ("書類のファイル・ページの画像・フィードバックの文章" if i.kind == "document"
                else "名前・フィードバックの文章・解析の中のこの人の場面とその画像・顔の切り抜き")
        lines.append(f"「{i.title}」の{i.name}さん：{what}、保存してあるPDFのこの人のページ")
    if p.linked:
        lines.append("成長の記録のつながり（ほかに人が残らない記録は、記録ごと）")
    return lines


def engine_note(p: Plan) -> list[str]:
    """What stays because the analysis was redone after the people were numbered again."""
    return [f"「{t}」は人の番号をつけ直したあとの解析のため、解析の作業用の記録にあるこの人の分は、"
            "動画を消すまで残ります。" for t in p.engine_skipped]


KEEPS = ("動画そのものと、全員がいっしょに写る画像（ほかの人の場面にも使う場面の画像を含みます）・"
         "動いた位置の記録・話した言葉の書き起こしは、"
         "作業を削除するか、動画を自動で消す設定で消えるまで残ります（ほかの人も写っています）。",
         "判定支援の判定材料は、この人の番号の場面の足もとの写真と短い動画を消します。"
         "場面の一覧、コーチの確認、チームの背番号で記録した場面の写真と動画は、"
         "作業を削除するか、動画を自動で消す設定で消えるまで残ります。",
         "すでに渡したPDFは、受け取った人の手元に残ります。")


def backup_keep_line(settings=None) -> str:
    """The backups keep a copy for as long as the rotation does: backup_keep days, then the
    weekly ones (ops.backup.max_age_days), 42 days at most with the defaults."""
    from pfb_app.config import Settings   # noqa: PLC0415
    from pfb_app.ops import backup         # noqa: PLC0415
    days = backup.max_age_days(settings if settings is not None else Settings())
    return f"毎晩のバックアップには、最長{days}日（入れ替わるまで）残ります。"


def keeps(settings=None) -> tuple[str, ...]:
    return (*KEEPS, backup_keep_line(settings))
MENTIONS = ("ほかの人の文章は自動では直しません。必要なら、その作業の「確認して承認」で直してください。")


def mentions(repo: Repo, p: Plan) -> list[str]:
    """Other people's approved texts that contain one of the person's names (disclosed in the
    confirm area; never edited automatically)."""
    names = {i.name.strip() for i in p.items
             if len(i.name.strip()) >= 2 and i.name.strip() != ERASED_NAME}
    mine = {i.subject_id for i in p.items}
    if not names:
        return []
    out = []
    for sid, other, title, output in repo.conn.execute(
            "SELECT s.id, s.display_name, a.title, r.output FROM subjects s"
            " JOIN stage_runs r ON r.id = s.approved_run_id JOIN submissions b"
            " ON s.submission_id = b.id JOIN assignments a ON b.assignment_id = a.id"
            " WHERE s.approval_status = 'approved' AND s.erased_at IS NULL"
            " ORDER BY a.id, s.id").fetchall():
        if sid in mine:
            continue
        text = str((json.loads(output or "{}") or {}).get("pfb") or "")
        for name in sorted(names):
            if name in text:
                out.append(f"「{title}」の{other}さんの文章に「{name}」が出てきます。")
                break
    return out


def people(repo: Repo) -> list[dict]:
    """Everyone who can be chosen: each person of a work not yet erased (a video's person with
    a participant id, a document's person), by work then name; `linked`: how many works their
    成長の記録 holds."""
    rows = repo.conn.execute(
        "SELECT s.id, s.display_name, a.id, a.title, a.kind, l.person_id FROM subjects s"
        " JOIN submissions b ON s.submission_id = b.id JOIN assignments a"
        " ON b.assignment_id = a.id LEFT JOIN growth_links l ON l.subject_id = s.id"
        " WHERE s.erased_at IS NULL AND (a.kind = 'document' OR s.participant_id IS NOT NULL)"
        " ORDER BY a.id, s.display_name, s.id").fetchall()
    counts = {r[0]: r[1] for r in repo.conn.execute(
        "SELECT person_id, COUNT(*) FROM growth_links GROUP BY person_id")}
    return [{"subject_id": r[0], "name": r[1], "assignment_id": r[2], "title": r[3],
             "kind": r[4], "linked": counts.get(r[5], 1) if r[5] is not None else 1}
            for r in rows]


def label(person: dict) -> str:
    more = person["linked"] - 1
    return f"{person['name']}（{person['title']}" + (f"ほか{more}件" if more > 0 else "") + "）"


# --- the removal ----------------------------------------------------------------------------

def _labels_of(repo: Repo, sid: int, pid: str) -> dict[str, set[tuple[int, str]]]:
    """{camera: {(track run id, label)}} the current mapping of each track run gives to the
    participant (a label means a person only within its track run; failed runs and mappings
    corrected since do not count)."""
    out: dict[str, set[tuple[int, str]]] = {}
    cameras = [r[0] for r in repo.conn.execute(
        "SELECT DISTINCT camera FROM stage_runs WHERE submission_id=? AND stage=?"
        " AND camera IS NOT NULL", (sid, Stage.identity.value))]
    from pfb_app.services.auto_people import current_mappings   # noqa: PLC0415
    for camera in cameras:
        for tid, mapping in current_mappings(repo, sid, camera).items():
            out.setdefault(camera, set()).update((tid, lb) for lb, p in mapping.items()
                                                 if p == pid)
    return out


def _update_output(repo: Repo, run_id: int, output: dict) -> None:
    repo.conn.execute("UPDATE stage_runs SET output=? WHERE id=?",
                      (json.dumps(output, ensure_ascii=False), run_id))


def _scrub(repo: Repo, sid: int, pid: str, labels: dict[str, set[str]]) -> list[str]:
    """Remove the participant's scenes from the group's shared runs; returns the paths of
    their crops the names runs named and of the engine's pictures of their scenes (removed
    later, path-checked)."""
    gone: set[str] = set()
    shots: list[str] = []
    for run_id, output in repo.conn.execute(
            "SELECT id, output FROM stage_runs WHERE submission_id=? AND stage=?",
            (sid, Stage.behavior.value)).fetchall():
        out = json.loads(output or "null")
        if not isinstance(out, dict) or not isinstance(out.get("events"), list):
            continue
        mine = {e.get("event_id") for e in out["events"]
                if isinstance(e, dict) and e.get("participant") == pid}
        if not mine:
            continue
        gone |= mine
        if isinstance(out.get("frames"), dict):   # the engine's pictures of their scenes
            shots += [str(p) for k, v in out["frames"].items() if k in mine
                      for p in (v if isinstance(v, list) else []) if isinstance(p, str)]
        out["events"] = [e for e in out["events"] if e.get("event_id") not in mine]
        for key in ("frames", "source_text"):
            if isinstance(out.get(key), dict):
                out[key] = {k: v for k, v in out[key].items() if k not in mine}
        _update_output(repo, run_id, out)
    for run_id, output in repo.conn.execute(
            "SELECT id, output FROM stage_runs WHERE submission_id=? AND stage=?",
            (sid, Stage.contribution.value)).fetchall():
        out = json.loads(output or "null")
        if not gone or not isinstance(out, dict) or not isinstance(out.get("judgments"), list):
            continue
        kept_judgments = [j for j in out["judgments"]
                          if not (isinstance(j, dict) and j.get("ref_id") in gone)]
        if len(kept_judgments) != len(out["judgments"]):
            out["judgments"] = kept_judgments
            _update_output(repo, run_id, out)
    crops: list[str] = []
    for run_id, camera, output in repo.conn.execute(
            "SELECT id, camera, output FROM stage_runs WHERE submission_id=? AND stage=?",
            (sid, Stage.names.value)).fetchall():
        out = json.loads(output or "null")
        if not isinstance(out, dict):
            continue
        mine = {lb for tid, lb in labels.get(camera) or set() if tid == out.get("track_run_id")}
        if not mine:
            continue
        changed = False
        for key, value in out.items():
            if isinstance(value, dict) and mine & set(value):
                if key == "crops":
                    crops += [str(value[lb]) for lb in mine if isinstance(value.get(lb), str)]
                out[key] = {k: v for k, v in value.items() if k not in mine}
                changed = True
        if changed:
            _update_output(repo, run_id, out)
    # a picture of a window shows everyone in it: it goes only when no remaining event of
    # any behavior run of the group still uses it (review R26)
    used: set[str] = set()
    for (output,) in repo.conn.execute(
            "SELECT output FROM stage_runs WHERE submission_id=? AND stage=?",
            (sid, Stage.behavior.value)).fetchall():
        out = json.loads(output or "null")
        frames = out.get("frames") if isinstance(out, dict) else None
        if isinstance(frames, dict):
            used |= {str(p) for v in frames.values() if isinstance(v, list) for p in v}
    shots = [p for p in dict.fromkeys(shots) if p not in used]
    return crops + shots


def _drop_calls(repo: Repo, sid: int) -> int:
    """The group's cached model answers that no other group uses (deletion's rule)."""
    from pfb_app.services import deletion   # noqa: PLC0415
    hashes = [r[0] for r in repo.conn.execute(
        "SELECT request_hash FROM call_log_owner WHERE submission_id=?", (sid,))]
    repo.conn.execute("DELETE FROM call_log_owner WHERE submission_id=?", (sid,))
    return deletion._drop_unowned_calls(repo, hashes)


def _scrub_names(repo: Repo, sid: int, pid: str | None, name: str) -> None:
    """The person's name out of the group's runs: their display_name in the identity mappings
    and every warning that names them (review._checked_mapping writes both)."""
    for run_id, stage, output, warnings in repo.conn.execute(
            "SELECT id, stage, output, warnings FROM stage_runs WHERE submission_id=?",
            (sid,)).fetchall():
        out = json.loads(output or "null")
        changed = False
        if pid and stage == Stage.identity.value and isinstance(out, dict):
            for p in out.get("participants") or []:
                if isinstance(p, dict) and p.get("participant_id") == pid \
                        and p.get("display_name") is not None:
                    p["display_name"] = None
                    changed = True
        if changed:
            _update_output(repo, run_id, out)
        notes = json.loads(warnings or "[]")
        if name and isinstance(notes, list) and any(isinstance(w, str) and name in w
                                                    for w in notes):
            notes = [w.replace(name, ERASED_NAME) if isinstance(w, str) else w for w in notes]
            repo.conn.execute("UPDATE stage_runs SET warnings=? WHERE id=?",
                              (json.dumps(notes, ensure_ascii=False), run_id))


def _erase_video_person(repo: Repo, item: Item, subject: dict) -> dict:
    pid = subject["participant_id"]
    labels = _labels_of(repo, item.submission_id, pid) if pid else {}
    engine, _skipped = engine_targets(repo, item.assignment_id, item.submission_id, labels)
    crops = _scrub(repo, item.submission_id, pid, labels) if pid else []
    crops += linecall_files(repo, item.submission_id, labels)
    _scrub_names(repo, item.submission_id, pid, (subject["display_name"] or "").strip())
    texts = repo.conn.execute("DELETE FROM stage_runs WHERE subject_id=?",
                              (item.subject_id,)).rowcount
    repo.conn.execute("DELETE FROM jobs WHERE subject_id=?", (item.subject_id,))
    # the group's practice tips were made from this person's text among others: they go too
    repo.conn.execute("DELETE FROM stage_runs WHERE submission_id=? AND stage=?"
                      " AND subject_id IS NULL", (item.submission_id, Stage.tips.value))
    repo.conn.execute(
        "UPDATE subjects SET display_name=?, appearance=NULL, roster='{}',"
        " initial_positions='{}', excluded=1, erased_at=?, approval_status='none',"
        " approved_run_id=NULL, approved_at=NULL, approved_auto=0 WHERE id=?",
        (ERASED_NAME, _now(), item.subject_id))
    return {"texts": texts, "labels": labels, "crops": crops, "engine": engine}


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.parent.resolve().is_relative_to(root) and path.name not in ("", ".", "..")
    except (OSError, RuntimeError):
        return False


def _unlink(path: Path, root: Path) -> int:
    if not _inside(path, root):
        return 0
    try:
        if path.is_symlink() or path.is_file():
            path.unlink()
            return 1
    except OSError:
        log.exception("ファイルを消せませんでした: %s", path)
    return 0


def _rmtree(path: Path, root: Path) -> None:
    """A real folder inside `root` goes with everything in it; a link loses only itself."""
    try:
        if path.is_symlink():
            _unlink(path, root)
        elif path.is_dir() and path.resolve().is_relative_to(root) and path.resolve() != root:
            shutil.rmtree(path)
    except OSError:
        log.exception("フォルダを消せませんでした: %s", path)


def _video_files(settings, item: Item, found: dict) -> int:
    root = kept.work_folder(settings, item.assignment_id)
    if root is None:
        return 0
    sub_dir = root / "submissions" / str(item.submission_id)
    n = 0
    people = sub_dir / "people"
    for camera, pairs in found["labels"].items():
        for tid, lb in pairs:   # auto_people._sheet_crop's name
            n += _unlink(people / f"{camera}_{lb}_t{tid}.jpg", root)
    for crop in found["crops"]:
        n += _unlink(Path(crop), root)
    _rmtree(sub_dir / "handout" / "pictures", root)   # a cache of everyone's pictures
    _rmtree(sub_dir / "handout" / "clips", root)      # the dashboard's excerpts (moment_clip)
    from pfb_app.services import explainer   # noqa: PLC0415 - 解説動画: every video of the group
    explainer.remove_files(settings, item.assignment_id, item.submission_id)
    for rows, sheet_dir, labels in found.get("engine") or []:
        n += _engine_files(root, rows, sheet_dir, set(labels))
    return n + kept.drop_person(settings, item.assignment_id, item.subject_id)


_VIDEO_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
ERASED_EVENT = {"label": None, "erased": True}   # keeps the others' event numbers


def _label_token(labels: set[str]) -> re.Pattern:
    return re.compile(r"(?:^|[_.-])(?:" + "|".join(map(re.escape, sorted(labels)))
                      + r")(?:[_.-]|$)")


def _rewrite(path: Path, root: Path, text: str) -> None:
    if _inside(path, root) and path.is_file() and not path.is_symlink():
        tmp = path.with_name(f".{path.name}.erase.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


def engine_targets(repo: Repo, aid: int, sid: int, labels: dict[str, set[tuple[int, str]]]
                   ) -> tuple[list[tuple[str | None, str | None, list[str]]], list[str]]:
    """Per camera, the engine files that hold the person, each with the labels that mean the
    person in THAT file: (the rows file of the camera's current behavior run, None, the labels
    the mapping of that run's upstream track run gives them) and (None, the current track
    run's sheet folder, the labels of the current track run). A label means a person only
    within its track run, so the rows are edited with the labels of the track run they were
    made from (also after a re-track that left the person unmapped on the newest one); when
    the person has no mapping on that track run, the camera is listed as skipped (its rows may
    hold them under a label we cannot tell). Review R27, final review HIGH 1."""
    from pfb_app.export.sheets import _rows_file   # noqa: PLC0415
    targets, skipped = [], []
    for camera, pairs in sorted(labels.items()):
        track = repo.latest_run(sid, Stage.track, camera)
        current = track is not None and track.status == "ok"
        mine = sorted({lb for tid, lb in pairs if current and tid == track.id})
        if mine:
            out = track.output or {}
            vid = str(out.get("video_id") or "")
            sheet = out.get("sheet")
            sheet_dir = (str(Path(sheet).parent) if isinstance(sheet, str)
                         and _VIDEO_ID.fullmatch(vid) and Path(sheet).parent.name == vid
                         and Path(sheet).parent.parent.name == "sheets" else None)
            if sheet_dir is not None:
                targets.append((None, sheet_dir, mine))
        behavior = repo.latest_run(sid, Stage.behavior, camera)
        if behavior is None:
            continue
        upstream = set(behavior.upstream_run_ids or [])
        theirs = sorted({lb for tid, lb in pairs if tid in upstream})
        if behavior.status != "ok" or not theirs:
            skipped.append(camera)
            continue
        found = _rows_file(behavior.output or {}, None, repo, aid, sid)
        if found is not None:
            targets.append((str(found), None, theirs))
    return targets, skipped


def linecall_files(repo: Repo, sid: int, labels: dict[str, set[tuple[int, str]]]) -> list[str]:
    """The person's own 判定支援 pictures and clips (final review LOW 8): per camera, the
    crop, overlay and clip of each episode of the latest linecall run whose label the mapping
    of that run's track run gives the person. Only an engine-tracked run (episodes.json
    source "engine": its labels are the app's P-labels) is matched; TrackID3x3 jersey labels
    (Aチーム #6) cannot be tied to the person and stay (KEEPS says so). Paths relative to the
    run's linecall folder only; the file phase checks they stay inside the work's folder."""
    out: list[str] = []
    for camera, pairs in sorted(labels.items()):
        run = repo.latest_run(sid, Stage.linecall, camera)
        if run is None or run.status != "ok" or not run.output:
            continue
        mine = {lb for tid, lb in pairs if tid in set(run.upstream_run_ids or [])}
        path = run.output.get("episodes_json")
        if not mine or not isinstance(path, str):
            continue
        base = Path(path).parent
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict) or doc.get("source") != "engine":
            continue
        for ep in doc.get("episodes") or []:
            if not isinstance(ep, dict) or str(ep.get("label")) not in mine:
                continue
            for key in ("crop", "overlay", "clip"):
                rel = ep.get(key)
                if isinstance(rel, str) and rel and not Path(rel).is_absolute() \
                        and ".." not in Path(rel).parts:
                    out.append(str(base / rel))
    return out


def _engine_files(root: Path, rows: str | None, sheet_dir: str | None, labels: set[str]) -> int:
    """The engine's own per-person records of these labels: their events in the one rows file
    are blanked in place (the others keep their numbers: 成長の記録 reads them), and the files
    of the current sheet folder named after a label go. Position records (tracks/) stay. All
    inside the work's folder, links never followed."""
    if not labels:
        return 0
    n = 0
    path = Path(rows) if rows else None
    if path is not None and _inside(path, root) and path.is_file() and not path.is_symlink():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            lines = []
        out, changed = [], False
        for line in lines:
            try:
                row = json.loads(line) if line.strip() else None
            except ValueError:
                row = None
            if isinstance(row, dict) and isinstance(row.get("events"), list) and any(
                    isinstance(e, dict) and e.get("label") in labels for e in row["events"]):
                row["events"] = [dict(ERASED_EVENT) if isinstance(e, dict)
                                 and e.get("label") in labels else e for e in row["events"]]
                line, changed = json.dumps(row, ensure_ascii=False), True
            out.append(line)
        if changed:
            _rewrite(path, root, "\n".join(out) + "\n")
            n += 1
    folder = Path(sheet_dir) if sheet_dir else None
    if folder is not None and not folder.is_symlink() and folder.is_dir() \
            and _inside(folder, root):
        token = _label_token(labels)
        for f in sorted(folder.rglob("*")):
            if not f.is_symlink() and f.is_file() and _inside(f, root) and token.search(f.stem):
                n += _unlink(f, root)
    return n


ERASING = "erasing"   # submissions meta: the files of an erasure are being edited (a stamp)


def _mark(repo: Repo, submission_ids, on: bool) -> None:
    """Set (or clear) the group's ERASING stamp: pipeline.runner.enqueue_once queues nothing
    for it meanwhile (Repo.erasing: a stamp older than ERASING_STALE_SEC no longer counts)."""
    for sid in sorted(set(submission_ids)):
        if on:
            repo.conn.execute(f"UPDATE submissions SET meta = json_set(meta, '$.{ERASING}', ?)"
                              " WHERE id=?", (_now(), sid))
        else:
            repo.conn.execute(f"UPDATE submissions SET meta = json_remove(meta, '$.{ERASING}')"
                              " WHERE id=?", (sid,))


def erase(repo: Repo, settings, subject_id: int) -> Plan:
    """Remove the person's data (see the module doc); returns what was removed.
    Busy: one of their groups is analysed (nothing changed). KeyError: no such person.
    While the files are edited, nothing is queued for their groups (ERASING)."""
    with repo._tx():
        p = plan(repo, subject_id)
        if p.busy:
            raise Busy()
        _mark(repo, (i.submission_id for i in p.items if i.kind != "document"), True)
        from pfb_app.services import explainer   # noqa: PLC0415
        repo.explainer_cancel([i.submission_id for i in p.items], explainer.CANCELLED)
        done = []
        for item in p.items:
            subject = repo.get_subject(item.subject_id)
            repo.unlink_growth(item.subject_id)
            repo.conn.execute("DELETE FROM growth_declined WHERE subject_a=? OR subject_b=?",
                              (item.subject_id, item.subject_id))
            calls = _drop_calls(repo, item.submission_id)
            if item.kind == "document":
                repo.conn.execute("DELETE FROM submissions WHERE id=?", (item.submission_id,))
                found = {"texts": item.texts}
            else:
                found = _erase_video_person(repo, item, subject)
            # what the file phase removes, so a stop half way can be finished (ids, labels
            # and paths inside the work's folder only: never a name)
            repo.audit(ERASED, f"subject:{item.subject_id}",
                       {"assignment": item.assignment_id, "submission": item.submission_id,
                        "kind": item.kind, "texts": found["texts"], "calls_removed": calls,
                        "files": _files_record(found)})
            done.append((item, found))
        repo.prune_growth_people()
    try:
        for item, found in done:
            _file_phase(repo, settings, item, found)
    finally:
        _mark(repo, (i.submission_id for i, _ in done if i.kind != "document"), False)
    return p


ERASED = "erase_person"            # audit: the rows are gone (with what the files phase does)
FILES_DONE = "erase_files_done"    # audit: the files of that erasure are gone too
GRACE_SEC = 600                    # an erasure without FILES_DONE older than this stopped


def _files_record(found: dict) -> dict:
    return {"labels": sorted([c, t, lb] for c, pairs in (found.get("labels") or {}).items()
                             for t, lb in pairs),
            "crops": list(found.get("crops") or []),
            "engine": [[r, d, list(lbs)] for r, d, lbs in found.get("engine") or []]}


def _found_from(record: dict) -> dict:
    labels: dict[str, set[tuple[int, str]]] = {}
    for c, t, lb in record.get("labels") or []:
        labels.setdefault(str(c), set()).add((int(t), str(lb)))
    return {"labels": labels, "crops": [str(c) for c in record.get("crops") or []],
            "engine": [(r, d, list(lbs)) for r, d, lbs in record.get("engine") or []]}


def _file_phase(repo: Repo, settings, item: Item, found: dict) -> None:
    """The files of one erased person (idempotent: a second run finds nothing more), then
    FILES_DONE."""
    if item.kind == "document":
        root = kept.work_folder(settings, item.assignment_id)
        if root is not None:
            _rmtree(root / "submissions" / str(item.submission_id), root)
        kept.drop_person(settings, item.assignment_id, item.subject_id)
    else:
        _video_files(settings, item, found)
    repo.audit(FILES_DONE, f"subject:{item.subject_id}", {})
    log.info("人のデータを消しました subject=%s assignment=%s", item.subject_id,
             item.assignment_id)


def unfinished(repo: Repo, older_than_sec: float = GRACE_SEC) -> list[dict]:
    """Erasures whose rows went but whose files were not all removed (a stop half way), older
    than `older_than_sec` (a running one is left alone): [{"subject_id", detail...}]."""
    stamp = (datetime.now(timezone.utc).timestamp() - older_than_sec)
    since = datetime.fromtimestamp(stamp, timezone.utc).isoformat(timespec="microseconds")
    out = []
    for target, detail in repo.conn.execute(
            "SELECT target, detail FROM audit_log l WHERE action=? AND ts <= ? AND NOT EXISTS"
            " (SELECT 1 FROM audit_log d WHERE d.action=? AND d.target = l.target"
            "  AND d.id > l.id) ORDER BY id", (ERASED, since, FILES_DONE)).fetchall():
        tail = str(target).split(":", 1)[-1]
        if tail.isascii() and tail.isdigit():
            out.append({"subject_id": int(tail), **json.loads(detail or "{}")})
    return out


def finish_unfinished(repo: Repo, settings) -> int:
    """The worker's pass: finish the file phase of every erasure that stopped half way (also
    the person's saved PDF and phone page), and clear their groups' erasing stamps."""
    done = 0
    for rec in unfinished(repo):
        log.warning("途中で止まった人のデータの消去を続けます subject=%s", rec["subject_id"])
        try:
            item = Item(rec["subject_id"], int(rec["submission"]), int(rec["assignment"]),
                        "", "", str(rec.get("kind") or "video"), 0, False)
            _file_phase(repo, settings, item, _found_from(rec.get("files") or {}))
            _mark(repo, [item.submission_id], False)
            done += 1
        except Exception:   # noqa: BLE001 - one bad record never stops the worker
            log.exception("途中で止まった消去を続けられませんでした subject=%s",
                          rec.get("subject_id"))
    return done


STOPPED_NOTE = ("途中で止まった「この人のデータを消す」がありました。残りのファイルは、"
                "ワーカーが自動で消します。")
