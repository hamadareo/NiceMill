"""Rosters (名簿 / 選手名簿): read an uploaded file (CSV/TSV/text or Excel) or pasted text,
recognise each column's role from its header, and turn the rows into the submission's subjects.

Real-world files are expected: Excel's Shift_JIS (cp932) or UTF-8-with-BOM CSV, UTF-16 tab text,
full-width or padded headers (「Ｎｏ．」, 「背番号 」), title rows and merged headers above the
real header row, blank rows, and numbers stored as floats ("10.0" → "10"). A number keeps its
zeros when it is all zeros ("00" stays distinct from "0"), as in the engine.

Without a recognised header, rows that start with a number (「10 山田」, 「10,山田,FW」) become
背番号 / 名前 / 列3…, and one value per row becomes a list of names (or numbers).

Roles are only guesses until the teacher sees them in the role table (`roster_input`); columns
that match no synonym are kept and shown with the role `ignore`.

Limits: 5 MB of input, 50 MB unpacked for an Excel file, 64 columns and 5000 people.
"""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import PurePath

from pfb_app.domain.models import JobStatus, Stage
from pfb_app.domain.vocab import default_person_name, number_name, words
from pfb_app.pipeline.runner import request_names
from pfb_app.services.review import (ADDED, BusyError, added_people, name_key,
                                     roster_subjects)
from pfb_app.storage.db import SIDE_STAGES, Repo

__all__ = ["HEADER_SYNONYMS", "JOBS_BUSY", "ROLES", "ROLE_LABELS", "Roster", "RosterEntry",
           "apply_roster",
           "empty_roster_message", "guess_place_cameras", "guess_roles", "normalise_header",
           "normalise_number", "parse_roster_file", "parse_roster_text", "roster_entries",
           "roster_text", "validate_roster"]

# apply_roster while a job that uses the people is queued or running (no internal words: the
# wizard shows its own sentence for it, other pages show this one)
JOBS_BUSY = "解析が進んでいるため、名簿は入れ替えられません。終わるまでお待ちください。"
ROLES = ("name", "number", "position", "grid", "batting", "team", "appearance", "place", "ignore")
ROLE_LABELS = {"name": "名前", "number": "背番号", "position": "守備位置・ポジション",
               "grid": "スタート位置", "batting": "打順", "team": "チーム", "appearance": "外見",
               "place": "初期位置", "ignore": "使わない"}
HEADER_SYNONYMS = {
    "name": ("名前", "氏名", "選手名", "name"),
    "number": ("背番号", "番号", "No", "No.", "#", "ゼッケン", "カーナンバー"),
    "position": ("ポジション", "守備位置", "守備番号", "position", "pos"),
    "grid": ("スタート位置", "グリッド", "grid"),
    "batting": ("打順", "打席順", "batting order"),
    "team": ("チーム", "team"),
    "appearance": ("外見", "服装"),
    "place": ("初期位置",),   # also 初期位置（カメラn） for camera n
}
DEFAULT_CAMERA = "cam1"
UNSUPPORTED = "名簿はCSVかExcel（.xlsx）で入れてください"
BLANK_ROLE_ERROR = "名前か背番号の列を選んでください"
REPEAT_ERROR = "同じ役割の列が2つあります"
MAX_BYTES = 5 * 1024 * 1024
MAX_XLSX_UNPACKED = 50 * 1024 * 1024
MAX_COLUMNS = 64
MAX_ROWS = 5000
TOO_BIG = "名簿が大きすぎます（5MBまで）。人の一覧だけにして入れ直してください"
TOO_BIG_XLSX = "Excelの名簿が大きすぎます。名簿のシートだけを別のファイルに保存して入れ直してください"
TOO_MANY_ROWS = f"名簿の行が多すぎます（{MAX_ROWS}人まで）"
UNREADABLE = "名簿を読めませんでした。CSVかExcel（.xlsx）で保存し直してください"
UNREADABLE_XLSX = "Excelの名簿を読めませんでした。.xlsx形式で保存し直してください"
TEXT_EXTENSIONS = (".csv", ".tsv", ".txt")

_ATTR_ROLES = ("number", "position", "grid", "batting", "team", "appearance")
_NUMERIC_ROLES = {"number", "grid", "batting"}     # "１", "1.0" -> "1"
_INVISIBLE = dict.fromkeys((0xFEFF, 0x200B, 0x200C, 0x200D, 0x2060))   # BOM, zero-width
_SYNONYM_ROLE = {unicodedata.normalize("NFKC", s).strip().lower(): role
                 for role, synonyms in HEADER_SYNONYMS.items() for s in synonyms}
_PLACE_CAMERA = re.compile(r"初期位置\s*\(\s*(?:カメラ|camera|cam)\s*(\d+)\s*\)")
_DEDUP_SUFFIX = re.compile(r"\s\(\d+\)$")
_NUMBER = re.compile(r"(?:#|no\.?)?\s*(\d+)(?:\.0+)?", re.I)
_HEADERLESS = re.compile(r"^\s*(#?\d{1,3})\s+(\S.*?)\s*$")
_NUMBER_ONLY = re.compile(r"^\s*#?\d{1,3}\s*$")
_HEADER_SCAN = 10        # the header row is looked for among the first 10 non-blank rows
_DELIMITER_SCAN = 10     # the delimiter is counted over the first 10 non-blank lines


@dataclass
class Roster:
    columns: list[str]
    rows: list[dict[str, str]]
    roles: dict[str, str]              # column -> role (one of ROLES)
    place_camera: dict[str, str]       # place column -> camera (cam1..); missing means cam1


@dataclass
class RosterEntry:
    participant_id: str
    display_name: str
    attrs: dict[str, str]              # role -> value: the raw name, the number/grid/batting
                                       # normalised like the engine's numbers, the others as written
    places: dict[str, str] = field(default_factory=dict)   # camera -> initial place


def empty_roster_message(domain: str | None) -> str:
    return f"{words(domain).roster}に人が見つかりませんでした"


# --- headers and values ---------------------------------------------------------------------

def normalise_header(h: str) -> str:
    """NFKC, trimmed (also of a BOM and zero-width characters), lower-cased, inner runs of
    white space collapsed to one space."""
    s = unicodedata.normalize("NFKC", str(h or "")).translate(_INVISIBLE)
    return " ".join(s.split()).lower()


def _header_role(column: str) -> str:
    key = normalise_header(column)
    role = _SYNONYM_ROLE.get(key) or _SYNONYM_ROLE.get(_DEDUP_SUFFIX.sub("", key))
    if role:
        return role
    return "place" if _PLACE_CAMERA.fullmatch(key) else "ignore"


def guess_roles(columns) -> dict[str, str]:
    """{column: role} from the header synonyms; anything else is `ignore`."""
    return {c: _header_role(c) for c in columns}


def guess_place_cameras(columns) -> dict[str, str]:
    """{place column: camera}: 「初期位置」 → cam1, 「初期位置（カメラn）」 → camn."""
    out = {}
    for c in columns:
        key = _DEDUP_SUFFIX.sub("", normalise_header(c))
        m = _PLACE_CAMERA.fullmatch(key)
        if m:
            out[c] = f"cam{int(m.group(1))}"
        elif _SYNONYM_ROLE.get(key) == "place":
            out[c] = DEFAULT_CAMERA
    return out


def normalise_number(value) -> str:
    """A roster number as the engine writes it: NFKC, "#07"/"No.7" → "7", "10.0" → "10";
    leading zeros dropped except for an all-zero number ("00" stays "00"). Anything that is not
    a plain number is kept as written (NFKC, trimmed); blank → ""."""
    s = unicodedata.normalize("NFKC", str(value or "")).translate(_INVISIBLE).strip()
    m = _NUMBER.fullmatch(s)
    if not m:
        return s
    digits = m.group(1)
    return digits.lstrip("0") or digits


def _cell_text(v) -> str:
    """An Excel cell value as text: floats without a fraction lose ".0", dates are ISO."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.15g}"
    if isinstance(v, datetime):
        return v.date().isoformat() if v.time() == time(0) else v.isoformat(sep=" ")
    if isinstance(v, (date, time)):
        return v.isoformat()
    return str(v)


# --- tables ---------------------------------------------------------------------------------

def _score(row: list[str]) -> int:
    return sum(1 for c in row if c and _header_role(c) != "ignore")


def _empty() -> Roster:
    return Roster(columns=[], rows=[], roles={}, place_camera={})


def _roster(columns: list[str], body: list[list[str]]) -> Roster:
    width = max([len(columns)] + [len(r) for r in body])
    names, keep, seen = [], [], set()
    for i in range(width):
        h = columns[i] if i < len(columns) else ""
        if not h and not any(i < len(r) and r[i] for r in body):
            continue                                   # a blank padding column
        h = h or f"列{i + 1}"
        base, k = h, 2
        while h in seen:
            h, k = f"{base} ({k})", k + 1
        seen.add(h)
        names.append(h)
        keep.append(i)
    rows = []
    for r in body:
        row = {name: (r[i] if i < len(r) else "") for name, i in zip(names, keep)}
        if any(row.values()):
            rows.append(row)
    if len(rows) > MAX_ROWS:
        raise ValueError(TOO_MANY_ROWS)
    return Roster(columns=names, rows=rows, roles=guess_roles(names),
                  place_camera=guess_place_cameras(names))


def _clean(raw) -> list[list[str]]:
    """Trimmed cells (at most MAX_COLUMNS), trailing blank cells dropped."""
    rows = []
    for r in raw:
        cells = [str(c).translate(_INVISIBLE).strip() for c in list(r)[:MAX_COLUMNS]]
        while cells and not cells[-1]:
            cells.pop()
        rows.append(cells)
    return rows


def _header_view(row: list[str]) -> list[str]:
    """A one-cell row may be a header written with spaces (「背番号 名前 ポジション」)."""
    if len(row) == 1:
        split = row[0].split()
        if len(split) > 1 and _score(split) > _score(row):
            return split
    return row


def _table(raw) -> Roster:
    """Find the header row (the most recognised headers among the first non-blank rows, so a
    title or merged group row above it is skipped) and build the roster below it. Without a
    recognised header: number-first rows → 背番号 / 名前 / 列3…; one value per row → a list."""
    rows = _clean(raw)
    nonblank = [i for i, r in enumerate(rows) if r]
    if not nonblank:
        return _empty()
    best = max(nonblank[:_HEADER_SCAN], key=lambda i: (_score(_header_view(rows[i])), -i))
    header = _header_view(rows[best])
    if _score(header) > 0:
        body = rows[best + 1:]
        if header is not rows[best]:          # a space-separated header: split rows alike
            body = [r[0].split(None, len(header) - 1) if len(r) == 1 else r for r in body]
        return _roster(header, body)
    data = [rows[i] for i in nonblank]
    split = []
    for r in data:
        m = _HEADERLESS.match(r[0]) if len(r) == 1 else None
        split.append(list(m.groups()) if m else r)
    if all(len(r) >= 2 and _NUMBER_ONLY.match(r[0]) for r in split):
        return _roster(["背番号", "名前"], split)
    if all(len(r) == 1 for r in data):
        column = "背番号" if all(_NUMBER_ONLY.match(r[0]) for r in data) else "名前"
        return _roster([column], data)
    return _roster(rows[nonblank[0]], rows[nonblank[0] + 1:])


def _delimiter(text: str) -> str:
    """Tab-separated when the first non-blank lines have at least as many tabs as commas (and
    some tab), else comma-separated."""
    lines = [ln for ln in text.splitlines() if ln.strip()][:_DELIMITER_SCAN]
    tabs = sum(ln.count("\t") for ln in lines)
    commas = sum(ln.count(",") for ln in lines)
    return "\t" if tabs and tabs >= commas else ","


def _split(text: str) -> list[list[str]]:
    try:
        return [list(r) for r in csv.reader(io.StringIO(text), delimiter=_delimiter(text))]
    except csv.Error:
        raise ValueError(UNREADABLE) from None


def _decode_text(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp932"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("名簿の文字コードを読めませんでした。UTF-8かShift_JISのCSVで保存してください")


def _read_xlsx(data: bytes) -> list[list[str]]:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            unpacked = sum(i.file_size for i in z.infolist())
    except Exception:
        raise ValueError(UNREADABLE_XLSX) from None
    if unpacked > MAX_XLSX_UNPACKED:
        raise ValueError(TOO_BIG_XLSX)
    import openpyxl

    out, people = [], 0
    try:
        wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
        try:
            if not wb.worksheets:
                return []
            for row in wb.worksheets[0].iter_rows(max_col=MAX_COLUMNS, values_only=True):
                cells = [_cell_text(v) for v in row]
                out.append(cells)
                people += any(c.strip() for c in cells)
                if people > MAX_ROWS + _HEADER_SCAN:
                    break                       # enough to know it is too long (see _roster)
        finally:
            wb.close()
    except Exception:
        raise ValueError(UNREADABLE_XLSX) from None
    return out


def parse_roster_file(name: str, data: bytes) -> Roster:
    """A roster file: `.csv`/`.tsv`/`.txt` (UTF-8 with or without a BOM, then cp932; UTF-16
    too; comma or tab separated) or `.xlsx` (the first sheet). Other types raise
    ValueError(UNSUPPORTED); more than MAX_BYTES raises before anything is read."""
    ext = PurePath(name or "").suffix.lower()
    if ext not in TEXT_EXTENSIONS + (".xlsx",):
        raise ValueError(UNSUPPORTED)
    if len(data) > MAX_BYTES:
        raise ValueError(TOO_BIG)
    if ext == ".xlsx":
        return _table(_read_xlsx(data))
    return _table(_split(_decode_text(data)))


def parse_roster_text(text: str) -> Roster:
    """Pasted rows: tab-separated (spreadsheet copy) or comma-separated, or 「10 山田」 lines."""
    text = (text or "").translate(_INVISIBLE)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError(TOO_BIG)
    if not text.strip():
        return _empty()
    return _table(_split(text))


# --- validation and entries -----------------------------------------------------------------

def _role(r: Roster, column: str) -> str:
    role = r.roles.get(column, "ignore")
    return role if role in ROLES else "ignore"


def _camera(r: Roster, column: str) -> str:
    return r.place_camera.get(column) or DEFAULT_CAMERA


def _camera_index(camera: str) -> int:
    m = re.fullmatch(r"cam(\d+)", camera)
    return int(m.group(1)) if m else 0


def validate_roster(r: Roster) -> list[str]:
    """Japanese messages; empty when the roles can be used. Place columns repeat only when
    they are for the same camera."""
    errors = []
    roles = [_role(r, c) for c in r.columns]
    if "name" not in roles and "number" not in roles:
        errors.append(BLANK_ROLE_ERROR)
    groups: dict[tuple, list[str]] = {}
    for c in r.columns:
        role = _role(r, c)
        if role != "ignore":
            groups.setdefault((role, _camera(r, c) if role == "place" else ""), []).append(c)
    for (role, _), cols in sorted(groups.items(), key=lambda kv: ROLES.index(kv[0][0])):
        if len(cols) > 1:
            errors.append(f"{REPEAT_ERROR}: {ROLE_LABELS[role]}（{'、'.join(cols)}）")
    return errors


def _disambiguate(entries: list[RosterEntry]) -> None:
    """Colliding display names only: different teams → 「#10（チームA）」; else named rows with
    different numbers → 「山田（#10）」; else 「山田」, 「山田（2）」… in row order."""
    groups: dict[str, list[RosterEntry]] = {}
    for e in entries:
        groups.setdefault(e.display_name, []).append(e)
    for name, same in groups.items():
        if len(same) < 2:
            continue
        teams = [e.attrs.get("team", "") for e in same]
        numbers = [e.attrs.get("number", "") for e in same]
        if all(teams) and len(set(teams)) > 1:
            suffixes = teams
        elif all("name" in e.attrs for e in same) and all(numbers) and len(set(numbers)) > 1:
            suffixes = [number_name(n) for n in numbers]
        else:
            suffixes = [""] + [str(k) for k in range(2, len(same) + 1)]
        for e, s in zip(same, suffixes):
            if s:
                e.display_name = f"{name}（{s}）"
    seen: set[str] = set()
    for e in entries:                       # whatever still collides is numbered
        base, k = e.display_name, 2
        while e.display_name in seen:
            e.display_name, k = f"{base}（{k}）", k + 1
        seen.add(e.display_name)


def roster_entries(r: Roster, domain: str | None) -> list[RosterEntry]:
    """One entry per non-blank row (blank: nothing in any used column), ids r1..rn.
    The display name is the name, else 「#10」, else the default 「選手n」/「参加者n」;
    colliding display names get a suffix (`_disambiguate`); attrs["name"] stays the raw name."""
    first: dict[str, str] = {}
    places: list[str] = []
    for c in r.columns:
        role = _role(r, c)
        if role == "place":
            places.append(c)
        elif role != "ignore":
            first.setdefault(role, c)
    places.sort(key=lambda c: _camera_index(_camera(r, c)))
    used = list(first.values()) + places
    entries = []
    for row in r.rows:
        values = {c: str(row.get(c) or "").strip() for c in used}
        if not any(values.values()):
            continue
        n = len(entries) + 1
        attrs = {}
        name = values[first["name"]] if "name" in first else ""
        if name:
            attrs["name"] = name
        for role in _ATTR_ROLES:
            if role in first:
                v = values[first[role]]
                v = normalise_number(v) if role in _NUMERIC_ROLES else v
                if v:
                    attrs[role] = v
        cams = {}
        for c in places:
            if values[c]:
                cams.setdefault(_camera(r, c), values[c])
        if cams:
            attrs["place"] = next(iter(cams.values()))
        display = name or (number_name(attrs["number"]) if attrs.get("number")
                           else default_person_name(domain, n))
        entries.append(RosterEntry(f"r{n}", display, attrs, cams))
    _disambiguate(entries)
    return entries


_TEXT_ROLES = ("name",) + _ATTR_ROLES
_CELL_SPACES = re.compile(r"[\t\r\n]+")


def _text_cell(value) -> str:
    # pasted rows are split on tabs or commas (whichever the lines have more of): keep both out
    return _CELL_SPACES.sub(" ", str(value or "")).replace(",", "、").strip()


def roster_text(subjects: list[dict]) -> str:
    """Roster subjects (apply_roster's, in order) as pasted rows, tab-separated under one
    header per used role (HEADER_SYNONYMS' first word), which parse_roster_text reads back
    into the same entries. Initial places are left out: they describe one video's start.
    A subject without roster attrs (a registered participant) gives its display name.
    Commas in a cell become 「、」. Empty when there are no subjects."""
    rows = []
    for s in subjects:
        attrs = dict(s.get("roster") or {})
        if not attrs:
            attrs = {"name": s["display_name"]}
        rows.append({role: _text_cell(attrs.get(role)) for role in _TEXT_ROLES})
    used = [role for role in _TEXT_ROLES if any(r[role] for r in rows)]
    if not used:
        return ""
    out = io.StringIO()
    writer = csv.writer(out, delimiter="\t", lineterminator="\n")
    writer.writerow([HEADER_SYNONYMS[role][0] for role in used])
    for r in rows:
        writer.writerow([r[role] for role in used])
    return out.getvalue()


# --- applying -------------------------------------------------------------------------------

def _has_pfb(repo: Repo, submission_id: int) -> bool:
    return repo.conn.execute("SELECT 1 FROM stage_runs WHERE submission_id=? AND stage=? LIMIT 1",
                             (submission_id, Stage.pfb.value)).fetchone() is not None


# The content check and frame preparation never read the people, so a roster may replace
# them while those jobs are queued or running (the wizard asks for it during the check).
# names: its runs are made stale by the roster change, and request_names makes new ones;
# 判定支援 (SIDE_STAGES, linecall) never reads the people
_ROSTER_SAFE_STAGES = (Stage.detect.value, Stage.prepare.value, Stage.names.value, *SIDE_STAGES)
_ACTIVE = (JobStatus.queued.value, JobStatus.running.value, JobStatus.waiting_gpu.value)


def _has_people_jobs(repo: Repo, submission_id: int) -> bool:
    """An active job of a stage that uses the subjects or the mapping (all but detect,
    prepare and names)."""
    marks = ",".join("?" * len(_ROSTER_SAFE_STAGES))
    return repo.conn.execute(
        "SELECT 1 FROM jobs WHERE submission_id=? AND status IN (?,?,?)"
        f" AND stage NOT IN ({marks})"
        " LIMIT 1", (submission_id, *_ACTIVE, *_ROSTER_SAFE_STAGES)).fetchone() is not None


def _subject_key(participant_id, display_name, appearance, places, attrs) -> tuple:
    return (participant_id, display_name, appearance or None, dict(places or {}),
            dict(attrs or {}))


ROSTER_ID_MAX = "roster_id_max"   # meta: the highest roster id number (r<n>) ever used
_ROSTER_ID = re.compile(r"^r(\d+)$")


def apply_roster(repo: Repo, submission_id: int, roster: Roster, domain: str | None) -> int:
    """Replace the submission's own subjects with the roster's entries; returns how many.
    People made on the mapping screen (review.added_people) are kept, except those whose name
    is a new roster row's display name (「山田」, 「#10」): the roster row replaces them.

    `place` goes into initial_positions (per camera) and `appearance` into appearance; every
    attr is kept in the subject's roster. Nothing happens when the subjects already are these
    entries. Otherwise: BusyError once the identity is confirmed, a PFB exists, or a job that
    uses the people is queued or running (detect and prepare jobs do not count); ValueError for an invalid or empty roster or a place for a camera the
    submission does not have. The rows get ids on from the highest roster id the submission
    ever used (meta roster_id_max; never reused, so an old mapping cannot name a new row).
    The replacement writes meta["roster_run_floor"] (the highest
    stage run id), so a mapping saved before it no longer counts as current (mapping_form),
    and then asks for names again (runner.request_names) for every tracked camera. An active
    names job does not block it: its run would be built from the old subjects, which the run's
    roster fingerprint shows (naming.current_names_run)."""
    errors = validate_roster(roster)
    if errors:
        raise ValueError("\n".join(errors))
    entries = roster_entries(roster, domain)
    if not entries:
        raise ValueError(empty_roster_message(domain))
    # compared without the ids: a replacement numbers its rows on from the highest id used
    wanted = [_subject_key(None, e.display_name, e.attrs.get("appearance"), e.places, e.attrs)
              for e in entries]
    # the checks and the replacement in one write transaction (or, inside confirm_check's,
    # a savepoint of that one)
    with repo._tx(join=True):
        submission = repo.get_submission(submission_id)
        subjects = repo.list_subjects(submission_id)
        current = [_subject_key(None, s["display_name"], s["appearance"],
                                s["initial_positions"], s["roster"])
                   for s in roster_subjects(subjects, submission["meta"])]
        if current == wanted:
            return len(entries)
        if submission["identity_confirmed"]:
            raise BusyError("人物の確認が済んでいるため、名簿は入れ替えられません。")
        if _has_pfb(repo, submission_id):
            raise BusyError("フィードバックを作り始めているため、名簿は入れ替えられません。")
        if _has_people_jobs(repo, submission_id):
            raise BusyError(JOBS_BUSY)
        meta = dict(submission["meta"] or {})
        cameras = meta.get("cameras")
        if cameras:
            unknown = sorted({c for e in entries for c in e.places} - set(cameras))
            if unknown:
                raise ValueError("名簿の初期位置に、登録されていないカメラがあります: "
                                 + "、".join(f"カメラ{c[3:]}" for c in unknown))
        # people made on the mapping screen stay, unless a new roster row has their name
        new_names = {name_key(e.display_name) for e in entries}
        added = added_people(meta)
        keep = {s["participant_id"] for s in subjects if s["participant_id"] in added
                and name_key(s["display_name"]) not in new_names}
        repo.delete_subjects(submission_id, [s["participant_id"] for s in subjects
                                             if s["participant_id"]
                                             and s["participant_id"] not in keep])
        repo.conn.execute("DELETE FROM subjects WHERE submission_id=? AND participant_id IS NULL",
                          (submission_id,))
        meta[ADDED] = {p: k for p, k in added.items() if p in keep}
        # never reuse a roster id: a mapping saved for the old roster must not name a new row
        base = max([int(meta.get(ROSTER_ID_MAX) or 0)]
                   + [int(m.group(1)) for s in subjects
                      if (m := _ROSTER_ID.match(s["participant_id"] or ""))])
        for k, e in enumerate(entries, start=base + 1):
            e.participant_id = f"r{k}"
        meta[ROSTER_ID_MAX] = base + len(entries)
        for e in entries:
            repo.add_subject(submission_id, e.participant_id, e.display_name,
                             e.attrs.get("appearance"), dict(e.places), dict(e.attrs))
        meta["roster_run_floor"] = repo.conn.execute(
            "SELECT COALESCE(MAX(id), 0) FROM stage_runs").fetchone()[0]
        repo.update_submission_meta(submission_id, meta)
    names_jobs = request_names(repo, submission_id)   # proposals for the new roster
    repo.audit("apply_roster", f"submission:{submission_id}",
               {"subjects": len(entries), "names_jobs": names_jobs})
    return len(entries)
