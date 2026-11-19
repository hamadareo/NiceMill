"""The roster input: a file (CSV/Excel) or pasted rows, a short preview, and the role table.

`roster_input` draws two tabs, 「ファイルから」 and 「貼り付け」 (a file wins when both are
given), the first rows of what was read, and one row per column with its role picked from a
select (guessed from the header; the teacher can change it). It returns the roster with the
chosen roles, or None when nothing was given, the input could not be read, or the roles are
not usable yet (the reason is shown with st.error).
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import streamlit as st

from pfb_app.domain.vocab import words
from pfb_app.services.roster import (ROLE_LABELS, Roster, empty_roster_message,
                                     guess_place_cameras, parse_roster_file, parse_roster_text,
                                     validate_roster)

FILE_TAB = "ファイルから"
PASTE_TAB = "貼り付け"
PREVIEW_ROWS = 5
COLUMN = "列"
EXAMPLE = "例"
ROLE = "役割"
_ROLE_OF_LABEL = {label: role for role, label in ROLE_LABELS.items()}


def suggestion_caption(suggested: list[str]) -> str:
    return f"{'、'.join(suggested)}の列があると、名前を自動で推定できます"


def _read(key: str, domain: str) -> Roster | None:
    w = words(domain)
    file_tab, paste_tab = st.tabs([FILE_TAB, PASTE_TAB])
    with file_tab:
        upload = st.file_uploader(f"{w.roster}のファイル（CSV・Excel）",
                                  type=["csv", "xlsx", "tsv", "txt"], key=f"{key}_file")
    with paste_tab:
        text = st.text_area(f"{w.roster}を貼り付け（表計算ソフトからコピー、または「10 山田」の形）",
                            key=f"{key}_text", height=160)
    try:
        if upload is not None:
            if text and text.strip():
                st.caption(f"ファイルの{w.roster}を使います（貼り付けた内容は使いません）。")
            return parse_roster_file(upload.name, upload.getvalue())
        if text and text.strip():
            return parse_roster_text(text)
    except ValueError as e:
        st.error(str(e))
    return None


def _examples(r: Roster) -> dict[str, str]:
    return {c: next((row.get(c, "") for row in r.rows if row.get(c)), "") for c in r.columns}


def roles_from_table(rows) -> dict[str, str]:
    """{column: role} from the role table's rows (a cleared or unknown label → ignore)."""
    return {row[COLUMN]: _ROLE_OF_LABEL.get(row[ROLE] or "", "ignore") for row in rows}


def _role_table(key: str, r: Roster) -> dict[str, str]:
    """The role editor; returns {column: role} as chosen."""
    example = _examples(r)
    table = [{COLUMN: c, EXAMPLE: example[c], ROLE: ROLE_LABELS[r.roles.get(c, "ignore")]}
             for c in r.columns]
    # a new roster (other columns) starts a new editor, so old edits never apply to it
    digest = hashlib.sha1("\x1f".join(r.columns).encode()).hexdigest()[:10]
    edited = st.data_editor(
        table, hide_index=True, num_rows="fixed", disabled=[COLUMN, EXAMPLE],
        key=f"{key}_roles_{digest}", width="stretch",
        column_config={ROLE: st.column_config.SelectboxColumn(
            ROLE, options=list(ROLE_LABELS.values()), required=True)})
    return roles_from_table(edited)


def roster_input(key: str, domain: str, suggested: list[str]) -> Roster | None:
    """The roster widget (see the module docstring). `suggested` names the columns that help
    the automatic naming for this sport, shown as a caption."""
    if suggested:
        st.caption(suggestion_caption(suggested))
    r = _read(key, domain)
    if r is None:
        return None
    if not r.rows:
        st.warning(f"{empty_roster_message(domain)}。1行目に見出し（名前・背番号など）を"
                   "入れてください。")
        return None
    st.dataframe(r.rows[:PREVIEW_ROWS], hide_index=True, width="stretch")
    more = f"（最初の{PREVIEW_ROWS}行を表示）" if len(r.rows) > PREVIEW_ROWS else ""
    st.caption(f"{len(r.rows)}人分を読み取りました{more}。"
               f"列の役割が違うときは「{ROLE}」を選び直してください。")
    roles = _role_table(key, r)
    cameras = guess_place_cameras(r.columns)
    chosen = replace(r, roles=roles,
                     place_camera={c: cameras.get(c, "cam1")
                                   for c, role in roles.items() if role == "place"})
    errors = validate_roster(chosen)
    for e in errors:
        st.error(e)
    return None if errors else chosen
