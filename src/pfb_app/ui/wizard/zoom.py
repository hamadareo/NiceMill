"""「大きく見る」: every picture of the wizard can be opened large.

`picture` draws an image (its alt text says who or what it shows: 「3番の人」, never a tracking
label) and under it a button that names it (「3番の人を大きく見る」), which opens a large
dialog with the whole image, as wide as the dialog (Streamlit sends images wider than 1460 px
scaled down to 1460 px, so a very large frame is not shown pixel for pixel). The button is a
real button (Tab, Enter) and the dialog closes with Esc, its ×, or a click outside.

The open picture is kept in the session (`OPEN`) and shown again on every run until the dialog
is dismissed, so a live refresh that reruns the whole page does not close it.

Images are file reads only (no database).
"""
from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

from pfb_app.ui import keyed

ZOOM = "大きく見る"
LABEL = "{name}を大きく見る"
OPEN = "_pfb_zoom_open"     # the key of the picture whose large view is open
VIEW_KEY = "pfb_zoom_view"
_MD = re.compile(r"([\\`*_~\[\]()<>#|:$])")


def image_bytes(path) -> bytes | None:
    """The file's bytes; None when it cannot be read."""
    try:
        return Path(path).read_bytes()
    except OSError:
        return None


def _title(text: str) -> str:
    """The dialog title as plain text (its Markdown characters escaped)."""
    return _MD.sub(r"\\\1", text) or ZOOM


def _open(key: str) -> None:
    st.session_state[OPEN] = key


def close() -> None:
    """on_dismiss of the large view: it stays closed on the next runs."""
    st.session_state.pop(OPEN, None)


def _view(data: bytes, alt: str, caption: str | None) -> None:
    with keyed.container(key=VIEW_KEY):
        st.image(data, caption=caption, alt=alt, width="stretch")


def show_large(data: bytes, alt: str, caption: str | None = None) -> None:
    """The large view (st.dialog, width "large") of one image."""
    st.dialog(_title(alt), width="large", on_dismiss=close)(_view)(data, alt, caption)


def picture(path, *, alt: str, key: str, width="stretch", caption: str | None = None,
            name: str | None = None) -> bool:
    """The image and 「<name>を大きく見る」 under it (name: the alt text unless given); its
    large view while it is open. False (nothing drawn) when the file is missing."""
    data = image_bytes(path)
    if data is None:
        return False
    st.image(data, caption=caption, alt=alt, width=width)
    st.button(LABEL.format(name=name or alt), key=f"pfb_zoom_{key}", icon=":material/zoom_in:",
              type="tertiary", on_click=_open, args=(key,))
    if st.session_state.get(OPEN) == key:
        show_large(data, alt, caption)
    return True
