"""Bigger pictures and 「大きく見る」 (UX quick wins C): the person cards (step 3 and the people
panel of step 5), the evidence frames and the people sheet. Synthetic images only."""
import os
import re

import streamlit as st

from pfb_app.services import auto_people
from pfb_app.ui.wizard import people_grid, people_panel, theme, zoom
from test_people_panel import make_sheet, set_track
from test_wizard_people_grid import LABELS, P_LABEL, scene
from test_wizard_ui_basic import app, env, ok, video_assignment  # noqa: F401
from test_wizard_ui_review_pdf import reviewable, subject


def zooms(at):
    return [b for b in at.button if b.label.endswith(zoom.ZOOM)]


def walk(node):
    yield node
    for child in getattr(node, "children", {}).values():
        yield from walk(child)


def dialogs(at):
    return [n.proto.dialog for n in walk(at._tree)
            if type(n).__name__ == "Block" and n.proto.WhichOneof("type") == "dialog"]


def alts(at):
    return [img.alt for el in at.get("image") for img in el.proto.imgs]


def spy_images(monkeypatch):
    calls = []
    real = st.image

    def image(data, *args, **kwargs):
        calls.append(kwargs)
        return real(data, *args, **kwargs)
    monkeypatch.setattr(st, "image", image)
    return calls


# --- the shared helper -----------------------------------------------------------------------

def test_image_bytes_read_the_file_as_it_is_now(tmp_path):
    path = tmp_path / "a.png"
    path.write_bytes(b"one")
    first = path.stat().st_mtime_ns
    assert zoom.image_bytes(path) == b"one"
    path.write_bytes(b"two")
    os.utime(path, ns=(first, first))                 # same mtime and size: no stale copy
    assert zoom.image_bytes(path) == b"two"
    assert zoom.image_bytes(tmp_path / "missing.png") is None


def test_dismissing_the_large_view_forgets_it(monkeypatch):
    state = {zoom.OPEN: "crop_1_P3"}
    monkeypatch.setattr(st, "session_state", state)
    zoom.close()
    assert zoom.OPEN not in state


def test_dialog_titles_are_plain_text():
    assert zoom._title("3番の人") == "3番の人"
    assert zoom._title("a*b_[c]") == r"a\*b\_\[c\]"


# --- step 3: the mapping grid ----------------------------------------------------------------

def test_cards_are_three_per_row_with_big_crops():
    assert people_grid.COLUMNS == 3
    css = theme.theme_css()
    rule = re.search(r'"st-key-pfb_person_"\], \[class\*="st-key-pfb_pcard_"\]\) img \{([^}]*)\}',
                     css).group(1)
    height = float(re.search(r"height: ([\d.]+)rem", rule).group(1))
    assert height >= 17 and "width: 100%" in rule        # at least twice the old 8.5rem
    nophoto = re.search(r"\.pfb-person__nophoto \{([^}]*)\}", css).group(1)
    assert f"height: {height:g}rem" in nophoto           # cards keep their place


def test_every_grid_picture_has_alt_text_and_a_large_view(env, monkeypatch):
    calls = spy_images(monkeypatch)
    aid, (sid,), runs = scene(env)
    tr = runs[sid]
    at = ok(app(session={"assignment_id": aid}))
    shown = alts(at)
    for n in range(1, len(LABELS) + 1):
        assert f"{n}番の人" in shown
    assert all(a and not P_LABEL.search(a) for a in shown)
    labels = [b.label for b in zooms(at)]
    assert len(labels) == len(LABELS) + 1 + 2              # crops, sheet, strips
    assert len(set(labels)) == len(labels)                 # each button says what it opens
    assert {"3番の人を大きく見る", "人物一覧を大きく見る", "左のチームを大きく見る"} <= set(labels)
    assert all(not P_LABEL.search(label) and len(label) <= 16 for label in labels)
    assert {"width": "stretch"}.items() <= next(
        c for c in calls if (c.get("alt") or "").startswith("人物一覧")).items()  # full width
    assert not dialogs(at)
    at.button(key=f"pfb_zoom_crop_{tr.id}_P3").click()
    ok(at.run())
    (dialog,) = dialogs(at)
    assert dialog.title == "3番の人" and dialog.width == dialog.DialogWidth.LARGE
    assert alts(at).count("3番の人") == 2                    # the card and the large view
    assert dialog.id                                         # dismissing it reruns (zoom.close)
    ok(at.run())                         # a live tick reruns the whole page: still open
    (dialog,) = dialogs(at)
    assert dialog.title == "3番の人" and at.session_state[zoom.OPEN] == f"crop_{tr.id}_P3"
    del at.session_state[zoom.OPEN]      # what zoom.close does when the dialog is dismissed
    ok(at.run())
    assert not dialogs(at)


def test_a_missing_sheet_file_says_so(env):
    settings, repo = env
    aid, (sid,), runs = scene(env, names=False)
    (settings.data_dir / f"sheet_{sid}.png").unlink()
    at = ok(app(session={"assignment_id": aid}))
    sheet = next(e for e in at.expander if e.label == people_grid.SHEET_MORE)
    assert [i.value for i in sheet.info] == [people_grid.NO_SHEET]
    assert not [b for b in zooms(at) if (b.key or "").startswith("pfb_zoom_sheet_")]


# --- step 5: the people panel and the evidence frames ----------------------------------------

def test_people_panel_has_two_big_cards_per_row_and_a_large_view(env):
    settings, repo = env
    assert people_panel.COLUMNS == 2
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    set_track(repo, sid, sheet=str(make_sheet(settings, sid)),
              sheet_geometry={"P1": [10, 10, 50, 80]})
    pa = subject(repo, sid, "Pa")
    assert pa["id"] in auto_people.person_crops(repo, settings, sid)
    at = ok(app(session={"assignment_id": aid}))
    key = f"pfb_zoom_person_{pa['id']}"
    assert pa["display_name"] in alts(at)
    assert at.button(key=key).label == f"{pa['display_name']}を大きく見る"
    at.button(key=key).click()
    ok(at.run())
    (dialog,) = dialogs(at)
    assert dialog.title == pa["display_name"] and dialog.width == dialog.DialogWidth.LARGE


def test_evidence_frames_are_320_wide_with_a_large_view(env, monkeypatch):
    calls = spy_images(monkeypatch)
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid)
    at = ok(app(session={"assignment_id": aid}))
    frames = [c for c in calls if "フレーム" in (c.get("alt") or "")]
    assert len(frames) == 3 and all(c["width"] >= 320 for c in frames)
    assert all("00:12" in c["alt"] and not P_LABEL.search(c["alt"]) for c in frames)
    frame_zooms = [b for b in zooms(at) if (b.key or "").startswith("pfb_zoom_frame_")]
    assert [b.label for b in frame_zooms] == [f"場面1の{k}枚目を大きく見る" for k in (1, 2, 3)]
    frame_zooms[0].click()
    ok(at.run())
    (dialog,) = dialogs(at)
    assert "フレーム 1/" in dialog.title and dialog.width == dialog.DialogWidth.LARGE
