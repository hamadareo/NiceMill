"""The drop zone with the resumable uploader (ui/wizard/big_upload.py), when the server has
the upload endpoint (ui/serve.py). Synthetic files only."""
from pathlib import Path

import pytest

from pfb_app.services import chunked_upload as cu
from pfb_app.services import intake
from pfb_app.ui import upload_api
from pfb_app.ui.wizard import big_upload, step_drop
from test_wizard_ui_basic import app, env, ok, screen_text  # noqa: F401 (env)


def test_the_upload_types_are_the_intakes():
    assert cu.VIDEO_EXTENSIONS == intake.VIDEO_EXTENSIONS
    assert cu.DOCUMENT_EXTENSIONS == intake.DOCUMENT_EXTENSIONS


def finished_upload(settings, name=b"x", filename="a.mp4"):
    token = cu.issue_token(settings)
    up = cu.create(settings, token, filename, len(name))
    cu.append(settings, token, up.id, 0, name)
    return token, up


def test_only_really_finished_uploads_become_picks(env):
    settings, _ = env
    token, up = finished_upload(settings)
    other = cu.issue_token(settings)
    half = cu.create(settings, token, "b.mp4", 4)
    entries = [{"id": up.id, "token": token, "name": "a.mp4"},
               {"id": up.id, "token": other},                  # not its owner
               {"id": half.id, "token": token},               # not finished
               {"id": "../../x", "token": token}, {"bogus": 1}]
    picks = big_upload.finished(settings, entries)
    assert [p.id for p in picks] == [up.id] and picks[0].name == "a.mp4"


def test_a_pick_is_handed_over_once(env):
    settings, _ = env
    token, up = finished_upload(settings, b"video")
    pick = big_upload.finished(settings, [{"id": up.id, "token": token}])[0]
    saved = []
    paths = step_drop.save_videos(settings, [pick], False, saved)
    assert saved == [] and paths[0].read_bytes() == b"video"   # not a copy of the drop zone's
    assert paths[0].parent == settings.data_dir / "uploads"
    assert pick.claim(settings) == paths[0]                    # after the registration
    with pytest.raises(cu.UnknownUpload):
        pick.claim(settings)


def test_a_document_pick_is_read_and_removed(env):
    settings, _ = env
    token, up = finished_upload(settings, b"%PDF-1.4", "memo.pdf")
    pick = big_upload.finished(settings, [{"id": up.id, "token": token}])[0]
    path = cu.status(settings, token, up.id).path
    assert pick.read_and_remove(settings) == b"%PDF-1.4"
    assert not Path(path).exists()


def test_the_drop_zone_uses_the_resumable_uploader_when_served(env, monkeypatch):
    settings, _ = env
    monkeypatch.setattr(upload_api, "MOUNTED", True)
    at = ok(app(session={"assignment_id": None, "step": 1, "submission_id": None}))
    text = screen_text(at)
    assert step_drop.BIG_NOTE.format(mb=settings.max_upload_mb) in text
    assert not at.get("file_uploader")
    assert cu.check_token(settings, at.session_state[big_upload.TOKEN]) is not None
    primary = next(b for b in at.button if b.proto.type == "primary")
    assert primary.label == step_drop.PRIMARY and primary.disabled


def test_without_the_endpoint_streamlits_uploader_stays(env):
    at = ok(app(session={"assignment_id": None, "step": 1, "submission_id": None}))
    assert at.get("file_uploader")


def test_the_uploader_is_drawn_in_theme_tokens():
    import re
    from pfb_app.ui.wizard import theme
    css = big_upload._css()
    colours = set(re.findall(r"#[0-9A-Fa-f]{3,8}\b", css))
    assert colours <= {c.upper() for c in theme.TOKENS.values()} | set(theme.TOKENS.values())
    assert "var(--pfb-board" in css and "var(--pfb-line" in css
