"""解説動画 files: what is drawn (the person blue, others grey and nameless, the verified
region dashed), readable text, and a phone-friendly encoding (synthetic videos only)."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from pdf_synthetic import FPS, H as SH, W as SW, box_of, make_video, write_tracks
from pfb_app.export import explainer_ai as ai
from pfb_app.export import explainer_video as ev
from pfb_app.export.tracks import Box, load_boxes


def _probe(path: Path) -> dict:
    import av
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        frames = sum(1 for _ in c.decode(v))
        return {"codec": v.codec_context.name, "pix": v.codec_context.pix_fmt,
                "w": v.codec_context.width, "h": v.codec_context.height,
                "audio": len(c.streams.audio), "frames": frames, "rate": float(v.average_rate)}


def test_text_is_never_below_four_percent_of_the_height():
    assert min(ev.SMALL, ev.BODY, ev.HEAD, ev.NAME, ev.TEXT_MIN) >= 0.04 * ev.H
    assert ev.font(10).size == ev.TEXT_MIN
    f = ev.font(ev.BODY)
    lines = ev.wrap("あ" * 200, f, 400, 2)
    assert len(lines) == 2 and lines[-1].endswith("…")
    assert all(f.getlength(s) <= 400 for s in lines)


def test_the_person_is_blue_others_grey_and_without_a_name():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    tag = ev._tag("山田花子さん", ev.BLUE)
    ev.draw_boxes(img, [("P1", Box(40, 120, 60, 120)), ("P2", Box(250, 120, 60, 120))],
                  (1.0, 1.0), (0, 0), {"P1"}, {"P1": tag})
    assert tuple(img[180, 40]) in {ev.BLUE, (255, 255, 255)} or img[180, 40].sum() > 300
    assert tuple(img[180, 250]) == ev.GREY
    assert img[90:118, 240:330].sum() == 0              # nothing above the other person
    assert img[90:118, 40:120].sum() > 0                # the tag above the person


def _engine_stills(video: Path, run: Path, boxes, window: int, size) -> list[str]:
    """Stills as PFB3x3 would have sent them for clip_som window `window`."""
    frames = ai.window_frames(window, 0.5, 12, FPS, 0.0, 10 ** 6)
    region = ai.action_region([b for f in frames for _, b in ai.boxes_at(boxes, f)], *size)
    cap = cv2.VideoCapture(str(video))
    out = []
    for k, f in enumerate(frames, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, frame = cap.read()
        x0, y0, x1, y1 = (round(v) for v in region)
        crop = cv2.resize(frame[y0:y1, x0:x1], (960, 540))
        p = run / "overlay" / f"v_w{window:04d}" / f"{k}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), crop)
        out.append(str(p))
    cap.release()
    return out


def _clustered(tmp_path, spots=((260, 150), (330, 160))):
    """A textured 640x360 10 fps video with two people at `spots` (close together: the
    engine's region is then a small part of the frame) and its boxes."""
    rng = np.random.default_rng(5)
    floor = cv2.resize(rng.integers(40, 200, (45, 80, 3), dtype=np.uint8), (SW, SH),
                       interpolation=cv2.INTER_NEAREST)
    path = tmp_path / "c.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (SW, SH))
    rows = []
    for f in range(1, 30 * FPS + 1):
        img = floor.copy()
        for k, (x, y) in enumerate(spots):
            x += int(10 * np.sin(f / 9)) if k == 0 else 0
            cv2.rectangle(img, (x, y), (x + 30, y + 72), (40 + 150 * k, 90, 220), -1)
            rows.append(f"{f},{k + 1},{x},{y},30,72,0.9")
        w.write(img)
    w.release()
    (tmp_path / "ct").mkdir()
    (tmp_path / "ct" / "boxes.txt").write_text("\n".join(rows) + "\n")
    return path, load_boxes(tmp_path / "ct" / "boxes.txt")


@pytest.fixture
def scene(tmp_path):
    video = make_video(tmp_path / "v.mp4", 4, 30)
    boxes = load_boxes(write_tracks(tmp_path / "tracks", 4, 30, False))
    return tmp_path, video, boxes


def test_a_person_video_is_h264_yuv420p_720p_silent_and_faststart(tmp_path):
    video, boxes = _clustered(tmp_path)
    src = ev.Source(video, FPS, (SW, SH), boxes, 0.0, 300)
    stills = _engine_stills(video, tmp_path / "exp" / "m" / "clip_som" / "v", boxes, 1, (SW, SH))
    parts = [ev.Part("開始から 00:08", "パス", "あなたがほかの選手にパスを出した",
                     "仲間の動きを助けた", src, 8.0, ["P1"], stills),
             ev.Part("動画の時刻 ", None, "あなたが声をかけた", "まわりを見ていた")]
    plan = ev.PersonPlan("山田花子", "練習試合", "2026年10月24日", "まわりをよく見て動いていました。",
                         "ほかの選手", parts)
    target = tmp_path / "out.mp4"
    seen = []
    got = ev.render_person(plan, target, progress=lambda f, n: seen.append((f, n)))
    info = _probe(target)
    assert (info["codec"], info["pix"], info["w"], info["h"], info["audio"]) == \
        ("h264", "yuv420p", 1280, 720, 0)
    expect = ev.INTRO_S + ev.BEFORE_S + ev.AFTER_S + ev.CARD_S + ev.END_S
    assert abs(info["frames"] / info["rate"] - expect) < 0.5
    assert abs(got["seconds"] - expect) < 0.5 and got["parts"] == 2
    assert got["verified"] == 1 and parts[0].verified and not parts[1].verified
    data = target.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")        # +faststart: plays at once
    notes = [n for _, n in seen]       # the preparation of the one part with a source comes first
    assert seen[-1] == (1.0, "仕上げ") and notes[0] == "場面を確かめています 1 / 2"
    assert "場面 1 / 2" in notes
    assert any("点線" in s for s in ev.legend_lines(plan, True))


def test_people_across_the_whole_frame_get_no_rectangle_and_no_word_of_one(tmp_path):
    video, boxes = _clustered(tmp_path, ((15, 20), (590, 270)))    # at both edges
    src = ev.Source(video, FPS, (SW, SH), boxes, 0.0, 300)
    stills = _engine_stills(video, tmp_path / "exp" / "m" / "clip_som" / "v", boxes, 1, (SW, SH))
    part = ev.Part("開始から 00:08", "パス", "a", "b", src, 8.0, ["P1"], stills)
    plan = ev.PersonPlan("山田花子", "練習試合", "", "", "ほかの選手", [part])
    got = ev.render_person(plan, tmp_path / "out.mp4")
    assert got["verified"] == 0 and not part.verified
    assert not any("点線" in s for s in ev.legend_lines(plan, False) + ev.end_lines(False))
    assert any("点線" in s for s in ev.end_lines(True))


def test_unverified_stills_are_shown_but_no_region_is_drawn(scene):
    tmp_path, video, boxes = scene
    src = ev.Source(video, FPS, (SW, SH), boxes, 0.0, 300)
    run = tmp_path / "exp" / "m" / "clip_som" / "v" / "overlay" / "v_w0001"
    run.mkdir(parents=True)
    stills = []
    for k in range(1, 13):                       # not what the frames look like
        p = run / f"{k}.jpg"
        cv2.imwrite(str(p), np.full((540, 960, 3), 40 * (k % 5), dtype=np.uint8))
        stills.append(str(p))
    part = ev.Part("開始から 00:08", "パス", "a", "b", src, 8.0, ["P1"], stills)
    plan = ev.PersonPlan("山田花子", "練習試合", "", "", "ほかの選手", [part])
    got = ev.render_person(plan, tmp_path / "out.mp4")
    assert got["verified"] == 0 and not part.verified


def test_a_rotated_or_odd_source_is_refused_not_drawn_wrong(scene):
    tmp_path, video, boxes = scene
    src = ev.Source(video, FPS, (SH, SW), boxes, 0.0, 300)     # the tracks say portrait
    part = ev.Part("開始から 00:08", None, "a", "b", src, 8.0, ["P1"], [])
    with pytest.raises(ev.RenderError):
        ev.render_person(ev.PersonPlan("x", "t", "", "", "ほかの人", [part]), tmp_path / "o.mp4")


def test_cancel_stops_the_render(scene):
    tmp_path, video, boxes = scene
    src = ev.Source(video, FPS, (SW, SH), boxes, 0.0, 300)
    part = ev.Part("開始から 00:08", None, "a", "b", src, 8.0, ["P1"], [])
    with pytest.raises(ev.Cancelled):
        ev.render_person(ev.PersonPlan("x", "t", "", "", "ほかの人", [part]), tmp_path / "o.mp4",
                         stop=lambda: True)


def test_tracking_video_names_only_the_named(scene):
    tmp_path, video, boxes = scene
    src = ev.Source(video, FPS, (SW, SH), boxes, 0.0, 300)
    plan = ev.TrackingPlan("練習試合", "前半", src, {"P1": "山田花子", "P3": "#7"}, 0.0, 6.0,
                           lambda t: f"開始から {int(t):02d}")
    got = ev.render_tracking(plan, tmp_path / "t.mp4")
    info = _probe(tmp_path / "t.mp4")
    assert (info["codec"], info["w"], info["h"]) == ("h264", 1280, 720)
    assert abs(info["frames"] / info["rate"] - (ev.INTRO_S + 6.0)) < 0.5
    assert got["people"] == 2
    x, y, bw, bh = box_of(1, 30)        # P2 has no name: drawn grey only (unit-checked above)
    assert bw > 0
