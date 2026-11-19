"""The short excerpt a dashboard jump plays (「この場面から見る」).

st.video reads a local file into memory whole (Streamlit's MemoryMediaFileStorage; it offers
no path or URL streaming without static serving), so the dashboard never hands it a match
video: it cuts an excerpt of about BEFORE_S + AFTER_S seconds around the moment (PyAV stream
copy, no re-encode: it starts on the key frame at or before BEFORE_S seconds ahead of the
moment) and plays that from the moment (`offset`). A clip larger than MAX_CLIP_BYTES is
refused (TOO_BIG) rather than loaded into memory.

Which file it is cut from (`source`), as services.proxy rules for a preview on screen. The
source is probed (size, codec); a size the caller knows from the tracks wins:
- a video that needs a proxy (long side > proxy.MIN_LONG_SIDE, a 4K phone video) or an HEVC
  one (many browsers cannot play it): only its ready 1080p H.264 proxy; none ready -> no
  player (NOT_READY), never the original;
- otherwise the original;
- no local video (moved, or removed with the work's media) -> NO_VIDEO; a file that cannot be
  read -> CANNOT_READ.

Clips live in the group's `handout/clips/` (a heavy folder: services.retention removes it
with the videos, services.erasure with a person's data, a work deletion with the work). The
file name holds the camera, the moment and the source's size and mtime, so a changed source is
cut again; a sidecar .json keeps the offset.

Writing (`clip`): one cut per clip at a time (an flock on `<key>.lock`; a second press waits
and then finds the clip); the cut goes to a unique temp file in the same folder (mkstemp),
is checked (it opens, has a frame and a duration) and only then renamed; on any failure the
temp file is removed, so a broken clip is never cached. A new clip is refused when the disk
is low (ops.disk.ensure_room, as proxy.start) and the folder is kept under MAX_FOLDER_BYTES
by removing the least recently played clips first.
A phone's display rotation travels with the copied stream parameters (FFmpeg's coded side
data), as the proxy keeps it.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from pfb_app.config import Settings
from pfb_app.services import proxy
from pfb_app.storage import files

log = logging.getLogger(__name__)

BEFORE_S, AFTER_S = 5.0, 20.0
MAX_FOLDER_BYTES = 1024 ** 3        # per group
MAX_CLIP_BYTES = 150 * 1024 ** 2    # st.video holds it in memory
BROWSER_UNSAFE = proxy.BROWSER_UNSAFE
NO_VIDEO = "この場面の動画はありません（動画が見つかりません）。"
NOT_READY = "大きな動画のため、見られるようにする準備がまだです。しばらくしてから開いてください。"
CANNOT_READ = "この動画は読めないため、場面の動画は見られません。"
LOW_DISK = "空きが少ないため、場面の動画は作りません。"
TOO_BIG = "この場面の動画は大きすぎるため、ここでは再生しません。"


class ClipRefused(Exception):
    """No clip, for a reason the coach can read (LOW_DISK, TOO_BIG)."""


class ClipFailed(Exception):
    """The cut did not give a playable clip (nothing was kept)."""


@dataclass(frozen=True)
class Probe:
    width: int
    height: int
    codec: str
    duration_s: float | None


def clips_dir(settings: Settings, aid: int, sid: int) -> Path:
    return files.submission_dir(settings, aid, sid, create=False) / "handout" / "clips"


def probe(path: Path) -> Probe | None:
    """The first video stream's size, codec and the duration; None when it cannot be read."""
    import av   # noqa: PLC0415
    try:
        with av.open(str(path)) as c:
            v = c.streams.video[0]
            dur = None
            if v.duration is not None and v.time_base:
                dur = float(v.duration * v.time_base)
            elif c.duration:
                dur = c.duration / 1_000_000
            return Probe(int(v.codec_context.width), int(v.codec_context.height),
                         str(v.codec_context.name or "").lower(), dur)
    except Exception:   # noqa: BLE001 - any unreadable file is just "cannot read"
        return None


def source(settings: Settings, video: Path | None,
           size: tuple[int, int] | None = None) -> tuple[Path | None, str | None]:
    """(the file to cut from, None) or (None, why there is no player)."""
    if video is None or not Path(video).is_file():
        return None, NO_VIDEO
    found = probe(Path(video))
    if found is None:
        return None, CANNOT_READ
    width, height = size if size else (found.width, found.height)
    if proxy.needs_playable(width, height, found.codec):
        ready = proxy.ready(settings, video)
        return (ready, None) if ready is not None else (None, NOT_READY)
    return Path(video), None


def _key(src: Path, camera: str, at_s: float) -> str:
    st = src.stat()
    h = hashlib.sha1(f"{src.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:10]
    safe = "".join(c if c.isalnum() else "_" for c in camera)[:20] or "cam"
    return f"{safe}_{int(round(at_s * 10)):07d}_{h}"


@contextmanager
def _locked(path: Path):
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _cached(dst: Path, meta: Path) -> float | None:
    try:
        if dst.is_file():
            offset = float(json.loads(meta.read_text(encoding="utf-8"))["offset"])
            os.utime(dst)                      # played now: kept longest by the cap
            return offset
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _estimate(src: Path) -> int:
    """About how many bytes a clip of `src` takes (its bitrate times the clip's length)."""
    size = src.stat().st_size
    found = probe(src)
    if found is None or not found.duration_s or found.duration_s <= 0:
        return size
    return min(size, int(size / found.duration_s * (BEFORE_S + AFTER_S + 10)))


def clip(folder: Path, camera: str, src: Path, at_s: float, *,
         settings: Settings | None = None) -> tuple[Path, float]:
    """The excerpt of `src` around `at_s` (seconds in the video) and the moment's place in it
    (seconds from the clip's start). Cached in `folder`. ClipRefused: low disk (`settings`
    given) or too big to play; ClipFailed (or the cut's own error): nothing playable."""
    src = Path(src)
    key = _key(src, camera, at_s)
    dst, meta = folder / f"{key}.mp4", folder / f"{key}.json"
    offset = _cached(dst, meta)
    if offset is not None:
        return dst, offset
    folder.mkdir(parents=True, exist_ok=True)
    with _locked(folder / f"{key}.lock"):
        offset = _cached(dst, meta)            # cut by the press we waited for
        if offset is not None:
            return dst, offset
        if settings is not None:
            from pfb_app.ops import disk   # noqa: PLC0415
            try:
                disk.ensure_room(settings, _estimate(src))
            except disk.DiskLow as e:
                raise ClipRefused(LOW_DISK) from e
        fd, name = tempfile.mkstemp(dir=folder, prefix=f".{key}.", suffix=".part.mp4")
        os.close(fd)
        part = Path(name)
        try:
            start = _cut(src, part, max(at_s - BEFORE_S, 0.0), at_s + AFTER_S)
            _check(part)
            if part.stat().st_size > MAX_CLIP_BYTES:
                raise ClipRefused(TOO_BIG)
            offset = max(at_s - start, 0.0)
            tmp = folder / f".{key}.json.tmp"
            tmp.write_text(json.dumps({"offset": offset, "at_s": at_s}), encoding="utf-8")
            tmp.replace(meta)
            os.replace(part, dst)              # the clip last: a clip always has its sidecar
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    _prune(folder, keep=dst)
    return dst, offset


def _check(path: Path) -> None:
    """ClipFailed unless the clip opens, decodes a frame and has a duration."""
    import av   # noqa: PLC0415
    try:
        with av.open(str(path)) as c:
            v = c.streams.video[0]
            frame = next(iter(c.decode(v)), None)
            dur = (float(v.duration * v.time_base) if v.duration is not None and v.time_base
                   else (c.duration or 0) / 1_000_000)
    except Exception as e:   # noqa: BLE001
        raise ClipFailed(str(e)) from e
    if frame is None or not dur or dur <= 0:
        raise ClipFailed(f"no frame or no duration in {path.name}")


def _prune(folder: Path, keep: Path) -> None:
    """Remove the least recently played clips (with their sidecar .json and their .lock)
    until the folder's clips take at most MAX_FOLDER_BYTES; `keep` (just made) stays."""
    clips = []
    for p in folder.glob("*.mp4"):
        if p.name.startswith("."):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        clips.append((st.st_mtime, st.st_size, p))
    total = sum(size for _, size, _ in clips)
    for _, size, p in sorted(clips, key=lambda c: c[0]):
        if total <= MAX_FOLDER_BYTES:
            break
        if p == keep:
            continue
        try:
            p.unlink()
            p.with_suffix(".json").unlink(missing_ok=True)
            p.with_suffix(".lock").unlink(missing_ok=True)
            total -= size
        except OSError:
            log.warning("場面の動画を消せませんでした: %s", p)


def _cut(src: Path, dst: Path, start_s: float, end_s: float) -> float:
    """Stream-copy [key frame <= start_s, end_s] of `src` to `dst`, timestamps from the
    key frame's decode time. Returns that source time (the clip's 0)."""
    import av   # noqa: PLC0415 - only a jump needs PyAV

    with av.open(str(src)) as inp:
        vin = inp.streams.video[0]
        ain = inp.streams.audio[0] if inp.streams.audio else None
        if start_s > 0 and vin.time_base:
            inp.seek(int(start_s / vin.time_base), stream=vin, backward=True, any_frame=False)
        with av.open(str(dst), "w", format="mp4", options={"movflags": "+faststart"}) as out:
            vout = out.add_stream_from_template(vin)
            aout = out.add_stream_from_template(ain) if ain is not None else None
            first: float | None = None     # the first key frame's time (pts)
            base = 0.0                     # its decode time: the clip's 0
            for packet in inp.demux(*([vin] + ([ain] if ain is not None else []))):
                if packet.dts is None or packet.pts is None:
                    continue
                t = float(packet.pts * packet.time_base)
                if packet.stream is vin:
                    if first is None:
                        if not packet.is_keyframe:
                            continue
                        first = t
                        base = float(packet.dts * packet.time_base)
                    elif t > end_s and packet.is_keyframe:
                        break
                    elif t > end_s + 2:
                        break
                    target = vout
                else:
                    if first is None or t < first or t > end_s:
                        continue
                    target = aout
                shift = int(round(base / packet.time_base))
                packet.pts -= shift
                packet.dts -= shift
                packet.stream = target
                out.mux(packet)
    if first is None:
        raise ClipFailed(f"no key frame to start a clip of {src.name} at {start_s}")
    return base
