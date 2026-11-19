"""A 1080p proxy of a big (4K) phone video (phone-first lane, audit #4).

Which stage reads which file (decided per stage from the P2 report, 2026-10-22):

| stage | file | why |
|---|---|---|
| detect (content check) | original | 8 sampled frames on the CPU; cheap, and its people sizes feed the people choice |
| prepare (frames, 854x480) | proxy when ready and SDR, else original | the same small frames, decoded from 1080p instead of 4K |
| transcribe (paper method) | proxy when ready | its audio is the original's, copied packet for packet |
| track, names, behavior, line calls (engine) | original, always | P2: the engine decodes on NVDEC and downsizes itself; boxes stay in original pixels for the number crops and line calls; ByteTrack and the linker react to small pixel changes (raw IoU .95, linked label agreement .75), so a re-encoded input would move P-labels |
| a preview on screen | proxy only | never stream a 4K original to a phone |

HDR (HLG/PQ) sources: PyAV's plain reformat does no tone mapping (its dst_color_trc only tags
the frame), so frames would be about 25 grey levels darker than OpenCV's tone-mapped ones (P2).
Since the 解説動画 lane (2026-10-24) an HDR source gets a BT.709 SDR proxy all the same, made
through libavfilter: `scale` to the proxy size, then `scale` with out_transfer / out_primaries /
out_color_matrix bt709 (FFmpeg 8's colour-managed swscale, which is what OpenCV's backend gets:
about 0.6 grey levels from cv2 on an iPhone HLG frame, measured 10/24), tagged BT.709. It is for
players only: `ready(for_analysis=True)` still never returns an HDR source's proxy.

A proxy is also wanted for an HEVC video of any size (`needs_playable`: many browsers cannot
play HEVC). The child writes its progress to `<key>.progress` (0..1) and, when it fails, a
`failed` sidecar for that source (size and mtime): a failure caused by the file is not
tried again until the file changes; a failure for a passing reason (a full disk, no memory,
too many open files, a busy device: `transient`) is tried again after RETRY_PAUSE_SEC, at
most MAX_ATTEMPTS times in all, and only then stays (`state`: ready / making / waiting /
failed / none; "failed" holds through the pause).

Rotation: a phone's portrait video is stored landscape with a display rotation. The proxy
carries the same rotation (PyAV `set_display_rotation`), and `ready` refuses a proxy whose
rotation (as OpenCV reads and applies it) differs from the source's, so frames never turn.

Disk: `start` makes no proxy when about a quarter of the source's size more would take the
free space below disk_refuse_gb (ops.disk.ensure_room, open uploads counted); the original is
used instead.

How: `start` launches `python -m pfb_app.services.proxy SRC DST` in its own session at low
priority (nice 10, `ionice -c3`) and returns at once; nothing waits for it. The children take
one lock (`proxies/.run.lock`), so one transcode runs at a time. The child decodes with PyAV
(threads), scales to a height of 1080 (even width, aspect kept), encodes H.264 (libx264
veryfast, CRF 23, yuv420p, +faststart, so a browser can play it from the start) and copies
the audio stream; timestamps are kept, so a time in the proxy is the same time in the
original. It writes `<key>.part.mp4` and renames it when complete, with a sidecar
`<key>.json` naming the source (size and mtime), the sizes and how long it took. `ready`
returns the proxy only when it is complete and the source is unchanged. Files live in
`data_dir/proxies/` (never next to a video in the inbox). They go with their work
(`remove_for`, from services.deletion.delete_work; a transcode still queued or running for
one is stopped first, after checking that the pid in `<key>.pid` is really our child for that
file, so its part file frees the disk at once), and `maybe_sweep` (hourly at most, when
proxies start, when an upload session starts and when a work is deleted) removes proxies whose
source is gone or changed.
"""
import errno
import fcntl
import hashlib
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

from pfb_app.config import Settings

log = logging.getLogger(__name__)

TARGET_HEIGHT = 1080
MIN_LONG_SIDE = 2560          # bigger than this (e.g. 3840x2160) gets a proxy
CRF = "23"
PRESET = "veryfast"
THREADS = 4                   # of the server's 12 shared cores, at nice 10
BROWSER_UNSAFE = frozenset({"hevc", "h265"})   # codecs many browsers cannot play
PROGRESS_EVERY = 30           # frames between progress writes
# the tone mapping of an HDR source (after the plain resize): BT.709 SDR, limited range
TONEMAP = "out_color_matrix=bt709:out_range=tv:out_transfer=bt709:out_primaries=bt709"
SWEEP_EVERY_SEC = 3600        # maybe_sweep: at most this often per process
RETRY_PAUSE_SEC = 1800        # a transient failure is tried again after this long ...
MAX_ATTEMPTS = 3              # ... at most this many failures in all, then it stays failed
# errors that say "not now" rather than "not this file" (PyAV's errors carry an errno too)
TRANSIENT_ERRNOS = frozenset({errno.ENOSPC, errno.ENOMEM, errno.EMFILE, errno.ENFILE,
                              errno.EAGAIN, errno.EBUSY, errno.EINTR, errno.EDQUOT})
RUN_LOCK = ".run.lock"        # one transcode at a time on the machine (the children queue)
SIZE_SHARE = 4                # a 1080p proxy of a 4K original: about a quarter of its bytes
_last_sweep = 0.0
HDR_TRC = {"arib-std-b67", "smpte2084", "bt2020-10", "bt2020-12"}   # transfer names of HDR
HDR_TRC_CODES = {14, 15, 16, 18}   # the same as FFmpeg's AVColorTransferCharacteristic
BT2020_PRIMARIES = 9               # AVCOL_PRI_BT2020


def is_transient(exc: BaseException) -> bool:
    """The failure is about the moment (disk full, memory, descriptors, a busy device), not
    about the file: trying again later may work."""
    return isinstance(exc, (MemoryError, TimeoutError, InterruptedError, BlockingIOError)) \
        or getattr(exc, "errno", None) in TRANSIENT_ERRNOS


def _retry_allowed(meta: dict, now: float) -> bool:
    """A `failed` sidecar that may be tried again now: transient, past its pause, and fewer
    than MAX_ATTEMPTS failures so far."""
    return bool(meta.get("transient")) and int(meta.get("attempts") or 0) < MAX_ATTEMPTS \
        and now >= float(meta.get("retry_at") or 0)


def is_hdr(trc, primaries) -> bool:
    """HLG/PQ/BT.2020 colour, from PyAV's values (ints in PyAV 17, names in some builds)."""
    def code(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    if code(trc) in HDR_TRC_CODES or code(primaries) == BT2020_PRIMARIES:
        return True
    text = f"{trc} {primaries}".lower()
    return any(t in text for t in HDR_TRC) or "2020" in text


def proxies_dir(settings: Settings) -> Path:
    return settings.data_dir / "proxies"


def _fingerprint(src: Path) -> dict:
    st = src.stat()
    return {"source": str(src.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def key_of(src: Path) -> str:
    fp = _fingerprint(src)
    return hashlib.sha1(f"{fp['source']}|{fp['size']}|{fp['mtime_ns']}".encode()).hexdigest()[:20]


def paths(settings: Settings, src: Path) -> tuple[Path, Path, Path]:
    """(proxy, sidecar json, part file) of `src`."""
    k = key_of(Path(src))
    d = proxies_dir(settings)
    return d / f"{k}.mp4", d / f"{k}.json", d / f"{k}.part.mp4"


def needs_proxy(width: int, height: int) -> bool:
    return max(width, height) > MIN_LONG_SIDE


def needs_playable(width: int, height: int, codec: str | None) -> bool:
    """A player needs a proxy: a big video, or an HEVC one (whatever its size)."""
    return needs_proxy(width, height) or (codec or "").lower() in BROWSER_UNSAFE


def progress_path(settings: Settings, src: str | Path) -> Path:
    return paths(settings, Path(src))[0].with_suffix(".progress")


def state(settings: Settings, src: str | Path) -> tuple[str, float | None]:
    """("ready", 1.0) | ("making", done 0..1) | ("waiting", done) (a sidecar asks for one but
    no child is making it now: the worker starts it again) | ("failed", None) | ("none", None)."""
    src = Path(src)
    if ready(settings, src) is not None:
        return "ready", 1.0
    try:
        proxy_file, meta_path, _ = paths(settings, src)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fp = _fingerprint(src)
    except (OSError, ValueError):
        return "none", None
    if any(meta.get(k) != fp[k] for k in ("source", "size", "mtime_ns")):
        return "none", None
    if meta.get("failed"):
        return ("none", None) if _retry_allowed(meta, time.time()) else ("failed", None)
    if not meta.get("pending"):
        return "none", None
    try:
        done = float(proxy_file.with_suffix(".progress").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        done = 0.0
    done = min(max(done, 0.0), 1.0)
    return ("making" if _running(proxy_file.with_suffix(".pid")) else "waiting"), done


def ready(settings: Settings, src: str | Path, *, for_analysis: bool = False) -> Path | None:
    """The finished proxy of `src` when its source is unchanged, else None. `for_analysis`:
    also None for an HDR source (its proxy's colours are not the original's)."""
    src = Path(src)
    try:
        proxy, meta_path, _ = paths(settings, src)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fp = _fingerprint(src)
    except (OSError, ValueError):
        return None
    if not meta.get("complete") or not proxy.is_file():
        return None
    if any(meta.get(k) != fp[k] for k in ("source", "size", "mtime_ns")):
        return None
    if for_analysis and meta.get("hdr"):
        return None
    if orientation(proxy) != orientation(src):   # a sideways proxy would turn every frame
        return None
    return proxy


def orientation(path: Path) -> int:
    """The display rotation OpenCV reads (and applies) for a video, in degrees; 0 if none."""
    import cv2   # noqa: PLC0415
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    finally:
        cap.release()


def _skipped(settings: Settings, src: Path) -> bool:
    """The child failed on this very file (it is not tried again until the file changes; a
    transient failure is, after its pause: `_retry_allowed`). An older "skipped": "hdr" record
    (before HDR proxies were made) no longer counts."""
    try:
        meta = json.loads(paths(settings, src)[1].read_text(encoding="utf-8"))
        fp = _fingerprint(src)
    except (OSError, ValueError):
        return False
    return bool(meta.get("failed")) and not _retry_allowed(meta, time.time()) \
        and all(meta.get(k) == fp[k] for k in ("source", "size", "mtime_ns"))


def _attempts(meta_path: Path, src: Path) -> int:
    """How many times a build of this very file failed for a passing reason so far (a
    changed file starts again from 0)."""
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fp = _fingerprint(src)
    except (OSError, ValueError):
        return 0
    if any(meta.get(k) != fp[k] for k in ("source", "size", "mtime_ns")):
        return 0
    return int(meta.get("attempts") or 0)


def _running(pid_file: Path) -> bool:
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return True            # no /proc (macOS): the pid is alive, trust it
    return b"pfb_app.services.proxy" in cmd


def busy(settings: Settings) -> bool:
    """A proxy child of ours is running now (or waiting for the run lock)."""
    d = proxies_dir(settings)
    return d.is_dir() and any(_running(p) for p in d.glob("*.pid"))


def start(settings: Settings, src: str | Path, width: int, height: int,
          codec: str | None = None) -> bool:
    """Begin making the proxy of `src` in the background when it is big (or HEVC: `codec`)
    and none is ready or being made. Never raises: a proxy is only an optimisation. True when
    one was started."""
    try:
        src = Path(src)
        if not needs_playable(width, height, codec) or ready(settings, src) is not None \
                or _skipped(settings, src):
            return False
        proxy, _, _ = paths(settings, src)
        proxy.parent.mkdir(parents=True, exist_ok=True)
        maybe_sweep(settings)
        pid_file = proxy.with_suffix(".pid")
        if _running(pid_file):
            return False
        from pfb_app.ops import disk   # noqa: PLC0415
        try:   # never eat the space uploads and analysis need (open uploads count too)
            disk.ensure_room(settings, src.stat().st_size // SIZE_SHARE)
        except disk.DiskLow:
            log.info("空きが少ないため1080pの縮小版は作りません（元の動画を使います）: %s", src)
            return False
        # a pending sidecar: remove_for (a deleted work) finds a queued or running proxy too,
        # and the child gives up when it is gone (_still_wanted)
        _write_meta(proxy.with_suffix(".json"),
                    {**_fingerprint(src), "complete": False, "pending": True,
                     "attempts": _attempts(proxy.with_suffix(".json"), src)})
        log_file = proxy.with_suffix(".log")
        with open(log_file, "ab") as out:
            proc = subprocess.Popen(
                [*_low_priority(), sys.executable, "-m", "pfb_app.services.proxy",
                 str(src), str(proxy)],
                stdin=subprocess.DEVNULL, stdout=out, stderr=out, start_new_session=True,
                env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
        pid_file.write_text(str(proc.pid))
        return True
    except Exception:   # noqa: BLE001 - never hold the caller up
        log.exception("1080pの縮小版を作り始められませんでした: %s", src)
        return False


def _low_priority() -> list[str]:
    """CPU nice 10 and, where there is ionice, the idle I/O class: only spare capacity."""
    prefix = ["ionice", "-c3"] if shutil.which("ionice") else []
    return [*prefix, "nice", "-n", "10"]


def remove_for(settings: Settings, sources) -> int:
    """Remove the proxies of these source videos (with their work: services.deletion); the
    sources may already be gone. Returns how many."""
    wanted = set()
    for src in sources:
        try:
            wanted.add(str(Path(src).resolve()))
        except OSError:
            continue
    d = proxies_dir(settings)
    if not wanted or not d.is_dir():
        return 0
    removed = 0
    for meta_path in d.glob("*.json"):
        try:
            source = json.loads(meta_path.read_text(encoding="utf-8")).get("source")
        except (OSError, ValueError):
            continue
        if source in wanted:
            _drop(d, meta_path.stem)
            removed += 1
    return removed


def _drop(d: Path, stem: str) -> None:
    _stop_child(d / f"{stem}.pid", d / f"{stem}.mp4")   # its open part file holds the disk
    for p in (d / f"{stem}.mp4", d / f"{stem}.part.mp4", d / f"{stem}.pid",
              d / f"{stem}.log", d / f"{stem}.json", d / f"{stem}.progress"):
        p.unlink(missing_ok=True)


def _is_child_for(pid: int, dst: Path) -> bool:
    """`pid` is our transcode of `dst`: its command line is `... -m pfb_app.services.proxy SRC
    DST` (from /proc; without /proc, as on macOS, nothing is verified, so False)."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except OSError:
        return False
    return b"pfb_app.services.proxy" in cmd and os.fsencode(str(dst)) in cmd


def _stop_child(pid_file: Path, dst: Path) -> None:
    """SIGTERM the transcode named in `pid_file`, only when it is really ours for `dst`
    (a pidfd pins the process between the check and the signal where there is one). Never
    raises; a stopped child leaves its part file, which the caller removes."""
    try:
        pid = int(pid_file.read_text())
    except (OSError, ValueError):
        return
    pidfd_open = getattr(os, "pidfd_open", None)
    send = getattr(signal, "pidfd_send_signal", None)
    try:
        if pidfd_open is not None and send is not None:
            fd = pidfd_open(pid)
            try:
                if _is_child_for(pid, dst):
                    send(fd, signal.SIGTERM)
                    log.info("縮小版づくりを止めました（作業の削除）: pid %s", pid)
            finally:
                os.close(fd)
        elif _is_child_for(pid, dst):
            os.kill(pid, signal.SIGTERM)
            log.info("縮小版づくりを止めました（作業の削除）: pid %s", pid)
    except OSError:   # gone already, or not ours to signal
        pass


def maybe_sweep(settings: Settings, now: float | None = None) -> bool:
    """`sweep`, at most once per SWEEP_EVERY_SEC in this process (called when proxies are
    started, when an upload session starts and when a work is deleted). True when it ran."""
    global _last_sweep
    now = time.time() if now is None else now
    if now - _last_sweep < SWEEP_EVERY_SEC:
        return False
    _last_sweep = now
    try:
        sweep(settings)
    except OSError:
        log.exception("縮小版の掃除に失敗しました")
    return True


def start_for(settings: Settings, videos) -> int:
    """`start` for each path of `videos`, probing its size; returns how many started."""
    from pfb_app.preprocess.frames import VideoError, probe_video   # noqa: PLC0415
    started = 0
    for path in videos:
        try:
            info = probe_video(path)
        except VideoError:
            continue
        started += start(settings, path, info.width, info.height)
    return started


def sweep(settings: Settings) -> int:
    """Remove proxies (and their files) whose source is gone or changed; returns how many."""
    removed = 0
    d = proxies_dir(settings)
    if not d.is_dir():
        return 0
    for meta_path in d.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            src = Path(meta["source"])
            gone = not src.is_file() or src.stat().st_size != meta.get("size") \
                or src.stat().st_mtime_ns != meta.get("mtime_ns")
        except (OSError, ValueError, KeyError):
            gone = True
        if gone:
            _drop(d, meta_path.stem)
            removed += 1
    return removed


# --- the child: one transcode ------------------------------------------------------------------

def _size(width: int, height: int) -> tuple[int, int]:
    if height <= width:
        h = TARGET_HEIGHT
        w = round(width * h / height / 2) * 2
    else:                       # a portrait phone video: 1080 wide
        w = TARGET_HEIGHT
        h = round(height * w / width / 2) * 2
    return w, h


class Cancelled(Exception):
    """The proxy was removed (its work deleted) while it was queued or being made."""


def _still_wanted(meta_path: Path) -> bool:
    """The sidecar `start` wrote is still there (remove_for drops it with the work)."""
    return meta_path.exists()


def _tonemap_graph(vin, w: int, h: int):
    """resize, then BT.709 SDR (TONEMAP), as yuv420p: a libavfilter graph for an HDR source."""
    import av.filter   # noqa: PLC0415
    g = av.filter.Graph()
    last = g.add_buffer(template=vin)
    for name, args in (("scale", f"{w}:{h}"), ("scale", TONEMAP), ("format", "yuv420p")):
        node = g.add(name, args)
        last.link_to(node)
        last = node
    sink = g.add("buffersink")
    last.link_to(sink)
    g.configure()
    return g


def make(src: Path, dst: Path, *, require_sidecar: bool = False,
         progress=None) -> dict:
    """Transcode `src` to a 1080p H.264 proxy at `dst` (atomic: a part file, then a rename);
    an HDR source is tone-mapped to BT.709 SDR. Returns the sidecar written next to it.
    `require_sidecar` (the child of `start`): give up (Cancelled, nothing left behind) when the
    pending sidecar is gone before the rename. `progress(done)`: 0..1 now and then."""
    import av   # noqa: PLC0415 - only the child needs PyAV

    src, dst = Path(src), Path(dst)
    meta_path = dst.with_suffix(".json")
    part = dst.with_name(dst.stem + ".part.mp4")
    started = time.monotonic()
    fp = _fingerprint(src)
    rotation, hdr = _first_frame_facts(src)
    with av.open(str(src)) as inp:
        vin = inp.streams.video[0]
        vin.thread_type = "AUTO"
        vin.codec_context.thread_count = THREADS
        ain = inp.streams.audio[0] if inp.streams.audio else None
        w, h = _size(vin.codec_context.width, vin.codec_context.height)
        total = vin.frames or (round(float(vin.duration * vin.time_base * vin.average_rate))
                               if vin.duration and vin.average_rate else 0)
        graph = _tonemap_graph(vin, w, h) if hdr else None
        with av.open(str(part), "w", format="mp4", options={"movflags": "+faststart"}) as out:
            rate = vin.average_rate or vin.guessed_rate or Fraction(30, 1)
            vout = out.add_stream("libx264", rate=rate)
            vout.width, vout.height, vout.pix_fmt = w, h, "yuv420p"
            vout.time_base = vin.time_base
            vout.options = {"crf": CRF, "preset": PRESET, "threads": str(THREADS)}
            if hdr:   # tagged as what it now is: BT.709 SDR, limited range
                cc = vout.codec_context
                cc.color_trc, cc.color_primaries, cc.colorspace, cc.color_range = 1, 1, 1, 1
            if rotation:   # the phone's display rotation, as in the original (before muxing)
                vout.set_display_rotation(rotation)
            aout = out.add_stream_from_template(ain) if ain is not None else None
            streams = [vin] + ([ain] if ain is not None else [])
            frames = 0
            for packet in inp.demux(*streams):
                if packet.stream is ain:
                    if packet.dts is None:
                        continue
                    packet.stream = aout
                    out.mux(packet)
                    continue
                for frame in packet.decode():
                    if graph is not None:
                        graph.push(frame)
                        small = graph.pull()
                    else:
                        small = frame.reformat(width=w, height=h, format="yuv420p")
                    small.pts, small.time_base = frame.pts, frame.time_base
                    for p in vout.encode(small):
                        out.mux(p)
                    frames += 1
                    if progress is not None and total and frames % PROGRESS_EVERY == 0:
                        progress(min(frames / total, 0.99))
            for p in vout.encode():
                out.mux(p)
    if require_sidecar and not _still_wanted(meta_path):
        part.unlink(missing_ok=True)
        raise Cancelled(str(src))
    os.replace(part, dst)
    if require_sidecar and not _still_wanted(meta_path):   # removed during the rename
        dst.unlink(missing_ok=True)
        raise Cancelled(str(src))
    meta = {**fp, "complete": True, "hdr": bool(hdr), "width": w, "height": h,
            "rotation": rotation, "frames": frames, "bytes": dst.stat().st_size,
            "seconds": round(time.monotonic() - started, 1)}
    if hdr:
        meta["tonemapped"] = "bt709"
    _write_meta(meta_path, meta)
    if progress is not None:
        progress(1.0)
    return meta


def _write_meta(meta_path: Path, meta: dict) -> None:
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    tmp.replace(meta_path)


def _first_frame_facts(src: Path) -> tuple[int, bool]:
    """(display rotation in degrees as PyAV reads it, whether the colour is HDR)."""
    import av   # noqa: PLC0415
    with av.open(str(src)) as inp:
        vin = inp.streams.video[0]
        hdr = is_hdr(getattr(vin.codec_context, "color_trc", None),
                     getattr(vin.codec_context, "color_primaries", None))
        for frame in inp.decode(vin):
            return int(round(getattr(frame, "rotation", 0) or 0)) % 360, hdr
    return 0, hdr


def _progress_writer(path: Path):
    def write(done: float) -> None:
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(f"{done:.3f}", encoding="utf-8")
            tmp.replace(path)
        except OSError:
            pass
    return write


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    src, dst = Path(argv[0]), Path(argv[1])
    progress_file = dst.with_suffix(".progress")
    try:
        with open(dst.parent / RUN_LOCK, "a") as lock:   # one transcode at a time: wait
            fcntl.flock(lock, fcntl.LOCK_EX)
            if src.is_file() and _still_wanted(dst.with_suffix(".json")):
                make(src, dst, require_sidecar=True, progress=_progress_writer(progress_file))
    except Cancelled:   # its work was deleted meanwhile: nothing to keep
        dst.with_name(dst.stem + ".part.mp4").unlink(missing_ok=True)
        return 0
    except Exception as e:   # noqa: BLE001 - logged and recorded (see the module docstring)
        logging.basicConfig(level=logging.INFO)
        log.exception("1080pの縮小版を作れませんでした: %s", src)
        dst.with_name(dst.stem + ".part.mp4").unlink(missing_ok=True)
        try:
            meta_path = dst.with_suffix(".json")
            if _still_wanted(meta_path):
                record = {**_fingerprint(src), "complete": False, "failed": True}
                if is_transient(e):   # for now, not for this file: tried again later
                    attempts = _attempts(meta_path, src) + 1
                    record.update(transient=True, attempts=attempts,
                                  retry_at=time.time() + RETRY_PAUSE_SEC)
                _write_meta(meta_path, record)
        except OSError:
            pass
        return 1
    finally:
        dst.with_suffix(".pid").unlink(missing_ok=True)
        progress_file.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
