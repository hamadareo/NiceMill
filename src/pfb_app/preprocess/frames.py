"""Frame extraction from classroom videos (OpenCV, 5 s interval, 854x480 JPEG)."""
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2


class VideoError(Exception):
    """動画を読めない・区間指定が不正な場合のエラー（メッセージは日本語）。"""


@dataclass(frozen=True)
class VideoInfo:
    duration_sec: float
    fps: float
    width: int
    height: int


def format_time(sec: float) -> str:
    """MM:SS, or H:MM:SS from one hour on."""
    total = int(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass(frozen=True)
class Frame:
    index: int
    video_sec: float
    video_time: str
    task_elapsed: str
    path: Path

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "video_sec": self.video_sec,
            "video_time": self.video_time,
            "task_elapsed": self.task_elapsed,
            "path": str(self.path),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(
            index=int(d["index"]),
            video_sec=float(d["video_sec"]),
            video_time=d["video_time"],
            task_elapsed=d["task_elapsed"],
            path=Path(d["path"]),
        )


def probe_video(path: str | Path) -> VideoInfo:
    path = Path(path)
    if not path.is_file():
        raise VideoError(f"動画ファイルが見つかりません: {path}")
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"動画を開けません（形式が未対応か、ファイルが壊れています）: {path.name}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # the header can be fine while the codec cannot be decoded here (AV1, some 10-bit
        # files): say so now, not as 「0秒のフレームを読み取れません」 after the check
        readable = cap.isOpened() and cap.read()[0]
    finally:
        cap.release()
    if not fps or fps <= 0 or not n or n <= 0 or w <= 0 or h <= 0:
        raise VideoError(f"動画の長さ・フレームレート・解像度を取得できません: {path.name}")
    if not readable:
        raise VideoError("この動画の形式は読み込めません（AV1など、対応していない圧縮かもしれません）。"
                         f"MP4（H.264）に変換して入れ直してください: {path.name}")
    return VideoInfo(duration_sec=n / fps, fps=fps, width=w, height=h)


def _times(start: float, end: float, interval: float) -> list[float]:
    count = math.ceil((end - start) / interval - 1e-9)
    return [start + k * interval for k in range(max(count, 0))]


def validate_interval(
    info: VideoInfo, start_sec: float | None, end_sec: float | None, interval: int
) -> tuple[float, float]:
    start = 0.0 if start_sec is None else float(start_sec)
    end = info.duration_sec if end_sec is None else float(end_sec)
    if start < 0:
        raise VideoError(f"開始時刻は0秒以上にしてください（指定: {start:g}秒）")
    if start >= end:
        raise VideoError(f"開始時刻が終了時刻以降になっています（開始 {start:g}秒、終了 {end:g}秒）")
    if end > info.duration_sec + 0.5:
        raise VideoError(
            f"終了時刻が動画の長さを超えています（終了 {end:g}秒、動画の長さ {info.duration_sec:.1f}秒）"
        )
    end = min(end, info.duration_sec)
    if not [t for t in _times(start, end, interval) if t < info.duration_sec]:
        raise VideoError(
            f"指定区間（{start:g}〜{end:g}秒）から取り出せるフレームが0枚です。区間を広げてください"
        )
    return start, end


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    start_sec: float,
    end_sec: float,
    interval_sec: int = 5,
    size: tuple[int, int] = (854, 480),
    workers: int | None = None,
) -> list[Frame]:
    """Seek to start + k*interval (< end) and save one JPEG (quality 90) each.

    Existing output files are reused. The frames still to save are split into up to
    `workers` (default: EXTRACT_WORKERS) runs of consecutive times, each read by its own
    VideoCapture in its own thread: every frame is found by its own seek (OpenCV seeks back
    to a key frame and decodes forward from there, whatever was read before), so the files
    are the same as one reader makes, only sooner.
    """
    info = probe_video(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Frame] = []
    todo: list[tuple[float, Path]] = []
    for k, t in enumerate(_times(start_sec, end_sec, interval_sec)):
        if t >= info.duration_sec - 0.5 / info.fps:   # a time within half a frame of the end has no frame
            break
        out = out_dir / f"frame_{k:04d}_{round(t * 1000)}ms.jpg"
        if not out.exists():
            todo.append((t, out))
        frames.append(Frame(k, t, format_time(t), format_time(t - start_sec), out))
    n = max(1, min(workers or EXTRACT_WORKERS, len(todo) // MIN_FRAMES_PER_WORKER or 1))
    runs = [todo[i * len(todo) // n:(i + 1) * len(todo) // n] for i in range(n)]
    if n == 1:
        _save_frames(path, info, runs[0], size)
    else:
        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="pfb-frames") as pool:
            for f in [pool.submit(_save_frames, path, info, run, size) for run in runs]:
                f.result()   # the first error (VideoError) is raised
    return frames


EXTRACT_WORKERS = 4        # VideoCaptures reading one video's frames at once
MIN_FRAMES_PER_WORKER = 8  # fewer frames than this per reader: not worth another one


def _save_frames(path, info: VideoInfo, todo: list[tuple[float, Path]],
                 size: tuple[int, int]) -> None:
    """Seek to each (time, file) in order with one VideoCapture and save it."""
    if not todo:
        return
    cap = cv2.VideoCapture(str(path))
    try:
        for t, out in todo:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round(t * info.fps))
            ok, img = cap.read()
            if not ok:
                raise VideoError(f"{t:g}秒のフレームを読み取れません: {Path(path).name}")
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            if not cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise VideoError(f"フレーム画像を保存できません: {out}")
    finally:
        cap.release()


def chunk(frames: list[Frame], n: int = 5) -> list[list[Frame]]:
    return [frames[i : i + n] for i in range(0, len(frames), n)]


def sample_points(frames: list[Frame], k: int = 3) -> list[Frame]:
    """k frames spread over start, middle and end (all frames if fewer than k)."""
    if len(frames) <= k:
        return list(frames)
    if k <= 1:
        return frames[:1]
    last = len(frames) - 1
    idx = sorted({int(i * last / (k - 1) + 0.5) for i in range(k)})
    return [frames[i] for i in idx]
