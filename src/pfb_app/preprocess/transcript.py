"""Transcript segments: slicing, pasted-text parsing, and faster-whisper transcription."""
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pfb_app.preprocess.frames import format_time

_TIME_RE = re.compile(r"^\s*\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?[\s　]*(.*)$")


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "text": self.text}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(start=float(d["start"]), end=float(d["end"]), text=d["text"])


def slice_transcript(segments: list[Segment], start_sec: float, end_sec: float) -> str:
    """Lines "[MM:SS] text" for segments overlapping [start_sec, end_sec)."""
    return "\n".join(
        f"[{format_time(s.start)}] {s.text}"
        for s in segments
        if s.start < end_sec and s.end > start_sec
    )


def parse_pasted_transcript(text: str) -> list[Segment]:
    """Parse "[MM:SS] text" / "MM:SS text" lines (H:MM:SS also accepted).

    Without any time stamp the whole text is one segment covering everything.
    The end of each segment is the next start; the last one is open-ended.
    """
    entries: list[tuple[float, list[str]]] = []
    untimed: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _TIME_RE.match(line)
        if m:
            a, b, c, rest = m.groups()
            sec = int(a) * 3600 + int(b) * 60 + int(c) if c is not None else int(a) * 60 + int(b)
            entries.append((float(sec), [rest.strip()] if rest.strip() else []))
        elif entries:
            entries[-1][1].append(line.strip())
        else:
            untimed.append(line.strip())
    if not entries:
        return [Segment(0.0, math.inf, "\n".join(untimed))] if untimed else []
    if untimed:
        entries.insert(0, (0.0, untimed))
    segments = []
    for i, (start, lines) in enumerate(entries):
        end = entries[i + 1][0] if i + 1 < len(entries) else math.inf
        segments.append(Segment(start, end, "\n".join(lines)))
    return segments


def transcribe(video_path: str | Path, settings) -> list[Segment]:
    """Transcribe with faster-whisper on the GPU.

    Only call this from the whisper session child process (GpuGuard-managed).
    """
    from faster_whisper import WhisperModel  # lazy: heavy, GPU-only

    model = WhisperModel(settings.whisper_model, device="cuda", compute_type="float16")
    segments, _info = model.transcribe(str(video_path), language="ja")
    return [Segment(float(s.start), float(s.end), s.text.strip()) for s in segments if s.text.strip()]
