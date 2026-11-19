"""A running job's own progress, as the worker writes it to jobs.progress / progress_note.

Pure helpers: each gives (progress 0..1 or None for unknown, a plain short note or None).
- behavior (tracking engine): the windows the engine has written / the windows it will have;
  never 1 before the job ends, since the translation still follows.
- track: the time spent / the usual time of a track (the history median), capped, since
  the tracker reports no count.
"""

TRACKING_NOTE = "人物を追跡しています"
BEHAVIOR_CAP = 0.99   # the rows are all there, the translation is still to come
TRACK_CAP = 0.95      # a slower track than usual stays just below the end


def behavior_progress(done: int, planned: int | None) -> tuple[float | None, str | None]:
    """(done / planned, 「done / planned 区間」); (None, None) when the plan is unknown or
    has no windows."""
    if not planned or planned <= 0:
        return None, None
    done = max(0, min(int(done), planned))
    return min(round(done / planned, 4), BEHAVIOR_CAP), f"{done} / {planned} 区間"


def track_progress(elapsed_s: float, expected_s: float | None) -> tuple[float | None, str]:
    """(elapsed / expected, capped, TRACKING_NOTE); the progress is None without a history."""
    if not expected_s or expected_s <= 0:
        return None, TRACKING_NOTE
    return min(round(max(0.0, elapsed_s) / expected_s, 4), TRACK_CAP), TRACKING_NOTE
