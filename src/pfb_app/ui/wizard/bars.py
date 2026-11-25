"""Progress bars: the overall bar, the list of steps, group bars, the home bar, approval.

Plain values in, HTML out: nothing here reads the database or the analysis model. Every bar is
a native ``<progress>`` (kept by ``st.html``'s sanitiser, no inline style, read out by screen
readers), carries an ``aria-label`` that repeats its visible text, and sits next to that text:
the percentage and the step name are always written out, and the status is always a chip with
its word, never colour alone. Callers pass screen texts already localised (``theme.L``); the
fixed copy here holds no domain words.

While the analysis waits for the coach (``YOUR_TURN``), the overall, group and home bars say
「あなたの番」 instead of a percentage: the bar keeps its fill, no number is written.
"""
from html import escape
from typing import NamedTuple, Sequence

from pfb_app.ui.wizard import theme

# status key -> (screen word, theme chip kind)
STATUS_CHIPS: dict[str, tuple[str, str]] = {
    "done": ("済み", "done"),
    "running": ("進行中", "processing"),
    "queued": ("待機中", "insufficient"),
    "waiting_gpu": ("順番待ち", "waiting_gpu"),
    "stopped": ("止まっています", "stopped"),
}
YOUR_TURN = "your_turn"   # the analysis waits for the coach: no percentage is written
_TURN_CHIP = ("あなたの番", "your_turn")
# A row with one of these statuses gets its own small bar and its note.
_ACTIVE = ("running", "waiting_gpu", "stopped")
MOVING_TEXT = "進んでいます"


class StageRow(NamedTuple):
    name: str
    status: str
    percent: int
    known: bool
    note: str


def _clamp(percent: int) -> int:
    return max(0, min(100, int(percent)))


def _status(status: str, turn: bool = False) -> str:
    if status not in STATUS_CHIPS and not (turn and status == YOUR_TURN):
        raise ValueError(f"unknown progress status: {status}")
    return status


def chip(status: str) -> str:
    word, kind = _TURN_CHIP if status == YOUR_TURN else STATUS_CHIPS[_status(status)]
    return theme.status_chip(kind, word)   # type: ignore[arg-type]


def _bar(value: int, label: str, status: str, size: str, *, maximum: int = 100,
         moving: bool = False) -> str:
    classes = f"pfb-bar pfb-bar--{size} pfb-bar--{status}" + (" pfb-bar--moving" if moving else "")
    # A moving bar's value is only its start: screen readers hear 「進んでいます」 instead.
    valuetext = f' aria-valuetext="{MOVING_TEXT}"' if moving else ""
    return (f'<progress class="{classes}" max="{maximum}" value="{value}" '
            f'aria-label="{escape(label)}"{valuetext}>{escape(label)}</progress>')


def overall_bar_html(percent: int, remaining: str | None, status: str) -> str:
    """The large bar: 「全体の進み具合 63%」, the time left when known, and the status chip
    (waiting for the coach: 「全体の進み具合」 and 「あなたの番」, no percentage)."""
    p = _clamp(percent)
    turn = _status(status, turn=True) == YOUR_TURN
    head = "全体の進み具合" if turn else f"全体の進み具合 {p}%"
    label = f"{head} {_TURN_CHIP[0]}" if turn else head
    left = (f'<span class="pfb-overall__left">{escape(remaining)}</span>' if remaining else "")
    return (f'<div class="pfb-overall"><p class="pfb-overall__head">'
            f'<span class="pfb-overall__pct">{escape(head)}</span>{chip(status)}{left}</p>'
            f'{_bar(p, label, status, "lg")}</div>')


def _stage_row(row: StageRow) -> str:
    status = _status(row.status)
    head = (f'<span class="pfb-stagelist__name">{escape(row.name)}</span>{chip(status)}')
    detail = ""
    if status in _ACTIVE:
        p = _clamp(row.percent)
        note = f'<span class="pfb-stagelist__note">{escape(row.note)}</span>' if row.note else ""
        if row.known:
            label = f"{row.name} {p}%"
            pct = f'<span class="pfb-stagelist__pct">{p}%</span>'
        else:   # unknown sub-progress: the bar holds its start value, the note says why
            label = f"{row.name} {row.note}".strip()
            pct = ""
        bar = _bar(p, label, status, "sm", moving=not row.known)
        detail = f'<div class="pfb-stagelist__detail">{bar}{pct}{note}</div>'
    return (f'<li class="pfb-stagelist__row pfb-stagelist__row--{status}">'
            f'<div class="pfb-stagelist__head">{head}</div>{detail}</li>')


def stage_list_html(rows: Sequence[StageRow]) -> str:
    """The steps of the analysis in order, each with its chip; the active one with its bar."""
    return (f'<ol class="pfb-stagelist" aria-label="解析の内訳">'
            f'{"".join(_stage_row(r) for r in rows)}</ol>')


def compact_bar_html(label: str, percent: int, caption: str, status: str) -> str:
    """One group or camera: 「1班」, a bar, 「63%」, 「したことの書き出し」 and the status chip."""
    p = _clamp(percent)
    turn = _status(status, turn=True) == YOUR_TURN
    pct = "" if turn else f'<span class="pfb-compact__pct">{p}%</span>'
    aria = f"{label} {_TURN_CHIP[0]}" if turn else f"{label} {p}%"
    return (f'<div class="pfb-compact">'
            f'<span class="pfb-compact__label">{escape(label)}</span>'
            f'{_bar(p, aria, status, "sm")}{pct}'
            f'<span class="pfb-compact__caption">{escape(caption)}</span>{chip(status)}</div>')


def card_bar_html(percent: int, stage_name: str, status: str) -> str:
    """The thin bar on a home card, labelled 「したことの書き出し・63%」, always with its chip word
    (waiting for the coach: the step's name and 「あなたの番」)."""
    p = _clamp(percent)
    turn = _status(status, turn=True) == YOUR_TURN
    text = stage_name if turn else f"{stage_name}・{p}%"
    aria = f"{stage_name}・{_TURN_CHIP[0]}" if turn else text
    return (f'<div class="pfb-cardbar"><p class="pfb-cardbar__label">'
            f'<span>{escape(text)}</span>{chip(status)}</p>'
            f'{_bar(p, aria, status, "thin")}</div>')


def nav_percent_html(percent: int) -> str:
    """The small percentage next to the current step in the step bar."""
    return (f'<span class="pfb-navpct"><span class="pfb-sr">進み具合</span>'
            f'{_clamp(percent)}%</span>')


def nav_turn_html() -> str:
    """Next to the current step while the analysis waits for the coach: no percentage."""
    return f'<span class="pfb-navpct">{_TURN_CHIP[0]}</span>'


def approval_bar_html(approved: int, total: int, label: str | None = None, auto: int = 0) -> str:
    """Step 5's counter, 「承認 1 / 全 3」, with a bar (the text alone when there is none).
    `approved` counts a person's approvals only; `auto` (machine-checked, no person looked) is
    said beside it and never added to it."""
    total = max(0, int(total))
    approved = max(0, min(total, int(approved)))
    text = f"承認 {approved} / 全 {total}"
    where = f'<span class="pfb-counter__group">{escape(label)}</span>' if label else ""
    bar = (_bar(approved, text, "approval", "sm", maximum=total) if total else "")
    extra = (f'<span class="pfb-counter__auto"> ＋ 自動承認 {int(auto)}（人は未確認）</span>'
             if auto else "")
    return (f'<p class="pfb-counter">{where}<span class="pfb-counter__n">{escape(text)}</span>'
            f'{extra}{bar}</p>')
