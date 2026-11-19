"""ナイスミル look and feel (direction 「A コートサイド」): colour tokens, self-hosted fonts,
the brand, stepper, step card, chips, approval stamp.

A scoreboard: court navy carries the structure (sidebar, the stepper band, the step card's
top edge), ball orange marks the one action of a screen (the primary button, nothing else),
and the numbers are shot-clock amber on navy (amber is never put on white: it does not read
there). Approval is a centre-circle double ring with the word 承認. Fonts (M PLUS 1 for text,
Dela Gothic One for headings, Barlow Condensed for numbers; all SIL OFL) are vendored under
ui/static/fonts and served by Streamlit static serving (relative ``app/static/`` URLs), so the
page never requests anything from the internet. Every state is spelled out in words as well
as colour. The token names are kept from the first theme (``board`` = the court navy), so
every screen's own CSS follows.
"""
from contextlib import contextmanager
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Iterator, Literal

import streamlit as st

from pfb_app.domain import vocab
from pfb_app.ui import keyed

TOKENS = {
    "board": "#0F2D5C",     # court navy: structure, the stepper band, the sidebar
    "board-2": "#21477F",   # navy, one step up: the sidebar's current page, hover on navy
    "paper": "#EDF1F6",     # the gym floor: background
    "ink": "#13202F",       # text
    "muted": "#4D5A6C",     # secondary text
    "line": "#C9D2DE",      # lines
    "wait": "#A87200",      # signal amber, darkened for marks on white: your turn, waiting
    "sky": "#1F6FB2",       # processing
    "pen": "#0F2D5C",       # approval: the centre-circle stamp (navy; the ring is the sign)
    "chalk": "#F5F8FC",     # text on navy (the content check's sentence)
    "chalk-2": "#C8D4E6",   # quieter text on navy
    "tray": "#FFB000",      # the amber rule under a navy board (as the scoreboard's edge)
    "mist": "#E6ECF4",      # a pale navy: hover of a secondary button or a drop zone
    # the steps of a one-hue chart on navy (この解析の確かさ), light to dark, all >= 3:1 on navy
    "chalk-3": "#A3B7D6",
    "chalk-4": "#86A0C8",
    "chalk-5": "#6A88B8",
    "ball": "#C8410C",      # ball orange: the one primary action (white on it 5.0:1)
    "ball-2": "#A9370A",    # its hover and press
    "amber": "#FFB000",     # shot-clock amber: numbers and 「いまここ」 on navy only (7.4:1)
}

UI_FONT = "M PLUS 1"
DISPLAY_FONT = "Dela Gothic One"   # page titles and home headings (one weight)
NUM_FONT = "Barlow Condensed"      # numbers: step marks, counts, the scoreboard tiles
FONT_FACES = (             # (family, weight, file)
    (UI_FONT, 400, "MPLUS1-Regular.woff2"),
    (UI_FONT, 500, "MPLUS1-Medium.woff2"),
    (UI_FONT, 700, "MPLUS1-Bold.woff2"),
    (UI_FONT, 800, "MPLUS1-ExtraBold.woff2"),
    (DISPLAY_FONT, 400, "DelaGothicOne-Regular.woff2"),
    (NUM_FONT, 600, "BarlowCondensed-SemiBold.woff2"),
    (NUM_FONT, 700, "BarlowCondensed-Bold.woff2"),
)
FONT_FILES = tuple(f for _, _, f in FONT_FACES)
FONT_URL = "app/static/fonts/"   # relative: resolves under the app's own origin and base path

# --- the brand --------------------------------------------------------------------------------
APP_NAME = "ナイスミル"
TAGLINE = "ナイスプレーを、全員ぶん。"
BRAND_DIR = Path(__file__).resolve().parents[1] / "static" / "brand"   # files, for st.logo
BRAND_URL = "app/static/brand/"                                          # the same, served
WORDMARK = "nicemiru_wordmark.svg"            # navy letters: for light backgrounds
WORDMARK_ONDARK = "nicemiru_wordmark_ondark.svg"   # white letters: the navy sidebar
MARK = "nicemiru_mark.svg"
MARK_MONO = "nicemiru_mark_mono.svg"          # black and white: the PDFs' footers
MARK_PNG = "nicemiru_mark_64.png"             # the browser tab's icon

ChipKind = Literal["your_turn", "processing", "waiting_gpu", "stopped", "approved",
                   "insufficient", "done", "guess", "sample"]
# kind -> (text, background, mark). Text colours are darkened where the token itself is
# below 4.5:1 (amber); the token stays on the mark, a non-text graphic.
CHIP_COLOURS: dict[str, tuple[str, str, str]] = {
    "your_turn": ("#5A3B00", "#FFEFC2", TOKENS["wait"]),
    "waiting_gpu": ("#5A3B00", "#FFFFFF", TOKENS["wait"]),
    "processing": ("#154A82", "#E0EBF8", TOKENS["sky"]),
    "stopped": (TOKENS["ink"], "#E5E9EF", TOKENS["ink"]),
    "approved": (TOKENS["pen"], "#FFFFFF", TOKENS["pen"]),
    "insufficient": ("#46525F", "#E5E9EF", TOKENS["muted"]),
    "done": (TOKENS["board"], "#DCE4F0", TOKENS["board"]),
    "guess": ("#154A82", "#E0EBF8", TOKENS["sky"]),   # 推定: a name filled in, to be checked
    "sample": (TOKENS["ink"], TOKENS["paper"], TOKENS["wait"]),   # 見本（見るだけ）
}

_RUN_KEY = "_pfb_theme_run"


# --- the approval stamp: a centre circle ----------------------------------------------------
# A double ring crossed by the court's centre line, with the word 承認 on it: CSS only (st.html
# drops inline <svg>), so it needs no file and no mask.

def approval_stamp(*, animate: bool = True) -> str:
    """The approval stamp of an approved person. With ``animate`` it is pressed on once,
    when the element first appears; under prefers-reduced-motion it simply appears."""
    cls = "pfb-stamp pfb-stamp--animate" if animate else "pfb-stamp"
    return (f'<span class="{cls}" role="img" aria-label="承認済み" title="承認済み">'
            '<span class="pfb-stamp__ring" aria-hidden="true"></span>'
            '<span class="pfb-stamp__word" aria-hidden="true">承認</span></span>')


def brand_html(*, tagline: bool = True) -> str:
    """The wordmark (an <img>: the letters are outlines) and the tagline, for the screens
    shown before anyone is signed in."""
    line = f'<p class="pfb-brand__tag">{escape(TAGLINE)}</p>' if tagline else ""
    return (f'<div class="pfb-brand"><img class="pfb-brand__word" src="{BRAND_URL}{WORDMARK}" '
            f'alt="{escape(APP_NAME)}" height="40">{line}</div>')


# --- CSS ------------------------------------------------------------------------------------

def _font_faces_css() -> str:
    return "\n".join(
        f'@font-face {{ font-family: "{family}"; font-weight: {weight}; font-style: normal; '
        f'font-display: swap; src: url({FONT_URL}{file}) format("woff2"); }}'
        for family, weight, file in FONT_FACES)


def _chip_css() -> str:
    rules = []
    for kind, (fg, bg, mark) in CHIP_COLOURS.items():
        rules.append(f".pfb-chip--{kind} {{ color: {fg}; background: {bg}; "
                     f"--pfb-chip-mark: {mark}; }}")
    return "\n".join(rules)


_CSS = """
:root {
%(tokens)s
  --pfb-font-ui: "M PLUS 1", "Hiragino Sans", "Noto Sans JP", sans-serif;
  --pfb-font-display: "Dela Gothic One", "M PLUS 1", "Hiragino Sans", sans-serif;
  --pfb-font-num: "Barlow Condensed", "M PLUS 1", "Hiragino Sans", sans-serif;
  /* the first theme's hand font: numerals and marks now take the numbers' face */
  --pfb-font-hand: var(--pfb-font-num);
}
%(faces)s

/* Less empty space above the page, so a step's button is more often in the first view. */
.stApp [data-testid="stMainBlockContainer"] { padding-top: 2.5rem; }

/* Buttons: ball orange for the one primary action, a navy outline for the rest. */
.stApp button[data-testid^="stBaseButton-primary"],
.stApp a[data-testid^="stBaseLinkButton-primary"] {
  background: var(--pfb-ball); border: 1.5px solid var(--pfb-ball); color: #FFFFFF;
  border-radius: 6px; min-height: 2.75rem; padding: 0.5rem 1.25rem; font-weight: 700;
}
.stApp button[data-testid^="stBaseButton-primary"]:hover,
.stApp a[data-testid^="stBaseLinkButton-primary"]:hover {
  background: var(--pfb-ball-2); border-color: var(--pfb-ball-2); color: #FFFFFF;
}
.stApp button[data-testid^="stBaseButton-primary"]:active { background: var(--pfb-ball-2); }
.stApp button[data-testid^="stBaseButton-secondary"],
.stApp a[data-testid^="stBaseLinkButton-secondary"] {
  background: #FFFFFF; border: 1.5px solid var(--pfb-board); color: var(--pfb-board);
  border-radius: 6px; min-height: 2.75rem; padding: 0.5rem 1.1rem; font-weight: 500;
}
.stApp button[data-testid^="stBaseButton-secondary"]:hover,
.stApp a[data-testid^="stBaseLinkButton-secondary"]:hover {
  background: var(--pfb-mist); border-color: var(--pfb-board); color: var(--pfb-board);
}
.stApp button[data-testid^="stBaseButton-primary"]:disabled,
.stApp button[data-testid^="stBaseButton-secondary"]:disabled {
  background: #E5E9EF; border-color: var(--pfb-line); color: #4D5A6C; cursor: not-allowed;
}
.stApp [data-testid^="stBaseButton"] p, .stApp [data-testid^="stBaseLinkButton"] p {
  font-weight: inherit; color: inherit;
}

/* Keyboard focus is always visible: a navy ring set off from the control; on navy (the
   sidebar, the stepper band) the ring is amber. */
.stApp :is(button, a, summary, [role="tab"], [role="radio"], [role="checkbox"],
           [tabindex="0"]):focus-visible {
  outline: 3px solid var(--pfb-board); outline-offset: 2px; box-shadow: none;
}
.stApp :is([data-testid="stSidebar"], .st-key-pfb_stepnav, .st-key-pfb_stepcompact)
    :is(button, a, summary, [tabindex="0"]):focus-visible {
  outline-color: var(--pfb-amber);
}

/* Stepper: where this work stands. */
.pfb-stepper {
  display: flex; list-style: none; margin: 0.25rem 0 1.75rem; padding: 0;
  font-family: var(--pfb-font-ui); counter-reset: none;
}
.pfb-step {
  flex: 1 1 0; position: relative; margin: 0; padding: 2.6rem 0.25rem 0;
  text-align: center; color: var(--pfb-muted); font-size: 0.875rem; line-height: 1.5;
}
.pfb-step::before { content: none; }
.pfb-step__mark {
  position: absolute; top: 0; left: 50%%; transform: translateX(-50%%);
  width: 2rem; height: 2rem; border-radius: 50%%; box-sizing: border-box;
  border: 2px solid var(--pfb-line); background: #FFFFFF; color: var(--pfb-muted);
  font-family: var(--pfb-font-num); font-weight: 700; font-size: 1.0625rem; line-height: 1.75rem;
  z-index: 1;
}
.pfb-step:not(:last-child)::after {
  content: ""; position: absolute; top: calc(1rem - 1px); height: 2px;
  left: calc(50%% + 1.25rem); right: calc(-50%% + 1.25rem); background: var(--pfb-line);
}
.pfb-step__label { display: block; letter-spacing: 0.02em; }
.pfb-step__state { display: block; font-size: 0.75rem; color: var(--pfb-muted); }
.pfb-step--done { color: var(--pfb-ink); }
.pfb-step--done .pfb-step__mark {
  background: var(--pfb-board); border-color: var(--pfb-board); color: #FFFFFF;
  font-family: var(--pfb-font-ui); font-weight: 700;
}
.pfb-step--done:not(:last-child)::after { background: var(--pfb-board); }
.pfb-step--current { color: var(--pfb-ink); font-weight: 700; }
.pfb-step--current .pfb-step__mark {
  border-width: 3px; border-color: var(--pfb-board); color: var(--pfb-board);
  line-height: 1.625rem;
}
.pfb-step--current.pfb-step--done .pfb-step__mark {
  color: #FFFFFF; box-shadow: 0 0 0 3px var(--pfb-paper), 0 0 0 5px var(--pfb-board);
}
.pfb-step--current .pfb-step__state { color: var(--pfb-board); font-weight: 700; }

/* Step card: the one task of this screen. */
[class*="st-key-pfb_step_card"] {
  max-width: 46rem; margin: 0 auto; box-sizing: border-box;
  background: #FFFFFF; border: 1px solid var(--pfb-line); border-top: 4px solid var(--pfb-board);
  border-radius: 12px; padding: 1.75rem 2rem 1.5rem;
}
.pfb-card-head { font-family: var(--pfb-font-ui); }
.pfb-card-head .pfb-card-title {
  margin: 0; padding: 0; color: var(--pfb-ink); font-family: var(--pfb-font-ui);
  font-size: 1.5rem; line-height: 1.4; font-weight: 800; letter-spacing: 0.02em;
  font-feature-settings: "palt";
}
.pfb-card-head .pfb-card-guide {
  margin: 0.5rem 0 0; max-width: 38em; color: var(--pfb-muted);
  font-size: 1rem; line-height: 1.8;
}

/* Status chips: a word, a shape and a colour, never colour alone. */
.pfb-chip {
  display: inline-flex; align-items: center; gap: 0.4em; padding: 0.1rem 0.65rem;
  vertical-align: middle;
  border-radius: 999px; font-family: var(--pfb-font-ui); font-size: 0.8125rem;
  font-weight: 700; line-height: 1.6; white-space: nowrap; border: 1px solid transparent;
}
.pfb-chip__mark {
  width: 0.6rem; height: 0.6rem; flex: none; box-sizing: border-box; border-radius: 50%%;
  background: var(--pfb-chip-mark);
}
.pfb-chip--waiting_gpu { border-color: var(--pfb-wait); }
.pfb-chip--waiting_gpu .pfb-chip__mark { background: transparent; border: 2px solid var(--pfb-chip-mark); }
.pfb-chip--processing .pfb-chip__mark {
  border: 2px solid var(--pfb-chip-mark);
  background: linear-gradient(90deg, var(--pfb-chip-mark) 50%%, transparent 50%%);
}
.pfb-chip--stopped .pfb-chip__mark { border-radius: 1px; }
.pfb-chip--approved { border-color: var(--pfb-pen); }
.pfb-chip--approved .pfb-chip__mark { background: transparent; border: 2px solid var(--pfb-chip-mark); }
.pfb-chip--insufficient .pfb-chip__mark { height: 2px; border-radius: 1px; }
.pfb-chip--guess { border-color: var(--pfb-sky); border-style: dashed; }
.pfb-chip--guess .pfb-chip__mark { background: transparent; border: 2px dotted var(--pfb-chip-mark); }
.pfb-chip--done .pfb-chip__mark {
  width: 0.35rem; height: 0.65rem; border-radius: 0; background: transparent;
  border: solid var(--pfb-chip-mark); border-width: 0 2px 2px 0;
  transform: translateY(-0.1rem) rotate(45deg);
}
%(chips)s

/* The approval stamp: a centre circle. A navy double ring crossed by the centre line, the
   word 承認 on a gap in the line. Pressed on once (a short scale and fade), only when motion
   is welcome. */
.pfb-stamp {
  position: relative; display: inline-flex; align-items: center; justify-content: center;
  width: 3.6rem; height: 3.6rem; flex: none; vertical-align: middle; margin: 0 0.45rem;
}
.pfb-stamp__ring {
  position: absolute; inset: 0; box-sizing: border-box; border-radius: 50%%;
  border: 3px solid var(--pfb-pen);
  box-shadow: inset 0 0 0 3px #FFFFFF, inset 0 0 0 5px var(--pfb-pen);
  background: #FFFFFF;
}
.pfb-stamp::before {
  content: ""; position: absolute; left: -0.45rem; right: -0.45rem; top: 50%%;
  border-top: 3px solid var(--pfb-pen); transform: translateY(-50%%);
}
.pfb-stamp__word {
  position: relative; z-index: 1; padding: 0 2px; background: #FFFFFF;
  color: var(--pfb-pen); font-family: var(--pfb-font-ui); font-weight: 800;
  font-size: 0.95rem; line-height: 1.2; letter-spacing: 0.04em;
}
@media (prefers-reduced-motion: no-preference) {
  .pfb-stamp--animate { animation: pfb-press 320ms cubic-bezier(0.3, 1.4, 0.5, 1) 60ms both; }
}
@keyframes pfb-press {
  from { opacity: 0; transform: scale(1.35); } to { opacity: 1; transform: scale(1); }
}

/* The stepper you can click (router.step_nav): one column per step, the mark on top, the
   step's name as a quiet button, its state in words below. */
.st-key-pfb_stepnav { margin: 0.25rem 0 1.5rem; }
.st-key-pfb_stepnav [data-testid="stColumn"] [data-testid="stVerticalBlock"] { gap: 0.2rem; }
.st-key-pfb_stepnav [data-testid="stColumn"] { position: relative; text-align: center; }
.pfb-navstep { position: relative; height: 2rem; padding: 0; }
.pfb-navstep:not(.pfb-navstep--last)::after {
  content: ""; position: absolute; top: calc(1rem - 1px); height: 2px;
  left: calc(50%% + 1.25rem); right: calc(-50%% + 1.25rem); background: var(--pfb-line);
}
.pfb-navstep.pfb-step--done:not(.pfb-navstep--last)::after { background: var(--pfb-board); }
.st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"] {
  min-height: 2rem; padding: 0.15rem 0.25rem; color: var(--pfb-ink); font-weight: 500;
  border-radius: 6px; justify-content: center;
}
.st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"] p {
  font-size: 0.875rem; text-decoration: underline; text-underline-offset: 0.2em;
  text-decoration-color: var(--pfb-line);
}
.st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"]:hover:not(:disabled) {
  background: var(--pfb-mist); color: var(--pfb-board);
}
.st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"]:disabled {
  color: var(--pfb-muted); cursor: default; background: transparent;
}
.st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"]:disabled p { text-decoration: none; }
.pfb-navstate { font-size: 0.75rem; line-height: 1.4; color: var(--pfb-muted); }
.pfb-navstate--current { color: var(--pfb-board); font-weight: 700; }
.st-key-pfb_stepnav [data-testid="stColumn"]:has(.pfb-step--current) button:disabled {
  color: var(--pfb-ink); font-weight: 700;
}

/* Phone stepper (router.step_nav): below 40rem one line, a thin bar and 「ほかの段階」 replace
   the stacked columns; from 40rem up only the columns show. */
.st-key-pfb_stepcompact { display: none; margin: 0 0 0.75rem; }
.st-key-pfb_stepcompact [data-testid="stVerticalBlock"] { gap: 0.4rem; }
.pfb-navline { display: flex; align-items: baseline; gap: 0.5rem; flex-wrap: wrap;
  font-weight: 700; color: var(--pfb-ink); }
.pfb-navbar { height: 4px; border-radius: 2px; background: var(--pfb-line); overflow: hidden; }
.pfb-navbar__fill { height: 100%%; background: var(--pfb-board); }
@media (max-width: 39.99rem) {
  .st-key-pfb_stepcompact { display: block; }
  .st-key-pfb_stepnav { display: none; }
}
@media (forced-colors: active) {
  .pfb-navbar { border: 1px solid CanvasText; }
  .pfb-navbar__fill { background: Highlight; }
}

/* The small stepper on a home card. */
.pfb-mini .pfb-stepper { margin: 0.5rem 0 0; }
.pfb-mini .pfb-step { padding-top: 1.7rem; font-size: 0.75rem; }
.pfb-mini .pfb-step__mark {
  width: 1.375rem; height: 1.375rem; font-size: 0.75rem; line-height: 1.125rem;
}
.pfb-mini .pfb-step--current .pfb-step__mark { line-height: 1rem; }
.pfb-mini .pfb-step:not(:last-child)::after {
  top: calc(0.6875rem - 1px); left: calc(50%% + 0.9rem); right: calc(-50%% + 0.9rem);
}
.pfb-mini .pfb-step__state { font-size: 0.6875rem; }

/* Work header, home head and work cards. */
.pfb-workhead, .pfb-workcard {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.75rem;
  font-family: var(--pfb-font-ui);
}
.pfb-workhead__title { font-size: 1.125rem; font-weight: 700; color: var(--pfb-ink); }
.pfb-workhead__meta, .pfb-workcard__meta { font-size: 0.875rem; color: var(--pfb-muted); }
.pfb-homehead__title {
  margin: 0; padding: 0; font-size: 1.75rem; font-weight: 900; color: var(--pfb-ink);
  letter-spacing: 0.04em;
}
.pfb-homehead__guide { margin: 0.25rem 0 0; color: var(--pfb-muted); }
[class*="st-key-pfb_work_"] {
  background: #FFFFFF; border-color: var(--pfb-line); border-radius: 12px;
  border-left: 4px solid var(--pfb-board);
}
.pfb-workcard__title {
  margin: 0; padding: 0; font-size: 1.1875rem; font-weight: 700; color: var(--pfb-ink);
}
.pfb-workcard__reason { margin: 0.35rem 0 0; color: var(--pfb-ink); font-size: 0.9375rem; }

/* Groups of a work: name, status in words, why. */
.pfb-groups { list-style: none; margin: 0.25rem 0 0.75rem; padding: 0; }
.pfb-groups .pfb-group { margin: 0; }

/* The sidebar menu (ui/app.py): 「ホーム」, then the quiet 管理用の画面 links. */
[class*="st-key-pfb_sidenav"] button[data-testid="stBaseButton-tertiary"] {
  justify-content: flex-start; min-height: 2.25rem; padding: 0.25rem 0.5rem;
  border-radius: 8px; color: var(--pfb-ink); font-weight: 500;
}
[class*="st-key-pfb_sidenav"] button[data-testid="stBaseButton-tertiary"] > div {
  justify-content: flex-start; width: 100%%;
}
[class*="st-key-pfb_sidenav"] button[data-testid="stBaseButton-tertiary"]:hover {
  background: var(--pfb-mist); color: var(--pfb-board);
}
.st-key-pfb_sidenav_here button[data-testid="stBaseButton-tertiary"] {
  background: var(--pfb-mist); color: var(--pfb-board); font-weight: 700;
}
.pfb-sidenav__section {
  margin: 0.75rem 0 0; padding: 0 0.5rem; color: var(--pfb-muted); font-size: 0.8125rem;
  font-weight: 700; letter-spacing: 0.06em;
}
.pfb-sidenav__who {
  margin: 1rem 0 0.25rem; padding: 0 0.5rem; color: var(--pfb-ink); font-size: 0.875rem;
  font-weight: 700;
}
.pfb-group {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem 0.75rem;
  padding: 0.55rem 0; border-bottom: 1px solid var(--pfb-line);
}
.pfb-group:last-child { border-bottom: 0; }
.pfb-group__name { font-weight: 700; color: var(--pfb-ink); min-width: 6em; }
.pfb-group__reason { color: var(--pfb-muted); font-size: 0.875rem; flex: 1 1 14em; }

/* Waiting screens: a group's line and students, live lines, the estimate. */
.pfb-live { margin: 0 0 0.75rem; }
.pfb-students, .pfb-camstates, .pfb-livelines {
  list-style: none; margin: 0.35rem 0 0; padding: 0;
  display: flex; flex-wrap: wrap; gap: 0.35rem 1rem;
}
.pfb-student, .pfb-camstate {
  display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.875rem;
}
.pfb-camstate__name { color: var(--pfb-muted); }
.pfb-student__name { color: var(--pfb-ink); font-weight: 500; }
.pfb-students { margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--pfb-line); }
.pfb-live > .pfb-group { border-bottom: 0; padding-bottom: 0.15rem; }   /* the students' line follows */
.pfb-livelines { flex-direction: column; gap: 0.35rem; margin: 0 0 0.5rem; }
.pfb-estimate { margin: 0.75rem 0 0; color: var(--pfb-ink); font-weight: 700; }
.pfb-subhead { margin: 0.75rem 0 0.25rem; padding: 0; font-size: 1rem; font-weight: 700; }

/* Step 5: the counter, the student, the text large, its evidence; the stamp line. */
.pfb-counter {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.25rem 0.75rem; margin: 0 0 0.5rem;
}
.pfb-counter__group { font-weight: 700; color: var(--pfb-ink); }
.pfb-counter__n { font-weight: 700; color: var(--pfb-board); font-size: 1.0625rem; }
.pfb-counter .pfb-bar { flex: 1 1 8rem; max-width: 16rem; align-self: center; }

/* Progress bars (bars.py): a native progress element, always next to its percentage and its word.
   Navy fills progress and done, amber a wait for the GPU, grey a stop. The track is
   pale enough for every fill to keep 3:1 against it; its edge keeps 3:1 against the paper.
   No stripes, no gradients. */
.pfb-bar {
  --pfb-bar-fill: var(--pfb-board); --pfb-bar-track: #E8ECF2; --pfb-bar-edge: #7F8B9C;
  -webkit-appearance: none; -moz-appearance: none; appearance: none;
  display: block; box-sizing: border-box; width: 100%%; height: 8px; margin: 0;
  border: 0; border-radius: 999px; overflow: hidden; vertical-align: middle;
  background: var(--pfb-bar-track); color: var(--pfb-bar-fill);
  box-shadow: inset 0 0 0 1px var(--pfb-bar-edge);
}
.pfb-bar::-webkit-progress-bar { background: var(--pfb-bar-track); border-radius: 999px;
  box-shadow: inset 0 0 0 1px var(--pfb-bar-edge); }
.pfb-bar::-webkit-progress-value { background: var(--pfb-bar-fill); border-radius: 999px; }
.pfb-bar::-moz-progress-bar { background: var(--pfb-bar-fill); border-radius: 999px; }
.pfb-bar--lg { height: 14px; }
.pfb-bar--sm { height: 8px; }
.pfb-bar--thin { height: 6px; }
.pfb-bar--waiting_gpu { --pfb-bar-fill: var(--pfb-wait); }
.pfb-bar--stopped { --pfb-bar-fill: var(--pfb-muted); }
/* Progress not yet measured: a gentle pulse, and only when motion is welcome. */
@media (prefers-reduced-motion: no-preference) {
  .pfb-bar--moving { animation: pfb-bar-pulse 1.6s ease-in-out infinite alternate; }
  @keyframes pfb-bar-pulse { from { opacity: 1; } to { opacity: 0.5; } }
}
/* Windows high contrast: system colours, a visible edge. */
@media (forced-colors: active) {
  .pfb-bar { forced-color-adjust: none; background: Canvas; border: 1px solid CanvasText;
    box-shadow: none; }
  .pfb-bar::-webkit-progress-bar { background: Canvas; box-shadow: none; }
  .pfb-bar::-webkit-progress-value { background: Highlight; }
  .pfb-bar::-moz-progress-bar { background: Highlight; }
}

.pfb-overall { margin: 0 0 1rem; }
.pfb-overall__head {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.35rem 0.75rem; margin: 0 0 0.45rem;
}
.pfb-overall__pct { color: var(--pfb-ink); font-weight: 900; font-size: 1.25rem;
  font-variant-numeric: tabular-nums; }
.pfb-overall__left { color: var(--pfb-ink); font-weight: 500; }

ol.pfb-stagelist { list-style: none; margin: 0 0 1rem; padding: 0 0 0 0.85rem;
  border-left: 2px solid var(--pfb-line); }
ol.pfb-stagelist > li.pfb-stagelist__row { margin: 0; padding: 0.3rem 0; }
.pfb-stagelist__head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem 0.6rem; }
.pfb-stagelist__name { color: var(--pfb-ink); font-weight: 500; min-width: 9.5em; }
.pfb-stagelist__row--queued .pfb-stagelist__name { color: var(--pfb-muted); }
.pfb-stagelist__row--running .pfb-stagelist__name,
.pfb-stagelist__row--waiting_gpu .pfb-stagelist__name,
.pfb-stagelist__row--stopped .pfb-stagelist__name { font-weight: 700; }
.pfb-stagelist__detail { display: flex; flex-wrap: wrap; align-items: center;
  gap: 0.25rem 0.6rem; margin: 0.3rem 0 0.1rem; }
.pfb-stagelist__detail .pfb-bar { flex: 1 1 10rem; max-width: 20rem; }
.pfb-stagelist__pct, .pfb-stagelist__note { color: var(--pfb-muted); font-size: 0.875rem;
  font-variant-numeric: tabular-nums; }

.pfb-compact { display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem 0.6rem;
  margin: 0 0 0.4rem; }
.pfb-compact__label { color: var(--pfb-ink); font-weight: 700; min-width: 3.5rem; }
.pfb-compact .pfb-bar { flex: 1 1 8rem; max-width: 16rem; }
.pfb-compact__pct { color: var(--pfb-ink); font-weight: 700; min-width: 2.75rem;
  font-variant-numeric: tabular-nums; }
.pfb-compact__caption { color: var(--pfb-muted); font-size: 0.875rem; }
.pfb-compacts { margin: 0 0 1rem; }

.pfb-cardbar { margin: 0.35rem 0 0; }
.pfb-cardbar__label { display: flex; flex-wrap: wrap; align-items: center; gap: 0.25rem 0.5rem;
  margin: 0 0 0.25rem; color: var(--pfb-muted); font-size: 0.8125rem;
  font-variant-numeric: tabular-nums; }
.pfb-cardbar .pfb-bar { max-width: 24rem; }

.pfb-navpct { margin-left: 0.3rem; color: var(--pfb-board); font-weight: 700;
  font-variant-numeric: tabular-nums; }
.pfb-who { display: flex; align-items: center; gap: 0.75rem; margin: 0.75rem 0 0.25rem; }
.pfb-who__name { font-size: 1.375rem; font-weight: 900; color: var(--pfb-ink); }
.pfb-text {
  margin: 0.5rem 0 0.75rem; padding: 1rem 1.25rem; background: #FFFFFF;
  border: 1px solid var(--pfb-line); border-radius: 10px; color: var(--pfb-ink);
  font-size: 1.3125rem; line-height: 1.85; white-space: pre-wrap; overflow-wrap: anywhere;
}
.pfb-evidence {
  display: grid; grid-template-columns: max-content 1fr; gap: 0.2rem 1rem;
  margin: 0.5rem 0 0.25rem; padding: 0.5rem 0 0; border-top: 1px dashed var(--pfb-line);
}
.pfb-evidence dt { color: var(--pfb-muted); font-size: 0.875rem; }
.pfb-evidence dd { margin: 0; color: var(--pfb-ink); }
.pfb-evidence__when { font-weight: 700; font-variant-numeric: tabular-nums; }
.pfb-stamped { display: flex; align-items: center; gap: 0.75rem; margin: 0.25rem 0 0.5rem; }
.pfb-stamped__text { color: var(--pfb-ink); font-weight: 700; }
.pfb-done { margin: 0.5rem 0; color: var(--pfb-ink); font-weight: 700; }

/* Step 6: who is in the PDF and who is not yet. */
.pfb-names { margin: 0 0 0.75rem; }
.pfb-names__title { margin: 0 0 0.25rem; color: var(--pfb-ink); font-weight: 700; }
.pfb-names__list {
  list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 0.25rem 1rem;
  color: var(--pfb-ink);
}

/* Facts of the assignment (step 1, view). */
.pfb-facts {
  display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 1.25rem; margin: 0 0 1rem;
}
.pfb-facts dt { color: var(--pfb-muted); font-size: 0.875rem; }
.pfb-facts dd { margin: 0; color: var(--pfb-ink); font-weight: 500; }

/* Streamlit's own English, in the teacher's words. The file uploader: the size line is a
   caption of ours (from the settings), and the button says ファイルを選ぶ. Its own label is
   taken out with display: none (not just made invisible), so a screen reader reads the
   generated Japanese label of the same real button. */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploaderDropzone"] button[data-testid^="stBaseButton"] > * { display: none; }
[data-testid="stFileUploaderDropzone"] button[data-testid^="stBaseButton"]::after {
  content: "ファイルを選ぶ"; font-weight: 500;
}
/* "Press Enter to apply" and the like: in plain words outside forms, gone inside them (a
   form is sent with its button). */
[data-testid="stForm"] [data-testid="InputInstructions"] { display: none; }
[data-testid="InputInstructions"] > * { display: none; }
[data-testid="InputInstructions"]:has(> *)::after {
  content: "Enterキーで反映されます"; color: var(--pfb-muted); font-size: 0.75rem;
}
[data-testid="stTextArea"] [data-testid="InputInstructions"]:has(> *)::after {
  content: "入力欄の外を押すと反映されます";
}
/* "Running..." in the corner: a word while the page works. */
[data-testid="stStatusWidget"]:has([data-testid="stStatusWidgetRunningIcon"])::after {
  content: "処理中…"; margin-left: 0.4rem; color: var(--pfb-muted); font-size: 0.875rem;
  white-space: nowrap;
}

/* Reduced motion: no movement anywhere (the stamp has its own guard above). */
@media (prefers-reduced-motion: reduce) {
  .stApp *, .stApp *::before, .stApp *::after {
    animation-duration: 0.01ms !important; animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important; scroll-behavior: auto !important;
  }
}

/* The drop zone (a new work, a later video group): a large dashed area with a line of its own
   above the button. */
.st-key-drop_files [data-testid="stFileUploaderDropzone"],
.st-key-drop_more_files [data-testid="stFileUploaderDropzone"] {
  min-height: 9rem; padding: 1.5rem 1rem; box-sizing: border-box;
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 0.75rem; background: var(--pfb-paper); border: 2px dashed var(--pfb-board-2);
  border-radius: 12px;
}
.st-key-drop_files [data-testid="stFileUploaderDropzone"]::before,
.st-key-drop_more_files [data-testid="stFileUploaderDropzone"]::before {
  content: "ここにファイルを置くか、下のボタンで選びます"; color: var(--pfb-ink);
  font-family: var(--pfb-font-ui); font-weight: 700; font-size: 1.0625rem; text-align: center;
}
.st-key-drop_files [data-testid="stFileUploaderDropzone"]:hover,
.st-key-drop_more_files [data-testid="stFileUploaderDropzone"]:hover {
  background: var(--pfb-mist); border-color: var(--pfb-board);
}

/* The content check's sentence, written on the board: the app's guess, which the coach then
   confirms. A thin amber rule under it, as a scoreboard's edge. (The UI font: the numbers' face lacks
   the kanji, and a mix of two faces in one sentence reads as a mistake.) */
.pfb-verdict {
  margin: 0.25rem 0 1.25rem; padding: 1rem 1.35rem 0.9rem; box-sizing: border-box;
  background: var(--pfb-board); color: var(--pfb-chalk); border-radius: 10px 10px 4px 4px;
  border-bottom: 5px solid var(--pfb-tray);
}
.pfb-verdict__label {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 0 0 0.4rem;
  color: var(--pfb-chalk-2); font-size: 0.8125rem; font-weight: 700; letter-spacing: 0.08em;
}
.pfb-verdict__text {
  margin: 0; color: var(--pfb-chalk); font-family: var(--pfb-font-ui); font-weight: 700;
  font-size: 1.5rem; line-height: 1.65; letter-spacing: 0.02em; font-feature-settings: "palt";
  overflow-wrap: anywhere;
}
.pfb-verdict__files {
  list-style: none; margin: 0.65rem 0 0; padding: 0.5rem 0 0; display: flex; flex-wrap: wrap;
  gap: 0.2rem 1.25rem; border-top: 1px dashed var(--pfb-chalk-2);
  color: var(--pfb-chalk-2); font-size: 0.875rem;
}
.pfb-verdict ul.pfb-verdict__files { padding-left: 0; margin-left: 0; }
.pfb-verdict ul.pfb-verdict__files > li { margin: 0; padding: 0; }
.pfb-verdict__cam { font-weight: 700; margin-right: 0.35rem; }
.pfb-verdict--waiting .pfb-verdict__text { font-size: 1.25rem; }
.pfb-verdict .pfb-chip { font-family: var(--pfb-font-ui); letter-spacing: 0; }

/* Step 3, the mapping grid (people_grid): the team question, one card per person. */
/* This screen's step card is wider than the others (three cards per row at 1280 px): more
   specific than the theme's own [class*="st-key-pfb_step_card"] rule, whatever the order. */
.stApp [class*="st-key-pfb_step_card"].st-key-pfb_step_card_people { max-width: 76rem; }
/* With the sidebar open at 1280 px the page's side margins leave the crops under 240 px:
   this screen uses narrower page margins on wide windows (small windows keep Streamlit's). */
@media (min-width: 64rem) {
  [data-testid="stMainBlockContainer"]:has(.st-key-pfb_step_card_people) {
    padding-left: 2rem; padding-right: 2rem; }
}
.pfb-team__side { margin: 0 0 0.25rem; color: var(--pfb-ink); font-weight: 700; }
.pfb-team__none { margin: 0; padding: 1.25rem 0.75rem; color: var(--pfb-muted);
  background: var(--pfb-paper); border: 1px dashed var(--pfb-line); border-radius: 8px;
  font-size: 0.875rem; text-align: center; }
[class*="st-key-pfb_strip_"] img {
  max-height: 10rem; width: auto; max-width: 100%%; border-radius: 6px;
  border: 1px solid var(--pfb-line); }
.pfb-teamnote { margin: 0.25rem 0 0.5rem; color: var(--pfb-muted); font-size: 0.9375rem; }
[class*="st-key-pfb_person_"] {
  background: #FFFFFF; border: 1px solid var(--pfb-line); border-left: 4px solid var(--pfb-line);
  border-radius: 10px; padding: 0.6rem 0.5rem 0.7rem; gap: 0.35rem; }
[class*="st-key-pfb_person_"][class*="_todo"] { border-left-color: var(--pfb-wait); }
[class*="st-key-pfb_person_"][class*="_guess"] { border-left-color: var(--pfb-sky); }
[class*="st-key-pfb_person_"][class*="_named"] { border-left-color: var(--pfb-board); }
[class*="st-key-pfb_person_"][class*="_out"] { background: var(--pfb-paper); }
[class*="st-key-pfb_person_"][class*="_out"] img { opacity: 0.55; }
/* Crops twice the old size: as wide as the card (at least 240 px), 18rem high, whole. */
:is([class*="st-key-pfb_person_"], [class*="st-key-pfb_pcard_"]) img {
  display: block; width: 100%%; height: 18rem; max-width: 100%%;
  object-fit: contain; background: var(--pfb-paper); border-radius: 6px; }
/* The name and its reason take the same room on every card, so the rows line up. */
/* No crop (yet): the same height as a crop, so the cards do not move when crops arrive. */
.pfb-person__nophoto {
  box-sizing: border-box; height: 18rem; margin: 0; display: flex; align-items: center;
  justify-content: center; background: var(--pfb-paper); border: 1px dashed var(--pfb-line);
  border-radius: 6px; color: var(--pfb-muted); font-size: 0.8125rem; }
.pfb-person { min-height: 3.9rem; font-family: var(--pfb-font-ui); }
.pfb-person__head { display: flex; flex-wrap: wrap; align-items: center; gap: 0.2rem 0.5rem; }
.pfb-person__mark {
  margin: 0; color: var(--pfb-ink); font-family: var(--pfb-font-hand); font-weight: 600;
  font-size: 1.1875rem; line-height: 1.4; }
.pfb-person__why { margin: 0.15rem 0 0; color: var(--pfb-ink); font-size: 0.8125rem;
  line-height: 1.5; }
.pfb-person__was { margin: 0.15rem 0 0; color: var(--pfb-muted); font-size: 0.8125rem;
  line-height: 1.5; }
.pfb-gridnone { margin: 0.5rem 0; color: var(--pfb-muted); }

/* 「大きく見る」 (zoom.py): the whole picture in the large dialog, as wide as the dialog and
   never taller than the window; evidence frames with a quiet frame. */
.st-key-pfb_zoom_view img {
  display: block; width: 100%%; height: auto; max-height: 78vh; object-fit: contain;
  background: var(--pfb-paper); border-radius: 6px; }
[class*="st-key-pfb_frames_"] img {
  border: 1px solid var(--pfb-line); border-radius: 6px; }

/* Screen-reader-only text. */
.pfb-sr {
  position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0;
}

/* Deleting a work (home): a quiet 「削除」 on the card; the confirm area's button is outlined
   in ink, bold, never orange (ball orange is for the one primary action only). */
[class*="st-key-pfb_delete_area_"] { border-top: 1px solid var(--pfb-line); padding-top: 0.75rem; }
.stApp :is([class*="st-key-pfb_delete_go_"], [class*="st-key-pfb_delete_stop_"])
    button[data-testid^="stBaseButton-secondary"]:not(:disabled) {
  background: #FFFFFF; border: 2px solid var(--pfb-ink); color: var(--pfb-ink); font-weight: 700;
}
.stApp :is([class*="st-key-pfb_delete_go_"], [class*="st-key-pfb_delete_stop_"])
    button[data-testid^="stBaseButton-secondary"]:not(:disabled):hover {
  background: var(--pfb-paper); border-color: var(--pfb-ink); color: var(--pfb-ink);
}
.pfb-delete__title { margin: 0; font-size: 1.0625rem; font-weight: 700; color: var(--pfb-ink); }
.pfb-delete p { margin: 0.35rem 0 0; color: var(--pfb-ink); }
.pfb-delete .pfb-delete__undo { font-weight: 700; }
.pfb-delete__pending { margin: 0.25rem 0 0; color: var(--pfb-muted); font-size: 0.875rem; }
%(court)s
"""


# The scoreboard parts of 「A コートサイド」: the navy sidebar and stepper band, amber numbers
# on navy, the display face on page titles, the home cards' state edge. Kept apart so the
# component rules above stay as they were. (Plain text: it is inserted into _CSS as it is.)
_COURTSIDE_CSS = """
/* ===== ナイスミル: コートサイド ===== */
/* Headings in our own HTML take the text face; only the page titles below take the display
   face (Streamlit gives every h1-h6 the theme's headingFont). */
.stApp [class*="pfb-"]:is(h1, h2, h3, h4, h5, h6) { font-family: var(--pfb-font-ui); }
.stApp [class*="pfb-"].pfb-homehead__title {
  font-family: var(--pfb-font-display); font-weight: 400; font-size: 2.3rem; line-height: 1.25;
  color: var(--pfb-board); letter-spacing: 0.02em; font-synthesis: none;
}
.stApp [class*="pfb-"].pfb-homegroup {
  font-family: var(--pfb-font-display); font-weight: 400; color: var(--pfb-board);
  font-size: 1.25rem; font-synthesis: none; margin-top: 1.75rem;
}
.stApp .pfb-homegroup__count { font-family: var(--pfb-font-ui); font-size: 1rem; }
.stApp :is(h1, h2, h3, h4, h5, h6) { font-synthesis: none; }

/* The sidebar: court navy (config.toml [theme.sidebar]); its header holds the wordmark. */
[data-testid="stSidebarHeader"] { padding-top: 1.25rem; padding-bottom: 0.5rem; }
[data-testid="stSidebar"] [class*="st-key-pfb_sidenav"] button[data-testid="stBaseButton-tertiary"] {
  color: #FFFFFF;
}
[data-testid="stSidebar"] [class*="st-key-pfb_sidenav"] button[data-testid="stBaseButton-tertiary"]:hover {
  background: var(--pfb-board-2); color: #FFFFFF;
}
[data-testid="stSidebar"] .st-key-pfb_sidenav_here button[data-testid="stBaseButton-tertiary"] {
  background: var(--pfb-board-2); color: #FFFFFF; font-weight: 700;
  box-shadow: inset 4px 0 0 var(--pfb-amber);
}
[data-testid="stSidebar"] .pfb-sidenav__section { color: #AFC0DA; }
[data-testid="stSidebar"] .pfb-sidenav__who { color: #FFFFFF; }
[data-testid="stSidebar"] [data-testid="stPageLink"] a:hover { background: var(--pfb-board-2); }

/* The stepper band: navy, the marks as rounded squares, the current one amber. */
.stApp .st-key-pfb_stepnav {
  background: var(--pfb-board); border-radius: 8px; padding: 1rem 0.5rem 0.75rem;
  box-sizing: border-box;
}
.st-key-pfb_stepnav .pfb-step__mark {
  border-radius: 6px; background: transparent; border-color: #6C84AC; color: #C8D4E6;
}
.st-key-pfb_stepnav .pfb-navstep:not(.pfb-navstep--last)::after { background: #3A5888; }
.st-key-pfb_stepnav .pfb-step--done .pfb-step__mark {
  background: #FFFFFF; border-color: #FFFFFF; color: var(--pfb-board);
}
.st-key-pfb_stepnav .pfb-navstep.pfb-step--done:not(.pfb-navstep--last)::after { background: #FFFFFF; }
.st-key-pfb_stepnav .pfb-step--current .pfb-step__mark,
.st-key-pfb_stepnav .pfb-step--current.pfb-step--done .pfb-step__mark {
  background: var(--pfb-amber); border-color: var(--pfb-amber); color: var(--pfb-board);
  box-shadow: none;
}
.stApp .st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"] { color: #FFFFFF; }
.stApp .st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"] p {
  text-decoration-color: #5C77A3;
}
.stApp .st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"]:hover:not(:disabled) {
  background: var(--pfb-board-2); color: #FFFFFF;
}
.stApp .st-key-pfb_stepnav button[data-testid="stBaseButton-tertiary"]:disabled { color: #AFC0DA; }
.stApp .st-key-pfb_stepnav [data-testid="stColumn"]:has(.pfb-step--current) button:disabled {
  color: #FFFFFF;
}
.st-key-pfb_stepnav .pfb-navstate { color: #C8D4E6; }
.st-key-pfb_stepnav .pfb-navstate--current { color: var(--pfb-amber); }
.st-key-pfb_stepnav .pfb-navpct { color: var(--pfb-amber); font-family: var(--pfb-font-num);
  font-size: 0.875rem; }

/* The phone stepper: the same band, one line, an amber bar. */
.stApp .st-key-pfb_stepcompact {
  background: var(--pfb-board); border-radius: 8px; padding: 0.65rem 0.9rem 0.6rem;
  box-sizing: border-box;
}
.st-key-pfb_stepcompact .pfb-navline { color: #FFFFFF; }
.st-key-pfb_stepcompact .pfb-navpct { color: var(--pfb-amber); }
.st-key-pfb_stepcompact .pfb-navbar { background: #2A4C82; }
.st-key-pfb_stepcompact .pfb-navbar__fill { background: var(--pfb-amber); }

/* The small stepper on a home card: the current step amber, inside a navy edge. */
.pfb-mini .pfb-step--current .pfb-step__mark,
.pfb-mini .pfb-step--current.pfb-step--done .pfb-step__mark {
  background: var(--pfb-amber); border-color: var(--pfb-board); color: var(--pfb-board);
  border-width: 2px; box-shadow: none; line-height: 1.125rem;
}
.pfb-mini .pfb-step__mark { border-radius: 4px; }

/* Cards: a thick left edge says the state again (the chip's word is the state). */
.stApp [class*="st-key-pfb_work_"] { border-radius: 8px; border-left: 6px solid var(--pfb-line); }
.stApp [class*="st-key-pfb_work_"]:has(.pfb-workcard .pfb-chip--your_turn) { border-left-color: var(--pfb-wait); }
.stApp [class*="st-key-pfb_work_"]:has(.pfb-workcard .pfb-chip--stopped) { border-left-color: var(--pfb-ink); }
.stApp [class*="st-key-pfb_work_"]:has(.pfb-workcard :is(.pfb-chip--processing, .pfb-chip--waiting_gpu)) {
  border-left-color: var(--pfb-sky); }
.stApp [class*="st-key-pfb_work_"]:has(.pfb-workcard .pfb-chip--done) { border-left-color: var(--pfb-board); }
.stApp [class*="st-key-pfb_step_card"] { border-radius: 8px; }
.pfb-workcard__title, .pfb-who__name { font-weight: 800; }
.pfb-text { border-left: 5px solid var(--pfb-board); font-weight: 500; }

/* Numbers: the numbers' face; amber where they sit on navy. */
.pfb-counter__n { font-family: var(--pfb-font-num); font-weight: 700; font-size: 1.375rem;
  color: var(--pfb-board); font-variant-numeric: tabular-nums; }
.pfb-overall__pct, .pfb-compact__pct, .pfb-evidence__when {
  font-family: var(--pfb-font-num); font-variant-numeric: tabular-nums; }
.pfb-overall__pct { font-size: 1.5rem; font-weight: 700; }
.stApp .pfb-tile__value { color: var(--pfb-amber); font-weight: 700; }
/* st.metric (試合・練習のまとめ, and elsewhere): a scoreboard tile. */
.stApp [data-testid="stMetric"] {
  background: var(--pfb-board); border-radius: 8px; padding: 0.85rem 1.1rem 0.9rem;
  box-sizing: border-box; height: 100%;
}
.stApp [data-testid="stMetric"] [data-testid="stMetricLabel"],
.stApp [data-testid="stMetric"] [data-testid="stMetricLabel"] * { color: #DCE4F0; }
.stApp [data-testid="stMetric"] [data-testid="stMetricValue"],
.stApp [data-testid="stMetric"] [data-testid="stMetricValue"] * {
  color: var(--pfb-amber); font-family: var(--pfb-font-num); font-weight: 700;
  font-variant-numeric: tabular-nums;
}
.stApp [data-testid="stMetric"] [data-testid="stMetricValue"] { font-size: 2.6rem; line-height: 1.15; }

/* The brand on the screens before sign-in. */
.pfb-brand { margin: 1.75rem 0 1.5rem; }   /* clear of the app's top bar */
.pfb-brand__word { display: block; height: 2.5rem; width: auto; }
.pfb-brand__tag { margin: 0.5rem 0 0; color: var(--pfb-muted); font-weight: 700; }

@media (forced-colors: active) {
  .stApp .st-key-pfb_stepnav, .stApp .st-key-pfb_stepcompact, .stApp [data-testid="stMetric"] {
    border: 1px solid CanvasText; }
}
"""


@lru_cache(maxsize=1)
def theme_css() -> str:
    """The wizard stylesheet: tokens, @font-face, buttons, focus ring and components."""
    tokens = "\n".join(f"  --pfb-{name}: {value};" for name, value in TOKENS.items())
    return _CSS % {"tokens": tokens, "faces": _font_faces_css(), "chips": _chip_css(),
                   "court": _COURTSIDE_CSS}


def _run_marker():
    """An object unique to the current script run, or None outside a Streamlit run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
    except Exception:   # noqa: BLE001 - a Streamlit internals change must not break the page
        return None
    # ScriptRunContext.reset() gives every run a fresh `cursors` dict. Holding it in session
    # state keeps it alive, so a later run can never get the same object back.
    return getattr(ctx, "cursors", None) if ctx is not None else None


def inject_theme() -> None:
    """Write the stylesheet once per page run (calling it again in the same run is a no-op)."""
    marker = _run_marker()
    if marker is not None:
        if st.session_state.get(_RUN_KEY) is marker:
            return
        st.session_state[_RUN_KEY] = marker
    st.html(f"<style>{theme_css()}</style>")


# --- components -----------------------------------------------------------------------------

def stepper_html(labels: list[str], current: int, done: set[int]) -> str:
    """The stepper as HTML: done steps show ✓, the current one carries aria-current."""
    items = []
    for i, label in enumerate(labels):
        is_done, is_current = i in done, i == current
        classes = ["pfb-step"]
        if is_done:
            classes.append("pfb-step--done")
        if is_current:
            classes.append("pfb-step--current")
        mark = "✓" if is_done else str(i + 1)
        if is_current:
            state = '<span class="pfb-step__state">いまここ</span>'
            if is_done:
                state += '<span class="pfb-sr">（済み）</span>'
        elif is_done:
            state = '<span class="pfb-step__state">済み</span>'
        else:
            state = '<span class="pfb-sr">これから</span>'
        aria = ' aria-current="step"' if is_current else ""
        items.append(
            f'<li class="{" ".join(classes)}"{aria}>'
            f'<span class="pfb-step__mark" aria-hidden="true">{mark}</span>'
            f'<span class="pfb-step__label">{escape(label)}</span>{state}</li>')
    return f'<ol class="pfb-stepper" aria-label="進み具合">{"".join(items)}</ol>'


def stepper(labels: list[str], current: int, done: set[int]) -> None:
    """Render the stepper across the top of a step screen."""
    st.html(stepper_html(labels, current, done))


DOMAIN_KEY = "pfb_domain"


def set_domain(domain: str | None) -> None:
    """The open work's domain for this run: screen texts are then localised (school: as is)."""
    st.session_state[DOMAIN_KEY] = domain or "school"


def domain() -> str:
    return st.session_state.get(DOMAIN_KEY) or "school"


def L(text: str) -> str:
    """A screen text in the open work's vocabulary."""
    return vocab.localize(text, domain())


@contextmanager
def step_card(title: str, guidance: str, *, key: str | None = None) -> Iterator:
    """The focus card of a step: a title, one sentence of guidance, then the body.

    ``key`` only needs to be given when one screen shows more than one card.
    """
    box = keyed.container(key="pfb_step_card" + (f"_{key}" if key else ""))
    with box:
        st.html(f'<div class="pfb-card-head"><h2 class="pfb-card-title">{escape(L(title))}</h2>'
                f'<p class="pfb-card-guide">{escape(L(guidance))}</p></div>')
        yield box


def status_chip(kind: ChipKind, text: str) -> str:
    """An inline status label: the text is always there; colour and mark only repeat it."""
    if kind not in CHIP_COLOURS:
        raise ValueError(f"unknown status kind: {kind}")
    return (f'<span class="pfb-chip pfb-chip--{kind}">'
            f'<span class="pfb-chip__mark" aria-hidden="true"></span>{escape(text)}</span>')


def help_expander(lines: list[str]) -> None:
    """The 「この画面の使い方」 expander: a few short lines on how to use this screen."""
    with st.expander("この画面の使い方"):
        st.markdown("\n".join(f"- {L(line)}" for line in lines))
