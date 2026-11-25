"""Telling the coach that a work has moved on without them watching it (UX round 2, #4 a/b).

- The browser tab's title follows the state: on a work 「（確認できます）練習試合」,
  「（解析中 47%）練習試合」, 「（あなたの番）…」, 「（止まっています）…」, 「（完了）…」, each
  followed by 「｜NiceMill」; on the home screen 「（あなたの番 2件）NiceMill」 or
  「（解析中 47%）NiceMill」.
- An optional soft chime, once, when a work goes from analysing to the coach's turn while
  this tab is open. It is off by default; the toggle 「終わったら音で知らせる」 sits on the home
  screen above the works being analysed. Under prefers-reduced-motion it never sounds.
  Opened over https (scripts/https_serve.sh, docs/https.md), the same moment also shows the
  device's notification when the tab is not in view: permission is asked on the press of the
  toggle itself, and nothing is shown without it. Over plain http browsers allow no
  notifications, so there is only the chime.
- The home screen, while anything is being analysed, looks again every `live.WATCH_SEC`
  seconds and reruns the page when a work changes state or moves on by ten percent, so the
  cards, the tab title and the chime follow without a reload.

How it reaches the browser: one hidden element per page run (`st.html` with its script
allowed) carries the title, whether the sound is on and the chime ids to play. The script
installs one small watcher in the page (once per browser tab) that keeps the tab title while
such an element is on the screen and gives the page its own title back when there is none
(the drop zone, the detail pages). Each chime id plays once per browser tab; an id stays in
the element for CHIME_KEEP_SEC seconds, so a rerun that cuts a page run short cannot lose it.
"""
import json
import time
import uuid
from html import escape
from typing import Iterable

import streamlit as st

from pfb_app.ui import keyed
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import live, progress_panel
from pfb_app.ui.wizard.progress import WorkProgress, all_work

APP = "NiceMill"          # the app's name (theme.APP_NAME; app.py's page_title)
KEY = "pfb_notify"                 # the hidden container (CSS: .st-key-pfb_notify)
SOUND = "_pfb_notify_sound"        # the coach's choice, kept apart from the widget
SOUND_WIDGET = "pfb_notify_sound"  # the toggle (its state goes when it is not shown)
SEEN = "_pfb_notify_seen"          # {assignment id: state} as this session last saw it
CHIMES = "_pfb_notify_chimes"      # [(chime id, time)] still handed to the page
HOLD = "_pfb_notify_hold"          # True while the coach renames or deletes: no auto rerun
CHIME_KEEP_SEC = 30
SOUND_LABEL = "終わったら音で知らせる"
SOUND_HELP = ("解析が終わってあなたの番になったとき、この画面を開いていれば短い音を一度鳴らします。"
              "端末で動きを減らす設定にしているときは鳴りません。"
              "https のアドレスで開いているときは、ほかの画面を見ていても端末の通知で知らせます"
              "（初めてオンにしたときに、通知してよいか聞かれます）。")
NOTICE = "確認できる人がいます"   # the device notification's text (with the work's tab title)

READY, RUNNING, STOPPED, DONE = "ready", "running", "stopped", "done"


def _now() -> float:
    return time.time()


# --- states and titles ------------------------------------------------------------------------

def work_state(work: WorkProgress) -> str:
    """"done" (complete), "ready" (the coach's turn), "stopped", else "running"."""
    if work.complete:
        return DONE
    if work.your_turn:
        return READY
    if work.stopped:
        return STOPPED
    return RUNNING


def percent(work: WorkProgress) -> int | None:
    """The analysis's percentage while it moves on by itself (as the card's bar shows it)."""
    a = work.analysis
    if a is None or not a.active or a.fraction <= 0:
        return None
    return progress_panel.held_percent(work.assignment_id, a)


def _running_word(work: WorkProgress) -> str:
    p = percent(work)
    if p is not None:
        return f"解析中 {p}%"
    statuses = {s.status for s in work.subs}
    return "順番待ち" if "processing" not in statuses and "waiting_gpu" in statuses else "解析中"


def work_title(work: WorkProgress) -> str:
    """The tab title of an open work: its state in words, its name, then the app's name."""
    state = work_state(work)
    if state == DONE:
        word = "完了"
    elif state == READY:
        at_review = work.turn_step is not None and work.keys[work.turn_step - 1] == "review"
        word = "確認できます" if at_review else "あなたの番"
    elif state == STOPPED:
        word = "止まっています"
    else:
        word = _running_word(work)
    return f"（{word}）{work.title}｜{APP}"


def home_title(works: list[WorkProgress]) -> str:
    """The home screen's tab title: how many works wait for the coach (stopped ones too), else
    the analysis under way, else the app's name."""
    turn = [w for w in works if work_state(w) in (READY, STOPPED)]
    if turn:
        return f"（あなたの番 {len(turn)}件）{APP}"
    running = [w for w in works if work_state(w) == RUNNING]
    if len(running) == 1:
        return f"（{_running_word(running[0])}）{APP}"
    if running:
        return f"（解析中 {len(running)}件）{APP}"
    return APP


# --- the chime ------------------------------------------------------------------------------

def newly_ready(seen: dict, works: Iterable, *, everyone: bool) -> list[int]:
    """The works that were analysing when this session last saw them and are the coach's
    turn now; `seen` is updated. A work seen for the first time never counts (a reload or a
    new tab does not chime). `everyone`: `works` is every work (home), so works no longer
    there are forgotten; one work's screen keeps what it knows of the others."""
    fresh, ids = [], set()
    for w in works:
        state = work_state(w)
        if seen.get(w.assignment_id) == RUNNING and state == READY:
            fresh.append(w.assignment_id)
        seen[w.assignment_id] = state
        ids.add(w.assignment_id)
    if everyone:
        for aid in [a for a in seen if a not in ids]:
            del seen[aid]
    return fresh


def sound_on() -> bool:
    return bool(st.session_state.get(SOUND, False))


def _keep_sound() -> None:
    st.session_state[SOUND] = bool(st.session_state.get(SOUND_WIDGET))


def sound_toggle() -> None:
    """「終わったら音で知らせる」, off until the coach turns it on (kept for the session)."""
    st.toggle(SOUND_LABEL, value=sound_on(), key=SOUND_WIDGET, on_change=_keep_sound,
              help=SOUND_HELP)


def _chimes(fresh: list[int]) -> list[str]:
    """The chime ids to hand to the page: new ones for `fresh` works when the sound is on,
    plus those handed out in the last CHIME_KEEP_SEC seconds (each plays once per tab)."""
    now = _now()
    kept = [(cid, t) for cid, t in st.session_state.get(CHIMES, []) if now - t < CHIME_KEEP_SEC]
    if sound_on():
        kept += [(f"{aid}-{uuid.uuid4().hex[:12]}", now) for aid in fresh]
    st.session_state[CHIMES] = kept
    return [cid for cid, _ in kept] if sound_on() else []


# --- the page part --------------------------------------------------------------------------

_JS = """
(function () {
  var w = window, n = w.__pfbNotify;
  if (!n) {
    n = w.__pfbNotify = {played: {}, ours: null, base: null, ctx: null, sound: false};
    var audio = function () {
      var C = w.AudioContext || w.webkitAudioContext;
      if (!C) return null;
      if (!n.ctx) n.ctx = new C();
      if (n.ctx.state === "suspended") {
        var r = n.ctx.resume();
        if (r && r.catch) r.catch(function () {});
      }
      return n.ctx;
    };
    var calm = function () {
      return !!(w.matchMedia && w.matchMedia("(prefers-reduced-motion: reduce)").matches);
    };
    /* the device's notification: https only (a secure context), with permission, and only
       while the tab is not in view (in view, the chime and the title say it) */
    n.notice = function (title, text) {
      try {
        if (!w.isSecureContext || !("Notification" in w)) return;
        if (w.Notification.permission !== "granted" || document.visibilityState === "visible") return;
        new w.Notification(title, {body: text, tag: "pfb-ready"});
      } catch (e) { /* none is fine */ }
    };
    n.chime = function () {
      if (calm()) return;
      try {
        var ctx = audio();
        /* only a context in the "run" state plays (locked on Safari, iOS: no sound). The
           state is matched by its stem: the screens' English-word checks read this script */
        if (!ctx || !/^run/.test(ctx.state)) return;
        var t = ctx.currentTime + 0.05;
        [[784, 0], [1175, 0.22]].forEach(function (p) {
          var o = ctx.createOscillator(), g = ctx.createGain();
          o.type = "sine";
          o.frequency.value = p[0];
          g.gain.setValueAtTime(0.0001, t + p[1]);
          g.gain.exponentialRampToValueAtTime(0.12, t + p[1] + 0.03);
          g.gain.exponentialRampToValueAtTime(0.0001, t + p[1] + 0.9);
          o.connect(g);
          g.connect(ctx.destination);
          o.start(t + p[1]);
          o.stop(t + p[1] + 1);
        });
      } catch (e) { /* no sound is fine */ }
    };
    /* phones allow sound only after a touch: wake the audio on one while the sound is on,
       and on the touch of the toggle itself (the page does not know yet it is on) */
    var wake = function (e) {
      var t = e && e.target, toggle = !!(t && t.closest && t.closest(".st-key-pfb_notify_sound"));
      if ((n.sound || toggle) && !calm()) { try { audio(); } catch (x) {} }
      if (toggle && w.isSecureContext && "Notification" in w
          && w.Notification.permission === "default") {
        try { w.Notification.requestPermission(); } catch (x) {}
      }
    };
    document.addEventListener("pointerdown", wake, true);
    document.addEventListener("keydown", wake, true);
    n.apply = function () {
      var all = Array.prototype.slice.call(document.querySelectorAll("[data-pfb-notify]"));
      var fresh = all.filter(function (e) { return !e.closest('[data-stale="true"]'); });
      var el = (fresh.length ? fresh : all).pop();
      if (!el) {
        if (n.ours !== null && document.title === n.ours && n.base) document.title = n.base;
        n.ours = null;
        return;
      }
      var d;
      try { d = JSON.parse(el.getAttribute("data-pfb-notify")); } catch (e) { return; }
      if (document.title !== d.title) {
        if (document.title !== n.ours) n.base = document.title;
        document.title = d.title;
      }
      n.ours = d.title;
      n.sound = !!d.sound;
      (d.chimes || []).forEach(function (id) {
        if (n.played[id]) return;
        n.played[id] = 1;
        n.chime();
        n.notice(d.title, d.notice);
      });
    };
    w.setInterval(n.apply, 1000);
  }
  n.apply();
})();
"""


def _html(title: str, chimes: list[str]) -> str:
    data = json.dumps({"title": title, "sound": sound_on(), "chimes": chimes,
                       "notice": NOTICE}, ensure_ascii=False)
    return ("<style>.st-key-pfb_notify { display: none; }</style>"
            f'<span data-pfb-notify="{escape(data, quote=True)}" hidden></span>'
            f"<script>{_JS}</script>")


def _emit(title: str, works: list[WorkProgress], *, everyone: bool) -> None:
    seen = st.session_state.get(SEEN)
    if not isinstance(seen, dict):
        seen = {}
    fresh = newly_ready(seen, works, everyone=everyone)
    st.session_state[SEEN] = seen
    st.html(_html(title, _chimes(fresh)), unsafe_allow_javascript=True)


def for_work(progress: WorkProgress) -> None:
    """On a work's screens (router.render): the tab title, and a chime if it just became the
    coach's turn."""
    with keyed.container(key=KEY):
        _emit(work_title(progress), [progress], everyone=False)


def signature(works: list[WorkProgress]) -> tuple:
    """What the home screen shows that can change by itself: each work's state and its
    percentage in tens."""
    return tuple((w.assignment_id, work_state(w), progress_panel.bucket(w)) for w in works)


def needs_rerun(base: tuple, now: tuple, held: bool) -> bool:
    """A tick reruns the page when what home shows has moved on, unless the coach is in the
    middle of renaming or deleting (HOLD)."""
    return not held and now != base


@st.fragment(run_every=live.WATCH_SEC)
def _home_watch(base: tuple, first: list) -> None:
    if first:                         # drawn with the page: its works are `base`, no re-read
        first.pop()
        return
    if st.session_state.get(HOLD):    # renaming or deleting: no read, no rerun
        return
    repo = get_repo()                 # a fragment rerun may run in another thread
    try:
        now = signature(all_work(repo))
    finally:
        repo.conn.close()
    if needs_rerun(base, now, bool(st.session_state.get(HOLD))):
        st.rerun()                    # the whole page: a work has moved on


def watching(works: list[WorkProgress]) -> bool:
    """Home looks again by itself while any work is being analysed (or moves on by itself)."""
    return any(work_state(w) == RUNNING or (w.analysis is not None and w.analysis.active)
               for w in works)


def for_home(works: list[WorkProgress], *, held: bool = False) -> None:
    """On the home screen: the tab title, a chime for works that just became the coach's
    turn, and, while anything is being analysed, a look every WATCH_SEC seconds. `held`:
    the coach is renaming or deleting, so the look never reruns the page (HOLD)."""
    st.session_state[HOLD] = held
    with keyed.container(key=KEY):
        _emit(home_title(works), works, everyone=True)
        if watching(works):
            _home_watch(signature(works), [True])
