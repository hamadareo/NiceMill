"""Step 1 of a work started from videos: the content check card.

A video group dropped on the drop zone (or added on step 2 of a checked work) is held for the
content check (meta["check"] "pending" / "ready"). The router shows this card on step 1 for the
oldest such group (the first file sets the work, so it is confirmed first).

- **Checking** (an active detect job, never the meta alone): the card says so and redraws
  itself every 10 seconds (live.watch), like step 4; the page reruns when the check has ended,
  and the card appears. The primary is disabled with 「いまは押すものはありません…」. The roster
  can already be entered meanwhile.
- **Ready or failed**: a short card. The sentence, large, on the board; three facts (種目 or
  場面, 評価する人, 解析する部分 with the time estimate), which follow what is chosen; and one
  primary 「この内容で始める」, disabled with a plain reason until everything needed is picked.
  Everything else is in the expander 「内容を変える」, pre-filled from the check (the kind of
  activity: 一般 / 話し合い・会議 / スポーツ / 発表・パフォーマンス / 研修・ワークショップ /
  学習活動, sport, title, activity, rubric, 解析の細かさ, method, people, target, analysed
  part, clock offset)
  with the optional roster; it is open when the check could not decide (needs_pick). The title
  defaults to 「{種目 or 場面} {M月D日}」, not the file name, and follows the kind of activity
  until the coach edits it; the rest of the card speaks in that kind's words (会議: 進行役…).
  A radio with a single option is not shown. The primary calls `confirm_check` and then
  follows the derived step.

A later group of a work that is already set keeps the work's kind, title and activity (shown,
not asked); it still gets its own method, people, analysed part, offset and roster. Its roster
starts as the first confirmed group's (same match, same players); the coach can edit it.
"""
import math
import re
from html import escape
from pathlib import Path

import streamlit as st

from pfb_app.domain import profiles
from pfb_app.domain.vocab import SPORT_KEYS, SPORT_LABELS, words
from pfb_app.services import content_check, eta
from pfb_app.services.content_check import (LONG_NOTE, MAX_PEOPLE, MOTION_NOTE, SHORT_PART_NOTE,
                                            CheckCard, CheckChoice, build_card, check_facts,
                                            confirm_check, default_title, earlier_roster_text,
                                            estimate_for, numbered, roster_roles, sport_phrase)
from pfb_app.services.review import BusyError
from pfb_app.services.roster import ROLE_LABELS
from pfb_app.ui.common import MODEL_LABELS, flash
from pfb_app.ui.wizard import live, router, theme
from pfb_app.ui.wizard.progress import View, held_submissions
from pfb_app.ui.wizard.roster_editor import roster_input
from pfb_app.ui.wizard.step_files import METHOD_CAPTIONS, METHODS

PRIMARY = "この内容で始める"
CHANGE = "内容を変える"
KINDS = tuple(profiles.PROFILES)   # the kinds of activity, in screen order
KIND_QUESTION = "どんな場面の動画ですか"
TARGET_LABELS = {"person": "人", "vehicle": "車・バイク"}
OFFSET_LABEL = "この動画の始まりの試合時間（分）"
CHECKING_TITLE = "動画の中身を確かめています"
CHECKING_GUIDE = "どんな場面か、映っている人は何人くらいかを、自動で確かめています。"
CARD_TITLE = "この内容で解析を始めます"
CARD_GUIDE = "違うところがあれば、「内容を変える」で直してから始めてください。"
DECIDED = "内容を決定しました。解析を始めています。"
# why the primary is disabled, in plain words
PICK_SCENE = "「内容を変える」でどんな場面の動画かを選ぶと、始められます。"
PICK_SPORT = "「内容を変える」でスポーツの種類を選ぶと、始められます。"
FIX_ROSTER = "「内容を変える」の{roster}を直すか、空にすると、始められます。"
EARLIER_ROSTER = "最初に決定した動画の{roster}を入れてあります。違うところは直してください。"
PAPER_NEEDS_ROSTER = "「内容を変える」で名簿を入れると、論文方式で始められます。"
EMPTY_TITLE = "「内容を変える」で{work}の名前を入れると、始められます。"
EMPTY_ACTIVITY = "「内容を変える」で{work}の内容を入れると、始められます。"
END_BEFORE_START = ("「内容を変える」で、解析の終わり（{end}分）を始まり（{start}分）より後にすると、"
                    "始められます。")
START_AFTER_END = ("「内容を変える」で、解析の始まり（{start}分）を動画の長さ（約{length}分）より前にすると、"
                   "始められます。")
FIX_PREFIX = "「内容を変える」で直してください："
PAPER_HINT = ("論文方式では、名簿の人の外見と初期位置を手がかりに人物を見分けます。名簿に「名前」と、"
              "できれば「外見」「初期位置（カメラ1）」のような列（カメラごと）を入れてください。")

HELP = [
    "入れた動画の中身（どんな場面か、映っている人は何人くらいか）を自動で確かめ、"
    "その結果で解析の内容を決めてあります。結果は推定なので、違うところは「内容を変える」で直します。",
    "「内容を変える」では、名前、内容、見てほしい観点、解析の細かさ、追跡する人数、解析する部分、"
    "名簿（任意）を変えられます。名簿を入れておくと、あとで人物に名前の候補が付きます。",
    "「この内容で始める」を押すと、その内容で解析が始まります。",
    "うまくいかないとき：中身を確かめられなかったときは、「内容を変える」でどんな場面の動画かを選べば"
    "先へ進めます。長い間「確かめています」のままのときは、管理者に知らせてください。",
]

_UPLOAD_PREFIX = re.compile(r"^[0-9a-f]{8}_")   # save_uploaded_video's unique prefix


def _held(screen: router.Screen) -> list[dict]:
    return held_submissions(screen.snap, screen.progress.assignment_id)


def _file_names(sub: dict) -> list[tuple[str, str]]:
    """[(カメラn, the file's name as the coach gave it)]"""
    cameras = (sub.get("meta") or {}).get("cameras") or {}
    return [(f"カメラ{i}", _UPLOAD_PREFIX.sub("", Path(str(path)).name, count=1))
            for i, path in enumerate(cameras.values(), 1)]


def _verdict_html(sub: dict, sentence: str, *, waiting: bool) -> str:
    files = "".join(f'<li><span class="pfb-verdict__cam">{escape(cam)}</span>{escape(name)}</li>'
                    for cam, name in _file_names(sub))
    if waiting:
        label = theme.status_chip("processing", "確かめています")
    else:
        label = "自動で確かめた結果（推定）"
    live_region = ' role="status"' if waiting else ""    # only the panel that changes by itself
    return (f'<section class="pfb-verdict{" pfb-verdict--waiting" if waiting else ""}"'
            f'{live_region} aria-label="動画の中身"><p class="pfb-verdict__label">{label}</p>'
            f'<p class="pfb-verdict__text">{escape(sentence)}</p>'
            f'<ul class="pfb-verdict__files" aria-label="入れた動画">{files}</ul></section>')


def _roster_given(key: str) -> bool:
    text = st.session_state.get(f"{key}_text") or ""
    return st.session_state.get(f"{key}_file") is not None or bool(text.strip())


def _roster(key: str, domain: str, roles: list[str], earlier: str = "", *,
            boxed: bool = True):
    """The optional roster. Returns (roster or None, usable): not usable when something was
    given that cannot be used yet (its reasons are shown by the editor). `earlier`: a later
    group's start, the first confirmed group's roster as pasted rows; it fills the paste box
    once (the coach's edits are kept). `boxed`: in its own expander (while checking); on the
    card it sits inside 「内容を変える」 under a heading (expanders cannot nest)."""
    text_key = f"{key}_text"
    if earlier and text_key not in st.session_state:
        st.session_state[text_key] = earlier
    label = f"{words(domain).roster}（任意）"
    if boxed:
        box = st.expander(label)
    else:
        box = st.container()
        box.markdown(f"**{label}**")
    with box:
        if earlier and st.session_state.get(text_key) == earlier:
            st.caption(EARLIER_ROSTER.format(roster=words(domain).roster))
        roster = roster_input(key, domain, [ROLE_LABELS[r] for r in roles if r in ROLE_LABELS])
    return roster, roster is not None or not _roster_given(key)


def _has_names(roster) -> bool:
    """A roster with at least one named person (what the paper method identifies)."""
    if roster is None:
        return False
    columns = [c for c, role in roster.roles.items() if role == "name"]
    return any((row.get(c) or "").strip() for row in roster.rows for c in columns)


def _others(held: list[dict]) -> None:
    if len(held) > 1:
        st.caption(f"ほかに{len(held) - 1}件の動画が、この次に中身の決定を待っています。")


# --- checking ---------------------------------------------------------------------------------

def _signature(view: View) -> tuple:
    return tuple((s.submission_id, s.step, s.status) for s in view.progress.subs)


def _checking_draw(sid: int, settings):
    def draw(view: View) -> None:
        sub = view.repo.get_submission(sid)
        st.html(_verdict_html(sub, content_check.WAITING_SENTENCE, waiting=True))
        live.show_worker_warning(view.repo, settings, [sid])
    return draw


def _checking(screen: router.Screen, sub: dict, card: CheckCard, held: list[dict]) -> None:
    sid = sub["id"]
    with theme.step_card(CHECKING_TITLE, CHECKING_GUIDE):
        live.watch(f"check_{sid}", screen.view, _signature, _checking_draw(sid, screen.settings))
        _others(held)
        st.caption(f"{words(card.domain).roster}は、確かめている間にも入れておけます。")
        _roster(f"chk_roster_{sid}", card.domain, card.roster_roles,
                "" if card.sets_work else earlier_roster_text(screen.repo, sid))
        st.button(PRIMARY, key=f"chk_go_{sid}", type="primary", disabled=True, width="stretch")
        st.caption(live.NOTHING_TO_PRESS)
    theme.help_expander(HELP)


# --- the card ---------------------------------------------------------------------------------

def _prefill(key: str, auto) -> None:
    """Pre-fill a keyed widget with `auto` and keep following the check's suggestion (when
    the scene or sport changes) until the coach edits the field."""
    shown = f"_pfb_auto_{key}"
    if key not in st.session_state or st.session_state[key] == st.session_state.get(shown):
        st.session_state[key] = auto
    st.session_state[shown] = auto


def _scene_for(card: CheckCard, profile: str | None) -> str | None:
    """The scene behind a kind of activity: スポーツ is a sport; any other kind keeps the
    check's group work / other (a sport the coach turned into another kind is group work)."""
    if profile is None:
        return None
    if profile == profiles.SPORTS:
        return "sport"
    return card.scene if card.scene in ("group_work", "other") else "group_work"


def _kind_and_sport(sid: int, card: CheckCard,
                    later: bool) -> tuple[str | None, str | None, str | None]:
    """(profile, scene, sport): the kind of activity picked (pre-filled from the check), the
    scene behind it and the sport; a later group's are the work's."""
    if later:
        return card.profile, card.scene, card.sport
    guess = None if card.needs_pick or card.profile not in KINDS else KINDS.index(card.profile)
    profile = st.radio(KIND_QUESTION, KINDS, format_func=profiles.label,
                       key=f"chk_profile_{sid}", horizontal=True, index=guess)
    sport = None
    if profile == profiles.SPORTS:
        guess = card.sport if card.sport in SPORT_KEYS else None
        sport = st.selectbox("スポーツの種類", SPORT_KEYS, format_func=SPORT_LABELS.get,
                             key=f"chk_sport_{sid}", placeholder="選んでください",
                             index=None if guess is None else SPORT_KEYS.index(guess))
    return profile, _scene_for(card, profile), sport


def _activity_default(card: CheckCard, scene: str | None, sport: str | None,
                      profile: str | None) -> str:
    if card.activity and (scene, sport, profile) == (card.scene, card.sport, card.profile):
        return card.activity
    if scene == "sport":
        return sport_phrase(sport) if sport else ""
    if profile is not None and profile not in profiles.UNCHANGED:
        return profiles.activity_text(profile)
    return "グループ活動" if scene == "group_work" else ""


def _title_default(card: CheckCard, scene: str | None, sport: str | None,
                   profile: str | None) -> str:
    """Ours follows the kind and sport chosen now; a title the coach gave stays."""
    if card.auto_title and card.day is not None:
        return numbered(default_title(scene, sport, card.day, profile), card.taken_titles)
    return card.title


def _work_fields(sid: int, card: CheckCard, later: bool, scene, sport, profile,
                 w) -> tuple[str, str, str]:
    """(title, activity, rubric): asked for the group that sets the work, else the work's."""
    if later:
        st.html(f'<dl class="pfb-facts"><dt>{escape(w.work)}の名前</dt>'
                f'<dd>{escape(card.title)}</dd><dt>{escape(w.work)}の内容</dt>'
                f'<dd>{escape(card.activity)}</dd></dl>')
        st.caption(f"この動画は、この{w.work}の動画として加わります。")
        return card.title, card.activity, ""
    _prefill(f"chk_title_{sid}", _title_default(card, scene, sport, profile))
    title = st.text_input(f"{w.work}の名前", key=f"chk_title_{sid}")
    _prefill(f"chk_activity_{sid}", _activity_default(card, scene, sport, profile))
    activity = st.text_area(f"{w.work}の内容", key=f"chk_activity_{sid}", height=90,
                            placeholder="例：サッカーの練習試合（前半）、紙で高い塔を作るグループ活動")
    rubric = st.text_area(w.rubric_label, key=f"chk_rubric_{sid}", height=90)
    return title, activity, rubric


def _method(sid: int, domain: str) -> str:
    options = ["pfb3x3"] if domain == "sports" else list(METHODS)
    if len(options) == 1:             # nothing to choose: no radio
        return options[0]
    captions = [theme.L(c) for c in METHOD_CAPTIONS[:len(options)]]
    return st.radio(theme.L("解析の方式"), options, format_func=METHODS.get,
                    key=f"chk_method_{sid}_{domain}", captions=captions, horizontal=True)


def _model(screen: router.Screen, sid: int, later: bool) -> str | None:
    """解析の細かさ (moved here from the drop zone): the work's own, for the group that sets
    the work; None (the work's) for a later group or when there is nothing to choose."""
    options = list(screen.settings.models)
    if later or len(options) < 2:
        return None
    current = screen.repo.get_assignment(screen.progress.assignment_id)["model_key"]
    return st.selectbox(theme.L("解析の細かさ"), options, key=f"chk_model_{sid}",
                        index=options.index(current) if current in options else 0,
                        format_func=lambda k: MODEL_LABELS.get(k, k),
                        help="高精度は時間がかかりますが、より細かく読み取ります。")


def _target(sid: int, sport: str | None) -> str:
    if sport != "motor_racing":
        return "person"
    return st.radio("追跡する対象", list(TARGET_LABELS), format_func=TARGET_LABELS.get,
                    key=f"chk_target_{sid}", horizontal=True, index=1)


def _people(sid: int, card: CheckCard, domain: str, target: str) -> int | None:
    limit = MAX_PEOPLE[domain]
    thing, unit = ("台数", "台") if target == "vehicle" else ("人数", "人")
    guess = card.max_people      # counted for the check's own scene; kept within the limit
    people = st.number_input(f"追跡する{thing}（{limit}{unit}まで）", min_value=1,
                             max_value=limit, step=1, key=f"chk_people_{sid}_{domain}",
                             value=None if guess is None else min(int(guess), limit))
    st.caption(f"空欄なら{limit}{unit}まで追跡します。確かめた結果から入れてあります。")
    return None if people is None else int(people)


def _part(sid: int, card: CheckCard) -> tuple[float, float | None]:
    """The analysed part (start, end) in seconds; asked in minutes."""
    if card.duration_s:
        st.caption(f"動画の長さは約{max(1, math.ceil(card.duration_s / 60))}分です。")
    left, right = st.columns(2)
    start = left.number_input("解析の始まり（分）", min_value=0.0, step=0.5, format="%.1f",
                              key=f"chk_start_{sid}", value=card.start_sec / 60)
    end = right.number_input("解析の終わり（分、空欄なら最後まで）", min_value=0.0, step=0.5,
                             format="%.1f", key=f"chk_end_{sid}",
                             value=None if card.end_sec is None else card.end_sec / 60)
    return float(start) * 60, None if end is None else float(end) * 60


def _min(seconds: float) -> str:
    return f"{round(seconds / 60, 1):g}"


def _part_reason(start: float, end: float | None, duration: float | None) -> str | None:
    """A part that cannot be analysed, in minutes: the end not after the start, or a start
    at or after the video's end."""
    if end is not None and end <= start:
        return END_BEFORE_START.format(start=_min(start), end=_min(end))
    if duration and start >= duration:
        return START_AFTER_END.format(start=_min(start), length=_min(duration))
    return None


def _estimate(screen: router.Screen, sub: dict, method: str, people: int | None,
              card: CheckCard, start: float, end: float | None) -> float | None:
    """The time the analysis may take for the part the coach has chosen (start, end)."""
    cameras = len((sub.get("meta") or {}).get("cameras") or {})
    return estimate_for(screen.repo, method, cameras, people or card.people or 0,
                        eta.window(card.duration_s, start, end))


def _facts_html(rows: list[tuple[str, str]]) -> str:
    items = "".join(f"<dt>{escape(k)}</dt><dd>{escape(v)}</dd>" for k, v in rows)
    return f'<dl class="pfb-facts pfb-checkfacts" aria-label="解析の内容">{items}</dl>'


def _notes(card: CheckCard) -> None:
    """Outside the expander: what the coach should read (camera motion, a later group's
    disagreement). The long-video note is the fact 「はじめの20分」, explained inside."""
    for note in card.notes:
        if note == MOTION_NOTE:
            st.warning(f"{MOTION_NOTE}。{SHORT_PART_NOTE}。")
        elif note not in (SHORT_PART_NOTE, LONG_NOTE):
            st.info(note)


def _confirm(screen: router.Screen, sid: int, choice: CheckChoice, roster) -> None:
    try:
        confirm_check(screen.repo, screen.settings, sid, choice, roster)
    except ValueError as e:
        st.error(f"{FIX_PREFIX}{e}")
        return
    except BusyError as e:           # decided meanwhile (another tab): show where it stands now
        flash(str(e), "error", raw=True)
        router.go(None)
        router.state()["submission_id"] = None
        st.rerun()
    flash(DECIDED)
    router.go(None)
    router.state()["submission_id"] = None
    st.rerun()


def _card(screen: router.Screen, sub: dict, card: CheckCard, held: list[dict]) -> None:
    sid = sub["id"]
    later = not card.sets_work
    with theme.step_card(CARD_TITLE, CARD_GUIDE):
        st.html(_verdict_html(sub, card.sentence, waiting=False))
        _others(held)
        facts_slot = st.empty()       # filled below, from what is chosen in the expander
        _notes(card)
        # opened when the check left something to pick or to write (scene 「その他」 has no
        # activity to start from)
        opened = card.needs_pick or (not later and not _activity_default(
            card, card.scene, card.sport, card.profile))
        with st.expander(CHANGE, expanded=opened):
            profile, scene, sport = _kind_and_sport(sid, card, later)
            domain = card.domain if later else ("sports" if scene == "sport" else "school")
            if not later:
                sport = sport if scene == "sport" else None
            wording = profile or domain
            theme.set_domain(wording)         # the rest of the card in the chosen vocabulary
            w = words(wording)
            title, activity, rubric = _work_fields(sid, card, later, scene, sport, profile, w)
            model_key = _model(screen, sid, later)
            method = _method(sid, domain)
            target, people = "person", None
            if method == "pfb3x3":
                target = _target(sid, sport)
                people = _people(sid, card, domain, target)
            start, end = _part(sid, card)
            if LONG_NOTE in card.notes:
                st.info(LONG_NOTE)
            estimate = _estimate(screen, sub, method, people, card, start, end)
            if estimate is not None:
                st.caption(live.ESTIMATE_NOTE)
            offset = None           # blank: the game clock is not known (times from the start)
            if domain == "sports":
                given = st.number_input(OFFSET_LABEL, min_value=0.0, step=1.0, format="%.1f",
                                        key=f"chk_offset_{sid}", value=None)
                offset = None if given is None else float(given)
            roles = (card.roster_roles if (domain, sport) == (card.domain, card.sport)
                     else roster_roles(domain, sport, card.numbers_visible))
            if method == "paper":
                roles = ["name", "appearance", "place"]
                st.info(theme.L(PAPER_HINT))
            roster, roster_ok = _roster(f"chk_roster_{sid}", wording, roles,
                                        earlier_roster_text(screen.repo, sid) if later else "",
                                        boxed=False)
        seen = card.people if card.target == target else None
        facts_slot.html(_facts_html(check_facts(
            scene=scene, sport=sport, people=seen, target=target, start_sec=start,
            end_sec=end, estimate_sec=estimate, duration_s=card.duration_s, limit=people,
            profile=profile)))
        reason = None
        if profile is None:
            reason = PICK_SCENE
        elif scene == "sport" and sport is None:
            reason = PICK_SPORT
        elif not (title or "").strip():
            reason = EMPTY_TITLE.format(work=w.work)
        elif not (activity or "").strip():
            reason = EMPTY_ACTIVITY.format(work=w.work)
        elif (part := _part_reason(start, end, card.duration_s)) is not None:
            reason = part
        elif not roster_ok:
            reason = FIX_ROSTER.format(roster=w.roster)
        elif method == "paper" and not _has_names(roster):
            reason = PAPER_NEEDS_ROSTER
        if st.button(PRIMARY, key=f"chk_go_{sid}", type="primary", disabled=reason is not None,
                     width="stretch"):
            _confirm(screen, sid, CheckChoice(
                title=title, activity=activity, rubric_text=rubric, domain=domain, sport=sport,
                method=method, max_people=people, target=target, start_sec=start, end_sec=end,
                clock_offset_min=offset, model_key=model_key,
                profile=None if later else profile), roster)
        if reason is not None:
            st.caption(reason)
    theme.help_expander(HELP)


def render(screen: router.Screen) -> None:
    held = _held(screen)
    sub = held[0]
    card = build_card(screen.repo, screen.settings, sub["id"])
    if card.state == "waiting":
        _checking(screen, sub, card, held)
    else:
        _card(screen, sub, card, held)
