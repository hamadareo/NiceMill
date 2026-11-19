"""Step 3's mapping grid (tracking method): one card per tracked person, for up to 30 people.

- `team_question`: 「自分のチームはどちらですか」 with the two teams' sample strips and a radio
  (左 / 右 / 分けない), shown when the names stage found two teams; when it could not, why in
  plain words (or nothing). 「相手チームも評価する」 (sports, off by default) is set for every
  group of the work at once.
- `filter_bar`: すべて / 推定あり / まだ選んでいません / 対象外, with how many people each has.
- `person_grid`: three cards per row: the person's crop (large, with 「大きく見る」), 「n番の人」, a 「推定」 chip and its
  reason while the choice is still the proposal, the select (key `pfb_map_{track}_{label}`)
  and 「名前をつける」 (a new person with the coach's name, selected at once).
- `bulk_outside`: the secondary 「残りを対象外にする」.
- `roster_expander`: 「（選手）名簿を入れる・直す」 → apply_roster, then names are asked for again.

The choices on screen are kept per (track run, saved mapping) in the session: only what the
coach changed is stored; every other person follows the form's prefill, so proposals that
arrive while the screen is open (a names job finishing, a team answer) fill in the people the
coach has not touched, and nothing the coach chose is ever thrown away (not by the filter, a
rerun or the live refresh). Each run the selects are set from these values, so the default
names (選手n, 相手選手n), which are numbered over the choices on screen, never move a select.
Persons are always 「n番の人」 on screen, never the tracking labels.
"""
from __future__ import annotations

import re
from html import escape

import streamlit as st

from pfb_app.domain.vocab import mark_name, words
from pfb_app.pipeline.runner import request_names
from pfb_app.services import review
from pfb_app.services.content_check import roster_roles
from pfb_app.services.review import BusyError
from pfb_app.services.roster import JOBS_BUSY, ROLE_LABELS, apply_roster
from pfb_app.storage.db import Repo
from pfb_app.ui import keyed
from pfb_app.ui.common import OUTSIDE, MappingForm, flash
from pfb_app.ui.wizard import theme, zoom
from pfb_app.ui.wizard.roster_editor import roster_input

CHOICES = "_pfb_map_choices"   # {(track run id, identity run id): {label: choice the coach made}}
_PICKS = "_pfb_map_picks"      # {widget key: value} a radio or checkbox was just changed to
_CLEAR = "_pfb_map_clear"      # {text field key: True} to empty on the next run
_UNSET = object()

COLUMNS = 3          # crops at least 240 px wide at 1280 px
TEAM_QUESTION = "自分のチームはどちらですか"
TEAM_GUIDE = "選んだチームの人に名前をつけます。もう一方のチームの人は「対象外」になります。"
TEAM_SIDES = {"T1": "左", "T2": "右", "none": "分けない"}
NO_TEAMS = "no_teams"          # meta: {camera: track run id} the coach answered 分けない for
TEAM_REASONS = {"too_few": "人数が少ないため、チーム分けはしていません。",
                "one_colour": "ユニフォームの色で分けられませんでした。",
                "not_person": None}          # nothing to say: no question, no note
TEAM_UNKNOWN = "チーム分けはできませんでした。"
NO_STRIP = "見本の写真がありません"
OPPONENTS = "相手チームも評価する"
OPPONENTS_HELP = "オンにすると、相手チームの人にも「相手選手1」のような名前がつき、フィードバックを作ります。"
OPPONENTS_KEPT = "決定済みのグループの相手選手は、オフにしてもそのまま残ります。"
FILTERS = {"all": "すべて", "guess": "推定あり", "unset": "まだ選んでいません", "outside": "対象外"}
FILTER_LABEL = "表示する人"
BULK_OUTSIDE = "残りを対象外にする"
BULK_HELP = "まだ選んでいない人を、まとめて「対象外」にします。「これで決定して次へ」を押すまでは保存されません。"
NAME_IT = "名前をつける"
NAME_FIELD = "名前"
NAME_ADD = "この名前で追加"
NAME_HELP = "{mark}に名前をつける"
NO_PHOTO = "写真なし"
NAMES_PENDING = "名簿から名前の候補を探しています。待たずに選んでもかまいません。"
GUESS = "推定"
UNSET = "まだ選んでいません"
PLACEHOLDER = "選んでください"
SHEET_CAPTION = "人物一覧（{camera}）"
SHEET_MORE = "人物一覧の写真を見る"
STRIP_ALT = "{side}のチームの見本"
STRIP_NAME = "{side}のチーム"
SHEET_NAME = "人物一覧"
NO_SHEET = "人物一覧の写真が見つかりません。管理用の画面で、人を見つけるところからやり直してください。"
NONE_HERE = "この表示にあてはまる人はいません。"
ROSTER_APPLY = "この名簿を使う"
ROSTER_DONE = "名簿を入れました。名前の候補を探し直します。"
ROSTER_BUSY = "いまは解析が進んでいるため名簿を入れ替えられません。終わってから、もう一度押してください。"
ROSTER_RESETS = "名簿を入れ替えると、決定したカメラのつなぎ方も選び直しになります。"

_LABEL = re.compile(r"\bP(\d+)\b")



def marks(text: str) -> str:
    """Every tracking label in a message as 「n番の人」 ('…: P7' -> '…: 7番の人')."""
    return _LABEL.sub(lambda m: mark_name(m.group(0)), str(text))


# --- the choices on screen -------------------------------------------------------------------

def choices_key(form: MappingForm) -> tuple[int, int | None]:
    """Choices kept for this track run and this saved mapping: a mapping saved on 管理用の画面
    (a new identity run) is shown as saved, not hidden by choices made before it."""
    return form.track.id, form.identity.id if form.identity is not None else None


def kept(form: MappingForm) -> dict[str, str | None]:
    """The choices the coach made on this form (and not yet saved)."""
    return st.session_state.setdefault(CHOICES, {}).setdefault(choices_key(form), {})


def drop_kept(form: MappingForm) -> None:
    st.session_state.get(CHOICES, {}).pop(choices_key(form), None)


def current(form: MappingForm) -> dict[str, str | None]:
    """{label: choice} for every person: the coach's choice, else the prefill; a value the
    person's select no longer offers (e.g. an opponent once 「相手チームも評価する」 is off)
    is unset."""
    mine = kept(form)
    out = {}
    for label in form.labels:
        value = mine[label] if label in mine else form.prefill.get(label)
        out[label] = value if value in form.options_for(label) else None
    return out


def _touch(ckey: tuple, label: str, wkey: str) -> None:
    """on_change of a person's select: the coach chose this."""
    st.session_state.setdefault(CHOICES, {}).setdefault(ckey, {})[label] = \
        st.session_state.get(wkey)


def _pick(wkey: str) -> None:
    """on_change of the team radio and the opponents checkbox: handled on the run itself
    (callbacks run before the page has its database connection)."""
    st.session_state.setdefault(_PICKS, {})[wkey] = st.session_state.get(wkey)


def _picked(wkey: str):
    return st.session_state.get(_PICKS, {}).pop(wkey, _UNSET)


def _proposed(form: MappingForm, label: str) -> bool:
    return label in form.reasons


def state_of(form: MappingForm, label: str, value) -> str:
    """todo (unset), out (対象外), guess (still the proposal), named (anything else)."""
    if value is None:
        return "todo"
    if value == OUTSIDE:
        return "out"
    if _proposed(form, label) and value == form.prefill.get(label):
        return "guess"
    return "named"


def _matches(form: MappingForm, label: str, value, filt: str) -> bool:
    if filt == "guess":
        return _proposed(form, label)
    if filt == "unset":
        return value is None
    if filt == "outside":
        return value == OUTSIDE
    return True


# --- the team question -----------------------------------------------------------------------

def team_note(form: MappingForm) -> str | None:
    """Why there is no team question, in plain words; None when there is nothing to say (no
    team cue ran, or the people are not people, e.g. cars)."""
    if form.teams or not form.team_reason:
        return None
    return TEAM_REASONS.get(form.team_reason, TEAM_UNKNOWN)


def team_answer(form: MappingForm, meta: dict | None, camera: str) -> str | None:
    """T1 / T2, "none" when the coach answered 分けない for this track run, else None (not
    answered yet)."""
    if form.own_team in review.CLUSTERS:
        return form.own_team
    if ((meta or {}).get(NO_TEAMS) or {}).get(camera) == form.track.id:
        return "none"
    return None


def _mark_no_teams(repo: Repo, sid: int, camera: str, track_id: int | None) -> None:
    with repo._tx():
        meta = dict(repo.get_submission(sid)["meta"] or {})
        marks_ = dict(meta.get(NO_TEAMS) or {})
        if track_id is None:
            marks_.pop(camera, None)
        else:
            marks_[camera] = track_id
        meta[NO_TEAMS] = marks_
        repo.update_submission_meta(sid, meta)


def team_question(form: MappingForm, sid: int, camera: str, *, repo: Repo,
                  meta: dict | None = None) -> None:
    """「自分のチームはどちらですか」: the two strips side by side and 左 / 右 / 分けない
    (set_team_choice; 分けない is kept as an answer of its own); only when the names stage
    found the teams. Nothing is picked until the coach answers; on a decided camera (its
    mapping is saved) the answer is shown but cannot be changed."""
    if not form.teams:
        note = team_note(form)
        if note:
            st.html(f'<p class="pfb-teamnote">{escape(theme.L(note))}</p>')
        return
    wkey = f"pfb_team_{form.track.id}"
    answer = team_answer(form, meta, camera)
    pick = _picked(wkey)
    if pick is not _UNSET and pick is not None and pick != answer and not form.current:
        try:
            review.set_team_choice(repo, sid, camera, None if pick == "none" else pick)
            _mark_no_teams(repo, sid, camera, form.track.id if pick == "none" else None)
        except ValueError as e:
            st.error(theme.L(marks(str(e))))
        else:
            st.rerun()
    st.html(f'<h3 class="pfb-subhead">{TEAM_QUESTION}</h3>')
    sides = st.columns(2, gap="medium")
    for col, team in zip(sides, ("T1", "T2")):
        with col, keyed.container(key=f"pfb_strip_{form.track.id}_{team}"):
            st.html(f'<p class="pfb-team__side">{TEAM_SIDES[team]}</p>')
            strip = form.strips.get(team)
            if strip is None or not zoom.picture(
                    strip, alt=STRIP_ALT.format(side=TEAM_SIDES[team]),
                    name=STRIP_NAME.format(side=TEAM_SIDES[team]),
                    key=f"strip_{form.track.id}_{team}", width="content"):
                st.html(f'<p class="pfb-team__none">{NO_STRIP}</p>')
    st.session_state[wkey] = answer
    st.radio(TEAM_QUESTION, list(TEAM_SIDES), index=None, format_func=TEAM_SIDES.get,
             horizontal=True, key=wkey, on_change=_pick, args=(wkey,),
             label_visibility="collapsed", disabled=form.current)
    st.caption(theme.L(TEAM_GUIDE))


def opponents_toggle(form: MappingForm, sid: int, assignment_id: int, *, repo: Repo) -> None:
    """「相手チームも評価する」 (sports with two teams found, off by default): set for every
    group of the work. Groups already decided keep the opponents they have."""
    if form.domain != "sports" or not form.teams:
        return
    wkey = f"pfb_opp_{sid}"
    pick = _picked(wkey)
    if pick is not _UNSET and bool(pick) != form.evaluate_opponents:
        for s in repo.list_submissions(assignment_id):
            review.set_evaluate_opponents(repo, s["id"], bool(pick))
        st.rerun()
    st.session_state[wkey] = form.evaluate_opponents
    st.checkbox(OPPONENTS, key=wkey, on_change=_pick, args=(wkey,), help=OPPONENTS_HELP)
    if any(s["identity_confirmed"] for s in repo.list_submissions(assignment_id)
           if s["id"] != sid):
        st.caption(OPPONENTS_KEPT)


def sync_opponents(repo: Repo, snap: Repo, assignment_id: int, sub: dict) -> bool:
    """A group added after 「相手チームも評価する」 was turned on takes the work's setting;
    True when it was changed (the page then reruns to read it)."""
    on = any((s["meta"] or {}).get(review.EVALUATE_OPPONENTS)
             for s in snap.list_submissions(assignment_id))
    if on and not (sub["meta"] or {}).get(review.EVALUATE_OPPONENTS):
        review.set_evaluate_opponents(repo, sub["id"], True)
        return True
    return False


# --- filter, bulk, grid ----------------------------------------------------------------------

def filter_bar(key: str, counts: dict[str, int] | None = None) -> str:
    """すべて / 推定あり / まだ選んでいません / 対象外 (with how many people each has). The pick is
    kept apart from the radio, whose option texts change with the counts."""
    def label(f: str) -> str:
        return FILTERS[f] + (f"（{counts[f]}人）" if counts and f in counts else "")
    store = key + "_shown"
    pick = _picked(key)
    if pick in FILTERS:
        st.session_state[store] = pick
    st.session_state[key] = st.session_state.get(store, "all")
    return st.radio(FILTER_LABEL, list(FILTERS), format_func=label, horizontal=True, key=key,
                    on_change=_pick, args=(key,))


def counts_of(form: MappingForm, values: dict[str, str | None]) -> dict[str, int]:
    return {f: sum(_matches(form, l, values[l], f) for l in form.labels) for f in FILTERS}


def bulk_outside(form: MappingForm, values: dict[str, str | None]) -> bool:
    """The secondary 「残りを対象外にする」: every unset person becomes 対象外 (kept as the
    coach's choice). True when it was pressed."""
    rest = sum(v is None for v in values.values())
    if not st.button(f"{BULK_OUTSIDE}（{rest}人）", key=f"pfb_bulk_{form.track.id}",
                     help=BULK_HELP, disabled=not rest):
        return False
    mine = kept(form)
    for label, value in values.items():
        if value is None:
            mine[label] = OUTSIDE
    return True


def _name_it(form: MappingForm, label: str, repo: Repo, sid: int) -> None:
    """「名前をつける」: a new person with the typed name, selected for this person."""
    tid = form.track.id
    field = f"pfb_name_{tid}_{label}"
    if st.session_state.get(_CLEAR, {}).pop(field, False):
        st.session_state[field] = ""          # the name was added: an empty field again
    with st.popover(NAME_IT, help=NAME_HELP.format(mark=mark_name(label))):
        name = st.text_input(NAME_FIELD, key=field, max_chars=review.MAX_NAME)
        if st.button(NAME_ADD, key=f"pfb_nameadd_{tid}_{label}"):
            try:
                pid = review.add_named_subject(repo, sid, name)
            except BusyError:
                st.error(theme.L("いまは解析が進んでいるため名前をつけられません。終わってから、もう一度押してください。"))
            except review.NameTaken as e:   # the fixed words localised, the typed name raw
                st.error(theme.L(review.NAME_TAKEN) + e.name)
            except ValueError as e:
                st.error(theme.L(str(e)))
            else:
                kept(form)[label] = pid
                st.session_state.setdefault(_CLEAR, {})[field] = True
                st.rerun()


def _card_html(form: MappingForm, label: str, value, names: dict[str, str]) -> str:
    head = f'<p class="pfb-person__mark">{escape(mark_name(label))}</p>'
    state = state_of(form, label, value)
    if state == "guess":
        head += theme.status_chip("guess", GUESS)
    elif state == "todo":
        head += theme.status_chip("your_turn", UNSET)
    body = ""
    if _proposed(form, label):
        reason = escape(form.reasons[label])
        if state == "guess":
            body = f'<p class="pfb-person__why">{reason}</p>'
        else:
            was = form.prefill.get(label)
            who = names.get(was) or form.name(was)
            body = f'<p class="pfb-person__was">推定: {escape(who)}（{reason}）</p>'
    return f'<div class="pfb-person"><div class="pfb-person__head">{head}</div>{body}</div>'


def person_grid(form: MappingForm, filt: str, *, repo: Repo, sid: int,
                values: dict[str, str | None] | None = None) -> dict[str, str | None]:
    """The cards of the people the filter shows, three per row; returns the choices of every
    person (the hidden ones as kept)."""
    values = dict(values if values is not None else current(form))
    names = form.names_for(values)     # the default names for exactly these choices

    def fmt(option) -> str:
        if option in form.students:
            return form.students[option]
        return names.get(option) or form.name(option)

    ckey = choices_key(form)
    shown = [l for l in form.labels if _matches(form, l, values[l], filt)]
    if not shown:
        st.html(f'<p class="pfb-gridnone">{NONE_HERE}</p>')
    for start in range(0, len(shown), COLUMNS):
        cols = st.columns(COLUMNS, gap="small")
        for col, label in zip(cols, shown[start:start + COLUMNS]):
            n = form.labels.index(label)
            value = values[label]
            with col, keyed.container(
                    key=f"pfb_person_{form.track.id}_{n}_{state_of(form, label, value)}"):
                crop = form.crops.get(label)
                if crop is None or not zoom.picture(
                        crop, alt=mark_name(label), key=f"crop_{form.track.id}_{label}"):
                    # the same space, so arriving crops do not move the cards
                    st.html(f'<p class="pfb-person__nophoto">{NO_PHOTO}</p>')
                st.html(_card_html(form, label, value, names))
                wkey = f"pfb_map_{form.track.id}_{label}"
                st.session_state[wkey] = value      # the select shows exactly this choice
                values[label] = st.selectbox(
                    mark_name(label), form.options_for(label), index=None, format_func=fmt,
                    placeholder=PLACEHOLDER, key=wkey, on_change=_touch,
                    args=(ckey, label, wkey), label_visibility="collapsed")
                _name_it(form, label, repo, sid)
    return values


def sheet(form: MappingForm, camera_name: str) -> None:
    """The people sheet, always in the same place: 「人物一覧の写真を見る」, open while there
    are no crops (the sheet is then the only picture of the people), as wide as the card,
    with 「大きく見る」."""
    with st.expander(SHEET_MORE, expanded=not form.crops):
        caption = SHEET_CAPTION.format(camera=camera_name)
        if form.sheet is None or not zoom.picture(
                form.sheet, alt=caption, caption=caption, name=SHEET_NAME,
                key=f"sheet_{form.track.id}"):
            st.info(theme.L(NO_SHEET))


# --- the roster ------------------------------------------------------------------------------

def roster_expander(repo: Repo, sid: int, domain: str, sport: str | None,
                    decided: bool = False) -> None:
    """「名簿を入れる・直す」 (「選手名簿を…」 for sports): apply_roster, then names again.
    `decided`: a camera's mapping is saved, and a new roster makes the coach choose again."""
    w = words(domain)
    with st.expander(f"{w.roster}を入れる・直す"):
        if decided:
            st.warning(theme.L(ROSTER_RESETS))
        suggested = [ROLE_LABELS[r] for r in roster_roles(domain, sport, False) if r != "name"]
        roster = roster_input(f"pfb_roster_{sid}", domain, suggested)
        if st.button(theme.L(ROSTER_APPLY), key=f"pfb_roster_apply_{sid}",
                     disabled=roster is None):
            try:
                apply_roster(repo, sid, roster, domain)
                request_names(repo, sid)     # apply_roster asks too; a pending one is kept
            except BusyError as e:   # the work is busy, or it is too late for a new roster
                st.error(theme.L(ROSTER_BUSY if str(e) == JOBS_BUSY else str(e)))
            except ValueError as e:
                st.error(theme.L(str(e)))
            else:
                flash(theme.L(ROSTER_DONE), raw=True)
                st.rerun()
