"""「映っている人」 on 確認して承認: who was evaluated, names optional.

One card per person with a participant id, two per row (the evaluated ones first, those
marked 「評価しない」 last; roster rows that are excluded and that no camera maps get no card):
the picture (large, with 「大きく見る」) or 「写真なし」, the name (with 「背番号から」 under a name
given from a read number), 「名前をつける」 (a text field, or a select of the roster's names no
evaluated person uses plus 「自分で入力」) and 「この人は評価しない」 / 「評価に戻す」. No button is
primary: 承認する is the one primary of the screen.

On a game, a person the participation rule marked unsure carries 「要確認」; the people it
left out (spectators, the bench, officials) follow under 「試合に参加していない人」, each with
their picture, why, and 「この人も評価する」 (services.auto_people.include_person). The people
the automatic choice left out because they were on screen only briefly follow under
「映っている時間が短い人」 with their picture, how long they were on screen and the same button.
"""
from html import escape

import streamlit as st

from pfb_app.services import auto_people, review
from pfb_app.services.auto_people import (is_evaluated, label_crop, name_notes,
                                          participation_view, person_crops, rename_person,
                                          set_person_excluded, short_people, unused_roster)
from pfb_app.storage.db import Repo
from pfb_app.ui import keyed
from pfb_app.ui.common import flash
from pfb_app.ui.wizard import zoom

COLUMNS = 2          # crops at least 240 px wide in the step card
NAME_IT = "名前をつける"
NAME_FIELD = "名前"
OWN_INPUT = "自分で入力"
EXCLUDE = "この人は評価しない"
RESTORE = "評価に戻す"
EXCLUDED = "評価しない"
NO_PHOTO = "写真なし"
UNNAMED = "名前のない人"   # the card's heading (and picture) when the person has no name
NAME_TAKEN = "同じ名前の人がすでにいます。別の名前にしてください。"
RENAMED = "名前をつけました。"
EXCLUDED_DONE = "評価しない人にしました。"
RESTORED = "評価に戻しました。"
CHECK_NOTE = "要確認：試合に出ていたか確かめてください。出ていなければ「この人は評価しない」を押します。"
LEFT_OUT = "試合に参加していない人"
LEFT_OUT_GUIDE = ("コートの外にいた人や、ずっと同じ場所にいた人です（見ている人・控え・審判など）。"
                  "評価には入っていません。試合に出ていた人は「この人も評価する」を押してください。")
LEFT_OUT_WHY = {"off_court": "コートの外にいました", "static": "ずっと同じ場所にいました"}
LEFT_OUT_DEFAULT = "試合に出ていないようです"
SHORT = "映っている時間が短い人"
SHORT_GUIDE = ("映っている時間が短かったため、評価には入っていません。"
               "評価したい人は「この人も評価する」を押してください。")
SHORT_FOR = "約{n}秒映っていました"
INCLUDE = "この人も評価する"
INCLUDED = "評価に入れました。この人の分も、もう一度解析します。"


def people_panel(repo: Repo, settings, sub: dict) -> None:
    sid = sub["id"]
    subjects = repo.list_subjects(sid)
    idle = {s["id"] for s in unused_roster(repo, sid)}
    people = [s for s in subjects if s.get("participant_id") and s["id"] not in idle
              and not s.get("erased_at")]   # removed on request (データの扱い): no card
    people = [s for s in people if is_evaluated(s)] + [s for s in people if not is_evaluated(s)]
    if not people:
        return
    used = {review.name_key(s["display_name"]) for s in subjects if is_evaluated(s)}
    roster = [s["display_name"] for s in review.roster_subjects(subjects, sub.get("meta"))
              if review.name_key(s["display_name"]) not in used]
    crops = person_crops(repo, settings, sid)
    notes = name_notes(repo, sid)
    view = participation_view(repo, sid)
    for start in range(0, len(people), COLUMNS):
        for col, person in zip(st.columns(COLUMNS), people[start:start + COLUMNS]):
            with col, keyed.container(key=f"pfb_pcard_{person['id']}"):
                _card(repo, person, crops.get(person["id"]), roster, notes.get(person["id"]),
                      check=person["id"] in view.check)
    if view.not_playing:
        _left_out(repo, settings, sub, view.not_playing)
    short = short_people(repo, sid).people
    if short:
        _short(repo, settings, sub, short)


def _left_out(repo: Repo, settings, sub: dict, people) -> None:
    """「試合に参加していない人」: the people the participation rule left out, with their
    picture and 「この人も評価する」."""
    st.html(f'<p class="pfb-person__group"><strong>{LEFT_OUT}</strong></p>'
            f'<p class="pfb-person__why">{escape(LEFT_OUT_GUIDE)}</p>')
    _unevaluated(repo, settings, sub, people, "pout",
                 lambda p: LEFT_OUT_WHY.get(p.reason, LEFT_OUT_DEFAULT))


def _short(repo: Repo, settings, sub: dict, people) -> None:
    """「映っている時間が短い人」: the people the automatic choice did not evaluate because
    they were on screen briefly, with their picture, for how long and 「この人も評価する」."""
    st.html(f'<p class="pfb-person__group"><strong>{SHORT}</strong></p>'
            f'<p class="pfb-person__why">{escape(SHORT_GUIDE)}</p>')
    _unevaluated(repo, settings, sub, people, "pshort",
                 lambda p: SHORT_FOR.format(n=max(1, round(p.seconds))) if p.seconds is not None else "")


def _unevaluated(repo: Repo, settings, sub: dict, people, kind: str, why) -> None:
    """Cards of tracked people nobody evaluates: picture, `why(person)` and 「この人も評価する」
    (`person`: camera, label, track_run_id)."""
    for start in range(0, len(people), COLUMNS):
        for col, person in zip(st.columns(COLUMNS), people[start:start + COLUMNS]):
            key = f"{person.camera}_{person.label}"
            with col, keyed.container(key=f"pfb_{kind}_{key}"):
                track = repo.get_run(person.track_run_id)
                crop = label_crop(settings, sub, person.camera, person.label, track)
                if crop is None or not zoom.picture(crop, alt=UNNAMED, key=f"{kind}_{key}"):
                    st.html(f'<p class="pfb-person__nophoto">{NO_PHOTO}</p>')
                text = why(person)
                if text:
                    st.html(f'<p class="pfb-person__why">{escape(text)}</p>')
                if st.button(INCLUDE, key=f"pfb_people_incl_{key}"):
                    try:
                        auto_people.include_person(repo, settings, sub["id"], person.camera,
                                                   person.label)
                    except (ValueError, review.BusyError) as exc:
                        st.error(str(exc))
                        return
                    flash(INCLUDED)
                    st.rerun()


def _card(repo: Repo, person: dict, crop, roster: list[str], note: str | None = None, *,
          check: bool = False) -> None:
    pid = person["id"]
    shown = person["display_name"] or UNNAMED
    if crop is None or not zoom.picture(crop, alt=shown, key=f"person_{pid}"):
        st.html(f'<p class="pfb-person__nophoto">{NO_PHOTO}</p>')
    st.html(f'<p class="pfb-person__mark"><strong>{escape(shown)}</strong>'
            + ("" if is_evaluated(person) else f"（{EXCLUDED}）") + "</p>")
    if note:   # where the name came from (the mapping form's reason line)
        st.html(f'<p class="pfb-person__why">{escape(note)}</p>')
    if check and is_evaluated(person):
        st.html(f'<p class="pfb-person__why">{escape(CHECK_NOTE)}</p>')
    name = None
    picked_roster = False
    if roster:
        options = [*roster, OWN_INPUT]
        picked = st.selectbox(NAME_FIELD, options, index=len(options) - 1,
                              key=f"pfb_people_pick_{pid}", label_visibility="collapsed")
        name = None if picked == OWN_INPUT else picked
        picked_roster = name is not None
    if name is None:
        name = st.text_input(NAME_FIELD, key=f"pfb_people_name_{pid}",
                             label_visibility="collapsed", placeholder=NAME_FIELD)
    if st.button(NAME_IT, key=f"pfb_people_rename_{pid}"):
        try:
            rename_person(repo, pid, name, roster=picked_roster)
        except review.NameTaken:
            st.error(NAME_TAKEN)
            return
        except (ValueError, review.BusyError) as exc:
            st.error(str(exc))
            return
        flash(RENAMED)
        st.rerun()
    excluded = not is_evaluated(person)
    if st.button(RESTORE if excluded else EXCLUDE, key=f"pfb_people_excl_{pid}"):
        try:
            set_person_excluded(repo, pid, not excluded)
        except (ValueError, review.BusyError) as exc:
            st.error(str(exc))
            return
        flash(RESTORED if excluded else EXCLUDED_DONE)
        st.rerun()
