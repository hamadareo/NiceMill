"""読み手に合わせた文章 on step 6 (services.plain_text makes and keeps them; the worker rewords).

A coach picks whose eyes the handouts are written for: 13歳以上 (the approved text as it is), 9〜12歳 or
6〜8歳 (`plain_checks.LEVELS`). Until the picked version of a person is ready, their handout keeps
the approved text, and the screen says so. 「言いかえた文章を作る」 asks the worker for everybody
who has none (a line that refreshes itself shows the waiting, the page reruns when it is done);
「一人ずつ見る・直す」 shows each person's approved text and the reworded one side by side, where the
coach can write the version in their own words (it must pass the same checks, whose reasons are
said in plain words), ask again for that person alone, or keep the approved text for them.

管理者 and 評価する人 only (ui.access.can_edit); nothing for a work whose videos were removed. Every
change is made on a fresh connection (ui.common.get_repo: read-only without an editing session).
No keyed containers here (only widget keys).
"""
from __future__ import annotations

import logging
from html import escape
from typing import Callable

import streamlit as st

from pfb_app.services import plain_text
from pfb_app.services.plain_text import (FALLBACK, LABELS, LEVELS, MAKING, NONE, ORIGINAL,
                                         OUTDATED, READY, UNAVAILABLE)
from pfb_app.storage.db import Repo
from pfb_app.ui import access
from pfb_app.ui.common import flash, get_repo
from pfb_app.ui.wizard import theme

log = logging.getLogger(__name__)

TITLE = "読み手に合わせた文章"
GUIDE = ("承認した文章を、読む人の年齢に合わせて言いかえた版を作れます。書かれている意味や事実は変えません。"
         "元の文はそのまま残り、いつでも元の文にもどせます。")
WHO = "参加者に渡すPDFの文章"
MAKE = "言いかえた文章を作る"
REMAKE = "全員分を作り直す"
EACH = "一人ずつ見る・直す"
ORIGINAL_TEXT = "元の文（承認した文章）"
THIS_TEXT = "言いかえた文"
SAVE = "この文にする"
AGAIN = "この人だけ作り直す"
KEEP_ORIGINAL = "この人は元の文を使う"
MAKING_LINE = "{n}人の言いかえを作っています。できると、ここに出ます。"
READY_LINE = "{n}人の文章を言いかえました。PDFには、この文が入ります。"
NONE_READY = "まだ言いかえた文章がありません。「{make}」を押してください。それまでは、元の文が入ります。"
FAILED = "うまくいきませんでした。もう一度ためしても直らないときは、管理者に知らせてください。"
SAVED = "言いかえた文を保存しました。"
ASKED = "{n}人分の言いかえを頼みました。しばらくすると、ここに出ます。"
NOTHING_TO_DO = "作る人がいません。"
STATE_WORDS = {
    NONE: "まだ作っていません",
    MAKING: "作っています",
    READY: "言いかえた文を使います",
    FALLBACK: "意味を変えずに言いかえられなかったため、元の文を使います",
    OUTDATED: "元の文が変わりました。作り直してください",
    UNAVAILABLE: "承認した文章がありません",
}
STATE_CHIP = {NONE: "insufficient", MAKING: "processing", READY: "approved", FALLBACK: "your_turn",
              OUTDATED: "your_turn", UNAVAILABLE: "insufficient"}
SOURCE_WORDS = {"model": "AIが作った文", "teacher": "手で直した文"}
POLL_SEC = 3
HELP = [
    "「参加者に渡すPDFの文章」で、PDFに入れる文章を選びます。「13歳以上向け」は、承認した文章のままです。",
    "6〜8歳向け・9〜12歳向けを選んで「言いかえた文章を作る」を押すと、承認した文章を、読む人の年齢に合わせた"
    "言葉に言いかえます。書かれていること（動作・数字・名前）は変えず、新しいことは足しません。",
    "機械の確かめでは見つけられない、意味の小さなずれが残ることがあります。とくに6〜8歳向けは、目を通してから渡してください。",
    "言いかえた文章がまだできていない人、確かめを通らなかった人、承認した文が変わった人は、元の文がPDFに入ります。",
    "「一人ずつ見る・直す」で、元の文と言いかえた文を見比べて、自分の言葉に直せます。直した文も、同じ確かめを通ったものだけが入ります。",
]


def _note(text: str) -> None:
    st.html(f'<p class="pfb-dash-note">{escape(theme.L(text))}</p>')


def _write(action: Callable[[Repo], str | None]) -> bool:
    """Run `action` on a fresh connection; its returned sentence (if any) is shown next run.
    Refusals are shown in their own plain words. True when it was done."""
    if access.refuse_unless_edit():
        return False
    repo = get_repo()
    try:
        message = action(repo)
    except plain_text.Refused as e:
        flash(str(e), kind="warning")
        return False
    except Exception as e:   # noqa: BLE001 - the screen stays
        if access.refused(e):
            flash(access.VIEW_ONLY, kind="warning")
        else:
            log.exception("読み手に合わせた文章の操作に失敗しました")
            flash(FAILED, kind="warning")
        return False
    finally:
        repo.conn.close()
    if message:
        flash(message)
    return True


def _picked(aid: int) -> None:
    """The radio changed: keep the work's choice (and who keeps the approved text)."""
    level = st.session_state.get(f"pfb_plain_level_{aid}")
    if level not in plain_text.CHOICES:
        return

    def action(repo: Repo):
        plain_text.choose(repo, aid, level, plain_text.choice(repo, aid).off)
    _write(action)


def _keep_changed(aid: int, subject_id: int) -> None:
    on = bool(st.session_state.get(f"pfb_plain_keep_{subject_id}"))

    def action(repo: Repo):
        now = plain_text.choice(repo, aid)
        off = set(now.off) | {subject_id} if on else set(now.off) - {subject_id}
        plain_text.choose(repo, aid, now.level, off)
    _write(action)


def _make(aid: int, level: str, subject_ids=None, *, redo: bool = False) -> None:
    def action(repo: Repo):
        n = plain_text.request(repo, aid, level, subject_ids, redo=redo)
        return ASKED.format(n=n) if n else NOTHING_TO_DO
    _write(action)


def _save(subject_id: int, level: str, key: str) -> None:
    text = st.session_state.get(key, "")
    _write(lambda repo: (plain_text.save_edit(repo, subject_id, level, text), SAVED)[1])


@st.fragment(run_every=POLL_SEC)
def _waiting(aid: int, level: str) -> None:
    """「作っています」 (redrawn by itself; the page reruns when nobody is being made any more).
    A revoked or demoted session sees nothing more."""
    if not access.can_edit():
        return
    repo = get_repo()
    try:
        n = plain_text.making(repo, aid, level)
    finally:
        repo.conn.close()
    if not n:
        st.rerun()
    st.info(theme.L(MAKING_LINE.format(n=n)), icon=":material/schedule:")


def _person(repo: Repo, aid: int, subject: dict, group: str, level: str, several: bool,
            off: frozenset[int], original: str) -> None:
    shown = plain_text.state(repo, subject, level)
    name = f"{subject['display_name']}さん" + (f"（{group}）" if several else "")
    head = f"**{name}**"
    st.markdown(head)
    chip = theme.status_chip(STATE_CHIP[shown.kind], STATE_WORDS[shown.kind])
    source = SOURCE_WORDS.get(shown.source or "", "")
    st.html(chip + (f' <span class="pfb-dash-note">{escape(source)}</span>' if source else ""))
    left, right = st.columns(2)
    with left:
        st.caption(ORIGINAL_TEXT)
        st.text(original)
    with right:
        st.caption(THIS_TEXT)
        run_id = shown.run.id if shown.run is not None else 0
        key = f"pfb_plain_text_{subject['id']}_{level}_{run_id}_{shown.kind}"
        st.text_area(THIS_TEXT, value=shown.text or "", key=key, height=140,
                     label_visibility="collapsed", disabled=shown.kind == MAKING)
        st.button(SAVE, key=f"pfb_plain_save_{subject['id']}_{level}", icon=":material/save:",
                  disabled=shown.kind == MAKING, on_click=_save, args=(subject["id"], level, key))
    again, keep = st.columns(2)
    with again:
        st.button(AGAIN, key=f"pfb_plain_again_{subject['id']}_{level}", type="tertiary",
                  icon=":material/refresh:", disabled=shown.kind == MAKING,
                  on_click=_make, args=(aid, level, [subject["id"]]), kwargs={"redo": True})
    with keep:
        st.checkbox(theme.L(KEEP_ORIGINAL), value=subject["id"] in off,
                    key=f"pfb_plain_keep_{subject['id']}", on_change=_keep_changed,
                    args=(aid, subject["id"]))


def section(repo: Repo, work: dict) -> None:
    """Step 6's part (nothing for anyone but 管理者 and 評価する人, for a work whose videos were
    removed, or while nobody has an approved text)."""
    if not access.can_edit() or work.get("media_removed_at"):
        return
    aid = work["id"]
    people = plain_text.people(repo, aid)
    if not people:
        return
    picked = plain_text.choice(repo, aid)
    st.html(f'<h3 class="pfb-subhead">{escape(theme.L(TITLE))}</h3>')
    _note(GUIDE)
    options = list(plain_text.CHOICES)
    st.radio(theme.L(WHO), options, index=options.index(picked.level), horizontal=True,
             format_func=LABELS.get, key=f"pfb_plain_level_{aid}", on_change=_picked, args=(aid,))
    if picked.level == ORIGINAL:
        return
    level = picked.level
    making = plain_text.making(repo, aid, level)
    use = plain_text.usage(repo, aid)
    if making:
        _waiting(aid, level)
    elif use.plain:
        st.success(theme.L(READY_LINE.format(n=len(use.plain))), icon=":material/check_circle:")
    else:
        st.info(theme.L(NONE_READY.format(make=MAKE)), icon=":material/info:")
    if use.original and not making:
        for name, why in use.original:
            st.caption(f"{name}さん：{theme.L(why)}")
    if not making and plain_text.pending(repo, aid, level):
        st.button(theme.L(MAKE), key=f"pfb_plain_make_{aid}", icon=":material/auto_fix_high:",
                  on_click=_make, args=(aid, level))
    elif not making and use.plain:
        st.button(theme.L(REMAKE), key=f"pfb_plain_remake_{aid}", type="tertiary",
                  icon=":material/refresh:", on_click=_make, args=(aid, level),
                  kwargs={"redo": True})
    several = len(repo.list_submissions(aid)) > 1
    with st.expander(EACH):
        from pfb_app.pipeline import plain as pipeline_plain   # noqa: PLC0415
        for subject, group in people:
            found = pipeline_plain.approved_text(repo, subject)
            if found is None:
                continue
            _person(repo, aid, subject, group, level, several, picked.off, found[1])
            st.divider()

