"""成長の記録: one person across several works (services.growth links, export.growth reads).

Two screens on one page (pages/growth.py), chosen by the address:
- the list (no query): 「元に戻す」 for the last accepted suggestion; the suggestions, each with
  「同じ人としてつなぐ」 and 「別の人」 (nothing is ever linked without that press); the records;
  each work's 「チーム・グループ」 (numbers like #5 are suggested only within one). The one
  primary: 「この人の記録を見る」 for the person picked.
- one person (`?person=<record>` or `?subject=<person in a work>`; a person in no record is
  shown alone): the statements, the counts grid, each work (date, activity, the points noted,
  quotes of the approved text, 「この作業を外す」), the suggestions for them and 「つなぐ」 (the
  one primary) to link someone of another work by hand.

Only approved texts are counted or quoted (export.growth). The words are the general ones
(人, 作業): one record may hold works of several kinds of activity.
"""
from __future__ import annotations

from html import escape

import streamlit as st

from pfb_app.export import charts
from pfb_app.export import growth as record
from pfb_app.services import growth
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import flash, get_repo, show_flash
from pfb_app.ui.wizard import svg, theme

TITLE = "成長の記録"
GUIDE = "同じ人の、いくつかの作業での記録をまとめて見られます。承認した文章だけを使います。"
EMPTY = "まだ記録がありません。作業で文章を承認すると、ここで見られます。"
OPEN = "この人の記録を見る"
PICK = "見る人"
SUGGEST_HEAD = "つながりの候補（{n}件）"
SUGGEST_NOTE = ("同じ人かもしれない人です。同じ人なら「同じ人としてつなぐ」を押してください。"
                "自動ではつなぎません。" + growth.APPROVED_ONLY)
ACCEPT = "同じ人としてつなぐ"
DECLINE = "別の人"
UNDO = "元に戻す"
ACCEPTED = "「{a}」と「{b}」を同じ人としてつなぎました。"
RECORDS_HEAD = "記録（{n}人）"
OPEN_ONE = "開く"
GROUPS = "作業のチーム・グループ（背番号でつなぐとき）"
GROUPS_NOTE = ("背番号（#5 など）の人は、同じチーム・グループを入れた作業どうしでだけ、"
               "同じ人の候補になります。")
SAVE_GROUPS = "チーム・グループを保存"
BACK = "記録の一覧へ"
PERSON_GUIDE = "承認した文章と、その文章のもとになった場面の数だけを使っています。"
CHART_NOTE = "数字は回数です。色が濃いほど多く記録されています。"
WORKS_HEAD = "作業ごとの記録"
SUGGESTED = "候補からつなぎました"
UNLINK = "この作業を外す"
LINK_HEAD = "ほかの作業の人をつなぐ"
LINK_NOTE = ("同じ人なのに候補に出ないとき（名前のない人など）は、ここでつなぎます。"
             + growth.APPROVED_ONLY)
LINK_PICK = "つなぐ人"
LINK = "つなぐ"
LINK_NOBODY = "つなげる人がいません。"
MOVE_NOTE = "ほかの記録に入っている人を選ぶと、この記録へ移します。"
RENAME = "記録の名前を変える"
RENAME_GO = "この名前にする"
MAX_SHOWN = 10
UNDO_KEY = "_pfb_growth_undo"     # (Accepted, the sentence, the screen) of the last accept

HELP = [
    "同じ人が映っている作業をつなぐと、作業ごとの記録が一人分にまとまります。",
    "名前（名簿の名前や、つけた名前）が同じ人は「つながりの候補」に出ます。"
    "同じ人なら「同じ人としてつなぐ」、違うときは「別の人」を押します。自動ではつなぎません。",
    "背番号の人（#5 など）は、「作業のチーム・グループ」に同じ名前を入れた作業どうしでだけ候補になります。",
    "「参加者1」のような名前の人は候補に出ません。記録の画面の「ほかの作業の人をつなぐ」でつなぎます。",
    "数と文章は、承認した文章だけから出します。まちがえてつないだときは「この作業を外す」で外せます。",
    "PDFの「成長のまとめ」のページは、「PDFを受け取る」の画面で入れるかどうかを選べます（はじめは入れません）。",
]
CSS = """<style>
.pfb-growth-lines { margin: 0 0 1rem; padding-left: 1.25rem; color: var(--pfb-ink); }
.pfb-growth-lines li { margin: 0 0 0.375rem; }
.pfb-growth-pair { margin: 0; color: var(--pfb-ink); }
.pfb-growth-why, .pfb-growth-meta { margin: 0.125rem 0 0; color: var(--pfb-muted);
  font-size: 0.875rem; }
.pfb-growth-work { margin: 0; padding: 0.5rem 0; border-top: 1px solid var(--pfb-line); }
.pfb-growth-work__title { margin: 0; font-weight: 700; color: var(--pfb-ink); }
.pfb-growth-work__points { margin: 0.125rem 0 0; color: var(--pfb-ink); }
.pfb-growth-quote { margin: 0.25rem 0 0; padding-left: 0.75rem;
  border-left: 3px solid var(--pfb-line); color: var(--pfb-ink); }
.pfb-growth-chart img { max-width: 100%; height: auto; }
</style>"""


def _works(people: list[growth.Member]) -> int:
    return len({m.assignment_id for m in people})


def _who(m: growth.Member) -> str:
    where = m.work_title + (f"・{m.group_label}" if m.group_label else "")
    return f"{m.name}（{where}・{m.when}）"


def _screen() -> str:
    """The screen shown, as its address says (the undo belongs to one screen)."""
    return "&".join(f"{k}={st.query_params.get(k)}" for k in ("person", "subject")
                    if st.query_params.get(k) is not None)


def _go(**query) -> None:
    st.session_state.pop(UNDO_KEY, None)     # 「元に戻す」 stays on the screen it was made on
    st.query_params.clear()
    for k, v in query.items():
        st.query_params[k] = str(v)
    st.rerun()


# --- suggestions -----------------------------------------------------------------------------

def _suggestion_rows(repo: Repo, found: list[growth.Suggestion], by_id: dict) -> None:
    if not found:
        return
    st.html(f'<h3 class="pfb-subhead">{escape(SUGGEST_HEAD.format(n=len(found)))}</h3>'
            f'<p class="pfb-growth-why">{escape(SUGGEST_NOTE)}</p>')
    for s in found[:MAX_SHOWN]:
        a, b = by_id[s.subject_id], by_id[s.other_id]
        with keyed.container(border=True, key=f"pfb_growth_sug_{s.subject_id}_{s.other_id}"):
            st.html(f'<p class="pfb-growth-pair">{escape(_who(a))} と {escape(_who(b))}</p>'
                    f'<p class="pfb-growth-why">理由：{escape(s.reason)}</p>')
            with st.container(horizontal=True):
                # callbacks bound to the pair shown: a press is never lost or moved to another
                # pair when the list changed meanwhile (another tab)
                st.button(ACCEPT, key=f"pfb_growth_yes_{s.subject_id}_{s.other_id}",
                          on_click=_accept, args=(s.subject_id, s.other_id, a.name, b.name))
                st.button(DECLINE, key=f"pfb_growth_no_{s.subject_id}_{s.other_id}",
                          type="tertiary", on_click=_decline, args=(s.subject_id, s.other_id))
    if len(found) > MAX_SHOWN:
        st.caption(f"ほかに{len(found) - MAX_SHOWN}件あります。上の候補に答えると出てきます。")


def _accept(subject_id: int, other_id: int, a: str, b: str) -> None:
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    repo = get_repo()
    try:
        done = growth.accept(repo, subject_id, other_id)
    except ValueError as e:
        st.session_state.pop(UNDO_KEY, None)
        flash(str(e), kind="warning")
        return
    finally:
        repo.conn.close()
    st.session_state[UNDO_KEY] = (done, ACCEPTED.format(a=a, b=b), _screen())


def _decline(subject_id: int, other_id: int) -> None:
    if access.refuse_unless_edit():   # a callback runs before the gate
        return
    repo = get_repo()
    try:
        growth.decline(repo, subject_id, other_id)
    finally:
        repo.conn.close()
    st.session_state.pop(UNDO_KEY, None)


def _undo_bar(repo: Repo) -> None:
    last = st.session_state.get(UNDO_KEY)
    if not last or last[2] != _screen():
        return
    done, sentence, _ = last
    with keyed.container(horizontal=True, vertical_alignment="center", key="pfb_growth_undo"):
        st.html(f'<p class="pfb-growth-pair">{escape(sentence)}</p>')
        if st.button(UNDO, key="pfb_growth_undo_go", icon=":material/undo:"):
            growth.undo(repo, done)
            st.session_state.pop(UNDO_KEY, None)
            st.rerun()


# --- the list ----------------------------------------------------------------------------------

def _list(repo: Repo) -> None:
    everyone = growth.members(repo)
    by_id = {m.subject_id: m for m in everyone}
    with theme.step_card(TITLE, GUIDE, key="growth"):
        if not everyone:
            st.info(EMPTY)
            st.button(OPEN, key="pfb_growth_open", type="primary", disabled=True)
            return
        records = repo.growth_people()
        linked: dict[int, list[growth.Member]] = {}
        for m in everyone:
            if m.person_id is not None:
                linked.setdefault(m.person_id, []).append(m)
        options = {f"p{p['id']}": f"{p['name']}（作業{_works(linked[p['id']])}件）"
                   for p in records if linked.get(p["id"])}
        options.update({m.subject_id: _who(m) for m in everyone if m.person_id is None})
        keys = list(options)
        if st.session_state.get("pfb_growth_pick") not in keys:
            st.session_state.pop("pfb_growth_pick", None)
        chosen = st.selectbox(PICK, keys, format_func=options.get, key="pfb_growth_pick")
        if st.button(OPEN, key="pfb_growth_open", type="primary"):
            if isinstance(chosen, str):
                _go(person=chosen[1:])
            _go(subject=chosen)
        _undo_bar(repo)
        _suggestion_rows(repo, growth.suggestions(repo, everyone=everyone), by_id)
        shown = [p for p in records if linked.get(p["id"])]
        if shown:
            st.html(f'<h3 class="pfb-subhead">{escape(RECORDS_HEAD.format(n=len(shown)))}</h3>')
            for p in shown:
                works = linked[p["id"]]
                text, action = st.columns([4, 1], vertical_alignment="center")
                with text:
                    st.html(f'<p class="pfb-growth-pair"><strong>{escape(p["name"])}</strong></p>'
                            f'<p class="pfb-growth-meta">作業{_works(works)}件・いちばん新しい作業 '
                            f'{escape(works[-1].when)}</p>')
                with action:
                    if st.button(OPEN_ONE, key=f"pfb_growth_open_{p['id']}", width="stretch"):
                        _go(person=p["id"])
    _groups(repo)


def _groups(repo: Repo) -> None:
    works = [w for w in repo.list_assignments() if w["kind"] == "video"]
    if not works:
        return
    with st.expander(GROUPS):
        st.caption(GROUPS_NOTE)
        for w in works:
            st.text_input(f"{w['title']}（{growth.date_text(w.get('created_at'))}）",
                          value=w.get("growth_group") or "", key=f"pfb_growth_group_{w['id']}",
                          max_chars=growth.MAX_GROUP)
        if st.button(SAVE_GROUPS, key="pfb_growth_groups_save"):
            for w in works:
                growth.set_group(repo, w["id"], st.session_state.get(f"pfb_growth_group_{w['id']}"))
            flash("チーム・グループを保存しました。")
            st.rerun()


# --- one person --------------------------------------------------------------------------------

def _chart(rec: record.Record) -> None:
    columns, rows = record.grid(rec)
    drawn = charts.growth_grid(columns, rows, title="観点ごとのよかった場面の数")
    if not drawn:
        return
    desc = "。".join(f"{cat.label}：" + "、".join(f"{c} {v}回" for c, v in zip(columns, values))
                    for cat, values in rows)
    st.html('<h3 class="pfb-subhead">観点ごとのよかった場面の数</h3>'
            f'<div class="pfb-growth-chart">{svg.img(drawn, "観点ごとのよかった場面の数。" + desc)}</div>'
            f'<p class="pfb-growth-why">{escape(CHART_NOTE)}</p>')
    if len(rec.works) > record.GRID_WORKS:
        st.caption(f"図は新しい作業{record.GRID_WORKS}件です。下の一覧には全部の作業があります。")


def _work(repo: Repo, rec: record.Record, e: record.Entry) -> None:
    head = f"{e.when}・{e.activity}"
    title = f"「{e.title}」" + (f"（{e.group_label}）" if e.group_label else "")
    if e.name != rec.name:
        title += f"　この作業での名前：{e.name}"
    parts = [f'<p class="pfb-growth-meta">{escape(head)}</p>',
             f'<p class="pfb-growth-work__title">{escape(title)}</p>']
    if e.approved:
        points = "・".join(f"{label} {n}回" for label, n in e.counts.items())
        parts.append(f'<p class="pfb-growth-work__points">'
                     f'{escape(points or "観点ごとの場面は記録されていません。")}</p>')
        parts += [f'<p class="pfb-growth-quote">「{escape(q)}」</p>' for q in e.quotes[:2]]
    else:
        parts.append(f'<p class="pfb-growth-why">{escape(record.NOT_APPROVED)}</p>')
    if e.how == "suggested":
        parts.append(f'<p class="pfb-growth-why">{escape(SUGGESTED)}</p>')
    st.html(f'<div class="pfb-growth-work">{"".join(parts)}</div>')
    if rec.person_id is not None:
        if st.button(UNLINK, key=f"pfb_growth_unlink_{e.subject_id}", type="tertiary",
                     icon=":material/link_off:"):
            growth.unlink(repo, e.subject_id)
            left = repo.growth_person(rec.person_id)
            if left is None:
                _go(subject=e.subject_id)
            st.rerun()


def _manual(repo: Repo, rec: record.Record, everyone: list[growth.Member]) -> None:
    """Link someone by hand: approved people of works not in the record yet (a person with no
    approved text in their work can neither start a record nor join one)."""
    mine = {e.subject_id for e in rec.entries}
    works = {e.assignment_id for e in rec.entries}
    st.html(f'<h3 class="pfb-subhead">{escape(LINK_HEAD)}</h3>'
            f'<p class="pfb-growth-why">{escape(LINK_NOTE)}</p>')
    if rec.person_id is None and not any(e.approved for e in rec.entries):
        st.caption(growth.APPROVED_ONLY)
        st.button(LINK, key="pfb_growth_link", type="primary", disabled=True)
        return
    choices = {m.subject_id: _who(m) for m in everyone
               if m.approved and m.subject_id not in mine and m.assignment_id not in works}
    if not choices:
        st.caption(LINK_NOBODY)
        st.button(LINK, key="pfb_growth_link", type="primary", disabled=True)
        return
    keys = list(choices)
    if st.session_state.get("pfb_growth_link_pick") not in keys:
        st.session_state.pop("pfb_growth_link_pick", None)
    chosen = st.selectbox(LINK_PICK, keys, format_func=choices.get, key="pfb_growth_link_pick")
    if any(m.person_id is not None for m in everyone if m.subject_id == chosen):
        st.caption(MOVE_NOTE)
    if st.button(LINK, key="pfb_growth_link", type="primary", icon=":material/link:"):
        try:
            person = rec.person_id
            if person is None:
                person = growth.link_manual(repo, rec.entries[0].subject_id, None)
            growth.link_manual(repo, chosen, person)
        except ValueError as e:
            flash(str(e), kind="warning")
            st.rerun()
        flash("つなぎました。")
        _go(person=person)


def _rename(repo: Repo, rec: record.Record) -> None:
    if rec.person_id is None:
        return
    with st.popover(RENAME, icon=":material/edit:"):
        name = st.text_input(RENAME, value=rec.name, key=f"pfb_growth_name_{rec.person_id}",
                             max_chars=growth.MAX_NAME, label_visibility="collapsed")
        if st.button(RENAME_GO, key=f"pfb_growth_name_go_{rec.person_id}"):
            try:
                growth.rename(repo, rec.person_id, name)
            except ValueError as e:
                flash(str(e), kind="warning")
            st.rerun()


def _person(repo: Repo, rec: record.Record) -> None:
    everyone = growth.members(repo)
    by_id = {m.subject_id: m for m in everyone}
    if st.button(BACK, key="pfb_growth_back", type="tertiary", icon=":material/arrow_back:"):
        _go()
    with theme.step_card(f"{rec.name}さんの{TITLE}", PERSON_GUIDE, key="growth_person"):
        lines = "".join(f"<li>{escape(line)}</li>" for line in record.statements(rec))
        st.html(f'<ul class="pfb-growth-lines">{lines}</ul>')
        _chart(rec)
        st.html(f'<h3 class="pfb-subhead">{escape(WORKS_HEAD)}</h3>')
        for e in rec.entries:
            _work(repo, rec, e)
        _undo_bar(repo)
        if rec.person_id is not None:
            found = growth.suggestions(repo, person_id=rec.person_id, everyone=everyone)
        else:
            found = growth.suggestions(repo, subject_id=rec.entries[0].subject_id,
                                       everyone=everyone)
        _suggestion_rows(repo, found, by_id)
        _manual(repo, rec, everyone)
        _rename(repo, rec)


def _number(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def render(repo: Repo) -> None:
    theme.inject_theme()
    theme.set_domain("school")     # the general words: a record may hold several activities
    st.html(CSS)
    show_flash()
    person = _number(st.query_params.get("person"))
    subject = _number(st.query_params.get("subject"))
    rec = None
    if person is not None:
        rec = record.for_person(repo, person)
    elif subject is not None:
        try:
            rec = record.for_subject(repo, subject)
        except (KeyError, StopIteration):
            rec = None
    if (person is not None or subject is not None) and rec is None:
        st.info("この記録は見つかりませんでした（外したか、作業を削除した可能性があります）。")
    if rec is None:
        _list(repo)
    else:
        _person(repo, rec)
    theme.help_expander(HELP)
