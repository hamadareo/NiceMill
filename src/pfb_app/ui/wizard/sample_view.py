"""「見本の作業」: the sample work, read-only, walked through the same six steps as a real one.

Every step shows what the coach would see there, filled with the fictional sample
(services.sample_work, pictures from sample_art); nothing on it can be pressed except
「新しく始める」 (the one primary), 「見本を消す」 and 「ホームへ」. It reuses the step screens'
own pieces (the stepper, the content check's board, the feedback text, the evidence list,
the hanamaru stamp), so what the sample shows is what the real screens look like.
"""
from html import escape

import streamlit as st

from pfb_app.services import sample_work
from pfb_app.storage.db import Repo
from pfb_app.ui import access, keyed
from pfb_app.ui.common import flash, show_flash
from pfb_app.ui.wizard import help as guide
from pfb_app.ui.wizard import sample_art, theme
from pfb_app.ui.wizard.progress import CHECK_FIRST, CHECK_VIDEOS, TRACKING_STEPS

READ_ONLY = "見本（見るだけ）"
VIEW_ONLY = "見るだけの見本なので、ここで文章を直したり承認したりはできません。"
START = "新しく始める"
REMOVE = "見本を消す"
REMOVED = "見本を消しました。「はじめての方へ」から、いつでも戻せます。"
APPROVAL_NOTE = {
    "as_is": "下書きのまま承認しました。",
    "edited": "コーチが言い回しを直してから承認しました。",
    "short": "映っている場面が少ないため、短い文で承認しました。",
}

CSS = """<style>
.pfb-sample { max-width: 52rem; min-width: 0; margin: 0 auto; font-family: var(--pfb-font-ui);
  color: var(--pfb-ink); overflow-wrap: anywhere; }
.pfb-sample__head { margin: 0.5rem 0 0.25rem; }
.pfb-sample__title { margin: 0; font-size: 1.75rem; line-height: 1.4; font-weight: 700;
  font-feature-settings: "palt"; }
.pfb-sample__meta { margin: 0.25rem 0 0; color: var(--pfb-muted); display: flex; flex-wrap: wrap;
  align-items: center; gap: 0.25rem 0.75rem; }
.pfb-sample__about { margin: 0.75rem 0 1.25rem; padding: 0.6rem 0.9rem; border: 1px dashed var(--pfb-line);
  border-radius: 8px; background: #FFFFFF; }
.pfb-sample__step { display: grid; grid-template-columns: 2.5rem 1fr; gap: 0 1rem;
  margin: 0 0 2rem; }
.pfb-sample__num { width: 2.25rem; height: 2.25rem; border-radius: 50%; background: var(--pfb-board);
  color: #FFFFFF; font-family: var(--pfb-font-hand); font-weight: 600; font-size: 1.125rem;
  line-height: 2.25rem; text-align: center; }
.pfb-sample__body { min-width: 0; }
.pfb-sample__h { margin: 0.2rem 0 0; font-size: 1.25rem; font-weight: 700; }
.pfb-sample__guide { margin: 0.2rem 0 0.75rem; color: var(--pfb-muted); }
.pfb-sample__scene { display: block; width: 100%; max-width: 300px; height: auto; border-radius: 10px; }
.pfb-sample__board { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 300px);
  gap: 1rem; align-items: center; }
.pfb-sample__people { list-style: none; margin: 0; padding: 0; display: grid;
  grid-template-columns: repeat(auto-fill, minmax(9rem, 1fr)); gap: 0.75rem; }
.pfb-sample__people li { margin: 0; display: flex; align-items: center; gap: 0.6rem; padding: 0.5rem;
  background: #FFFFFF; border: 1px solid var(--pfb-line); border-radius: 10px; }
.pfb-sample__face { width: 48px; height: 48px; flex: none; }
.pfb-sample__who { font-weight: 700; }
.pfb-sample__num-small { display: block; color: var(--pfb-muted); font-size: 0.8125rem; font-weight: 500; }
.pfb-sample__stages { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column;
  gap: 0.35rem; }
.pfb-sample__stages li { margin: 0; display: flex; align-items: center; gap: 0.75rem; }
.pfb-sample__person { margin: 0 0 1.5rem; padding: 1rem 1.25rem; background: #FFFFFF;
  border: 1px solid var(--pfb-line); border-radius: 12px; }
.pfb-sample__person .pfb-who { margin-top: 0; }
.pfb-sample__note { margin: 0 0 0.5rem; color: var(--pfb-muted); font-size: 0.9375rem; }
.pfb-sample__draft { margin: 0 0 0.75rem; color: var(--pfb-muted); font-size: 0.9375rem; }
.pfb-sample__draft summary { cursor: pointer; color: var(--pfb-board); font-weight: 500; }
.pfb-sample__moments { list-style: none; margin: 0.25rem 0 0; padding: 0; display: grid;
  grid-template-columns: repeat(auto-fill, minmax(15rem, 1fr)); gap: 1rem; }
.pfb-sample__moments li { margin: 0; }
.pfb-sample__moments .pfb-evidence { border-top: 0; padding-top: 0.25rem; }
.pfb-sample__label { margin: 0.5rem 0 0; font-weight: 700; color: var(--pfb-ink); }
.pfb-sample__sheet { padding: 1.5rem 1.75rem; background: #FFFFFF; border: 1px solid var(--pfb-line);
  border-radius: 4px; box-shadow: 0 1px 0 var(--pfb-line), 0 4px 0 -2px #FFFFFF, 0 5px 0 -2px var(--pfb-line); }
.pfb-sample__sheet h4 { margin: 0 0 0.25rem; font-size: 1.0625rem; }
.pfb-sample__sheet p { margin: 0 0 1rem; line-height: 1.85; }
.pfb-sample__sheet p:last-child { margin-bottom: 0; }
@media (max-width: 40rem) {
  .pfb-sample .pfb-stepper { display: none; }   /* the numbered steps below say the same */
  .pfb-sample__board { grid-template-columns: 1fr; }
  .pfb-sample__people { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .pfb-sample__step { grid-template-columns: 1fr; }
  .pfb-sample__num { margin-bottom: 0.35rem; }
}
</style>"""


def steps(work: sample_work.SampleWork) -> list[str]:
    """The sample's steps, named as a work started from the drop zone names them."""
    return [CHECK_FIRST, CHECK_VIDEOS.format(n=work.cameras), *TRACKING_STEPS[2:]]


def _step(n: int, title: str, guide_text: str, body: str) -> str:
    return (f'<section class="pfb-sample__step" aria-label="{n}. {escape(title)}">'
            f'<div class="pfb-sample__num" aria-hidden="true">{n}</div>'
            f'<div class="pfb-sample__body"><h3 class="pfb-sample__h">{escape(title)}</h3>'
            f'<p class="pfb-sample__guide">{escape(guide_text)}</p>{body}</div></section>')


def _face(person: sample_work.Person) -> str:
    return sample_art.player_img(person.team, person.number,
                                 f"{person.name}さん（{person.number}番）のイラスト")


def _check(work) -> str:
    board = (f'<section class="pfb-verdict" aria-label="動画の中身">'
             f'<p class="pfb-verdict__label">自動で確かめた結果（推定）</p>'
             f'<p class="pfb-verdict__text">{escape(work.check)}</p></section>')
    court = sample_art.scene_img(work, work.overview, "4人の選手がコートにいる様子のイラスト")
    return f'<div class="pfb-sample__board"><div>{board}</div><div>{court}</div></div>'


def _video(work) -> str:
    return (f'<dl class="pfb-facts"><dt>動画</dt><dd>{escape(work.video_name)}</dd>'
            f'<dt>長さ</dt><dd>約{work.minutes}分</dd>'
            f'<dt>カメラ</dt><dd>{work.cameras}台</dd></dl>')


def _people(work) -> str:
    items = "".join(f'<li>{_face(p)}<span class="pfb-sample__who">{escape(p.name)}'
                    f'<span class="pfb-sample__num-small">{p.number}番</span></span></li>'
                    for p in work.people)
    return f'<ul class="pfb-sample__people" aria-label="映っている選手">{items}</ul>'


def _stages(work) -> str:
    items = "".join(f'<li>{theme.status_chip("done", "済み")}{escape(name)}</li>'
                    for name in work.stages)
    return f'<ul class="pfb-sample__stages" aria-label="解析の段階">{items}</ul>'


def _mark(person) -> str:
    if person.approval == "short":
        return theme.status_chip("insufficient", "短い文") + theme.status_chip("approved", "承認済み")
    return theme.approval_stamp(animate=False)


def _moment(work, person, m: sample_work.Moment) -> str:
    alt = f"{m.at}の場面のイラスト：{m.did}"
    return (f'<li>{sample_art.scene_img(work, m.scene, alt)}'
            f'<dl class="pfb-evidence"><dt>時刻</dt><dd class="pfb-evidence__when">{escape(m.at)}</dd>'
            f'<dt>したこと</dt><dd>{escape(m.did)}</dd>'
            f'<dt>よかった理由</dt><dd>{escape(m.why)}</dd></dl></li>')


def _person(work, person: sample_work.Person) -> str:
    draft = ""
    if person.draft:
        draft = (f'<details class="pfb-sample__draft"><summary>直す前の下書きを見る</summary>'
                 f'<p>{escape(person.draft)}</p></details>')
    moments = "".join(_moment(work, person, m) for m in person.moments)
    return (f'<article class="pfb-sample__person" aria-label="{escape(person.name)}さん">'
            f'<div class="pfb-who">{_face(person)}<span class="pfb-who__name">'
            f'{escape(person.name)}さん</span>{_mark(person)}</div>'
            f'<p class="pfb-text">{escape(person.text)}</p>'
            f'<p class="pfb-sample__note">{escape(APPROVAL_NOTE[person.approval])}</p>{draft}'
            f'<p class="pfb-sample__label">この文章の根拠</p>'
            f'<ul class="pfb-sample__moments">{moments}</ul></article>')


def _sheet(work) -> str:
    items = "".join(f'<h4>{escape(p.name)}さんへ</h4><p>{escape(p.text)}</p>' for p in work.people)
    return f'<div class="pfb-sample__sheet" aria-label="選手に渡すPDFの中身">{items}</div>'


def page_html(work: sample_work.SampleWork) -> str:
    """The whole read-only walk-through (everything but the buttons)."""
    labels = steps(work)
    head = (f'<div class="pfb-sample__head"><h1 class="pfb-sample__title">{escape(work.title)}</h1>'
            f'<p class="pfb-sample__meta">動画・1グループ・選手{len(work.people)}人'
            f'{theme.status_chip("done", "完了")}{theme.status_chip("sample", READ_ONLY)}</p></div>'
            f'<p class="pfb-sample__about">{escape(work.about)}{escape(VIEW_ONLY)}</p>')
    stepper = theme.stepper_html(labels, len(labels), set(range(len(labels))))
    sections = [
        _step(1, labels[0], "入れた動画の中身を、自動で確かめます。", _check(work)),
        _step(2, labels[1], "解析する動画です。同じ場面を別のカメラで撮った動画も、ここに入ります。",
              _video(work)),
        _step(3, labels[2], "映っている人は自動で選ばれます。名前はあとからでもつけられます。",
              _people(work)),
        _step(4, labels[3], "解析は自動で進みます。見本では、もう終わっています。", _stages(work)),
        _step(5, labels[4], "一人ずつ、文章と根拠の場面を見てから承認します。",
              "".join(_person(work, p) for p in work.people)),
        _step(6, labels[5], "選手に渡すPDFには、一人ずつ名前と文章、よかった場面の写真、場面の種類と時間の流れの図が"
              "入ります（見本では文章だけを見せています）。全員の一覧と、どの場面を根拠にしたかは、"
              "コーチ用のPDFに入ります。",
              _sheet(work)),
    ]
    return CSS + f'<div class="pfb-sample">{head}{stepper}{"".join(sections)}</div>'


SEE = "見本を見てみる"
CARD_REASON = "架空の選手とイラストで作った見本です。はじめから終わりまでの流れを、見るだけで確かめられます。"
CARD_CSS = """<style>
@media (max-width: 40rem) { .st-key-pfb_samplecard .pfb-mini { display: none; } }
.st-key-pfb_samplecard { background: #FFFFFF; border-color: var(--pfb-line); border-radius: 12px;
  border-left: 4px dashed var(--pfb-board); }
</style>"""


@st.cache_data(ttl=60, show_spinner=False)
def _shown(db: str, _repo: Repo) -> bool:
    return sample_work.is_shown(_repo)


def shown(repo: Repo) -> bool:
    """sample_work.is_shown, kept a minute per database: the empty home and the guide do not
    read the records on every run (set_shown forgets it at once)."""
    where = repo.settings.data_dir if repo.settings is not None else id(repo.conn)
    return _shown(str(where), repo)


def set_shown(repo: Repo, value: bool) -> None:
    sample_work.set_shown(repo, value)
    _shown.clear()


def home_card(repo: Repo) -> None:
    """On an empty home: the sample as a work card 「見本の作業」, with 「見本を見てみる」
    (secondary: home's primary stays 「新しく始める」) and the one-tap 「見本を消す」."""
    if not shown(repo):
        return
    work = sample_work.load()
    labels = steps(work)
    st.html(CARD_CSS)
    with keyed.container(border=True, key="pfb_samplecard"):
        main, action = st.columns([5, 1], vertical_alignment="center")
        with main:
            st.html(f'<div class="pfb-workcard"><h3 class="pfb-workcard__title">'
                    f'{escape(guide.SAMPLE_TITLE)}</h3><span class="pfb-workcard__meta">'
                    f'動画・1グループ・選手{len(work.people)}人</span>'
                    f'{theme.status_chip("sample", READ_ONLY)}</div>'
                    f'<p class="pfb-workcard__reason">{escape(CARD_REASON)}</p>'
                    f'<div class="pfb-mini">{theme.stepper_html(labels, len(labels), set(range(len(labels))))}</div>')
        with action:
            if st.button(SEE, key="pfb_home_sample", icon=":material/visibility:",
                         width="stretch"):
                st.switch_page(guide.SAMPLE_PAGE)
            if access.can_edit() and st.button(REMOVE, key="pfb_home_sample_remove", type="tertiary",
                         icon=":material/delete:", width="stretch"):
                set_shown(repo, False)
                flash(REMOVED)
                st.rerun()


RESTORE = "見本をもう一度出す"
HIDDEN = "見本の作業は消してあります。もう一度出すと、ホームと「はじめての方へ」から開けます。"


def render(repo: Repo) -> None:
    theme.inject_theme()
    theme.set_domain(sample_work.load().domain)
    if st.button("ホームへ", key="pfb_sample_home", icon=":material/home:"):
        guide.go_home()
    show_flash()
    if not shown(repo):
        with theme.step_card(guide.SAMPLE_TITLE, HIDDEN, key="sample_hidden"):
            if access.can_edit() and st.button(RESTORE, key="pfb_sample_restore", type="primary",
                         icon=":material/undo:"):
                set_shown(repo, True)
                st.rerun()
        return
    st.html(page_html(sample_work.load()))
    if not access.can_edit():             # 見るだけの人: the sample only
        return
    with theme.step_card("自分の動画で始めましょう",
                         "動画か書類を入れると、この見本と同じ流れで下書きができます。", key="sample_end"):
        with keyed.container(horizontal=True, key="pfb_sample_actions"):
            if st.button(START, key="pfb_sample_start", type="primary", icon=":material/add:"):
                guide.start_new()
            if st.button(REMOVE, key="pfb_sample_remove", type="tertiary",
                         icon=":material/delete:"):
                set_shown(repo, False)
                flash(REMOVED)
                guide.go_home()
