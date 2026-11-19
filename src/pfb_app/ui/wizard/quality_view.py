"""「この解析の確かさ」: what this server measured, what the research checked, what nobody has.

Three parts, never mixed:
1. **このサーバーでの実績**: numbers counted now from this install's own records
   (services.quality_stats), written in chalk on the green board, with the split of the
   approved texts as one bar (inline SVG, a one-hue chalk ramp validated on the board).
2. **研究で確かめたこと**: a fixed list, each with its source; the numbers come from the
   research and pilot records (pfb_report dossier, 2026-10-22), not from this server.
3. **わかっている弱いところ / まだ確かめていないこと**: what was measured and fell short,
   and what was never measured. Nothing is claimed beyond these lists.
"""
from dataclasses import dataclass
from html import escape

import streamlit as st

from pfb_app.services import quality_stats
from pfb_app.services.quality_stats import AS_IS, AUTO, EDITED, REWRITTEN, SHORT, QualityStats
from pfb_app.storage.db import Repo
from pfb_app.ui.wizard import help as guide
from pfb_app.ui.wizard import svg, theme
from pfb_app.ui.wizard.theme import TOKENS

TITLE = guide.QUALITY_TITLE
LEAD = ("このサーバーで実際に数えた数字と、研究で確かめたこと、まだ確かめていないことを"
        "分けて書いています。")

# the approved people: a person's approvals, least to most changed, then the automatic ones;
# one hue, light to dark on the board, all theme tokens (validated: dataviz
# validate_palette.py --mode dark --surface board --ordinal, all pass)
KIND_LABELS = {AS_IS: "下書きのまま承認", EDITED: "直して承認", REWRITTEN: "自分で書いて承認",
               SHORT: "短い文で承認", AUTO: "自動で承認"}
RAMP = {AS_IS: TOKENS["chalk"], EDITED: TOKENS["chalk-2"], REWRITTEN: TOKENS["chalk-3"],
        SHORT: TOKENS["chalk-4"], AUTO: TOKENS["chalk-5"]}
BAR_W, BAR_H, GAP = 600.0, 36.0, 2.0

EMPTY = "まだ承認された文章がありません。承認すると、ここに数字が出ます。"
FEW = "まだ{n}人分の記録です。人数が少ないうちは、目安として見てください。"
SCOPE = "いまある作業の記録から数えています。削除した作業は入りません。"


@dataclass(frozen=True)
class Claim:
    text: str
    source: str


RESEARCH = "研究データ（開発中の試験。このサーバーの記録ではありません）"
PILOT = "試験運用の記録（2026年10月、2つの作業）"
DESIGN = "このアプリのつくり"

VALIDATED = (
    Claim("解析はこのサーバーの中だけで行います。動画や文章を、インターネット上のサービスには送りません。",
          DESIGN),
    Claim("文章の一つ一つは、観察した場面（時刻つき）にもとづいて作られます（短い文と手書きの文を除く）。"
          "評価者用のPDFで、どの場面を根拠にしたかを確かめられます。", DESIGN),
    Claim("人が承認するまで、文章はだれにも渡りません。「確認せずに自動で承認する」は、"
          "オンにしたときだけ働きます（はじめはオフです）。", DESIGN),
    Claim("文章づくりは安定していました。2,592件のうち2,451件（94.6%）で文章ができ、書くのを"
          "断ったことは0件でした。残りは、場面が足りなかった45件（1.7%）と、途中で切れた回答"
          "96件（3.7%）でした。", RESEARCH),
    Claim("試験運用の2つの作業（13人と4人）で、解析の段階29件がすべて1回目で終わりました。"
          "13人の作業では、決定から下書きがそろうまで約34分、10人の承認は34秒でした。", PILOT),
)
WEAK = (
    Claim("場面が少ない人には、文章が書けないことがあります。試験運用では13人中3人（23%）が"
          "そうでした。いまは短い文を用意しますが、短い文は具体的ではありません。", PILOT),
    Claim("人の見分け（同じ人を最後までたどること）は、3対3の試合の動画で、つなぎの確かさが"
          "0.4〜0.6でした。カメラが大きく動くと、取り違えが増えます。", RESEARCH),
    Claim("先行研究の参考の文章と、4人分の文章を文の似かたで比べたところ、対応する人の文章がいちばん"
          "似ていたのは36回中12回で、偶然（9回）と同じくらいでした。この比べ方では、その人ならではの文かどうかは"
          "確かめられていません。", RESEARCH),
    Claim("名前の自動推定は、確かさの基準（9割）に届かないため、使っていません。", RESEARCH),
    Claim("「判定支援」は、線の近くの場面を写真と動画で見せる材料で、審判の代わりではありません。"
          "実際の3対3の試合の動画3本（線の近くの場面343件・373件・296件）では、足が線を踏んだか"
          "どうかを自動で決められた場面は0件でした。くわしく決める方法は、いまも確かめている途中です。",
          RESEARCH),
)
NOT_YET = (
    Claim("専門家が書いたフィードバックとの比べ合わせ。バスケットボールの専門家の文章をまだ"
          "用意していないため、していません。", RESEARCH),
    Claim("別の人についての文が混ざる割合。まだ測っていません。", RESEARCH),
    Claim("3対3以外のスポーツや、会議・研修などの映像での確かさ。まだ測っていません。", RESEARCH),
    Claim("点数や記録（ボックススコア）。正解のデータがないため、出していません。", DESIGN),
)
SO = ("そのため、このアプリは「正確な評価」や「専門家と同じ」とは言いません。"
      "文章は下書きとして、あなたが確かめてから渡してください。")
SOURCE_NOTE = ("研究データと試験運用の数字は、開発チームの記録（2026年10月22日まとめ）からの"
               "引用です。このサーバーで測った数字ではありません。")

CSS = """<style>
.pfb-quality { max-width: 52rem; margin: 0 auto; font-family: var(--pfb-font-ui); color: var(--pfb-ink); }
.pfb-quality__title { margin: 0.5rem 0 0.25rem; font-size: 1.75rem; font-weight: 700; line-height: 1.4; }
.pfb-quality__lead { margin: 0 0 1.25rem; font-size: 1.0625rem; line-height: 1.8; max-width: 36em; }
.pfb-measured { padding: 1.25rem 1.5rem 1.1rem; background: var(--pfb-board); color: var(--pfb-chalk);
  border-radius: 12px 12px 4px 4px; border-bottom: 6px solid var(--pfb-tray); }
.pfb-measured__h { margin: 0; font-size: 1.1875rem; font-weight: 700; color: var(--pfb-chalk); }
.pfb-measured__scope { margin: 0.2rem 0 1rem; color: var(--pfb-chalk-2); font-size: 0.875rem; }
.pfb-tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(8.25rem, 1fr));
  gap: 1rem 1.25rem; margin: 0 0 1.25rem; padding: 0; }
.pfb-tile { margin: 0; padding: 0.1rem 0 0 0.75rem; border-left: 1px dashed var(--pfb-chalk-2); }
.pfb-tile dt { color: var(--pfb-chalk-2); font-size: 0.875rem; line-height: 1.5; }
.pfb-tile dd { margin: 0; }
.pfb-tile__value { display: block; font-family: var(--pfb-font-hand); font-weight: 600;
  font-size: 2.5rem; line-height: 1.2; color: var(--pfb-chalk); font-variant-numeric: tabular-nums; }
.pfb-tile__unit { font-size: 1.125rem; margin-left: 0.15rem; }
.pfb-tile__of { display: block; color: var(--pfb-chalk-2); font-size: 0.8125rem; }
.pfb-split__h { margin: 0 0 0.4rem; font-size: 0.9375rem; font-weight: 700; color: var(--pfb-chalk); }
.pfb-split__bar { display: block; width: 100%; height: auto; }
.pfb-split__legend { display: flex; flex-wrap: wrap;
  gap: 0.3rem 1.25rem; margin: 0.6rem 0 0; padding: 0; }
.pfb-split__legend div { display: flex; align-items: baseline; gap: 0.45rem; white-space: nowrap; }
.pfb-split__legend dt { color: var(--pfb-chalk); font-size: 0.9375rem; }
.pfb-split__legend dd { margin: 0; color: var(--pfb-chalk-2); font-size: 0.875rem;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.pfb-swatch { flex: none; width: 0.8rem; height: 0.8rem; border-radius: 3px; align-self: center;
  outline: 1px solid var(--pfb-chalk-2); }
.pfb-measured__note { margin: 0.9rem 0 0; color: var(--pfb-chalk-2); font-size: 0.875rem; }
.pfb-measured__empty { margin: 0 0 0.5rem; color: var(--pfb-chalk); }
.pfb-claims__h { margin: 2rem 0 0.25rem; font-size: 1.1875rem; font-weight: 700; }
.pfb-claims__guide { margin: 0 0 0.75rem; color: var(--pfb-muted); font-size: 0.9375rem; }
.pfb-claims { list-style: none; margin: 0; padding: 0; }
.pfb-claims li { position: relative; margin: 0 0 0.85rem; padding-left: 1.9rem; line-height: 1.75; }
.pfb-claims li::before { position: absolute; left: 0.2rem; top: 0; font-weight: 700; }
.pfb-claims--ok li::before { content: "✓"; color: var(--pfb-board); }
.pfb-claims--weak li::before { content: "△"; color: var(--pfb-wait); }
.pfb-claims--open li::before { content: "？"; color: var(--pfb-muted); }
.pfb-claims__src { display: block; color: var(--pfb-muted); font-size: 0.8125rem; }
.pfb-quality__so { margin: 1.75rem 0 0.5rem; padding: 0.9rem 1.1rem; background: #FFFFFF;
  border: 1px solid var(--pfb-line); border-left: 4px solid var(--pfb-board); border-radius: 0 10px 10px 0;
  font-weight: 700; line-height: 1.75; }
.pfb-quality__cite { margin: 0.5rem 0 0; color: var(--pfb-muted); font-size: 0.8125rem; }
@media (max-width: 40rem) {
  .pfb-measured { padding: 1rem 1rem 0.9rem; }
  .pfb-tile__value { font-size: 2rem; }
}
</style>"""


def percent(share: float | None) -> str:
    return "—" if share is None else f"{round(share * 100)}"


def minutes(seconds: float | None) -> str:
    return "—" if seconds is None else f"{max(1, round(seconds / 60))}"


def _tile(label: str, value: str, unit: str, of: str) -> str:
    unit_html = f'<span class="pfb-tile__unit">{escape(unit)}</span>' if value != "—" else ""
    return (f'<div class="pfb-tile"><dt>{escape(label)}</dt><dd><span class="pfb-tile__value">'
            f'{escape(value)}{unit_html}</span><span class="pfb-tile__of">{escape(of)}</span></dd></div>')


def tiles_html(s: QualityStats) -> str:
    n = s.approved_total
    of = f"承認した{n}人のうち" if n else "まだ承認がありません"
    turn = (f"{len(s.turnaround_sec)}グループの中央値" if s.turnaround_sec
            else "まだ記録がありません")
    first = (f"終わった{s.jobs_done}件のうち{s.jobs_first_try}件。いま止まっているもの{s.jobs_stopped}件"
             if s.jobs_done or s.jobs_stopped else "まだ記録がありません")
    items = (
        _tile("下書きのまま承認", percent(s.share(AS_IS)), "%", of),
        _tile("自動で承認", percent(s.auto_share), "%", of),
        _tile("短い文で承認", percent(s.share(SHORT)), "%", of),
        _tile("解析を始めてから下書きがそろうまで", minutes(s.turnaround_median_sec), "分", turn),
        _tile("1回目で終わった解析の段階", percent(s.first_try_share), "%", first),
    )
    return f'<dl class="pfb-tiles">{"".join(items)}</dl>'


def split_svg(s: QualityStats) -> str:
    """The approved texts as one bar: a segment per kind, 2px board gaps between them,
    rounded ends, the share written inside a segment wide enough for it."""
    total = s.approved_total
    parts = [(k, s.approved[k]) for k in quality_stats.KINDS if s.approved[k]]
    gaps = GAP * (len(parts) - 1)
    body, x = [], 0.0
    for i, (kind, count) in enumerate(parts):
        w = (BAR_W - gaps) * count / total
        rx = 4 if i in (0, len(parts) - 1) else 0
        body.append(f'<rect x="{x:.2f}" y="0" width="{w:.2f}" height="{BAR_H:g}" rx="{rx}" '
                    f'fill="{RAMP[kind]}"><title>{escape(KIND_LABELS[kind])} {count}人</title></rect>')
        if 0 < rx and len(parts) > 1:   # square off the inner side of an end segment
            inner = x + w - 4 if i == 0 else x
            body.append(f'<rect x="{inner:.2f}" y="0" width="4" height="{BAR_H:g}" '
                        f'fill="{RAMP[kind]}"/>')
        if w >= 48:
            body.append(f'<text x="{x + w / 2:.2f}" y="{BAR_H / 2 + 5:.1f}" text-anchor="middle" '
                        f'font-size="15" font-weight="700" font-family="sans-serif" '
                        f'fill="{TOKENS["ink"]}">{round(count * 100 / total)}%</text>')
        x += w + GAP
    labels = "、".join(f"{KIND_LABELS[k]} {c}人" for k, c in parts)
    return svg.doc(BAR_W, BAR_H, "".join(body), f"承認した文章の内わけ：{labels}")


def split_alt(s: QualityStats) -> str:
    """The chart's text alternative: every kind with its people and share."""
    total = s.approved_total
    parts = "、".join(f"{KIND_LABELS[k]} {s.approved[k]}人（{round(s.approved[k] * 100 / total)}%）"
                     for k in quality_stats.KINDS)
    return f"承認した文章の内わけ、{total}人：{parts}。"


def split_html(s: QualityStats) -> str:
    total = s.approved_total
    chart = svg.img(split_svg(s), split_alt(s), "pfb-split__bar")
    rows = "".join(
        f'<div><span class="pfb-swatch" style="background:{RAMP[k]}" aria-hidden="true"></span>'
        f'<dt>{escape(KIND_LABELS[k])}</dt><dd>{s.approved[k]}人・{round(s.approved[k] * 100 / total)}%'
        f'</dd></div>' for k in quality_stats.KINDS)
    return (f'<p class="pfb-split__h">承認した文章の内わけ（{total}人）</p>{chart}'
            f'<dl class="pfb-split__legend" aria-label="承認した文章の内わけ">{rows}</dl>')


def measured_html(s: QualityStats) -> str:
    scope = SCOPE
    if s.since:
        y, m, d = s.since.split("-")
        scope = f"{int(y)}年{int(m)}月{int(d)}日からの、いまある作業{s.works}件の記録から数えています。削除した作業は入りません。"
    body = tiles_html(s)
    if s.approved_total:
        body += split_html(s)
        if s.few:
            body += f'<p class="pfb-measured__note">{escape(FEW.format(n=s.approved_total))}</p>'
    else:
        body += f'<p class="pfb-measured__empty">{escape(EMPTY)}</p>'
    if s.retries:
        body += (f'<p class="pfb-measured__note">「もう一度ためす」を押した回数：{s.retries}回</p>')
    return (f'<section class="pfb-measured" aria-label="このサーバーでの実績">'
            f'<h2 class="pfb-measured__h">このサーバーでの実績</h2>'
            f'<p class="pfb-measured__scope">{escape(scope)}</p>{body}</section>')


def claims_html(title: str, guide_text: str, claims: tuple[Claim, ...], kind: str) -> str:
    items = "".join(f'<li>{escape(c.text)}<span class="pfb-claims__src">出典：{escape(c.source)}'
                    '</span></li>' for c in claims)
    return (f'<h2 class="pfb-claims__h">{escape(title)}</h2>'
            f'<p class="pfb-claims__guide">{escape(guide_text)}</p>'
            f'<ul class="pfb-claims pfb-claims--{kind}" aria-label="{escape(title)}">{items}</ul>')


def page_html(s: QualityStats) -> str:
    return CSS + (
        f'<div class="pfb-quality"><h1 class="pfb-quality__title">{escape(TITLE)}</h1>'
        f'<p class="pfb-quality__lead">{escape(LEAD)}</p>{measured_html(s)}'
        + claims_html("研究で確かめたこと", "このアプリのつくりと、研究・試験運用で確かめたことです。",
                      VALIDATED, "ok")
        + claims_html("わかっている弱いところ", "測ってみて、まだ十分ではなかったことです。", WEAK, "weak")
        + claims_html("まだ確かめていないこと", "測っていないので、できるとは言えないことです。",
                      NOT_YET, "open")
        + f'<p class="pfb-quality__so">{escape(SO)}</p>'
        f'<p class="pfb-quality__cite">{escape(SOURCE_NOTE)}</p></div>')


CACHE_SEC = 60   # the numbers are re-counted at most once a minute per database


@st.cache_data(ttl=CACHE_SEC, show_spinner=False)
def _measured(db: str, marker: tuple, _repo: Repo) -> QualityStats:
    return quality_stats.measure(_repo)


def measured(repo: Repo) -> QualityStats:
    """quality_stats.measure, kept for CACHE_SEC per database and change marker (the guide's
    FAQ and this page share it): a page run does not rescan the records every time, and a
    change to them (an approval, a deletion, a finished step) is counted at once."""
    where = repo.settings.data_dir if repo.settings is not None else id(repo.conn)
    return _measured(str(where), quality_stats.change_marker(repo), repo)


def render(repo: Repo) -> None:
    theme.inject_theme()
    theme.set_domain("school")
    st.html(page_html(measured(repo)))
