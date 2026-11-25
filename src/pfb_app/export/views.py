"""Sheets -> what the templates draw: captions in the reader's words and the charts as SVG.

`person(report, sheet, audience)` is one person's blocks for the handout ("handout": the
person reads it, 本人 is 「あなた」) or the evaluator PDF ("evaluator": 本人 is the name). A
block with nothing recorded is None, and the template leaves it out:
- pictures / moments: the cited moments (with their pictures when there are any);
- categories: a bar chart (two or more categories), or one line for a single category;
- timeline: needs the analysed length and something to show (on-screen time or moments);
- court: needs court samples (the engine's geometry; never without it).
A short-text person (too few scenes) gets the text, the photo and the on-screen timeline
only: no moments, no charts that would look empty, no claims.
"""
from __future__ import annotations

import re

from markupsafe import Markup

from pfb_app.export import charts, labels
from pfb_app.export.charts import Bar, Mark
from pfb_app.export.sheets import Report, Sheet
from pfb_app.pipeline.pfb3x3_bridge import format_game_clock
from pfb_app.preprocess.frames import format_time

CATEGORY_TITLE = "どんな場面で活躍したか"
GROWTH_TITLE = "成長のまとめ"
GROWTH_CHART = "観点ごとのよかった場面の数"
GROWTH_CHART_NOTE = "数字は回数です。色が濃いほど多く記録されています。"
GROWTH_FOOTER = "このページは、承認した文章と、その文章のもとになった場面の数からまとめています。"
GROWTH_FOOTER_LIVE = ("このページは、作業ごとの文章と、その文章のもとになった場面の数からまとめています。"
                      "AIが作成した文章は、人の確認を経ていない場合があります（作業ごとに確認状況を書いています）。")
TIMELINE_TITLE = "時間の流れ"
COURT_TITLE = "コートのどこにいたか"


def initial(name: str) -> str:
    """The monogram of a person without a photo: the first letter or digit of the name
    (「#7」 -> 「7」)."""
    m = re.search(r"[^\W_]", name or "")
    return m.group(0) if m else "・"


def _mono(sheet: Sheet) -> str:
    """The monogram of a person without a photo: the first letter of the name, or (draft export,
    a jersey label) the jersey digits, 「?」 when the number is unknown."""
    if not sheet.jersey:
        return initial(sheet.name)
    m = re.search(r"\d+", sheet.jersey)
    return m.group(0) if m else "?"


def distance_text(metres: float | None) -> str | None:
    if metres is None or metres < 1:
        return None
    step = 1 if metres < 50 else 5 if metres < 200 else 10
    return f"動いた距離 約{int(round(metres / step) * step)} m"


def _tick(sheet: Sheet):
    if sheet.clock_offset_s is not None:
        offset = sheet.clock_offset_s
        return lambda t: format_game_clock(offset + t)
    return format_time


def timeline_note(report: Report, sheet: Sheet) -> str:
    if sheet.clock_offset_s is not None:
        return f"目もりは{report.words.time_anchor}です。"
    return "目もりは開始からの時間です。"


def caption(report: Report, sheet: Sheet, text: str, audience: str) -> str:
    """The handout speaks to the person (本人 -> あなた); the evaluator PDF keeps the
    observation exactly as recorded (as on the review screen)."""
    return text if audience == "evaluator" else labels.self_words(text, report.work)


def person(report: Report, sheet: Sheet, audience: str, *, accent: str = charts.BAR,
           width: float = 640) -> dict:
    short = sheet.short
    cats = report.categories
    moments = [] if short else [m for m in sheet.moments if m.cited]
    pictures = [m for m in moments if m.picture is not None] if audience == "handout" else [
        m for m in moments if m.picture is not None or m.thumb is not None]
    pictures.sort(key=lambda m: (m.number or 99, m.t_s or 0.0))
    texts_only = [m for m in moments if m not in pictures]
    counts = [] if short else sheet.counts(cats)
    bars = single = None
    if len(counts) >= 2:
        bars = Markup(charts.bar_chart([Bar(c.label, n, c) for c, n in counts],
                                       title=CATEGORY_TITLE, unit="回", width=width * 0.48,
                                       colour=accent))
    elif len(counts) == 1:
        single = {"glyph": Markup(charts.glyph(counts[0][0], 14)), "label": counts[0][0].label,
                  "n": counts[0][1]}
    timeline = None
    if sheet.retracking:
        pass
    elif sheet.duration_s and (sheet.spans or (sheet.moments and not short)):
        marks = [] if short else [Mark(m.t_s, m.category, m.number) for m in sheet.moments
                                  if m.t_s is not None]
        svg = charts.timeline(sheet.duration_s, sheet.spans, marks, tick=_tick(sheet),
                              title=TIMELINE_TITLE, width=width)
        timeline = Markup(svg) if svg else None
    court = None
    if sheet.court_points and not short:
        svg = charts.court(sheet.court_points, title=COURT_TITLE, width=width * 0.38)
        court = Markup(svg) if svg else None
    return {
        "sheet": sheet, "name": sheet.shown, "initial": _mono(sheet),
        "jersey": bool(sheet.jersey), "jersey_id": sheet.jersey_id, "text": sheet.text,
        "short": short,
        "tips": [] if short else sheet.tips,
        "team_tips": [] if short else sheet.team_tips,
        "team_title": report.team_tips_title,
        "photo": sheet.photo,
        "footer": ("" if sheet.draft and audience == "handout" else
               labels.live_footer(sheet.live_version) if sheet.live else
                   labels.footer_line(report.work, report.video, auto=sheet.auto, draft=sheet.draft)),
        "draft": sheet.draft, "live": sheet.live, "live_label": sheet.live_label,
        "live_lines": sheet.live_lines, "ident_note": sheet.ident_note,
        "pictures": [{"m": m, "image": m.picture or m.thumb, "caption":
                      caption(report, sheet, m.what, audience)} for m in pictures],
        "moments_text": [] if audience == "evaluator" else [
            {"m": m, "caption": caption(report, sheet, m.what, audience)} for m in texts_only],
        "evidence": [] if audience != "evaluator" else [
            {"m": m, "caption": caption(report, sheet, m.what, audience),
             "glyph": Markup(charts.glyph(m.category, 11)) if m.category else ""}
            for m in sheet.moments],
        "bars": bars, "single": single, "timeline": timeline,
        "timeline_note": timeline_note(report, sheet), "court": court,
        "distance": None if short else distance_text(sheet.distance_m),
        "on_screen_min": None if sheet.on_screen_s is None else round(sheet.on_screen_s / 60, 1),
    }


def summary(report: Report, *, accent: str = charts.BAR, width: float = 640) -> dict:
    """The evaluator's cover charts: good moments per person and minutes on screen per
    person, each sorted by its own value (two charts, never two axes)."""
    people = [s for s in report.sheets if not s.ident_blocked]   # 同一人物の確認
    by_count = sorted((s for s in people if s.counted), key=lambda s: (-s.counted, s.shown))
    by_time = sorted((s for s in people if s.on_screen_s), key=lambda s: (-s.on_screen_s, s.shown))
    moments = charts.bar_chart([Bar(s.shown_long, s.counted) for s in by_count],
                               title="よかった場面の数", unit="回", width=width * 0.48,
                               row_h=20, thick=10, colour=accent)
    minutes = charts.bar_chart([Bar(s.shown_long, round(s.on_screen_s / 60, 1)) for s in by_time],
                               title="映っていた時間", unit="分", width=width * 0.48,
                               row_h=20, thick=10, colour=accent,
                               value_text=lambda v: f"{v:.1f}分")
    return {"moments": Markup(moments) if moments else None,
            "minutes": Markup(minutes) if minutes else None}


def contact(report: Report, sheet: Sheet) -> dict:
    top = sheet.top_category(report.categories)
    return {"sheet": sheet, "photo": sheet.photo, "name": sheet.shown,
            "initial": _mono(sheet), "count": sheet.counted, "jersey_id": sheet.jersey_id,
            "top": None if top is None else {"label": top.label,
                                             "glyph": Markup(charts.glyph(top, 11))},
            "chips": sheet.chips, "reason": sheet.reason}


def team_groups(report: Report) -> list[tuple[str | None, list[Sheet]]]:
    """[(team, people)]: two columns when teams are known for at least two people."""
    known = [s for s in report.sheets if s.team]
    if len({s.team for s in known}) < 2:
        return [(None, report.sheets)]
    groups = [(t, [s for s in report.sheets if s.team == t]) for t in report.teams]
    rest = [s for s in report.sheets if not s.team]
    if rest:
        groups.append(("チームがわからない人", rest))
    return groups


def team_courts(report: Report, width: float = 300) -> list[dict]:
    out = []
    for team, people in team_groups(report):
        if team is None:
            continue
        pts = [p for s in people for p in s.court_points]
        svg = charts.court(pts, title=f"{team}のいた所", width=width) if pts else ""
        if svg:
            out.append({"team": team, "svg": Markup(svg)})
    return out


def growth_page(rec, *, accent: str = charts.BAR, width: float = 640) -> dict:
    """The handout's 「成長のまとめ」 page of one record (export.growth.Record): the
    statements, the counts grid and the works with one quote each (approved works only)."""
    from pfb_app.export import growth   # noqa: PLC0415 - growth imports sheets, as we do
    columns, rows = growth.grid(rec)
    svg = charts.growth_grid(columns, rows, title=GROWTH_CHART, width=width)
    works = []
    for e in rec.works:                     # one per work (its groups added up)
        points = "・".join(f"{label} {n}回" for label, n in e.counts.items())
        works.append({"when": e.when, "title": e.title, "group": e.group_label,
                      "activity": e.activity, "points": points,
                      "quote": e.quotes[0] if e.quotes else None,
                      "state": f"AIが作成した文・確認状況：{e.live_label}" if e.live else None})
    return {"title": GROWTH_TITLE, "lines": growth.statements(rec),
            "chart": Markup(svg) if svg else None, "chart_title": GROWTH_CHART,
            "chart_note": GROWTH_CHART_NOTE, "works": works,
            "footer": GROWTH_FOOTER_LIVE if rec.live else GROWTH_FOOTER, "accent": accent}
