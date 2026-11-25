"""The demo edition (デモ版) of the booklet: what is really confirmed, up front, nothing bent.

A separate output next to `booklet.build` (the full booklet keeps every person). Here each
person's scenes are judged one by one (`demo_scenes`: the person's number is read near the scene,
the box is continuous, nobody else's box or number overlaps) and
- a person gets a page only with at least `min_confirmed` confirmed scenes (strongest first),
  and the page uses only its confirmed scenes (photos, counts, the table);
- "検出の例" pages follow one person's box through 121 consecutive frames of the public clip
  (9 pictures, 0.5 s apart) where it never breaks or jumps and nobody else overlaps it;
- "検出の数字" pages carry only measured values (this clip) and the two recall figures of the
  development branch, marked as such;
- the colophon states that the pages are a selection, with the measured counts.

Pages: cover, [front spread], person spreads, example spreads (two per spread), numbers spread,
colophon (a multiple of 4 in all).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from pfb_app.export import booklet, labels, pdf as _pdf, views
from pfb_app.export.sheets import Sheet, build_report

SUBTITLE = "公開クリップ（Samsunspor 3分版）の処理結果 デモ版"
MIN_CONFIRMED = 2
FLOOR = 4                # fewer people than this at MIN_CONFIRMED: the minimum drops to 1 (推定)
MAX_EXAMPLES = 8
MAX_PHOTOS = 4

# the development branch's measured recall (not in production; other clips): see
# docs/recall_extra_boxes.md. before -> after, people on the court boxed, false boxes.
RECALL = (
    {"clip": "Samsunspor 60秒版（等間隔6場面、74人）", "before": 47, "after": 85,
     "n_before": 35, "n_after": 63, "total": 74, "fp_before": 0, "fp_after": 6},
    {"clip": "車いすバスケの公開クリップ（4場面、35人）", "before": 74, "after": 86,
     "n_before": 26, "n_after": 30, "total": 35, "fp_before": 4, "fp_after": 6},
)
RECALL_NOTE = ("開発版での測定値です。本番への反映と、この3分版での測定はこれからです。"
               "正解は、コート上の人（選手と審判。ベンチ・コーチ・観客は数えない）を目で数えたものです。"
               "枠が正解の人に合うのは、枠の重なり（IoU）が0.4以上のときで、1枠は1人だけに数えます。"
               "「再現率」は、正解の人のうち枠が合った人の割合です。改善の中身は、人物として追跡"
               "できなかった人にも、名前のない枠を付けることです。")
FP_NOTE = ("選手以外の枠は、正解（コート上の選手と審判）にない枠の数です。ベンチ・コーチ・座っている人に付いた枠も"
           "含みます。改善後は枠が増えるため、この数も増えています。評価は2本のクリップ・10場面（正解109人）で、"
           "件数には数人ぶんの幅があります。")
AI_LINE = "AIが作成した文です。気づいた点は訂正できます。"
LIMIT_NOTE = "本資料は公開動画1本による実処理例です。人物の対応づけには手がかりに基づく部分があります。"
CONTACT = "お問い合わせ・ご意見：上の GitHub へどうぞ。"


@dataclass
class Example:
    who: str
    jersey_id: str | None
    a: int
    b: int
    fps: float
    images: list[tuple[str, Path | None]]      # (time label, picture)
    run_s: float                               # the whole unbroken run the window is in
    number: str | None                         # the person's number if it is read in the window
    scene: bool                                # a confirmed scene is inside the window
    clock: str = "開始から"
    other_label: str = ""
    run_a: int | None = None                   # the whole unbroken run (frames), for the position bar
    run_b: int | None = None
    clip_frames: int | None = None

    @property
    def frames(self) -> int:
        return self.b - self.a + 1

    @property
    def seconds(self) -> float:
        return round((self.b - self.a) / self.fps, 1)


@dataclass
class Entry:
    """One person with text: the scene judgment and what the demo draws from it."""
    sheet: Sheet
    refs: set[str]                  # confirmed scenes (moment ref ids)
    n_all: int
    dominant: str | None = None
    reads: int = 0
    numbered: int = 0
    reach_s: float = 9.0
    examples: list[Example] = field(default_factory=list)
    longest_run_s: float | None = None
    runs_4s: int = 0
    hero: "Example | None" = None   # the longest unbroken run, sampled over its whole length
    estimated: bool = False         # shown with fewer than MIN_CONFIRMED scenes

    @property
    def n_ok(self) -> int:
        return len(self.refs)


@dataclass
class Evidence:
    entries: list[Entry]
    numbers: dict = field(default_factory=dict)      # measured values of the clip
    visuals: dict = field(default_factory=dict)      # pictures of the intro (frame, boxes, jersey)


def choose(entries: list[Entry], min_confirmed: int = MIN_CONFIRMED, floor: int = FLOOR
           ) -> tuple[list[Entry], int]:
    """(people with a page, strongest first; the minimum used). Fewer than `floor` people
    reach `min_confirmed`: it drops to 1 and those with one scene are marked 推定."""
    used = min_confirmed
    shown = [e for e in entries if e.n_ok >= used]
    if len(shown) < floor and used > 1:
        used = 1
        shown = [e for e in entries if e.n_ok >= 1]
    for e in shown:
        e.estimated = e.n_ok < min_confirmed
    shown.sort(key=lambda e: (-e.n_ok, -e.reads, e.sheet.shown_long))
    return shown, used


def pick_examples(shown: list[Entry], cap: int = MAX_EXAMPLES) -> list[Example]:
    """Each person's first example, then the second ones, strongest person first; an even
    number if it can be (an odd one gets a 見方 page as its partner)."""
    out: list[Example] = []
    for rnd in range(2):
        for e in shown:
            if len(e.examples) > rnd and len(out) < cap:
                out.append(e.examples[rnd])
    return out


def layout(n_people: int, n_examples: int) -> dict:
    strips = math.ceil(n_examples / 2)
    units = n_people + strips + 3                  # + intro, numbers and 今/これから spreads
    lay = booklet.plan(units)
    return {**lay, "strips": strips, "units": units}


# --- the real data -------------------------------------------------------------------------

def _clock(t_s: float, offset: float | None) -> str:
    s = max(t_s + (offset or 0.0), 0.0)
    return f"{int(s // 60)}:{s % 60:04.1f}"


def gather(repo, assignment_id: int, report) -> Evidence:
    """Judge every person's scenes and collect the pictures and numbers (one clip, one camera:
    the first submission's first camera). Reads the database and the video's frames only for
    pictures; nothing is written to the database."""
    import numpy as np   # noqa: PLC0415
    from pfb_app.domain.models import Stage   # noqa: PLC0415
    from pfb_app.export import demo_scenes as D, pictures, sheets as S   # noqa: PLC0415
    from pfb_app.export.tracks import Box   # noqa: PLC0415
    work = report.work
    sub0 = repo.list_submissions(assignment_id)[0]
    sub = S._Submission(repo, work, sub0)
    cam = sub.camera(sub.camera_names[0]) if sub.camera_names else None
    if cam is None or cam.boxes is None or not cam.fps:
        return Evidence([Entry(s, set(), len(s.moments)) for s in report.exported])
    fps = float(cam.fps)
    cache = sub.cache_dir / "demo"
    cache.mkdir(parents=True, exist_ok=True)
    others: dict = {}
    entries: list[Entry] = []
    reqs = []
    for sheet in report.exported:
        subject = repo.get_subject(sheet.subject_id)
        run = repo.latest_run(sub0["id"], Stage.pfb, subject_id=subject["id"])
        ps = D.check_person(repo, sub, cam, subject, sheet, S._chain(repo, sub, run), others)
        e = Entry(sheet, {s.ref_id for s in ps.confirmed}, ps.n_all, ps.dominant, ps.reads,
                  ps.numbered, ps.reach_s)
        runs = D.unbroken_runs(cam, ps.labels)
        e._ps, e._runs = ps, runs                     # type: ignore[attr-defined]
        if runs:
            e.longest_run_s = round(max(b - a + 1 for a, b in runs) / fps, 1)
            e.runs_4s = sum(1 for a, b in runs if (b - a + 1) / fps >= 4.0)
        if e.n_ok:
            for w in D.windows(ps, runs):
                frames = [w["a"] + D.STRIP_STEP * k for k in range(D.STRIP_FRAMES)]
                boxes = []
                for f in frames:
                    j = int(np.searchsorted(ps.series[0], f))
                    boxes.append(Box(*map(float, ps.series[1][j])))
                    reqs.append(pictures.PictureRequest(cam.video, f, boxes[-1]))
                num = ps.dominant if w["number"] else None
                e.examples.append(Example(
                    sheet.shown, sheet.jersey_id, w["a"], w["b"], fps,
                    [(_clock((f - 1) / fps - sub.start_s, sheet.clock_offset_s), None) for f in frames],
                    round((w["run"][1] - w["run"][0] + 1) / fps, 1), num, w["scene"],
                    "試合時間" if sheet.clock_offset_s is not None else "開始から",
                    run_a=w["run"][0], run_b=w["run"][1], clip_frames=cam.meta.n_frames if cam.meta else None))
                e.examples[-1]._frames = list(zip(frames, boxes))        # type: ignore[attr-defined]
            _photos(repo, sheet, e, sub)
            if runs:      # the longest unbroken run, 8 pictures over its whole length
                a, b = max(runs, key=lambda r: r[1] - r[0])
                fr8 = [int(round(a + (b - a) * k / 7)) for k in range(8)]
                bx8 = []
                for f in fr8:
                    j = int(np.searchsorted(ps.series[0], f))
                    bx8.append(Box(*map(float, ps.series[1][j])))
                    reqs.append(pictures.PictureRequest(cam.video, f, bx8[-1]))
                e.hero = Example(sheet.shown, sheet.jersey_id, a, b, fps,
                                 [(_clock((f - 1) / fps - sub.start_s, sheet.clock_offset_s), None) for f in fr8],
                                 round((b - a + 1) / fps, 1), None, False,
                                 "試合時間" if sheet.clock_offset_s is not None else "開始から")
                e.hero._frames = list(zip(fr8, bx8))          # type: ignore[attr-defined]
        entries.append(e)
    made = pictures.make_pictures(reqs, cache) if reqs else {}
    for e in entries:
        for ex in e.examples + ([e.hero] if e.hero else []):
            ex.images = [(lab, made.get(pictures.PictureRequest(cam.video, f, b)))
                         for (lab, _), (f, b) in zip(ex.images, ex._frames)]   # type: ignore[attr-defined]
    return Evidence(entries, _numbers(repo, sub, cam, sub0, fps, entries),
                    _visuals(cam, others, entries, cache))


def _visuals(cam, others: dict, entries: list[Entry], cache: Path) -> dict:
    """The intro's pictures from one frame of the clip (the strongest person's first example,
    its middle): the frame, the same frame with every tracked box (the person's in the brand
    colour), and the person's upper body (where the number is read)."""
    import cv2   # noqa: PLC0415
    import numpy as np   # noqa: PLC0415
    best = [e for e in entries if e.n_ok and e.examples]
    if not best:
        return {}
    e = max(best, key=lambda e: (e.n_ok, e.reads))
    ex = e.examples[0]
    f = (ex.a + ex.b) // 2
    cap = cv2.VideoCapture(str(cam.video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f - 1)
        ok, img = cap.read()
    finally:
        cap.release()
    if not ok or img is None:
        return {}
    H, W = img.shape[:2]
    k = 960 / max(H, W)
    raw = cv2.resize(img, (round(W * k), round(H * k)), interpolation=cv2.INTER_AREA)
    boxed = raw.copy()
    mine = set(e._ps.labels)                                # type: ignore[attr-defined]
    for lb, (fr, bx) in others.items():
        j = int(np.searchsorted(fr, f))
        if j < len(fr) and fr[j] == f:
            x, y, w, h = (v * k for v in bx[j])
            own = lb in mine
            cv2.rectangle(boxed, (round(x), round(y)), (round(x + w), round(y + h)),
                          (12, 65, 200) if own else (255, 255, 255), 4 if own else 2)
    fr, bx = e._ps.series                                   # type: ignore[attr-defined]
    j = int(np.searchsorted(fr, f))
    x, y, w, h = bx[j]
    x0, x1 = int(max(x - 0.1 * w, 0)), int(min(x + 1.1 * w, W))
    y0, y1 = int(max(y + 0.08 * h, 0)), int(min(y + 0.5 * h, H))
    crop = img[y0:y1, x0:x1]
    crop = cv2.resize(crop, (480, max(round(480 * crop.shape[0] / max(crop.shape[1], 1)), 1)),
                      interpolation=cv2.INTER_CUBIC)
    out = {}
    for name, im in (("raw", raw), ("boxed", boxed), ("jersey", crop)):
        path = cache / f"intro_{name}.jpg"
        cv2.imwrite(str(path), im, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[name] = path
    out["who"] = e.sheet.shown
    return out


def _photos(repo, sheet: Sheet, e: Entry, sub) -> None:
    """The person page's pictures: up to MAX_PHOTOS confirmed scenes spread over the clip."""
    from pfb_app.export import pictures   # noqa: PLC0415
    from pfb_app.export import sheets as S   # noqa: PLC0415
    pid = repo.get_subject(sheet.subject_id).get("participant_id")
    for m in sheet.moments:
        if m.ref_id in e.refs:
            m.cited = True          # the page draws the confirmed scenes (not only those the text cites)
            if m._request is None and not m.other_person:
                m._request = S._picture_request(sub, m, pid)
    mine = sorted((m for m in sheet.moments if m.ref_id in e.refs and m._request is not None),
                  key=lambda m: (m.t_s or 0.0, m.ref_id))
    if len(mine) > MAX_PHOTOS:
        idx = sorted({round(i * (len(mine) - 1) / (MAX_PHOTOS - 1)) for i in range(MAX_PHOTOS)})
        mine = [mine[i] for i in idx]
    made = pictures.make_pictures([m._request for m in mine], sub.cache_dir)
    for k, m in enumerate(mine, 1):
        m.picture = made.get(m._request)
        if m.picture is not None:
            m.number = k


def _numbers(repo, sub, cam, sub0, fps: float, entries: list[Entry]) -> dict:
    """Values measured on this clip (the boxes and the number reads the app holds)."""
    from pfb_app.export import demo_scenes as D   # noqa: PLC0415
    from pfb_app.domain.models import Stage   # noqa: PLC0415
    from pfb_app.services import auto_people   # noqa: PLC0415
    labels_ = list(cam.boxes.labels)
    rows = 0
    longest, longest_id, runs4 = 0, None, 0
    for lb in labels_:
        fr, _ = cam.boxes.boxes(lb)
        rows += len(fr)
        for a, b in D.unbroken_runs(cam, [lb]):
            n = b - a + 1
            runs4 += n / fps >= 4.0
            if n > longest:
                longest, longest_id = n, lb
    n_frames = cam.meta.n_frames if cam.meta else None
    reads = ((auto_people._number_run(repo, sub.sid, cam.name, cam.track).output or {}).get("cues") or {}).get("number") or {}
    return {
        "fps": fps, "frames": n_frames, "clip_s": round(n_frames / fps) if n_frames else None,
        "ids": len(labels_), "ids_with_number": sum(1 for r in reads.values() if isinstance(r, dict) and r.get("dominant") is not None),
        "boxes_per_frame": round(rows / n_frames, 1) if n_frames else None,
        "longest_run_s": round(longest / fps, 1), "longest_run_id": longest_id, "runs_4s": int(runs4),
        "scenes_all": sum(e.n_all for e in entries), "scenes_ok": sum(e.n_ok for e in entries),
    }


# --- the booklet ----------------------------------------------------------------------------

def _pct(a: int, b: int) -> int:
    return round(100 * a / b) if b else 0


def build(repo, assignment_id: int, *, credits=None, contact=None, subtitle=SUBTITLE,
          credit_short=None, title="NiceMill 処理出力", authors=booklet.AUTHORS,
          repo_url=booklet.REPO_URL, repo_note=booklet.REPO_NOTE, edition=booklet.EDITION,
          series=booklet.SERIES, min_confirmed=MIN_CONFIRMED, floor=FLOOR) -> bytes:
    from pfb_app.export import sheets as _sheets   # noqa: PLC0415
    keep, _sheets.MAX_PICTURES = _sheets.MAX_PICTURES, 99
    try:
        report = build_report(repo, assignment_id, drafts=True, identity_gate=False,
                              with_pictures=False)
    finally:
        _sheets.MAX_PICTURES = keep
    import re   # noqa: PLC0415
    for s in report.exported:          # no uncertain number on a page: a person without a confirmed number is 選手 ID n
        if s.jersey_state != "confirmed":
            m = re.search(r"\d+", s.jersey_id or "")
            s.jersey = f"選手 ID{m.group()}" if m else s.name
            s.jersey_id = None
    ev = gather(repo, assignment_id, report)
    shown, used = choose(ev.entries, min_confirmed, floor)
    for e in shown:
        s = e.sheet
        s.moments = [m for m in s.moments if m.ref_id in e.refs]
        s.spans, s.court_points, s.distance_m = [], [], None
        s.ident_note = f"写真と件数は、同一人物と確認できた場面{e.n_ok}件（記録された場面{e.n_all}件のうち）を使っています。"
    examples = pick_examples(shown)
    n_people_all = len(ev.entries)
    ok_all = sum(e.n_ok for e in ev.entries)
    all_all = sum(e.n_all for e in ev.entries)
    ok_shown = sum(e.n_ok for e in shown)
    all_shown = sum(e.n_all for e in shown)
    lay = layout(len(shown), len(examples))
    org = _pdf._org(repo)
    env = _pdf._env.get_template("booklet.html.j2")
    credit_lines = _pdf._credit_lines(credits)
    common = _pdf._common(report, org, None, credits, contact)
    common.pop("src", None)
    common.update(stamp=_pdf._stamp(), credits=credit_lines)
    reach = int(max((e.reach_s for e in ev.entries), default=9.0))
    selection = (f"掲載は、同一人物と確認できた場面のある人物の例です。全{n_people_all}人のうち、"
                 f"人物ページを載せたのは{len(shown)}人です。{LIMIT_NOTE}")
    measured = (f"確認できた場面は、全{n_people_all}人の記録された場面{all_all}件のうち{ok_all}件"
                f"（{_pct(ok_all, all_all)}%）。掲載した{len(shown)}人では{all_shown}件のうち{ok_shown}件です。"
                "人物ページの文章は記録された全場面から作り、写真と件数は確認できた場面を使っています。"
                "この冊子の数字は、この処理の枠の記録と背番号の読み取り記録から測った値です（開発版の再現率だけは別のクリップでの測定）。")
    criteria = (f"場面を確認した基準は三つです。(1)その人物IDで最も多く読み取られた背番号が、場面の前後{reach}秒以内に"
                "読み取られている。(2)場面の前後1秒、人物の枠がほぼ全フレームにあり、飛んでいない。"
                "(3)その間、ほかの人物の枠と半分以上重ならず、ほかの人物が同じ背番号を読まれていない。"
                "背番号は約6秒ごとに読み取っています。")
    about = {
        "what": f"この冊子は、{report.title}の映像を NiceMill で処理した出力のうち、確認できた部分を前に出したデモ版です。"
                "はじめに NiceMill とは何かを示し、人物ごとに見開き2ページ、続いて「検出の例」「検出の数字」「できること・これから」を載せています。",
        "ai": AI_LINE + " 人物の呼び方は選手IDです。背番号は確認できたものだけ載せています。",
        "fix": "気づいた点は、評価する人に伝えてください。訂正は版として履歴に残ります。",
        "left_out": selection,
        "selection": selection, "measured": measured, "criteria": criteria,
        "tracking_note": "",
    }
    qr = booklet._qr_svg_path(repo_url)
    cover = {"title": title, "work": report.title, "subtitle": subtitle, "people": len(shown),
             "credit_short": credit_short, "series": series, "authors": list(authors),
             "repo_url": repo_url, "repo_note": repo_note, "edition": edition, "qr": qr,
             "date": booklet._today()}
    first = lay["first"]
    intro_page = first
    toc = [{"name": "NiceMillとは", "id": None, "page": intro_page}]
    toc += [{"name": views.person(report, e.sheet, "handout")["name"], "id": e.sheet.jersey_id,
             "page": first + 2 + 2 * k} for k, e in enumerate(shown)]
    ex_first = first + 2 + 2 * len(shown)
    toc += [{"name": "検出の例", "id": None, "page": ex_first}] if examples else []
    num_page = first + 2 + 2 * (len(shown) + lay["strips"])
    toc += [{"name": "検出の数字", "id": None, "page": num_page},
            {"name": "できること・これから", "id": None, "page": num_page + 2}]
    hero = max((e.hero for e in shown if e.hero), key=lambda h: h.b - h.a, default=None)
    numbers = _numbers_ctx(ev, shown, used, hero, n_people_all, len(examples))
    ctx = dict(common, cols=2, page_title=report.title, kind="cover", n=1, cover=cover, title=report.title,
               date=report.date, words=report.words, toc=toc, toc_back=None, about=about, p=None,
               contact=contact or CONTACT, ai_line=AI_LINE, demo=True, numbers=numbers)
    A = _pdf._Assets
    R = booklet._render
    done: dict[int, object] = {1: R(env, A, **ctx)}
    n = 2
    if lay["front"]:
        done[2] = R(env, A, **dict(ctx, kind="front_l", n=2))
        done[3] = R(env, A, **dict(ctx, kind="front_r", n=3))
        n = 4
    assert n == intro_page
    n += 2                                           # the intro spread is made last (it shows the pages)
    lefts = []
    for e in shown:
        s = e.sheet
        p = views.person(report, s, "handout", accent=org.accent_hex, width=520)
        rows = [{"when": m.when, "what": views.caption(report, s, m.what, "handout"), "why": m.why or ""}
                for m in sorted(s.moments, key=lambda m: (m.t_s or 0.0, m.ref_id))]
        facts = [f"確認できた場面：{e.n_ok}件（記録された場面{e.n_all}件のうち）"]
        if e.longest_run_s:
            facts.append(f"枠が途切れず追えた最長の区間：{e.longest_run_s}秒")
        counts = s.counts(report.categories)
        if counts:
            facts.append("確認できた場面の種類：" + "、".join(f"{c.label} {k}回" for c, k in counts))
        npic = len(p["pictures"])
        left = dict(ctx, kind="left", p=p, credits=[], all_rows=rows, rows_title="確認できた場面", smax=5.0,
                    cols=1 if npic <= 1 else 2, estimated=False, confirmed_n=e.n_ok)
        right = dict(ctx, kind="right", p=p, credits=credit_lines[:1], facts=facts, rows=rows, rows_title="確認できた場面", smax=5.0,
                     estimated=False)
        assert n % 2 == 0, n
        done[n] = R(env, A, **dict(left, n=n))
        lefts.append(n)
        done[n + 1] = R(env, A, variants=({"extras": True}, {"extras": False}), **dict(right, n=n + 1))
        n += 2
    k = 0
    for sp in range(lay["strips"]):
        for side in (0, 1):
            ex = examples[k] if k < len(examples) else None
            k += 1
            if ex is not None:
                page = dict(ctx, kind="strip", ex=ex, idx=k, cols=3, total=len(examples), per=len(ex.images),
                            step=round(demo_step(ex), 1))
            else:
                page = dict(ctx, kind="strip_note", examples=examples)
            done[n] = R(env, A, **dict(page, n=n))
            n += 1
    done[n] = R(env, A, **dict(ctx, kind="numbers_l", n=n, smax=5.0))
    done[n + 1] = R(env, A, **dict(ctx, kind="numbers_r", n=n + 1, smax=6.0))
    done[n + 2] = R(env, A, **dict(ctx, kind="can_l", n=n + 2, smax=5.0))
    done[n + 3] = R(env, A, **dict(ctx, kind="can_r", n=n + 3, smax=5.0))
    n += 4
    # the intro spread: real pictures of this clip and the real pages of the first people
    thumbs = [_thumb(done[ln]) for ln in lefts[:4]]
    vis = ev.visuals or {}
    scene = next((m.picture for e in shown for m in e.sheet.moments if m.picture is not None), None)
    first_text = (shown[0].sheet.text or "") if shown else ""
    quote = first_text.split("。")[0] + "。" if first_text else ""
    intro = {"raw": vis.get("raw"), "boxed": vis.get("boxed"), "jersey": vis.get("jersey"), "scene": scene,
             "quote": quote, "thumbs": [t for t in thumbs if t], "people": n_people_all,
             "clip_s": ev.numbers.get("clip_s")}
    done[intro_page] = R(env, A, **dict(ctx, kind="intro_l", n=intro_page, intro=intro, smax=5.0))
    done[intro_page + 1] = R(env, A, **dict(ctx, kind="intro_r", n=intro_page + 1, intro=intro, smax=5.0))
    done[n] = R(env, A, **dict(ctx, kind="back", n=n, toc_back=None if lay["front"] else toc, tightmin=0.7))
    pages = [done[i] for i in range(1, n + 1)]
    assert n == lay["total"] and len(pages) == lay["total"], (n, len(pages), lay)
    combined = pages[0].copy([pg for d in pages for pg in d.pages])
    combined.metadata.title = f"{title} {report.title} デモ版"
    combined.metadata.generator = _pdf.BRAND_NAME
    build.last = {"people_all": n_people_all, "people_shown": len(shown), "min": used,
                  "scenes_ok": ok_all, "scenes_all": all_all, "examples": len(examples),
                  "pages": len(pages)}
    return combined.write_pdf()


def demo_step(ex: Example) -> float:
    return (ex.b - ex.a) / max(len(ex.images) - 1, 1) / ex.fps


def _thumb(doc) -> Path | None:
    """A PNG of a rendered one-page document (the real output page, shown small in the intro)."""
    import tempfile   # noqa: PLC0415
    import pypdfium2 as pdfium   # noqa: PLC0415
    try:
        img = pdfium.PdfDocument(doc.write_pdf())[0].render(scale=0.9).to_pil().convert("RGB")
        path = Path(tempfile.mkdtemp(prefix="nicemill_thumb_")) / "page.png"
        img.save(path)
        return path
    except Exception:   # noqa: BLE001 - a missing thumbnail only leaves the picture out
        return None


def _numbers_ctx(ev: Evidence, shown: list[Entry], used: int, hero, n_people_all: int, n_examples: int = 0) -> dict:
    m = ev.numbers
    rows = []
    for e in shown:
        rows.append({"who": e.sheet.shown, "id": e.sheet.jersey_id, "ok": e.n_ok, "all": e.n_all,
                     "longest": e.longest_run_s, "runs4": e.runs_4s})
    return {"clip": m, "people": rows, "recall": RECALL, "recall_note": RECALL_NOTE, "fp_note": FP_NOTE,
            "bars": [{**r} for r in RECALL], "hero": hero, "n_people": n_people_all, "examples": n_examples,
            "hero_who": hero.jersey_id if hero else None}
