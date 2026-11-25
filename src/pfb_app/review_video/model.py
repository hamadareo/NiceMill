"""The data a review video and the review player are made from (no database, no model)."""
from __future__ import annotations

from dataclasses import dataclass, field

from pfb_app.export.explainer_video import Source

# DRAFT_NOTICE never says 承認, 渡す or 下書き: an output may go out before anyone checked it
# (「あとから訂正」). BAND_NOTE, EVAL_AI_NOTE and the player still say 「AIの下書き」 (2026-11-23:
# only this notice was changed)
DRAFT_NOTICE = "AIが作成した文です。違っていれば、あとから直せます。"
CHECK_NOTE = "AIの判定・要確認"
SUSPECT_NOTE = "人物が別人かも・要確認。別の人の動きかもしれません。"
NUMBER_NOTE = "背番号はAIが読んだもの。違っていたら直してください。"
NUMBER_NOTE_UNVERIFIED = "背番号はAIの読み取りで、精度は未検証です。違っていたら直してください。"
GATE_NOTE = ("背番号の読み取りは、まだ十分に確かめられていません（試験では7人中6人。人数も少ない）。"
             "「？」の付いた番号は、AIが読んだものです。")
BAND_NOTE = "AIの下書き・背番号は未検証"
BAND_NOTE_VERIFIED = "AIの下書き・要確認"
NO_MARK_NOTE = "「別の人かも」の印が無くても、別の人でないとは限りません（判定の材料がない場合があります）。"
CHECKED = {True: "人物の照合：番号の時系列で確かめました", False: "人物の照合：材料がなく、確かめていません"}
FIX_NOTE = "直すときは、NiceMillの「人物と参加者の対応づけ」で直します。"
# the number-change note says 「読み取った」 only while the number reading has passed the check; until then
# it says what the AI saw, with 「？」 like every other number on screen
CHANGE_NOTE = ("この付近で、AIが読んだ背番号が {a}番？ から {b}番？ に変わったように見えます（読み取りは未検証）。"
               "人物が別人かも・要確認。")
CHANGE_NOTE_VERIFIED = "この付近で、読み取った背番号が {a}番 から {b}番 に変わりました。人物が別人かも・要確認。"


def change_note(a, b, verified: bool = False) -> str:
    return (CHANGE_NOTE_VERIFIED if verified else CHANGE_NOTE).format(a=a, b=b)
UNKNOWN = "番号不明"
KIND_GOOD, KIND_EVENT, KIND_CHANGE = "good", "event", "change"
KIND_WORDS = {KIND_GOOD: "よかった所", KIND_EVENT: "行動の記録", KIND_CHANGE: "番号が変わった所"}
# One colour per TEAM (not per person): A, B and, for anybody the colour groups could not place,
# a neutral dark grey. The groups come from the shirt colour (the person sheets' torso colour, or
# the names stage's T1/T2); "A"/"B" are only the two groups, never a club's name or kit colour.
# Reserved elsewhere: red = ball holder, yellow = court outline, flags are symbols (not colours).
# Distinctness (Lab dE, normal / deuteranopia / protanopia / tritanopia) is checked in the tests.
TEAM_COLORS = {"A": "#0072B2", "B": "#CC79A7", None: "#222222"}
TEAM_WORDS = {"A": "チームA", "B": "チームB", None: "チーム不明"}


def _lum(hex_: str) -> float:
    def lin(v: int) -> float:
        c = v / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_[i:i + 2], 16) for i in (1, 3, 5))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast(a: str, b: str) -> float:
    x, y = _lum(a), _lum(b)
    return (max(x, y) + 0.05) / (min(x, y) + 0.05)


def team_color(letter) -> dict:
    """{hex, rgb, ink (text colour on it, contrast >= 4.5)} of a team letter ("A"/"B"/None)."""
    hex_ = TEAM_COLORS.get(letter, TEAM_COLORS[None])
    ink = "#ffffff" if contrast(hex_, "#ffffff") >= contrast(hex_, "#1a1f1d") else "#1a1f1d"
    return {"hex": hex_, "rgb": tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5)), "ink": ink,
            "ink_rgb": tuple(int(ink[i:i + 2], 16) for i in (1, 3, 5))}


BALL_NOTE = "赤い丸は、ボールを持っている人の推定です。ボールが小さく見えない時間が多いため、出ない時間のほうが長く、別の人を指すこともあります。"
NONPLAYER_NOTE = "選手以外かも"   # a hint, never a removal: nobody is dropped on it
NONPLAYER_WHY_OTHER = "服の色が両チームのどちらとも違う（審判・控えなど、選手以外かも）"
NONPLAYER_WHY_CHECK = "コートの中で動いている根拠が弱い（選手以外かも）"
NONPLAYER_WHY_IMAGE = "服の色が両チームのどちらとも違う（映像からの推定。審判・コーチなど選手以外かも）"
NONPLAYER_SET_APART = "選手以外（推定）。違えば「選手に戻す」で戻せます。"
TEAM_LETTERS = {"T1": "Aチーム", "T2": "Bチーム"}   # the colour is not decided: A / B


@dataclass
class Moment:
    ref_id: str
    camera: str | None
    t_s: float | None            # seconds in the (playable) video; None: no time recorded
    when: str                    # 「試合時間 12:30」 / 「動画の時刻 03:05」 as the handout says it
    what: str                    # the generated observation, as generated
    why: str                     # the generated reason it counted, as generated
    category: str | None = None
    cited: bool = False          # the text cites it as evidence
    suspect: bool = False        # the preliminary identity check: the box may be on someone else
    kind: str = KIND_GOOD        # good (a よかった所) | event (any recorded action) | change (the
                                 # read number changed on this track)
    note: str = ""               # a fixed explanation (change markers)
    why_orig: str = ""           # the sentence as generated, when the action check weakened `why`
    claim: dict | None = None    # the action check's verdict (pipeline.claim_check), None: not checked
    category_orig: str | None = None   # the category label, withheld while the sentence is weakened
    weak: bool = False           # `why` was weakened by the action check (shown as 要確認)
    nums: tuple | None = None    # a number-change marker: the (before, after) numbers read


@dataclass
class Person:
    subject_id: int
    name: str
    labels: dict[str, list[str]] = field(default_factory=dict)   # camera -> track labels
    status: str = "draft"        # "approved" | "draft"  (what the text's state is, as the app says)
    text: str | None = None      # the generated feedback text (draft or approved), as generated
    moments: list[Moment] = field(default_factory=list)       # the よかった所 (kind good)
    events: list[Moment] = field(default_factory=list)        # the other recorded actions + change marks
    label: str = ""              # how the person is shown: 「Aチーム 3番」 / 「3番」 / 「番号不明（P2）」
    label_note: str = ""         # e.g. 「番号が重複・要確認」
    number: str | None = None    # the read jersey number the label stands on (None: unknown)
    team: str | None = None      # "T1" | "T2" | None
    internal: str = ""           # the tracker's own id (P2), kept small
    conflict: bool = False       # the person's tracks read different numbers
    verified: bool = False       # the number cue passed the precision gate (else: AI reading, 「？」)
    team_letter: str | None = None   # "A" | "B" | None: the colour group (shirt colour), drawn in that colour
    np_score: float | None = None    # 0..1 from the picture (nonplayer.py); None: not judged
    excluded: bool = False       # set apart as 選手以外（推定）: hidden by default, one click brings back
    nonplayer: str = ""          # why this may not be a player (the app's own cues); "" = no sign
    checked: bool = False        # a number timeline existed, so the identity check could run
    eval_text: str | None = None     # the evaluation text made from the checked scenes only (claims.apply_eval_texts), None: none made
    eval_source: str = ""            # "ai" (the local model) | "short" (the app's fact-only short text)
    eval_hedged: int = 0             # sentences of the text weakened or dropped by the text guard
    eval_why: str = ""               # why no evaluation text could be made

    def shown(self) -> str:
        """The label (the name when none was made)."""
        return self.label or self.name

    def tag(self) -> str:
        """The label for a box tag / a row name: with the tracker id when the number alone does
        not tell this person from another (a duplicate or a conflict)."""
        s = self.shown()
        if self.label_note and self.internal and self.internal not in s:
            s += f"（{self.internal}）"
        return s

    @property
    def dashed(self) -> bool:
        """Drawn with a dashed box: a number the gate has not vouched for."""
        return self.number is not None and not self.verified

    def everything(self) -> list["Moment"]:
        return sorted(self.moments + self.events,
                      key=lambda m: (m.t_s if m.t_s is not None else 1e18, m.ref_id))

    def highlights(self) -> list[Moment]:
        """The moments an edit shows: the cited ones (all positive ones when none is cited),
        with a time, in time order."""
        timed = [m for m in self.moments if m.t_s is not None]
        pick = [m for m in timed if m.cited] or timed
        return sorted(pick, key=lambda m: (m.t_s, m.ref_id))


@dataclass
class Package:
    title: str
    date: str
    others: str                                   # 「ほかの選手」
    people: list[Person]
    sources: dict[str, Source]                    # camera -> playable file + tracks
    credit_lines: list[str] = field(default_factory=list)   # e.g. the CC BY attribution
    notes: list[str] = field(default_factory=list)          # what was left out, and why
    number_verified: bool = False                 # the number cue passed the gate (all cameras)
    holders: list = field(default_factory=list)   # [[first_frame, last_frame, track label], ...]: the ball holder (an estimate)
    extras: dict = field(default_factory=dict)    # camera -> ndarray rows of unnamed detections (export.extra_boxes)
    court: dict | None = None                     # {"frames": [[frame, x1, y1, ... x4, y4]], "note"}: the registered court's outline in the picture
    ball_frames: int = 0                          # frames with a ball sighting (0: no ball detection was given)
    court_est: bool = False                       # the court's sidelines are to be estimated from the picture (no registered court)
    pose_path: str | None = None                  # a pose file (an external pose tool (not included)): hands above the head / crouching
    every: dict | None = None                     # cache of everyone.analyse (made once per package)

    def person(self, subject_id: int) -> Person:
        for p in self.people:
            if p.subject_id == subject_id:
                return p
        raise KeyError(subject_id)


def date_in_title(title: str, date: str) -> bool:
    """True when the title already says the date (then it is not written a second time)."""
    day = date.split("年")[-1] if date else ""
    return not date or date in title or bool(day and day in title)


COURT_NOTE = "コート外周は、アプリに登録されたコートの位置（登録がある映像だけ）から描いています。映像から新しく推定した線ではありません。"


def ink_for(hex_: str) -> str:
    """The text colour (white or near black) that reads on ."""
    return "#ffffff" if contrast(hex_, "#ffffff") >= contrast(hex_, "#1a1f1d") else "#1a1f1d"


def court_at(court, frame: int):
    """The court outline [(x, y) x4] of the sample nearest `frame`, or None (no court / too far)."""
    if not court or not court.get("frames"):
        return None
    row = min(court["frames"], key=lambda r: abs(r[0] - frame))
    if abs(row[0] - frame) > 45:
        return None
    return [(row[1 + 2 * i], row[2 + 2 * i]) for i in range(4)]


EXTRA_NOTE = ("薄い灰色の細い枠は「そのほかの人」です。追跡が名前を付けなかった人の検出で、コーチや座っている人が入ることもあります。"
              "選んだ人の枠が途切れる時間や、検出されない人は、直っていません。")


CLAIM_NOTE = ("「要確認」の文は、AIの文を断定しない形に直したものです。動作（シュート・ドリブルなど）を映像で"
              "確かめていない文や、確かめて合わなかった文が入っています。")
NO_CUES_NOTE = ("動作（シュート・ドリブルなど）を映像で確かめる検査はしていません。文に書かれた動作は、"
                "AIの記録のままです。映像で確かめてください。")
EVENT_NOTE = "この行動の文は、AIの記録のままで、動作を映像で確かめていません。"
EVAL_AI_NOTE = ("AIの下書きです。動作の確認を通った場面だけをもとに作りました。確かめられない動作の文は書かず、"
                "断定できない文は「要確認」の形にしてあります。評価する人が映像で確かめてから渡します。")
EVAL_SHORT_NOTE = "映っている場面が少なく、AIの文章は作れませんでした。確かな事実だけの短い文です。"
EVAL_NONE_NOTE = "この人の評価文は、まだありません。"
EVENT_WEAK_NOTE = "動作を映像で確かめていないため、断定しない形にしています。"


TEAM_COLOUR_NOTE = ("枠の色は、映像から推定した服の色です（人物の上半身の色の中央値。映像の明るさで、実際の色と少し違って見えます）。"
                    "背景に溶けないよう、太さと黒か白の縁取りで補っています。チームの名前ではありません。")
TEAM_ADJUST_NOTE = "2チームの服の色が似ていて見分けにくいため、{name}のチームの枠は、見分けやすい色に寄せています（実際の服の色とは違います）。"
ACT_NOTE = ("頭上の吹き出しの「いまの様子」は、映像の中の位置の動き（カメラの動きは除いています）、体の向き、ボールの位置から"
            "機械が推定した短い言葉です。外れることがあります。シュート・パスなどの動作は、確かめた場面（◆）のほかには書きません。")
COURT_EST_NOTE = ("赤い線「コートの枠（推定）」は、床の色とコート外の縁取りの色の境目から推定した、見えている範囲の長い辺（サイドライン）です。"
                  "主に奥側の辺が出て、手前側の辺と短い辺・隅は、映像に映らなかったり見分けにくかったりして、ほとんど描けていません。"
                  "線が外れる場面や、出ない場面があります。選手たちの足の位置と合わない線は出していません。")
