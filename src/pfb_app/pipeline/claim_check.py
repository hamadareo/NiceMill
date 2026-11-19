"""Re-check what an action sentence claims (シュート, ドリブル, パス, 守り, 走る) against cues the
sentence was not made from, and weaken the sentence when the cues do not carry it.

Pure functions, no model and no database. Two independent cues are read by the caller:

* `Cues`: numbers measured on the frames around the moment (pose: is a wrist above the head;
  ball detections near the person). `scripts/claims_cues.py` measures them (engine venv).
* `Votes`: the answers of a second, separate question to the local vision model, asked about the
  evidence frames only (`ask_votes`): one question about the claimed action and one about "only
  something else". Each answer is はい / いいえ / 分からない.

`check(text, cues, votes)` returns a `Verdict`; `weaken(text, verdict)` the sentence to show.
Nothing here invents a fact: a weakened sentence only says less than the original.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

YES, NO, UNKNOWN = "はい", "いいえ", "分からない"

# claimed action -> words of the sentence (first match wins, so 「シュートをブロック」 is a block)
_ORDER = (
    ("defend", r"ブロック|防衛|防いだ|防ぎ|ディフェンス|守っ|守り"),
    ("shoot", r"シュート|投げ(?:た|る)|打っ|放(?:っ|つ)"),
    ("dribble", r"ドリブル|ドライブ"),
    ("pass", r"パス"),
    ("run", r"走っ|走る|走り"),
)

# a sentence that is only about a state ("ボールを持つ") claims no action to check
NO_CLAIM = "none"

# wrist above the nose by at least this share of the box height: a shot needs one (physical)
SHOOT_ARMS_UP = 0.05
MIN_POSE_FRAMES = 3
# 3 or more frames measured; fewer is "not measured"


def claimed_action(text: str) -> str:
    """The one action a sentence claims (shoot / dribble / pass / defend / run), else NO_CLAIM."""
    for key, rx in _ORDER:
        if re.search(rx, text or ""):
            return key
    return NO_CLAIM


@dataclass
class Cues:
    """Measured numbers; None = not measured (never a reason to veto)."""
    wrist_above_head: float | None = None   # best (head - wrist) / box height over the frames
    pose_frames: int = 0                    # frames the pose was found in


@dataclass
class Votes:
    pos: str = UNKNOWN    # 「その動作をしていますか」
    ctl: str = UNKNOWN    # 「その動作はせず、別の動きだけでしたか」


@dataclass
class Verdict:
    action: str
    status: str                  # supported | unclear | contradicted | unchecked
    confidence: str              # high | low | none
    show: str                    # keep | hedge | fallback
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"action": self.action, "status": self.status, "confidence": self.confidence,
                "show": self.show, "reasons": list(self.reasons)}


def physical_veto(action: str, cues: Cues | None) -> str | None:
    """The reason a necessary physical condition is not met, else None."""
    if cues is None or action != "shoot":
        return None
    if (cues.pose_frames >= MIN_POSE_FRAMES and cues.wrist_above_head is not None
            and cues.wrist_above_head < SHOOT_ARMS_UP):
        return "腕が頭より上に上がった場面がない"
    return None


def check(text: str, cues: Cues | None = None, votes: Votes | None = None) -> Verdict:
    """Combine the physical cue and the second question.

    * a physical necessary condition missing, or both answers against the claim (no to "did
      it", yes to "only something else"): contradicted -> the safe weak sentence;
    * "did it" yes and "only something else" not yes, and no veto: supported -> kept;
    * anything else (a lone no, answers that disagree, 分からない, a veto the model contradicts):
      unclear -> hedged ("〜しているように見える（要確認）"), never stated as a fact.
    Without any re-check the sentence is left as it was ("unchecked")."""
    action = claimed_action(text)
    if action == NO_CLAIM:
        return Verdict(action, "unchecked", "none", "keep")
    veto = physical_veto(action, cues)
    if votes is None and not veto:
        return Verdict(action, "unchecked", "none", "keep", ["動作の再質問をしていない"])
    v = votes or Votes()
    reasons = [veto] if veto else []
    if veto and v.pos == YES and v.ctl != YES:
        return Verdict(action, "unclear", "low", "hedge",
                       reasons + ["映像の再質問は見えたと答えたが、腕の動きの手がかりと合わない"])
    if veto or (v.pos == NO and v.ctl == YES):
        if not veto:
            reasons.append("映像の再質問で、その動作は見えず、別の動きだけだった")
        return Verdict(action, "contradicted", "high", "fallback", reasons)
    if v.pos == YES and v.ctl != YES:
        if action == "shoot" and not (cues and cues.wrist_above_head is not None
                                      and cues.pose_frames >= MIN_POSE_FRAMES):
            # a shot is only called supported when its necessary arm cue was measured and met
            return Verdict(action, "unclear", "low", "hedge",
                           ["映像の再質問は見えたと答えたが、腕の動きの手がかりを測れていない"])
        return Verdict(action, "supported", "high", "keep", ["映像の再質問で、その動作が見えた"])
    if v.pos == NO:
        reasons.append("映像の再質問で、その動作は見えない")
    elif v.pos == YES:
        reasons.append("再質問の答えが食い違う")
    else:
        reasons.append("映像からは判断できない")
    return Verdict(action, "unclear", "low", "hedge", reasons)


SAFE_TEXT = "この場面に映っていた（行動の内容は要確認）"
HEDGE_TAIL = "ように見える（要確認）"
# the plain past endings of an action sentence and the stem each hedges from
_PAST = (("した", "し"), ("った", "っ"), ("んだ", "ん"), ("いだ", "い"), ("れた", "れ"), ("きた", "き"),
         ("試みた", "試み"), ("みた", "み"))


def hedge(text: str) -> str:
    """「〜した。」 -> 「〜しているように見える（要確認）。」: the claim stays, but is no longer
    stated as a fact. A sentence in another form only gets 「（要確認）」."""
    t = re.sub(r"[。.\s]+$", "", (text or "").strip())
    for past, stem in _PAST:
        if t.endswith(past):
            return t[: -len(past)] + stem + "ている" + HEDGE_TAIL + "。"
    if t.endswith("する"):
        return t[:-2] + "している" + HEDGE_TAIL + "。"
    return t + "（要確認）。"


def weaken(text: str, verdict: Verdict) -> str:
    """The sentence to show for a verdict: kept, hedged, or the safe weak fact."""
    if verdict.show == "keep":
        return text
    if verdict.show == "hedge":
        return hedge(text)
    return SAFE_TEXT


def annotate(moment: dict, verdict: Verdict) -> dict:
    """A moment's package entry with the check recorded (keys added, the original kept):
    `claim_check` (the verdict) and `why_shown` (the sentence to show; `why` stays as made)."""
    out = dict(moment)
    out["claim_check"] = verdict.as_dict()
    out["why_shown"] = weaken(moment.get("why") or "", verdict)
    return out


# --- the second question (asked of a vision model about the evidence frames only) ----------------

# action -> (what the claimed action is, what "only something else" is)
QUESTIONS = {
    "shoot": ("ゴールに向けてボールを投げ上げる（シュートする）動作",
              "シュートはせず、立つ・歩く・走る・守る・ドリブルなど別の動きだけ"),
    "dribble": ("ボールを床につきながら運ぶ（ドリブルする）動作",
                "ドリブルはせず、ボールを持たずに動く、または別の動きだけ"),
    "pass": ("ボールをほかの選手に投げ渡す（パスする）動作", "パスはせず、別の動きだけ"),
    "defend": ("相手の選手にくっついて、または相手のシュートを防ぐように守る動作",
               "相手を守る動きはせず、別の動きだけ"),
    "run": ("ゴールの方向へ走る動作", "ゴールの方向へ走ってはおらず、立つ・歩く・別方向へ動くだけ"),
}
FRAME_INTRO = ("バスケットボールの試合を約1秒おきに時間順に並べた{n}枚の画像です。黄色い枠で囲まれた"
               "1人の人物だけに注目してください（ほかの人の動きは無視）。")
ANSWER_RULE = ("\n画像から確かに読み取れるときだけ答えてください。見えない・はっきりしないときは"
               "「分からない」と答えてください。答えは「はい」「いいえ」「分からない」のどれか1つだけ。")


def question_texts(action: str, n_frames: int) -> tuple[str, str] | None:
    """The two prompts (positive, control) for an action, or None when it has no question."""
    if action not in QUESTIONS:
        return None
    pos, ctl = QUESTIONS[action]
    head = FRAME_INTRO.format(n=n_frames)
    return (head + f"黄色い枠の人物は、この間に「{pos}」をしていますか。" + ANSWER_RULE,
            head + f"黄色い枠の人物は、この間「{ctl}」をしていましたか。" + ANSWER_RULE)


def parse_answer(reply: str | None) -> str:
    """はい / いいえ / 分からない from a reply; an empty or unreadable reply is 分からない (the
    model ran out of room or did not answer: never a yes)."""
    m = re.search(r"はい|いいえ|分からない|わからない", reply or "")
    return UNKNOWN if m is None or m.group(0) == "わからない" else m.group(0)


def ask_votes(action: str, images: list, ask) -> Votes | None:
    """Ask the two questions. `ask(images, prompt) -> str` is the caller's model call (the
    frames are the evidence frames of the moment with the person's box drawn). None when the
    action has no question."""
    texts = question_texts(action, len(images))
    if texts is None:
        return None
    return Votes(pos=parse_answer(ask(images, texts[0])), ctl=parse_answer(ask(images, texts[1])))


def check_moment(moment: dict, cues: Cues | None = None, images: list | None = None, ask=None) -> dict:
    """The one call the review video makes for a moment: reads `moment["why"]`, asks the second
    question when images and an `ask` function are given, and returns the moment with
    `claim_check` and `why_shown` added (see `annotate`)."""
    text = moment.get("why") or ""
    votes = None
    if images and ask is not None:
        votes = ask_votes(claimed_action(text), images, ask)
    return annotate(moment, check(text, cues, votes))
