"""Apply the action-claim check (pipeline.claim_check) to a package's moments.

Reads measured pose cues from a JSON file made by an external pose tool (not included) ({ref_id: {"wrist_above_head",
"pose_frames"}}), so the check runs with the app's own Python and no model; `ask` (optional) is the
local vision model's second question. A weakened sentence only says less than the original: `why`
becomes the weaker text, `why_orig` keeps the original, `weak` marks it for 「要確認」. Good moments
without a cue or without an action to check are left as they were (unchecked); an unverified action
sentence of the other recorded actions is hedged. The database is not touched.
"""
from __future__ import annotations

import json
import logging
import re

from pfb_app.pipeline import claim_check as cc
from pfb_app.review_video.model import KIND_EVENT, KIND_GOOD

log = logging.getLogger(__name__)


def load_cues(path) -> dict:
    try:
        return json.load(open(path, encoding="utf8"))
    except (OSError, ValueError):
        log.warning("claim cues unreadable: %s", path)
        return {}


def _weaken(m, why: str, claim: dict) -> None:
    # the category label (「シュート・得点」…) names the same action as the sentence: it goes with it, or the
    # card's detail line and the highlight caption would state the action the sentence no longer does
    m.why_orig, m.why, m.weak, m.claim = m.why or m.what, why, True, claim
    m.category_orig, m.category = m.category, None


def apply(people, cues: dict, images_of=None, ask=None) -> dict:
    """Check the action sentences of `people`; returns {"checked", "weakened", "fallback", "events"}.

    * the good moments (よかった所): checked against the cues / the second question as before; a moment
      with no cue and no question stays as generated (status "unchecked").
    * the other recorded actions (行動の記録): checked the same way where a cue exists (a shot needs
      an arm above the head), and otherwise never stated as a fact: a sentence that claims an action
      (shoot, dribble, pass, defend, run) and was not verified is hedged ("〜ているように見える（要確認）").
      A sentence that claims no action to check ("倒れた", "両手を挙げた") is left as it was.
    The sentence the moment shows is `why` (else `what`): it is the one that is checked and weakened.
    """
    n = weak = fb = ev = 0
    for p in people:
        for m in list(p.moments) + list(p.events):
            if m.kind not in (KIND_GOOD, KIND_EVENT):
                continue
            text = m.why or m.what
            if not text:
                continue
            c = cues.get(m.ref_id)
            cue = cc.Cues(c.get("wrist_above_head"), int(c.get("pose_frames") or 0)) if c else None
            imgs = images_of(p, m) if (images_of and ask) else None
            out = cc.check_moment({"why": text}, cue, imgs, ask)
            claim = out["claim_check"]
            m.claim = claim
            if claim["status"] != "unchecked":
                n += 1
            if claim["show"] != "keep":
                _weaken(m, out["why_shown"], claim)
                weak += 1
                fb += claim["show"] == "fallback"
                ev += m.kind == KIND_EVENT
            elif m.kind == KIND_EVENT and claim["action"] != cc.NO_CLAIM:
                # nothing verified this action: not stated as a fact
                _weaken(m, cc.hedge(text), dict(claim, show="hedge",
                                                reasons=list(claim.get("reasons") or []) + ["行動の記録は動作を確かめていない"]))
                weak += 1
                ev += 1
    return {"checked": n, "weakened": weak, "fallback": fb, "events": ev}


# --- the evaluation text of a person (made from the checked scenes only) ------------------------------
#
# The generated paragraph of the app (the pfb stage) restates the actions of every counted scene, also
# of those the action check weakened. For a text of everyone, the local model is asked again with the
# counted scenes that passed the check only (evalgen.py), and what it writes is checked once more here:
# a sentence that claims an action no checked scene carries is hedged, a shot is never stated without a
# verified shot, a result (得点...) no scene states is hedged, a comparison between people is dropped,
# and a jersey number always carries its 「？」. Nothing is added: a sentence only ever says less.

_RESULT = re.compile(r"得点|ゴール|決め|決ま|成功|アシスト|勝|リード|逆転|ファウル")
_COMPARE = re.compile(r"一番|最も|もっとも|随一|トップ|ナンバー|ほかの選手より|他の選手より|チーム(?:内|で)|上位|最多|最高|1位|一位|エース|MVP|誰より")
_NUMBER = re.compile(r"(\d+)番(?!？)")


def sentence_actions(sentence: str) -> set[str]:
    """Every action a sentence claims (a block is a block, not also a shot)."""
    found = {key for key, rx in cc._ORDER if re.search(rx, sentence or "")}
    if "defend" in found:
        found.discard("shoot")
    return found


def eligible(person) -> list:
    """The counted scenes a text may be made from: the action check did not weaken them, the identity
    check did not flag them, and a scene that states a shot is never one (it was not verified here)."""
    out = []
    for m in person.moments:
        if m.weak or m.suspect or m.t_s is None:
            continue
        if "shoot" in sentence_actions(m.why or m.what) and (m.claim or {}).get("status") != "supported":
            continue
        out.append(m)
    return out


def _hedge_polite(sentence: str) -> str:
    t = re.sub(r"[。.\s]+$", "", sentence.strip())
    if t.endswith("ました"):
        return t[:-3] + "たように見えます（要確認）。"
    if t.endswith("ています") or t.endswith("です") or t.endswith("ます"):
        return t[:-2] + "ているように見えます（要確認）。" if t.endswith("ています") else t + "（要確認）。"
    return cc.hedge(t)


def guard_text(text: str, scenes: list) -> tuple[str | None, list[str]]:
    """(the text to show, what was changed). `scenes`: the person's eligible scenes."""
    allowed: set[str] = set()
    said = "\n".join((m.why or m.what or "") for m in scenes)
    shot_ok = any("shoot" in sentence_actions(m.why or m.what) for m in scenes)
    for m in scenes:
        allowed |= sentence_actions(m.why or m.what)
    changes: list[str] = []
    out: list[str] = []
    for s in re.split(r"(?<=。)", text or ""):
        if not s.strip():
            continue
        if _COMPARE.search(s):
            changes.append("人どうしの比較に読める文を外した")
            continue
        acts = sentence_actions(s)
        bad = {a for a in acts if a not in allowed or (a == "shoot" and not shot_ok)}
        res = [w for w in _RESULT.findall(s) if w not in said]
        if bad or res:
            changes.append("確かめた場面にない動作・結果を言う文を、断定しない形にした")
            s = _hedge_polite(s)
        out.append(_NUMBER.sub(r"\1番？", s))
    return ("".join(out).strip() or None), changes


def apply_eval_texts(people, data: dict) -> dict:
    """Attach the evaluation texts of a file made by evalgen.py ({"<subject id>": {"text", "source"}}).

    Every person gets one of: the guarded AI text (source "ai"), the app's fact-only short text
    (source "short"), or no text with the reason in `eval_why`. People set apart as non-players are
    left alone. Returns the counts for the page's notes."""
    n_ai = n_short = hedged = 0
    none: list[str] = []
    for p in people:
        if p.excluded:
            p.eval_why = "選手以外（推定）のため、評価文は作っていません。"
            continue
        d = (data or {}).get(str(p.subject_id)) or {}
        text, src = d.get("text"), d.get("source")
        if text and src == "ai":
            text, changes = guard_text(text, eligible(p))
            p.eval_hedged = len(changes)
            hedged += len(changes)
        if text:
            p.eval_text, p.eval_source = text, src or "ai"
            n_ai += src == "ai"
            n_short += src == "short"
        else:
            p.eval_why = d.get("why") or "評価文の材料（確かめた場面）がありません。"
            none.append(p.shown())
    return {"ai": n_ai, "short": n_short, "hedged": hedged, "none": none}
