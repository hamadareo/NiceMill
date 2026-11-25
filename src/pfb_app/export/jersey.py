"""How a person is called in the draft export: by jersey number, never as a fact unless it is one.

What the app stores (services.auto_people / services.review):
- a person made automatically is 「選手n」 until a number is read on them (the number cue, which is
  provisional: the precision gate is not passed); then `number_names` renames them 「#10」
  (「#10（チーム1）」 when two share it) and records `meta.name_source[participant_id] = "number"`;
- a name the coach typed has `name_source` "coach" (and any other name is the coach's);
- a roster number (`subject.roster["number"]`) is typed by a person: confirmed.

So: confirmed (roster number, or a 「#n」 name the coach gave) -> 「背番号 8」;
a 「#n」 name made from the AI's reading -> 「背番号 8番？（推定）」;
a default 「選手n」 -> 「背番号 不明」. The tracker id (「人物ID 2」, from participant_id) goes with
every label so that people can be told apart. No 「さん」 is added anywhere.
"""
import re

CONFIRMED, ESTIMATED, UNKNOWN = "confirmed", "estimated", "unknown"
_HASH = re.compile(r"[#＃]\s*(\d+)\s*(?:[（(]\s*(.+?)\s*[）)])?")
_DEFAULT = re.compile(r"選手\d+")


def person_id(subject: dict) -> str | None:
    m = re.search(r"\d+", subject.get("participant_id") or "")
    return m.group(0) if m else None


def label(subject: dict, name_source: dict | None) -> tuple[str, str | None, str]:
    """(label, small id line, state) for a subject; the state is confirmed / estimated / unknown
    (or "named" for a name the coach gave, shown as it is)."""
    name = (subject.get("display_name") or "").strip()
    pid = person_id(subject)
    small = f"人物ID {pid}" if pid else None
    roster = (subject.get("roster") or {}) if isinstance(subject.get("roster"), dict) else {}
    source = (name_source or {}).get(subject.get("participant_id"))
    rn = re.sub(r"\D", "", str(roster.get("number") or ""))
    m = _HASH.fullmatch(name)
    team = f"・{m.group(2)}" if m and m.group(2) else ""
    if rn:
        return f"背番号 {rn}", small, CONFIRMED
    if m and source == "coach":
        return f"背番号 {m.group(1)}{team}", small, CONFIRMED
    if m:
        return f"背番号 {m.group(1)}番？（推定）{team}", small, ESTIMATED
    if not name or _DEFAULT.fullmatch(name):
        return "背番号 不明", small, UNKNOWN
    return name, small, "named"
