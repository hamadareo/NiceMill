"""Load zero-shot prompt files and assemble the data part of the user text."""
import hashlib
from dataclasses import dataclass

from pfb_app.config import Settings
from pfb_app.domain import profiles
from pfb_app.domain.models import Kind, Stage

_FORBIDDEN_MARKERS = ("例:", "例えば", "例）", "good example", "bad example", "sample")


@dataclass(frozen=True)
class Prompt:
    text: str
    hash: str


def load_prompt(settings: Settings, kind: Kind, stage: Stage, domain: str = "school",
                profile: str | None = None) -> Prompt:
    """The stage's prompt. A video of an activity-type profile with its own prompts
    (話し合い・会議, 発表, 研修, 学習活動: prompts/video_<profile>/<stage>.md, contribution and
    pfb) uses them; a sports video uses prompts/video_sports/<stage>.md when that file exists
    (contribution, pfb); else the general prompt prompts/video/<stage>.md. 一般 and スポーツ
    have no folder of their own: their prompts are the ones from before profiles."""
    if kind == Kind.video and profile and profile not in profiles.UNCHANGED:
        path = settings.prompts_dir / f"video_{profile}" / f"{stage.value}.md"
        if path.is_file():
            return _load(path)
    if domain == "sports" and kind == Kind.video:
        path = settings.prompts_dir / "video_sports" / f"{stage.value}.md"
        if path.is_file():
            return _load(path)
    return load_prompt_file(settings, kind, stage.value)


def load_prompt_file(settings: Settings, kind: Kind, name: str) -> Prompt:
    """A prompt that is not a stage's own (e.g. video/translate.md)."""
    return _load(settings.prompts_dir / kind.value / f"{name}.md")


def load_prompt_named(settings: Settings, kind: Kind, name: str, domain: str = "school") -> Prompt:
    """A prompt that is not a stage's own, with the sports variant when there is one
    (prompts/video_sports/<name>.md), else prompts/<kind>/<name>.md."""
    if domain == "sports" and kind == Kind.video:
        path = settings.prompts_dir / "video_sports" / f"{name}.md"
        if path.is_file():
            return _load(path)
    return load_prompt_file(settings, kind, name)


def _load(path) -> Prompt:
    text = path.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return Prompt(text=text, hash=digest)


def build_user_text(sections: list[tuple[str, str]]) -> str:
    """Wrap each data section so the model can tell data from instructions."""
    return "\n\n".join(
        f"【{name}・データ開始】\n{body}\n【{name}・データ終了】" for name, body in sections
    )


def zero_shot_violations(text: str) -> list[str]:
    """Return the example markers found in text (case-insensitive)."""
    lowered = text.lower()
    return [m for m in _FORBIDDEN_MARKERS if m in lowered]
