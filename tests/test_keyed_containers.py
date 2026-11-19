"""A keyed container never shares its React key with a sibling (owner's report 2026-10-24).

Streamlit's frontend uses a block's id as its React key, and a keyed container's id comes from
its key alone, not from where it is. When a keyed container moves up among its siblings from one
run to the next (a one-time message or the approval stamp above it is gone), the new block and
the old, not yet cleared one are siblings with the same key for a moment; React then loses the
old block, and its DOM stays on the screen for good: step 5's buttons, greyed, with their
tooltips, still drawn over step 6. So every keyed block is the only child of an unkeyed one
(ui/keyed), whose React key is its position. The browser check: scripts/browser_stale_check.py.

Only synthetic data: invented names and texts, frames are tiny black PNGs."""
import ast
from pathlib import Path

from streamlit.testing.v1.element_tree import Block

from pfb_app.services import review
from test_wizard_ui_basic import app, env, ok, shown_step, video_assignment  # noqa: F401
from test_wizard_ui_review_pdf import reviewable, subject

UI = Path(__file__).resolve().parents[1] / "src" / "pfb_app" / "ui"
BLOCKS = {"container", "expander", "popover", "tabs"}


def _keyed_blocks(node, parent=None, found=None):
    """[(block, its parent)] for every block of the run that has an id (a key, or state)."""
    found = [] if found is None else found
    proto = getattr(node, "proto", None)
    if parent is not None and isinstance(node, Block) and getattr(proto, "id", ""):
        found.append((node, parent))
    for child in getattr(node, "children", {}).values():
        _keyed_blocks(child, node, found)
    return found


def assert_alone_in_an_unkeyed_block(at):
    keyed = _keyed_blocks(at._tree)
    assert keyed                                       # the step card has a key
    for block, parent in keyed:
        assert not getattr(parent.proto, "id", ""), block.proto.id
        assert [id(c) for c in parent.children.values()] == [id(block)], block.proto.id


def test_step_5s_keyed_blocks_each_sit_alone_in_an_unkeyed_one(env):
    settings, repo = env
    aid = video_assignment(repo)
    reviewable(settings, repo, aid, insufficient=("Pa", "Pb"))      # short texts on screen
    at = ok(app(session={"assignment_id": aid}))
    assert shown_step(at) == 5
    assert_alone_in_an_unkeyed_block(at)


def test_step_6s_keyed_blocks_each_sit_alone_in_an_unkeyed_one(env):
    settings, repo = env
    aid = video_assignment(repo)
    sid = reviewable(settings, repo, aid)
    for pid in ("Pa", "Pb", "Pc"):
        review.approve(repo, subject(repo, sid, pid)["id"])
    at = ok(app(session={"assignment_id": aid, "step": 6, "submission_id": None}))
    assert shown_step(at) == 6
    assert_alone_in_an_unkeyed_block(at)


def test_the_screens_make_keyed_blocks_only_through_ui_keyed():
    """No st.container / expander / popover / tabs with a key or a state of its own anywhere
    in the screens, except in ui/keyed itself."""
    bare = []
    for path in sorted(UI.rglob("*.py")):
        if path.name == "keyed.py" and path.parent == UI:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in BLOCKS
                    and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "keyed")
                    and {k.arg for k in node.keywords} & {"key", "on_change"}):
                bare.append(f"{path.relative_to(UI)}:{node.lineno}")
    assert bare == []
