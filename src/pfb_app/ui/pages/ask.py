"""ききたいことを聞く: ask a work's records a question (ui/wizard/ask_view)."""
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import ask_view
from pfb_app.ui.access import require_page

require_page("edit")   # first: also when this file runs without ui/app.py's gate

ask_view.render(get_repo())
