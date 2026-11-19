"""試合・活動のまとめ: one video group's numbers and pictures (ui/wizard/dashboard)."""
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import dashboard
from pfb_app.ui.access import require_page

require_page("edit")   # first: also when this file runs without ui/app.py's gate

dashboard.render(get_repo())
