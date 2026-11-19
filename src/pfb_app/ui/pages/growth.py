"""成長の記録: one person across several works (ui/wizard/growth_view)."""
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import growth_view
from pfb_app.ui.access import require_page

require_page("edit")   # first: also when this file runs without ui/app.py's gate

growth_view.render(get_repo())
