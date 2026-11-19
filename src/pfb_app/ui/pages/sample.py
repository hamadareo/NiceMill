"""見本の作業：架空の選手とイラストで作った、見るだけの作業。"""
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import sample_view
from pfb_app.ui.access import require_page

require_page()   # first: also when this file runs without ui/app.py's gate

sample_view.render(get_repo())
