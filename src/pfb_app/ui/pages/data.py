"""データの扱い: where the data lives, 「動画を自動で消す」 and 「この人のデータを消す」
(ui/wizard/data_view)."""
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import data_view
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

data_view.render(get_repo(), get_settings())
