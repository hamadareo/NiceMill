"""はじめての方へ：使う流れ、見本の作業、撮り方のこつ、よくある質問、くわしい使い方。"""
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import guide_view
from pfb_app.ui.access import require_page

require_page()   # first: also when this file runs without ui/app.py's gate

guide_view.render(get_repo())
