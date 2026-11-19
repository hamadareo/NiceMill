"""この解析の確かさ：このサーバーでの実績と、確かめたこと・まだ確かめていないこと。"""
from pfb_app.ui.common import get_repo
from pfb_app.ui.wizard import quality_view
from pfb_app.ui.access import require_page

require_page()   # first: also when this file runs without ui/app.py's gate

quality_view.render(get_repo())
