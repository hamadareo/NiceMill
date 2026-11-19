"""利用する人: who may use the app, their roles, invitations and reset codes (管理者 only;
ui/wizard/users_view)."""
from pfb_app.ui.common import get_repo, get_settings
from pfb_app.ui.wizard import users_view
from pfb_app.ui.access import require_page

require_page("admin")   # first: also when this file runs without ui/app.py's gate

users_view.render(get_repo(), get_settings())
