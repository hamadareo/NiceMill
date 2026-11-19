"""The app's server entry: the screens (ui/app.py) plus the resumable upload endpoint
(ui/upload_api.py) on the same server and origin.

    streamlit run src/pfb_app/ui/serve.py --server.address 127.0.0.1 --server.port 8501 ...

Streamlit finds `app` (st.App) and serves it with uvicorn; every `--server.*` option and
.streamlit/config.toml apply as with `streamlit run src/pfb_app/ui/app.py`, which still works
(then without the endpoint: the screens fall back to Streamlit's own uploader).
"""
import streamlit as st

from pfb_app.ui import access, auth_api, upload_api

access.no_pages_folder_mode()   # every run goes through app.py's gate (ui/access.require_page)
app = st.App("app.py", routes=upload_api.mount() + auth_api.mount())
