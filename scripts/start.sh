#!/usr/bin/env bash
# Start the PFB app: tmux sessions pfb-ui (Streamlit) and pfb-worker. Existing sessions are kept.
# (With the systemd --user units installed and enabled, systemd keeps them running instead.)
# Settings: the built-in defaults, or the YAML named by PFB_CONFIG (passed to both).
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
mkdir -p inbox

WORKER_ARGS=""
if [ -n "${PFB_CONFIG:-}" ]; then
    WORKER_ARGS="--settings $(printf %q "$PFB_CONFIG")"
fi
# Models are already in the local Hugging Face cache: never contact huggingface.co.
# Set per command: a running tmux server does not pass this shell's environment on, and
# its own environment may carry another project's VIRTUAL_ENV.
ENV_PREFIX="env -u VIRTUAL_ENV HF_HUB_OFFLINE=1 "
if [ -n "${PFB_CONFIG:-}" ]; then
    ENV_PREFIX="${ENV_PREFIX}PFB_CONFIG=$(printf %q "$PFB_CONFIG") "
fi

# kept by systemd (scripts/install_user_services.sh --enable): never start a second copy,
# also when the units are enabled but stopped right now (systemd starts them again)
for u in pfb-ui.service pfb-worker.service; do
    if systemctl --user is-active --quiet "$u" 2>/dev/null \
            || systemctl --user is-enabled --quiet "$u" 2>/dev/null; then
        echo "systemd が管理しています（systemctl --user status pfb-ui pfb-worker）。tmux では起動しません"
        exit 0
    fi
done

# Where the screens listen (docs/https.md):
#   PFB_UI_HOST  default 127.0.0.1 (the Tailscale IP, http). 127.0.0.1 for HTTPS through
#                `tailscale serve` (scripts/https_serve.sh on), so nothing but the proxy reaches it.
#   PFB_UI_PORT  default 8501.
#   PFB_UI_ENTRY default serve (ui/serve.py: the screens plus the resumable upload endpoint);
#                app runs ui/app.py alone (Streamlit's own uploader only).
UI_HOST="${PFB_UI_HOST:-127.0.0.1}"
UI_PORT="${PFB_UI_PORT:-8501}"
if ! [[ "$UI_HOST" =~ ^[0-9A-Za-z.:-]+$ && "$UI_PORT" =~ ^[0-9]+$ ]]; then
    echo "PFB_UI_HOST / PFB_UI_PORT が正しくありません" >&2
    exit 2
fi
case "${PFB_UI_ENTRY:-serve}" in
    serve) UI_SCRIPT=src/pfb_app/ui/serve.py ;;
    app) UI_SCRIPT=src/pfb_app/ui/app.py ;;
    *) echo "PFB_UI_ENTRY は serve か app にしてください" >&2; exit 2 ;;
esac
if tmux has-session -t pfb-ui 2>/dev/null; then
    echo "pfb-ui はすでに起動しています"
else
    tmux new-session -d -s pfb-ui -c "$APP_DIR" \
        "${ENV_PREFIX}$UV run streamlit run $UI_SCRIPT --server.address $UI_HOST --server.port $UI_PORT --server.maxUploadSize 4096 --server.headless true"
    echo "pfb-ui を起動しました（http://$UI_HOST:$UI_PORT）"
fi

if tmux has-session -t pfb-worker 2>/dev/null; then
    echo "pfb-worker はすでに起動しています"
else
    tmux new-session -d -s pfb-worker -c "$APP_DIR" \
        "env -u VIRTUAL_ENV HF_HUB_OFFLINE=1 $UV run python -m pfb_app.worker.main $WORKER_ARGS"
    echo "pfb-worker を起動しました"
fi
