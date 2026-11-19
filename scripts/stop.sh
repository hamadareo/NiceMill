#!/usr/bin/env bash
# Stop the PFB app: end only the tmux sessions pfb-ui and pfb-worker.
# Ending pfb-worker sends SIGHUP to the worker and its inference session; both leave through
# their cleanup (pfb_app.worker.session.exit_on_signals), which returns the VRAM and stops the
# PFB3x3 engine's command and vLLM server (they run in their own process groups). Anything a
# hard kill still leaves behind is stopped by the next worker at start-up (identity-checked).
# Before restarting after a deploy, check: no job running / waiting_gpu, :8200 free and no
# <data_dir>/pfb3x3_runtime/serve.pid (docs/pfb3x3_engine.md).
set -uo pipefail
for s in pfb-ui pfb-worker; do
    if tmux has-session -t "$s" 2>/dev/null; then
        tmux kill-session -t "$s"
        echo "$s を終了しました"
    else
        echo "$s は起動していません"
    fi
done
for u in pfb-ui.service pfb-worker.service; do
    if systemctl --user is-active --quiet "$u" 2>/dev/null \
            || systemctl --user is-enabled --quiet "$u" 2>/dev/null; then
        echo "systemd が管理しています。止めるには: systemctl --user stop pfb-ui pfb-worker"
        echo "（次の起動でも始めないようにするには: systemctl --user disable pfb-ui pfb-worker）"
        break
    fi
done
