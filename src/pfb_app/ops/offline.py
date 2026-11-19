"""Nothing the app runs reaches outside this machine on its own: the switches that keep the
model libraries offline and their usage reports off. They are set for the worker process
(and so for its spawned sessions) at start, and passed to every engine subprocess
(engine.pfb3x3: `_env`), whatever the environment said before.

- HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE: models load from local files only (no download,
  no update check); HF_HUB_DISABLE_TELEMETRY: no hub telemetry.
- VLLM_NO_USAGE_STATS / DO_NOT_TRACK: vLLM sends no usage report.
- YOLO_OFFLINE: ultralytics (the person detector of the tracking engine) counts as offline,
  so it sends no usage events and checks no updates (ultralytics 8.4: utils.is_online returns
  False, and utils.events.Events is enabled only when online).
"""
import os

OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "HF_HUB_DISABLE_TELEMETRY": "1", "VLLM_NO_USAGE_STATS": "1",
               "DO_NOT_TRACK": "1", "YOLO_OFFLINE": "1"}


def apply(environ=os.environ) -> None:
    """Set the switches in this process's environment (children inherit them)."""
    environ.update(OFFLINE_ENV)
