# Demo assets

This directory is where `scripts/record_demo_gif.py` writes a **real** rollout
GIF plus its `.json` metadata (trace/GIF SHA-256, success, final reward).

It is intentionally empty in the source release: the V14 checkpoint and a
BrowserGym/Playwright installation were not available on the release machine,
so no episode could be recorded. The recorder refuses to build a GIF without
the per-step screenshots captured by `agent/environment.run_episode`, so no
placeholder or synthetic demo can be produced by accident.
