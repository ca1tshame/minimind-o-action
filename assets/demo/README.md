# Demo assets

Real rollouts of the V14 student checkpoint, recorded with
`scripts/record_demo_gif.py` from the per-step screenshots that
`agent/environment.run_episode` saved while BrowserGym was actually running.

| File | Content |
|---|---|
| `scroll-text-2_seed20008.gif` | 2 steps, success, reward 1.0 — scroll then click |
| `scroll-text-2_seed20008.json` | GIF metadata (trace/GIF SHA-256, replay note) |
| `scroll-text-2_20008_trace.json` | full episode trace (action, tokens, reward per step) |
| `login-user_seed20001.gif` | 5 steps, success, reward 1.0 — multi-field form |
| `login-user_seed20001.json` | GIF metadata |
| `login-user_20001_trace.json` | full episode trace |

The recorder refuses to build a GIF without the real screenshots, and the
overlay only annotates the action string the policy emitted plus the reward the
environment returned — no coordinate or success flag is authored by hand.

The `scroll-text-2` pair also documents a fixed defect: at the release default
`max_new=24` the four-argument `scroll_at(cx,cy,dx,dy)` was truncated mid-token
and the episode failed; at the protocol budget of 48 the same checkpoint and
seed succeed in 2 steps.
