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

## V18 OpenApps Pure-Student Closed-Loop Demos (`v18_iter0/final_00001200`, `1920×1080`)

| File | Content |
|---|---|
| `v18_demo1_composite_19step.gif` | **19 steps, success (48.8s)** — cross-app Calendar + Todo + Messages autonomous execution (`charlie_birthday_surprise`) |
| `v18_demo2_frozen_msg_formal.gif` | **5 steps, success (15.9s)** — **ONE-SHOT Frozen-20** unseen formal-phrasing Messages execution (`v18_fz_msg_bob_formal`, where V17 failed) |
| `v18_demo3_ood_unrelated_ctx.gif` | **5 steps, success (15.0s)** — **OOD-Dev 24** unrelated-context distractor robustness (`message_bob_to_meet__unrelated_context_1`, where V17 failed) |

All three V18 GIFs are rendered strictly from real OpenApps BrowserGym screenshots saved in `rollouts.jsonl`, with the overlay annotating only the student's raw `<action>...</action>` output, predicted crosshair coordinate, and environment terminal reward.
