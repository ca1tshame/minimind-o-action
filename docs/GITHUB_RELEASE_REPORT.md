# MiniMind-O Action — GitHub release engineering report

Date: 2026-11-18. Target repository: `ca1tshame/minimind-o-action`.
Baseline audited by Carbon: `01a4007` (unchanged at the start of this work).

Status vocabulary used below: **DONE** (implemented here), **VERIFIED**
(executed here, with the command/result), **NOT RUN** (implemented but not
executed on this machine), **BLOCKED** (external input missing), **N/A**.

## 1. Carbon audit issues — disposition

| # | Issue | Status | Where |
|---|---|---|---|
| R1.1 | `FrozenSigLIP2.forward` Tensor vs `.last_hidden_state` | DONE | `scripts/inference.py`, `scripts/train_gui_sft.py` (encoder output used directly) |
| R1.2 | Projector key mismatch `proj.net.*` vs `proj.mlp.*` | DONE / VERIFIED | `models/vision.py:remap_action_keys`, unit test |
| R1.3 | compressor/tile_pos attached after checkpoint load | DONE | `build_predictor` / `build_model` attach first, then load |
| R1.4 | `strict=False` hides missing weights | DONE / VERIFIED | `load_action_model`: explicit missing/unexpected checks, `ActionWeightError` |
| R1.5 | click/scroll regex broken (unescaped parens) | DONE | `agent/environment.py` no longer re-parses with a second regex; single `agent/dsl.py` |
| R1.6 | stop returns `noop(0)` instead of terminating | DONE / VERIFIED | `run_episode`: `status="model_stop"`, env not stepped |
| R1.7 | history double-wrapping | DONE | history stores the parsed action string once |
| R1.8 | unquoted `@S:E`, `|lower`, `|upper` | DONE / VERIFIED | `agent/dsl.py` + `agent/copy_resolve.py` |
| R1.9 | `<think>` prefix rejected | DONE / VERIFIED | `extract_action` scans the last `<action>…</action>` |
| R1.10 | hover/invalid coordinates, NaN/Inf | DONE | range checks and `math.isfinite` |
| R1.11 | scroll semantics (V3 vs V12) | DONE | single V14 protocol: `scroll(dx,dy)`, negative dy = down, executor `scroll_at(cx,cy,-dx*W,-dy*H)` |
| R1.12 | eval never calls `configure_browser` | DONE | called in `eval/miniwob_eval.main` |
| R1.13 | `reward` vs `final_reward` mismatch | DONE / VERIFIED | trace writes both; metrics falls back |
| R1.14 | `hist_used` not sliced | DONE | `assemble_prompt(hist_used=…)` |
| R1.15 | last action error not fed back | DONE / VERIFIED | next prompt carries `Last action failed: …` |
| R1.16 | protocol `unseen/long` empty task list | DONE | protocols resolved from `configs/default.yaml` |
| R1.17 | sys.path set after imports | DONE | imports deferred / package entry points |
| R1.18 | 210 episodes passed off as 240 | DONE | historical = 15 tasks × 16 seeds = 240 |
| R1.19 | trainer missing `optimizer.zero_grad` | DONE / VERIFIED | once per optimizer step, honoured accumulation |
| R1.20 | gradient accumulation broken | DONE / VERIFIED | `test_trainer_cpu` accumulation test |
| R1.21 | `truncated_samples` missing | DONE | counted per epoch, stored in progress |
| R1.22 | validation loaded but unused | DONE | validation runs every `eval_interval` |
| R1.23 | random-init fallback | DONE | strict LM load + parameter count + explicit weight gates |
| R1.24 | 64 vs 320 vision tokens inconsistency | DONE | default 320; layouts table in config |
| R1.25 | V13 entropy double normalisation | DONE / VERIFIED | `rl.sequence_logprobs` returns per-token mean; caller no longer divides |
| R1.26 | hard-coded HPC/Windows paths, `.cuda()` | DONE | `MINIMIND_WORKSPACE` / `--workspace-root`, `resolve_device` |

## 2. Files changed / added

Rewritten: `agent/dsl.py`, `agent/copy_resolve.py`, `agent/ctx_pack.py`,
`agent/environment.py`, `agent/records.py`, `configs/__init__.py`,
`configs/default.yaml`, `models/vision.py`, `eval/miniwob_eval.py`,
`eval/metrics.py`, `scripts/inference.py`, `scripts/train_gui_sft.py`,
`requirements.txt`, `README.md`.

Added: `agent/vision_tiles.py` (existing, extended use), `models/grounding.py`,
`scripts/{filter_teacher,collect_oracle,build_distill_dataset,prepare_gui_sft_data}.py`,
`scripts/train_grpo.py`, `scripts/lib/rl.py`, `scripts/demo_gui_agent.py`,
`scripts/record_demo_gif.py`, `scripts/gui_v9_{grounding,ground_collect,mix,ground_eval}.py`,
`legacy/action_v1.py`, `data/README.md`, `data/schemas/record.schema.json`,
`configs/demo.yaml`, `tests/` (12 files, 109 tests), `LICENSE`, `NOTICE`,
`pyproject.toml`, `.gitattributes`, this report.

`agent/action.py` is now a deprecation shim over `legacy.action_v1`.

## 3. Model loading and SFT mainline

- Loader order is enforced: language backbone (strict, 63,912,192 params) →
  `GuiActionModel` → `attach_tile_compression` → action checkpoint (strict,
  `proj.net.*` remapped) → frozen encoder. Any missing tensor raises
  `ActionWeightError`; `strict=False` is never used for the action model.
- Verified: remap test, missing-compressor test, 320-marker injection test,
  parameter-count test (`tests/test_vision_load.py`).
- `scripts/train_gui_sft.py` now implements the V11/V14 recipe: 5-tile/320-token
  interface, `hist_used` cap 6, three weighted streams (0.75/0.15/0.10), action
  kind weights, action-level CE, one AdamW step per accumulation boundary,
  truncation accounting, validation, resumable checkpoints with contract hash.
- The default config is the published V14 recipe; `configs/demo.yaml` is the
  separate smoke config.

## 4. DSL / BrowserGym

A single parser (`agent/dsl.py`) feeds both inference and evaluation. Supported:
click, type (quoted literal or `@S:E[|lower|upper]`), scroll (V14 semantics),
stop, key. Invalid coordinates, NaN/Inf, bad pointers and unknown actions raise
`ActionParseError`, which the episode loop feeds back as `Last action failed`
without stepping the environment. V1 JSON is in `legacy/` and no longer used by
the agent line.

## 5. V9 grounding

`models/grounding.py` + 4 scripts port the V9 experiment: GroundingHead
(463,363 parameters), 320-cell 8×8×5 patch grid, focal classification +
offset regression, DOM-box collection, mix and eval. The evaluator reproduces
the stored numbers rather than inventing them:

```
n=80: v8_ctrl 10/80 | v9_dsl 14/80 | v9_head2 DSL 28/80 vs head 3/80
n=30: v8 8/30 | v9_dsl 8/30 | v9_head2 DSL 15/30 vs head 1/30
```

## 6. V12/V13 GRPO

`scripts/lib/rl.py` provides group-relative advantage, policy log-prob,
reference KL, rollout grouping, zero-variance diagnostics, masked entropy and
grad-norm logging; `scripts/train_grpo.py` is the runnable (BrowserGym) loop.
The V13 entropy bug is fixed at the source (no second division), and
`tests/test_rl.py` pins the numeric behaviour. Historical V13 entropy values
are not re-derived because the original logits are gone; the README reports the
bug and the V12 0.60–0.72 reference instead.

## 7. V14 data pipeline

`filter_teacher.py` → `collect_oracle.py` → `build_distill_dataset.py` →
`prepare_gui_sft_data.py` implement verified-trajectory → filter → convert →
dedup → balance → manifest. The V14 accounting is documented and enforced:
3,010 raw (2,259 V11 + 705 Oracle + 46 GUI-Owl) → 2,950 after blocked seeds →
2,915 unique after content-hash dedup → 2,783 after easy-task cap
(train 2,552 / val 231). Oracle and GUI-Owl are labelled separately. No raw
screenshots or trajectories are committed.

## 8. CPU tests

Command (VERIFIED):

```
D:/Anaconda/envs/minimind_action/python.exe -m unittest discover -s tests -t .
Ran 109 tests in ~1.5s — OK
```

Coverage: config resolution, model forward/parameter count, projector remap and
strict loading, 5-tile injection, token-aware packing and overflow, DSL and copy
pointers, episode state/reward/error feedback, action-level SFT spans, gradient
accumulation, checkpoint round-trip, GRPO advantage/entropy, distillation
filtering and split isolation, grounding head numerics, GIF recorder (real
screenshots only).

## 9. Real-checkpoint integration

**BLOCKED.** The release machine has no V14/V11 GUI checkpoint (only P1.5/P1.6
language checkpoints and the SigLIP2 weights), no BrowserGym/Playwright install,
and no reachable HPC (`10.120.21.247:22` timed out). Consequently:

- NOT RUN: loading a real V14 checkpoint, generating a real action, executing a
  BrowserGym episode, recording a demo GIF.
- NOT RUN: re-measuring 75/240. The README marks it as a historical result.

## 10. README metric corrections

- 2.34436 → 1.57849 is P1.5 Full SFT vs P1.6 Full SFT on the same frozen
  2048-token windows (n=53, 108,544 targets, paired Δ −0.766 [−0.858, −0.674]);
  not a base-model comparison.
- 35.0% vs 3.75% are the two output paths of the same `v9_head2` checkpoint.
- V14 75/240 is shown next to V11 64/240, V14 control 68/240, long 17/64,
  V12 seed-32 11/32 and V13 weak-48 8/48, with the long-horizon regression.
- GRPO is reported as a negative result; the entropy log artefact is explained.
- Parameter counts separate 63,912,192 (LM), 65,297,409 (action model) and
  94,552,320 (frozen encoder).

## 11. License / NOTICE

Apache-2.0 `LICENSE` at the root; `NOTICE` records the MiniMind/MiniMind-O
upstream, the modified-file list, SigLIP2 and BrowserGym/MiniWoB dependencies,
the CenterNet-inspired focal loss, and the fact that weights/data are not
redistributed. The project does not claim copyright over third-party packages.

## 12. Git

See the final section of the delivery message and `git log` for the commit
hash. Push is performed with a normal fast-forward `git push origin main`; no
force push, no history rewrite.

## 13. Remaining gaps

| Item | Status |
|---|---|
| Real V14 checkpoint integration / 240-episode re-run | BLOCKED (no weights, no browser, HPC unreachable) |
| Demo GIF | BLOCKED (depends on the above); recorder and trace format are ready |
| GUI SFT full 700-step re-run | NOT RUN (out of scope: no new long training authorised) |
| Agentic RL re-run | NOT RUN (out of scope) |
| Speech modality | not implemented in this release |
