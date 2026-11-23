# MiniMind-O Action — V15–V18 Generalization Sprint Report

[← Back to README](../README.md) · Date: 2026-11-23 · Target repo: `ca1tshame/minimind-o-action`.
Model constraint unchanged: from-scratch **63,912,192**-parameter MiniMind Dense backbone + frozen **94.6M** SigLIP2 (`siglip2-base-p32-256`) + **~1.4M** multimodal interface (`~65.3M` trainable action model).

---

## 1. Architecture & Protocol Upgrades (V15 → V18)

1. **V16 6-Tile Focus Crop (`vt=384, max_ctx=768, hist_cap=8`)**:
   - Added a deterministic native `480×360` 6th focus tile (`agent/vision_tiles.py: LAYOUT[384], TILE_NAMES_FOCUS`) alongside the global thumbnail and 2×2 quadrant tiles, avoiding 1920→256 downsampling distortion on centered desktop forms.
   - Controlled V16 paired ablation (`vt=320` vs `vt=384`, 2,400 steps): `Input Focus 85.7% → 100.0%`, `Repeated Click 24.4% → 0.0%`, unlocking 19-step 3-app composite `charlie_birthday_surprise`.
2. **V17 Balanced DAgger (`--balance-dagger`)**:
   - Corrected 16:1 task-family skew in DAgger traces without extra data collection, raising OpenApps Historical Dev (12 tasks) from `5/12 (41.7%)` to `8/12 (66.7%)`.
   - Exposed 3 generalization gaps on the first strict frozen set (`2/24 = 8.33%`) and unseen non-16:9 viewports (`0%`): phrasing sensitivity, off-screen bottom Todo scroll supervision, and cross-viewport shift.
3. **V18 Canonical `Screen: {W}x{H}` Protocol & Dock-Aware Scroll**:
   - `agent/ctx_pack.py`: added `resolve_screen_wh`, `format_screen_tag`, and optional `screen_wh` in `build_user_prompt` / `assemble_prompt` (`Screen: {W}x{H}` extracted from actual screenshot `(width, height)`, 0% truncation across 11,663 multi-resolution samples; backwards-compatible default `screen_wh=None`).
   - Added `--layout-per-batch 1` with bucketed multi-resolution sampling (`W ∈ [860, 1680], H ∈ [700, 1050]`, excluding test viewports `1100×820` and `800×900`) and fixed bottom floating Dock obstruction (`dock_safe_bottom`) before re-measuring off-screen bounding boxes.

---

## 2. Strictly Disjoint Splits (`configs/v18_splits.json`)

Verified zero subtask-signature overlap (`tests/test_v18_protocol.py` + automated split audit):
- **`train_gen` (105 tasks)**: standard/casual/formal/unrelated-context variations + off-screen Todo delete/mark items.
- **`historical_dev_12` (12 tasks, `1920×1080`)**: historical regression benchmark.
- **`diagnostic_dev_15` (`1100×820`)**: unseen within-range interpolation viewport.
- **`ood_dev_24` (24 tasks, `1920×1080`)**: OOD Dev used exclusively for checkpoint selection.
- **`frozen_20` (20 tasks, ONE-SHOT confirmed + 12-task `800×900` extrapolation subset)**: **0 executions** during all Dev tuning and checkpoint selection (`v18_iter0 / final_00001200` selected strictly on Dev before running Frozen-20).

---

## 3. Experimental Findings & Paired V17 vs V18 Results

### 3.1 Dev Selection & Negative OOD-DAgger Channel Finding

| Candidate | Historical Dev (12t) | OOD-Dev (24t) | `1100×820` Dev (15t) | Notes |
|---|---:|---:|---:|---|
| `v18_iter0 / step_600` | `2/12` | `4/24` | `0/15` | Underfit |
| `v18_iter0 / step_900` | **`6/12`** | `3/24` | `0/15` | Runner-up |
| **`v18_iter0 / step_1200` (Selected)** | `5/12 (41.7%)` | **`5/24 (20.8%)`** | `0/15` (Dock `15/15`) | Highest Dev sum (`10/51`) & OOD-Dev |
| `v18_iter1` (unfixed sampler) | `2/12` | `1/24` | `0/15` | `540` RV DAgger rows crowded out `1920` DAgger (`oa_1920` draws `-30.5%`), causing `1200`-step collapse |
| `v18_iter1_bal` (fixed `dag_1920` vs `dag_var`) | **`6/12`** | `2/24` | `0/15` | Restored Hist12, but multi-intervention RV traces (`mark_todo=0`) diluted OOD-Dev |

### 3.2 Paired V17 (`v17_b_balanced`) vs V18 (`v18_iter0/final_00001200`) Confirmation

| Benchmark | V17 (`--no-screen-tag`) | V18 (`Screen: {W}x{H}`) | Paired Delta / Notes |
|---|---:|---:|---|
| `OOD-Dev 24t` (`1920×1080`) | `2/24` (`8.33%`, CI `[2.32%, 25.85%]`) | **`5/24` (`20.83%`, CI `[9.24%, 40.47%]`)** | **`+3 / -0`** net wins (`2.5×`, McNemar `p=0.25`) |
| `Interp 1100×820` (`15t`) | `0/15` (Step-0 Dock `9/15 = 60.0%`) | `0/15` (**Step-0 Dock `15/15 = 100.0%`**) | Step-0 shifted Dock grounded cleanly; multi-step inner control still fails |
| `Extrap 800×900` (`12t`) | `0/12` (Dock `6/12 = 50.0%`) | `0/12` (Dock `4/12 = 33.3%`) | Width `800` lies outside training `[860, 1680]`; no extrapolation transfer |
| **ONE-SHOT `Frozen-20`** (`1920×1080`) | `2/20` (`10.00%`, CI `[2.79%, 30.10%]`) | **`3/20` (`15.00%`, CI `[5.24%, 36.04%]`)** | `+1` (`+3/-2` disjoint tasks, McNemar `p=1.0`, wide small-N CI) |
| **MiniWoB `Indep32`** (`768/384/8`) | `23/32` (`71.88%`) | **`23/32` (`71.88%`)** | Full preservation (`0` overflows) |
| **MiniWoB `Weak48`** (`768/384/8`) | `15/48` (`31.25%`) | `13/48` (`27.08%`) | Slight regression (`-2`, `0` overflows) |
| **MiniWoB `Long64`** (`768/384/8`) | `23/64` (`35.94%`) | **`23/64` (`35.94%`)** | Exact match to V17 (`0` overflows) |
| **MiniWoB `Hist240`** (`768/384/8`) | `98/240` (`40.83%`) | `92/240` (`38.33%`) | Mild trade-off (`-6` / `-2.50%`, `0` overflows) |

### 3.3 `W×H` Tag Causal Probe, Language Check & Runtime Efficiency

- **`W×H` Diagnostic Ablation (`800×900` & `1100×820`, Step-0)**: Spoofing `Screen: 1920x1080` vs true `(W, H)` changes Step-0 predictions (`800×900 OpenTodos`: `0.3690` vs `0.2388`; `1100×820 OpenMessages`: `0.2389` vs `0.1920`), confirming V18 conditions on the tag alongside visual geometry.
- **Language Degradation Probe (10 prompts)**: `first_token_eos = 0/10`, `empty_output = 0/10`; `5/10` answered in natural language while `5/10` pure-text prompts triggered GUI `<action>click(...)</action>` bias after intensive multimodal SFT, and `"Is 17 prime?"` answered `"No"` (illustrating real 63.9M instruction/capacity limits).
- **Real-Checkpoint & Unified Hardware Verification (`v18_unified_bench_and_gh_verify.json`)**:
  - Verified this GitHub repo (`scripts/inference.py:build_predictor` + `models/vision.py`) loads **both** real `v17_b_balanced` and `v18_iter0` `vt=384` checkpoints (`65,298,177` action params, `0` missing/unexpected keys) and fixed two real inference bugs during verification (`load_lm_checkpoint` tied-embedding safetensors check + CPU strided GQA `flash_attn` NaN).
  - **A40 GPU (Public repo KV-Cache + `bf16`, isolated single-model process)**: V17 **`517.59 ms/step`** (`343.3 MB` alloc / `370.0 MB` rsv); V18 **`524.04 ms/step`** (`343.3 MB` alloc / `372.0 MB` rsv). *(Non-KV-Cache eval script: `702.76–712.57 ms/step`, `647.2 MB` alloc; an earlier V17 `1,877.2 MB` figure was traced to an uncollected `MethodType` reference cycle stacking 3 model copies in one loop, not a 65% VRAM reduction.)*
  - **8-Thread CPU (`gpu3-1` Xeon 8358P, KV-Cache `fp32`)**: V17 `27,579.4 ms/step`, V18 `27,373.9 ms/step` with bit-identical `<action>` outputs to GPU (`~73–75.8 s/step` without KV-cache). We explicitly avoid claiming real-time CPU deployment.

---

## 4. Pure-Student Demo GIFs (`assets/demo/`)

- `v18_demo1_composite_19step.gif`: `charlie_birthday_surprise` (**19 steps, 48.8s, reward 1.0**) — cross-app Calendar + Todo + Messages execution preserved.
- `v18_demo2_frozen_msg_formal.gif`: `v18_fz_msg_bob_formal` (**5 steps, 15.9s, reward 1.0**) — ONE-SHOT Frozen-20 formal phrasing pass where V17 failed.
- `v18_demo3_ood_unrelated_ctx.gif`: `message_bob_to_meet__unrelated_context_1` (**5 steps, 15.0s, reward 1.0**) — OOD-Dev unrelated-context robustness pass where V17 failed.
