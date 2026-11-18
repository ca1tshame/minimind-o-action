# MiniMind-O Action

MiniMind-O Action is a small **vision-language GUI agent trained from scratch**:
a 63.9M-parameter MiniMind language backbone, a frozen SigLIP2 image encoder,
a 5-tile multi-scale vision interface, and a single `<action>…</action>` DSL
that is executed in **BrowserGym / MiniWoB++**.

This repository is the cleaned-up, runnable release of the V1 → V14 research
line. It contains the model code, the training and evaluation entry points,
the teacher-distillation pipeline, the V9 grounding ablation, the V12/V13 GRPO
experiments, and the README-level corrections for every metric that the
earlier draft reported incorrectly.

> **Scope of this release.** Code, configs and protocols are complete and
> unit-tested on CPU. A real V14 checkpoint and a BrowserGym/Playwright install
> are **not** redistributed here (see [Weights and data](#weights-and-data)),
> so the closed-loop demo is documented and executable but was **not** re-run
> on the release machine. Numbers in [Results](#results) are the historical,
> verified measurements of the V1–V14 runs, not a new measurement.

---

## 1. Overview

The project asks a narrow question: *how much GUI competence can a small,
from-scratch language model reach when it is given a frozen, well-trained
vision encoder and a strictly verifiable action space?* Concretely:

- the language backbone is **not** a downloaded LLM — it is MiniMind Dense,
  pretrained from random initialisation by this project (P1.5), then extended
  to long context (P1.6) and language-SFT'd;
- the vision encoder **is** pretrained and **frozen** (SigLIP2 base, 94.6M);
- the interface between them is trained: a projector, a learned 5-tile
  compressor and per-tile positional embeddings;
- actions are not free text: every step must be a parseable
  `<action>click(x,y)</action>`, `<action>type("…")</action>`,
  `<action>type(@S:E|lower)</action>`, `<action>scroll(dx,dy)</action>` or
  `<action>stop()</action>`.

The agent is evaluated on MiniWoB++ through BrowserGym with a **fixed 240-episode
protocol** (15 tasks × 16 seeds, `hist_used=6`) so that V11/V12/V14 numbers are
directly comparable.

## 2. Architecture

```
 screenshot 1280×720
        │
        ▼
 5-tile split  (global + 2×2 crops, 256×256 each)
        │
        ▼
 Frozen SigLIP2-base-p32-256  ── 64 patches × 5 tiles = 320 vision tokens
        │
        ▼
 TileCompressor (per-tile keep-1 softmax merge)  +  tile_pos
        │
        ▼
 Projector 768→768  ──►  <|image_pad|> × 320 inside the prompt
        │
        ▼
 MiniMind Dense 63.9M  (8 layers, 768 hidden, GQA 8/4, RoPE θ=1e6)
        │
        ▼
 action DSL  <action>click(0.48,0.42)</action>
        │
        ▼
 BrowserGym / MiniWoB++  ──►  reward, terminated, last_action_error
        │
        └────────────► next prompt (token-aware history packing)
```

| Component | Parameters | Trained here? |
|---|---:|---|
| MiniMind language backbone (P1.5/P1.6/SFT) | 63,912,192 | **from scratch** |
| Projector (`proj.mlp`) | 1,182,720 | yes |
| TileCompressor | 198,657 | yes |
| `tile_pos` | 3,840 | yes |
| **Action model total** | **65,297,409** | — |
| Frozen SigLIP2 vision encoder | 94,552,320 | no (external, frozen) |
| V9 grounding head (ablation only) | 463,363 | yes |

So the whole multimodal system is **~160M parameters**, of which only the
63.9M language backbone and the ~1.4M interface are trained by this project.
Saying "a 65M vision-language model" is imprecise; the action model is 65.3M
**plus** a frozen 94.6M encoder.

Resolution of the vision interface is configurable — 320 tokens is the V11–V14
setting, `160/128/64` are supported legacy layouts (`configs/default.yaml:
vision.tile_layout`). At 320 tokens each tile keeps all 64 patches, so the
compressor is a no-op there by design (it matters at 160/128/64).

## 3. Training pipeline

| Stage | Script | What it does |
|---|---|---|
| P1.5 pretrain | `scripts/pretrain_p15.py` | 63.9M LM, 380-token windows, 3.61B target tokens, 65,504 updates |
| P1.6 long context | `scripts/pretrain_p16.py` | continues P1.5 to 2048 context, 175.4M tokens, 40,001 updates |
| Language SFT | `scripts/full_sft.py` | 768-token chat SFT, 46,667 updates, 321.3M supervised tokens |
| GUI SFT | `scripts/train_gui_sft.py` | V11/V14 recipe: 5 tiles, 320 vision tokens, action-level CE, three replay streams |
| Teacher distillation | `scripts/filter_teacher.py` → `build_distill_dataset.py` | GUI-Owl/V11/Oracle trajectories → verified, deduped, balanced manifest |
| Agentic RL | `scripts/train_grpo.py` | V12/V13 GRPO experiments (negative result, kept for the record) |
| Grounding ablation | `scripts/gui_v9_grounding.py` | V9 auxiliary patch-classification head |

The GUI SFT recipe (the default in `configs/default.yaml`) is the V14 one:

- context 512, `hist_used=6`, 320 vision tokens, 700 steps;
- per step: new GUI batch 4 (weight 0.75) + language replay 1 (0.15) + old GUI
  replay 2 (0.10);
- action-class weights `click 0.35, type 1.0, scroll 2.0, stop 2.0`;
- AdamW β=(0.9, 0.95), wd 0.01, lr 1e-6 (LM layer 7 + norm), 2e-5 (projector,
  compressor, `tile_pos`), 20-step warmup then cosine to 0.1×, clip 1.0;
- only `lm.model.layers.7.*`, `lm.model.norm.*`, `proj.*`, `compressor.*`,
  `tile_pos.*` are trainable.

`configs/demo.yaml` is a deliberately tiny smoke configuration; it is **not**
the published recipe.

## 4. Reproducing

### 4.1 Install

```bash
git clone https://github.com/ca1tshame/minimind-o-action.git
cd minimind-o-action
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch, transformers, Pillow, PyYAML
pip install -e ".[eval]"                 # BrowserGym/Playwright, optional
python -m playwright install chromium    # needs network
```

Verified versions: Python 3.10, torch 2.4.1, transformers 4.57.6,
safetensors 0.7.0, numpy 1.26.4, PyYAML 6.0.3, Pillow 9.5.0.

### 4.2 Workspace

Every path is resolved against a workspace root (`--workspace-root` or
`MINIMIND_WORKSPACE`); no absolute machine paths are baked into the code.

```
$MINIMIND_WORKSPACE/
├── tokenizer/                      # tokenizer.json (vocab 6400, bos/eos/pad 1/2/0)
├── siglip2-base-p32-256/           # config.json + model.safetensors
├── checkpoints/
│   ├── p1_6_sft/                   # language backbone after SFT
│   └── gui_v14/payload/model.safetensors
├── data/
│   ├── gui_v14_scaled_distill/train.jsonl, val.jsonl
│   └── frames/                     # screenshots referenced by the records
└── outputs/
```

Missing inputs fail loudly (`require_paths` / `ActionWeightError`) instead of
silently falling back to random weights.

### 4.3 CPU tests

```bash
python -m unittest discover -s tests -t .
# or: pytest -q
```

109 tests, no GPU, no network, no browser.

### 4.4 Inference

```bash
python scripts/inference.py \
  --workspace-root "$MINIMIND_WORKSPACE" \
  --checkpoint checkpoints/gui_v14/payload/model.safetensors \
  --task click-dialog --seed 20001 --max-steps 8 \
  --record outputs/demo/click-dialog_20001.json
```

The loader checks the projector keys (`proj.net.*` from the original
checkpoints is mapped to `proj.mlp.*`), requires the compressor/tile_pos tensors
to be present, and refuses to continue if any expected tensor is missing.
`--dump-load-report` prints the mapping and parameter counts.

### 4.5 BrowserGym evaluation (240-episode protocol)

```bash
python eval/miniwob_eval.py \
  --workspace-root "$MINIMIND_WORKSPACE" \
  --checkpoint checkpoints/gui_v14/payload/model.safetensors \
  --protocol historical --out outputs/eval/v14_hist240
```

`--protocol historical` is exactly 15 tasks × seeds 20001–20016 = 240
episodes, `max_steps=8`, `max_new=48`, `hist_used=6`. Subsets:
`seen`, `related_unseen`, `distinct_unseen`, `unseen` (32 tasks at seed
15101), `long` (8 tasks × 8 seeds, `max_steps=10`). Episodes are JSONL with
both `reward` and `final_reward`; `eval/metrics.py` aggregates them.

`MINIWOB_URL` must point at a local MiniWoB++ `index.html`; set
`PLAYWRIGHT_BROWSERS_PATH` if Chromium is not in the default cache.

### 4.6 GUI SFT

```bash
python scripts/train_gui_sft.py \
  --workspace-root "$MINIMIND_WORKSPACE" \
  --config configs/default.yaml \
  --init-checkpoint checkpoints/v11_teacher_distill/payload/model.safetensors \
  --train-data data/gui_v14_scaled_distill/train.jsonl \
  --val-data data/gui_v14_scaled_distill/val.jsonl \
  --frames-dir data/frames --steps 700 \
  --experiment-dir outputs/experiments/v14
```

Gradient accumulation is honoured (one `zero_grad` per optimizer step, not per
micro-batch), `truncated_samples` is counted, validation actually runs, and the
checkpoint contract records the accumulation boundary.

### 4.7 Distillation data

```bash
python scripts/filter_teacher.py --workspace-root "$MINIMIND_WORKSPACE" \
    --episodes-dir raw/teacher_episodes --frames-root raw/frames \
    --out data/teacher_filtered.jsonl
python scripts/collect_oracle.py --workspace-root "$MINIMIND_WORKSPACE" \
    --out data/oracle_v14.jsonl --frames-root data/frames
python scripts/build_distill_dataset.py --workspace-root "$MINIMIND_WORKSPACE" \
    --v11-teacher data/v11_teacher.jsonl --v14-oracle data/oracle_v14.jsonl \
    --gui-owl data/gui_owl.jsonl --frames-root data/frames \
    --out data/gui_v14_scaled_distill
python scripts/prepare_gui_sft_data.py --workspace-root "$MINIMIND_WORKSPACE" \
    --records data/gui_v14_scaled_distill/train.jsonl --out data/final
```

Records keep provenance (`data_source` ∈ {`gui_owl_teacher`, `v11_teacher`,
`oracle`, `old_gui`, `language_replay`}), episode/task/seed ids, screenshot
hash, DSL target and reward; the frozen test seeds are blocked from training by
config and by the builder.

### 4.8 Agentic RL (experiments)

```bash
python scripts/train_grpo.py --workspace-root "$MINIMIND_WORKSPACE" \
    --checkpoint checkpoints/gui_v14/payload/model.safetensors \
    --steps 40 --group-size 4 --rollouts-per-step 12
```

This is the V12-style loop (real BrowserGym rollouts, group-relative
advantage, KL to the frozen reference, resumable checkpoints). It is a research
entry point, not a recipe that improved the model — see Results.

### 4.9 V9 grounding ablation

```bash
python scripts/gui_v9_grounding.py --forward-check
python scripts/gui_v9_ground_collect.py --out data/v9_ground.jsonl
python scripts/gui_v9_mix.py --base data/v9_lang.jsonl \
    --ground data/v9_ground.jsonl --ratio 0.1 --out data/v9_mix
python scripts/gui_v9_ground_eval.py --result-json evals/v9_80.json
```

## 5. Results

All GUI numbers use MiniWoB++, BrowserGym, the 240-episode protocol unless a
column says otherwise. "V14 scaled" is the model trained on the 2,783-row
balanced distillation set; "V14 control" is the same trainer on the 2,259-row
V11 set. These are **historical verified runs**, not re-measured on this
release machine.

### 5.1 Language backbone

| Stage | Metric | Value |
|---|---|---|
| P1.5 pretrain (65,504 updates, 3.61B tokens) | validation NLL / PPL | 1.6539 / 5.2271 |
| P1.6 long-context (40,001 updates, 175.4M tokens) | validation NLL @512 / 1024 / 2048 | 1.7162 / 1.3587 / 1.1970 |
| P1.5 Full SFT | final validation NLL / PPL | 1.23924 / 3.45299 |
| P1.6 Full SFT | final validation NLL / PPL | 1.23841 / 3.45011 |

**P1.5 → P1.6 long-context NLL (the "2.344 → 1.578" number).** This is *P1.5
Full SFT vs P1.6 Full SFT*, evaluated on the same frozen 2048-token real-text
windows (n=53 documents, 108,544 target tokens, no extra EOS):
**2.34436 → 1.57849**, paired document-mean difference **−0.766**
(95% CI [−0.858, −0.674]). It is **not** a base-vs-SFT comparison and not a
random-sample estimate; the 768 bucket is a defined prefix experiment
(2.46373 → 1.63629), the 512 bucket shows no meaningful difference
(1.77938 → 1.77791).

### 5.2 GUI agent (240-episode protocol, `hist_used=6`)

| Model | Episodes | Success |
|---|---:|---:|
| V11 teacher-distilled SFT | 240 | 64/240 = 26.67% |
| V12 after GRPO (480 on-policy episodes) | 240 | 64/240 = 26.67% |
| V14 control (V11 data, continued SFT) | 240 | 68/240 = 28.33% |
| **V14 scaled (2,552 train rows)** | 240 | **75/240 = 31.25%** |

Secondary slices (same models, different protocols — do not compare across
rows):

| Model | Protocol | Success |
|---|---|---:|
| V11/V12 | independent 32 tasks | 15/32 |
| V11/V12 | long 64 (8 tasks × 8 seeds) | 19/64 |
| V11/V12 | unseen weak set 48 | 8/48 |
| V14 scaled | long 64 | 17/64 |
| V12 new seed 32 | independent 32 | 11/32 |
| V13 weak set 48 | unseen weak 48 | 8/48 |

The V14 gain is real but modest (+7 episodes over V11 on 240), and it comes
with a **long-horizon regression** (19/64 → 17/64). Both are reported.

### 5.3 V9 grounding ablation (new seeds, n=80)

Same checkpoint, two output paths:

| Variant | DSL output | Grounding-head output |
|---|---:|---:|
| V8 control (no grounding data) | 10/80 = 12.5% | — |
| V9 DSL-only (text mix) | 14/80 = 17.5% | — |
| **V9 `head2` (one model)** | **28/80 = 35.0%** | **3/80 = 3.75%** |

On the 30-sample slice: V8 8/30, V9 DSL 8/30, `v9_head2` DSL 15/30 (50.0%) and
head 1/30 (3.33%). The 35.0% and 3.75% are **two decoding paths of the same
model**, not two trained models; the head path is a negative result for
point-prediction-as-action and must not be read as "all visual grounding heads
fail".

### 5.4 V12/V13 GRPO: a negative result

- **V12**: 480 real on-policy episodes, 40 updates. Final 240-episode success
  is **identical to V11 (64/240)** — no measurable gain.
- **V13**: two continuation rounds; the high-LR round completed 40 updates /
  480 episodes. On a fresh 32-task seed it went **15/32 → 13/32** (worse).
- 50–75% of rollout groups had **zero reward variance** (no learning signal),
  KL stayed ≈0, and gradient norm (27–48) sat against the clip of 0.3.
- **Logged entropy bug.** V13 divided the already per-action-token mean entropy
  by the action length a second time, so its `0.026–0.068` log is an artefact.
  V12's single-normalisation value is **0.60–0.72**. The raw V13 logits are no
  longer available, so no "corrected" V13 entropy is reported — only the
  definition is fixed in `scripts/lib/rl.py` and unit-tested.
- PRM / GUI-transition-verifier were **not** implemented; nothing in this
  repository claims they were.

## 6. Demo

A real rollout looks like this (command, not a mock):

```bash
python scripts/demo_gui_agent.py --workspace-root "$MINIMIND_WORKSPACE" \
  --checkpoint checkpoints/gui_v14/payload/model.safetensors \
  --task enter-text-2 --seed 20001 \
  --output outputs/demo/enter-text-2_20001.json
python scripts/record_demo_gif.py \
  --trace outputs/demo/enter-text-2_20001.json \
  --frames outputs/demo/enter-text-2_20001_frames \
  --out assets/demo/enter-text-2_20001.gif
```

`demo_gui_agent.py` executes BrowserGym and saves the trace plus one real
screenshot per step; `record_demo_gif.py` only overlays the step number, sampled
action, reward and environment error on **those** screenshots, and writes
`trace_sha256` / `gif_sha256` / `replay_note` next to the GIF.

**Status: produced, from the real V14 checkpoint.** On 2026-11-18 the V14
action model was loaded with the code in this repository (65,297,409
parameters, `proj.net.*` remapped, compressor/tile_pos attached before load)
and run against BrowserGym/MiniWoB++ on an A40. Two episodes completed
successfully; both GIFs are in `assets/demo/` with their traces:

| Demo | Steps | Result | What it shows |
|---|---:|---|---|
| `scroll-text-2_seed20008` | 2 | **success, reward 1.0** | `scroll_at(...)` then `click` on the revealed button |
| `login-user_seed20001` | 5 | **success, reward 1.0** | click → type → click → type → submit (multi-field form) |

The scroll demo is worth reading closely: with the release-time default of
`max_new=24` the model's `scroll_at(cx,cy,dx,dy)` output was cut to 24 tokens,
so **every scroll step became a parse error and the episode failed**. Raising
the budget to the protocol's 48 and accepting the four-argument spelling turned
the same checkpoint, same seed, into a 2-step success. That is a code defect
being fixed, not a model being retrained — see `docs/GITHUB_RELEASE_REPORT.md`
§9 for the full list of blockers found while producing these runs.

Frames are the environment's own screenshots; the overlay only annotates the
action text the policy emitted and the reward the environment returned. No
coordinate, step or success flag was authored by hand.

## 7. Limitations

- Many GUI tasks still fail (≈69% of the 240-episode suite at V14).
- Long-horizon behaviour is the weak spot: V14 regressed from 19/64 to 17/64
  on the long slice.
- Agentic RL (V12/V13) produced no stable gain; the negative result is kept.
- The vision encoder is external and frozen; the "from scratch" claim applies
  to the language backbone only.
- Speech/audio modality is not integrated in this release.
- The 240-episode numbers are historical; re-running them requires the V14
  checkpoint and a working BrowserGym environment.

## 8. License and acknowledgments

- Project code: **Apache-2.0** (see [LICENSE](LICENSE)).
- MiniMind and MiniMind-O upstream code: Apache-2.0; modified files are listed
  in [NOTICE](NOTICE).
- SigLIP2 (`google/siglip2-base-p32-256-ve`): external pretrained weights, not
  redistributed here; check its model card before use.
- BrowserGym / MiniWoB++ / Gymnasium: their own licenses; installed as optional
  evaluation dependencies.
- The V9 grounding head follows the CenterNet-style focal-loss idea (MIT); the
  implementation here is original.
- Training data, screenshots, teacher (GUI-Owl) weights and all V1–V14
  checkpoints are **not** redistributed. See `data/README.md` for the schema,
  provenance fields and regeneration commands.

## Repository layout

```
agent/       DSL, token-aware prompt packing, BrowserGym loop, records, 5-tile vision
models/      MiniMind backbone, FrozenSigLIP2, projector, grounding head
scripts/     pretrain / SFT / GUI SFT / distillation / GRPO / inference / demo
  lib/       checkpoint transaction, language SFT data, RL primitives
eval/        BrowserGym protocol runner and metrics
configs/     default.yaml (V14 recipe), demo.yaml, model_config.json
data/        record schema and pipeline documentation (no raw data)
legacy/      V1 JSON action prototype (not used by the agent)
tests/       109 CPU unittest/pytest tests
docs/        release report and experiment notes
```
