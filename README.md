# MiniMind-O Action

A small-scale vision-language GUI agent built for reproducible research on real-web
closed-loop control. We train a 63.9M-parameter Dense language backbone from random
initialization, attach a frozen SigLIP2 vision encoder through a trainable projector,
and produce constrained GUI actions that execute multi-step tasks in the BrowserGym /
MiniWoB++ sandbox. Every reported success is the environment's terminal reward, never
a model self-report.

## 1. Positioning and open-source boundary

- **From-scratch backbone.** The MiniMind-O Dense architecture (63,912,192 parameters,
  vocabulary 6400) is randomly initialized and pretrained on 3.6B tokens. No pretrained
  language weights are downloaded.
- **Disclosed external component.** SigLIP2 is used strictly as a frozen visual
  encoder; it is not trained and is never written to checkpoints. Only the cross-modal
  projector is newly trained.
- **Real sandbox evaluation.** The agent perceives screenshots and emits a constrained
  action DSL (click / type / scroll / key / stop). Success is measured by MiniWoB++ task
  completion (environment reward = 1), not by answer plausibility.
- **Scope of release.** This repository publishes code and configuration only. Owing
  to upstream corpus licensing, the raw training datasets and pretrained weights are
  **not** included. See Section 7.

## 2. Core contributions

### 2.1 From-scratch pretraining and long-context extension

- **3.6B-token pretraining** of the Dense backbone (sequence length 380), reaching
  validation NLL 1.6539 / PPL 5.2271.
- **2048-token context extension** followed by Full SFT on a frozen 746,672-row subset.
  On a unified long-text window, validation NLL at 2048 tokens drops from 2.344
  (base) to 1.578 (long-context model).
- Data integrity, resumable cursors, and token-accounting conservation are enforced
  at every optimizer boundary.

### 2.2 Constrained DSL action space and visual-language alignment

- A fail-closed action DSL restricts the model to a small, typed action vocabulary.
  Malformed, out-of-range, or unsupported actions are rejected rather than executed.
- A trainable **vision projector** maps frozen SigLIP2 patch features into the language
  model's embedding space by replacing dedicated image-marker tokens, yielding a single
  unified token stream.
- **Multi-scale visual compression** (global + 2x2 local tiles) preserves spatial
  grounding while keeping the token budget fixed.

### 2.3 Token-aware history compression (ctx_pack)

- The `ctx_pack` module keeps the instruction, the last error, and the most recent K
  actions verbatim, then degrades older actions to kind-only stubs under a strict token
  budget with reserved output space.
- Under the unified `hist_used=6` protocol this reduced long-horizon context overflow
  from **42.2% (27/64) to 0% (0/64)**.

## 3. Key experimental results (unified hist_used=6)

All closed-loop numbers below use the same history setting (`hist_used=6`). Results
from different history settings are not directly comparable.

### 3.1 Language backbone

| Setting | Validation NLL | PPL |
|---|---:|---:|
| P1.5 pretraining (seq_len=380) | 1.6539 | 5.2271 |
| Base model, 2048-token window | 2.344 | 10.427 |
| P1.6 long-context Full SFT, 2048-token window | 1.578 | 4.848 |

### 3.2 Real-web closed-loop success (MiniWoB++)

| Protocol | Result | Notes |
|---|---:|---|
| Independent unseen-seed set (seed 15101) | 46.9% (15/32) | zero-shot baseline 0/18 |
| Long-horizon 20-step tasks | 29.7% (19/64) | 8 tasks x 8 seeds |
| Historical combined task set | 26.7% (64/240) | unified hist_used=6 |

## 4. Key ablations and engineering findings

- **Data-centric gains beat add-on structure.** 160 text-coordinate samples with real
  DOM bounding boxes reached a held-out grounding hit rate of **35.0%**, substantially
  better than a bolt-on CenterNet detection head at **3.7%**. Labeled data routed
  through the text coordinate space was the more effective investment.
- **On-policy RL (GRPO) boundary diagnosis.** After SFT convergence the policy entropy
  is very low (~0.03), so group sampling collapses toward argmax, intra-group variance
  approaches zero, and the advantage signal disappears. This indicates that broader
  task coverage and higher exploration entropy are prerequisites before applying RL.
  Under the unified protocol, the RL stage provided no net success gain over the
  distilled model.

## 5. Repository structure

```
configs/    default.yaml and model_config.json: model, path, and eval hyperparameters
models/     MiniMind Dense backbone, frozen SigLIP2 wrapper, vision projector, GUI model
agent/      constrained action DSL, ctx_pack history compression, BrowserGym loop
eval/       MiniWoB++ closed-loop runner and metric aggregation
scripts/    pretraining, Full SFT, GUI SFT, distillation, inference, shared lib/
assets/     placeholder for architecture diagrams and demo GIFs
```

## 6. Installation and quick evaluation

```bash
# 1. Create environment
conda create -n minimind-o-action python=3.10 -y
conda activate minimind-o-action
pip install -r requirements.txt

# 2. Point configuration at your local assets (no paths are baked into source)
#    Edit configs/default.yaml, or override on the command line.
#    Required: paths.* directories, tokenizer_dir, vision_encoder_dir, checkpoint,
#              miniwob_url, playwright_browsers_path

# 3. Install BrowserGym's local sandbox assets (MiniWoB++ html + Playwright browsers)
#    and set paths.miniwob_url / paths.playwright_browsers_path accordingly.

# 4. Single-step / single-episode demo
python -m agent.environment --task click-dialog --seed 15101 --out demo_trace.json

# 5. Closed-loop evaluation (unified hist_used=6)
python -m eval.miniwob_eval --protocol historical --output outputs/eval_h6 --hist-used 6

# 6. Aggregate metrics
python -m eval.metrics --traces outputs/eval_h6 --hist-used 6 --out summary.json
```

Training entry points (all require explicit `--execute-training` and a configured
workspace; see each script's `--help`):

```bash
python -m scripts.pretrain_p15  --dry-run --config configs/default.yaml
python -m scripts.pretrain_p16  --dry-run --config configs/default.yaml
python -m scripts.full_sft      --check-config
python -m scripts.train_gui_sft --check
python -m scripts.build_distill_dataset --out <dir> --base-train <file> --base-val <file>
python -m scripts.convert_teacher_sft --eval-dir <dir> --out <dir> --teacher <name>
```

## 7. Acknowledgements and license

This work builds on the open-source **MiniMind / MiniMind-O** project and its Dense
architecture, uses **SigLIP2** as the frozen visual encoder, and relies on
**BrowserGym / MiniWoB++** for real-web closed-loop evaluation. We thank the authors
and maintainers of these projects.

**License and data statement.** This repository releases code and configuration under
the terms described in the included license files (`licenses/`). Restricted by upstream corpus
licensing agreements, it does **not** include the original training datasets or any
pretrained model weights. Users must obtain those assets independently and comply
with their respective licenses. External encoders remain the property of their
respective owners.
