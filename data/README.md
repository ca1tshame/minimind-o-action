# GUI SFT distillation data

This directory documents the GUI SFT distillation dataset shipped by the
pipeline in `scripts/`. The repository contains **code and metadata only**:
screenshots, raw teacher outputs and model weights are deliberately **not**
included (see [Redistribution limits](#redistribution-limits)).

Files:

| Path | Contents |
| --- | --- |
| `data/schemas/record.schema.json` | JSON Schema (draft 2020-12) for one JSONL record |
| `data/README.md` | this document: schema, sources, counts, limits |
| `datasets/**` (git-ignored) | generated `*.jsonl`, `frames/`, `manifest.json` |

## Record schema

One JSON object per line, UTF-8, exactly these 16 fields and no others:

```json
{"episode_id": "click-dialog_20101", "step": 0, "task": "click-dialog", "seed": 20101,
 "instruction": "Click the button to close the dialog.",
 "frame": "frames/v14_click-dialog_20101_0000.jpg",
 "frame_sha256": "<64 hex chars>", "history": [], "error_feedback": null,
 "target_action": "<action>click(0.5000,0.5000)</action>", "action_kind": "click",
 "data_source": "v14_oracle", "split": "train", "reward": 1.0, "success": true,
 "target_point": [0.5, 0.5]}
```

| Field | Type | Meaning |
| --- | --- | --- |
| `episode_id` | string | stable rollout id, shared by all steps of one episode |
| `step` | int >= 0 | 0-based action index inside the episode |
| `task` | string | task id (MiniWoB id for oracle data) |
| `seed` | int | environment seed of the rollout |
| `instruction` | string | instruction shown to the model (raw goal text for oracle rows) |
| `frame` | string | screenshot path **relative to the frames root**, POSIX separators |
| `frame_sha256` | string | SHA-256 of the screenshot bytes; used for integrity + dedup |
| `history` | string[] | earlier `<action>...</action>` strings of the same episode |
| `error_feedback` | string or null | last environment/executor error, if any |
| `target_action` | string | one constrained action, see below |
| `action_kind` | enum | `click` \| `type` \| `scroll` \| `stop`, must agree with `target_action` |
| `data_source` | enum | `v11_teacher` \| `v14_oracle` \| `gui_owl` \| `v5_old` |
| `split` | enum | `train` \| `validation` (a frozen test set lives in its own file) |
| `reward` | float or null | episode reward, finite |
| `success` | bool or null | environment-verified success; `null` = unverified |
| `target_point` | `[x, y]` or null | normalized click point, `null` for non-click actions |

### Action DSL

| Kind | Form |
| --- | --- |
| click | `<action>click(0.4321,0.7654)</action>` — coordinates in `[0, 1]` |
| type (literal) | `<action>type("hello")</action>` — JSON string, e.g. `type("say \"hi\"")` |
| type (copy pointer) | `<action>type(@10:16|lower)</action>` — copy `instruction[10:16]`, optionally `|lower` / `|upper` |
| scroll | `<action>scroll(0.0000,0.9000)</action>` — deltas in `[-1, 1]`, optional third `inside` boolean |
| stop | `<action>stop(done)</action>` |

Copy pointers keep the label anchored to the instruction the policy sees; the
converter (`scripts/build_distill_dataset.py:to_type_pointer`) only emits one
when the literal text really occurs in the instruction (exact, then lower-case,
then upper-case span), otherwise it keeps `type("...")`.

## Sources

| `data_source` | Provenance | Rows | Episodes |
| --- | --- | --- | --- |
| `v11_teacher` | GUI-Owl-1.5-8B teacher rollouts on MiniWoB, filtered by `scripts/filter_teacher.py` | 2259 | — |
| `v14_oracle` | live BrowserGym/MiniWoB oracle rollouts from `scripts/collect_oracle.py` | 705 | 183 of 280 successful |
| `gui_owl` | GUI-Owl adapter output converted to the action DSL | 46 | 22 of 60 successful |
| `v5_old` | earlier in-house base corpus (legacy rows, kept only as a source tag) | — | — |

`v14_oracle` rows are **oracle** rows: the environment really executed the
action and rewarded it. They are never GUI-Owl rows, and a row is only admitted
when the rollout terminated with `reward > 0`.

## Published counts

Totals of the V14 run (all sources merged):

| Stage | Count | Delta |
| --- | --- | --- |
| raw sources (2259 + 705 + 46) | 3010 | — |
| after `BLOCKED_SEEDS` (frozen-eval seeds removed) | 2950 | -60 |
| after content dedup, key `(frame_sha256, target_action, instruction[:80])` | **2915 unique rows** | -35 |
| after task balancing, `easy_cap = 1200` | **2783** | -132 |
| train split | **2552** | — |
| validation split | **231** | — |

**2915 is the number of unique rows after dedup, not the number of training
rows.** Training consumes the 2552 rows of `train.jsonl`; 231 rows are held out
for validation, and any frozen test set is separate from both. The builder never
claims that all 2915 unique rows were trained on.

`manifest.json` written next to the splits carries the same numbers with a
per-source breakdown at every stage, the dedup hash, the split seed and a
SHA-256 for each output file. Counts are computed by walking the actual rows, so
a different source mix produces different numbers rather than a copied claim.

## Reproducing

All scripts accept `--workspace-root` and `--config` and resolve relative paths
through `configs/`; no machine paths are baked in. `--workspace-root` falls back
to `paths.workspace_root` from the config file and then to the current directory.

```bash
# 1. filter teacher rollouts (kept.jsonl / rejected.jsonl / stats.json)
python scripts/filter_teacher.py --workspace-root . \
  --episodes-dir datasets/v11_teacher/episodes \
  --frames-root datasets/v11_teacher --out datasets/v11_teacher_filtered

# 2. collect environment-verified oracle rollouts (needs BrowserGym + MiniWoB)
python scripts/collect_oracle.py --workspace-root . \
  --out datasets/v14_oracle --seeds 20101-20140 --handoff

# 3. merge sources, dedup, balance, split
python scripts/build_distill_dataset.py --workspace-root . \
  --v11-teacher datasets/v11_teacher_filtered/kept.jsonl \
  --v14-oracle datasets/v14_oracle/trajectories.jsonl \
  --gui-owl datasets/gui_owl/converted.jsonl \
  --out datasets/gui_distill_v14

# 4. package canonical SFT JSONL with frozen-test isolation
python scripts/prepare_gui_sft_data.py --workspace-root . \
  --records datasets/gui_distill_v14/train.jsonl datasets/gui_distill_v14/validation.jsonl \
  --frozen-test datasets/frozen_test.json --out datasets/gui_sft
```

Tests (CPU only, no BrowserGym, no pytest needed):

```bash
python -m unittest tests.test_distill -v
```

## Frozen test isolation

A frozen test set is never written into a training split:

* `--frozen-test` accepts a JSON manifest with `tasks` / `seeds` / `episodes`
  (whole-task exclusion applies only when `tasks` is non-empty) or a JSONL
  record list (episode/seed-level exclusion).
* `scripts/build_distill_dataset.py` removes conflicting rows before dedup and
  records the removals under `stages.after_frozen_test`.
* `scripts/prepare_gui_sft_data.py` calls `assert_isolated()` and fails closed:
  a test episode, test seed or declared test task appearing in train/validation
  raises `IsolationError` before anything is written.
* `test.jsonl` is a separate file; `train.jsonl` and `validation.jsonl` never
  contain it.

## Redistribution limits

* **Screenshots are not included.** The rendered MiniWoB pages and teacher
  screenshots are not redistributed with this repository; `frame` values only
  point into a local frames root. Regenerate them with `scripts/collect_oracle.py`
  (your own BrowserGym run) or your own teacher run, and honour the upstream
  terms of the environments and of the teacher model.
* **Weights are not included.** The GUI-Owl teacher weights and this project's
  trained adapters are not part of the data directory; obtain the teacher from
  its upstream release and train student weights locally.
* **Only derived labels are shipped**: `target_action`, `instruction`, episode
  identity, hashes and counts. The repository's code license covers the scripts;
  the upstream data/environment licenses continue to apply to anything you
  regenerate locally.
* Do not commit `datasets/**`, `*.jsonl`, `*.jpg`, `*.png` or model weights;
  `.gitignore` already excludes them.

## Notes and caveats

* The rows are a mix of teacher (model-generated) and oracle (environment-executed)
  labels; `data_source` is the only field that distinguishes them. Do not report
  the oracle portion as GUI-Owl data.
* `success: null` means "not environment-verified" (legacy teacher rows). The
  teacher filter only keeps episodes whose recorded `success` is `true`, so
  shipped teacher rows are `success: true`.
* Task balancing uses `easy_cap = 1200` per task and keeps the small hard tasks
  (`click-pie`, `click-option`, `click-link`, `login-user-popup`, ...) intact, so
  the cap removes rows only from the largest tasks.
* Determinism: source rows are sorted by `(task, seed, episode_id, step,
  frame_sha256, target_action)`; dedup keeps the first row in that order;
  balancing and the split use `random.Random(split_seed)` (default 14).
