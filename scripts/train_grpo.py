"""Portable online GRPO trainer for the MiniMind-O Action GUI agent.

This is the V12/V13 agentic-RL experiment ported out of the read-only research
tree and onto the public repo's APIs (``agent.environment``,
``agent.ctx_pack``, ``agent.vision_tiles``, ``models.vision``).  It trains on
real BrowserGym MiniWoB++ rollouts only: there is no synthetic reward path and
no fabricated trace.

Entropy logging (V13 bug fixed)
-------------------------------
``seq_logp`` in the V13 research script already returned the per-action-token
mean entropy (``ent / max(1, len(aids))``), but the outer update log divided the
accumulated entropy by ``n_tok`` *again*.  That double normalisation produced
the published 0.026-0.068 monitor values; V12's single-normalised values were
0.60-0.72.  The historical V13 entropy numbers are therefore invalid.  Here the
per-token entropies come from ``scripts.lib.rl.sequence_logprobs`` (already
normalised) and are aggregated with action-token weights -- never divided again.

Honest experimental record (kept from the research tree)
--------------------------------------------------------
* V12: 480 on-policy episodes over 40 updates; no measured gain over V11.
* V13: high learning rate, 40 updates / 480 episodes; the new-seed-32
  evaluation regressed from 15/32 to 13/32 successes.
* Both runs: 50-75% zero-variance groups, KL approximately 0, gradient norm
  27-48 effectively pinned to the 0.3 clip.  With that history, treat this
  trainer as an experiment harness, not as a recipe that is known to help.

Configuration
-------------
``configs/default.yaml`` has no ``rl:`` section (and this port must not edit it),
so every knob below mirrors the V13 defaults and can be overridden either by an
optional ``rl:`` block in any user-supplied config or by the CLI::

    python scripts/train_grpo.py --config configs/default.yaml \
        --checkpoint outputs/experiments/action.safetensors --device cpu

Checkpoint loading order (important)
------------------------------------
``attach_tile_compression`` must run *before* the action weights are loaded,
otherwise ``compressor``/``tile_pos`` stay randomly initialised while the model
looks "loaded".  ``models.vision.load_action_model`` enforces exactly that when
present; this module also refuses to proceed if those weights are missing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib import rl

DEFAULT_TASKS = ("click-test", "click-dialog", "click-checkboxes", "enter-text")
DEFAULT_SEED_LO = 17101
DEFAULT_PREPROCESS = {
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}

# V13 defaults: group 3-4 episodes per task+seed, 4 tasks per update, lr 2e-6,
# KL 0.02, temperature 1.2, gradient clip 0.3.  Clip kept at 0.3 because that
# is where V13's 27-48 gradient norm was pinned; lower it only deliberately.
DEFAULT_RL: Dict[str, Any] = {
    "group_size": 4,
    "learning_rate": 2e-6,
    "kl_coefficient": 0.02,
    "temperature": 0.9,
    "grad_clip": 0.3,
    "weight_decay": 0.0,
    "max_steps": 6,
    # 48 is the evaluation protocol budget. At 24 a 4-argument
    # scroll_at(cx,cy,dx,dy) is cut mid-token and every scroll step turns into
    # a parse error, which silently understates the checkpoint's real ability.
    "max_new": 48,
    "hist_used": 6,
    "vision_tokens": None,
    "context": None,
    "reward_penalty": 0.05,
    "save_every": 10,
    "ref_checkpoint": None,
}


# ---------------------------------------------------------------------------
# Small config helpers
# ---------------------------------------------------------------------------

def rl_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge V13 defaults, an optional ``rl:`` config block and eval defaults."""
    eval_cfg = cfg.get("eval") or {}
    settings = dict(DEFAULT_RL)
    section = cfg.get("rl") or {}
    if isinstance(section, dict):
        for key, value in section.items():
            if value is not None:
                settings[key] = value
    if not settings.get("vision_tokens"):
        settings["vision_tokens"] = int(eval_cfg.get("vision_tokens", 320))
    if not settings.get("context"):
        settings["context"] = int(eval_cfg.get("context", 512))
    if not settings.get("max_new"):
        settings["max_new"] = int(eval_cfg.get("max_new", 48))
    if settings.get("hist_used") is None:
        settings["hist_used"] = int(eval_cfg.get("hist_used", 6))
    settings["group_size"] = int(settings["group_size"])
    settings["max_steps"] = int(settings["max_steps"])
    settings["max_new"] = int(settings["max_new"])
    settings["hist_used"] = int(settings["hist_used"])
    settings["context"] = int(settings["context"])
    settings["vision_tokens"] = int(settings["vision_tokens"])
    settings["save_every"] = int(settings["save_every"])
    return settings


def workspace_root(cfg: Dict[str, Any], override: Optional[str] = None) -> Path:
    raw = override or (cfg.get("paths") or {}).get("workspace_root")
    return (Path(raw) if raw else ROOT).resolve()


def resolve_asset(root: Path, value: Optional[str]) -> Optional[Path]:
    """Resolve a config path against the workspace root, the repo, or cwd."""
    if value in (None, ""):
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (root, ROOT, Path.cwd()):
        resolved = (base / candidate)
        if resolved.exists():
            return resolved
    return root / candidate


def _device_from(cfg: Dict[str, Any], override: Optional[str]) -> torch.device:
    name = override or (cfg.get("device") or {}).get("device") or "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "device %r requested but torch.cuda.is_available() is False; pass --device cpu"
            % name
        )
    return device


# ---------------------------------------------------------------------------
# Model / tokenizer construction
# ---------------------------------------------------------------------------

def load_tokenizer(cfg: Dict[str, Any], root: Path):
    """Load the tokenizer described by the config and verify the image token."""
    from transformers import AutoTokenizer

    from models import IMAGE_ID, IMAGE_TOKEN

    tokenizer_dir = resolve_asset(root, (cfg.get("paths") or {}).get("tokenizer_dir"))
    if tokenizer_dir is None:
        raise SystemExit(
            "paths.tokenizer_dir is not set; point it at the local tokenizer directory"
        )
    if not Path(tokenizer_dir).is_dir():
        raise SystemExit("tokenizer directory not found: %s" % tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True, trust_remote_code=False
    )
    if tokenizer.convert_tokens_to_ids(IMAGE_TOKEN) != IMAGE_ID:
        raise SystemExit(
            "tokenizer image token %r does not map to id %d" % (IMAGE_TOKEN, IMAGE_ID)
        )
    return tokenizer


def _load_base_lm(lm, path: Path) -> None:
    from safetensors.torch import load_model

    if not Path(path).is_file():
        raise SystemExit("base checkpoint not found: %s" % path)
    try:
        missing, unexpected = load_model(lm, str(path), strict=True, device="cpu")
    except RuntimeError as exc:  # strict load_state_dict reports through RuntimeError
        raise SystemExit(
            "base checkpoint %s does not match the configured LM (strict load failed): %s"
            % (path, exc)
        ) from exc
    if missing or unexpected:
        raise SystemExit(
            "base checkpoint %s mismatch: missing=%s unexpected=%s"
            % (path, missing[:10], unexpected[:10])
        )


def _load_action_weights(model, checkpoint: Path) -> Dict[str, Any]:
    """Load action weights through ``models.vision.load_action_model`` if present.

    The fallback mirrors the essential safety property (vision-encoder keys are
    dropped, every remaining key must match) but cannot know about legacy
    projector aliases; it therefore fails loudly instead of silently skipping.
    """
    try:
        from models.vision import load_action_model as shared_loader
    except (ImportError, AttributeError):  # pragma: no cover - older tree
        shared_loader = None
    if shared_loader is not None:
        try:
            report = shared_loader(model, str(checkpoint))
        except Exception as exc:
            raise SystemExit(
                "could not load action checkpoint %s: %s" % (checkpoint, exc)
            ) from exc
        return dict(report)

    from safetensors.torch import load_file

    state = load_file(str(checkpoint))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("vision_encoder."):
            continue
        cleaned[key[11:] if key.startswith("_orig_mod.") else key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise SystemExit(
            "action checkpoint %s does not match the model: missing=%s unexpected=%s"
            " (models.vision.load_action_model is unavailable, so legacy key "
            "aliases cannot be remapped)"
            % (
                checkpoint,
                incompatible.missing_keys[:10],
                incompatible.unexpected_keys[:10],
            )
        )
    return {
        "path": str(checkpoint),
        "renamed_keys": 0,
        "dropped_vision_encoder_keys": len(state) - len(cleaned),
        "missing_keys": [],
        "unexpected_keys": [],
    }


def build_action_model(
    cfg: Dict[str, Any],
    checkpoint: Path,
    *,
    device: torch.device,
    encoder=None,
    settings: Optional[Dict[str, Any]] = None,
):
    """Build and load a real action checkpoint with the correct ordering.

    Returns ``(model, load_report)``.  The frozen SigLIP2 wrapper is loaded
    from ``paths.vision_encoder_dir`` unless a shared ``encoder`` is passed
    (the reference model reuses the student's frozen encoder and is therefore
    not duplicated in memory).
    """
    settings = settings or rl_settings(cfg)
    paths = cfg.get("paths") or {}
    root = workspace_root(cfg)

    model_config = resolve_asset(
        root, paths.get("model_config") or "configs/model_config.json"
    )
    if model_config is None or not Path(model_config).is_file():
        raise SystemExit("model config not found: %s" % model_config)
    lm_config = json.loads(Path(model_config).read_text(encoding="utf8"))
    capacity = int(lm_config.get("max_position_embeddings", 0) or 0)
    needed = int(settings["context"]) + int(settings["max_new"]) + 8
    if capacity and capacity < needed:
        raise SystemExit(
            "model config max_position_embeddings=%d is smaller than the prompt "
            "budget (%d); raise it in %s" % (capacity, needed, model_config)
        )

    from models.minimind import MiniMindConfig, MiniMindForCausalLM
    from models.vision import FrozenSigLIP2, GuiActionModel

    from agent.vision_tiles import attach_tile_compression

    lm = MiniMindForCausalLM(MiniMindConfig(**lm_config))
    base_checkpoint = resolve_asset(root, paths.get("base_checkpoint"))
    if base_checkpoint is not None:
        _load_base_lm(lm, Path(base_checkpoint))

    model = GuiActionModel(lm)
    if encoder is None:
        encoder_dir = resolve_asset(root, paths.get("vision_encoder_dir"))
        if encoder_dir is None or not Path(encoder_dir).is_dir():
            raise SystemExit(
                "paths.vision_encoder_dir is not set or missing: %s" % encoder_dir
            )
        encoder = FrozenSigLIP2(str(encoder_dir))
    object.__setattr__(model, "vision_encoder", encoder)
    object.__setattr__(
        model,
        "vision_preprocess",
        dict((cfg.get("vision") or {}).get("preprocess") or DEFAULT_PREPROCESS),
    )

    # Compression must exist before the checkpoint is applied, otherwise
    # compressor/tile_pos would silently stay random.
    attach_tile_compression(model, int(settings["vision_tokens"]))
    report = _load_action_weights(model, Path(checkpoint))

    model.to(device)
    model.vision_encoder.to(device).eval()
    return model, report


def configure_trainable(model) -> List[torch.nn.Parameter]:
    """Freeze everything except the V13 trainable set.

    V13 trained the last transformer layer, lm_head (tied with the token
    embedding), the vision projector, the tile compressor and tile_pos.  The
    rest of the backbone and the frozen encoder stay fixed.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.lm.model.layers[-1].parameters():
        parameter.requires_grad_(True)
    for parameter in model.lm.lm_head.parameters():
        parameter.requires_grad_(True)
    for parameter in model.proj.parameters():
        parameter.requires_grad_(True)
    for module_name in ("compressor", "tile_pos"):
        module = getattr(model, module_name, None)
        if module is not None:
            for parameter in module.parameters():
                parameter.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Policy adapter (sampling and re-scoring of the exact sampled tokens)
# ---------------------------------------------------------------------------

class GuiPolicy:
    """Autoregressive GUI policy with GRPO re-scoring.

    ``sample()`` draws one action and returns ``(text, prompt_ids, action_ids)``
    so the update can re-score exactly the tokens that were sampled.
    ``score()`` returns ``(sum_logp, mean_action_token_entropy, n_tokens)`` from
    :func:`scripts.lib.rl.sequence_logprobs` -- the entropy is already a
    per-action-token mean.
    """

    def __init__(self, model, tokenizer, *, device, vision_tokens: int,
                 context: int, max_new: int):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.vision_tokens = int(vision_tokens)
        self.context = int(context)
        self.max_new = int(max_new)
        self.vision_preprocess = getattr(model, "vision_preprocess", DEFAULT_PREPROCESS)

    def _encode_prompt(self, prompt: str) -> List[int]:
        ids = [int(self.tokenizer.bos_token_id)]
        ids += list(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if len(ids) + self.max_new > self.context:
            raise ValueError(
                "prompt plus generation exceeds context budget (%d + %d > %d)"
                % (len(ids), self.max_new, self.context)
            )
        return ids

    def _visual(self, screen_rgb) -> torch.Tensor:
        from PIL import Image

        from models.vision import IN_DIM, TOKENS_PER_IMAGE

        from agent.vision_tiles import preprocess_tiles

        image = Image.fromarray(np.asarray(screen_rgb, dtype=np.uint8)).convert("RGB")
        pixels, _boxes = preprocess_tiles(image, self.vision_preprocess)
        pixels = pixels.to(self.device)
        with torch.no_grad():
            features = self.model.vision_encoder(pixels)
            if not torch.is_tensor(features):  # wrappers may return a model output
                features = features.last_hidden_state
            features = features.reshape(1, 5 * TOKENS_PER_IMAGE, IN_DIM).float()
        return features[0].detach()

    def visual_for_frame(self, frame_path) -> torch.Tensor:
        from PIL import Image

        with Image.open(frame_path) as image:
            screen = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return self._visual(screen)

    def _hidden(self, prompt_ids: Sequence[int], action_ids: Sequence[int],
                visual: torch.Tensor):
        ids = list(prompt_ids) + list(action_ids)
        tokens = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(tokens)
        embedded = self.model.lm.model.embed_tokens(tokens)
        hidden = self.model.inject(tokens, embedded, attention, [[visual]])
        return tokens, attention, hidden

    def sample(self, screen_rgb, prompt: str, temperature: float = 0.9):
        """Sample one action autoregressively; returns ``(text, prompt_ids, action_ids)``."""
        from models.vision import lm_hidden_forward

        prompt_ids = self._encode_prompt(prompt)
        visual = self._visual(screen_rgb)
        with torch.inference_mode():
            tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
            attention = torch.ones_like(tokens)
            hidden = self.model.inject(
                tokens, self.model.lm.model.embed_tokens(tokens), attention, [[visual]]
            )
            generated: List[int] = []
            eos_ids = {int(self.tokenizer.eos_token_id), 2}
            for _ in range(self.max_new):
                logits = lm_hidden_forward(self.model.lm, hidden, attention)
                scores = logits[0, -1].float() / max(float(temperature), 1e-6)
                next_id = int(torch.multinomial(torch.softmax(scores, dim=-1), 1).item())
                if next_id in eos_ids:
                    break
                generated.append(next_id)
                tokens = torch.cat(
                    [tokens, torch.tensor([[next_id]], dtype=torch.long, device=self.device)],
                    dim=1,
                )
                attention = torch.ones_like(tokens)
                hidden = torch.cat(
                    [hidden, self.model.lm.model.embed_tokens(tokens[:, -1:])], dim=1
                )
        text = self.tokenizer.decode(generated, skip_special_tokens=False)
        return text, prompt_ids, generated

    @torch.no_grad()
    def predict(self, screen_rgb, prompt: str) -> str:
        """Greedy action for evaluation/demo (``agent.environment.run_episode``)."""
        from models.vision import lm_hidden_forward

        prompt_ids = self._encode_prompt(prompt)
        visual = self._visual(screen_rgb)
        tokens = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        attention = torch.ones_like(tokens)
        hidden = self.model.inject(
            tokens, self.model.lm.model.embed_tokens(tokens), attention, [[visual]]
        )
        generated: List[int] = []
        eos_ids = {int(self.tokenizer.eos_token_id), 2}
        for _ in range(self.max_new):
            logits = lm_hidden_forward(self.model.lm, hidden, attention)
            next_id = int(torch.argmax(logits[0, -1]).item())
            if next_id in eos_ids:
                break
            generated.append(next_id)
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], dtype=torch.long, device=self.device)],
                dim=1,
            )
            attention = torch.ones_like(tokens)
            hidden = torch.cat(
                [hidden, self.model.lm.model.embed_tokens(tokens[:, -1:])], dim=1
            )
        return self.tokenizer.decode(generated, skip_special_tokens=False)

    def score(self, frame, prompt_ids: Sequence[int], action_ids: Sequence[int],
              visual: Optional[torch.Tensor] = None):
        """Re-score sampled tokens; entropy is already per action token."""
        from models.vision import lm_hidden_forward

        if visual is None:
            visual = self.visual_for_frame(frame)
        tokens, attention, hidden = self._hidden(prompt_ids, action_ids, visual)
        logits = lm_hidden_forward(self.model.lm, hidden, attention)
        action_mask = torch.zeros(tokens.shape, dtype=torch.bool, device=self.device)
        action_mask[:, len(prompt_ids):] = True
        return rl.sequence_logprobs(logits, tokens, action_mask)


# ---------------------------------------------------------------------------
# Rollout sampling
# ---------------------------------------------------------------------------

def make_act(policy: GuiPolicy, settings: Dict[str, Any], frame_dir: Path,
             counter: Iterable[int]):
    """Build the ``act`` hook used by ``rl.rollout_group``."""
    from PIL import Image

    from agent.ctx_pack import assemble_prompt

    vision_tokens = settings["vision_tokens"]
    context = settings["context"]
    max_new = settings["max_new"]
    hist_used = settings["hist_used"]
    temperature = settings["temperature"]
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)

    def act(observation, goal, history, error):
        packed = assemble_prompt(
            policy.tokenizer, goal, vision_tokens, history, context, max_new,
            error=error, hist_used=hist_used,
        )
        screen = np.asarray(observation["screenshot"], dtype=np.uint8)
        text, prompt_ids, action_ids = policy.sample(screen, packed.prompt, temperature)
        index = next(counter)
        frame = frame_dir / ("ep%06d_st%02d.png" % (index, len(history)))
        Image.fromarray(screen).save(frame)
        return {
            "text": text,
            "prompt_ids": prompt_ids,
            "action_ids": action_ids,
            "frame": str(frame),
            "input_tokens": packed.input_tokens,
        }

    return act


def sample_update(
    policy: GuiPolicy,
    settings: Dict[str, Any],
    tasks: Sequence[str],
    seeds: Sequence[int],
    frame_dir: Path,
    counter: Iterable[int],
):
    """Sample one GRPO update: one group per (task, seed), all on-policy."""
    from agent.dsl import format_action
    from agent.environment import decode_action, make_env

    act = make_act(policy, settings, frame_dir, counter)

    def env_factory(task: str):
        return make_env(task, headless=True)

    groups: List[Tuple[Tuple[str, int], List[Dict[str, Any]]]] = []
    for task in tasks:
        for seed in seeds:
            group = rl.rollout_group(
                env_factory,
                act,
                task,
                int(seed),
                settings["group_size"],
                max_steps=settings["max_steps"],
                decode=decode_action,
                format_action=format_action,
                reward_penalty=float(settings["reward_penalty"]),
            )
            groups.append(((task, int(seed)), group))
    return groups


# ---------------------------------------------------------------------------
# One optimisation step
# ---------------------------------------------------------------------------

def optimise_update(
    policy: GuiPolicy,
    ref_policy: GuiPolicy,
    optimizer,
    scheduler,
    trainable: Sequence[torch.nn.Parameter],
    groups,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-score all sampled actions and apply one accumulated GRPO update.

    ``optimizer.zero_grad`` happens exactly once, here: the update boundary.
    Episodes whose advantage is ~0 are skipped, matching V12/V13 (most episodes
    in those runs landed in zero-variance groups and contributed nothing).
    """
    episodes = [episode for _key, group in groups for episode in group]
    rewards = [float(episode["reward"]) for episode in episodes]
    group_ids = [group_index for group_index, (_key, group) in enumerate(groups)
                 for _episode in group]
    advantages, stats = rl.group_relative_advantage(rewards, group_ids)

    optimizer.zero_grad(set_to_none=True)

    entropy_values: List[float] = []
    entropy_weights: List[int] = []
    kl_values: List[float] = []
    kl_weights: List[int] = []
    loss_total = 0.0
    loss_terms = 0
    action_tokens = 0
    skipped = 0
    kl_coefficient = float(settings["kl_coefficient"])

    for episode, advantage in zip(episodes, advantages):
        if abs(advantage) <= 1e-12:
            skipped += 1
            continue
        for record in episode["steps"]:
            extra = record.get("policy") or {}
            prompt_ids = extra.get("prompt_ids")
            action_ids = extra.get("action_ids")
            frame = extra.get("frame")
            if not prompt_ids or not action_ids:
                continue
            if not frame or not Path(frame).is_file():
                raise RuntimeError(
                    "no frame recorded for re-scoring (task=%s seed=%s step=%s)"
                    % (episode.get("task"), episode.get("seed"), record.get("step"))
                )
            visual = policy.visual_for_frame(frame)
            logp, entropy, n_tokens = policy.score(
                frame, prompt_ids, action_ids, visual=visual
            )
            if n_tokens == 0:
                continue
            action_tokens += n_tokens
            with torch.no_grad():
                ref_logp, _ref_entropy, _ref_tokens = ref_policy.score(
                    frame, prompt_ids, action_ids, visual=visual
                )
                # sequence_logprobs returns a sum, so divide the sums by the
                # action-token count before the sampled-token estimator.  The
                # monitored KL is therefore per action token, like V12's.
                kl_value = float(
                    rl.kl_divergence(ref_logp / n_tokens, logp / n_tokens)
                )
            shaped_advantage = float(advantage) - kl_coefficient * kl_value
            loss = rl.policy_gradient_loss(
                logp, torch.full_like(logp, shaped_advantage, requires_grad=False)
            )
            loss.backward()
            loss_total += float(loss.detach())
            loss_terms += 1
            entropy_values.append(float(entropy))
            entropy_weights.append(int(n_tokens))
            kl_values.append(kl_value)
            kl_weights.append(int(n_tokens))

    grad_norm_value = 0.0
    if loss_terms > 0:
        grad_norm_value = rl.grad_norm(trainable, max_norm=float(settings["grad_clip"]))
        optimizer.step()
        scheduler.step()
    else:
        # No learning signal this update (all groups had zero variance); keep the
        # LR schedule in step so a resumed run has the same schedule position.
        scheduler.step()

    entropy_summary = rl.entropy_stats(entropy_values, entropy_weights) \
        if entropy_values else {"n": 0, "mean": None}
    kl_summary = rl.entropy_stats(kl_values, kl_weights) \
        if kl_values else {"n": 0, "mean": None}
    return {
        "episodes": len(episodes),
        "successes": sum(1 for episode in episodes if episode.get("success")),
        "loss_terms": loss_terms,
        "skipped_zero_advantage": skipped,
        "reward_mean": stats["reward_mean"],
        "reward_std": stats["reward_std"],
        "advantage_std": stats["advantage_std"],
        "num_groups": stats["num_groups"],
        "zero_variance_groups": stats["zero_variance_groups"],
        "zero_variance_group_fraction": stats["zero_variance_group_fraction"],
        "entropy": entropy_summary["mean"],
        "entropy_n": entropy_summary.get("n", 0),
        "kl": kl_summary["mean"],
        "grad_norm": grad_norm_value,
        "action_tokens": action_tokens,
        "loss": (loss_total / loss_terms) if loss_terms else None,
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def checkpoint_contract(cfg: Dict[str, Any], settings: Dict[str, Any],
                        checkpoint: Path, ref_checkpoint: Path,
                        tasks: Sequence[str], seed_lo: int) -> Dict[str, Any]:
    """Values a resume must match exactly (``scripts/lib/checkpoint.py``)."""
    return {
        "format": "minimind-grpo-v1",
        "checkpoint": str(checkpoint),
        "ref_checkpoint": str(ref_checkpoint),
        "tasks": list(tasks),
        "group_size": int(settings["group_size"]),
        "max_steps": int(settings["max_steps"]),
        "max_new": int(settings["max_new"]),
        "temperature": float(settings["temperature"]),
        "kl_coefficient": float(settings["kl_coefficient"]),
        "grad_clip": float(settings["grad_clip"]),
        "vision_tokens": int(settings["vision_tokens"]),
        "context": int(settings["context"]),
        "seed_lo": int(seed_lo),
    }


def latest_checkpoint(root: Path) -> Optional[str]:
    root = Path(root)
    if not root.is_dir():
        return None
    tags = [entry.name for entry in root.iterdir() if (entry / "READY.json").is_file()]
    if not tags:
        return None

    def step_of(tag: str) -> int:
        manifest = root / tag / "payload" / "manifest.json"
        try:
            return int(json.loads(manifest.read_text(encoding="utf8"))["progress"]["step"])
        except (OSError, KeyError, ValueError):
            return -1

    return max(tags, key=step_of)


def save_update_checkpoint(root: Path, step: int, model, optimizer, scheduler,
                           contract: Dict[str, Any], action_tokens: int) -> Path:
    from scripts.lib.checkpoint import save_checkpoint

    progress = {
        "step": int(step),
        "epoch": 0,
        "next_sample": 0,
        "supervised_tokens": int(action_tokens),
        "truncated_samples": 0,
        "order": [],
        "accumulation_boundary": True,
    }
    tag = "step_%06d" % int(step)
    return save_checkpoint(root, tag, model, optimizer, scheduler, progress, contract)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_seeds(text: Optional[str]) -> Optional[List[int]]:
    """Parse ``--seeds 17101,17102`` or ``--seeds 17101:4``; ``None`` = derive."""
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if ":" in text:
        low, count = text.split(":", 1)
        low, count = int(low), int(count)
        if count <= 0:
            raise ValueError("seed count must be positive")
        return [low + index for index in range(count)]
    return [int(part) for part in text.split(",") if part.strip()]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable online GRPO trainer (V12/V13 port).")
    parser.add_argument("--config", default=None,
                        help="YAML config path (default: configs/default.yaml)")
    parser.add_argument("--workspace-root", default=None,
                        help="root directory all config paths resolve against")
    parser.add_argument("--steps", type=int, default=10,
                        help="number of GRPO updates to run (after --resume)")
    parser.add_argument("--group-size", type=int, default=None,
                        help="episodes sampled per task+seed group")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--seeds", default=None,
                        help="comma list or lo:count; omitted = one fresh seed per task/update")
    parser.add_argument("--device", default=None, help="cpu / cuda / cuda:0")
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="resume from a checkpoint tag, or 'latest'")
    parser.add_argument("--checkpoint", default=None, help="action checkpoint for the student")
    parser.add_argument("--ref-checkpoint", default=None,
                        help="frozen reference checkpoint (default: same as --checkpoint)")
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--vision-encoder-dir", default=None)
    parser.add_argument("--output", default=None, help="run directory (default: <experiment_dir>/grpo)")
    parser.add_argument("--max-steps", type=int, default=None, help="environment steps per episode")
    parser.add_argument("--max-new", type=int, default=None, help="generated tokens per action")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--kl", type=float, default=None, help="KL coefficient")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--seed-lo", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="run seed (torch/numpy/python)")
    parser.add_argument("--check", action="store_true",
                        help="validate configuration and paths, then exit without rolling out")
    return parser.parse_args(argv)


def _apply_cli(settings: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    settings = dict(settings)
    for cli_name, key in (
        ("group_size", "group_size"),
        ("max_steps", "max_steps"),
        ("max_new", "max_new"),
        ("temperature", "temperature"),
        ("grad_clip", "grad_clip"),
        ("save_every", "save_every"),
    ):
        value = getattr(args, cli_name)
        if value is not None:
            settings[key] = value
    if args.lr is not None:
        settings["learning_rate"] = args.lr
    if args.kl is not None:
        settings["kl_coefficient"] = args.kl
    return settings


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    from configs import load_config

    cfg = load_config(args.config, {
        "paths.workspace_root": args.workspace_root,
        "paths.checkpoint": args.checkpoint,
        "paths.tokenizer_dir": args.tokenizer_dir,
        "paths.vision_encoder_dir": args.vision_encoder_dir,
        "device.device": args.device,
    })
    root = workspace_root(cfg, args.workspace_root)
    settings = _apply_cli(rl_settings(cfg), args)
    device = _device_from(cfg, args.device)

    paths = cfg.get("paths") or {}
    checkpoint = resolve_asset(root, paths.get("checkpoint"))
    if checkpoint is None:
        raise SystemExit(
            "no action checkpoint configured: set paths.checkpoint in the config "
            "or pass --checkpoint"
        )
    if not Path(checkpoint).is_file():
        raise SystemExit("action checkpoint not found: %s" % checkpoint)
    ref_checkpoint = resolve_asset(
        root, args.ref_checkpoint or settings.get("ref_checkpoint")
        or paths.get("ref_checkpoint") or str(checkpoint)
    )
    if ref_checkpoint is None or not Path(ref_checkpoint).is_file():
        raise SystemExit("reference checkpoint not found: %s" % ref_checkpoint)

    tasks = [task.strip() for task in str(args.tasks).split(",") if task.strip()]
    if not tasks:
        raise SystemExit("--tasks is empty")
    seed_lo = int(args.seed_lo if args.seed_lo is not None
                  else (cfg.get("device") or {}).get("seed", DEFAULT_SEED_LO))
    seeds = parse_seeds(args.seeds)
    contract = checkpoint_contract(cfg, settings, Path(checkpoint),
                                   Path(ref_checkpoint), tasks, seed_lo)

    if args.check:
        print("GRPO_CONFIG_OK")
        print("  workspace_root :", root)
        print("  checkpoint     :", checkpoint)
        print("  ref_checkpoint :", ref_checkpoint)
        print("  device         :", device)
        print("  tasks          :", ",".join(tasks))
        print("  group_size     :", settings["group_size"])
        print("  steps          :", args.steps)
        print("  entropy        : per action token (already normalised)")
        return 0

    run_seed = int(args.seed if args.seed is not None else (cfg.get("device") or {}).get("seed", 4225))
    random.seed(run_seed)
    np.random.seed(run_seed)
    torch.manual_seed(run_seed)

    # Fail before building anything heavy if BrowserGym is absent: this trainer
    # only accepts real environment rewards.
    from agent.environment import make_env  # noqa: F401  (raises if unavailable)
    try:
        import browsergym.miniwob  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "BrowserGym is not installed, so no real rollout is possible and this "
            "trainer refuses to fabricate rewards. Install the eval extra: "
            "pip install 'minimind-o-action[eval]'"
        ) from exc

    tokenizer = load_tokenizer(cfg, root)
    model, load_report = build_action_model(
        cfg, Path(checkpoint), device=device, settings=settings
    )
    ref_model, ref_report = build_action_model(
        cfg, Path(ref_checkpoint), device=device,
        encoder=model.vision_encoder, settings=settings,
    )
    for parameter in ref_model.parameters():
        parameter.requires_grad_(False)
    ref_model.eval()
    model.train()

    trainable = configure_trainable(model)
    if not trainable:
        raise SystemExit("no trainable parameters were selected")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)

    out_root = Path(args.output) if args.output else Path(
        paths.get("experiment_dir") or "outputs/experiments"
    ) / "grpo"
    if not out_root.is_absolute():
        out_root = root / out_root
    out_root.mkdir(parents=True, exist_ok=True)
    frame_dir = out_root / "frames"
    checkpoint_dir = out_root / "checkpoints"
    log_path = out_root / "run_log.jsonl"

    start_step = 0
    action_tokens_total = 0
    if args.resume:
        from scripts.lib.checkpoint import load_checkpoint

        tag = args.resume
        if tag == "latest":
            tag = latest_checkpoint(checkpoint_dir)
            if tag is None:
                raise SystemExit("no resumable checkpoint under %s" % checkpoint_dir)
        try:
            progress = load_checkpoint(
                checkpoint_dir, tag, model, optimizer, scheduler, contract
            )
        except ValueError as exc:
            raise SystemExit(
                "checkpoint %s cannot be resumed with this configuration: %s"
                % (tag, exc)
            ) from exc
        start_step = int(progress["step"])
        action_tokens_total = int(progress.get("supervised_tokens", 0))
        print("RESUMED", tag, "next_step", start_step)

    policy = GuiPolicy(
        model, tokenizer, device=device,
        vision_tokens=settings["vision_tokens"],
        context=settings["context"],
        max_new=settings["max_new"],
    )
    ref_policy = GuiPolicy(
        ref_model, tokenizer, device=device,
        vision_tokens=settings["vision_tokens"],
        context=settings["context"],
        max_new=settings["max_new"],
    )

    print("GRPO_START", json.dumps({
        "workspace_root": str(root),
        "checkpoint": str(checkpoint),
        "device": str(device),
        "tasks": tasks,
        "group_size": settings["group_size"],
        "temperature": settings["temperature"],
        "kl_coefficient": settings["kl_coefficient"],
        "grad_clip": settings["grad_clip"],
        "load_report": {
            "renamed_keys": load_report.get("renamed_keys"),
            "missing_keys": len(load_report.get("missing_keys") or []),
        },
    }, ensure_ascii=False))

    frame_counter = iter(range(10 ** 12))
    total_tokens = action_tokens_total
    final_step = start_step
    for step in range(start_step, start_step + int(args.steps)):
        started = time.time()
        step_seeds = seeds if seeds is not None else [
            seed_lo + step * len(tasks) + task_index for task_index in range(len(tasks))
        ]
        groups = sample_update(policy, settings, tasks, step_seeds, frame_dir, frame_counter)
        update = optimise_update(
            policy, ref_policy, optimizer, scheduler, trainable, groups, settings
        )
        total_tokens += int(update["action_tokens"])
        record = {
            "step": step,
            "tasks": list(tasks),
            "seeds": list(step_seeds),
            "episodes": update["episodes"],
            "successes": update["successes"],
            "reward_mean": round(update["reward_mean"], 6),
            "reward_std": round(update["reward_std"], 6),
            "advantage_std": round(update["advantage_std"], 6),
            "entropy": None if update["entropy"] is None else round(update["entropy"], 6),
            "kl": None if update["kl"] is None else round(update["kl"], 8),
            "kl_coefficient": float(settings["kl_coefficient"]),
            "grad_norm": round(update["grad_norm"], 6),
            "grad_clip": float(settings["grad_clip"]),
            "lr": update["lr"],
            "loss": None if update["loss"] is None else round(update["loss"], 6),
            "zero_variance_groups": update["zero_variance_groups"],
            "zero_variance_group_fraction": round(
                update["zero_variance_group_fraction"], 4
            ),
            "num_groups": update["num_groups"],
            "skipped_zero_advantage": update["skipped_zero_advantage"],
            "action_tokens": update["action_tokens"],
            "entropy_tokens": update["entropy_n"],
            "seconds": round(time.time() - started, 2),
        }
        with log_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("STEP", json.dumps(record, ensure_ascii=False))
        final_step = step + 1
        if final_step % max(1, int(settings["save_every"])) == 0:
            saved = save_update_checkpoint(
                checkpoint_dir, final_step, model, optimizer, scheduler,
                contract, total_tokens,
            )
            print("CHECKPOINT", saved)

    if final_step > start_step and final_step % max(1, int(settings["save_every"])) != 0:
        saved = save_update_checkpoint(
            checkpoint_dir, final_step, model, optimizer, scheduler, contract, total_tokens
        )
        print("CHECKPOINT", saved)
    print("GRPO_DONE", json.dumps({"steps": final_step, "log": str(log_path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
