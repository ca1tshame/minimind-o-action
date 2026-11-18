"""GRPO math and rollout hooks for the V12/V13 agentic-RL port.

The module is deliberately dependency-light: only ``torch`` is imported at
module scope.  BrowserGym, the tokenizer and the ``agent.*`` helpers are
imported lazily inside the rollout functions, so ``import scripts.lib.rl``
works on a CPU-only host without BrowserGym installed.  ``tests/test_rl.py``
exercises the math on CPU in a couple of seconds.

Entropy normalisation (the V13 logging bug)
-------------------------------------------
``sequence_logprobs`` returns ``mean_action_token_entropy`` as the mean
per-token entropy over the *action* tokens of the sequence, i.e. it has ALREADY
been divided by ``n_action_tokens``.  Callers must not divide it by the token
count a second time.

Why this note exists: the V13 research script
(``project/gui_agent_rl_v13/rl_train_v13.py`` in the read-only research tree)
already returned ``ent / max(1, len(aids))`` from ``seq_logp`` (line ~104), but
the outer update log divided the accumulated entropy by ``n_tok`` again (line
~267).  That double normalisation produced the published 0.026-0.068 monitor
values, whereas V12's single-normalised values were 0.60-0.72.  The historical
V13 entropy numbers are therefore invalid.  This port computes entropy
correctly from now on; it does not reconstruct "corrected" historical numbers.

Honest experimental record carried over from the research tree
--------------------------------------------------------------
* V12: 480 on-policy episodes across 40 updates; no measured gain over V11.
* V13: high learning rate, 40 updates / 480 episodes; the new-seed-32
  evaluation regressed from 15/32 to 13/32 successes.
* Both runs: 50-75% of groups had zero reward variance, KL stayed ~0, and the
  gradient norm was 27-48, effectively pinned to the 0.3 clip threshold.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

__all__ = [
    "entropy_stats",
    "grad_norm",
    "group_relative_advantage",
    "kl_divergence",
    "policy_gradient_loss",
    "rollout_episode",
    "rollout_group",
    "sequence_logprobs",
]


# ---------------------------------------------------------------------------
# Group-relative advantages
# ---------------------------------------------------------------------------

def group_relative_advantage(
    rewards: Sequence[float],
    group_ids: Sequence[Any],
    eps: float = 1e-8,
) -> Tuple[List[float], Dict[str, Any]]:
    """Whitened, group-relative advantages: ``(r - mean_r) / std_r`` per group.

    ``group_ids`` groups samples that came from the same task+seed rollout
    group (GRPO normalises inside a group, not across the batch).  A group
    whose reward standard deviation is ``<= eps`` (all members equal, including
    singleton groups) receives all-zero advantages instead of dividing by zero;
    such groups are counted in the returned statistics.  Advantages are
    bounded by ``sqrt(n - 1)`` for a group of size ``n``, so tiny groups cannot
    explode.

    Returns ``(advantages, stats)`` where ``advantages`` is aligned with
    ``rewards`` and ``stats`` contains::

        num_groups, num_items, zero_variance_groups,
        zero_variance_group_fraction, group_sizes,
        reward_mean, reward_std, advantage_std

    ``zero_variance_group_fraction`` is the fraction of groups that produced
    no learning signal -- in the V12/V13 runs this was 0.5-0.75, which is why
    most of the updates moved nothing.
    """
    rewards = [float(value) for value in rewards]
    group_ids = list(group_ids)
    if len(rewards) != len(group_ids):
        raise ValueError("rewards and group_ids must have the same length")

    buckets: Dict[Any, List[Tuple[int, float]]] = {}
    order: List[Any] = []
    for index, (reward, gid) in enumerate(zip(rewards, group_ids)):
        members = buckets.get(gid)
        if members is None:
            members = buckets[gid] = []
            order.append(gid)
        # Any unhashable group id would have failed above on dict lookup.
        members.append((index, reward))

    advantages = [0.0] * len(rewards)
    zero_variance_groups = 0
    for gid in order:
        members = buckets[gid]
        values = [value for _, value in members]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        std = math.sqrt(max(variance, 0.0))
        if std <= eps:
            zero_variance_groups += 1
            continue
        for index, value in members:
            advantages[index] = (value - mean) / std

    def _std(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return math.sqrt(max(variance, 0.0))

    num_groups = len(order)
    stats = {
        "num_groups": num_groups,
        "num_items": len(rewards),
        "zero_variance_groups": zero_variance_groups,
        "zero_variance_group_fraction": (
            zero_variance_groups / num_groups if num_groups else 0.0
        ),
        "group_sizes": [len(buckets[gid]) for gid in order],
        "reward_mean": (sum(rewards) / len(rewards)) if rewards else 0.0,
        "reward_std": _std(rewards),
        "advantage_std": _std(advantages),
    }
    return advantages, stats


# ---------------------------------------------------------------------------
# Log-probability / entropy of one sampled sequence
# ---------------------------------------------------------------------------

def sequence_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    action_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Score the action tokens of an already-sampled sequence.

    ``logits`` is ``(batch, seq, vocab)`` (raw, pre-softmax), ``input_ids`` is
    ``(batch, seq)`` and ``action_mask`` is a ``(batch, seq)`` boolean tensor
    marking the *target* tokens to score.  Position 0 cannot be scored (it has
    no predecessor) and must not be marked.

    Returns ``(sum_logp, mean_action_token_entropy, n_action_tokens)``:

    * ``sum_logp`` has shape ``(batch,)`` -- the summed log-probability of each
      sequence's action tokens;
    * ``mean_action_token_entropy`` is a scalar and is ALREADY divided by the
      number of action tokens, i.e. it is a per-action-token mean.  **Callers
      must not divide it by ``n_action_tokens`` again.**  It is detached: it is
      a monitoring statistic, not a loss term;
    * ``n_action_tokens`` is the total number of scored tokens.
    """
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, seq, vocab)")
    batch, seq, vocab = logits.shape
    if tuple(input_ids.shape) != (batch, seq):
        raise ValueError("input_ids must have shape (batch, seq)")
    if tuple(action_mask.shape) != (batch, seq):
        raise ValueError("action_mask must have shape (batch, seq)")
    if seq < 2:
        return logits.new_zeros((batch,)), logits.new_zeros(()), 0
    if int(input_ids.min().item()) < 0 or int(input_ids.max().item()) >= vocab:
        raise ValueError("input_ids out of vocabulary range")

    mask = action_mask.to(dtype=torch.bool)
    if bool(mask[:, 0].any()):
        raise ValueError("action_mask must not mark position 0 (no predecessor)")

    # Logits at position t predict the token at position t + 1.
    logp_all = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = input_ids[:, 1:]
    token_logp = logp_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    token_entropy = -(logp_all.exp() * logp_all).sum(dim=-1)

    mask = mask[:, 1:]
    sum_logp = (token_logp * mask).sum(dim=-1)
    n_action_tokens = int(mask.sum().item())
    if n_action_tokens == 0:
        mean_entropy = logits.new_zeros(())
    else:
        mean_entropy = (token_entropy * mask).sum() / n_action_tokens
    return sum_logp, mean_entropy.detach(), n_action_tokens


# ---------------------------------------------------------------------------
# Losses / diagnostics
# ---------------------------------------------------------------------------

def policy_gradient_loss(
    logprobs: torch.Tensor,
    advantages: torch.Tensor,
    old_logprobs: Optional[torch.Tensor] = None,
    clip: Optional[float] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """REINFORCE / PPO surrogate loss with a detached advantage.

    Without ``old_logprobs`` this is the plain policy gradient
    ``-mean(advantage * logprob)``.  Supplying ``old_logprobs`` switches on the
    PPO clipped ratio; ``clip`` is required in that case (a float such as 0.3).
    ``advantages`` must already be detached by the caller: the gradient flows
    only through ``logprobs``.
    """
    if not torch.is_tensor(logprobs):
        logprobs = torch.as_tensor(logprobs, dtype=torch.float32)
    advantages = torch.as_tensor(
        advantages, dtype=logprobs.dtype, device=logprobs.device
    )
    if advantages.shape != logprobs.shape:
        advantages = torch.broadcast_tensors(advantages, logprobs)[0]
    if old_logprobs is not None:
        old_logprobs = torch.as_tensor(
            old_logprobs, dtype=logprobs.dtype, device=logprobs.device
        )
        if clip is None:
            raise ValueError("clip is required when old_logprobs is provided")
        ratio = torch.exp(logprobs - old_logprobs)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages
        per_item = -torch.minimum(unclipped, clipped)
    else:
        per_item = -(advantages * logprobs)
    if reduction == "mean":
        return per_item.mean()
    if reduction == "sum":
        return per_item.sum()
    if reduction == "none":
        return per_item
    raise ValueError("reduction must be 'mean', 'sum' or 'none'")


def kl_divergence(
    ref_logprobs: torch.Tensor,
    logprobs: torch.Tensor,
) -> torch.Tensor:
    """Sampled-token KL estimator ``mean(log pi(a) - log pi_ref(a))``.

    The expectation is over tokens sampled from the current policy, which is
    the estimator used by V12/V13 (it is unbiased up to the samples).  Both
    arguments must already be log-probabilities; for summed sequence
    log-probabilities the result is the summed divergence divided by the
    number of elements.  Identical policies return exactly zero.
    """
    ref_logprobs = torch.as_tensor(ref_logprobs, dtype=torch.float32) \
        if not torch.is_tensor(ref_logprobs) else ref_logprobs
    logprobs = torch.as_tensor(logprobs, dtype=torch.float32) \
        if not torch.is_tensor(logprobs) else logprobs
    if ref_logprobs.shape != logprobs.shape:
        raise ValueError("ref_logprobs and logprobs must have the same shape")
    return (logprobs - ref_logprobs).mean()


def entropy_stats(
    entropies: Sequence[float] | torch.Tensor,
    weights: Optional[Sequence[float] | torch.Tensor] = None,
) -> Dict[str, Any]:
    """Mean/min/max/std of entropy values, optionally token-weighted.

    ``weights`` (e.g. action-token counts) turns the mean into a properly
    token-weighted mean; without it each entry counts once.  Empty input
    returns ``n=0`` with ``None`` values instead of NaN.
    """
    values = torch.as_tensor(entropies, dtype=torch.float64).detach().reshape(-1)
    if values.numel() == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    if weights is None:
        mean = values.mean()
        std = values.std(unbiased=False)
    else:
        weight_tensor = torch.as_tensor(weights, dtype=torch.float64).detach().reshape(-1)
        if weight_tensor.numel() != values.numel():
            raise ValueError("weights must match entropies")
        total = float(weight_tensor.sum().item())
        if total <= 0:
            raise ValueError("weights must sum to a positive value")
        mean = (values * weight_tensor).sum() / total
        std = ((values - mean) ** 2 * weight_tensor).sum() / total
        std = std.sqrt()
    return {
        "n": int(values.numel()),
        "mean": float(mean.item()),
        "std": float(std.item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def grad_norm(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: Optional[float] = None,
) -> float:
    """Global gradient norm; clips in place when ``max_norm`` is given.

    With ``max_norm`` this returns the *pre-clip* norm (the value
    ``torch.nn.utils.clip_grad_norm_`` returns), which is the number that
    shows whether the update is being pinned to the clip.
    """
    params = [p for p in parameters if p is not None and p.grad is not None]
    if not params:
        return 0.0
    if max_norm is not None:
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))
    total = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in params))
    return float(total)


# ---------------------------------------------------------------------------
# Rollout hooks (BrowserGym imported lazily)
# ---------------------------------------------------------------------------

def _default_decode(text: str, width: int, height: int, instruction: Optional[str]):
    """Decode one model output with the repository's canonical decoder."""
    from agent.environment import decode_action

    return decode_action(text, width, height, instruction)


def _default_format_action(kind: str, value: Any) -> str:
    from agent.dsl import format_action

    return format_action(kind, value)


def _reset_env(env, seed: Optional[int]) -> Mapping[str, Any]:
    if seed is None:
        reset = env.reset()
    else:
        try:
            reset = env.reset(seed=seed)
        except TypeError:  # pragma: no cover - older gym takes no seed
            reset = env.reset()
    if isinstance(reset, tuple) and len(reset) == 2:
        observation, _info = reset
    else:  # pragma: no cover - gym < 0.26 resets to a bare observation
        observation = reset
    return observation


def _step_env(env, command: str):
    outcome = env.step(command)
    if len(outcome) == 5:
        return outcome
    observation, reward, done, info = outcome  # pragma: no cover - legacy gym
    return observation, reward, done, False, info


def _close_env(env) -> None:
    try:
        env.close()
    except Exception:  # pragma: no cover - best-effort cleanup
        pass


def rollout_episode(
    env,
    act: Callable[..., Any],
    *,
    task: Optional[str] = None,
    seed: Optional[int] = None,
    max_steps: int = 6,
    decode: Optional[Callable[..., Tuple[Any, str, Any]]] = None,
    format_action: Optional[Callable[[str, Any], str]] = None,
    reward_penalty: float = 0.05,
    instruction: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one real episode and keep everything GRPO needs to re-score it.

    ``act(observation, goal, history, error)`` must return either the raw model
    text or a mapping whose ``"text"`` key is the raw model text.  Any other
    keys of that mapping (e.g. ``prompt_ids``, ``action_ids``, ``frame``) are
    preserved verbatim under the step record's ``"policy"`` entry, which is how
    ``scripts/train_grpo.py`` later re-scores the exact sampled tokens.

    ``decode(text, width, height, instruction)`` defaults to
    ``agent.environment.decode_action`` and returns
    ``(command, kind, value)``.  A parse failure costs ``reward_penalty`` and
    feeds the error back on the next step; it does not touch the environment.
    ``kind == "stop"`` ends the episode without stepping.

    The returned ``reward`` is the shaped return (environment rewards minus
    parse penalties) used as the GRPO reward; ``env_reward`` is the raw
    environment sum and ``final_reward`` the last raw environment reward.
    """
    decode = decode or _default_decode
    format_action = format_action or _default_format_action
    observation = _reset_env(env, seed)
    goal = instruction or str(observation.get("goal") or task or "")
    history: List[str] = []
    steps: List[Dict[str, Any]] = []
    status = "max_steps"
    total_reward = 0.0
    env_reward = 0.0
    last_reward = 0.0
    terminated = truncated = False
    error = ""

    for index in range(int(max_steps)):
        try:
            output = act(observation, goal, list(history), error)
        except (ValueError, RuntimeError) as exc:
            if "context budget" not in str(exc):
                raise
            status = "context_overflow"
            steps.append({"step": index, "context_error": str(exc)})
            break
        if isinstance(output, str):
            output = {"text": output}
        text = str(output.get("text", ""))
        policy_extra = {key: value for key, value in output.items() if key != "text"}
        record: Dict[str, Any] = {
            "step": index,
            "output": text,
            "policy": policy_extra,
            "history_used": len(history),
        }

        viewport = observation.get("viewport") or {}
        screenshot = observation.get("screenshot")
        height, width = (screenshot.shape[0], screenshot.shape[1]) if screenshot is not None else (600, 800)
        height = int(viewport.get("height", height))
        width = int(viewport.get("width", width))

        try:
            command, kind, value = decode(text, width, height, goal)
        except Exception as exc:  # parse errors are policy feedback
            message = str(exc)[:120]
            record["parse_error"] = message
            record["reward"] = -float(reward_penalty)
            steps.append(record)
            total_reward -= float(reward_penalty)
            error = message
            continue

        record["action"] = format_action(kind, value)
        record["action_kind"] = kind
        if kind == "stop":
            record["stopped"] = value or "done"
            steps.append(record)
            status = "model_stop"
            break

        observation, reward, terminated, truncated, info = _step_env(env, command)
        last_reward = float(reward)
        env_reward += last_reward
        total_reward += last_reward
        record["reward"] = last_reward
        last_error = ""
        if isinstance(info, Mapping):
            last_error = str(info.get("last_action_error") or "")
        record["env_error"] = last_error
        error = last_error[:120]
        history.append(format_action(kind, value))
        steps.append(record)
        if terminated or truncated:
            status = "truncated" if truncated else ("success" if last_reward > 0 else "failed")
            break

    success = bool(status == "success" and last_reward > 0)
    return {
        "task": task or goal,
        "seed": seed,
        "goal": goal,
        "steps": steps,
        "reward": total_reward,
        "env_reward": env_reward,
        "final_reward": last_reward,
        "success": success,
        "status": status,
        "n_steps": len(steps),
    }


def rollout_group(
    env_factory: Callable[[str], Any],
    act: Callable[..., Any],
    task: str,
    seed: Optional[int],
    group_size: int,
    *,
    max_steps: int = 6,
    decode: Optional[Callable[..., Tuple[Any, str, Any]]] = None,
    format_action: Optional[Callable[[str, Any], str]] = None,
    reward_penalty: float = 0.05,
    instruction: Optional[str] = None,
    on_episode: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> List[Dict[str, Any]]:
    """Draw ``group_size`` on-policy episodes for one task+seed (one GRPO group).

    Each episode gets its own environment from ``env_factory(task)`` so a
    BrowserGym browser cannot leak state between group members.  The caller is
    responsible for comparing rewards inside the returned group.
    """
    episodes: List[Dict[str, Any]] = []
    for group_index in range(max(1, int(group_size))):
        env = env_factory(task)
        try:
            episode = rollout_episode(
                env,
                act,
                task=task,
                seed=seed,
                max_steps=max_steps,
                decode=decode,
                format_action=format_action,
                reward_penalty=reward_penalty,
                instruction=instruction,
            )
        finally:
            _close_env(env)
        episode["group_index"] = group_index
        episodes.append(episode)
        if on_episode is not None:
            on_episode(episode)
    return episodes
