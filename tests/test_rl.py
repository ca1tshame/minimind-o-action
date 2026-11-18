"""CPU-only unit tests for ``scripts/lib/rl.py`` (V12/V13 GRPO math port).

These tests are deliberately pure in-memory: they never touch BrowserGym, the
tokenizer, or the filesystem, so they run in well under five seconds on CPU.
They cover the math that the V13 logging bug corrupted:

* group-relative advantages, including all-equal groups (zero variance) and
  tiny groups, where a naive ``(r - mean) / std`` would divide by zero or
  explode;
* the entropy normalisation regression: ``sequence_logprobs`` returns the mean
  per *action token* entropy, and the V13 monitor divided it by the token count
  a second time (0.026-0.068 instead of the 0.60-0.72 range of V12);
* the sampled-token KL estimator, which must be exactly zero for identical
  policies;
* the REINFORCE/PPO policy-gradient loss sign and magnitude;
* the rollout hooks with a fake environment (no BrowserGym import).
"""

from __future__ import annotations

import math
import unittest

import torch

from scripts.lib import rl


class GroupAdvantageTests(unittest.TestCase):
    def test_known_values_within_group(self):
        rewards = [0.0, 1.0, 2.0]
        advantages, stats = rl.group_relative_advantage(rewards, [7, 7, 7])
        expected_std = math.sqrt(2.0 / 3.0)
        for value, advantage in zip(rewards, advantages):
            self.assertAlmostEqual(advantage, (value - 1.0) / expected_std, places=7)
        self.assertEqual(stats["num_groups"], 1)
        self.assertEqual(stats["num_items"], 3)
        self.assertEqual(stats["zero_variance_groups"], 0)
        self.assertAlmostEqual(stats["reward_mean"], 1.0, places=7)
        self.assertEqual(stats["group_sizes"], [3])

    def test_all_equal_group_is_zero_and_std_safe(self):
        advantages, stats = rl.group_relative_advantage([1.0, 1.0, 1.0], ["a", "a", "a"])
        self.assertEqual(advantages, [0.0, 0.0, 0.0])
        self.assertEqual(stats["zero_variance_groups"], 1)
        self.assertEqual(stats["zero_variance_group_fraction"], 1.0)
        self.assertEqual(stats["advantage_std"], 0.0)
        self.assertTrue(math.isfinite(stats["reward_std"]))

    def test_zero_variance_group_mixed_with_varying_group(self):
        # First group is all-zero (no learning signal); second group is [0, 1].
        rewards = [0.0, 0.0, 0.0, 0.0, 1.0]
        groups = [0, 0, 0, 1, 1]
        advantages, stats = rl.group_relative_advantage(rewards, groups)
        self.assertEqual(advantages[:3], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(advantages[3], -1.0, places=7)
        self.assertAlmostEqual(advantages[4], 1.0, places=7)
        self.assertEqual(stats["num_groups"], 2)
        self.assertEqual(stats["zero_variance_groups"], 1)
        self.assertAlmostEqual(stats["zero_variance_group_fraction"], 0.5, places=7)

    def test_singleton_group_does_not_divide_by_zero(self):
        advantages, stats = rl.group_relative_advantage([5.0], [0])
        self.assertEqual(advantages, [0.0])
        self.assertEqual(stats["zero_variance_groups"], 1)
        self.assertEqual(stats["zero_variance_group_fraction"], 1.0)
        self.assertTrue(math.isfinite(stats["advantage_std"]))

    def test_tiny_group_advantages_are_bounded(self):
        # A group of n members cannot produce |advantage| > sqrt(n - 1).
        for group_size in (1, 2, 3, 5):
            rewards = [float(index) for index in range(group_size)]
            advantages, _stats = rl.group_relative_advantage(
                rewards, [0] * group_size
            )
            bound = math.sqrt(max(group_size - 1, 1)) + 1e-9
            for advantage in advantages:
                self.assertLessEqual(abs(advantage), bound)

    def test_tiny_reward_differences_do_not_explode(self):
        # Differences below eps are treated as a zero-variance group, not
        # amplified into a huge advantage.
        advantages, stats = rl.group_relative_advantage(
            [1.0, 1.0 + 1e-13], [0, 0]
        )
        self.assertEqual(advantages, [0.0, 0.0])
        self.assertEqual(stats["zero_variance_groups"], 1)

    def test_empty_input_is_defined(self):
        advantages, stats = rl.group_relative_advantage([], [])
        self.assertEqual(advantages, [])
        self.assertEqual(stats["num_groups"], 0)
        self.assertEqual(stats["zero_variance_group_fraction"], 0.0)
        self.assertEqual(stats["advantage_std"], 0.0)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            rl.group_relative_advantage([1.0, 2.0], [0])

    def test_large_rewards_stay_normalised(self):
        advantages, _stats = rl.group_relative_advantage([0.0, 1e6], [0, 0])
        self.assertAlmostEqual(advantages[0], -1.0, places=7)
        self.assertAlmostEqual(advantages[1], 1.0, places=7)


class SequenceLogprobEntropyTests(unittest.TestCase):
    def test_entropy_is_mean_over_action_tokens(self):
        # Position 0 predicts t1 from two equal logits -> entropy ln(2).
        # Position 1 predicts t2 from four equal logits -> entropy ln(4).
        vocab, seq = 4, 3
        logits = torch.zeros(1, seq, vocab)
        logits[0, 0, 2] = -1e9
        logits[0, 0, 3] = -1e9
        input_ids = torch.zeros(1, seq, dtype=torch.long)
        mask = torch.tensor([[False, True, True]])

        sum_logp, entropy, n_tokens = rl.sequence_logprobs(logits, input_ids, mask)

        self.assertEqual(n_tokens, 2)
        expected_entropy = 0.5 * (math.log(2.0) + math.log(4.0))
        self.assertAlmostEqual(float(entropy), expected_entropy, places=5)
        # sum_logp: -ln(2) for the first action token, -ln(4) for the second.
        self.assertAlmostEqual(float(sum_logp), -(math.log(2.0) + math.log(4.0)),
                               places=5)

    def test_regression_second_division_would_differ(self):
        # Uniform logits: entropy is ln(vocab) for every token, independent of
        # sequence length.  The V13 monitor divided by n_action_tokens again,
        # which is exactly the bug this asserts against.
        vocab, seq = 4, 6
        logits = torch.zeros(1, seq, vocab)
        input_ids = torch.zeros(1, seq, dtype=torch.long)
        mask = torch.zeros(1, seq, dtype=torch.bool)
        mask[0, 1:] = True

        _sum_logp, entropy, n_tokens = rl.sequence_logprobs(logits, input_ids, mask)

        self.assertEqual(n_tokens, 5)
        self.assertAlmostEqual(float(entropy), math.log(4.0), places=6)
        # The double-normalised value (what V13 logged) is ln(4)/5 ~ 0.277,
        # which is clearly different: it must never be reported as entropy.
        double_normalised = float(entropy) / n_tokens
        self.assertGreater(abs(float(entropy) - double_normalised), 0.2)

    def test_entropy_is_detached_but_logprob_is_differentiable(self):
        logits = torch.zeros(1, 3, 3, requires_grad=True)
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.tensor([[False, True, True]])
        sum_logp, entropy, _n = rl.sequence_logprobs(logits, input_ids, mask)
        self.assertTrue(sum_logp.requires_grad)
        self.assertFalse(entropy.requires_grad)
        sum_logp.sum().backward()
        self.assertIsNotNone(logits.grad)

    def test_empty_action_mask(self):
        logits = torch.zeros(1, 3, 3, requires_grad=True)
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        sum_logp, entropy, n_tokens = rl.sequence_logprobs(logits, input_ids, mask)
        self.assertEqual(n_tokens, 0)
        self.assertEqual(float(sum_logp), 0.0)
        self.assertEqual(float(entropy), 0.0)

    def test_marking_position_zero_is_rejected(self):
        logits = torch.zeros(1, 3, 3)
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        mask = torch.tensor([[True, True, False]])
        with self.assertRaises(ValueError):
            rl.sequence_logprobs(logits, input_ids, mask)

    def test_out_of_vocabulary_token_is_rejected(self):
        logits = torch.zeros(1, 3, 3)
        input_ids = torch.tensor([[0, 9, 0]], dtype=torch.long)
        mask = torch.tensor([[False, True, True]])
        with self.assertRaises(ValueError):
            rl.sequence_logprobs(logits, input_ids, mask)

    def test_wrong_shapes_are_rejected(self):
        with self.assertRaises(ValueError):
            rl.sequence_logprobs(torch.zeros(3), torch.zeros(1, 3, dtype=torch.long),
                                 torch.zeros(1, 3, dtype=torch.bool))
        with self.assertRaises(ValueError):
            rl.sequence_logprobs(torch.zeros(1, 3, 4), torch.zeros(1, 2, dtype=torch.long),
                                 torch.zeros(1, 3, dtype=torch.bool))


class KLLossTests(unittest.TestCase):
    def test_identical_distributions_give_exactly_zero(self):
        logprobs = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        self.assertEqual(float(rl.kl_divergence(logprobs, logprobs)), 0.0)

    def test_sequence_logprobs_kl_is_zero_for_the_same_policy(self):
        logits = torch.randn(2, 5, 7)
        ids = torch.randint(0, 7, (2, 5))
        mask = torch.zeros(2, 5, dtype=torch.bool)
        mask[:, 1:] = True
        sum_logp, _entropy, _n = rl.sequence_logprobs(logits, ids, mask)
        self.assertEqual(float(rl.kl_divergence(sum_logp, sum_logp)), 0.0)

    def test_shifted_policy_is_nonzero(self):
        ref = torch.log(torch.tensor([0.5, 0.5]))
        current = torch.log(torch.tensor([0.9, 0.1]))
        self.assertNotEqual(float(rl.kl_divergence(ref, current)), 0.0)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            rl.kl_divergence(torch.zeros(2), torch.zeros(3))


class PolicyGradientTests(unittest.TestCase):
    def test_sign_and_magnitude(self):
        logprobs = torch.tensor([[-2.0, -0.5]], requires_grad=True)
        advantages = torch.tensor([[1.0, -1.0]])
        loss = rl.policy_gradient_loss(logprobs, advantages)
        # -mean(adv * logp) = -((1 * -2) + (-1 * -0.5)) / 2 = 0.75
        self.assertAlmostEqual(float(loss), 0.75, places=6)
        loss.backward()
        # d loss / d logp = -adv / n: raising the logprob of a positive-advantage
        # token lowers the loss (negative gradient); the opposite holds for the
        # negative-advantage token.
        self.assertAlmostEqual(float(logprobs.grad[0, 0]), -0.5, places=6)
        self.assertAlmostEqual(float(logprobs.grad[0, 1]), 0.5, places=6)

    def test_zero_advantage_gives_zero_loss_and_gradient(self):
        logprobs = torch.tensor([[-1.0, -1.0]], requires_grad=True)
        loss = rl.policy_gradient_loss(logprobs, torch.zeros(1, 2))
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(float(logprobs.grad.abs().sum()), 0.0)

    def test_reduction_options(self):
        logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=False)
        advantages = torch.tensor([[1.0, 1.0]])
        self.assertAlmostEqual(
            float(rl.policy_gradient_loss(logprobs, advantages, reduction="sum")), 3.0, places=6
        )
        per_item = rl.policy_gradient_loss(logprobs, advantages, reduction="none")
        self.assertEqual(tuple(per_item.shape), (1, 2))
        with self.assertRaises(ValueError):
            rl.policy_gradient_loss(logprobs, advantages, reduction="bogus")

    def test_ppo_clip_limits_the_surrogate(self):
        logprobs = torch.tensor([[-0.5]])
        old_logprobs = torch.tensor([[-1.0]])
        advantages = torch.tensor([[1.0]])
        unclipped = rl.policy_gradient_loss(
            logprobs, advantages, old_logprobs=old_logprobs, clip=10.0
        )
        clipped = rl.policy_gradient_loss(
            logprobs, advantages, old_logprobs=old_logprobs, clip=0.2
        )
        ratio = math.exp(0.5)
        self.assertAlmostEqual(float(unclipped), -ratio, places=5)
        self.assertAlmostEqual(float(clipped), -1.2, places=5)
        with self.assertRaises(ValueError):
            rl.policy_gradient_loss(logprobs, advantages, old_logprobs=old_logprobs)


class DiagnosticsTests(unittest.TestCase):
    def test_entropy_stats_weighted_mean(self):
        stats = rl.entropy_stats([1.0, 3.0], weights=[1.0, 3.0])
        self.assertEqual(stats["n"], 2)
        self.assertAlmostEqual(stats["mean"], 2.5, places=7)
        self.assertAlmostEqual(stats["min"], 1.0, places=7)
        self.assertAlmostEqual(stats["max"], 3.0, places=7)

    def test_entropy_stats_empty(self):
        stats = rl.entropy_stats([])
        self.assertEqual(stats["n"], 0)
        self.assertIsNone(stats["mean"])

    def test_grad_norm_clips_and_reports_pre_clip_value(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0, 0.0]))
        parameter.grad = torch.tensor([3.0, 4.0])
        self.assertAlmostEqual(rl.grad_norm([parameter]), 5.0, places=6)
        reported = rl.grad_norm([parameter], max_norm=1.0)
        self.assertAlmostEqual(reported, 5.0, places=6)
        self.assertAlmostEqual(rl.grad_norm([parameter]), 1.0, places=6)

    def test_grad_norm_without_gradients(self):
        parameter = torch.nn.Parameter(torch.zeros(2))
        self.assertEqual(rl.grad_norm([parameter]), 0.0)


class _Frame:
    """Minimal stand-in for a screenshot array (only ``shape`` is used)."""

    shape = (600, 800, 3)


class FakeEnv:
    """Duck-typed BrowserGym env: reset(seed=) / step(command) / close()."""

    def __init__(self, rewards=(0.0, 1.0)):
        self.rewards = list(rewards)
        self.position = 0
        self.closed = False
        self.commands = []

    def _observation(self):
        return {
            "goal": "close the dialog",
            "screenshot": _Frame(),
            "viewport": {"width": 800, "height": 600},
        }

    def reset(self, seed=None):
        self.position = 0
        return self._observation(), {}

    def step(self, command):
        self.commands.append(command)
        reward = float(self.rewards[self.position])
        self.position += 1
        terminated = self.position >= len(self.rewards) and reward > 0
        return self._observation(), reward, terminated, False, {"last_action_error": ""}

    def close(self):
        self.closed = True


def _stub_decode(text, width, height, instruction):
    if text == "stop":
        return None, "stop", "done"
    if text == "bad":
        raise ValueError("invalid_action_syntax")
    return "click(0.5,0.5)", "click", (0.5, 0.5)


def _stub_format(kind, value):
    return "%s:%s" % (kind, value)


class RolloutHookTests(unittest.TestCase):
    def test_episode_keeps_policy_extras_for_rescoring(self):
        seen = []

        def act(observation, goal, history, error):
            seen.append({"goal": goal, "history": list(history), "error": error})
            return {
                "text": "click",
                "prompt_ids": [1, 12, 13],
                "action_ids": [7, 8],
                "frame": "frame-0.png",
            }

        env = FakeEnv(rewards=(0.0, 1.0))
        episode = rl.rollout_episode(
            env, act, task="click-dialog", seed=11, max_steps=3,
            decode=_stub_decode, format_action=_stub_format,
        )

        self.assertEqual(episode["task"], "click-dialog")
        self.assertEqual(episode["seed"], 11)
        self.assertTrue(episode["success"])
        self.assertEqual(episode["status"], "success")
        self.assertEqual(episode["final_reward"], 1.0)
        self.assertEqual(episode["env_reward"], 1.0)
        self.assertEqual(len(episode["steps"]), 2)
        self.assertNotIn("text", episode["steps"][0]["policy"])
        self.assertEqual(episode["steps"][0]["policy"]["frame"], "frame-0.png")
        self.assertEqual(episode["steps"][0]["action"], "click:(0.5, 0.5)")
        self.assertEqual(episode["steps"][0]["action_kind"], "click")
        # History grows by one formatted action per executed step.
        self.assertEqual(seen[0]["history"], [])
        self.assertEqual(len(seen[1]["history"]), 1)
        self.assertEqual(len(env.commands), 2)

    def test_parse_failure_costs_penalty_and_feeds_error(self):
        seen = []

        def act(observation, goal, history, error):
            seen.append(error)
            return {"text": "bad"}

        env = FakeEnv(rewards=(0.0, 0.0))
        episode = rl.rollout_episode(
            env, act, task="click-test", seed=None, max_steps=2,
            decode=_stub_decode, format_action=_stub_format, reward_penalty=0.25,
        )
        self.assertEqual(episode["steps"][0]["parse_error"], "invalid_action_syntax")
        self.assertEqual(episode["steps"][0]["reward"], -0.25)
        self.assertEqual(episode["reward"], -0.5)
        self.assertEqual(env.commands, [])
        self.assertEqual(seen[1], "invalid_action_syntax")

    def test_model_stop_ends_episode_without_stepping(self):
        def act(observation, goal, history, error):
            return {"text": "stop"}

        env = FakeEnv()
        episode = rl.rollout_episode(
            env, act, task="click-test", seed=1, max_steps=4,
            decode=_stub_decode, format_action=_stub_format,
        )
        self.assertEqual(episode["status"], "model_stop")
        self.assertFalse(episode["success"])
        self.assertEqual(env.commands, [])

    def test_context_overflow_is_recorded(self):
        def act(observation, goal, history, error):
            raise ValueError("prompt plus generation exceeds context budget (500 + 24 > 512)")

        env = FakeEnv()
        episode = rl.rollout_episode(
            env, act, task="click-test", seed=1, max_steps=3,
            decode=_stub_decode, format_action=_stub_format,
        )
        self.assertEqual(episode["status"], "context_overflow")
        self.assertEqual(len(episode["steps"]), 1)
        self.assertIn("context budget", episode["steps"][0]["context_error"])

    def test_rollout_group_uses_fresh_envs_and_indexes(self):
        created = []

        def factory(task):
            env = FakeEnv(rewards=(0.0, 1.0))
            env.task = task
            created.append(env)
            return env

        def act(observation, goal, history, error):
            return {"text": "click"}

        group = rl.rollout_group(
            factory, act, "click-dialog", 5, 3, max_steps=2,
            decode=_stub_decode, format_action=_stub_format,
        )
        self.assertEqual(len(group), 3)
        self.assertEqual([episode["group_index"] for episode in group], [0, 1, 2])
        self.assertEqual(len(created), 3)
        self.assertTrue(all(env.closed for env in created))
        self.assertTrue(all(episode["success"] for episode in group))


if __name__ == "__main__":
    unittest.main()
