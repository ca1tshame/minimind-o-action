import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agent.environment import decode_action, run_episode

from tests.helpers import CharTokenizer

SCREEN = np.zeros((360, 480, 3), dtype=np.uint8)


class FakeEnv:
    """Two-step MiniWoB stand-in: step 1 fails execution, step 2 succeeds."""

    def __init__(self, fail_first=True):
        self.fail_first = fail_first
        self.calls = 0
        self.closed = False

    def reset(self, seed=None):
        return {"goal": "click the submit button", "screenshot": SCREEN,
                "viewport": {"width": 480, "height": 360}}, {}

    def step(self, command):
        self.calls += 1
        terminated = self.calls >= 2
        info = {"last_action_error": "element not found" if self.calls == 1 else ""}
        return ({"goal": "click the submit button", "screenshot": SCREEN,
                 "viewport": {"width": 480, "height": 360}},
                1.0 if terminated else 0.0, terminated, False, info)

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.tokenizer = CharTokenizer()
        self.prompts = []

    def predict(self, screen, prompt):
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class DecodeTest(unittest.TestCase):
    def test_scroll_command_matches_v14_semantics(self):
        command, kind, value = decode_action(
            "<action>scroll(0.0000,-0.9000)</action>", 480, 360)
        self.assertEqual(kind, "scroll")
        self.assertEqual(command, "scroll_at(240,180,0,324)")

    def test_stop_returns_no_command(self):
        command, kind, _ = decode_action("<action>stop()</action>", 480, 360)
        self.assertIsNone(command)
        self.assertEqual(kind, "stop")


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.cfg = {"eval": {"max_steps": 4, "max_new": 8, "context": 8192,
                             "hist_used": 6, "vision_tokens": 8}}

    def test_success_trace_and_error_feedback(self):
        policy = FakePolicy(["<action>click(0.5,0.5)</action>",
                             "<action>click(0.6,0.6)</action>"])
        env = FakeEnv()
        with mock.patch("agent.environment.make_env", return_value=env):
            trace = run_episode(policy, "click-test", self.cfg, seed=20001,
                                vision_tokens=8, context=8192, max_new_tokens=8)
        self.assertTrue(trace["success"])
        self.assertEqual(trace["final_reward"], 1.0)
        self.assertEqual(trace["reward"], 1.0)
        self.assertEqual(trace["status"], "success")
        self.assertEqual(trace["num_steps"], 2)
        self.assertIn("Last action failed: element not found", policy.prompts[1])
        self.assertEqual(trace["steps"][0]["env_error"], "element not found")
        self.assertEqual(trace["steps"][0]["action_kind"], "click")

    def test_model_stop_terminates_the_episode(self):
        policy = FakePolicy(["<action>stop(done)</action>"])
        env = FakeEnv()
        with mock.patch("agent.environment.make_env", return_value=env):
            trace = run_episode(policy, "click-test", self.cfg, seed=20001,
                                vision_tokens=8, context=8192, max_new_tokens=8)
        self.assertEqual(trace["status"], "model_stop")
        self.assertFalse(trace["success"])
        self.assertEqual(env.calls, 0)
        self.assertEqual(trace["num_steps"], 1)

    def test_parse_failure_is_fed_back_without_stepping(self):
        policy = FakePolicy(["<action>click(2.0,0.5)</action>",
                             "<action>click(0.5,0.5)</action>",
                             "<action>click(0.6,0.6)</action>"])
        env = FakeEnv()
        with mock.patch("agent.environment.make_env", return_value=env):
            trace = run_episode(policy, "click-test", self.cfg, seed=20001,
                                vision_tokens=8, context=8192, max_new_tokens=8)
        self.assertIn("parse_error", trace["steps"][0])
        self.assertEqual(env.calls, 2)
        self.assertTrue(trace["success"])

    def test_record_path_writes_one_jsonl_trace(self):
        policy = FakePolicy(["<action>stop()</action>"])
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "episodes.jsonl"
            with mock.patch("agent.environment.make_env", return_value=env):
                run_episode(policy, "click-test", self.cfg, seed=20001,
                            vision_tokens=8, context=8192, max_new_tokens=8,
                            record_path=path)
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 1)
        trace = json.loads(lines[0])
        self.assertEqual(trace["status"], "model_stop")


if __name__ == "__main__":
    unittest.main()
