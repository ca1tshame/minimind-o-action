import unittest

from agent.records import RecordError, validate_record
from eval.metrics import summarize_episodes

GOOD = {
    "episode_id": "ep1", "step": 0, "task": "click-dialog", "seed": 20001,
    "instruction": "click the OK button", "frame": None, "frame_sha256": None,
    "history": ["<action>click(0.1,0.2)</action>"],
    "error_feedback": "", "target_action": "<action>click(0.5,0.5)</action>",
    "action_kind": "click", "data_source": "oracle", "split": "train",
    "reward": 1.0, "success": True, "target_point": [0.5, 0.5],
}


class RecordTest(unittest.TestCase):
    def test_valid_click(self):
        out = validate_record(GOOD)
        self.assertEqual(out["action_kind"], "click")
        self.assertEqual(out["target_point"], [0.5, 0.5])

    def test_pointer_type_record(self):
        rec = dict(GOOD, target_action='<action>type(@0:7|lower)</action>',
                   action_kind="type", target_point=None,
                   instruction="HELLO WORLD")
        out = validate_record(rec)
        self.assertEqual(out["target_point"], [0, 7])

    def test_rejects_bad_split_source_kind(self):
        for patch in ({"split": "holdout"}, {"data_source": "unknown"},
                      {"action_kind": "scroll"}):
            with self.assertRaises(RecordError):
                validate_record(dict(GOOD, **patch))

    def test_rejects_invalid_history(self):
        with self.assertRaises(RecordError):
            validate_record(dict(GOOD, history=["not an action"]))

    def test_click_needs_point(self):
        with self.assertRaises(RecordError):
            validate_record(dict(GOOD, target_point=None))


class MetricsTest(unittest.TestCase):
    def test_final_reward_and_legacy_reward_both_count(self):
        summary = summarize_episodes([
            {"task": "a", "success": True, "final_reward": 1.0},
            {"task": "a", "success": False, "reward": 0.0},
        ])
        self.assertEqual(summary["episodes"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["mean_reward"], 0.5)
        self.assertEqual(summary["by_task"]["a"], {"success": 1, "total": 2})


if __name__ == "__main__":
    unittest.main()
