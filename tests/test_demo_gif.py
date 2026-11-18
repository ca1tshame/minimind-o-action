import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.record_demo_gif import build_gif


class DemoTraceTests(unittest.TestCase):
    def test_gif_is_built_from_real_frames_and_records_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frames = root / "frames"
            frames.mkdir()
            Image.new("RGB", (64, 36), (10, 20, 30)).save(frames / "step_000.png")
            Image.new("RGB", (64, 36), (30, 20, 10)).save(frames / "step_001.png")
            trace = root / "trace.json"
            trace.write_text(json.dumps({
                "task": "click-dialog", "seed": 20001, "success": True,
                "final_reward": 1.0,
                "steps": [
                    {"step": 0, "action": "<action>click(0.5,0.5)</action>",
                     "reward": 0.0, "env_error": None},
                    {"step": 1, "action": "<action>stop()</action>",
                     "reward": 1.0, "env_error": ""},
                ],
            }), encoding="utf8")
            out = root / "demo.gif"
            meta = build_gif(trace, frames, out)
            self.assertTrue(out.is_file() and out.stat().st_size > 0)
            self.assertEqual(meta["steps"], 2)
            self.assertEqual(meta["final_reward"], 1.0)
            self.assertEqual(meta["trace_sha256"], build_gif(trace, frames, root / "b.gif")["trace_sha256"])
            saved = json.loads(out.with_suffix(".json").read_text(encoding="utf8"))
            self.assertEqual(saved["gif_sha256"], meta["gif_sha256"])
            self.assertIn("real rollout", saved["replay_note"])

    def test_missing_frame_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "trace.json"
            trace.write_text(json.dumps({"steps": [{"step": 3, "action": "x"}]}), encoding="utf8")
            (root / "frames").mkdir()
            with self.assertRaises(SystemExit):
                build_gif(trace, root / "frames", root / "x.gif")


if __name__ == "__main__":
    unittest.main()
