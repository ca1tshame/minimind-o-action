import os
import tempfile
import unittest
from pathlib import Path

from configs import (apply_overrides, load_config, require_paths, resolve_paths,
                     resolve_device, workspace_root)


class ConfigTest(unittest.TestCase):
    def test_default_config_240_protocol(self):
        cfg = load_config()
        protocol = cfg["eval"]["protocols"]["historical"]
        self.assertEqual(len(protocol["tasks"]), 15)
        self.assertEqual(protocol["seeds"][0], 20001)
        self.assertEqual(protocol["seeds"][-1], 20016)
        self.assertEqual(len(protocol["tasks"]) * len(protocol["seeds"]), 240)
        self.assertEqual(cfg["eval"]["hist_used"], 6)
        self.assertEqual(cfg["eval"]["max_steps"], 8)
        self.assertEqual(cfg["gui_sft"]["vision_tokens"], 320)

    def test_overrides_ignore_none(self):
        cfg = load_config(overrides={"paths.workspace_root": "/tmp/x",
                                     "paths.checkpoint": None})
        self.assertEqual(cfg["paths"]["workspace_root"], "/tmp/x")
        self.assertIsNone(cfg["paths"]["checkpoint"])

    def test_workspace_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config()
            root = workspace_root(cfg, tmp)
            self.assertEqual(root, Path(tmp).resolve())
            paths = resolve_paths({"paths": {"checkpoint": "ckpt/model.safetensors",
                                             "miniwob_url": "file:///a/b"}}, root)
            self.assertEqual(paths["checkpoint"], root / "ckpt" / "model.safetensors")
            self.assertEqual(paths["miniwob_url"], "file:///a/b")

    def test_env_var_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("MINIMIND_WORKSPACE")
            os.environ["MINIMIND_WORKSPACE"] = tmp
            try:
                self.assertEqual(workspace_root(load_config()), Path(tmp).resolve())
            finally:
                if old is None:
                    os.environ.pop("MINIMIND_WORKSPACE", None)
                else:
                    os.environ["MINIMIND_WORKSPACE"] = old

    def test_require_paths_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            require_paths({"checkpoint": None}, ["checkpoint"])
        self.assertIn("checkpoint", str(ctx.exception))

    def test_device_fallback(self):
        self.assertEqual(resolve_device({"device": {"device": "cpu"}}), "cpu")


if __name__ == "__main__":
    unittest.main()
