"""CPU-only tests for the V9 grounding port (unittest; pytest also collects)."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from models.grounding import (
    NUM_TOKENS, GroundingHead, build_targets, decode_points, focal_loss,
    patch_to_token, point_to_token, token_offset_to_point, token_to_patch,
)
from scripts import gui_v9_mix as mix


class _CleanupDir(object):
    """Minimal ``TemporaryDirectory`` stand-in for the fallback path."""

    def __init__(self, path):
        self.name = str(path)
        self._path = path

    def cleanup(self):
        shutil.rmtree(self._path, ignore_errors=True)


def _writable_tempdir():
    """Return a directory handle for the mix round-trip test.

    A per-process directory beside this module is preferred, because on
    restricted Windows hosts a ``tempfile.mkdtemp`` directory can be created
    while every write inside it is denied. A system temp directory is the
    fallback for read-only source trees.
    """
    local = Path(__file__).resolve().parent / ("_mix_tmp_%d" % os.getpid())
    try:
        shutil.rmtree(local, ignore_errors=True)
        local.mkdir()
        return _CleanupDir(local)
    except OSError:
        pass
    return _CleanupDir(Path(tempfile.mkdtemp(prefix="v9_mix_")))


class GroundingHeadTest(unittest.TestCase):
    def test_shapes_and_offset_range(self):
        torch.manual_seed(0)
        head = GroundingHead(dim=16, hidden=8)
        feats = torch.randn(2, NUM_TOKENS, 16)
        instr = torch.randn(2, 16)
        heat, offsets = head(feats, instr)
        self.assertEqual(tuple(heat.shape), (2, NUM_TOKENS))
        self.assertEqual(tuple(offsets.shape), (2, NUM_TOKENS, 2))
        self.assertTrue(bool((offsets >= 0).all()) and bool((offsets <= 1).all()))
        self.assertEqual(float(head.hm.bias[0]), -4.0)
        self.assertEqual(head.dim, 16)
        self.assertEqual(head.hidden, 8)

    def test_shape_validation(self):
        head = GroundingHead(dim=16, hidden=8)
        with self.assertRaises(ValueError):
            head(torch.randn(2, NUM_TOKENS + 1, 16), torch.randn(2, 16))
        with self.assertRaises(ValueError):
            head(torch.randn(2, NUM_TOKENS, 16), torch.randn(1, 16))
        with self.assertRaises(ValueError):
            head(torch.randn(2, NUM_TOKENS, 8), torch.randn(2, 8))

    def test_parameters_receive_gradients(self):
        head = GroundingHead(dim=8, hidden=4)
        feats = torch.randn(1, NUM_TOKENS, 8, requires_grad=True)
        instr = torch.randn(1, 8)
        heat, offsets = head(feats, instr)
        (heat.sum() + offsets.sum()).backward()
        self.assertIsNotNone(feats.grad)
        for name, parameter in head.named_parameters():
            self.assertIsNotNone(parameter.grad, name)


class TargetTest(unittest.TestCase):
    def test_build_targets_known_points(self):
        heat, offsets, mask = build_targets(
            [(5, 0.25, 0.75), (None, 0.0, 0.0), (319, -0.2, 0.9)], "cpu"
        )
        self.assertEqual(tuple(heat.shape), (3, NUM_TOKENS))
        self.assertEqual(tuple(offsets.shape), (3, NUM_TOKENS, 2))
        self.assertEqual(mask.tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(int(heat[0].argmax()), 5)
        self.assertEqual(int(heat[2].argmax()), 319)
        self.assertEqual(float(heat[0, 5]), 1.0)
        self.assertEqual(float(heat[0, patch_to_token(0, 0, 4)]), 0.5)
        self.assertEqual(float(heat[0, patch_to_token(0, 1, 4)]), 0.25)
        self.assertEqual(float(heat[1].abs().sum()), 0.0)
        self.assertEqual(float(offsets[0, 5, 0]), 0.25)
        self.assertEqual(float(offsets[0, 5, 1]), 0.75)
        self.assertAlmostEqual(float(offsets[2, 319, 0]), -0.2, places=5)
        self.assertTrue(bool((offsets.abs() <= 1.0).all()))
        with self.assertRaises(ValueError):
            build_targets([(NUM_TOKENS, 0.0, 0.0)], "cpu")
        with self.assertRaises(ValueError):
            build_targets([(0, 2.0, 0.0)], "cpu")

    def test_point_to_token_round_trip(self):
        for x, y in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0),
                     (0.5, 0.5), (0.25, 0.75), (0.9, 0.1), (0.75, 0.25)]:
            token, ox, oy = point_to_token(x, y)
            self.assertTrue(0 <= token < NUM_TOKENS)
            self.assertTrue(0.0 <= ox < 1.0 and 0.0 <= oy < 1.0)
            rx, ry = token_offset_to_point(token, ox, oy)
            self.assertAlmostEqual(rx, x, delta=3e-3)
            self.assertAlmostEqual(ry, y, delta=3e-3)
        with self.assertRaises(ValueError):
            point_to_token(1.5, 0.5)
        with self.assertRaises(ValueError):
            point_to_token(float("nan"), 0.5)

    def test_token_patch_round_trip(self):
        for token in range(NUM_TOKENS):
            tile, row, col = token_to_patch(token)
            self.assertEqual(patch_to_token(tile, row, col), token)
        with self.assertRaises(ValueError):
            token_to_patch(NUM_TOKENS)

    def test_decode_points(self):
        heat = torch.full((1, NUM_TOKENS), -20.0)
        heat[0, 42] = 20.0
        offsets = torch.zeros(1, NUM_TOKENS, 2)
        points, scores = decode_points(heat, offsets)
        self.assertEqual(points[0][0], token_offset_to_point(42, 0.0, 0.0))
        self.assertGreater(float(scores[0, 0]), 0.9)
        with self.assertRaises(ValueError):
            decode_points(heat[:, :-1], offsets[:, :-1])


class FocalLossTest(unittest.TestCase):
    def test_confident_correct_is_lower(self):
        target = torch.zeros(2, NUM_TOKENS)
        target[0, 10] = 1.0
        target[1, 20] = 1.0
        good = torch.full_like(target, -8.0)
        good[0, 10] = 8.0
        good[1, 20] = 8.0
        bad = torch.full_like(target, -8.0)
        self.assertLess(float(focal_loss(good, target)), float(focal_loss(bad, target)))
        self.assertGreater(float(focal_loss(good, target)), 0.0)
        with self.assertRaises(ValueError):
            focal_loss(good, target[:, :-1])
        with self.assertRaises(ValueError):
            focal_loss(good, -torch.ones_like(target))


class MixTest(unittest.TestCase):
    @staticmethod
    def _base(count):
        return [{"instruction": "click", "frame": "base/%d.jpg" % i} for i in range(count)]

    @staticmethod
    def _ground(count, frame_prefix="frames"):
        rows = []
        for i in range(count):
            rows.append({
                "task": "click-test", "seed": 6201 + i,
                "episode_id": "click-test_%d" % (6201 + i), "step": 0,
                "instruction": "click the button",
                "frame": "%s/%d.jpg" % (frame_prefix, i),
                "target_point": [0.5, 0.5], "target_bbox": [0.4, 0.4, 0.2, 0.2],
                "action_kind": "click",
            })
        return rows

    def test_ratio_math(self):
        self.assertEqual(mix.plan_mix(90, 0.1), 10)
        self.assertEqual(mix.plan_mix(90, 0.5), 90)
        self.assertEqual(mix.plan_mix(0, 0.1), 0)
        with self.assertRaises(ValueError):
            mix.plan_mix(10, 0.0)
        with self.assertRaises(ValueError):
            mix.plan_mix(10, 1.0)

    def test_manifest_counts(self):
        mixed = mix.build_mix(
            self._base(90), [], self._ground(40), 0.1,
            frames_prefix="datasets/gui_v9_ground_v1",
        )
        manifest = mixed["manifest"]
        self.assertEqual(manifest["ground_in_train"], 10)
        self.assertEqual(manifest["total_train"], 100)
        self.assertAlmostEqual(manifest["ratio_achieved"], 0.1, places=4)
        self.assertLessEqual(manifest["ground_in_train"], manifest["ground_train_available"])
        self.assertEqual(manifest["rows_with_target_point"], 10)
        self.assertEqual(manifest["files"]["train"]["rows"], manifest["total_train"])
        self.assertEqual(manifest["files"]["validation"]["rows"], manifest["val_rows"])
        self.assertEqual(len(mixed["train"]), 100)
        ground_train = [r for r in mixed["train"] if r.get("mix_source") == "v9_ground"]
        self.assertTrue(all(r["frame"].startswith("datasets/gui_v9_ground_v1/")
                            for r in ground_train))

    def test_manifest_deterministic_and_order_independent(self):
        base = self._base(90)
        ground = self._ground(30)
        first = mix.build_mix(base, [], ground, 0.2)
        second = mix.build_mix(list(base), [], list(reversed(ground)), 0.2)
        self.assertEqual(first["manifest"], second["manifest"])
        self.assertEqual(mix.dump_rows(first["train"]), mix.dump_rows(second["train"]))
        self.assertEqual(mix.dump_rows(first["validation"]),
                         mix.dump_rows(second["validation"]))

    def test_write_round_trip(self):
        tmp = _writable_tempdir()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        mixed = mix.build_mix(
            self._base(90), self._base(2), self._ground(20), 0.1,
            frames_prefix="datasets/gui_v9_ground_v1",
        )
        manifest = mix.write_mix(root / "out", mixed)
        train = mix.read_jsonl(root / "out" / "train.jsonl")
        validation = mix.read_jsonl(root / "out" / "validation.jsonl")
        self.assertEqual(len(train), manifest["total_train"])
        self.assertEqual(len(validation), manifest["val_rows"])
        on_disk = json.loads((root / "out" / "manifest.json").read_text())
        self.assertEqual(on_disk, manifest)
        self.assertNotIn(str(root), json.dumps(manifest))
        with self.assertRaises(FileExistsError):
            mix.write_mix(root / "out", mixed)

    def test_absolute_paths_refused(self):
        with self.assertRaises(ValueError):
            mix.join_frames(None, "C:/data/frame.jpg")
        with self.assertRaises(ValueError):
            mix.join_frames("datasets", "../frame.jpg")
        with self.assertRaises(ValueError):
            mix.build_mix(self._base(2), [], [
                {"task": "t", "seed": 1, "frame": "/abs/f.jpg"}
            ], 0.9)


if __name__ == "__main__":
    unittest.main()
