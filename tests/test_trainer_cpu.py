import tempfile
import unittest
from pathlib import Path

import torch

from scripts.lib.checkpoint import load_checkpoint, save_checkpoint
from scripts.train_gui_sft import (cosine_schedule, forward_gui, forward_language,
                                   pad_gui_batch)

from tests.helpers import FakeEncoder, tiny_action_model


def gui_batch(kinds=("click", "type")):
    batch = []
    for index, kind in enumerate(kinds):
        length = 12 + index
        ids = [1, 5, 6, 7] + [12] * 320 + [2]
        labels = [-100] * (len(ids) - 4) + [8, 9, 10, 2]
        batch.append({"input_ids": ids, "labels": labels,
                      "tiles": torch.zeros(5, 3, 256, 256), "kind": kind})
    return pad_gui_batch(batch)


class TrainerMathTest(unittest.TestCase):
    def test_pad_batch_shapes(self):
        batch = gui_batch()
        self.assertEqual(batch["input_ids"].shape, (2, 325))
        self.assertEqual(batch["mask"][0, :325].sum().item(), 325)
        self.assertEqual(batch["kinds"], ["click", "type"])

    def test_forward_gui_backward_reaches_projector(self):
        model = tiny_action_model()
        model.vision_encoder = FakeEncoder()
        model.train()
        batch = gui_batch()
        loss, kinds = forward_gui(model, batch, "cpu")
        self.assertEqual(kinds, ["click", "type"])
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertIsNotNone(model.proj.mlp[1].weight.grad)
        self.assertGreater(model.proj.mlp[1].weight.grad.abs().sum().item(), 0)

    def test_forward_language_is_token_mean(self):
        model = tiny_action_model()
        model.vision_encoder = FakeEncoder()
        model.train()
        batch = {
            "input_ids": torch.tensor([[1, 5, 6, 7, 2], [1, 8, 9, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]),
            "labels": torch.tensor([[-100, 5, 6, 7, 2], [-100, 8, 9, 2, -100]]),
        }
        loss = forward_language(model, batch, "cpu")
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))

    def test_zero_grad_at_real_accumulation_boundary(self):
        model = tiny_action_model()
        model.vision_encoder = FakeEncoder()
        model.train()
        opt = torch.optim.AdamW(model.proj.parameters(), lr=1e-3)
        batch = gui_batch()
        opt.zero_grad(set_to_none=True)
        for _ in range(2):
            loss, _ = forward_gui(model, batch, "cpu")
            (loss.mean() / 2).backward()
        grad_once = model.proj.mlp[1].weight.grad.abs().sum().item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        loss, _ = forward_gui(model, batch, "cpu")
        (loss.mean() / 2).backward()
        grad_after_reset = model.proj.mlp[1].weight.grad.abs().sum().item()
        self.assertGreater(grad_once, 0)
        self.assertGreater(grad_after_reset, 0)
        self.assertLess(grad_after_reset, grad_once * 1.5)

    def test_cosine_schedule_warmup_then_decay(self):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        sched = cosine_schedule(opt, warmup=10, total=100)
        lrs = []
        for step in range(100):
            lrs.append(opt.param_groups[0]["lr"])
            opt.step()
            sched.step()
        self.assertAlmostEqual(lrs[0], 0.1, places=4)
        self.assertAlmostEqual(max(lrs), 1.0, places=4)
        self.assertLess(lrs[-1], lrs[50])


class CheckpointTest(unittest.TestCase):
    CONTRACT = {"context": 512, "vision_tokens": 320, "layout": [64, 64, 64, 64, 64]}

    def _progress(self):
        return {"step": 3, "epoch": 0, "next_sample": 24, "supervised_tokens": 100,
                "truncated_samples": 0, "order": [0, 1], "accumulation_boundary": True}

    def test_save_and_resume_roundtrip(self):
        model = tiny_action_model()
        opt = torch.optim.AdamW(model.proj.parameters(), lr=1e-4)
        sched = cosine_schedule(opt, 2, 10)
        with tempfile.TemporaryDirectory() as tmp:
            dest = save_checkpoint(tmp, "step_0003", model, opt, sched,
                                   self._progress(), self.CONTRACT)
            self.assertTrue((Path(dest) / "READY.json").is_file())
            fresh = tiny_action_model()
            opt2 = torch.optim.AdamW(fresh.proj.parameters(), lr=1e-4)
            sched2 = cosine_schedule(opt2, 2, 10)
            progress = load_checkpoint(tmp, "step_0003", fresh, opt2, sched2,
                                       self.CONTRACT)
            self.assertEqual(progress["step"], 3)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, fresh.state_dict()[key]), key)

    def test_save_requires_accumulation_boundary(self):
        model = tiny_action_model()
        opt = torch.optim.AdamW(model.proj.parameters(), lr=1e-4)
        sched = cosine_schedule(opt, 2, 10)
        progress = self._progress()
        progress["accumulation_boundary"] = False
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_checkpoint(tmp, "bad", model, opt, sched, progress, self.CONTRACT)

    def test_contract_mismatch_is_rejected(self):
        model = tiny_action_model()
        opt = torch.optim.AdamW(model.proj.parameters(), lr=1e-4)
        sched = cosine_schedule(opt, 2, 10)
        with tempfile.TemporaryDirectory() as tmp:
            save_checkpoint(tmp, "step_0003", model, opt, sched,
                            self._progress(), self.CONTRACT)
            with self.assertRaises(ValueError):
                load_checkpoint(tmp, "step_0003", tiny_action_model(), opt, sched,
                                {"context": 1024})


if __name__ == "__main__":
    unittest.main()
