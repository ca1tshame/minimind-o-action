import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from agent.vision_tiles import attach_tile_compression
from models.minimind import MiniMindConfig, MiniMindForCausalLM
from models.vision import (ActionWeightError, GuiActionModel, load_action_model,
                           remap_action_keys)

from tests.helpers import TINY, tiny_action_model


class ParameterCountTest(unittest.TestCase):
    def test_language_backbone_is_63_912_192(self):
        cfg = json.loads(Path("configs/model_config.json").read_text(encoding="utf-8"))
        cfg["flash_attn"] = False
        model = MiniMindForCausalLM(MiniMindConfig(**cfg))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 63_912_192)


class ProjectorKeyTest(unittest.TestCase):
    def test_legacy_proj_net_keys_are_remapped(self):
        mapping = {"proj.net.2.weight": 1, "proj.mlp.0.weight": 2,
                   "_orig_mod.compressor.score.0.weight": 3, "lm.lm_head.weight": 4}
        mapped, renamed = remap_action_keys(mapping)
        self.assertEqual(renamed, 2)
        self.assertEqual(mapped["proj.mlp.2.weight"], 1)
        self.assertEqual(mapped["compressor.score.0.weight"], 3)
        self.assertIn("lm.lm_head.weight", mapped)

    def test_roundtrip_through_legacy_keys(self):
        model = tiny_action_model()
        state = {k.replace("proj.mlp.", "proj.net."): v.detach().clone()
                 for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.safetensors"
            save_file(state, str(path))
            target = tiny_action_model()
            report = load_action_model(target, str(path))
            self.assertGreater(report["renamed_keys"], 0)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, target.state_dict()[key]), key)


class AttachOrderTest(unittest.TestCase):
    def test_attach_twice_is_an_error(self):
        model = tiny_action_model()
        with self.assertRaises(ValueError):
            attach_tile_compression(model, 320)

    def test_missing_compressor_in_checkpoint_is_fatal(self):
        model = tiny_action_model()
        state = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if not k.startswith(("compressor.", "tile_pos."))}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.safetensors"
            save_file(state, str(path))
            with self.assertRaises(ActionWeightError):
                load_action_model(tiny_action_model(), str(path))

    def test_load_before_attach_is_fatal(self):
        model = GuiActionModel(MiniMindForCausalLM(MiniMindConfig(**TINY)), in_dim=768)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ckpt.safetensors"
            save_file(state, str(path))
            with self.assertRaises(ActionWeightError):
                load_action_model(model, str(path))


class TileInjectionTest(unittest.TestCase):
    def test_320_markers_are_replaced_without_changing_shape(self):
        model = tiny_action_model()
        ids = [1, 5, 6] + [12] * 320 + [2]
        tokens = torch.tensor([ids])
        mask = torch.ones_like(tokens)
        hidden = model.lm.model.embed_tokens(tokens)
        features = torch.randn(320, 768) * 0.02
        out = model.inject(tokens, hidden, mask, [[features]])
        self.assertEqual(out.shape, hidden.shape)
        self.assertFalse(torch.allclose(out[:, 3:323], hidden[:, 3:323]))
        self.assertTrue(torch.allclose(out[:, :3], hidden[:, :3]))
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(model.proj.mlp[0].weight.grad)
        self.assertIsNotNone(model.tile_pos.weight.grad)
        # At 320 tokens every tile keeps all 64 patches, so the compressor is
        # deliberately bypassed (it serves the 160/128/64 layouts).
        self.assertIsNone(model.compressor.score[1].weight.grad)


if __name__ == "__main__":
    unittest.main()


class TiedEmbeddingTest(unittest.TestCase):
    """A V14 checkpoint stores only lm_head.weight; embed_tokens is tied.

    Loading must succeed and the two tensors must still be the same object,
    otherwise the model would run with a randomly initialised embedding.
    """

    def test_embedding_only_in_checkpoint_is_not_missing(self):
        model = tiny_action_model()
        # emulate a real checkpoint: embeddings are tied, so only lm_head is saved
        state = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if k != "lm.model.embed_tokens.weight"}
        self.assertIn("lm.lm_head.weight", state)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            save_file(state, str(path))
            report = load_action_model(model, path)
        missing = [k for k in report["missing_keys"] if "embed_tokens" not in k]
        self.assertEqual(missing, [])
        self.assertIs(model.lm.model.embed_tokens.weight, model.lm.lm_head.weight)

    def test_diverged_tie_is_an_error(self):
        model = tiny_action_model()
        # break the tie: now a checkpoint missing embed_tokens must be rejected
        model.lm.model.embed_tokens.weight = torch.nn.Parameter(
            model.lm.model.embed_tokens.weight.detach().clone()
        )
        state = {k: v.detach().clone() for k, v in model.state_dict().items()
                 if k != "lm.model.embed_tokens.weight"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.safetensors"
            save_file(state, str(path))
            with self.assertRaises(ActionWeightError):
                load_action_model(model, path)
