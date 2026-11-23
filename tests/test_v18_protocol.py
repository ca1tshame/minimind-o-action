import json
import unittest
from pathlib import Path

import torch
from PIL import Image

from agent.ctx_pack import (
    assemble_prompt,
    build_user_prompt,
    format_screen_tag,
    resolve_screen_wh,
)
from agent.vision_tiles import (
    TILE_NAMES_FOCUS,
    attach_tile_compression,
    focus_box_from_xy,
    next_focus_xy,
    preprocess_tiles,
    split_tiles,
    tokens_per_tile,
)
from models.minimind import MiniMindConfig, MiniMindForCausalLM
from models.vision import GuiActionModel, lm_hidden_forward, load_lm_checkpoint
from safetensors.torch import save_model
import tempfile
from scripts.inference import GUIPolicy
from tests.helpers import TINY, CharTokenizer, FakeEncoder, tiny_action_model
from tests.test_inference_cpu import MarkerTokenizer


class _SpecialAwareTokenizer(CharTokenizer):
    """Treats <|image_pad|> and chat markers as single tokens, matching MiniMind."""

    def encode(self, text, add_special_tokens=False):
        parts = text.replace("<|image_pad|>", "\x01").replace("<|vision_end|>", "\x02")
        parts = parts.replace("<|im_start|>", "\x03").replace("<|im_end|>", "\x04")
        return super().encode(parts, add_special_tokens=add_special_tokens)


class ScreenProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tok = _SpecialAwareTokenizer()

    def test_format_and_resolve_screen_wh(self):
        self.assertEqual(format_screen_tag(None), "")
        self.assertEqual(resolve_screen_wh([1100, 820]), (1100, 820))
        self.assertEqual(format_screen_tag((1920, 1080)), "Screen: 1920x1080")
        self.assertEqual(format_screen_tag((800, 900)), "Screen: 800x900")
        with self.assertRaises(ValueError):
            resolve_screen_wh((0, 900))
        with self.assertRaises(ValueError):
            resolve_screen_wh((1920,))

    def test_v14_default_unchanged_and_v18_tag_inserted(self):
        p_v14 = build_user_prompt("Add milk", 8)
        self.assertNotIn("Screen:", p_v14)
        p_v18 = build_user_prompt("Add milk", 8, screen_wh=(1100, 820))
        self.assertIn("Add milk\nScreen: 1100x820\nTask screenshot follows.", p_v18)

    def test_assemble_prompt_fits_v18_focus_budget(self):
        hist = [f"<action>click(0.1000,0.200{i})</action>" for i in range(8)]
        packed = assemble_prompt(
            self.tok,
            "Add milk",
            384,
            hist,
            768,
            36,
            hist_used=8,
            screen_wh=(1100, 820),
        )
        self.assertIn("Screen: 1100x820", packed.prompt)
        self.assertLessEqual(packed.input_tokens + packed.max_new_tokens, 768)


class FocusCrop384Test(unittest.TestCase):
    def test_focus_helpers_match_v16_v18_contract(self):
        self.assertEqual(focus_box_from_xy(1920, 1080, 0.5, 0.3), (720, 144, 1200, 504))
        self.assertEqual(next_focus_xy([]), (0.50, 0.30))
        self.assertEqual(
            next_focus_xy(["<action>click(0.3187,0.9556)</action>"]),
            (0.50, 0.30),
        )
        self.assertEqual(
            next_focus_xy(["<action>click(0.4659,0.1679)</action>", '<action>type("abc")</action>']),
            (0.4659, 0.1679),
        )

    def test_layout_384_and_6tile_injection_and_policy_predict(self):
        self.assertEqual(tokens_per_tile(384), (64,) * 6)
        self.assertEqual(len(TILE_NAMES_FOCUS), 6)
        img = Image.new("RGB", (1100, 820), color=(20, 40, 60))
        tiles, boxes = split_tiles(img, focus_box=(100, 100, 600, 500))
        self.assertEqual(len(tiles), 6)
        self.assertEqual(boxes[-1], (100, 100, 600, 500))
        pix, _ = preprocess_tiles(
            img,
            {"image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]},
            focus_box=(100, 100, 600, 500),
        )
        self.assertEqual(tuple(pix.shape), (6, 3, 256, 256))

        lm = MiniMindForCausalLM(MiniMindConfig(**TINY))
        model = GuiActionModel(lm, in_dim=768)
        attach_tile_compression(model, 384)
        tokens = torch.tensor([[1, 5] + [12] * 384 + [2]])
        mask = torch.ones_like(tokens)
        hidden = model.lm.model.embed_tokens(tokens)
        feat = torch.randn(384, 768) * 0.02
        out = model.inject(tokens, hidden, mask, [[feat]])
        self.assertEqual(tuple(out.shape), tuple(hidden.shape))

        policy = GUIPolicy(
            tiny_action_model(vision_tokens=384),
            FakeEncoder(),
            MarkerTokenizer(),
            device="cpu",
            context=768,
            max_new=4,
            dtype=None,
            max_position=768,
        )
        prompt = build_user_prompt("Add milk", 384, screen_wh=(1100, 820))
        reply = policy.predict(img, prompt)
        self.assertIsInstance(reply, str)

    def test_load_lm_checkpoint_handles_deduplicated_tied_safetensors(self):
        action = tiny_action_model(vision_tokens=384)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "model.safetensors"
            save_model(action, str(ckpt))
            lm2 = MiniMindForCausalLM(MiniMindConfig(**TINY))
            total = load_lm_checkpoint(lm2, str(ckpt), expected_parameters=sum(p.numel() for p in lm2.parameters()))
            self.assertEqual(total, sum(p.numel() for p in lm2.parameters()))

    def test_cpu_flash_no_nan_and_kv_cache_parity(self):
        cfg = dict(TINY)
        cfg["flash_attn"] = True
        lm = MiniMindForCausalLM(MiniMindConfig(**cfg)).eval()
        hidden = torch.randn(1, 12, cfg["hidden_size"])
        mask = torch.ones(1, 12, dtype=torch.long)
        with torch.no_grad():
            full_logits = lm_hidden_forward(lm, hidden, mask)
            self.assertTrue(torch.isfinite(full_logits).all())
            prefill, kvs = lm_hidden_forward(lm, hidden[:, :11, :], mask[:, :11], start=0, use_cache=True)
            step_logits, _ = lm_hidden_forward(lm, hidden[:, 11:12, :], None, start=11, past_key_values=kvs, use_cache=True)
            self.assertTrue(torch.allclose(full_logits[:, -1:, :], step_logits, atol=1e-4, rtol=1e-4))


class V18SplitsDisjointTest(unittest.TestCase):
    def test_splits_exist_and_frozen_disjoint(self):
        splits = json.loads(Path("configs/v18_splits.json").read_text(encoding="utf-8"))
        frz = set(splits["frozen_20"]["tasks"])
        self.assertEqual(len(frz), 20)
        self.assertEqual(len(splits["frozen_20"]["frozen_12_subset_for_800x900"]), 12)
        for key in ("historical_dev_12", "diagnostic_dev_15", "ood_dev_24", "train_gen"):
            self.assertEqual(frz & set(splits[key]["tasks"]), set(), key)
        tr = set(splits["train_gen"]["tasks"])
        for key in ("historical_dev_12", "diagnostic_dev_15", "ood_dev_24", "frozen_20"):
            self.assertEqual(tr & set(splits[key]["tasks"]), set(), key)


if __name__ == "__main__":
    unittest.main()
