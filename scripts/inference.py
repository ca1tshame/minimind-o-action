"""Single-step inference adapter for the trained GUI policy.

Wraps a loaded checkpoint behind a ``predict(screen, prompt) -> text`` interface
consumed by agent.environment. No model weights or datasets are downloaded here;
all paths resolve from configuration.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs import load_config
from agent.vision_tiles import attach_tile_compression, preprocess_tiles
from models import GuiActionModel, MiniMindConfig, MiniMindForCausalLM
from models.vision import FrozenSigLIP2, lm_hidden_forward
from safetensors.torch import load_model


class GUIPolicy:
    """Greedy autoregressive policy over the vision-language action model."""

    def __init__(self, model, tokenizer, device, vision_tokens, max_new, context):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.vision_tokens = vision_tokens
        self.max_new = max_new
        self.context = context

    def predict(self, screen_rgb, prompt):
        from io import BytesIO

        model = self.model
        ids = [self.tokenizer.bos_token_id] + self.tokenizer(
            prompt, add_special_tokens=False
        )["input_ids"]
        if len(ids) + self.max_new > self.context:
            raise ValueError("prompt plus generation exceeds context budget")
        x = torch.tensor([ids], device=self.device)
        mask = torch.ones_like(x)

        buffer = BytesIO()
        Image.fromarray(screen_rgb).convert("RGB").save(buffer, format="PNG")
        from PIL import Image as PILImage
        pixels, _ = preprocess_tiles(PILImage.open(BytesIO(buffer.getvalue())).convert("RGB"),
                                     model.vision_preprocess)
        pixels = pixels.to(self.device)
        with torch.inference_mode():
            visual = model.vision_encoder(pixel_values=pixels).last_hidden_state
            visual = visual.reshape(1, 320, visual.shape[-1]).float()
            hidden = model.inject(x, model.lm.model.embed_tokens(x), mask, [[visual[0]]])
            output = []
            for _ in range(self.max_new):
                logits = lm_hidden_forward(model.lm, hidden, mask)
                next_id = int(logits[0, -1].float().argmax())
                if next_id in (self.tokenizer.eos_token_id, 2):
                    break
                output.append(next_id)
                x = torch.cat([x, torch.tensor([[next_id]], device=self.device)], 1)
                mask = torch.ones_like(x)
                hidden = torch.cat([hidden, model.lm.model.embed_tokens(x[:, -1:])], 1)
        return self.tokenizer.decode(output, skip_special_tokens=False)


def build_predictor(cfg):
    """Construct a GUIPolicy from a loaded checkpoint described by config."""
    paths = cfg["paths"]
    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["tokenizer_dir"]), local_files_only=True, trust_remote_code=False
    )
    model_cfg = json.loads((ROOT / paths["model_config"]).read_text(encoding="utf8"))
    model_cfg["max_position_embeddings"] = cfg["eval"]["context"] + 128
    lm = MiniMindForCausalLM(MiniMindConfig(**model_cfg))
    model = GuiActionModel(lm)
    object.__setattr__(model, "vision_encoder", FrozenSigLIP2(paths["vision_encoder_dir"]))
    object.__setattr__(model, "vision_preprocess", cfg["vision"].get("preprocess", {
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
    }))
    load_model(model, str(paths["checkpoint"]), strict=False, device="cpu")
    attach_tile_compression(model, cfg["eval"]["vision_tokens"])
    device = cfg["device"]["device"]
    model.to(device).eval()
    model.vision_encoder.to(device).eval()
    return GUIPolicy(
        model, tokenizer, device,
        cfg["eval"]["vision_tokens"],
        cfg["inference"]["max_new"],
        cfg["eval"]["context"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Single-step model smoke check.")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--vision-encoder-dir")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, {
        "paths.checkpoint": args.checkpoint,
        "paths.tokenizer_dir": args.tokenizer_dir,
        "paths.vision_encoder_dir": args.vision_encoder_dir,
    })
    policy = build_predictor(cfg)
    print("PREDICTOR_READY", "context", policy.context, "vision_tokens", policy.vision_tokens)


if __name__ == "__main__":
    main()
