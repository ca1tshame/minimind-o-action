"""Shared CPU-test helpers (tiny MiniMind + fake frozen encoder)."""
import torch
from torch import nn

from models.minimind import MiniMindConfig, MiniMindForCausalLM
from agent.vision_tiles import attach_tile_compression
from models.vision import GuiActionModel

TINY = dict(
    vocab_size=128, hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, head_dim=8, intermediate_size=64,
    max_position_embeddings=512, tie_word_embeddings=True, use_moe=False,
    bos_token_id=1, eos_token_id=2, pad_token_id=0,
)


def tiny_lm():
    return MiniMindForCausalLM(MiniMindConfig(**TINY))


def tiny_action_model(vision_tokens=320):
    lm = tiny_lm()
    model = GuiActionModel(lm, in_dim=768)
    attach_tile_compression(model, vision_tokens)
    model.eval()
    return model


class FakeEncoder(nn.Module):
    """Returns a fixed (5, 64, 768) feature map; never needs SigLIP2 weights."""

    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros(1))

    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        return torch.randn(batch, 64, 768) * 0.02 + self.bias


class CharTokenizer:
    """Deterministic stand-in tokenizer: one token per character."""
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.vocab = {"<bos>": 1, "<eos>": 2}
        self.inverse = {1: "<bos>", 2: "<eos>"}

    def encode(self, text, add_special_tokens=False):
        ids = []
        for char in text:
            if char not in self.vocab:
                index = len(self.vocab)
                self.vocab[char] = index
                self.inverse[index] = char
            ids.append(self.vocab[char])
        return ids

    def decode(self, ids):
        return "".join(self.inverse.get(i, "?") for i in ids)

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}
