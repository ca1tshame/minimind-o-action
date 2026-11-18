import unittest

import numpy as np

from agent.ctx_pack import ContextOverflow, build_user_prompt
from scripts.inference import GUIPolicy

from tests.helpers import CharTokenizer, FakeEncoder, tiny_action_model

IMAGE = 12


class MarkerTokenizer(CharTokenizer):
    """Counts ``<|image_pad|>`` as exactly one token (id 12).

    Ordinary characters start at id 100 so they can never collide with the
    image marker or the eos/pad ids.
    """

    def __init__(self):
        super().__init__()
        self.inverse = {1: "<bos>", 2: "<eos>"}
        self._next = 3

    def _char(self, char):
        if char not in self.vocab:
            while self._next in (0, 2, 12):
                self._next += 1
            self.vocab[char] = self._next
            self.inverse[self._next] = char
            self._next += 1
        return self.vocab[char]

    def encode(self, text, add_special_tokens=False):
        parts = text.split("<|image_pad|>")
        ids = []
        for index, part in enumerate(parts):
            if index:
                ids.append(IMAGE)
            ids.extend(self._char(char) for char in part)
        return ids

    def decode(self, ids):
        return "".join(self.inverse.get(int(i), "?") for i in ids)


class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.model = tiny_action_model()
        self.policy = GUIPolicy(self.model, FakeEncoder(), MarkerTokenizer(),
                                device="cpu", context=4096, max_new=4,
                                dtype=None, max_position=4096)
        self.screen = np.random.randint(0, 255, (360, 480, 3), dtype=np.uint8)

    def test_predict_accepts_tensor_encoder_output(self):
        prompt = build_user_prompt("click the button", 320)
        text = self.policy.predict(self.screen, prompt)
        self.assertIsInstance(text, str)

    def test_budget_is_checked_before_the_forward_pass(self):
        prompt = build_user_prompt("g" * 5000, 320)
        with self.assertRaises(ContextOverflow):
            self.policy.predict(self.screen, prompt)

    def test_prompt_marker_count_must_match_vision_tokens(self):
        prompt = build_user_prompt("click", 64)
        with self.assertRaises(Exception):
            self.policy.predict(self.screen, prompt)


if __name__ == "__main__":
    unittest.main()
