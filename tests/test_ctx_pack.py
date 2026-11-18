import unittest

from agent.ctx_pack import ContextOverflow, assemble_prompt, compact_history

from tests.helpers import CharTokenizer

HISTORY = ["<action>click(0.1000,0.2000)</action>",
           "<action>type(\"hi\")</action>",
           '<action>scroll(0.0000,-0.9000)</action>',
           "<action>key(Enter)</action>",
           "<action>click(0.3000,0.4000)</action>",
           "<action>click(0.5000,0.6000)</action>",
           "<action>click(0.7000,0.8000)</action>"]


class CtxPackTest(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer()

    def test_keeps_last_hist_used_verbatim(self):
        packed = assemble_prompt(self.tok, "do the thing", 8, HISTORY, 4096, 32,
                                 hist_used=3)
        self.assertEqual(packed.history_used, 3)
        self.assertIn(HISTORY[-1], packed.prompt)
        self.assertNotIn(HISTORY[0], packed.prompt)
        self.assertLessEqual(packed.input_tokens + packed.max_new_tokens, packed.context)

    def test_history_is_stubbed_then_dropped_under_pressure(self):
        packed = assemble_prompt(self.tok, "g" * 40, 8, HISTORY, 420, 20,
                                 hist_used=7, min_history=1)
        self.assertLessEqual(packed.input_tokens + 20, 420)
        self.assertGreater(packed.history_stubbed + packed.history_dropped, 0)
        self.assertIn(HISTORY[-1], packed.prompt)

    def test_overflow_raises_before_model_call(self):
        with self.assertRaises(ContextOverflow):
            assemble_prompt(self.tok, "g" * 400, 32, [], 128, 32, hist_used=6)

    def test_error_feedback_counts_against_budget(self):
        without = assemble_prompt(self.tok, "goal", 8, [], 600, 16)
        with_error = assemble_prompt(self.tok, "goal", 8, [], 600, 16,
                                     error="click coordinate outside viewport")
        self.assertGreater(with_error.input_tokens, without.input_tokens)

    def test_compact_history_reports_dropped(self):
        text, kept, dropped = compact_history(HISTORY, self.tok, 4096, reserve_out=32,
                                              kind_only_after=2)
        self.assertIn("Previous actions:", text)
        self.assertIn(HISTORY[-1], text)
        # The budget is tight: compact_history stops as soon as it fits and
        # reports the rest as dropped (stubs still count as kept entries).
        self.assertLess(len(kept), len(HISTORY))
        self.assertEqual(len(kept) + dropped, len(HISTORY))


if __name__ == "__main__":
    unittest.main()
