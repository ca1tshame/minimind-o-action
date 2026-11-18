import unittest

from agent.dsl import ActionParseError, extract_action, format_action, parse_action, to_browsergym


class DSLTest(unittest.TestCase):
    def test_click(self):
        self.assertEqual(parse_action("<action>click(0.5000,0.2500)</action>"),
                         ("click", (0.5, 0.25)))
        command = to_browsergym(("click", (0.5, 0.25)), 800, 600)
        self.assertEqual(command, "mouse_click(399,149)")

    def test_click_bounds_and_nan(self):
        for raw in ("<action>click(1.5,0.2)</action>", "<action>click(nan,0.2)</action>",
                    "<action>click(0.1,inf)</action>", "<action>click(0.1)</action>"):
            with self.assertRaises(ActionParseError):
                parse_action(raw)

    def test_scroll_is_displacement_not_coordinates(self):
        kind, (dx, dy) = parse_action("<action>scroll(0.0000,-0.9000)</action>")
        self.assertEqual(kind, "scroll")
        # dy=-0.9 scrolls the page down: wheel deltaY is positive.
        self.assertEqual(to_browsergym((kind, (dx, dy)), 800, 600),
                         "scroll_at(400,300,0,540)")
        with self.assertRaises(ActionParseError):
            parse_action("<action>scroll(0.0,-1.5)</action>")

    def test_type_quoted(self):
        kind, value = parse_action('<action>type("hello world")</action>')
        self.assertEqual((kind, value), ("type", "hello world"))
        self.assertEqual(to_browsergym((kind, value), 800, 600),
                         'keyboard_type("hello world")')
        with self.assertRaises(ActionParseError):
            parse_action("<action>type(hello)</action>")

    def test_type_pointer_unquoted_and_case(self):
        kind, value = parse_action("<action>type(@0:5|upper)</action>")
        self.assertEqual(kind, "type")
        self.assertEqual(str(value), "@0:5|upper")
        self.assertEqual(to_browsergym((kind, value), 800, 600, instruction="hello world"),
                         'keyboard_type("HELLO")')
        kind, value = parse_action("<action>type(@0:5|lower)</action>", "HeLLo")
        self.assertEqual(value, "hello")

    def test_stop_forms_and_termination(self):
        self.assertEqual(parse_action("<action>stop()</action>"), ("stop", ""))
        self.assertEqual(parse_action("<action>stop(done)</action>"), ("stop", "done"))
        self.assertIsNone(to_browsergym(("stop", "done"), 800, 600))
        with self.assertRaises(ActionParseError):
            parse_action("<action>stop(maybe)</action>")

    def test_key(self):
        self.assertEqual(parse_action("<action>key(Enter)</action>"), ("key", "Enter"))
        with self.assertRaises(ActionParseError):
            parse_action("<action>key(F13)</action>")

    def test_think_prefix_and_trailing_prose(self):
        text = "<think>click the button</think><action>click(0.1,0.2)</action>"
        self.assertEqual(parse_action(text), ("click", (0.1, 0.2)))
        self.assertEqual(extract_action("done: <action>stop()</action> ok"),
                         "<action>stop()</action>")

    def test_roundtrip(self):
        for text in ("<action>click(0.1234,0.5678)</action>",
                     "<action>scroll(0.0000,-0.9000)</action>",
                     '<action>type("abc")</action>', "<action>key(Tab)</action>",
                     "<action>stop(failed)</action>"):
            kind, value = parse_action(text)
            self.assertEqual(format_action(kind, value), text)


if __name__ == "__main__":
    unittest.main()


class ScrollSpellingTest(unittest.TestCase):
    """Real V14 checkpoints emit both scroll spellings; both must survive.

    ``scroll(dx,dy)`` is the documented form. Checkpoints trained on data that
    predates it emit ``scroll_at(cx,cy,dx,dy)``. Rejecting the second one makes
    the model look far worse than it is (every scroll step becomes a parse
    error), so the parser normalises both and keeps the V14 sign convention.
    """

    def test_two_arg_scroll(self):
        kind, value = parse_action("<action>scroll(0.0000,-0.9000)</action>")
        self.assertEqual(kind, "scroll")
        self.assertEqual(value, (0.0, -0.9))

    def test_four_arg_scroll_is_accepted_and_keeps_semantics(self):
        text = "<action>scroll_at(0.2410,0.5140,0.0000,0.9000)</action>"
        kind, value = parse_action(text)
        self.assertEqual(kind, "scroll")
        self.assertEqual(value, (0.0, 0.9, 0.2410, 0.5140))
        # dy>0 must still mean "page up" under the V14 convention
        command = to_browsergym((kind, value), 498, 321)
        self.assertEqual(command, "scroll_at(119,164,0,-289)")

    def test_scroll_roundtrip_preserves_the_spelling(self):
        text = "<action>scroll_at(0.2410,0.5140,0.0000,0.9000)</action>"
        kind, value = parse_action(text)
        self.assertEqual(format_action(kind, value), text)

    def test_bad_arity_is_rejected(self):
        for bad in (
            "<action>scroll()</action>",
            "<action>scroll(1,2,3)</action>",
            "<action>scroll(0,0,0,0,0)</action>",
        ):
            with self.assertRaises(ActionParseError):
                parse_action(bad)

    def test_scroll_centre_out_of_range_is_rejected(self):
        with self.assertRaises(ActionParseError):
            parse_action("<action>scroll_at(1.5,0.5,0.0,0.9)</action>")

    def test_scroll_centre_zero_consumes_no_new_tokens(self):
        # sanity: the legacy spelling must not silently shift the delta
        _, value = parse_action("<action>scroll_at(0.1,0.2,0.5,-0.5)</action>")
        self.assertEqual(value, (0.5, -0.5, 0.1, 0.2))
