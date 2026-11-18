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
