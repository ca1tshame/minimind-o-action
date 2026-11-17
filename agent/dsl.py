"""Constrained GUI action DSL and its BrowserGym translation (V2-V13 main line).

The model emits ``<action>kind(args)</action>``. Only coordinate and a small
keyboard subset are translated into BrowserGym code; unsupported kinds are
rejected rather than silently substituted.
"""
import json
import math
import re

ACTION_RE = re.compile(
    r"<action>(click|longpress|scroll|scroll_at|type|stop|key)\((.*)\)</action>",
    re.S,
)
KEY_ALLOWED = ("Enter", "Tab", "Backspace", "Escape", "ArrowUp", "ArrowDown")


def parse_action(text):
    """Parse a DSL string into ``(kind, args)``.

    Numeric coordinates are normalized to [0, 1]; scroll displacements to
    [-1, 1]. Raises ValueError on any malformed or out-of-range action.
    """
    match = ACTION_RE.fullmatch(text.strip())
    if not match:
        raise ValueError("invalid_action_syntax")
    kind, raw = match.groups()
    if kind in ("click", "longpress", "scroll", "scroll_at"):
        try:
            values = [float(x) for x in raw.split(",")]
        except ValueError as exc:
            raise ValueError("invalid_numeric_argument") from exc
        expected = 4 if kind == "scroll_at" else 2
        if len(values) != expected or not all(math.isfinite(v) for v in values):
            raise ValueError("invalid_numeric_argument")
        if kind == "scroll_at":
            if not (all(0 <= v <= 1 for v in values[:2]) and
                    all(-1 <= v <= 1 for v in values[2:])):
                raise ValueError("scroll_out_of_range")
        elif kind == "scroll":
            if not all(-1 <= v <= 1 for v in values):
                raise ValueError("scroll_out_of_range")
        elif not all(0 <= v <= 1 for v in values):
            raise ValueError("click_out_of_range")
        return kind, values
    if kind == "type":
        try:
            value = json.loads(raw)
        except ValueError as exc:
            raise ValueError("invalid_type_string") from exc
        if not isinstance(value, str) or not value or len(value) > 512:
            raise ValueError("invalid_type_string")
        return kind, value
    if kind == "stop" and raw in ("done", "failed"):
        return kind, raw
    if kind == "key" and raw in KEY_ALLOWED:
        return kind, raw
    raise ValueError("invalid_action_argument")


def to_browsergym(action, width, height):
    """Translate ``(kind, args)`` into BrowserGym Python code, or None for stop."""
    if width <= 0 or height <= 0:
        raise ValueError("invalid_viewport")
    kind, args = action
    if kind == "click":
        return "mouse_click(%r,%r)" % (
            min(args[0] * width, width - 1),
            min(args[1] * height, height - 1),
        )
    if kind == "type":
        return "keyboard_type(%r)" % args
    if kind == "key":
        return "keyboard_press(%r)" % args
    # Displacement is a finger gesture; wheel direction is inverse. V2 had no
    # scroll origin, so the viewport center is an explicit fallback.
    if kind == "scroll":
        return "scroll_at(%r,%r,%r,%r)" % (
            width // 2, height // 2,
            round(-args[0] * width), round(-args[1] * height),
        )
    if kind == "scroll_at":
        return "scroll_at(%r,%r,%r,%r)" % (
            min(round(args[0] * width), width - 1),
            min(round(args[1] * height), height - 1),
            round(-args[2] * width), round(-args[3] * height),
        )
    if kind == "stop":
        return None
    # Long press has no duration in this DSL; do not substitute a click.
    raise ValueError("longpress_duration_unsupported")
