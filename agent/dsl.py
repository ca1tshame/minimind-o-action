"""Canonical MiniMind-O Action GUI action DSL (V14 protocol).

Single grammar shared by data collection, SFT, inference and evaluation::

    <action>click(X,Y)</action>
    <action>scroll(DX,DY)</action>
    <action>type("text")</action>
    <action>type(@S:E)</action>            # copy pointer, optional |lower / |upper
    <action>key(Enter)</action>
    <action>stop()</action>
    <action>stop(done)</action>
    <action>stop(failed)</action>

Coordinates are viewport fractions in [0, 1].  ``scroll`` follows the V14
content-displacement convention: ``scroll(dx, dy)`` where ``dy = -0.9``
scrolls *down*; the executor maps it to ``scroll_at(cx, cy, -dx*W, -dy*H)``
(wheel deltaY positive scrolls down).  Do not reinterpret the two scroll
arguments as coordinates: V1's ``scroll_at(x, y, dx, dy)`` prototype was a
different action and is kept only in ``legacy/``.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional, Tuple

from agent.copy_resolve import CopyPointer, parse_pointer

__all__ = [
    "ActionParseError",
    "KEY_NAMES",
    "format_action",
    "extract_action",
    "parse_action",
    "strip_think",
    "to_browsergym",
]

KEY_NAMES = ("Enter", "Tab", "Backspace", "Escape", "ArrowUp", "ArrowDown")

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ACTION_RE = re.compile(
    r"<action>\s*(click|scroll|type|key|stop)\s*\((.*?)\)\s*</action>", re.S
)
_STOP_VALUES = ("", "done", "failed")


class ActionParseError(ValueError):
    """Raised when a model output is not a valid V14 action."""


def strip_think(text: str) -> str:
    """Remove a leading ``<think>...</think>`` block (V14 sometimes emits it)."""
    if not isinstance(text, str):
        raise ActionParseError("action text must be a string")
    stripped = text.strip()
    match = _THINK_RE.match(stripped)
    if match:
        return stripped[match.end():].strip()
    return stripped


def extract_action(text: str) -> Optional[str]:
    """Return the last complete ``<action>...</action>`` substring, if any."""
    if not isinstance(text, str):
        return None
    matches = list(ACTION_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0).strip()


def _finite_float(raw: str, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ActionParseError("invalid %s value %r" % (name, raw)) from exc
    if not math.isfinite(value):
        raise ActionParseError("%s must be finite, got %r" % (name, raw))
    return value


def _split_args(raw: str) -> list:
    if raw is None:
        return []
    if raw.strip() == "":
        return []
    return [part.strip() for part in raw.split(",")]


def _parse_type(raw: str, instruction: Optional[str]) -> Any:
    body = raw.strip()
    if not body:
        raise ActionParseError("type() requires text or a copy pointer")
    if body.startswith('"'):
        try:
            value = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ActionParseError("type() string is not valid JSON: %s" % exc) from exc
        if not isinstance(value, str):
            raise ActionParseError("type() must contain a JSON string")
        if not value:
            raise ActionParseError("type() text is empty")
        return value
    if body.startswith("@"):
        pointer = parse_pointer(body)
        if instruction is not None:
            from agent.copy_resolve import resolve

            return resolve(pointer, instruction)[0]
        return pointer
    raise ActionParseError(
        "type() needs a quoted string (type(\"text\")) or a copy pointer "
        "(type(@S:E|lower)); got %r" % body
    )


def parse_action(text: str, instruction: Optional[str] = None) -> Tuple[str, Any]:
    """Parse one action.

    Accepts a leading ``<think>`` block and surrounding prose; the last
    complete action tag wins.  ``instruction`` is only needed to resolve a
    copy pointer to the actual text; without it the pointer is returned as a
    :class:`CopyPointer` for the executor to resolve at run time.

    Returns ``(kind, value)``:

    * click  -> ``(float x, float y)``
    * scroll -> ``(float dx, float dy)``
    * type   -> ``str`` or :class:`CopyPointer`
    * key    -> ``str``
    * stop   -> ``""`` / ``"done"`` / ``"failed"``
    """
    canonical = extract_action(text) or strip_think(text)
    match = ACTION_RE.fullmatch(canonical.strip())
    if match is None:
        raise ActionParseError(
            "expected one <action>kind(args)</action> tag, got %r" % (text,)
        )
    kind = match.group(1)
    raw = match.group(2).strip()

    if kind == "click":
        parts = _split_args(raw)
        if len(parts) != 2:
            raise ActionParseError("click() requires x,y; got %r" % raw)
        x = _finite_float(parts[0], "click x")
        y = _finite_float(parts[1], "click y")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ActionParseError("click coordinates must be in [0,1], got (%r,%r)" % (x, y))
        return kind, (x, y)

    if kind == "scroll":
        parts = _split_args(raw)
        if len(parts) != 2:
            raise ActionParseError("scroll() requires dx,dy; got %r" % raw)
        dx = _finite_float(parts[0], "scroll dx")
        dy = _finite_float(parts[1], "scroll dy")
        for value in (dx, dy):
            if not -1.0 <= value <= 1.0:
                raise ActionParseError("scroll deltas must be in [-1,1], got (%r,%r)" % (dx, dy))
        return kind, (dx, dy)

    if kind == "type":
        return kind, _parse_type(raw, instruction)

    if kind == "key":
        if raw not in KEY_NAMES:
            raise ActionParseError("key() must be one of %s; got %r" % (list(KEY_NAMES), raw))
        return kind, raw

    if raw not in _STOP_VALUES:
        raise ActionParseError("stop() accepts nothing, done or failed; got %r" % raw)
    return kind, raw


def format_action(kind: str, value: Any) -> str:
    """Serialise a parsed action back to the exact DSL string."""
    if kind == "click":
        return "<action>click(%.4f,%.4f)</action>" % (value[0], value[1])
    if kind == "scroll":
        return "<action>scroll(%.4f,%.4f)</action>" % (value[0], value[1])
    if kind == "type":
        if isinstance(value, CopyPointer):
            return "<action>type(%s)</action>" % (value,)
        return "<action>type(%s)</action>" % json.dumps(value)
    if kind == "key":
        return "<action>key(%s)</action>" % value
    if kind == "stop":
        return "<action>stop(%s)</action>" % (value or "")
    raise ActionParseError("unknown action kind %r" % kind)


def to_browsergym(action: Tuple[str, Any], width: int, height: int,
                  instruction: Optional[str] = None) -> Optional[str]:
    """Translate a parsed action into a BrowserGym high-level command.

    Returns ``None`` for ``stop`` (the caller must terminate the episode).
    """
    kind, value = action
    if kind == "stop":
        return None
    if kind == "click":
        x = min(max(value[0], 0.0), 1.0)
        y = min(max(value[1], 0.0), 1.0)
        return "mouse_click(%d,%d)" % (int(x * (width - 1)), int(y * (height - 1)))
    if kind == "scroll":
        dx, dy = value
        return "scroll_at(%d,%d,%d,%d)" % (
            width // 2, height // 2, int(round(-dx * width)), int(round(-dy * height))
        )
    if kind == "type":
        if isinstance(value, CopyPointer):
            if instruction is None:
                raise ActionParseError("copy pointer needs the task instruction to resolve")
            from agent.copy_resolve import resolve

            value = resolve(value, instruction)[0]
        return "keyboard_type(%s)" % json.dumps(value)
    if kind == "key":
        return "keyboard_press(%s)" % value
    raise ActionParseError("unknown action kind %r" % kind)
