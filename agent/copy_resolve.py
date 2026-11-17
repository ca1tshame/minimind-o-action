"""Copy-pointer resolution for type actions.

Grammar inside the DSL string argument:
  @S:E          copy instruction[S:E] verbatim
  @S:E|lower    copy, then lowercase
  @S:E|upper    copy, then uppercase

The resolver never invents text: the result is always a substring of the given
instruction, optionally case-transformed.
"""
import re

POINTER = re.compile(r"^@(\d+):(\d+)(?:\|(lower|upper))?$")


def resolve(args, instruction):
    if not isinstance(args, str):
        return None, {"ok": False, "reason": "not_str"}
    match = POINTER.match(args.strip())
    if not match:
        return args, {"ok": True, "copy": False}
    start, end = int(match.group(1)), int(match.group(2))
    transform = match.group(3)
    if not (0 <= start < end <= len(instruction)):
        return None, {
            "ok": False, "reason": "out_of_range",
            "s": start, "e": end, "len": len(instruction),
        }
    segment = instruction[start:end]
    if transform == "lower":
        segment = segment.lower()
    elif transform == "upper":
        segment = segment.upper()
    if not segment:
        return None, {"ok": False, "reason": "empty"}
    return segment, {
        "ok": True, "copy": True, "s": start, "e": end,
        "transform": transform, "source": instruction[start:end],
    }


def resolve_or_fallback(args, instruction):
    """Return ``(text, meta)``; fall back to the literal argument if invalid."""
    text, meta = resolve(args, instruction)
    if text is None:
        return args, {"ok": False, "fallback_literal": True, **meta}
    return text, meta
