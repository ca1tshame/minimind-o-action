"""Resolve ``@S:E`` / ``@S:E|lower`` / ``@S:E|upper`` copy pointers.

The pointer refers to a character span of the task instruction (the text the
policy is asked to copy).  Invalid spans are reported instead of silently
producing an empty string; callers decide whether to feed the error back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

__all__ = ["CopyPointer", "POINTER", "parse_pointer", "resolve", "resolve_or_fallback"]

POINTER = re.compile(r"^@([0-9]+):([0-9]+)(?:\|(lower|upper))?$")


@dataclass(frozen=True)
class CopyPointer:
    start: int
    end: int
    transform: Optional[str] = None

    def __str__(self) -> str:
        suffix = "|%s" % self.transform if self.transform else ""
        return "@%d:%d%s" % (self.start, self.end, suffix)


def parse_pointer(raw: str) -> CopyPointer:
    match = POINTER.match(raw.strip())
    if match is None:
        raise ValueError("not a copy pointer: %r" % raw)
    start, end = int(match.group(1)), int(match.group(2))
    if end <= start:
        raise ValueError("copy pointer %r has an empty span" % raw)
    return CopyPointer(start, end, match.group(3))


def _apply(text: str, transform: Optional[str]) -> str:
    if transform == "lower":
        return text.lower()
    if transform == "upper":
        return text.upper()
    return text


def resolve(args: Union[CopyPointer, str], instruction: str) -> Tuple[str, Dict[str, Any]]:
    """Resolve a pointer against ``instruction``.

    Returns ``(text, meta)``.  ``meta['error']`` is set when the pointer is
    invalid; ``text`` is then empty and the caller should report the failure.
    """
    meta: Dict[str, Any] = {"kind": "copy_pointer", "instruction_length": len(instruction)}
    if isinstance(args, str):
        try:
            pointer = parse_pointer(args)
        except ValueError as exc:
            meta["error"] = str(exc)
            return "", meta
    else:
        pointer = args
    if not isinstance(instruction, str):
        meta["error"] = "instruction is not text"
        return "", meta
    if pointer.start < 0 or pointer.end > len(instruction) or pointer.start >= pointer.end:
        meta["error"] = "copy span %d:%d outside instruction of length %d" % (
            pointer.start, pointer.end, len(instruction)
        )
        return "", meta
    raw = instruction[pointer.start:pointer.end]
    if not raw or raw.strip() == "":
        meta["error"] = "copy span %d:%d selects whitespace" % (pointer.start, pointer.end)
        return "", meta
    meta.update({"start": pointer.start, "end": pointer.end, "transform": pointer.transform})
    return _apply(raw, pointer.transform), meta


def resolve_or_fallback(args: Union[CopyPointer, str, None], instruction: str
                        ) -> Tuple[str, Dict[str, Any]]:
    """Resolve a pointer, falling back to a literal string when it is invalid.

    Used by the executor for old checkpoints that emitted a literal inside
    ``type()``; a bad pointer is still flagged in ``meta`` so evaluation can
    record it as an action error.
    """
    if args is None:
        return "", {"kind": "empty", "error": "no text"}
    if isinstance(args, CopyPointer):
        return resolve(args, instruction)
    text = str(args)
    if not text.startswith("@"):
        return text, {"kind": "literal", "fallback_literal": False}
    resolved, meta = resolve(text, instruction)
    if not meta.get("error"):
        return resolved, meta
    return text, {"kind": "literal", "fallback_literal": True, "error": meta["error"]}
