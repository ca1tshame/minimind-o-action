"""Token-aware prompt assembly and history packing.

The V12-V14 evaluation protocol keeps the last ``hist_used`` actions (6),
compresses older ones into kind-only stubs and, when the context still does
not fit, drops the oldest entries.  The budget is computed from the real
tokenizer, the rendered prompt (goal + image placeholders + error feedback)
and the reserved generation tokens, so a long task can never overflow::

    actual_input_tokens + max_new_tokens <= context_length

``compact_history`` (V12 API) is kept for callers that only need the history
string; new code should use :func:`assemble_prompt`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ContextOverflow",
    "PackedPrompt",
    "assemble_prompt",
    "build_user_prompt",
    "compact_history",
    "count_tokens",
    "kind_stub",
    "render_history",
]

_ACTION_KIND_RE = re.compile(r"<action>\s*([a-z_]+)")
KIND_ONLY = "<%s>"


class ContextOverflow(RuntimeError):
    """The prompt plus reserved generation tokens exceeds the context length."""


@dataclass(frozen=True)
class PackedPrompt:
    prompt: str
    input_tokens: int          # tokenizer tokens + BOS
    max_new_tokens: int
    context: int
    history_used: int          # entries kept verbatim
    history_stubbed: int       # entries demoted to kind-only stubs
    history_dropped: int       # entries dropped entirely
    budget: int                # tokens available for history + error feedback


def count_tokens(tokenizer: Any, text: str) -> int:
    if not text:
        return 0
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise TypeError("tokenizer must expose encode(text, add_special_tokens=False)")
    try:
        return len(encode(text, add_special_tokens=False))
    except TypeError:
        return len(encode(text))


def kind_stub(entry: str) -> str:
    match = _ACTION_KIND_RE.search(entry)
    if match:
        return KIND_ONLY % match.group(1)
    return "<action>"


def render_history(entries: Sequence[str]) -> str:
    return "; ".join(entry for entry in entries if entry)


def build_user_prompt(goal: str, image_tokens: int, history_text: str = "",
                      error: str = "") -> str:
    """Render the V14 user turn (image placeholders are one token each)."""
    if image_tokens < 0:
        raise ValueError("image_tokens must be >= 0")
    lines = ["<|im_start|>user", goal, "Task screenshot follows." + "<|image_pad|>" * image_tokens
             + "<|vision_end|>"]
    if history_text:
        lines.append("Previous actions: " + history_text)
    if error:
        lines.append("Last action failed: " + error)
    lines.extend(["<|im_end|>", "<|im_start|>assistant"])
    return "\n".join(lines) + "\n"


def assemble_prompt(tokenizer: Any, goal: str, image_tokens: int, history: Iterable[str],
                    context: int, max_new_tokens: int, *, error: str = "",
                    hist_used: int = 6, min_history: int = 0) -> PackedPrompt:
    """Pack history under the true context budget and verify it fits.

    ``history`` entries are already-formatted action strings
    (``<action>click(...)</action>``).  When the prompt is too long the oldest
    verbatim entry is demoted to a kind-only stub, then the oldest stubs are
    dropped; ``min_history`` verbatim entries are protected from demotion.
    """
    if context <= 0 or max_new_tokens < 0:
        raise ValueError("context must be positive and max_new_tokens non-negative")
    goal = str(goal or "")
    entries: List[str] = [str(e) for e in history if e]
    if hist_used > 0:
        entries = entries[-hist_used:]
    states: List[str] = ["verbatim"] * len(entries)
    dropped = 0
    fixed_no_history = build_user_prompt(goal, image_tokens, "", error)
    # BOS is added by the caller; count it explicitly.
    base = 1 + count_tokens(tokenizer, fixed_no_history)
    budget = context - max_new_tokens - base
    if budget <= 0:
        raise ContextOverflow(
            "no room for history: context=%d max_new=%d prompt=%d"
            % (context, max_new_tokens, base)
        )

    def render() -> str:
        items = []
        for entry, state in zip(entries, states):
            items.append(entry if state == "verbatim" else kind_stub(entry))
        return render_history(items)

    while True:
        history_text = render()
        prompt = build_user_prompt(goal, image_tokens, history_text, error)
        input_tokens = 1 + count_tokens(tokenizer, prompt)
        if input_tokens + max_new_tokens <= context:
            used = sum(1 for state in states if state == "verbatim")
            stubbed = sum(1 for state in states if state == "stub")
            return PackedPrompt(prompt, input_tokens, max_new_tokens, context,
                                used, stubbed, dropped, budget)
        verbatim = [i for i, state in enumerate(states) if state == "verbatim"]
        if len(verbatim) > min_history:
            states[verbatim[0]] = "stub"
            continue
        if entries:
            entries.pop(0)
            states.pop(0)
            dropped += 1
            continue
        raise ContextOverflow(
            "prompt does not fit even without history: context=%d max_new=%d "
            "input=%d (goal/image/error dominate)" % (context, max_new_tokens, input_tokens)
        )


def compact_history(history: Iterable[str], tokenizer: Any, budget_tokens: int,
                    reserve_out: int = 32, kind_only_after: int = 2
                    ) -> Tuple[str, List[str], int]:
    """V12 history compaction: keep the newest ``kind_only_after`` verbatim.

    Returns ``(text, kept_entries, dropped)``.  This API only knows the
    history budget; use :func:`assemble_prompt` for full-prompt budgeting.
    """
    entries = [str(e) for e in history if e]
    kept: List[str] = []
    for index, entry in enumerate(reversed(entries)):
        kept.append(entry if index < max(kind_only_after, 0) else kind_stub(entry))
        text = "Previous actions: " + render_history(list(reversed(kept))) + ".\n"
        if count_tokens(tokenizer, text) <= budget_tokens - reserve_out:
            kept.reverse()
            return text, kept, max(0, len(entries) - len(kept))
    return "", [], len(entries)
