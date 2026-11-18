"""Agent modules: canonical action DSL, context packing, and the env loop."""
from .copy_resolve import CopyPointer
from .ctx_pack import ContextOverflow, PackedPrompt, assemble_prompt, compact_history
from .dsl import ActionParseError, format_action, parse_action, to_browsergym

# Deprecated V1 JSON prototype, kept importable for old notebooks only.
from legacy.action_v1 import ActionError, BrowserAction
from legacy.action_v1 import parse_action as parse_json_action

__all__ = [
    "ActionError",
    "ActionParseError",
    "BrowserAction",
    "ContextOverflow",
    "CopyPointer",
    "PackedPrompt",
    "assemble_prompt",
    "compact_history",
    "format_action",
    "parse_action",
    "parse_json_action",
    "to_browsergym",
]
