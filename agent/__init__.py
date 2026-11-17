"""Agent modules: constrained action parsing, context packing, and environment loop."""
from .action import parse_action as parse_json_action, BrowserAction, ActionError
from .ctx_pack import compact_history
