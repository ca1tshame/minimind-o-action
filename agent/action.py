"""Deprecated V1 JSON action protocol.

The formal GUI agent line is the ``<action>click(0.5,0.5)</action>`` DSL in
:mod:`agent.dsl`; the old JSON prototype is kept only so historical records
still load. Import from :mod:`legacy.action_v1` directly in new code.
"""

from legacy.action_v1 import (  # noqa: F401
    ActionError,
    BrowserAction,
    parse_action,
    step_environment,
)

__all__ = ["ActionError", "BrowserAction", "parse_action", "step_environment"]
