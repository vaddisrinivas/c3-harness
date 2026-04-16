"""Backward-compatible harness exports for integrations."""

from __future__ import annotations

from .core import (
    ConfigurableHarness,
    HarnessEvent,
    HarnessHooks,
    HarnessPolicy,
    HarnessSettings,
    HarnessState,
    PRESETS,
)
from .core import AgentHarness as CoreAgentHarness
from .integration import IntegrationHarness
from .core import strip_code_fences as _strip_code_fences

AgentHarness = CoreAgentHarness

__all__ = [
    "AgentHarness",
    "ConfigurableHarness",
    "HarnessEvent",
    "HarnessHooks",
    "HarnessPolicy",
    "HarnessSettings",
    "HarnessState",
    "IntegrationHarness",
    "PRESETS",
    "_strip_code_fences",
]
