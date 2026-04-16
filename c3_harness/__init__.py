"""c3_harness: Configurable planning/execute/reflection harness for agent turns.

Core classes:
    - AgentHarness: Factory for creating harnesses from mode strings
    - IntegrationHarness: Base class for framework integrations
    - PlanningHarness, ReflectionHarness, FullHarness: Pre-configured modes

Provider interface:
    - HarnessProvider: Protocol for implementing custom providers
    - HarnessProviderConfig: Configuration for provider instances
    - HarnessCapabilities: Describes provider features

Usage:
    from c3_harness import AgentHarness, PRESETS

    harness = AgentHarness.from_config(
        mode="planning",
        llm=my_llm_function,
        tool_registry=my_tools,
    )
    result = harness.run(turn_context)

Integration:
    Subclass IntegrationHarness to integrate with your framework:
        class MyHarness(IntegrationHarness):
            def _create_engine(self, tool_registry):
                return MyEngine(tool_registry)

Provider implementation:
    Implement the HarnessProvider protocol for custom providers:
        class MyProvider:
            async def run(self, turn_ctx): ...
            def capabilities(self): ...
"""

from .core import (
    AgentHarness,
    ConfigurableHarness,
    HarnessEvent,
    HarnessHooks,
    HarnessPolicy,
    HarnessSettings,
    HarnessState,
    PRESETS,
)
from .integration import (
    IntegrationHarness,
    FullHarness,
    PlanningHarness,
    ReflectionHarness,
)
from .provider import (
    HarnessCapabilities,
    HarnessProvider,
    HarnessProviderConfig,
    HarnessProviderFactory,
)
from .c3_runtime import C3HarnessRuntime
from .harness_manager import HarnessManager

__all__ = [
    # Core
    "AgentHarness",
    "ConfigurableHarness",
    "HarnessEvent",
    "HarnessHooks",
    "HarnessPolicy",
    "HarnessSettings",
    "HarnessState",
    "PRESETS",
    # Integration
    "IntegrationHarness",
    "PlanningHarness",
    "ReflectionHarness",
    "FullHarness",
    # Provider
    "HarnessProvider",
    "HarnessProviderConfig",
    "HarnessCapabilities",
    "HarnessProviderFactory",
    "C3HarnessRuntime",
    "HarnessManager",
]
