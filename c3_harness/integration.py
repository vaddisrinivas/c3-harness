"""Integration helpers for using c3_harness in agent frameworks.

This module provides wrapper classes and utilities for integrating c3_harness
into agent frameworks. It handles the glue between your framework's execution
engine and the harness's planning/reflection capabilities.
"""

from __future__ import annotations

from typing import Any, Callable

from .core import (
    ConfigurableHarness as _BaseConfigurableHarness,
    HarnessEvent,
    HarnessHooks,
    HarnessPolicy,
    HarnessSettings,
    HarnessState,
    PRESETS,
)


def _default_hooks() -> HarnessHooks:
    """Default hooks for integrations that don't provide custom settings."""
    return HarnessHooks(
        get_settings=lambda: HarnessSettings(
            side_effect_tools=("send_message", "send_image"),
        ),
        get_planner_model=lambda: "",
        build_llm_kwargs=lambda: {},
    )


class IntegrationHarness(_BaseConfigurableHarness):
    """Base harness class for framework integrations.

    Wraps the base ConfigurableHarness with framework-specific engine handling.
    Subclass this and override _create_engine() to integrate with your framework.

    Example:
        class MyFrameworkHarness(IntegrationHarness):
            def _create_engine(self, tool_registry):
                return MyEngine(tool_registry)

        harness = MyFrameworkHarness(
            policy=PRESETS["planning"],
            llm=my_llm_function,
            tool_registry=my_tools,
        )
        result = harness.run(turn_context)
    """

    def __init__(
        self,
        policy: HarnessPolicy,
        llm: Callable[..., Any],
        tool_registry: Any,
        hooks: HarnessHooks | None = None,
        event_bus: Any = None,
        engine: Any | None = None,
    ) -> None:
        """Initialize the integration harness.

        Args:
            policy: Harness policy (use PRESETS or create custom)
            llm: LLM call function, signature: (messages, **kwargs) -> str
            tool_registry: Your framework's tool registry or engine
            hooks: Optional hooks for config/settings injection
            event_bus: Optional event bus for logging/telemetry
            engine: Pre-built engine. If None, _create_engine() is called.
        """
        self.policy = policy
        self._llm = llm
        self._event_bus = event_bus

        if engine is not None:
            self._engine = engine
        elif hasattr(tool_registry, "execute"):
            # Tool registry has execute method — use it directly
            self._engine = tool_registry
        else:
            # Create engine from tool registry
            self._engine = self._create_engine(tool_registry)

        super().__init__(
            policy=policy,
            llm=llm,
            engine=self._engine,
            hooks=hooks or _default_hooks(),
        )

    def _create_engine(self, tool_registry: Any) -> Any:
        """Create an execution engine from your tool registry.

        Override this in your subclass to wrap your framework's tool execution.

        Default implementation assumes tool_registry is callable.

        Args:
            tool_registry: Your framework's tool registry

        Returns:
            An engine with an execute(tools: list) -> list method
        """
        if callable(tool_registry):
            return tool_registry
        raise NotImplementedError(
            "Tool registry not callable. Override _create_engine() in your subclass "
            "to return an engine with an execute(tools) method."
        )


class AgentHarness(IntegrationHarness):
    """Convenience factory for creating harnesses from mode strings.

    Example:
        # Planning mode
        harness = AgentHarness.from_config("planning", llm, tools)

        # Full mode (plan + reflect)
        harness = AgentHarness.from_config("full", llm, tools)
    """

    @classmethod
    def from_config(
        cls,
        mode: str,
        llm: Callable[..., Any],
        tool_registry: Any,
        hooks: HarnessHooks | None = None,
    ) -> IntegrationHarness:
        """Create a harness from a mode string.

        Args:
            mode: One of: default, planning, reflection, full
            llm: LLM call function
            tool_registry: Your framework's tool registry
            hooks: Optional hooks for config/settings injection

        Returns:
            A configured IntegrationHarness instance
        """
        policy = PRESETS.get((mode or "default").lower(), PRESETS["default"])
        return cls(
            policy=policy,
            llm=llm,
            tool_registry=tool_registry,
            hooks=hooks,
        )


class PlanningHarness(IntegrationHarness):
    """Pre-configured harness for planning mode.

    Runs a planning step before execution.
    """

    def __init__(
        self,
        llm: Callable[..., Any],
        tool_registry: Any,
        hooks: HarnessHooks | None = None,
        event_bus: Any = None,
    ) -> None:
        super().__init__(
            PRESETS["planning"],
            llm,
            tool_registry,
            hooks=hooks,
            event_bus=event_bus,
        )


class ReflectionHarness(IntegrationHarness):
    """Pre-configured harness for reflection mode.

    Executes, then reflects and retries if incomplete.
    """

    def __init__(
        self,
        llm: Callable[..., Any],
        tool_registry: Any,
        hooks: HarnessHooks | None = None,
        event_bus: Any = None,
    ) -> None:
        super().__init__(
            PRESETS["reflection"],
            llm,
            tool_registry,
            hooks=hooks,
            event_bus=event_bus,
        )


class FullHarness(IntegrationHarness):
    """Pre-configured harness for full mode.

    Plans → executes → reflects → retries if needed.
    """

    def __init__(
        self,
        llm: Callable[..., Any],
        tool_registry: Any,
        hooks: HarnessHooks | None = None,
        event_bus: Any = None,
    ) -> None:
        super().__init__(
            PRESETS["full"],
            llm,
            tool_registry,
            hooks=hooks,
            event_bus=event_bus,
        )
        self._planner = self
        self._reflector = self


__all__ = [
    "AgentHarness",
    "IntegrationHarness",
    "PlanningHarness",
    "ReflectionHarness",
    "FullHarness",
    "HarnessEvent",
    "HarnessHooks",
    "HarnessPolicy",
    "HarnessSettings",
    "HarnessState",
    "PRESETS",
]
