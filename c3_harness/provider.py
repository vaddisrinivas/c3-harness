"""Provider contract for pluggable harness implementations.

This module defines stable interfaces for implementing custom harness providers,
enabling integration with external agent frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .core import HarnessHooks


@dataclass
class HarnessCapabilities:
    """Describes what features a harness provider supports."""

    supports_planning: bool = False
    supports_reflection: bool = False
    supports_streaming: bool = False
    provider_name: str = "unknown"


@dataclass
class HarnessProviderConfig:
    """Configuration for a harness provider instance."""

    provider: str = "c3_harness"
    mode: str = "default"
    app_name: str = "unknown"
    app_reason: str = ""
    settings: dict[str, Any] = field(default_factory=dict)


class HarnessProvider(Protocol):
    """Protocol for harness provider implementations.

    A harness provider wraps the execution loop with optional planning
    and reflection steps.
    """

    async def run(self, turn_ctx: Any) -> Any:
        """Execute a turn with the harness.

        Args:
            turn_ctx: Framework-specific turn context

        Returns:
            The turn result
        """
        ...

    def capabilities(self) -> HarnessCapabilities:
        """Return the provider's capabilities."""
        ...


class HarnessProviderFactory(Protocol):
    """Factory protocol for creating harness provider instances."""

    def create(
        self,
        config: HarnessProviderConfig,
        llm: Any,
        engine: Any,
        hooks: HarnessHooks,
    ) -> HarnessProvider:
        """Create a new harness provider instance.

        Args:
            config: Provider configuration
            llm: LLM call function
            engine: Tool execution engine
            hooks: Optional hooks for config injection

        Returns:
            A configured harness provider
        """
        ...
