"""Compatibility manager module migrated from c3-py into c3_harness."""

from __future__ import annotations

from typing import Any

from .c3_runtime import C3HarnessRuntime
from .provider import HarnessProviderConfig


class HarnessManager:
    """Compatibility wrapper that delegates to ``C3HarnessRuntime``."""

    def __init__(self, runtime: C3HarnessRuntime) -> None:
        self._runtime = runtime

    def resolve_config(self, app_config: dict[str, Any] | None = None) -> HarnessProviderConfig:
        return self._runtime.resolve_config(app_config)

    def should_use_turn_engine(self, app_config: dict[str, Any] | None = None) -> bool:
        return self._runtime.should_use_turn_engine(app_config)

    def create_provider(self, llm: Any, engine: Any, app_config: dict[str, Any] | None = None) -> Any:
        return self._runtime.create_harness(llm=llm, engine=engine, app_config=app_config)


__all__ = ["HarnessManager"]
