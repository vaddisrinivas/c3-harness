"""c3-focused runtime adapter helpers for c3_harness.

This module keeps c3-specific harness wiring in the harness package, while
remaining framework-agnostic via injectable callbacks.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Awaitable, Callable

from .core import AgentHarness, HarnessEvent, HarnessHooks, HarnessSettings
from .provider import HarnessProviderConfig

DefaultsGetter = Callable[[], dict[str, Any]]
PlannerDefaultsGetter = Callable[[], dict[str, Any]]
Logger = Callable[[str], None]
AuditEmitter = Callable[[HarnessEvent], Awaitable[None]]


class C3HarnessRuntime:
    """Resolve harness config and construct c3-compatible hooks/providers."""

    def __init__(
        self,
        defaults_getter: DefaultsGetter,
        planner_defaults_getter: PlannerDefaultsGetter,
        logger: Logger,
        audit_emitter: AuditEmitter,
    ) -> None:
        self._defaults_getter = defaults_getter
        self._planner_defaults_getter = planner_defaults_getter
        self._logger = logger
        self._audit_emitter = audit_emitter

    def defaults(self) -> dict[str, Any]:
        return self._defaults_getter() or {}

    def planner_model(self) -> str:
        defaults = self.defaults()
        if model := str(defaults.get("harness_planner_model", "")).strip():
            return model
        router_cfg = self._planner_defaults_getter() or {}
        return router_cfg.get("cheap_model", "") or router_cfg.get("model", "azure/gpt-4.1-nano")

    def llm_extra_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if key := os.environ.get("AZURE_AI_API_KEY"):
            kwargs["api_key"] = key
        if base := os.environ.get("AZURE_AI_API_BASE"):
            kwargs["api_base"] = base
        return kwargs

    def settings_from_config(self) -> HarnessSettings:
        defaults = self.defaults()
        return HarnessSettings(
            plan_prefix=str(defaults.get("harness_plan_prefix", "Plan for this turn: ")),
            plan_prompt_template=str(defaults.get("harness_plan_prompt", "")),
            reflect_prompt_template=str(defaults.get("harness_reflect_prompt", "")),
            repair_prefix=str(defaults.get("harness_repair_prefix", "The previous attempt was incomplete. Specifically:")),
            repair_suffix=str(defaults.get("harness_repair_suffix", "Address only the missing items. Do not repeat what was already correct.")),
            side_effect_warning=str(defaults.get("harness_side_effect_warning", "Previous attempt already called: {tools}. Do NOT repeat these side-effecting actions.")),
            side_effect_tools=tuple(str(tool) for tool in defaults.get("harness_side_effect_tools", [])),
            max_attempts=int(defaults.get("harness_max_attempts", 2)),
            planner_timeout_s=float(defaults.get("harness_planner_timeout", 3.0)),
            reflector_timeout_s=float(defaults.get("harness_reflector_timeout", 3.0)),
            plan_max_tokens=int(defaults.get("harness_plan_max_tokens", 150)),
            reflect_max_tokens=int(defaults.get("harness_reflect_max_tokens", 120)),
        )

    def hooks(self) -> HarnessHooks:
        return HarnessHooks(
            get_settings=self.settings_from_config,
            get_planner_model=self.planner_model,
            build_llm_kwargs=self.llm_extra_kwargs,
            get_sender=lambda turn_ctx: str(getattr(turn_ctx, "meta", {}).get("sender_raw", "")),
            get_app_name=lambda turn_ctx: str(getattr(turn_ctx, "app_name", "unknown")),
            emit_event=self._audit_emitter,
            log=self._logger,
        )

    def resolve_config(self, app_config: dict[str, Any] | None = None) -> HarnessProviderConfig:
        app_config = app_config or {}
        defaults = self.defaults()
        provider = str(defaults.get("harness_provider", "c3_harness")).strip() or "c3_harness"
        mode = str(defaults.get("harness", "default")).strip() or "default"
        app_name = str(app_config.get("name", "unknown"))
        reason = ""
        if app_config.get("harness_override"):
            mode = str(app_config.get("harness_mode", mode) or mode)
            reason = str(app_config.get("harness_reason", "") or "")
        return HarnessProviderConfig(
            provider=provider,
            mode=mode,
            app_name=app_name,
            app_reason=reason,
            settings=asdict(self.settings_from_config()),
        )

    def should_use_turn_engine(self, app_config: dict[str, Any] | None = None) -> bool:
        config = self.resolve_config(app_config)
        return bool(self.defaults().get("use_turn_engine", False) or config.mode != "default")

    def create_harness(self, llm: Any, engine: Any, app_config: dict[str, Any] | None = None) -> Any:
        config = self.resolve_config(app_config)
        if config.provider != "c3_harness":
            raise ValueError(f"Unknown harness provider: {config.provider}")
        return AgentHarness.from_config(config.mode, llm, engine, hooks=self.hooks())
