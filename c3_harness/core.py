from __future__ import annotations

import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Protocol


def strip_code_fences(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw


class LLMClient(Protocol):
    async def acompletion(self, **kwargs: Any) -> Any: ...


class TurnEngineLike(Protocol):
    async def execute(self, turn_ctx: Any) -> Any: ...


@dataclass
class HarnessSettings:
    plan_prefix: str = "Plan for this turn: "
    plan_prompt_template: str = (
        "Break this task into at most 3 concise steps as a JSON array.\n"
        "Goal: {goal}\n"
        "Tools: {tools}"
    )
    reflect_prompt_template: str = (
        "Check whether the task is complete. Return JSON with keys: "
        "status, missing.\nGoal: {goal}\nTools used: {tools_used}\nResponse: {response}"
    )
    repair_prefix: str = "The previous attempt was incomplete. Specifically:"
    repair_suffix: str = "Address only the missing items. Do not repeat what was already correct."
    side_effect_warning: str = "Previous attempt already called: {tools}. Do NOT repeat these side-effecting actions."
    side_effect_tools: tuple[str, ...] = ()
    max_attempts: int = 2
    planner_timeout_s: float = 3.0
    reflector_timeout_s: float = 3.0
    plan_max_tokens: int = 150
    reflect_max_tokens: int = 120


@dataclass
class HarnessPolicy:
    enable_planning: bool = False
    enable_reflection: bool = False
    max_attempts: int | None = None
    planner_timeout_s: float | None = None
    reflector_timeout_s: float | None = None

    def effective_max_attempts(self, settings: HarnessSettings) -> int:
        return self.max_attempts or settings.max_attempts

    def effective_planner_timeout(self, settings: HarnessSettings) -> float:
        return self.planner_timeout_s or settings.planner_timeout_s

    def effective_reflector_timeout(self, settings: HarnessSettings) -> float:
        return self.reflector_timeout_s or settings.reflector_timeout_s


PRESETS: dict[str, HarnessPolicy] = {
    "default": HarnessPolicy(),
    "planning": HarnessPolicy(enable_planning=True),
    "reflection": HarnessPolicy(enable_reflection=True),
    "full": HarnessPolicy(enable_planning=True, enable_reflection=True),
}


@dataclass
class HarnessState:
    attempt: int = 0
    plan: list[str] = field(default_factory=list)
    augmentations: list[str] = field(default_factory=list)
    verification: dict[str, Any] | None = None

    def render_system_prefix(self, settings: HarnessSettings) -> str:
        parts: list[str] = []
        if self.plan:
            steps = " -> ".join(f"{i + 1}. {str(step)[:80]}" for i, step in enumerate(self.plan[:3]))
            parts.append(settings.plan_prefix + steps)
        parts.extend(self.augmentations)
        return "\n\n".join(parts)


@dataclass
class HarnessEvent:
    event_type: str
    details: dict[str, Any]


@dataclass
class HarnessHooks:
    get_settings: Callable[[], HarnessSettings]
    get_planner_model: Callable[[], str]
    build_llm_kwargs: Callable[[], dict[str, Any]] = field(default_factory=dict)
    get_sender: Callable[[Any], str] = lambda _turn_ctx: ""
    get_app_name: Callable[[Any], str] = lambda _turn_ctx: "unknown"
    before_execute: Callable[[Any, HarnessState, HarnessSettings], Any] | None = None
    emit_event: Callable[[HarnessEvent], Awaitable[None]] | None = None
    log: Callable[[str], None] | None = None


class AgentHarness(ABC):
    @abstractmethod
    async def run(self, turn_ctx: Any) -> Any: ...

    @classmethod
    def from_config(
        cls,
        mode: str,
        llm: LLMClient,
        engine: TurnEngineLike,
        hooks: HarnessHooks,
    ) -> "AgentHarness":
        return ConfigurableHarness(PRESETS.get((mode or "default").lower(), PRESETS["default"]), llm, engine, hooks)


class ConfigurableHarness(AgentHarness):
    def __init__(
        self,
        policy: HarnessPolicy,
        llm: LLMClient,
        engine: TurnEngineLike,
        hooks: HarnessHooks,
    ) -> None:
        self.policy = policy
        self._llm = llm
        self._engine = engine
        self._hooks = hooks

    async def run(self, turn_ctx: Any) -> Any:
        state = HarnessState()
        settings = self._hooks.get_settings()
        mode = self._mode_name()
        await self._emit("harness.start", turn_ctx, mode, settings)
        if self.policy.enable_planning:
            started = time.time()
            await self._plan(turn_ctx, state, settings)
            await self._emit(
                "harness.plan",
                turn_ctx,
                mode,
                settings,
                success=bool(state.plan),
                plan=state.plan,
                duration_ms=round((time.time() - started) * 1000, 1),
            )
        result: Any = None
        while True:
            exec_ctx = self._augment_ctx(turn_ctx, state, settings)
            if self._hooks.before_execute:
                self._hooks.before_execute(exec_ctx, state, settings)
            started = time.time()
            result = await self._engine.execute(exec_ctx)
            state.attempt += 1
            await self._emit(
                "harness.execute",
                turn_ctx,
                mode,
                settings,
                attempt=state.attempt,
                max_attempts=self.policy.effective_max_attempts(settings),
                success=True,
                duration_ms=round((time.time() - started) * 1000, 1),
            )
            if not self.policy.enable_reflection:
                break
            started = time.time()
            verdict = await self._reflect(turn_ctx, result, state, settings)
            await self._emit(
                "harness.reflect",
                turn_ctx,
                mode,
                settings,
                success=verdict is not None,
                status=(verdict or {}).get("status", "unknown"),
                should_retry=(verdict or {}).get("should_retry", False),
                missing=(verdict or {}).get("missing", []),
                duration_ms=round((time.time() - started) * 1000, 1),
            )
            if not verdict or not verdict.get("should_retry"):
                break
            if state.attempt >= self.policy.effective_max_attempts(settings):
                await self._emit(
                    "harness.gave_up",
                    turn_ctx,
                    mode,
                    settings,
                    attempts=state.attempt,
                    max_attempts=self.policy.effective_max_attempts(settings),
                )
                self._log(f"harness/reflect: gave up after {state.attempt} attempts")
                break
            self._add_repair_hint(state, result, verdict, settings)
        return result

    def _mode_name(self) -> str:
        if self.policy.enable_planning and self.policy.enable_reflection:
            return "full"
        if self.policy.enable_planning:
            return "planning"
        if self.policy.enable_reflection:
            return "reflection"
        return "default"

    async def _emit(self, event_type: str, turn_ctx: Any, mode: str, settings: HarnessSettings, **details: Any) -> None:
        if not self._hooks.emit_event:
            return
        payload = {
            "app": self._hooks.get_app_name(turn_ctx),
            "user": self._hooks.get_sender(turn_ctx),
            "mode": mode,
            "planning_enabled": self.policy.enable_planning,
            "reflection_enabled": self.policy.enable_reflection,
            **details,
        }
        await self._hooks.emit_event(HarnessEvent(event_type=event_type, details=payload))

    async def _plan(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings) -> None:
        tool_names = [
            str((tool.get("function") or {}).get("name") or "")
            for tool in (getattr(turn_ctx, "tool_schemas", None) or [])[:15]
        ]
        prompt = (
            settings.plan_prompt_template
            .replace("{goal}", str(getattr(turn_ctx, "user_content", ""))[:300])
            .replace("{tools}", ", ".join(tool_names[:10]) or "none")
        )
        try:
            response = await asyncio.wait_for(
                self._llm.acompletion(
                    model=self._hooks.get_planner_model(),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings.plan_max_tokens,
                    **self._hooks.build_llm_kwargs(),
                ),
                timeout=self.policy.effective_planner_timeout(settings),
            )
            raw = strip_code_fences((response.choices[0].message.content or "").strip())
            steps = json.loads(raw)
            if isinstance(steps, list) and steps:
                state.plan = [str(step)[:80] for step in steps[:3] if step]
                self._log(f"harness/plan: {' -> '.join(state.plan)[:120]}")
        except Exception as exc:
            self._log(f"harness/plan: skipped ({type(exc).__name__}: {exc})")

    async def _reflect(self, turn_ctx: Any, result: Any, state: HarnessState, settings: HarnessSettings) -> dict[str, Any] | None:
        if getattr(result, "reply_sent_via_tool", False):
            return None
        if not getattr(result, "reply", "") and not getattr(result, "tools_called", []):
            return None
        tools_used = ", ".join(tool.get("name", "") for tool in getattr(result, "tools_called", [])[:8]) or "none"
        prompt = (
            settings.reflect_prompt_template
            .replace("{goal}", str(getattr(turn_ctx, "user_content", ""))[:300])
            .replace("{tools_used}", tools_used)
            .replace("{response}", str(getattr(result, "reply", ""))[:400])
        )
        try:
            response = await asyncio.wait_for(
                self._llm.acompletion(
                    model=self._hooks.get_planner_model(),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=settings.reflect_max_tokens,
                    **self._hooks.build_llm_kwargs(),
                ),
                timeout=self.policy.effective_reflector_timeout(settings),
            )
            raw = strip_code_fences((response.choices[0].message.content or "").strip())
            verdict = json.loads(raw)
            if not isinstance(verdict, dict):
                self._log("harness/reflect: done (non-dict response)")
                return None
            status = str(verdict.get("status", "done")).lower()
            missing = verdict.get("missing", [])
            verdict["should_retry"] = status == "incomplete" and isinstance(missing, list) and bool(missing)
            state.verification = verdict
            self._log(f"harness/reflect: {'incomplete' if verdict['should_retry'] else 'done'}")
            return verdict
        except Exception as exc:
            self._log(f"harness/reflect: skipped ({type(exc).__name__}: {exc})")
            return None

    def _augment_ctx(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings) -> Any:
        prefix = state.render_system_prefix(settings)
        if not prefix:
            return turn_ctx
        return replace(turn_ctx, system_prompt=prefix + "\n\n" + getattr(turn_ctx, "system_prompt", ""))

    def _add_repair_hint(self, state: HarnessState, result: Any, verdict: dict[str, Any], settings: HarnessSettings) -> None:
        lines = [settings.repair_prefix]
        for gap in (verdict.get("missing", []) or [])[:5]:
            lines.append(f"  - {str(gap)[:100]}")
        lines.append(settings.repair_suffix)
        side_effect_names = [
            tool.get("name", "")
            for tool in getattr(result, "tools_called", [])
            if tool.get("name", "") in settings.side_effect_tools
        ]
        if side_effect_names:
            lines.append(settings.side_effect_warning.replace("{tools}", ", ".join(side_effect_names)))
        state.augmentations.append("\n".join(lines))

    def _log(self, message: str) -> None:
        if self._hooks.log:
            self._hooks.log(message)
