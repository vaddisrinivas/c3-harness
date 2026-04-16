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


@dataclass(frozen=True)
class HookPoints:
    before_turn: str = "before_turn"
    before_plan: str = "before_plan"
    after_plan: str = "after_plan"
    before_execute: str = "before_execute"
    after_execute: str = "after_execute"
    before_reflect: str = "before_reflect"
    after_reflect: str = "after_reflect"
    on_error: str = "on_error"
    on_event: str = "on_event"
    before_tool_call: str = "before_tool_call"
    after_tool_call: str = "after_tool_call"
    on_tool_error: str = "on_tool_error"
    augment_system_prompt: str = "augment_system_prompt"
    augment_user_prompt: str = "augment_user_prompt"
    filter_tools: str = "filter_tools"
    inject_turn_metadata: str = "inject_turn_metadata"
    authorize_tool_call: str = "authorize_tool_call"
    redact_for_model: str = "redact_for_model"
    redact_for_logs: str = "redact_for_logs"


HOOK_POINTS = HookPoints()


@dataclass
class HookContext:
    point: str
    mode: str
    turn_ctx: Any
    state: "HarnessState"
    settings: "HarnessSettings"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolAuthorization:
    allow: bool = True
    reason: str = ""


@dataclass
class HookHandler:
    point: str
    fn: Callable[[HookContext], Any]
    name: str = ""
    priority: int = 100
    timeout_s: float | None = None
    fail_open: bool = True


class HookRegistry:
    """Ordered hook registry with optional timeout and fail-open semantics."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[HookHandler]] = {}

    def register(
        self,
        point: str,
        fn: Callable[[HookContext], Any],
        *,
        name: str = "",
        priority: int = 100,
        timeout_s: float | None = None,
        fail_open: bool = True,
    ) -> None:
        handlers = self._handlers.setdefault(point, [])
        handlers.append(
            HookHandler(
                point=point,
                fn=fn,
                name=name or getattr(fn, "__name__", "anonymous_hook"),
                priority=priority,
                timeout_s=timeout_s,
                fail_open=fail_open,
            )
        )
        handlers.sort(key=lambda item: item.priority)

    def handlers_for(self, point: str) -> list[HookHandler]:
        return list(self._handlers.get(point, []))

    async def run(self, point: str, context: HookContext) -> list[Any]:
        outputs: list[Any] = []
        for handler in self.handlers_for(point):
            try:
                value = handler.fn(context)
                if asyncio.iscoroutine(value):
                    value = await asyncio.wait_for(value, timeout=handler.timeout_s) if handler.timeout_s else await value
                outputs.append(value)
            except Exception:
                if not handler.fail_open:
                    raise
        return outputs


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

    def render_system_prefix(self, settings: HarnessSettings | None = None) -> str:
        settings = settings or HarnessSettings()
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
    hook_registry: HookRegistry | None = None
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
        self._hook_registry = hooks.hook_registry or HookRegistry()

    async def run(self, turn_ctx: Any) -> Any:
        state = HarnessState()
        settings = self._hooks.get_settings()
        mode = self._mode_name()

        await self._run_hooks(HOOK_POINTS.before_turn, turn_ctx, state, settings, mode)
        await self._emit("harness.start", turn_ctx, mode, settings, state=state)

        if self.policy.enable_planning:
            started = time.time()
            await self._run_hooks(HOOK_POINTS.before_plan, turn_ctx, state, settings, mode)
            await self._plan(turn_ctx, state, settings, mode)
            await self._run_hooks(
                HOOK_POINTS.after_plan,
                turn_ctx,
                state,
                settings,
                mode,
                payload={"plan": list(state.plan)},
            )
            await self._emit(
                "harness.plan",
                turn_ctx,
                mode,
                settings,
                state=state,
                success=bool(state.plan),
                plan=state.plan,
                duration_ms=round((time.time() - started) * 1000, 1),
            )

        result: Any = None
        while True:
            exec_ctx = self._augment_ctx(turn_ctx, state, settings, mode)
            if self._hooks.before_execute:
                self._hooks.before_execute(exec_ctx, state, settings)
            await self._run_hooks(HOOK_POINTS.before_execute, exec_ctx, state, settings, mode)

            started = time.time()
            try:
                result = await self._engine.execute(exec_ctx)
            except Exception as exc:
                await self._run_hooks(
                    HOOK_POINTS.on_error,
                    exec_ctx,
                    state,
                    settings,
                    mode,
                    payload={"stage": "execute", "error": exc},
                )
                raise

            state.attempt += 1
            await self._run_hooks(
                HOOK_POINTS.after_execute,
                exec_ctx,
                state,
                settings,
                mode,
                payload={"result": result, "attempt": state.attempt},
            )
            await self._run_tool_hooks(exec_ctx, result, state, settings, mode)

            await self._emit(
                "harness.execute",
                turn_ctx,
                mode,
                settings,
                state=state,
                attempt=state.attempt,
                max_attempts=self.policy.effective_max_attempts(settings),
                success=True,
                duration_ms=round((time.time() - started) * 1000, 1),
            )

            if not self.policy.enable_reflection:
                break

            started = time.time()
            await self._run_hooks(
                HOOK_POINTS.before_reflect,
                turn_ctx,
                state,
                settings,
                mode,
                payload={"result": result},
            )
            verdict = await self._reflect(turn_ctx, result, state, settings, mode)
            await self._run_hooks(
                HOOK_POINTS.after_reflect,
                turn_ctx,
                state,
                settings,
                mode,
                payload={"result": result, "verdict": verdict},
            )

            await self._emit(
                "harness.reflect",
                turn_ctx,
                mode,
                settings,
                state=state,
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
                    state=state,
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

    async def _emit(
        self,
        event_type: str,
        turn_ctx: Any,
        mode: str,
        settings: HarnessSettings,
        *,
        state: HarnessState,
        **details: Any,
    ) -> None:
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

        redact_ctx = HookContext(
            point=HOOK_POINTS.redact_for_logs,
            mode=mode,
            turn_ctx=turn_ctx,
            state=state,
            settings=settings,
            payload={"event_type": event_type, "payload": payload},
        )
        redacted = await self._hook_registry.run(HOOK_POINTS.redact_for_logs, redact_ctx)
        for candidate in redacted:
            if isinstance(candidate, dict):
                payload = candidate

        await self._run_hooks(
            HOOK_POINTS.on_event,
            turn_ctx,
            state,
            settings,
            mode,
            payload={"event_type": event_type, "payload": payload},
        )

        await self._hooks.emit_event(HarnessEvent(event_type=event_type, details=payload))

    async def _plan(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings, mode: str) -> None:
        tool_names = [
            str((tool.get("function") or {}).get("name") or "")
            for tool in (getattr(turn_ctx, "tool_schemas", None) or [])[:15]
        ]
        prompt = (
            settings.plan_prompt_template
            .replace("{goal}", str(getattr(turn_ctx, "user_content", ""))[:300])
            .replace("{tools}", ", ".join(tool_names[:10]) or "none")
        )

        redactions = await self._run_hooks(
            HOOK_POINTS.redact_for_model,
            turn_ctx,
            state,
            settings,
            mode,
            payload={"prompt": prompt, "stage": "plan"},
        )
        for candidate in redactions:
            if isinstance(candidate, str):
                prompt = candidate
            elif isinstance(candidate, dict) and isinstance(candidate.get("prompt"), str):
                prompt = candidate["prompt"]

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

    async def _reflect(
        self,
        turn_ctx: Any,
        result: Any,
        state: HarnessState,
        settings: HarnessSettings,
        mode: str,
    ) -> dict[str, Any] | None:
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

        redactions = await self._run_hooks(
            HOOK_POINTS.redact_for_model,
            turn_ctx,
            state,
            settings,
            mode,
            payload={"prompt": prompt, "stage": "reflect"},
        )
        for candidate in redactions:
            if isinstance(candidate, str):
                prompt = candidate
            elif isinstance(candidate, dict) and isinstance(candidate.get("prompt"), str):
                prompt = candidate["prompt"]

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

    def _augment_ctx(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings, mode: str) -> Any:
        self._apply_metadata_hooks(turn_ctx, state, settings, mode)
        self._apply_filter_tools_hooks(turn_ctx, state, settings, mode)

        prefix = state.render_system_prefix(settings)
        prefix = self._apply_system_augment_hooks(prefix, turn_ctx, state, settings, mode)
        self._apply_user_augment_hooks(turn_ctx, state, settings, mode)

        if not prefix:
            return turn_ctx

        updated_prompt = prefix + "\n\n" + getattr(turn_ctx, "system_prompt", "")
        try:
            return replace(turn_ctx, system_prompt=updated_prompt)
        except TypeError:
            setattr(turn_ctx, "system_prompt", updated_prompt)
            return turn_ctx

    def _apply_metadata_hooks(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings, mode: str) -> None:
        results = self._run_sync_hooks(HOOK_POINTS.inject_turn_metadata, turn_ctx, state, settings, mode)
        for candidate in results:
            if isinstance(candidate, dict):
                meta = getattr(turn_ctx, "meta", {}) or {}
                meta.update(candidate)
                try:
                    setattr(turn_ctx, "meta", meta)
                except Exception:
                    continue

    def _apply_filter_tools_hooks(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings, mode: str) -> None:
        payload = {"tool_schemas": list(getattr(turn_ctx, "tool_schemas", []) or [])}
        results = self._run_sync_hooks(HOOK_POINTS.filter_tools, turn_ctx, state, settings, mode, payload=payload)
        for candidate in results:
            if isinstance(candidate, list):
                try:
                    setattr(turn_ctx, "tool_schemas", candidate)
                except Exception:
                    continue
            elif isinstance(candidate, dict) and isinstance(candidate.get("tool_schemas"), list):
                try:
                    setattr(turn_ctx, "tool_schemas", candidate["tool_schemas"])
                except Exception:
                    continue

    def _apply_system_augment_hooks(
        self,
        prefix: str,
        turn_ctx: Any,
        state: HarnessState,
        settings: HarnessSettings,
        mode: str,
    ) -> str:
        payload = {"system_prompt": getattr(turn_ctx, "system_prompt", "")}
        results = self._run_sync_hooks(HOOK_POINTS.augment_system_prompt, turn_ctx, state, settings, mode, payload=payload)
        rendered = prefix
        for candidate in results:
            if isinstance(candidate, str) and candidate:
                rendered = (rendered + "\n\n" + candidate).strip() if rendered else candidate
            elif isinstance(candidate, dict) and isinstance(candidate.get("system_prompt"), str):
                text = candidate["system_prompt"]
                rendered = (rendered + "\n\n" + text).strip() if rendered else text
        return rendered

    def _apply_user_augment_hooks(self, turn_ctx: Any, state: HarnessState, settings: HarnessSettings, mode: str) -> None:
        payload = {"user_content": getattr(turn_ctx, "user_content", "")}
        results = self._run_sync_hooks(HOOK_POINTS.augment_user_prompt, turn_ctx, state, settings, mode, payload=payload)
        for candidate in results:
            if isinstance(candidate, str) and candidate:
                current = getattr(turn_ctx, "user_content", "")
                try:
                    setattr(turn_ctx, "user_content", f"{current}\n\n{candidate}".strip())
                except Exception:
                    continue
            elif isinstance(candidate, dict) and isinstance(candidate.get("user_content"), str):
                try:
                    setattr(turn_ctx, "user_content", candidate["user_content"])
                except Exception:
                    continue

    async def _run_hooks(
        self,
        point: str,
        turn_ctx: Any,
        state: HarnessState,
        settings: HarnessSettings,
        mode: str,
        payload: dict[str, Any] | None = None,
    ) -> list[Any]:
        context = HookContext(
            point=point,
            mode=mode,
            turn_ctx=turn_ctx,
            state=state,
            settings=settings,
            payload=payload or {},
        )
        return await self._hook_registry.run(point, context)

    def _run_sync_hooks(
        self,
        point: str,
        turn_ctx: Any,
        state: HarnessState,
        settings: HarnessSettings,
        mode: str,
        payload: dict[str, Any] | None = None,
    ) -> list[Any]:
        context = HookContext(
            point=point,
            mode=mode,
            turn_ctx=turn_ctx,
            state=state,
            settings=settings,
            payload=payload or {},
        )

        outputs: list[Any] = []
        for handler in self._hook_registry.handlers_for(point):
            try:
                value = handler.fn(context)
                if asyncio.iscoroutine(value):
                    self._log(f"harness/hooks: skipped async hook in sync point {point} ({handler.name})")
                    continue
                outputs.append(value)
            except Exception:
                if not handler.fail_open:
                    raise
        return outputs

    async def _run_tool_hooks(
        self,
        turn_ctx: Any,
        result: Any,
        state: HarnessState,
        settings: HarnessSettings,
        mode: str,
    ) -> None:
        tools_called = getattr(result, "tools_called", []) or []
        for tool in tools_called:
            tool_name = str(tool.get("name", ""))
            arguments = tool.get("arguments", {})
            source = str(tool.get("source", "local"))

            auth_results = await self._run_hooks(
                HOOK_POINTS.authorize_tool_call,
                turn_ctx,
                state,
                settings,
                mode,
                payload={"tool_name": tool_name, "arguments": arguments, "source": source},
            )
            blocked = next(
                (
                    candidate
                    for candidate in auth_results
                    if isinstance(candidate, ToolAuthorization) and not candidate.allow
                ),
                None,
            )

            await self._run_hooks(
                HOOK_POINTS.before_tool_call,
                turn_ctx,
                state,
                settings,
                mode,
                payload={"tool_name": tool_name, "arguments": arguments, "source": source},
            )

            if blocked:
                await self._run_hooks(
                    HOOK_POINTS.on_tool_error,
                    turn_ctx,
                    state,
                    settings,
                    mode,
                    payload={
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "source": source,
                        "error": blocked.reason or "blocked by authorize_tool_call",
                        "retryable": False,
                    },
                )
                continue

            await self._run_hooks(
                HOOK_POINTS.after_tool_call,
                turn_ctx,
                state,
                settings,
                mode,
                payload={
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "source": source,
                    "result": tool,
                },
            )

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
