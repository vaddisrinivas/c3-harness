from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

from c3_harness import (
    AgentHarness,
    C3HarnessRuntime,
    ConfigurableHarness,
    FullHarness,
    HarnessCapabilities,
    HarnessHooks,
    HarnessManager,
    HarnessPolicy,
    HarnessProviderConfig,
    HarnessSettings,
    HookContext,
    HookRegistry,
    IntegrationHarness,
    PlanningHarness,
    PRESETS,
    ReflectionHarness,
    ToolAuthorization,
)
from c3_harness.core import HOOK_POINTS, strip_code_fences
from c3_harness import harness as harness_module
from c3_harness.integration import AgentHarness as IntegrationAgentHarness
from c3_harness.harness_contract import (
    HarnessCapabilities as ContractCapabilities,
    HarnessProvider,
    HarnessProviderConfig as ContractProviderConfig,
    HarnessProviderFactory,
)


@dataclass
class TurnCtx:
    user_content: str = "goal"
    system_prompt: str = "system"
    tool_schemas: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    app_name: str = "app"


@dataclass
class TurnResult:
    reply: str = "done"
    tools_called: list[dict[str, Any]] = field(default_factory=list)
    reply_sent_via_tool: bool = False


class DummyLLM:
    def __init__(self, responses: list[str] | None = None, *, fail: bool = False) -> None:
        self.responses = responses or []
        self.fail = fail

    async def acompletion(self, **_: Any) -> Any:
        if self.fail:
            raise RuntimeError("llm failure")
        content = self.responses.pop(0) if self.responses else "[]"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class SequenceEngine:
    def __init__(self, outputs: list[Any], *, fail_on_call: int | None = None) -> None:
        self.outputs = outputs
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.last_ctx: Any = None

    async def execute(self, turn_ctx: Any) -> Any:
        self.calls += 1
        self.last_ctx = turn_ctx
        if self.fail_on_call == self.calls:
            raise RuntimeError("engine boom")
        if self.outputs:
            return self.outputs.pop(0)
        return TurnResult(reply="fallback")


class MinimalIntegrationHarness(IntegrationHarness):
    pass


class RuntimeHarnessStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def resolve_config(self, app_config=None):
        self.calls.append(("resolve", app_config))
        return HarnessProviderConfig(mode="planning")

    def should_use_turn_engine(self, app_config=None):
        self.calls.append(("should", app_config))
        return True

    def create_harness(self, llm, engine, app_config=None):
        self.calls.append(("create", app_config, llm, engine))
        return "provider"


def _make_hooks(registry: HookRegistry | None = None, events: list[Any] | None = None, logs: list[str] | None = None) -> HarnessHooks:
    return HarnessHooks(
        get_settings=lambda: HarnessSettings(side_effect_tools=("send_email",)),
        get_planner_model=lambda: "cheap-model",
        build_llm_kwargs=lambda: {"temperature": 0},
        get_sender=lambda ctx: getattr(ctx, "sender", ""),
        get_app_name=lambda ctx: getattr(ctx, "app_name", "unknown"),
        emit_event=(lambda event: _append_async(events, event)) if events is not None else None,
        hook_registry=registry,
        log=(lambda m: logs.append(m)) if logs is not None else None,
    )


async def _append_async(container: list[Any], item: Any) -> None:
    container.append(item)


def test_strip_code_fences() -> None:
    assert strip_code_fences("```json\n[1,2]\n```") == "[1,2]"
    assert strip_code_fences("plain") == "plain"


def test_hook_registry_order_timeout_and_fail_modes() -> None:
    registry = HookRegistry()
    seen: list[str] = []

    async def slow(_: HookContext):
        await asyncio.sleep(0.01)
        seen.append("slow")
        return "slow"

    def first(_: HookContext):
        seen.append("first")
        return "first"

    def broken(_: HookContext):
        raise ValueError("boom")

    registry.register(HOOK_POINTS.before_turn, slow, priority=20, timeout_s=0.1)
    registry.register(HOOK_POINTS.before_turn, first, priority=10)
    registry.register(HOOK_POINTS.before_turn, broken, priority=30, fail_open=True)

    ctx = HookContext(HOOK_POINTS.before_turn, "default", TurnCtx(), state=None, settings=HarnessSettings())
    out = asyncio.run(registry.run(HOOK_POINTS.before_turn, ctx))
    assert out == ["first", "slow"]
    assert seen == ["first", "slow"]

    strict = HookRegistry()
    strict.register(HOOK_POINTS.before_turn, broken, fail_open=False)
    with pytest.raises(ValueError):
        asyncio.run(strict.run(HOOK_POINTS.before_turn, ctx))


def test_default_mode_execution_and_sync_hook_skip_async() -> None:
    events: list[Any] = []
    logs: list[str] = []
    registry = HookRegistry()

    def add_system(_: HookContext):
        return "extra-system"

    async def async_user(_: HookContext):
        return "should-be-skipped"

    registry.register(HOOK_POINTS.augment_system_prompt, add_system)
    registry.register(HOOK_POINTS.augment_user_prompt, async_user)

    result = TurnResult(reply="ok")
    engine = SequenceEngine([result])
    llm = DummyLLM([])

    hooks = _make_hooks(registry=registry, events=events, logs=logs)
    harness = ConfigurableHarness(PRESETS["default"], llm, engine, hooks)
    ctx = TurnCtx()
    out = asyncio.run(harness.run(ctx))

    assert out.reply == "ok"
    assert "extra-system" in engine.last_ctx.system_prompt
    assert any("skipped async hook in sync point augment_user_prompt" in m for m in logs)
    assert any(evt.event_type == "harness.execute" for evt in events)


def test_full_mode_plan_reflect_retry_and_tool_hooks() -> None:
    events: list[Any] = []
    seen: list[str] = []
    registry = HookRegistry()

    def on_before_turn(ctx: HookContext):
        seen.append(ctx.point)
        return {"session": "abc"}

    def filter_tools(_: HookContext):
        return [{"function": {"name": "filtered_tool"}}]

    def redact_model(ctx: HookContext):
        return {"prompt": ctx.payload["prompt"].replace("goal", "goal-redacted")}

    def redact_logs(ctx: HookContext):
        payload = dict(ctx.payload["payload"])
        payload["redacted"] = True
        return payload

    def authorize(ctx: HookContext):
        if ctx.payload.get("tool_name") == "send_email":
            return ToolAuthorization(allow=False, reason="blocked")
        return ToolAuthorization(allow=True)

    registry.register(HOOK_POINTS.before_turn, on_before_turn)
    registry.register(HOOK_POINTS.filter_tools, filter_tools)
    registry.register(HOOK_POINTS.redact_for_model, redact_model)
    registry.register(HOOK_POINTS.redact_for_logs, redact_logs)
    registry.register(HOOK_POINTS.authorize_tool_call, authorize)
    registry.register(HOOK_POINTS.before_tool_call, lambda ctx: seen.append(f"before:{ctx.payload['tool_name']}"))
    registry.register(HOOK_POINTS.after_tool_call, lambda ctx: seen.append(f"after:{ctx.payload['tool_name']}"))
    registry.register(HOOK_POINTS.on_tool_error, lambda ctx: seen.append(f"toolerr:{ctx.payload['tool_name']}"))
    registry.register(HOOK_POINTS.on_event, lambda ctx: seen.append(f"event:{ctx.payload['event_type']}"))

    llm = DummyLLM(
        responses=[
            '["discover", "execute"]',
            '{"status":"incomplete","missing":["item-1"]}',
            '{"status":"done","missing":[]}',
        ]
    )
    engine = SequenceEngine(
        [
            TurnResult(
                reply="attempt1",
                tools_called=[
                    {"name": "lookup", "arguments": {"q": "x"}, "source": "mcp"},
                    {"name": "send_email", "arguments": {"to": "x"}, "source": "external"},
                ],
            ),
            TurnResult(reply="attempt2", tools_called=[{"name": "lookup", "arguments": {}, "source": "plugin"}]),
        ]
    )
    hooks = _make_hooks(registry=registry, events=events, logs=[])
    harness = ConfigurableHarness(PRESETS["full"], llm, engine, hooks)

    ctx = TurnCtx(tool_schemas=[{"function": {"name": "orig_tool"}}])
    out = asyncio.run(harness.run(ctx))

    assert out.reply == "attempt2"
    assert engine.calls == 2
    assert HOOK_POINTS.before_turn in seen
    assert engine.last_ctx.tool_schemas[0]["function"]["name"] == "filtered_tool"
    assert any(s.startswith("before:lookup") for s in seen)
    assert any(s.startswith("after:lookup") for s in seen)
    assert any(s.startswith("toolerr:send_email") for s in seen)
    assert any(getattr(evt, "details", {}).get("redacted") for evt in events)


def test_reflect_early_exits_and_invalid_verdict() -> None:
    hooks = _make_hooks(registry=HookRegistry(), events=[])

    llm = DummyLLM(responses=['["x"]', '"not-a-dict"'])
    engine = SequenceEngine([TurnResult(reply="x", tools_called=[{"name": "lookup"}])])
    harness = ConfigurableHarness(PRESETS["reflection"], llm, engine, hooks)
    out = asyncio.run(harness.run(TurnCtx()))
    assert out.reply == "x"

    llm2 = DummyLLM([])
    engine2 = SequenceEngine([TurnResult(reply_sent_via_tool=True)])
    harness2 = ConfigurableHarness(PRESETS["reflection"], llm2, engine2, hooks)
    out2 = asyncio.run(harness2.run(TurnCtx()))
    assert out2.reply_sent_via_tool is True

    # No reply and no tools short-circuit
    engine3 = SequenceEngine([TurnResult(reply="", tools_called=[])])
    harness3 = ConfigurableHarness(PRESETS["reflection"], DummyLLM([]), engine3, hooks)
    out3 = asyncio.run(harness3.run(TurnCtx()))
    assert out3.reply == ""


def test_error_paths_for_engine_and_strict_sync_hook() -> None:
    registry = HookRegistry()
    seen: list[str] = []

    def strict_bomb(_: HookContext):
        raise RuntimeError("strict")

    registry.register(HOOK_POINTS.augment_system_prompt, strict_bomb, fail_open=False)
    hooks = _make_hooks(registry=registry, events=[], logs=[])
    harness = ConfigurableHarness(PRESETS["default"], DummyLLM([]), SequenceEngine([TurnResult()]), hooks)
    with pytest.raises(RuntimeError):
        asyncio.run(harness.run(TurnCtx()))

    registry2 = HookRegistry()
    registry2.register(HOOK_POINTS.on_error, lambda _: seen.append("on_error"))
    hooks2 = _make_hooks(registry=registry2, events=[])
    harness2 = ConfigurableHarness(PRESETS["default"], DummyLLM([]), SequenceEngine([], fail_on_call=1), hooks2)
    with pytest.raises(RuntimeError):
        asyncio.run(harness2.run(TurnCtx()))
    assert "on_error" in seen


def test_plain_object_ctx_replace_fallback_and_plan_reflect_exceptions() -> None:
    class PlainCtx:
        def __init__(self):
            self.user_content = "goal"
            self.system_prompt = "sys"
            self.tool_schemas = []
            self.meta = {}
            self.app_name = "plain"

    registry = HookRegistry()
    registry.register(HOOK_POINTS.augment_system_prompt, lambda _: "forced-prefix")
    hooks = _make_hooks(registry=registry, events=[], logs=[])
    harness = ConfigurableHarness(PRESETS["full"], DummyLLM(fail=True), SequenceEngine([TurnResult(reply="ok")]), hooks)
    out = asyncio.run(harness.run(PlainCtx()))
    assert out.reply == "ok"


def test_give_up_and_before_execute_legacy_hook_and_mode_names() -> None:
    seen: list[str] = []
    events: list[Any] = []
    logs: list[str] = []
    hooks = _make_hooks(registry=HookRegistry(), events=events, logs=logs)
    hooks.before_execute = lambda *_: seen.append("legacy-before-exec")
    hooks.get_settings = lambda: HarnessSettings(max_attempts=1)

    llm = DummyLLM(
        responses=[
            '["plan"]',
            '{"status":"incomplete","missing":["still missing"]}',
        ]
    )
    engine = SequenceEngine([TurnResult(reply="attempt", tools_called=[{"name": "lookup"}])])
    harness = ConfigurableHarness(PRESETS["full"], llm, engine, hooks)
    asyncio.run(harness.run(TurnCtx()))
    assert "legacy-before-exec" in seen
    assert any(evt.event_type == "harness.gave_up" for evt in events)
    assert any("gave up after 1 attempts" in msg for msg in logs)

    planning_harness = ConfigurableHarness(PRESETS["planning"], DummyLLM(["[]"]), SequenceEngine([TurnResult()]), _make_hooks(HookRegistry(), [], []))
    assert asyncio.run(planning_harness.run(TurnCtx())) is not None
    reflection_harness = ConfigurableHarness(PRESETS["reflection"], DummyLLM([]), SequenceEngine([TurnResult(reply="r", tools_called=[{"name": "t"}])]), _make_hooks(HookRegistry(), [], []))
    assert asyncio.run(reflection_harness.run(TurnCtx())) is not None


def test_sync_hook_error_branches_for_setattr_paths() -> None:
    class RestrictedCtx:
        def __init__(self) -> None:
            object.__setattr__(self, "user_content", "goal")
            object.__setattr__(self, "system_prompt", "sys")
            object.__setattr__(self, "tool_schemas", [{"function": {"name": "x"}}])
            object.__setattr__(self, "meta", {})
            object.__setattr__(self, "app_name", "restricted")

        def __setattr__(self, name: str, value: Any) -> None:
            if name in {"meta", "tool_schemas", "user_content"}:
                raise RuntimeError("blocked setattr")
            object.__setattr__(self, name, value)

    registry = HookRegistry()
    registry.register(HOOK_POINTS.inject_turn_metadata, lambda _: {"k": "v"})
    registry.register(HOOK_POINTS.filter_tools, lambda _: [{"function": {"name": "y"}}])
    registry.register(HOOK_POINTS.filter_tools, lambda _: {"tool_schemas": [{"function": {"name": "z"}}]})
    registry.register(HOOK_POINTS.augment_system_prompt, lambda _: {"system_prompt": "sys-dict"})
    registry.register(HOOK_POINTS.augment_user_prompt, lambda _: "add-user-str")
    registry.register(HOOK_POINTS.augment_user_prompt, lambda _: {"user_content": "replacement"})

    harness = ConfigurableHarness(PRESETS["default"], DummyLLM([]), SequenceEngine([TurnResult()]), _make_hooks(registry=registry, events=[], logs=[]))
    out = asyncio.run(harness.run(RestrictedCtx()))
    assert out.reply == "done"


def test_integration_harness_variants_and_factory() -> None:
    async def llm(**_: Any):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])

    class CallableEngine:
        async def __call__(self, ctx):
            return TurnResult(reply=getattr(ctx, "user_content", "x"))

        async def execute(self, ctx):
            return await self.__call__(ctx)

    # Uses provided engine because it has execute
    integ = MinimalIntegrationHarness(policy=PRESETS["default"], llm=llm, tool_registry=CallableEngine())
    out = asyncio.run(integ.run(TurnCtx(user_content="hello")))
    assert out.reply == "hello"

    # from_config path
    agent = AgentHarness.from_config("planning", llm=llm, engine=CallableEngine(), hooks=_make_hooks(HookRegistry(), [], []))
    assert isinstance(agent, ConfigurableHarness)
    integration_agent = IntegrationAgentHarness.from_config("UNKNOWN", llm=llm, tool_registry=CallableEngine())
    assert isinstance(integration_agent, IntegrationHarness)

    # Convenience wrappers
    PlanningHarness(llm=llm, tool_registry=CallableEngine())
    ReflectionHarness(llm=llm, tool_registry=CallableEngine())
    full = FullHarness(llm=llm, tool_registry=CallableEngine())
    assert full._planner is full
    assert full._reflector is full

    # _create_engine error path
    with pytest.raises(NotImplementedError):
        MinimalIntegrationHarness(policy=HarnessPolicy(), llm=llm, tool_registry=object())
    # explicit engine path
    explicit = MinimalIntegrationHarness(policy=HarnessPolicy(), llm=llm, tool_registry=object(), engine=CallableEngine())
    assert asyncio.run(explicit.run(TurnCtx(user_content="explicit"))).reply == "explicit"
    # callable tool_registry path
    callable_registry = MinimalIntegrationHarness(policy=HarnessPolicy(), llm=llm, tool_registry=CallableEngine().__call__)
    assert callable(callable_registry._engine)


def test_c3_runtime_and_manager_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[Any] = []
    logs: list[str] = []

    def defaults():
        return {
            "harness_provider": "c3_harness",
            "harness": "planning",
            "harness_plan_prefix": "Plan: ",
            "harness_plan_prompt": "Prompt {goal} {tools}",
            "harness_reflect_prompt": "Reflect {goal} {tools_used} {response}",
            "harness_repair_prefix": "Fix:",
            "harness_repair_suffix": "Only missing",
            "harness_side_effect_warning": "No repeat {tools}",
            "harness_side_effect_tools": ["send_email"],
            "harness_max_attempts": 3,
            "harness_planner_timeout": 1.2,
            "harness_reflector_timeout": 1.3,
            "harness_plan_max_tokens": 77,
            "harness_reflect_max_tokens": 88,
            "use_turn_engine": False,
        }

    runtime = C3HarnessRuntime(
        defaults_getter=defaults,
        planner_defaults_getter=lambda: {"cheap_model": "nano"},
        logger=logs.append,
        audit_emitter=lambda evt: _append_async(events, evt),
    )

    monkeypatch.setenv("AZURE_AI_API_KEY", "k")
    monkeypatch.setenv("AZURE_AI_API_BASE", "b")

    assert runtime.planner_model() == "nano"
    assert runtime.llm_extra_kwargs() == {"api_key": "k", "api_base": "b"}

    settings = runtime.settings_from_config()
    assert settings.plan_prefix == "Plan: "
    assert settings.max_attempts == 3

    config = runtime.resolve_config({"name": "my-app", "harness_override": True, "harness_mode": "full", "harness_reason": "forced"})
    assert config.mode == "full"
    assert config.app_reason == "forced"
    assert runtime.should_use_turn_engine({"name": "x"}) is True

    class E:
        async def execute(self, _):
            return TurnResult(reply="ok")

    provider = runtime.create_harness(DummyLLM(["[]"]), E(), {"name": "x"})
    assert provider is not None

    bad_runtime = C3HarnessRuntime(
        defaults_getter=lambda: {"harness_provider": "other"},
        planner_defaults_getter=lambda: {},
        logger=lambda _: None,
        audit_emitter=lambda _: _append_async([], None),
    )
    with pytest.raises(ValueError):
        bad_runtime.create_harness(DummyLLM(), E(), {"name": "x"})

    # manager wrapper
    manager = HarnessManager(runtime)
    assert manager.resolve_config({"name": "m"}).provider == "c3_harness"
    assert manager.should_use_turn_engine({"name": "m"}) is True
    assert manager.create_provider(DummyLLM(["[]"]), E(), {"name": "m"}) is not None


def test_contract_exports_and_dataclasses() -> None:
    cap = HarnessCapabilities(supports_planning=True, provider_name="x")
    assert cap.supports_planning is True

    cfg = HarnessProviderConfig(mode="reflection")
    assert cfg.mode == "reflection"

    # contract aliases from harness_contract
    cap2 = ContractCapabilities(provider_name="y")
    cfg2 = ContractProviderConfig(provider="p")
    assert cap2.provider_name == "y"
    assert cfg2.provider == "p"

    # protocol runtime-check usage by assignment
    _provider: HarnessProvider | None = None
    _factory: HarnessProviderFactory | None = None
    assert _provider is None
    assert _factory is None


def test_env_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_AI_API_BASE", raising=False)
    runtime = C3HarnessRuntime(
        defaults_getter=lambda: {},
        planner_defaults_getter=lambda: {"model": "fallback"},
        logger=lambda _: None,
        audit_emitter=lambda _: _append_async([], None),
    )
    assert runtime.planner_model() == "fallback"
    assert runtime.llm_extra_kwargs() == {}


def test_c3_runtime_prefers_explicit_planner_model_and_module_exports() -> None:
    runtime = C3HarnessRuntime(
        defaults_getter=lambda: {"harness_planner_model": "explicit-model"},
        planner_defaults_getter=lambda: {"cheap_model": "ignored"},
        logger=lambda _: None,
        audit_emitter=lambda _: _append_async([], None),
    )
    assert runtime.planner_model() == "explicit-model"

    # ensure harness.py export surface is imported/executed
    assert "AgentHarness" in harness_module.__all__
    assert hasattr(harness_module, "HookRegistry")


def test_redact_for_model_string_branch_in_plan_and_reflect() -> None:
    registry = HookRegistry()
    registry.register(HOOK_POINTS.redact_for_model, lambda ctx: ctx.payload["prompt"] + " [string-redacted]")
    hooks = _make_hooks(registry=registry, events=[], logs=[])
    llm = DummyLLM(
        responses=[
            '["step1"]',
            '{"status":"done","missing":[]}',
        ]
    )
    engine = SequenceEngine([TurnResult(reply="ok", tools_called=[{"name": "lookup"}])])
    harness = ConfigurableHarness(PRESETS["full"], llm, engine, hooks)
    out = asyncio.run(harness.run(TurnCtx()))
    assert out.reply == "ok"
