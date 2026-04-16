# c3-harness

Configurable planning/execute/reflection harness for agent turns.

## Features

- **Planning mode**: Generate execution plan before running tools
- **Reflection mode**: Evaluate result and retry if incomplete
- **Full mode**: Plan → execute → reflect → retry
- **Hook-first extension model**: Add lifecycle, tool, context, and policy hooks
- **Framework-agnostic**: Integrate with any agent framework
- **Async-first**: Built for async/await patterns

## Installation

```bash
pip install c3-harness
# Or for development:
uv pip install -e /path/to/harness
```

## Quick Start

```python
from c3_harness import AgentHarness

# Define your LLM function
async def my_llm(messages, **kwargs):
    # Call your LLM here (OpenAI, Anthropic, etc.)
    return response

# Define your tool registry
tools = {
    "search": lambda query: search_db(query),
    "write": lambda path, content: write_file(path, content),
}

# Create a planning harness
harness = AgentHarness.from_config(
    mode="planning",
    llm=my_llm,
    tool_registry=tools,
)

# Run a turn
result = await harness.run(turn_context={
    "goal": "Find the latest sales report and summarize it",
    "tools": list(tools.keys()),
})
```

## Integration

### Basic Integration

```python
from c3_harness import IntegrationHarness, PRESETS

class MyHarness(IntegrationHarness):
    def _create_engine(self, tool_registry):
        # Wrap your framework's tool execution
        class Engine:
            def execute(self, tool_calls):
                return [self._run_tool(tc) for tc in tool_calls]
        return Engine(tool_registry)

    def _run_tool(self, tool_call):
        # Execute a single tool call
        fn = self._engine.registry[tool_call.name]
        return fn(**tool_call.arguments)

harness = MyHarness(
    policy=PRESETS["full"],
    llm=my_llm,
    tool_registry=my_tools,
)
```

### Provider Implementation

For advanced integrations, implement the `HarnessProvider` protocol:

```python
from c3_harness import HarnessProvider, HarnessCapabilities

class MyProvider:
    def __init__(self, config, llm, engine, hooks):
        self.config = config
        self._harness = self._create_harness(config.mode)

    async def run(self, turn_ctx):
        return await self._harness.run(turn_ctx)

    def capabilities(self):
        return HarnessCapabilities(
            supports_planning=True,
            supports_reflection=True,
            provider_name="my_provider",
        )
```

## Modes

| Mode | Planning | Reflection | Description |
|------|----------|------------|-------------|
| `default` | ❌ | ❌ | Direct execution, no orchestration |
| `planning` | ✅ | ❌ | Plan before executing |
| `reflection` | ❌ | ✅ | Evaluate and retry if incomplete |
| `full` | ✅ | ✅ | Plan → execute → reflect → retry |

## Configuration

```python
from c3_harness import HarnessSettings, HarnessHooks

settings = HarnessSettings(
    plan_prefix="Plan: ",
    max_attempts=3,
    planner_timeout_s=5.0,
    side_effect_tools=("send_message", "send_email"),
)

hooks = HarnessHooks(
    get_settings=lambda: settings,
    get_planner_model=lambda: "gpt-4o-mini",
    build_llm_kwargs=lambda: {"temperature": 0.7},
)

harness = AgentHarness.from_config(
    mode="full",
    llm=my_llm,
    tool_registry=my_tools,
    hooks=hooks,
)
```

## Hook Registry

```python
from c3_harness import HarnessHooks, HarnessSettings, HookRegistry, HOOK_POINTS, ToolAuthorization

registry = HookRegistry()

# Context hook
registry.register(
    HOOK_POINTS.augment_system_prompt,
    lambda ctx: "Prefer calling MCP tools before local fallbacks.",
    priority=10,
)

# Policy hook
def authorize(ctx):
    tool = ctx.payload.get("tool_name", "")
    if tool == "send_email":
        return ToolAuthorization(allow=False, reason="email disabled in this environment")
    return ToolAuthorization(allow=True)

registry.register(HOOK_POINTS.authorize_tool_call, authorize, priority=5)

hooks = HarnessHooks(
    get_settings=lambda: HarnessSettings(),
    get_planner_model=lambda: "gpt-4o-mini",
    hook_registry=registry,
)
```

Available hook points include:
- `before_turn`, `before_plan`, `after_plan`, `before_execute`, `after_execute`, `before_reflect`, `after_reflect`, `on_error`, `on_event`
- `before_tool_call`, `after_tool_call`, `on_tool_error`
- `augment_system_prompt`, `augment_user_prompt`, `filter_tools`, `inject_turn_metadata`
- `authorize_tool_call`, `redact_for_model`, `redact_for_logs`

## License

MIT
