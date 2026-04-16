# c3-harness

Configurable planning/execute/reflection harness for agent turns.

## Features

- **Planning mode**: Generate execution plan before running tools
- **Reflection mode**: Evaluate result and retry if incomplete
- **Full mode**: Plan → execute → reflect → retry
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

## License

MIT
