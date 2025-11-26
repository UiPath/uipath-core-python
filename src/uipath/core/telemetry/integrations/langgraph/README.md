# LangGraph Integration

Automatic instrumentation for LangGraph workflows using OpenTelemetry callbacks.

## Quick Start

```python
from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

# Install once - zero code modification!
LangGraphInstrumentor().instrument()

# Normal LangGraph code - automatically traced!
from langgraph.graph import StateGraph

graph = StateGraph(...)
graph.add_node("agent", my_agent_function)
graph.add_node("tools", my_tools_function)
graph.add_edge("agent", "tools")

app = graph.compile()
result = app.invoke({"messages": ["Research AI trends"]})
```

## Features

- **Zero-code instrumentation**: Install once, trace all workflows
- **Automatic span creation**: Each node execution becomes a span
- **State tracking**: Captures input/output state with privacy controls
- **Semantic types**: Auto-detects AGENT, TOOL, LLM, CHAIN spans
- **Error handling**: Captures exceptions with stack traces
- **Parent-child hierarchy**: Maintains graph execution structure

## Manual Control

For advanced use cases, you can use the `LangGraphTracer` directly:

```python
from uipath.core import otel
from uipath.core.otel.integrations.langgraph import LangGraphTracer

otel.init(mode="auto")

tracer = LangGraphTracer(
    otel_client=otel.get_client(),
    trace_state=True,      # Capture state changes
    trace_edges=True,      # Capture edge transitions
    max_state_size=10_000, # 10KB limit
)

app = graph.compile()
result = app.invoke(
    {"messages": ["Hello"]},
    config={"callbacks": [tracer]}
)
```

## State Filtering

By default, all state is captured up to 10KB. For privacy-sensitive workflows:

```python
# Option 1: Disable state tracking
tracer = LangGraphTracer(otel_client=otel.get_client(), trace_state=False)

# Option 2: Reduce size limit
tracer = LangGraphTracer(otel_client=otel.get_client(), max_state_size=1_000)

# States larger than max_state_size will be replaced with:
# {"_truncated": True, "_size": 15000, "_limit": 10000}
```

## Span Attributes

Each LangGraph node span includes:

- `openinference.span.kind`: Semantic type (CHAIN, AGENT, TOOL, LLM, RETRIEVER)
- `node.name`: LangGraph node name
- `input.value`: Input state (JSON)
- `output.value`: Output state (JSON)
- `execution.id`: UiPath execution ID (if set)

## Mixing with Manual Instrumentation

Combine automatic instrumentation with manual wrappers:

```python
from uipath.core import otel
from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

# Automatic instrumentation for graph structure
LangGraphInstrumentor().instrument()

# Manual instrumentation for custom logic
@otel.agent()
def complex_agent_node(state):
    with otel.trace("sub-workflow") as trace:
        result = do_complex_work(state)
        trace.set_attribute("complexity", "high")
    return result

graph.add_node("agent", complex_agent_node)
app = graph.compile()
result = app.invoke({"messages": ["Hello"]})
```

## Implementation Details

The instrumentor works by:

1. Wrapping `StateGraph.compile()` to intercept compilation
2. Injecting a `LangGraphTracer` callback into the compiled app's `invoke()` method
3. Using LangChain's callback system to receive node start/end/error events
4. Creating OpenTelemetry spans for each node execution
5. Capturing state changes as span attributes with size limits

## Troubleshooting

**Issue**: Duplicate spans when using manual + auto instrumentation

**Solution**: Use OpenTelemetry's standard suppression API to disable auto-instrumentation selectively:

```python
from opentelemetry import context
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY

def my_node(state):
    # Standard OpenTelemetry suppression mechanism
    token = context.attach(context.set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        # This won't be traced by the instrumentor
        result = external_api_call()
    finally:
        context.detach(token)
    return result
```

**Issue**: State not captured

**Solution**: Ensure `trace_state=True` (default) and check state size:

```python
tracer = LangGraphTracer(
    otel_client=otel.get_client(),
    trace_state=True,
    max_state_size=50_000,  # Increase if needed
)
```

## Requirements

- `langgraph >= 0.1.0`
- `opentelemetry-sdk >= 1.20.0`
- `uipath-core >= 1.0.0` (with otel module)

## See Also

- [UiPath OTel User Guide](../../../docs/otel-user-guide.md)
- [Integrations README](../README.md)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
