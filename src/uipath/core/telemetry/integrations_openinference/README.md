# Telemetry Integrations OpenInference

**Simple wrapper around OpenInference with UiPath session context**

This package provides a minimal wrapper (~92 LOC) around [OpenInference](https://github.com/Arize-ai/openinference) that adds UiPath session context (session_id, thread_id) to all spans.

**Key Benefits:**
- ✅ **Minimal code** - Only 92 implementation lines (vs ~1,800 for full integrations)
- ✅ **Self-contained** - Zero dependencies on other telemetry modules
- ✅ **Simple API** - Only 4 public functions
- ✅ **Rich metadata** - Tokens, costs, models from OpenInference
- ✅ **Automatic instrumentation** - Zero LangChain/LangGraph code changes

## Quick Start

```python
from uipath.core.telemetry import init
from uipath.core.telemetry.integrations_openinference import (
    instrument_langchain,
    set_session_context,
)

# Initialize telemetry
init(enable_console_export=True)

# Instrument (adds UiPath features on top of OpenInference)
instrument_langchain()

# Set session context (optional)
set_session_context(session_id="session-123", thread_id="thread-456")

# Now ALL LangChain/LangGraph operations are automatically traced!
from langgraph.graph import StateGraph

builder = StateGraph(dict)
builder.add_node("process", lambda x: {"result": x["a"] * x["b"]})
builder.set_entry_point("process")
builder.set_finish_point("process")
graph = builder.compile()

# Automatic tracing with both OpenInference and UiPath attributes!
result = graph.invoke({"a": 5, "b": 3})
```

## What You Get

### From OpenInference (Delegated)

- ✅ **Automatic LangChain/LangGraph instrumentation** - Zero code changes needed
- ✅ **Rich metadata extraction** - Token counts, costs, model names
- ✅ **Node-level tracing** - Individual LangGraph node execution
- ✅ **OpenInference conventions** - llm.*, tool.*, embedding.* attributes
- ✅ **Production-tested** - Battle-tested by wider community

### From UiPath Layer (92 LOC)

- ✅ **UiPath session attributes** - session.id, thread.id
- ✅ **ContextVar-based** - Async-safe context propagation
- ✅ **Self-contained** - No dependencies on other telemetry modules

## Complete Example

See `tests/telemetry/integrations_openinference/test_e2e.py` for a comprehensive example with a real LangGraph calculator agent:

```python
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath.core.telemetry.integrations_openinference import (
    instrument_langchain,
    set_session_context,
)


class AgentState(TypedDict):
    """State for the calculator agent."""
    messages: Annotated[list, add_messages]
    current_value: float
    operation: str | None


def parse_input_node(state: AgentState) -> AgentState:
    """Parse the input message and extract operation."""
    messages = state.get("messages", [])
    if not messages:
        return {"operation": None, "current_value": 0.0}

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Simple parsing: "add 5", "multiply 3", etc.
    parts = content.lower().split()
    if len(parts) >= 2:
        operation = parts[0]
        try:
            value = float(parts[1])
            return {"operation": operation, "current_value": value}
        except ValueError:
            return {"operation": None, "current_value": 0.0}

    return {"operation": None, "current_value": 0.0}


def calculate_node(state: AgentState) -> AgentState:
    """Perform the calculation based on operation."""
    current_value = state.get("current_value", 0.0)
    operation = state.get("operation")
    messages = state.get("messages", [])

    # Get previous result if exists
    previous_result = 0.0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            try:
                previous_result = float(msg.content)
                break
            except ValueError:
                pass

    # Perform operation
    result = previous_result
    if operation == "add":
        result = previous_result + current_value
    elif operation == "subtract":
        result = previous_result - current_value
    elif operation == "multiply":
        result = previous_result * current_value
    elif operation == "divide" and current_value != 0:
        result = previous_result / current_value
    elif operation == "set":
        result = current_value

    return {
        "messages": [AIMessage(content=str(result))],
        "current_value": result,
    }


def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    operation = state.get("operation")
    if operation in ["add", "subtract", "multiply", "divide", "set"]:
        return "calculate"
    return "end"


# Setup instrumentation
provider = TracerProvider()
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))

instrument_langchain(tracer_provider=provider)
set_session_context(session_id="my-session", thread_id="my-thread")

# Build the calculator agent
builder = StateGraph(AgentState)
builder.add_node("parse", parse_input_node)
builder.add_node("calculate", calculate_node)
builder.set_entry_point("parse")
builder.add_conditional_edges(
    "parse",
    should_continue,
    {
        "calculate": "calculate",
        "end": "__end__",
    },
)
builder.add_edge("calculate", "__end__")
graph = builder.compile()

# Execute with state accumulation
operations = ["set 10", "add 5", "multiply 2", "subtract 3"]
state = {"messages": []}
results = []

for op in operations:
    state["messages"].append(HumanMessage(content=op))
    result = graph.invoke(state)
    results.append(result)
    state = {
        "messages": list(result["messages"]),
        "current_value": result.get("current_value", 0.0)
    }

# Verify telemetry export
spans = exporter.get_finished_spans()
print(f"Total spans: {len(spans)}")

# Verify OpenInference attributes
openinference_spans = [
    span for span in spans
    if span.attributes and "openinference.span.kind" in span.attributes
]
print(f"OpenInference spans: {len(openinference_spans)}")

# Verify UiPath session context
session_spans = [
    span for span in spans
    if span.attributes
    and span.attributes.get("session.id") == "my-session"
    and span.attributes.get("thread.id") == "my-thread"
]
print(f"UiPath session spans: {len(session_spans)}")
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  UiPath integrations_openinference (92 LOC)         │
│                                                      │
│  - Adds UiPath attributes (session.id, thread.id)  │
│  - Self-contained session context (ContextVar)      │
│                                                      │
│  Delegates to ↓                                      │
└──────────────────────────────────────────────────────┘
                      │
                      ↓
┌──────────────────────────────────────────────────────┐
│  OpenInference LangChainInstrumentor (external)      │
│                                                      │
│  - Automatic LangChain/LangGraph instrumentation    │
│  - Rich metadata extraction                          │
│  - Node-level tracing                                │
│  - OpenInference semantic conventions                │
└──────────────────────────────────────────────────────┘
```

**Principle:** Let OpenInference do the heavy lifting (DRY), we add UiPath session context (KISS).

## Session Context

Session and thread IDs are stored using ContextVar (async-safe) and automatically added to all spans:

```python
from uipath.core.telemetry.integrations_openinference import (
    set_session_context,
    clear_session_context,
)

# Set session context (propagates to all spans in this context)
set_session_context(session_id="user-123", thread_id="conv-456")

# All spans will now have:
# - session.id = "user-123"
# - thread.id = "conv-456"

# Clear context when done
clear_session_context()
```

## Attributes Reference

### UiPath Attributes (Added by Our Layer)

- `session.id` - Session identifier (from ContextVar)
- `thread.id` - Thread identifier (from ContextVar)

### OpenInference Attributes (Added by OpenInference)

Common attributes from OpenInference semantic conventions:

- `openinference.span.kind` - Span type (LLM, TOOL, CHAIN, etc.)
- `llm.model_name` - Model name (e.g., "gpt-4")
- `llm.token_count.prompt` - Input tokens
- `llm.token_count.completion` - Output tokens
- `llm.token_count.total` - Total tokens
- `llm.input_messages` - Input messages
- `llm.output_messages` - Output messages
- `tool.name` - Tool name
- `tool.description` - Tool description
- `embedding.model_name` - Embedding model
- `document.content` - Document text
- `document.id` - Document ID

See [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md) for full list.

## API Reference

### Instrumentation

```python
def instrument_langchain(**kwargs) -> None:
    """Instrument LangChain/LangGraph with OpenInference + UiPath features.

    Automatically traces LangChain and LangGraph operations with OpenInference
    attributes (llm.*, tool.*) plus UiPath attributes (session.id, thread.id).

    Args:
        **kwargs: Arguments passed to OpenInference LangChainInstrumentor
            (e.g., tracer_provider, skip_dep_check).

    Raises:
        ImportError: If openinference-instrumentation-langchain not installed.
        RuntimeError: If already instrumented.
    """

def uninstrument_langchain() -> None:
    """Remove LangChain/LangGraph instrumentation.

    Note: UiPathSpanProcessor remains attached (can't be removed from TracerProvider).

    Raises:
        RuntimeError: If not currently instrumented.
    """
```

### Session Context

```python
def set_session_context(
    session_id: str,
    thread_id: str | None = None,
) -> None:
    """Set session context for current execution context.

    Uses ContextVar for async-safe context propagation.

    Args:
        session_id: Session identifier.
        thread_id: Thread identifier (optional).
    """

def clear_session_context() -> None:
    """Clear session context.

    Resets both session_id and thread_id to None.
    """
```

## Testing

```bash
# Run all tests
pytest tests/telemetry/integrations_openinference/ -v

# Run e2e test
pytest tests/telemetry/integrations_openinference/test_e2e.py -v

# Run with coverage
pytest tests/telemetry/integrations_openinference/ --cov=src/uipath/core/telemetry/integrations_openinference
```

## Implementation Stats

**File Structure:**
- `__init__.py` - 114 lines (public API + docs)
- `_instrumentor.py` - 59 lines (OpenInference wrapper)
- `_span_processor.py` - 60 lines (UiPath attribute injection)
- `_session_context.py` - 72 lines (ContextVar session management)

**Total:**
- 305 total lines
- 92 implementation lines (excluding `__init__.py`)
- 127 comment/docstring lines
- 4 public exports

**Dependencies:**
- `opentelemetry-sdk>=1.38.0`
- `openinference-instrumentation-langchain>=0.1.55`

## Design Principles

### KISS (Keep It Simple, Stupid)

- **Single responsibility:** Only adds session context to OpenInference spans
- **No speculation:** Removed privacy config system (YAGNI)
- **Minimal API:** Only 4 public functions
- **Simple wrapper:** Delegates all heavy lifting to OpenInference

### DRY (Don't Repeat Yourself)

We **don't** reimplement what OpenInference provides:
- ❌ No LangChain instrumentation code
- ❌ No LangGraph instrumentation code
- ❌ No metadata extraction code

We **only** add what's unique to UiPath:
- ✅ Session context attributes (session.id, thread.id)
- ✅ Self-contained ContextVar-based context management

### YAGNI (You Aren't Gonna Need It)

Removed in this implementation:
- ❌ Privacy configuration system (was 147 lines, never needed)
- ❌ Privacy marker attributes (speculative feature)
- ❌ Unnecessary public getters (get_session_id, get_thread_id)
- ❌ is_instrumented() function (YAGNI)

## Migration Guide

### From integrations_full

**Before:**
```python
from uipath.core.telemetry.integrations_full._shared import set_session_context
from uipath.core.telemetry.integrations_full.langchain import instrument_langchain

instrument_langchain()
set_session_context(session_id="123")
```

**After:**
```python
from uipath.core.telemetry.integrations_openinference import (
    instrument_langchain,
    set_session_context,
)

instrument_langchain()
set_session_context(session_id="123")
```

**Benefits:**
- ✅ Simpler: ~92 LOC vs ~1,800 LOC
- ✅ Self-contained: No cross-module dependencies
- ✅ Same features: Session context + OpenInference attributes

## When to Use This

### ✅ Use integrations_openinference If:

1. Want **rich metadata** (tokens, costs, models) + UiPath session context
2. Want **node-level tracing** for LangGraph
3. Want **automatic instrumentation** (zero code changes)
4. Want **minimal maintenance** (92 LOC vs 1,800 LOC)
5. Okay with **one external dependency** (openinference-instrumentation-langchain)

### ❌ Don't Use If:

1. Need **zero dependencies** → Use `integrations_lite`
2. Need **custom extractors** → Use `integrations_full`
3. Running in **airgapped environment** → Use `integrations_lite` or `integrations_full`

## See Also

- [OpenInference Documentation](https://github.com/Arize-ai/openinference)
- [OpenInference Semantic Conventions](https://github.com/Arize-ai/openinference/blob/main/spec/semantic_conventions.md)
- [Arize Phoenix](https://phoenix.arize.com/) - OpenInference-compatible observability platform
