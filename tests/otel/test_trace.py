"""Tests for Trace context manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from uipath.core.otel.client import OTelClient, OTelConfig
from uipath.core.otel.trace import Trace, get_current_trace, require_trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_trace_creation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test trace creation with metadata.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    metadata = {"custom_key": "custom_value"}
    with Trace(
        tracer,
        "test-trace",
        execution_id="exec-123",
        user_id="user-456",
        metadata=metadata,
    ):
        pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1

    root_span = spans[0]
    assert root_span.name == "test-trace"
    assert root_span.attributes.get("execution.id") == "exec-123"
    assert root_span.attributes.get("user.id") == "user-456"
    assert root_span.attributes.get("custom_key") == "custom_value"


def test_trace_exit_on_exception(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test trace records exception when raised.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute - raise exception in trace
    with pytest.raises(ValueError, match="test error"):
        with Trace(tracer, "error-trace"):
            raise ValueError("test error")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1

    root_span = spans[0]
    assert root_span.status.status_code == StatusCode.ERROR

    # Verify exception recorded
    events = root_span.events
    assert len(events) == 1
    assert events[0].name == "exception"


def test_generation_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test generation() creates GenerationObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.generation("llm-call", model="gpt-4") as obs:
            obs.set_attribute("prompt", "test prompt")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "llm-call")

    assert gen_span.attributes.get("span.type") == "generation"
    assert gen_span.attributes.get("gen_ai.request.model") == "gpt-4"
    assert gen_span.attributes.get("prompt") == "test prompt"


def test_tool_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test tool() creates ToolObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.tool("calculator") as obs:
            obs.set_attribute("operation", "add")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    tool_span = next(s for s in spans if s.name == "calculator")

    assert tool_span.attributes.get("span.type") == "tool"
    assert tool_span.attributes.get("tool.name") == "calculator"
    assert tool_span.attributes.get("operation") == "add"


def test_agent_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test agent() creates AgentObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.agent("my-agent") as obs:
            obs.set_attribute("task", "reasoning")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "my-agent")

    assert agent_span.attributes.get("span.type") == "agent"
    assert agent_span.attributes.get("agent.name") == "my-agent"


def test_retriever_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test retriever() creates RetrieverObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.retriever("vector-search"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    retriever_span = next(s for s in spans if s.name == "vector-search")

    assert retriever_span.attributes.get("span.type") == "retriever"


def test_embedding_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test embedding() creates EmbeddingObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.embedding("embed-text", model="text-embedding-ada-002"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    embed_span = next(s for s in spans if s.name == "embed-text")

    assert embed_span.attributes.get("span.type") == "embedding"
    assert embed_span.attributes.get("gen_ai.request.model") == "text-embedding-ada-002"


def test_workflow_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test workflow() creates WorkflowObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.workflow("my-workflow"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    workflow_span = next(s for s in spans if s.name == "my-workflow")

    assert workflow_span.attributes.get("span.type") == "workflow"


def test_activity_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test activity() creates ActivityObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.activity("my-activity"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    activity_span = next(s for s in spans if s.name == "my-activity")

    assert activity_span.attributes.get("span.type") == "activity"


def test_get_url(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test get_url() returns trace viewer URL.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "test") as trace_ctx:
        url = trace_ctx.get_url()

    # Verify URL format
    assert url.startswith("https://telemetry.uipath.com/trace/")
    assert len(url) > len("https://telemetry.uipath.com/trace/")


def test_ambient_trace_context(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test ambient trace context is accessible.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Before trace context
    assert get_current_trace() is None

    # Inside trace context
    with Trace(tracer, "ambient-test") as trace_ctx:
        retrieved = get_current_trace()
        assert retrieved is not None
        assert retrieved is trace_ctx

        # Can use retrieved trace to create observations
        with retrieved.generation("nested"):
            pass

    # After trace context
    assert get_current_trace() is None

    # Verify nested observation created
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 2
    assert any(s.name == "nested" for s in spans)


def test_require_trace_success(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test require_trace() returns trace inside context.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "test") as trace_ctx:
        required = require_trace()
        assert required is trace_ctx


def test_require_trace_fails_outside_context() -> None:
    """Test require_trace() raises error outside context."""
    with pytest.raises(RuntimeError, match="No active trace context"):
        require_trace()


def test_nested_traces_isolated(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test nested trace contexts maintain isolation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute - create two separate traces
    with Trace(tracer, "trace1", execution_id="exec1"):
        with Trace(tracer, "trace2", execution_id="exec2"):
            # Inner trace should be current
            current = get_current_trace()
            assert current is not None
            assert current._name == "trace2"

        # Outer trace restored
        current = get_current_trace()
        assert current is not None
        assert current._name == "trace1"

    # Both traces ended
    assert get_current_trace() is None

    # Verify both root spans exported
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 2

    trace1_span = next(s for s in spans if s.name == "trace1")
    trace2_span = next(s for s in spans if s.name == "trace2")

    assert trace1_span.attributes.get("execution.id") == "exec1"
    assert trace2_span.attributes.get("execution.id") == "exec2"
