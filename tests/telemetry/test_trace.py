"""Tests for Trace context manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from uipath.core.telemetry.client import init_client
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_trace_creation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test trace creation with metadata.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    metadata = {"custom_key": "custom_value"}
    with Trace(
        tracer,
        "test-trace",
        execution_id="exec-123",
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
    assert root_span.attributes.get("custom_key") == "custom_value"


def test_trace_exit_on_exception(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test trace records exception when raised.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
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
    """Test span() with kind=generation creates observation with correct attributes.

    Metadata like model name should come from update() with actual response,
    not from kwargs.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("llm-call", kind="generation") as obs:
            obs.set_attribute("prompt", "test prompt")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "llm-call")

    assert gen_span.attributes.get("openinference.span.kind") == "GENERATION"
    assert gen_span.attributes.get("prompt") == "test prompt"


def test_tool_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test span() with kind=tool creates observation with correct attributes.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("calculator", kind="tool") as obs:
            obs.set_attribute("operation", "add")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    tool_span = next(s for s in spans if s.name == "calculator")

    assert tool_span.attributes.get("openinference.span.kind") == "TOOL"
    assert tool_span.attributes.get("operation") == "add"


def test_agent_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test span() with kind=agent creates observation with correct attributes.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("my-agent", kind="agent") as obs:
            obs.set_attribute("task", "reasoning")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "my-agent")

    assert agent_span.attributes.get("openinference.span.kind") == "AGENT"
    assert agent_span.attributes.get("task") == "reasoning"


def test_retriever_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test retriever() creates RetrieverObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("vector-search", kind="retriever"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    retriever_span = next(s for s in spans if s.name == "vector-search")

    assert retriever_span.attributes.get("openinference.span.kind") == "RETRIEVER"


def test_embedding_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test span() with kind=embedding creates observation with correct attributes.

    Metadata like model name should come from update() with actual response,
    not from kwargs.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("embed-text", kind="embedding"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    embed_span = next(s for s in spans if s.name == "embed-text")

    assert embed_span.attributes.get("openinference.span.kind") == "EMBEDDING"


def test_workflow_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test workflow() creates WorkflowObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("my-workflow", kind="workflow"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    workflow_span = next(s for s in spans if s.name == "my-workflow")

    assert workflow_span.attributes.get("openinference.span.kind") == "WORKFLOW"


def test_activity_observation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test activity() creates ActivityObservation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("my-activity", kind="activity"):
            pass

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    activity_span = next(s for s in spans if s.name == "my-activity")

    assert activity_span.attributes.get("openinference.span.kind") == "ACTIVITY"
