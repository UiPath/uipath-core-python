"""Tests for Observation classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from uipath.core.telemetry.client import init_client
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_set_attribute(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test set_attribute() sets span attributes.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.set_attribute("key1", "value1")
            obs.set_attribute("key2", 42)
            obs.set_attribute("key3", True)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    assert gen_span.attributes.get("key1") == "value1"
    assert gen_span.attributes.get("key2") == 42
    assert gen_span.attributes.get("key3") is True


def test_set_attribute_dict_serialization(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test set_attribute() serializes dicts to JSON.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    test_dict = {"nested": {"key": "value"}, "list": [1, 2, 3]}
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.set_attribute("dict_attr", test_dict)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    import json

    dict_attr = gen_span.attributes.get("dict_attr")
    assert dict_attr is not None
    # Should be JSON string
    parsed = json.loads(dict_attr)
    assert parsed == test_dict


def test_set_attribute_list_serialization(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test set_attribute() serializes lists to JSON.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    test_list = ["a", "b", {"key": "value"}]
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.set_attribute("list_attr", test_list)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    import json

    list_attr = gen_span.attributes.get("list_attr")
    assert list_attr is not None
    # Should be JSON string
    parsed = json.loads(list_attr)
    assert parsed == test_list


def test_set_status(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test set_status() sets span status.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.set_status(StatusCode.ERROR, "Something went wrong")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    assert gen_span.status.status_code == StatusCode.ERROR
    assert gen_span.status.description == "Something went wrong"


def test_record_exception(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test record_exception() records exception event.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    test_error = ValueError("test error")
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.record_exception(test_error, attributes={"custom": "attr"})

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Check exception event
    events = gen_span.events
    assert len(events) == 1
    assert events[0].name == "exception"
    assert events[0].attributes.get("exception.type") == "ValueError"
    assert "test error" in events[0].attributes.get("exception.message", "")


def test_observation_method_chaining(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test observation methods can be chained.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.set_attribute("a", 1).set_attribute("b", 2).set_status(StatusCode.OK)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    assert gen_span.attributes.get("a") == 1
    assert gen_span.attributes.get("b") == 2
    assert gen_span.status.status_code == StatusCode.OK


def test_generation_observation_update(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test GenerationObservation.update() with mock response.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Mock OpenAI-like response
    mock_response = {
        "model": "gpt-4",
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        "choices": [{"message": {"content": "Hello!"}}],
    }

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("llm-call", kind="generation") as obs:
            obs.update(mock_response)

    # Verify - parser should extract attributes
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "llm-call")

    # Check if any attributes were set by parser
    # (Exact attributes depend on parser implementation)
    assert len(gen_span.attributes) > 1  # More than just openinference.span.kind


def test_observation_exit_on_exception(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test observation records exception when raised in context.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute - raise in observation
    with pytest.raises(ValueError, match="test error"):
        with Trace(tracer, "root") as trace_ctx:
            with trace_ctx.span("test", kind="generation"):
                raise ValueError("test error")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Exception should be recorded
    assert gen_span.status.status_code == StatusCode.ERROR
    events = gen_span.events
    assert len(events) == 1
    assert events[0].name == "exception"


def test_tool_observation_specific(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test ToolObservation-specific attributes.

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
            obs.set_attribute("input", "2+2")
            obs.set_attribute("output", "4")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    tool_span = next(s for s in spans if s.name == "calculator")

    assert tool_span.attributes.get("openinference.span.kind") == "TOOL"
    assert tool_span.attributes.get("input") == "2+2"
    assert tool_span.attributes.get("output") == "4"


def test_agent_observation_specific(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test AgentObservation-specific attributes.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("reasoning-agent", kind="agent") as obs:
            obs.set_attribute("thought", "I need to analyze this")
            obs.set_attribute("action", "search")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "reasoning-agent")

    assert agent_span.attributes.get("openinference.span.kind") == "AGENT"
    assert agent_span.attributes.get("thought") == "I need to analyze this"
    assert agent_span.attributes.get("action") == "search"


def test_embedding_observation_with_model(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test embedding span with kind attribute.

    Model metadata should come from update() with actual response in real usage.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("embed", kind="embedding") as obs:
            obs.set_attribute("input_text", "hello world")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    embed_span = next(s for s in spans if s.name == "embed")

    assert embed_span.attributes.get("openinference.span.kind") == "EMBEDDING"
    assert embed_span.attributes.get("input_text") == "hello world"


def test_observation_update_unknown_type(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test update() with unknown type doesn't crash.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute - update with unknown object
    class UnknownType:
        """Unknown type for testing."""

        pass

    unknown_obj = UnknownType()

    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            # Should not crash
            obs.update(unknown_obj)

    # Verify - span still exported
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 2  # root + test
