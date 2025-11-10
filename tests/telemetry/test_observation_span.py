"""Tests for ObservationSpan wrapper and semantic methods."""

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from uipath.core.telemetry import TelemetryClient

if TYPE_CHECKING:
    pass


def test_observation_span_set_attribute(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test basic attribute setting."""
    with telemetry_client.start_as_current_span("test_span") as span:
        span.set_attribute("key1", "value1")
        span.set_attribute("key2", 42)
        span.set_attribute("key3", 3.14)
        span.set_attribute("key4", True)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert span_data.attributes["key1"] == "value1"
    assert span_data.attributes["key2"] == 42
    assert span_data.attributes["key3"] == 3.14
    assert span_data.attributes["key4"] is True


def test_observation_span_set_attributes(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test bulk attribute setting."""
    with telemetry_client.start_as_current_span("test_span") as span:
        span.set_attributes(
            {
                "attr1": "value1",
                "attr2": 100,
                "attr3": False,
            }
        )

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert span_data.attributes["attr1"] == "value1"
    assert span_data.attributes["attr2"] == 100
    assert span_data.attributes["attr3"] is False


def test_observation_span_update_input(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test input recording (JSON serialization)."""
    with telemetry_client.start_as_current_span("test_span") as span:
        span.update_input({"param1": "value1", "param2": 42})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert "input" in span_data.attributes
    # Should be JSON-serialized
    import json

    input_data = json.loads(span_data.attributes["input"])
    assert input_data["param1"] == "value1"
    assert input_data["param2"] == 42


def test_observation_span_update_output(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test output recording (JSON serialization)."""
    with telemetry_client.start_as_current_span("test_span") as span:
        span.update_output({"result": "success", "count": 10})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert "output" in span_data.attributes
    # Should be JSON-serialized
    import json

    output_data = json.loads(span_data.attributes["output"])
    assert output_data["result"] == "success"
    assert output_data["count"] == 10


def test_observation_span_hide_input(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test input hiding (privacy control)."""
    with telemetry_client.start_as_current_span("test_span", hide_input=True) as span:
        span.update_input({"sensitive": "password123"})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    # Input should NOT be recorded
    assert "input" not in span_data.attributes


def test_observation_span_hide_output(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test output hiding (privacy control)."""
    with telemetry_client.start_as_current_span("test_span", hide_output=True) as span:
        span.update_output({"secret": "api-key-xyz"})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    # Output should NOT be recorded
    assert "output" not in span_data.attributes


def test_observation_span_fail(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test manual error marking."""
    with telemetry_client.start_as_current_span("test_span") as span:
        try:
            raise ValueError("Test error")
        except ValueError as e:
            span.fail(e)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert span_data.status.status_code == StatusCode.ERROR
    assert "Test error" in span_data.status.description

    # Should have recorded exception event
    assert len(span_data.events) == 1
    event = span_data.events[0]
    assert event.name == "exception"


def test_observation_span_automatic_exception_recording(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test automatic exception recording on context exit."""
    with pytest.raises(RuntimeError):
        with telemetry_client.start_as_current_span("test_span"):
            raise RuntimeError("Automatic exception")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    # Should automatically mark as error
    assert span_data.status.status_code == StatusCode.ERROR
    assert "Automatic exception" in span_data.status.description

    # Should have recorded exception event(s)
    # Note: Both ObservationSpan and OTel context manager record exceptions
    assert len(span_data.events) >= 1
    assert any(event.name == "exception" for event in span_data.events)


def test_observation_span_end(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test manual span ending (non-context-manager usage)."""
    span = telemetry_client.start_as_current_span("test_span")
    span.__enter__()
    span.set_attribute("key", "value")
    span.end()

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert span_data.name == "test_span"
    assert span_data.attributes["key"] == "value"


def test_observation_span_span_id(
    telemetry_client: TelemetryClient,
):
    """Test span_id property (debugging utility)."""
    with telemetry_client.start_as_current_span("test_span") as span:
        span_id = span.span_id
        # Should be hex string (16 characters for 64-bit span ID)
        assert isinstance(span_id, str)
        assert len(span_id) == 16
        assert all(c in "0123456789abcdef" for c in span_id)


def test_observation_span_trace_id(
    telemetry_client: TelemetryClient,
):
    """Test trace_id property (debugging utility)."""
    with telemetry_client.start_as_current_span("test_span") as span:
        trace_id = span.trace_id
        # Should be hex string (32 characters for 128-bit trace ID)
        assert isinstance(trace_id, str)
        assert len(trace_id) == 32
        assert all(c in "0123456789abcdef" for c in trace_id)


def test_observation_span_complex_serialization(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test JSON serialization of complex objects."""
    with telemetry_client.start_as_current_span("test_span") as span:
        # Dict with nested structures
        span.set_attribute("config", {"timeout": 30, "retry": {"max": 3, "delay": 1.5}})
        # List
        span.set_attribute("tags", ["python", "telemetry", "otel"])

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]

    # Dict should be JSON-serialized
    import json

    config = json.loads(span_data.attributes["config"])
    assert config["timeout"] == 30
    assert config["retry"]["max"] == 3

    # List should be preserved (OpenTelemetry converts to tuple)
    tags = span_data.attributes["tags"]
    assert list(tags) == ["python", "telemetry", "otel"]


def test_observation_span_runtime_guard_set_attribute(
    telemetry_client: TelemetryClient,
):
    """Test that set_attribute raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        span.set_attribute("key", "value")


def test_observation_span_runtime_guard_update_input(
    telemetry_client: TelemetryClient,
):
    """Test that update_input raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        span.update_input({"data": "test"})


def test_observation_span_runtime_guard_update_output(
    telemetry_client: TelemetryClient,
):
    """Test that update_output raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        span.update_output({"result": "success"})


def test_observation_span_runtime_guard_fail(
    telemetry_client: TelemetryClient,
):
    """Test that fail raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        span.fail(ValueError("test"))


def test_observation_span_runtime_guard_end(
    telemetry_client: TelemetryClient,
):
    """Test that end raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        span.end()


def test_observation_span_runtime_guard_span_id(
    telemetry_client: TelemetryClient,
):
    """Test that span_id property raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        _ = span.span_id


def test_observation_span_runtime_guard_trace_id(
    telemetry_client: TelemetryClient,
):
    """Test that trace_id property raises RuntimeError when span not active."""
    span = telemetry_client.start_as_current_span("test_span")

    # Should raise RuntimeError before __enter__
    with pytest.raises(RuntimeError, match="Span not active"):
        _ = span.trace_id


def test_observation_span_methods_work_after_enter(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test that all methods work correctly after __enter__."""
    span = telemetry_client.start_as_current_span("test_span")

    # Enter context
    span.__enter__()

    try:
        # All methods should work fine inside context
        span.set_attribute("key", "value")
        span.update_input({"data": "test"})
        span.update_output({"result": "success"})
        _ = span.span_id
        _ = span.trace_id

        # Manual exception marking
        try:
            raise ValueError("test error")
        except ValueError as e:
            span.fail(e)
    finally:
        # Exit context
        span.__exit__(None, None, None)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1
    span_data = spans[0]
    assert span_data.attributes.get("key") == "value"
    assert "input" in span_data.attributes
    assert "output" in span_data.attributes
    assert span_data.status.status_code == StatusCode.ERROR


def test_invalid_span_with_nullcontext(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test INVALID_SPAN wrapped with nullcontext works correctly."""
    from contextlib import nullcontext

    from opentelemetry.trace import INVALID_SPAN

    from uipath.core.telemetry.observation import ObservationSpan

    # Create ObservationSpan with INVALID_SPAN (as client.py does for disabled telemetry)
    span = ObservationSpan(nullcontext(INVALID_SPAN))

    # Should work inside context manager
    with span:
        span.set_attribute("key", "value")
        span.update_input({"data": "test"})
        span.update_output({"result": "success"})

    # No spans should be recorded (INVALID_SPAN doesn't record)
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 0
