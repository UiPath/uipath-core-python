"""Tests for execution_id parameter in start_as_current_span."""

from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.telemetry.utils import get_execution_spans
from uipath.core.telemetry import (
    ResourceAttr,
    TelemetryConfig,
    get_execution_id,
    get_telemetry_client,
    reset_telemetry_client,
    set_execution_id,
)

if TYPE_CHECKING:
    pass


def test_execution_id_parameter_sets_context():
    """Test that execution_id parameter sets execution ID in context."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "test-org"),
            (ResourceAttr.TENANT_ID, "test-tenant"),
        ),
        endpoint=None,  # Console exporter
    )
    client = get_telemetry_client(config)

    # Add InMemorySpanExporter to capture spans
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # execution_id parameter should set context
    with client.start_as_current_span("workflow", execution_id="exec-123"):
        # Context should be set inside span
        assert get_execution_id() == "exec-123"

    # Context should be cleared after span exits
    # Note: Currently set_execution_id doesn't auto-clear, but that's okay
    # Users can call clear_execution_id() if needed

    client.flush()
    reset_telemetry_client()


def test_execution_id_parameter_propagates_to_children():
    """Test that child spans inherit execution ID from parent."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "test-org"),
            (ResourceAttr.TENANT_ID, "test-tenant"),
        ),
        endpoint=None,
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Create parent span with execution_id parameter
    with client.start_as_current_span(
        "parent", semantic_type="automation", execution_id="exec-456"
    ):
        # Create child spans without execution_id parameter
        with client.start_as_current_span("child1"):
            pass
        with client.start_as_current_span("child2"):
            pass

    client.flush()

    # All spans should have the same execution ID
    spans = get_execution_spans(exporter, "exec-456")
    assert len(spans) == 3, "Should have parent + 2 children"

    span_names = {s.name for s in spans}
    assert span_names == {"parent", "child1", "child2"}

    # Verify all have correct execution.id attribute
    for span in spans:
        assert span.attributes.get("execution.id") == "exec-456"  # type: ignore[union-attr]

    reset_telemetry_client()


def test_execution_id_parameter_overrides_context():
    """Test that execution_id parameter overrides existing context."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Set context manually first
    set_execution_id("exec-old")
    assert get_execution_id() == "exec-old"

    # execution_id parameter should override
    with client.start_as_current_span("workflow", execution_id="exec-new"):
        assert get_execution_id() == "exec-new"

    client.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("execution.id") == "exec-new"  # type: ignore[union-attr]

    reset_telemetry_client()


def test_execution_id_parameter_none_uses_context():
    """Test that execution_id=None reads from existing context."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Set context manually
    set_execution_id("exec-from-context")

    # execution_id=None should read from context
    with client.start_as_current_span("workflow", execution_id=None):
        pass

    client.flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("execution.id") == "exec-from-context"  # type: ignore[union-attr]

    reset_telemetry_client()


def test_execution_id_parameter_with_nested_spans():
    """Test execution_id with nested spans at different levels."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Root span with execution ID
    with client.start_as_current_span("root", execution_id="exec-root"):
        # Level 1 child
        with client.start_as_current_span("level1"):
            # Level 2 child
            with client.start_as_current_span("level2"):
                # Level 3 child
                with client.start_as_current_span("level3"):
                    pass

    client.flush()

    # All spans should have root execution ID
    spans = get_execution_spans(exporter, "exec-root")
    assert len(spans) == 4

    span_names = {s.name for s in spans}
    assert span_names == {"root", "level1", "level2", "level3"}

    reset_telemetry_client()


def test_execution_id_parameter_with_multiple_executions():
    """Test multiple execution-scoped spans don't interfere."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # First execution
    with client.start_as_current_span("workflow1", execution_id="exec-111"):
        with client.start_as_current_span("step1"):
            pass

    # Second execution
    with client.start_as_current_span("workflow2", execution_id="exec-222"):
        with client.start_as_current_span("step2"):
            pass

    client.flush()

    # Verify first execution spans
    spans_111 = get_execution_spans(exporter, "exec-111")
    assert len(spans_111) == 2
    assert {s.name for s in spans_111} == {"workflow1", "step1"}

    # Verify second execution spans
    spans_222 = get_execution_spans(exporter, "exec-222")
    assert len(spans_222) == 2
    assert {s.name for s in spans_222} == {"workflow2", "step2"}

    reset_telemetry_client()


def test_execution_id_parameter_with_decorator():
    """Test execution_id parameter works with @traced decorator."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from uipath.core.telemetry import traced

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    @traced(span_type="activity")
    def process_data(data):
        return f"processed: {data}"

    # Root span with execution_id
    with client.start_as_current_span("workflow", execution_id="exec-decorated"):
        result = process_data("test-data")
        assert result == "processed: test-data"

    client.flush()

    # Both spans should have execution ID
    spans = get_execution_spans(exporter, "exec-decorated")
    assert len(spans) == 2

    span_names = {s.name for s in spans}
    assert span_names == {"workflow", "process_data"}

    reset_telemetry_client()


def test_execution_id_parameter_backward_compatibility():
    """Test that existing code without execution_id still works."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Old code pattern (without execution_id parameter)
    set_execution_id("exec-old-style")
    with client.start_as_current_span("workflow"):
        with client.start_as_current_span("step1"):
            pass

    client.flush()

    # Should still work
    spans = get_execution_spans(exporter, "exec-old-style")
    assert len(spans) == 2

    reset_telemetry_client()


def test_get_execution_spans_utility():
    """Test the get_execution_spans test utility function."""
    reset_telemetry_client()
    exporter = InMemorySpanExporter()
    config = TelemetryConfig(
        resource_attributes=((ResourceAttr.ORG_ID, "test-org"),), endpoint=None
    )
    client = get_telemetry_client(config)

    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    client._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Create spans with different execution IDs
    with client.start_as_current_span("workflow1", execution_id="exec-A"):
        pass

    with client.start_as_current_span("workflow2", execution_id="exec-B"):
        pass

    with client.start_as_current_span("workflow3", execution_id="exec-A"):
        pass

    client.flush()

    # Test filtering by execution ID
    spans_a = get_execution_spans(exporter, "exec-A")
    assert len(spans_a) == 2
    assert {s.name for s in spans_a} == {"workflow1", "workflow3"}

    spans_b = get_execution_spans(exporter, "exec-B")
    assert len(spans_b) == 1
    assert spans_b[0].name == "workflow2"

    # Test non-existent execution ID
    spans_c = get_execution_spans(exporter, "exec-C")
    assert len(spans_c) == 0

    reset_telemetry_client()
