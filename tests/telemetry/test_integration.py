"""Integration tests for end-to-end telemetry workflows."""

from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath.core.telemetry import (
    TelemetryClient,
    TelemetryConfig,
    get_telemetry_client,
    reset_telemetry_client,
    set_execution_id,
)

if TYPE_CHECKING:
    pass


def test_full_workflow_with_nested_spans(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test complete workflow with nested spans and attributes."""
    set_execution_id("execution-123")

    with telemetry_client.start_as_current_span(
        "workflow", semantic_type="automation"
    ) as workflow_span:
        workflow_span.set_attribute("workflow_name", "invoice_processing")
        workflow_span.update_input({"invoice_id": "INV-001"})

        # Step 1
        with telemetry_client.start_as_current_span("validate") as validate_span:
            validate_span.set_attribute("validation_result", "passed")

        # Step 2
        with telemetry_client.start_as_current_span("process") as process_span:
            process_span.set_attribute("status", "completed")
            process_span.set_attribute("amount", 1500.00)

        workflow_span.update_output({"status": "success", "processed_count": 1})

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 3

    # Verify hierarchy
    validate_span_data = spans[0]
    process_span_data = spans[1]
    workflow_span_data = spans[2]

    # All spans should have the same trace ID
    assert validate_span_data.context.trace_id == workflow_span_data.context.trace_id
    assert process_span_data.context.trace_id == workflow_span_data.context.trace_id

    # Validate parent-child relationships
    assert validate_span_data.parent.span_id == workflow_span_data.context.span_id
    assert process_span_data.parent.span_id == workflow_span_data.context.span_id

    # Verify execution.id propagation
    assert validate_span_data.attributes["execution.id"] == "execution-123"
    assert process_span_data.attributes["execution.id"] == "execution-123"
    assert workflow_span_data.attributes["execution.id"] == "execution-123"

    # Verify semantic types
    assert workflow_span_data.attributes["span.type"] == "automation"
    assert validate_span_data.attributes["span.type"] == "span"


def test_multiple_independent_traces(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test multiple independent trace hierarchies."""
    # First trace
    set_execution_id("exec-1")
    with telemetry_client.start_as_current_span("trace1"):
        pass

    # Second trace
    set_execution_id("exec-2")
    with telemetry_client.start_as_current_span("trace2"):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2

    span1, span2 = spans

    # Should have different trace IDs
    assert span1.context.trace_id != span2.context.trace_id

    # Should have different execution IDs
    assert span1.attributes["execution.id"] == "exec-1"
    assert span2.attributes["execution.id"] == "exec-2"


def test_error_handling_in_nested_spans(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test error propagation and recording in nested spans."""
    from opentelemetry.trace import StatusCode

    try:
        with telemetry_client.start_as_current_span("parent") as parent_span:
            parent_span.set_attribute("status", "running")

            with telemetry_client.start_as_current_span("child") as child_span:
                child_span.set_attribute("task", "risky_operation")
                raise ValueError("Something went wrong")
    except ValueError:
        pass  # Expected

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2

    child_span_data = spans[0]
    parent_span_data = spans[1]

    # Child span should be marked as error
    assert child_span_data.status.status_code == StatusCode.ERROR
    assert "Something went wrong" in child_span_data.status.description

    # Parent span should also be marked as error (automatic propagation)
    assert parent_span_data.status.status_code == StatusCode.ERROR


def test_privacy_controls_integration(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test hide_input and hide_output throughout workflow."""
    with telemetry_client.start_as_current_span(
        "secure_operation", hide_input=True, hide_output=True
    ) as span:
        span.update_input({"password": "secret123"})
        span.update_output({"api_key": "key-xyz"})
        span.set_attribute("public_data", "safe_to_log")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]

    # Input and output should NOT be present
    assert "input" not in span_data.attributes
    assert "output" not in span_data.attributes

    # Public attributes should still work
    assert span_data.attributes["public_data"] == "safe_to_log"


def test_config_environment_variable_override():
    """Test that environment variables override config defaults."""
    import os

    # Set environment variables (correct names per config.py)
    os.environ["UIPATH_ORG_ID"] = "env-org-id"
    os.environ["UIPATH_TENANT_ID"] = "env-tenant-id"
    os.environ["UIPATH_TELEMETRY_SAMPLE_RATE"] = "0.5"

    try:
        # Reset to pick up env vars
        reset_telemetry_client()

        config = TelemetryConfig()
        assert config.org_id == "env-org-id"
        assert config.tenant_id == "env-tenant-id"
        assert config.sample_rate == 0.5

    finally:
        # Clean up
        del os.environ["UIPATH_ORG_ID"]
        del os.environ["UIPATH_TENANT_ID"]
        del os.environ["UIPATH_TELEMETRY_SAMPLE_RATE"]
        reset_telemetry_client()


def test_resource_attributes_integration(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test that resource attributes are correctly set."""
    with telemetry_client.start_as_current_span("test"):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    resource = span_data.resource.attributes

    # Verify UiPath resource attributes
    assert resource["uipath.org_id"] == "test-org-123"
    assert resource["uipath.tenant_id"] == "test-tenant-456"
    assert resource["uipath.user_id"] == "test-user-789"

    # Verify service attributes
    assert resource["service.name"] == "uipath-core"
    assert "service.version" in resource


def test_span_attribute_types_integration(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test various attribute value types in real spans."""
    with telemetry_client.start_as_current_span("test") as span:
        # Native types
        span.set_attribute("string_attr", "value")
        span.set_attribute("int_attr", 42)
        span.set_attribute("float_attr", 3.14)
        span.set_attribute("bool_attr", True)

        # Collections
        span.set_attribute("list_attr", ["a", "b", "c"])
        span.set_attribute("dict_attr", {"key": "value"})

    spans = memory_exporter.get_finished_spans()
    span_data = spans[0]

    # Verify native types preserved
    assert span_data.attributes["string_attr"] == "value"
    assert span_data.attributes["int_attr"] == 42
    assert span_data.attributes["float_attr"] == 3.14
    assert span_data.attributes["bool_attr"] is True

    # List converted to tuple by OTel
    assert list(span_data.attributes["list_attr"]) == ["a", "b", "c"]

    # Dict serialized to JSON
    import json

    dict_value = json.loads(span_data.attributes["dict_attr"])
    assert dict_value["key"] == "value"


def test_disabled_telemetry_zero_overhead(
    memory_exporter: InMemorySpanExporter,
):
    """Test that disabled telemetry has minimal overhead."""
    config = TelemetryConfig(sample_rate=0.0)  # Disabled
    reset_telemetry_client()
    client = get_telemetry_client(config)

    # Should return no-op spans
    with client.start_as_current_span("test") as span:
        span.set_attribute("key", "value")
        span.update_input({"data": "input"})

    # No spans should be recorded
    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 0

    reset_telemetry_client()


def test_manual_span_lifecycle(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test manual span start/end without context manager."""
    span = telemetry_client.start_as_current_span("manual_span")
    span.__enter__()

    try:
        span.set_attribute("step", 1)
        span.update_input({"request": "data"})

        # Simulate work
        result = {"status": "success"}

        span.update_output(result)
        span.set_attribute("step", 2)

    finally:
        span.end()

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]
    assert span_data.attributes["step"] == 2

    import json

    output = json.loads(span_data.attributes["output"])
    assert output["status"] == "success"


def test_deeply_nested_spans(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test deeply nested span hierarchy."""
    with telemetry_client.start_as_current_span("level1") as s1:
        s1.set_attribute("level", 1)
        with telemetry_client.start_as_current_span("level2") as s2:
            s2.set_attribute("level", 2)
            with telemetry_client.start_as_current_span("level3") as s3:
                s3.set_attribute("level", 3)
                with telemetry_client.start_as_current_span("level4") as s4:
                    s4.set_attribute("level", 4)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 4

    # Verify hierarchy (spans are returned in reverse order)
    level4, level3, level2, level1 = spans

    # All same trace
    assert level4.context.trace_id == level1.context.trace_id

    # Verify parent chain
    assert level4.parent.span_id == level3.context.span_id
    assert level3.parent.span_id == level2.context.span_id
    assert level2.parent.span_id == level1.context.span_id
    assert level1.parent is None  # Root span


def test_parent_resolver_integration(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test custom parent resolver integration."""
    from opentelemetry import trace

    # Create an external span
    external_tracer = trace.get_tracer("external")
    with external_tracer.start_as_current_span("external_parent") as external_span:
        # Register resolver that returns the external span
        def custom_resolver():
            return external_span

        telemetry_client.register_parent_resolver(custom_resolver)

        # Create a span - should use external parent
        with telemetry_client.start_as_current_span("child"):
            pass

        telemetry_client.unregister_parent_resolver()

    spans = memory_exporter.get_finished_spans()
    # Should have 2 spans: external_parent + child
    assert len(spans) == 2

    child_span = spans[0]
    external_parent_span = spans[1]

    # Child should have external_parent as parent
    assert child_span.parent.span_id == external_parent_span.context.span_id


def test_execution_id_context_propagation(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test execution_id propagates through context without explicit passing."""
    from uipath.core.telemetry import get_execution_id

    # Set execution ID once
    set_execution_id("global-exec-id")

    # Verify it's retrievable
    assert get_execution_id() == "global-exec-id"

    # Create multiple spans - all should have the execution ID
    with telemetry_client.start_as_current_span("span1"):
        pass

    with telemetry_client.start_as_current_span("span2"):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2

    # Both spans should have the execution ID
    assert spans[0].attributes["execution.id"] == "global-exec-id"
    assert spans[1].attributes["execution.id"] == "global-exec-id"
