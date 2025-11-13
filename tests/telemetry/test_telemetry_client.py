"""Tests for TelemetryClient singleton, configuration, and lifecycle."""

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath.core.telemetry import (
    ResourceAttr,
    TelemetryClient,
    TelemetryConfig,
    get_telemetry_client,
    reset_telemetry_client,
    set_execution_id,
)

if TYPE_CHECKING:
    pass


def test_telemetry_client_singleton(telemetry_config: TelemetryConfig):
    """Test that get_telemetry_client returns the same instance."""
    client1 = get_telemetry_client(telemetry_config)
    client2 = get_telemetry_client(telemetry_config)

    assert client1 is client2, "get_telemetry_client should return singleton"

    reset_telemetry_client()


def test_telemetry_client_with_config(telemetry_config: TelemetryConfig):
    """Test TelemetryClient properly applies configuration."""
    client = get_telemetry_client(telemetry_config)

    assert client._config == telemetry_config

    reset_telemetry_client()


def test_telemetry_client_tracer_provider(
    telemetry_client: TelemetryClient,
):
    """Test TracerProvider initialization."""
    assert telemetry_client._tracer_provider is not None
    assert telemetry_client._tracer is not None

    # Verify tracer metadata (when available - ProxyTracer doesn't have instrumentation_info)
    if hasattr(telemetry_client._tracer, "instrumentation_info"):
        assert telemetry_client._tracer.instrumentation_info.name == "uipath-core"


def test_telemetry_client_reset():
    """Test reset_telemetry_client() cleanup."""
    config = TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "org-1"),
            (ResourceAttr.TENANT_ID, "tenant-1"),
        )
    )
    client1 = get_telemetry_client(config)

    reset_telemetry_client()

    # New client after reset
    config2 = TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "org-2"),
            (ResourceAttr.TENANT_ID, "tenant-2"),
        )
    )
    client2 = get_telemetry_client(config2)

    assert client1 is not client2

    reset_telemetry_client()


def test_telemetry_client_start_as_current_span(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test context manager span creation."""
    with telemetry_client.start_as_current_span("test_span") as span:
        span.set_attribute("test_key", "test_value")

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.name == "test_span"
    assert span.attributes["test_key"] == "test_value"


def test_telemetry_client_register_parent_resolver(
    telemetry_client: TelemetryClient,
):
    """Test custom parent resolver registration."""
    mock_span = None
    call_count = 0

    def mock_resolver():
        nonlocal call_count
        call_count += 1
        return mock_span

    telemetry_client.register_parent_resolver(mock_resolver)

    # Create span (should call resolver)
    with telemetry_client.start_as_current_span("test_span"):
        pass

    assert call_count > 0, "Parent resolver should be called"


def test_telemetry_client_parent_resolution_basic(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test basic parent span resolution."""
    # Create parent span
    with telemetry_client.start_as_current_span("parent"):
        # Create child span (should automatically link to parent)
        with telemetry_client.start_as_current_span("child"):
            pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2

    child_span = spans[0]
    parent_span_data = spans[1]

    # Verify parent-child relationship
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span_data.context.span_id


def test_telemetry_client_parent_resolution_fallback(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test fallback to OTel context when resolver returns None."""

    def resolver_returns_none():
        return None

    telemetry_client.register_parent_resolver(resolver_returns_none)

    # Create parent span in OTel context
    with telemetry_client.start_as_current_span("parent"):
        # Should fall back to OTel context parent
        with telemetry_client.start_as_current_span("child"):
            pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 2

    child_span = spans[0]
    parent_span = spans[1]

    # Verify fallback worked
    assert child_span.parent is not None
    assert child_span.parent.span_id == parent_span.context.span_id


def test_telemetry_client_resource_attributes(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test resource attributes (org_id, tenant_id, user_id) are set."""
    with telemetry_client.start_as_current_span("test_span"):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    resource_attrs = span.resource.attributes

    assert resource_attrs["uipath.org_id"] == "test-org-123"
    assert resource_attrs["uipath.tenant_id"] == "test-tenant-456"
    assert resource_attrs["uipath.user_id"] == "test-user-789"


def test_telemetry_client_semantic_type(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test semantic_type attribute is set correctly."""
    with telemetry_client.start_as_current_span(
        "test_span", semantic_type="automation"
    ):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes["span.type"] == "automation"


def test_telemetry_client_execution_id_propagation(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test execution.id attribute is automatically added from context."""
    set_execution_id("test-execution-123")

    with telemetry_client.start_as_current_span("test_span"):
        pass

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes["execution.id"] == "test-execution-123"


def test_telemetry_client_unregister_parent_resolver(
    telemetry_client: TelemetryClient,
):
    """Test unregister_parent_resolver removes resolver."""

    def mock_resolver():
        return None

    telemetry_client.register_parent_resolver(mock_resolver)
    assert telemetry_client._parent_resolver is not None

    telemetry_client.unregister_parent_resolver()
    assert telemetry_client._parent_resolver is None


# ============================================================================
# Skipped tests for missing functionality
# ============================================================================


@pytest.mark.skip(reason="start_execution_span() not yet implemented (~20 LOC)")
def test_telemetry_client_start_execution_span():
    """Test start_execution_span() convenience helper with manual flush.

    Note: Per consensus, this helper does NOT auto-flush.
    User must manually call client.flush_spans().
    """
    config = TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "test-org"),
            (ResourceAttr.TENANT_ID, "test-tenant"),
        )
    )
    client = get_telemetry_client(config)

    # Start execution span
    with client.start_execution_span("execution-123", "workflow-run"):
        # Do work
        pass

    # Manual flush required
    client.flush_spans()

    reset_telemetry_client()


@pytest.mark.skip(reason="get_execution_spans() test utility not yet implemented")
def test_telemetry_client_get_execution_spans_test_utility():
    """Test get_execution_spans() retrieves spans by execution_id.

    Note: This is a test/development utility, not production API.
    Should only be available when using InMemorySpanExporter.
    """
    config = TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "test-org"),
            (ResourceAttr.TENANT_ID, "test-tenant"),
        )
    )
    client = get_telemetry_client(config)

    set_execution_id("exec-123")

    with client.start_as_current_span("span1"):
        pass

    with client.start_as_current_span("span2"):
        pass

    # Retrieve spans for execution
    spans = client.get_execution_spans("exec-123")
    assert len(spans) == 2

    reset_telemetry_client()
