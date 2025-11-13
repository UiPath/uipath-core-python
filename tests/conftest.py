"""Shared pytest fixtures for all tests."""

from typing import TYPE_CHECKING, Generator

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

if TYPE_CHECKING:
    from uipath.core.telemetry import TelemetryClient, TelemetryConfig


class SpanCapture:
    """Helper to capture and analyze spans."""

    def __init__(self):
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        trace.set_tracer_provider(self.provider)

    def get_spans(self):
        """Get all captured spans."""
        return self.exporter.get_finished_spans()

    def clear(self):
        """Clear captured spans."""
        self.exporter.clear()

    def print_hierarchy(self):
        """Print the span hierarchy for debugging."""
        spans = self.get_spans()
        print("\n=== Span Hierarchy ===")
        for span in spans:
            parent_id = span.parent.span_id if span.parent else "ROOT"
            print(f"  {span.name}")
            print(f"    Span ID: {span.context.span_id}")
            print(f"    Parent ID: {parent_id}")
            print(f"    Trace ID: {span.context.trace_id}")
        print("======================\n")


@pytest.fixture(scope="session")
def span_capture() -> SpanCapture:
    """Fixture to capture spans - created once for entire test session."""
    return SpanCapture()


@pytest.fixture(autouse=True)
def clear_spans_between_tests(span_capture: SpanCapture):
    """Clear captured spans before each test."""
    span_capture.clear()
    yield


# ============================================================================
# Telemetry-specific fixtures
# ============================================================================


@pytest.fixture
def telemetry_config() -> "TelemetryConfig":
    """Standard test configuration for telemetry tests.

    Returns a TelemetryConfig with test-specific settings using InMemorySpanExporter.
    """
    from uipath.core.telemetry import ResourceAttr, TelemetryConfig

    return TelemetryConfig(
        resource_attributes=(
            (ResourceAttr.ORG_ID, "test-org-123"),
            (ResourceAttr.TENANT_ID, "test-tenant-456"),
            (ResourceAttr.USER_ID, "test-user-789"),
        )
    )


@pytest.fixture
def telemetry_client(
    telemetry_config: "TelemetryConfig",
) -> Generator["TelemetryClient", None, None]:
    """Fixture providing TelemetryClient with InMemorySpanExporter.

    Automatically resets the client after each test.
    """
    from opentelemetry import trace as trace_api
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from uipath.core.telemetry import get_telemetry_client, reset_telemetry_client

    # Shutdown any existing provider to ensure clean state
    existing_provider = trace_api.get_tracer_provider()
    if hasattr(existing_provider, "shutdown"):
        existing_provider.shutdown()

    # Reset the client singleton
    reset_telemetry_client()

    # Force reset of global provider to allow new provider with resource attributes
    trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    # Reset the once flag to allow setting a new provider
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    # Create client (will create new TracerProvider with resource attributes)
    client = get_telemetry_client(telemetry_config)

    # Add InMemorySpanExporter alongside the ConsoleSpanExporter
    # Both will receive spans, but we'll only check InMemorySpanExporter in tests
    memory_exporter = InMemorySpanExporter()
    memory_processor = SimpleSpanProcessor(memory_exporter)
    client._tracer_provider.add_span_processor(memory_processor)

    yield client

    # Clean up after test - shutdown and reset
    client._tracer_provider.shutdown()
    reset_telemetry_client()
    trace_api._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace_api._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]


@pytest.fixture
def memory_exporter(
    telemetry_client: "TelemetryClient",
) -> InMemorySpanExporter:
    """Extract InMemorySpanExporter from TelemetryClient for span verification.

    This fixture was identified as critical (RED FLAG) in consensus analysis.
    Implementation accesses internal TelemetryClient structure.
    """
    # Access the tracer provider's span processors
    tracer_provider = telemetry_client._tracer_provider

    # The provider has an _active_span_processor which may be:
    # - A MultiSpanProcessor containing multiple processors
    # - A single processor directly

    active_processor = tracer_provider._active_span_processor

    # Check if it's a MultiSpanProcessor (has _span_processors attribute)
    if hasattr(active_processor, "_span_processors"):
        processors = active_processor._span_processors
    else:
        # Single processor
        processors = [active_processor]

    # Find the exporter
    for processor in processors:
        # SimpleSpanProcessor has span_exporter attribute
        # BatchSpanProcessor has span_exporter attribute
        if hasattr(processor, "span_exporter"):
            exporter = processor.span_exporter
            if isinstance(exporter, InMemorySpanExporter):
                return exporter

    raise RuntimeError(
        "InMemorySpanExporter not found in TelemetryClient. "
        "Found processors: " + str([type(p).__name__ for p in processors])
    )


@pytest.fixture(params=[1.0, 0.0])
def telemetry_enabled(request) -> "TelemetryConfig":
    """Parametrize tests for enabled/disabled telemetry states.

    Recommended by consensus for testing zero-overhead when disabled.
    sample_rate=1.0 means enabled, sample_rate=0.0 means disabled.
    """
    from uipath.core.telemetry import TelemetryConfig

    return TelemetryConfig(
        org_id="test-org",
        tenant_id="test-tenant",
        sample_rate=request.param,
    )
