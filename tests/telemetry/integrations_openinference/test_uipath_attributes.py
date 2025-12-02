"""Tests for UiPath session attributes on spans."""

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.integrations_openinference import (
    clear_session_context,
    instrument_langchain,
    set_session_context,
    uninstrument_langchain,
)


@pytest.fixture
def instrumented_provider():
    """Create instrumented tracer provider."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrument_langchain(tracer_provider=provider)

    yield provider, exporter

    uninstrument_langchain()
    provider.shutdown()
    clear_session_context()


def test_session_id_added_to_spans(instrumented_provider):
    """Test that session_id is added to spans."""
    provider, exporter = instrumented_provider

    set_session_context(session_id="test-session-123")

    # Create a test span
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes is not None
    assert span.attributes.get("session.id") == "test-session-123"


def test_thread_id_added_to_spans(instrumented_provider):
    """Test that thread_id is added to spans."""
    provider, exporter = instrumented_provider

    set_session_context(session_id="session-123", thread_id="thread-456")

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    assert span.attributes is not None
    assert span.attributes.get("session.id") == "session-123"
    assert span.attributes.get("thread.id") == "thread-456"


def test_no_session_context_no_attributes(instrumented_provider):
    """Test that no session attributes added when not set."""
    provider, exporter = instrumented_provider

    # Don't set session context

    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1

    span = spans[0]
    # Should not have session or thread attributes
    if span.attributes:
        assert "session.id" not in span.attributes
        assert "thread.id" not in span.attributes
