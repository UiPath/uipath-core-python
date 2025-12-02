"""Tests for basic instrumentation functionality."""

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.integrations_openinference import (
    instrument_langchain,
    uninstrument_langchain,
)


@pytest.fixture
def tracer_provider():
    """Create a fresh TracerProvider for each test."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def test_instrument_basic(tracer_provider):
    """Test basic instrumentation."""
    provider, exporter = tracer_provider

    instrument_langchain(tracer_provider=provider)

    # Clean up
    uninstrument_langchain()


def test_instrument_twice_raises(tracer_provider):
    """Test that instrumenting twice raises RuntimeError."""
    provider, _ = tracer_provider

    instrument_langchain(tracer_provider=provider)

    with pytest.raises(RuntimeError, match="already instrumented"):
        instrument_langchain(tracer_provider=provider)

    # Clean up
    uninstrument_langchain()


def test_uninstrument_not_instrumented_raises():
    """Test that uninstrumenting when not instrumented raises RuntimeError."""
    with pytest.raises(RuntimeError, match="not instrumented"):
        uninstrument_langchain()


def test_uninstrument_basic(tracer_provider):
    """Test basic uninstrumentation."""
    provider, _ = tracer_provider

    instrument_langchain(tracer_provider=provider)

    uninstrument_langchain()


def test_reinstrument_after_uninstrument(tracer_provider):
    """Test that we can re-instrument after uninstrumenting."""
    provider, _ = tracer_provider

    # Instrument
    instrument_langchain(tracer_provider=provider)

    # Uninstrument
    uninstrument_langchain()

    # Re-instrument
    instrument_langchain(tracer_provider=provider)

    # Clean up
    uninstrument_langchain()


def test_instrument_with_default_provider():
    """Test instrumentation with default tracer provider."""
    # Set default provider
    provider = TracerProvider()
    trace.set_tracer_provider(provider)

    try:
        instrument_langchain()  # Should use default provider

        uninstrument_langchain()
    except RuntimeError:
        # Clean up if still instrumented
        try:
            uninstrument_langchain()
        except RuntimeError:
            pass
