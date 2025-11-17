"""Shared fixtures for otel tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from opentelemetry.sdk.trace import TracerProvider


@pytest.fixture
def in_memory_exporter() -> InMemorySpanExporter:
    """Create in-memory exporter for capturing spans.

    Returns:
        InMemorySpanExporter instance for testing
    """
    return InMemorySpanExporter()


@pytest.fixture
def tracer_provider_with_exporter(
    in_memory_exporter: InMemorySpanExporter,
) -> TracerProvider:
    """Create TracerProvider with in-memory exporter.

    Args:
        in_memory_exporter: In-memory exporter fixture

    Returns:
        TracerProvider configured with in-memory exporter
    """
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    return provider


@pytest.fixture
def sample_openai_response() -> dict[str, Any]:
    """Mock OpenAI ChatCompletion response.

    Returns:
        Dict mimicking OpenAI response structure
    """
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "total_tokens": 19,
        },
    }


@pytest.fixture
def sample_anthropic_response() -> dict[str, Any]:
    """Mock Anthropic Message response.

    Returns:
        Dict mimicking Anthropic response structure
    """
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "text",
                "text": "Hello! How can I help you today?",
            }
        ],
        "model": "claude-3-opus-20240229",
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10,
            "output_tokens": 9,
        },
    }


@pytest.fixture(autouse=True)
def reset_client() -> None:
    """Reset global client before each test.

    This ensures test isolation by clearing the singleton client.
    """
    from uipath.core.otel.client import reset_client

    reset_client()


@pytest.fixture(autouse=True)
def setup_global_tracer_provider(
    request: FixtureRequest,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Set up global tracer provider with in-memory exporter for tests.

    This ensures all tests use the same InMemorySpanExporter for span capture.

    Tests can skip this fixture by using the 'no_auto_tracer' marker:
        @pytest.mark.no_auto_tracer
        def test_custom_setup(): ...

    Args:
        request: Pytest request fixture
        in_memory_exporter: In-memory exporter fixture
    """
    # Skip auto-setup if test is marked with no_auto_tracer
    if "no_auto_tracer" in request.keywords:
        return

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Reset global provider state to allow setting a new one
    # This is necessary because OpenTelemetry prevents overriding once set
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    # Create TracerProvider with in-memory exporter
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))

    # Set as global provider
    trace.set_tracer_provider(provider)
