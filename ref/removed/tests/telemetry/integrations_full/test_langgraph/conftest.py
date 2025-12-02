"""Shared fixtures for LangGraph integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

if TYPE_CHECKING:
    from uipath.core.telemetry.client import TelemetryClient


@pytest.fixture
def exporter(in_memory_exporter: InMemorySpanExporter) -> InMemorySpanExporter:
    """Provide InMemorySpanExporter for tests.

    Args:
        in_memory_exporter: Global exporter from parent conftest

    Returns:
        InMemorySpanExporter instance
    """
    return in_memory_exporter


@pytest.fixture
def telemetry_client() -> TelemetryClient:
    """Create TelemetryClient instance for tests.

    Returns:
        TelemetryClient configured with test settings
    """
    from opentelemetry import trace

    from uipath.core.telemetry.client import TelemetryClient
    from uipath.core.telemetry.config import TelemetryConfig

    # Get the global TracerProvider set by setup_global_tracer_provider
    provider = trace.get_tracer_provider()

    # Create TelemetryClient with disabled mode
    config = TelemetryConfig(
        service_name="test-langgraph",
        endpoint=None,
        enable_console_export=False,
    )
    client = TelemetryClient(config)

    # Use the global test provider (already configured with InMemorySpanExporter)
    client._provider = provider  # type: ignore[attr-defined]
    client._tracer = provider.get_tracer("uipath.core.telemetry")

    return client
