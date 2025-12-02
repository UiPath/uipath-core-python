"""Shared fixtures for integrations_lite tests."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )


@pytest.fixture(autouse=True)
def init_telemetry_client(in_memory_exporter: "InMemorySpanExporter") -> None:
    """Initialize telemetry client for all integrations_lite tests.

    This fixture ensures the telemetry client is initialized before each test
    runs, which is required by the @traced decorator.

    Args:
        in_memory_exporter: In-memory span exporter from parent conftest
    """
    from uipath.core.telemetry.client import init_client
    from uipath.core.telemetry.config import TelemetryConfig

    config = TelemetryConfig(enable_console_export=True)
    init_client(config)
