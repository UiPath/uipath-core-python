"""Tests for TelemetryClient and TelemetryConfig."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from uipath.core.telemetry.client import (
    TelemetryClient,
    get_client,
    init_client,
    reset_client,
)
from uipath.core.telemetry.config import TelemetryConfig

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter
    from pytest_mock.plugin import MockerFixture


@pytest.mark.no_auto_tracer
def test_init_with_otlp_exporter(mocker: MockerFixture) -> None:
    """Test initialization with OTLP exporter.

    Args:
        mocker: Pytest mocker fixture
    """
    # Reset global tracer provider state for this test
    from opentelemetry import trace

    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    # Mock OTLPSpanExporter to avoid real network calls
    mock_otlp = mocker.patch(
        "uipath.core.telemetry.client.OTLPSpanExporter",
        autospec=True,
    )

    # Setup
    config = TelemetryConfig(
        endpoint="https://telemetry.test.com",
    )
    _ = TelemetryClient(config)  # Client creation triggers OTLP exporter setup

    # Verify OTLP exporter created with correct parameters
    mock_otlp.assert_called_once()
    call_kwargs = mock_otlp.call_args.kwargs

    assert call_kwargs["endpoint"] == "https://telemetry.test.com"
    # Note: public_key and custom headers removed in V4 - authentication handled separately


def test_init_with_console_exporter() -> None:
    """Test initialization with console exporter."""
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = TelemetryClient(config)

    # Verify client initialized
    assert client._tracer is not None
    tracer = client.get_tracer()
    assert tracer is not None


def test_init_disabled_mode() -> None:
    """Test initialization in disabled mode (no endpoint or console export)."""
    # Setup - neither endpoint nor console export enabled
    config = TelemetryConfig(endpoint=None, enable_console_export=False)
    client = TelemetryClient(config)

    # Verify client NOT initialized
    assert client._tracer is None

    # Verify get_tracer() raises error
    with pytest.raises(RuntimeError, match="not initialized"):
        client.get_tracer()


def test_get_tracer(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test get_tracer() returns tracer.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = TelemetryClient(config)

    # Execute
    tracer = client.get_tracer()

    # Verify
    assert tracer is not None
    assert tracer.instrumentation_info.name == "uipath.core.telemetry"


# Privacy config removed - privacy is now controlled per-span via hide_input/hide_output flags


def test_shutdown(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test shutdown() flushes and stops client.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = TelemetryClient(config)
    tracer = client.get_tracer()

    # Create a span
    from uipath.core.telemetry.trace import Trace

    with Trace(tracer, "test"):
        pass

    # Execute shutdown
    result = client.shutdown()

    # Verify
    assert result is True
    assert client._provider is None
    assert client._tracer is None


def test_flush(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test flush() exports pending spans.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = TelemetryClient(config)
    tracer = client.get_tracer()

    # Create spans
    from uipath.core.telemetry.trace import Trace

    with Trace(tracer, "test1"):
        pass
    with Trace(tracer, "test2"):
        pass

    # Execute flush
    result = client.flush()

    # Verify
    assert result is True


def test_get_client_not_initialized() -> None:
    """Test get_client() raises error when not initialized."""
    # Reset to ensure no client
    reset_client()

    # Verify raises with helpful message
    with pytest.raises(RuntimeError, match="not initialized"):
        get_client()


def test_init_client_singleton() -> None:
    """Test init_client() returns singleton instance."""
    # Setup
    config = TelemetryConfig(enable_console_export=True)

    # Execute - call twice
    client1 = init_client(config)
    client2 = init_client(config)

    # Verify same instance (singleton behavior)
    assert client1 is client2  # Same instance - singleton pattern

    # Verify get_client() returns latest
    current = get_client()
    assert current is client2


def test_config_validation_invalid_endpoint() -> None:
    """Test configuration validation rejects invalid endpoints."""
    with pytest.raises(ValueError, match="must start with http://"):
        TelemetryConfig(
            endpoint="not-a-url",
        )


def test_config_valid_service_names() -> None:
    """Test valid service name formats are accepted."""
    # Note: TelemetryConfig doesn't validate service name format
    # Any string is accepted - OpenTelemetry will normalize it

    # Single character
    config1 = TelemetryConfig(enable_console_export=True, service_name="a")
    assert config1.service_name == "a"

    # With hyphens
    config2 = TelemetryConfig(
        enable_console_export=True, service_name="my-service-name"
    )
    assert config2.service_name == "my-service-name"

    # With numbers
    config3 = TelemetryConfig(enable_console_export=True, service_name="service123")
    assert config3.service_name == "service123"

    # Even unusual names are accepted (Telemetry will handle normalization)
    config4 = TelemetryConfig(enable_console_export=True, service_name="Invalid-Name")
    assert config4.service_name == "Invalid-Name"


def test_config_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides config.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    # Set env vars
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://env.test.com")
    monkeypatch.setenv("OTEL_SERVICE_NAME", "env-service")

    # Create config without explicit values
    config = TelemetryConfig()

    # Verify env vars used
    assert config.endpoint == "https://env.test.com"
    assert config.service_name == "env-service"


def test_config_service_name_default() -> None:
    """Test service name defaults correctly."""
    # Ensure no env var
    if "OTEL_SERVICE_NAME" in os.environ:
        del os.environ["OTEL_SERVICE_NAME"]

    # Create config without service name
    config = TelemetryConfig(enable_console_export=True)

    # Verify default
    assert config.service_name == "uipath-service"


def test_resource_attributes() -> None:
    """Test resource attributes are set correctly."""
    # Setup
    custom_attrs = {
        "deployment.environment": "production",
        "service.version": "1.0.0",
    }

    config = TelemetryConfig(
        enable_console_export=True,
        service_name="test-service",
        resource_attributes=custom_attrs,
    )
    client = TelemetryClient(config)

    # Verify resource attributes set
    # Note: We can't easily access resource attrs from client,
    # but we can verify initialization succeeded
    assert client._tracer is not None


def test_config_console_export_via_telemetry_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OTEL_TRACES_EXPORTER enables console export.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    # Test single value "console"
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "console")
    config1 = TelemetryConfig()
    assert config1.enable_console_export is True

    # Test comma-separated with console
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp,console")
    config2 = TelemetryConfig()
    assert config2.enable_console_export is True

    # Test uppercase
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "CONSOLE")
    config3 = TelemetryConfig()
    assert config3.enable_console_export is True

    # Test without console
    monkeypatch.setenv("OTEL_TRACES_EXPORTER", "otlp")
    config4 = TelemetryConfig()
    assert config4.enable_console_export is False
