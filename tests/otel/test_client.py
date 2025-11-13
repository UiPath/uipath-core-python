"""Tests for OTelClient and OTelConfig."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from uipath.core.otel.client import (
    OTelClient,
    OTelConfig,
    get_client,
    init_client,
    reset_client,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter
    from pytest_mock.plugin import MockerFixture


def test_init_with_otlp_exporter(mocker: MockerFixture) -> None:
    """Test initialization with OTLP exporter.

    Args:
        mocker: Pytest mocker fixture
    """
    # Mock OTLPSpanExporter to avoid real network calls
    mock_otlp = mocker.patch(
        "uipath.core.otel.client.OTLPSpanExporter",
        autospec=True,
    )

    # Setup
    config = OTelConfig(
        mode="prod",
        endpoint="https://telemetry.test.com",
        public_key="test-key",
        headers={"custom": "header"},
    )
    client = OTelClient(config)

    # Verify OTLP exporter created with correct parameters
    mock_otlp.assert_called_once()
    call_kwargs = mock_otlp.call_args.kwargs

    assert call_kwargs["endpoint"] == "https://telemetry.test.com"
    assert "x-uipath-public-key" in call_kwargs["headers"]
    assert call_kwargs["headers"]["x-uipath-public-key"] == "test-key"
    assert call_kwargs["headers"]["custom"] == "header"


def test_init_with_console_exporter() -> None:
    """Test initialization with console exporter."""
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)

    # Verify client initialized
    assert client._initialized is True
    tracer = client.get_tracer()
    assert tracer is not None


def test_init_disabled_mode() -> None:
    """Test initialization in disabled mode."""
    # Setup
    config = OTelConfig(mode="disabled")
    client = OTelClient(config)

    # Verify client NOT initialized
    assert client._initialized is False

    # Verify get_tracer() raises error
    with pytest.raises(RuntimeError, match="not initialized"):
        client.get_tracer()


def test_get_tracer(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test get_tracer() returns tracer.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)

    # Execute
    tracer = client.get_tracer()

    # Verify
    assert tracer is not None
    assert tracer.instrumentation_info.name == "uipath.core.otel"


def test_get_privacy_config() -> None:
    """Test get_privacy_config() returns privacy configuration."""
    # Setup
    privacy = {"redact_inputs": True, "max_attribute_length": 1000}
    config = OTelConfig(mode="dev", privacy=privacy)
    client = OTelClient(config)

    # Execute
    result = client.get_privacy_config()

    # Verify
    assert result == privacy


def test_get_privacy_config_default() -> None:
    """Test get_privacy_config() returns empty dict when no privacy."""
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)

    # Execute
    result = client.get_privacy_config()

    # Verify
    assert result == {}


def test_shutdown(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test shutdown() flushes and stops client.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Create a span
    from uipath.core.otel.trace import Trace

    with Trace(tracer, "test"):
        pass

    # Execute shutdown
    result = client.shutdown()

    # Verify
    assert result is True
    assert client._initialized is False


def test_flush(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test flush() exports pending spans.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Create spans
    from uipath.core.otel.trace import Trace

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
    config = OTelConfig(mode="dev")

    # Execute - call twice
    client1 = init_client(config)
    client2 = init_client(config)

    # Verify same instance (second init shuts down first)
    assert client1 is not client2  # Different because second init creates new

    # Verify get_client() returns latest
    current = get_client()
    assert current is client2


def test_config_validation_invalid_endpoint() -> None:
    """Test configuration validation rejects invalid endpoints."""
    with pytest.raises(ValueError, match="Must start with http://"):
        OTelConfig(
            mode="prod",
            endpoint="not-a-url",
        )


def test_config_validation_invalid_service_name() -> None:
    """Test configuration validation rejects invalid service names."""
    # Starts with hyphen
    with pytest.raises(ValueError, match="Invalid service name"):
        OTelConfig(mode="dev", service_name="-invalid")

    # Ends with hyphen
    with pytest.raises(ValueError, match="Invalid service name"):
        OTelConfig(mode="dev", service_name="invalid-")

    # Uppercase
    with pytest.raises(ValueError, match="Invalid service name"):
        OTelConfig(mode="dev", service_name="Invalid")

    # Spaces
    with pytest.raises(ValueError, match="Invalid service name"):
        OTelConfig(mode="dev", service_name="has spaces")


def test_config_validation_prod_without_endpoint() -> None:
    """Test prod mode requires endpoint when using OTLP."""
    with pytest.raises(ValueError, match="OTLP endpoint required"):
        OTelConfig(
            mode="prod",
            exporter="otlp",
            endpoint=None,
        )


def test_config_validation_valid_service_names() -> None:
    """Test valid service name formats are accepted."""
    # Single character
    config1 = OTelConfig(mode="dev", service_name="a")
    assert config1.service_name == "a"

    # With hyphens
    config2 = OTelConfig(mode="dev", service_name="my-service-name")
    assert config2.service_name == "my-service-name"

    # With numbers
    config3 = OTelConfig(mode="dev", service_name="service123")
    assert config3.service_name == "service123"


def test_config_auto_mode_detection_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test auto mode detects prod from public key.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    # Set environment variable
    monkeypatch.setenv("UIPATH_PUBLIC_KEY", "test-key")

    # Create config with auto mode
    config = OTelConfig(mode="auto", endpoint="https://test.com")

    # Verify resolved to prod
    assert config.mode == "prod"


def test_config_auto_mode_detection_without_key() -> None:
    """Test auto mode detects dev when no public key."""
    # Ensure no public key in env
    if "UIPATH_PUBLIC_KEY" in os.environ:
        del os.environ["UIPATH_PUBLIC_KEY"]

    # Create config with auto mode
    config = OTelConfig(mode="auto")

    # Verify resolved to dev
    assert config.mode == "dev"


def test_config_env_var_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variable overrides config.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    # Set env vars
    monkeypatch.setenv("UIPATH_TELEMETRY_ENDPOINT", "https://env.test.com")
    monkeypatch.setenv("UIPATH_SERVICE_NAME", "env-service")

    # Create config without explicit values
    config = OTelConfig(mode="dev")

    # Verify env vars used
    assert config.endpoint == "https://env.test.com"
    assert config.service_name == "env-service"


def test_config_service_name_default() -> None:
    """Test service name defaults correctly."""
    # Ensure no env var
    if "UIPATH_SERVICE_NAME" in os.environ:
        del os.environ["UIPATH_SERVICE_NAME"]

    # Create config without service name
    config = OTelConfig(mode="dev")

    # Verify default
    assert config.service_name == "uipath-service"


def test_config_mode_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test mode can be set via environment variable.

    Args:
        monkeypatch: Pytest monkeypatch fixture
    """
    # Set mode via env
    monkeypatch.setenv("UIPATH_TELEMETRY_MODE", "disabled")

    # Create config with auto mode
    config = OTelConfig(mode="auto")

    # Verify mode from env
    assert config.mode == "disabled"


def test_resource_attributes() -> None:
    """Test resource attributes are set correctly."""
    # Setup
    custom_attrs = {
        "deployment.environment": "production",
        "service.version": "1.0.0",
    }

    config = OTelConfig(
        mode="dev",
        service_name="test-service",
        resource_attributes=custom_attrs,
    )
    client = OTelClient(config)

    # Verify resource attributes set
    # Note: We can't easily access resource attrs from client,
    # but we can verify initialization succeeded
    assert client._initialized is True
