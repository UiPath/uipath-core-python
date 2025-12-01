"""Telemetry configuration module."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration for UiPath OpenTelemetry client.

    All parameters support standard OpenTelemetry environment variable overrides:
    - OTEL_EXPORTER_OTLP_ENDPOINT
    - OTEL_SERVICE_NAME
    - OTEL_TRACES_EXPORTER (set to "console" for console export)

    Args:
        endpoint: OTLP endpoint URL (None = console exporter for dev)
        service_name: Service identifier for resource attributes
        enable_console_export: Enable console exporter (for debugging)
        resource_attributes: Additional resource attributes
        max_generator_items: Maximum items to buffer from generators (default: 10000)
    """

    endpoint: str | None = None
    service_name: str = "uipath-service"
    enable_console_export: bool = False
    resource_attributes: dict[str, str] | None = None
    max_generator_items: int = 10000

    def __post_init__(self) -> None:
        """Resolve configuration from environment variables."""
        if env_endpoint := os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            object.__setattr__(self, "endpoint", env_endpoint)

        if env_service := os.getenv("OTEL_SERVICE_NAME"):
            object.__setattr__(self, "service_name", env_service)

        if env_exporter := os.getenv("OTEL_TRACES_EXPORTER"):
            object.__setattr__(
                self, "enable_console_export", "console" in env_exporter.lower()
            )

        self._validate()

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        if self.endpoint and not self.endpoint.startswith(("http://", "https://")):
            raise ValueError(
                f"endpoint must start with http:// or https://, got {self.endpoint}"
            )
