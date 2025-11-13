"""OpenTelemetry client for UiPath telemetry.

This module provides the core client implementation for OpenTelemetry-based
telemetry with UiPath-specific resource attributes and lifecycle management.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)

# Import version from parent module
try:
    from . import __version__
except ImportError:
    # Fallback if __version__ not available
    __version__ = "1.0.0"

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Tracer

logger = logging.getLogger(__name__)

# Singleton client instance with thread-safe access
_client: OTelClient | None = None
_client_lock = threading.Lock()


@dataclass
class OTelConfig:
    """Configuration for OpenTelemetry client.

    Args:
        public_key: UiPath public key for authentication
        mode: Operating mode - auto (env detection), dev, prod, or disabled
        service_name: Service name for resource attributes
        resource_attributes: Additional resource attributes
        privacy: Privacy configuration dict
        exporter: Exporter type - otlp or console
        endpoint: OTLP endpoint URL (defaults to env var or UiPath cloud)
        headers: Headers for OTLP exporter
        auto_instrument: Whether to enable auto-instrumentation
    """

    public_key: str | None = None
    mode: Literal["auto", "dev", "prod", "disabled"] = "auto"
    service_name: str | None = None
    resource_attributes: dict[str, str] | None = None
    privacy: dict[str, Any] | None = None
    exporter: Literal["otlp", "console"] = "otlp"
    endpoint: str | None = None
    headers: dict[str, str] | None = None
    auto_instrument: bool = False

    def __post_init__(self) -> None:
        """Resolve configuration from environment variables and validate."""
        # Resolve mode
        if self.mode == "auto":
            self.mode = self._detect_mode()

        # Resolve endpoint
        if self.endpoint is None and self.mode != "dev":
            self.endpoint = os.getenv(
                "UIPATH_TELEMETRY_ENDPOINT",
                "https://telemetry.uipath.com",
            )

        # Resolve service name
        if self.service_name is None:
            self.service_name = os.getenv(
                "UIPATH_SERVICE_NAME",
                "uipath-service",
            )

        # HIGH FIX: Validate configuration
        self._validate()

    def _detect_mode(self) -> Literal["dev", "prod", "disabled"]:
        """Auto-detect mode from environment.

        Returns:
            Detected mode based on environment variables
        """
        mode_env = os.getenv("UIPATH_TELEMETRY_MODE", "").lower()
        if mode_env in ("dev", "prod", "disabled"):
            return mode_env  # type: ignore[return-value]

        # Detect from environment
        if os.getenv("UIPATH_PUBLIC_KEY"):
            return "prod"
        return "dev"

    def _validate(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate endpoint URL if provided
        if self.endpoint:
            if not self.endpoint.startswith(("http://", "https://")):
                raise ValueError(
                    f"Invalid endpoint URL: {self.endpoint}. "
                    "Must start with http:// or https://"
                )

        # Validate prod mode has endpoint
        if self.mode == "prod" and self.exporter == "otlp" and not self.endpoint:
            raise ValueError(
                "OTLP endpoint required in prod mode. "
                "Set endpoint parameter or UIPATH_TELEMETRY_ENDPOINT env var."
            )

        # Validate service name format (lowercase alphanumeric with hyphens)
        if self.service_name:
            import re

            if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", self.service_name):
                raise ValueError(
                    f"Invalid service name: '{self.service_name}'. "
                    "Must be lowercase alphanumeric with hyphens, "
                    "cannot start or end with hyphen."
                )

        # Validate privacy configuration
        if self.privacy:
            if "max_attribute_length" in self.privacy:
                max_len = self.privacy["max_attribute_length"]
                if not isinstance(max_len, int) or max_len < 0:
                    raise ValueError(
                        "privacy.max_attribute_length must be a non-negative integer"
                    )

            if "redact_inputs" in self.privacy:
                if not isinstance(self.privacy["redact_inputs"], bool):
                    raise ValueError("privacy.redact_inputs must be a boolean")

            if "redact_outputs" in self.privacy:
                if not isinstance(self.privacy["redact_outputs"], bool):
                    raise ValueError("privacy.redact_outputs must be a boolean")


class OTelClient:
    """OpenTelemetry client with UiPath-specific configuration.

    Handles TracerProvider setup, resource attributes, span processors,
    and lifecycle management (shutdown/flush).
    """

    def __init__(self, config: OTelConfig) -> None:
        """Initialize OTel client with configuration.

        Args:
            config: Client configuration
        """
        self._config = config
        self._provider: TracerProvider | None = None
        self._tracer: Tracer | None = None
        self._initialized = False
        # CRITICAL FIX: Store privacy configuration for Observation layer
        self._privacy_config = config.privacy or {}

        # Initialize if not disabled
        if config.mode != "disabled":
            self._initialize()

    def _initialize(self) -> None:
        """Initialize OpenTelemetry TracerProvider and processors."""
        # Create resource with UiPath attributes
        resource_attrs = {
            "service.name": self._config.service_name or "uipath-service",
            "telemetry.sdk.name": "uipath-otel",
            "telemetry.sdk.language": "python",
        }

        # Add custom resource attributes
        if self._config.resource_attributes:
            resource_attrs.update(self._config.resource_attributes)

        resource = Resource.create(resource_attrs)

        # Create TracerProvider
        self._provider = TracerProvider(resource=resource)

        # Add span processor (batch processor for both prod and dev)
        exporter = self._create_exporter()
        processor = BatchSpanProcessor(exporter)
        self._provider.add_span_processor(processor)

        # Set global tracer provider
        trace.set_tracer_provider(self._provider)

        # Get tracer
        self._tracer = self._provider.get_tracer(
            instrumenting_module_name="uipath.core.otel",
            instrumenting_library_version=__version__,
        )

        self._initialized = True
        logger.info(
            "OTel client initialized: mode=%s, exporter=%s",
            self._config.mode,
            self._config.exporter,
        )

    def _create_exporter(self) -> SpanExporter:
        """Create span exporter based on configuration.

        Returns:
            Configured span exporter
        """
        if self._config.mode == "dev" or self._config.exporter == "console":
            logger.info("Using ConsoleSpanExporter for development")
            return ConsoleSpanExporter()

        # Production: OTLP exporter
        endpoint = self._config.endpoint
        headers = self._config.headers or {}

        # Add public key to headers if provided
        if self._config.public_key:
            headers["x-uipath-public-key"] = self._config.public_key

        logger.info("Using OTLPSpanExporter: endpoint=%s", endpoint)
        return OTLPSpanExporter(
            endpoint=endpoint,
            headers=headers,
        )

    def get_tracer(self) -> Tracer:
        """Get OpenTelemetry tracer.

        Returns:
            Tracer instance

        Raises:
            RuntimeError: If client not initialized
        """
        if not self._initialized or self._tracer is None:
            raise RuntimeError("OTel client not initialized")
        return self._tracer

    def get_privacy_config(self) -> dict[str, Any]:
        """Get privacy configuration for attribute sanitization.

        Returns:
            Privacy configuration dictionary
        """
        return self._privacy_config

    def shutdown(self, timeout_seconds: float = 10.0) -> bool:
        """Shutdown telemetry and flush remaining spans.

        Args:
            timeout_seconds: Timeout for shutdown

        Returns:
            True if shutdown successful
        """
        if self._provider is None:
            return True

        try:
            logger.info("Shutting down OTel client (timeout=%.1fs)", timeout_seconds)
            result = self._provider.shutdown()
            self._initialized = False
            return result
        except Exception as e:
            logger.error("Error during OTel client shutdown: %s", e, exc_info=True)
            return False

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        """Flush pending spans to exporter.

        Args:
            timeout_seconds: Timeout for flush

        Returns:
            True if flush successful
        """
        if self._provider is None:
            return True

        try:
            logger.debug("Flushing OTel spans (timeout=%.1fs)", timeout_seconds)
            result = self._provider.force_flush(timeout_millis=int(timeout_seconds * 1000))
            return result
        except Exception as e:
            logger.error("Error during OTel flush: %s", e, exc_info=True)
            return False


def get_client() -> OTelClient:
    """Get singleton OTel client instance.

    Returns:
        Global OTel client instance

    Raises:
        RuntimeError: If client not initialized via init()
    """
    global _client
    if _client is None:
        raise RuntimeError(
            "OTel client not initialized. Call otel.init() first."
        )
    return _client


def init_client(config: OTelConfig) -> OTelClient:
    """Initialize global OTel client with configuration (thread-safe).

    Args:
        config: Client configuration

    Returns:
        Initialized OTel client
    """
    global _client

    # CRITICAL FIX: Thread-safe singleton initialization with double-checked locking
    with _client_lock:
        # Shutdown existing client if any
        if _client is not None:
            logger.warning("Reinitializing OTel client (shutting down existing)")
            _client.shutdown()

        # Create new client
        _client = OTelClient(config)
        return _client


def reset_client() -> None:
    """Reset global client instance (for testing, thread-safe)."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.shutdown()
        _client = None
