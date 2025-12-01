"""OpenTelemetry client for UiPath telemetry.

This module provides the core client implementation for OpenTelemetry-based
telemetry with UiPath-specific resource attributes and lifecycle management.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
)

from .config import TelemetryConfig

try:
    from . import __version__  # type: ignore[attr-defined]
except ImportError:
    __version__ = "1.0.0"

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import Tracer

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "uipath-service"
DEFAULT_SDK_NAME = "uipath-telemetry"
DEFAULT_SDK_LANGUAGE = "python"
DEFAULT_INSTRUMENTING_MODULE_NAME = "uipath.core.telemetry"

_client: TelemetryClient | None = None
_client_lock = threading.Lock()


class TelemetryClient:
    """OpenTelemetry client with UiPath-specific configuration.

    Handles TracerProvider setup, resource attributes, span processors,
    and lifecycle management (shutdown/flush).
    """

    def __init__(self, config: TelemetryConfig) -> None:
        """Initialize Telemetry client with configuration.

        Args:
            config: Client configuration
        """
        self._config = config
        self._provider: TracerProvider | None = None
        self._tracer: Tracer | None = None
        self._is_shutdown = False

        # Initialize if any exporter is configured (either OTLP or console)
        if config.endpoint or config.enable_console_export:
            self._initialize()

    def _initialize(self) -> None:
        """Initialize OpenTelemetry TracerProvider and processors."""
        existing_provider = trace.get_tracer_provider()

        if isinstance(existing_provider, TracerProvider):
            logger.info(
                "Reusing existing global TracerProvider (e.g., from test fixture)"
            )
            self._provider = existing_provider
        else:
            resource_attrs = {
                "service.name": self._config.service_name or DEFAULT_SERVICE_NAME,
                "telemetry.sdk.name": DEFAULT_SDK_NAME,
                "telemetry.sdk.language": DEFAULT_SDK_LANGUAGE,
            }

            if self._config.resource_attributes:
                resource_attrs.update(self._config.resource_attributes)

            resource = Resource.create(resource_attrs)

            self._provider = TracerProvider(resource=resource)

            exporter = self._create_exporter()
            processor = BatchSpanProcessor(exporter)
            self._provider.add_span_processor(processor)

            trace.set_tracer_provider(self._provider)
            logger.info(
                "Created new TracerProvider: endpoint=%s, console_export=%s",
                self._config.endpoint,
                self._config.enable_console_export,
            )

        self._tracer = self._provider.get_tracer(  # type: ignore[assignment]
            instrumenting_module_name=DEFAULT_INSTRUMENTING_MODULE_NAME,
            instrumenting_library_version=__version__,
        )
        self._is_shutdown = False

    def _create_exporter(self) -> SpanExporter:
        """Create span exporter based on configuration.

        Returns:
            Configured span exporter
        """
        if self._config.endpoint is None or self._config.enable_console_export:
            logger.info("Using ConsoleSpanExporter for development")
            return ConsoleSpanExporter()

        endpoint = self._config.endpoint
        headers: dict[str, str] = {}

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
        if self._tracer is None:
            raise RuntimeError("Telemetry client not initialized")
        return self._tracer

    def get_tracer_provider(self) -> TracerProvider:
        """Get OpenTelemetry TracerProvider.

        Returns:
            TracerProvider instance

        Raises:
            RuntimeError: If client not initialized
        """
        if self._provider is None:
            raise RuntimeError("Telemetry client not initialized")
        return self._provider

    def shutdown(self, timeout_seconds: float = 10.0) -> bool:
        """Shutdown telemetry and flush remaining spans.

        Args:
            timeout_seconds: Timeout for shutdown

        Returns:
            True if shutdown successful

        Raises:
            Exception: If shutdown fails
        """
        if self._provider is None:
            self._is_shutdown = True
            return True

        logger.info("Shutting down Telemetry client (timeout=%.1fs)", timeout_seconds)
        self._provider.shutdown()
        self._provider = None
        self._tracer = None
        self._is_shutdown = True
        return True

    def flush(self, timeout_seconds: float = 5.0) -> bool:
        """Flush pending spans to exporter.

        Args:
            timeout_seconds: Timeout for flush

        Returns:
            True if flush successful

        Raises:
            Exception: If flush fails
        """
        if self._provider is None:
            return True

        logger.debug("Flushing Telemetry spans (timeout=%.1fs)", timeout_seconds)
        result = self._provider.force_flush(timeout_millis=int(timeout_seconds * 1000))
        return result

    def is_shutdown(self) -> bool:
        """Check if client has been shutdown.

        Returns:
            True if shutdown() has been called
        """
        return self._is_shutdown


def get_client() -> TelemetryClient:
    """Get singleton Telemetry client instance.

    Returns:
        Global Telemetry client instance

    Raises:
        RuntimeError: If client not initialized via init()
    """
    global _client
    if _client is None:
        raise RuntimeError(
            "Telemetry client not initialized. Call telemetry.init() first."
        )
    return _client


def init_client(config: TelemetryConfig) -> TelemetryClient:
    """Initialize global Telemetry client with configuration (thread-safe).

    This function is idempotent - calling it multiple times returns the same
    client instance. If client was shutdown, creates new instance.
    Use reset_client() in tests to clear state.

    Args:
        config: Client configuration

    Returns:
        Initialized Telemetry client
    """
    global _client

    if _client is not None and not _client.is_shutdown():
        return _client

    with _client_lock:
        if _client is not None and not _client.is_shutdown():
            return _client

        _client = TelemetryClient(config)
        return _client


def reset_client() -> None:
    """Reset global client instance (for testing only, thread-safe).

    Warning:
        This function should only be used in test teardown. Do not call
        in production code as it will lose all telemetry state.
    """
    global _client
    with _client_lock:
        if _client is not None:
            _client.shutdown()
        _client = None
