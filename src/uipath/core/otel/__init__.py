"""UiPath OpenTelemetry integration.

This module provides a simple, batteries-included OpenTelemetry integration
for UiPath Python applications with AI/LLM observability.

Example:
    from uipath.core import otel

    # Initialize telemetry (development mode)
    otel.init(enable_console_export=True)

    # Production mode
    otel.init(
        endpoint="https://telemetry.uipath.com",
        service_name="invoice-processor",
    )

    # Decorate functions
    @otel.traced(kind="generation")
    def extract_invoice(prompt: str) -> dict:
        response = openai.chat.completions.create(...)
        return response

    # Use trace context
    with otel.trace("workflow", execution_id="exec-123") as trace:
        result = extract_invoice("Extract from PDF...")
"""

from __future__ import annotations

from typing import Any

from .client import TelemetryClient, get_client, init_client
from .config import TelemetryConfig
from .decorator import traced
from .observation import ObservationSpan
from .trace import Trace

__all__ = [
    # Client
    "TelemetryClient",
    "init",
    "shutdown",
    "flush",
    # Tracing
    "trace",
    "traced",
    # Observation
    "ObservationSpan",
    # Config
    "TelemetryConfig",
]


# ============================================================================
# Initialization API
# ============================================================================


def init(
    endpoint: str | None = None,
    service_name: str | None = None,
    enable_console_export: bool = False,
    resource_attributes: dict[str, str] | None = None,
) -> TelemetryClient:
    """Initialize OpenTelemetry client with UiPath configuration.

    Privacy is controlled per-span using hide_input/hide_output flags.
    Sampling rate is hardcoded to 1.0 (100% sampling).

    Args:
        endpoint: OTLP endpoint URL (None = console exporter for dev)
        service_name: Service name for resource attributes
        enable_console_export: Enable console exporter (for debugging)
        resource_attributes: Additional resource attributes

    Returns:
        Initialized TelemetryClient

    Example:
        import otel

        # Development mode (console output)
        otel.init(enable_console_export=True)

        # Production mode
        otel.init(
            endpoint="https://telemetry.uipath.com",
            service_name="invoice-processor",
        )

        # Privacy controlled at decorator level
        @otel.traced(kind="tool", hide_input=True)
        def authenticate(api_key: str):
            ...
    """
    config = TelemetryConfig(
        endpoint=endpoint,
        service_name=service_name or "uipath-service",
        enable_console_export=enable_console_export,
        resource_attributes=resource_attributes,
    )
    return init_client(config)


def shutdown(timeout_seconds: float = 10.0) -> bool:
    """Shutdown telemetry and flush remaining spans.

    Args:
        timeout_seconds: Timeout for shutdown

    Returns:
        True if shutdown successful

    Raises:
        RuntimeError: If client not initialized (call init() first)
    """
    client = get_client()
    return client.shutdown(timeout_seconds)


def flush(timeout_seconds: float = 5.0) -> bool:
    """Flush pending spans to exporter.

    Args:
        timeout_seconds: Timeout for flush

    Returns:
        True if flush successful

    Raises:
        RuntimeError: If client not initialized (call init() first)
    """
    client = get_client()
    return client.flush(timeout_seconds)


# ============================================================================
# Trace Context API
# ============================================================================


def trace(
    name: str,
    execution_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Create trace context manager for workflow execution.

    Args:
        name: Trace name
        execution_id: Execution ID for correlation
        metadata: Additional metadata attributes

    Returns:
        Trace context manager

    Example:
        with otel.trace("invoice-workflow", execution_id="exec-123") as trace:
            result = extract_invoice(...)
            url = trace.get_url()
            print(f"View trace: {url}")
    """
    client = get_client()
    tracer = client.get_tracer()
    return Trace(
        tracer=tracer,
        name=name,
        execution_id=execution_id,
        metadata=metadata,
    )
