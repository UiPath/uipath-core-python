"""UiPath OpenTelemetry-based telemetry library.

This module provides a developer-friendly API for instrumenting Python applications
with OpenTelemetry tracing, featuring smart provider response parsing and
both decorator and context manager patterns.

Example:
    from uipath.core import otel
    import openai

    # Initialize once
    otel.init(public_key="pk_...", mode="auto")

    # Pattern 1: Decorator with auto-update
    @otel.generation(model="gpt-4")
    def extract_invoice(prompt: str) -> dict:
        result = openai.chat.completions.create(...)
        return result  # Auto-updated

    # Pattern 2: Context manager for complex workflows
    with otel.trace("workflow", execution_id="exec-123") as trace:
        with trace.generation(name="llm", model="gpt-4") as gen:
            result = openai.chat.completions.create(...)
            gen.update(result)  # Smart parsing

    # Pattern 3: Hybrid (both)
    with otel.trace("workflow") as trace:
        @otel.generation(model="gpt-4")
        def nested_call():
            return openai.chat.completions.create(...)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from .client import OTelClient, OTelConfig, get_client, init_client
from .decorator import agent, generation, tool, traced
from .trace import Trace

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Library version (sync with pyproject.toml)
__version__ = "1.0.0"

# Public API exports
__all__ = [
    # Initialization
    "init",
    # Context managers
    "trace",
    # Decorators
    "generation",
    "tool",
    "agent",
    "traced",
    # Lifecycle
    "shutdown",
    "flush",
]


def init(
    public_key: str | None = None,
    mode: Literal["auto", "dev", "prod", "disabled"] = "auto",
    service_name: str | None = None,
    resource_attributes: dict[str, str] | None = None,
    privacy: dict[str, Any] | None = None,
    exporter: Literal["otlp", "console"] = "otlp",
    endpoint: str | None = None,
    headers: dict[str, str] | None = None,
    auto_instrument: bool = False,
) -> OTelClient:
    """Initialize OpenTelemetry telemetry client.

    Args:
        public_key: UiPath public key for authentication
        mode: Operating mode - auto (env detection), dev, prod, or disabled
        service_name: Service name for resource attributes
        resource_attributes: Additional resource attributes
        privacy: Privacy configuration dict
        exporter: Exporter type - otlp or console
        endpoint: OTLP endpoint URL
        headers: Additional headers for OTLP exporter
        auto_instrument: Whether to enable auto-instrumentation

    Returns:
        Initialized OTel client

    Example:
        otel.init(
            public_key="pk_...",
            mode="auto",
            service_name="invoice-processor",
            resource_attributes={"uipath.org_id": "org-123"},
        )
    """
    config = OTelConfig(
        public_key=public_key,
        mode=mode,
        service_name=service_name,
        resource_attributes=resource_attributes,
        privacy=privacy,
        exporter=exporter,
        endpoint=endpoint,
        headers=headers,
        auto_instrument=auto_instrument,
    )

    client = init_client(config)
    logger.info(
        "OTel telemetry initialized: mode=%s, service=%s",
        config.mode,
        config.service_name,
    )
    return client


def trace(
    name: str,
    execution_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trace:
    """Create a trace context manager for instrumenting a workflow.

    Args:
        name: Trace name
        execution_id: Execution ID for correlation
        user_id: User ID for correlation
        metadata: Additional metadata attributes

    Returns:
        Trace context manager

    Example:
        with otel.trace("invoice-workflow", execution_id="exec-123") as trace:
            with trace.generation(name="llm", model="gpt-4") as gen:
                result = openai.chat.completions.create(...)
                gen.update(result)

            with trace.tool(name="validate") as tool:
                validation = validate_invoice(result)
                tool.set_attribute("valid", validation.passed)

            print(f"View trace: {trace.get_url()}")
    """
    client = get_client()
    tracer = client.get_tracer()
    return Trace(
        tracer=tracer,
        name=name,
        execution_id=execution_id,
        user_id=user_id,
        metadata=metadata,
    )


def shutdown(timeout_seconds: float = 10.0) -> bool:
    """Shutdown telemetry and flush remaining spans.

    Args:
        timeout_seconds: Timeout for shutdown

    Returns:
        True if shutdown successful

    Example:
        try:
            # Run application
            process_workflows()
        finally:
            otel.shutdown(timeout_seconds=10)
    """
    try:
        client = get_client()
        return client.shutdown(timeout_seconds=timeout_seconds)
    except RuntimeError:
        # Client not initialized
        logger.debug("Client not initialized, nothing to shutdown")
        return True


def flush(timeout_seconds: float = 5.0) -> bool:
    """Flush pending spans to exporter.

    Args:
        timeout_seconds: Timeout for flush

    Returns:
        True if flush successful

    Example:
        # After processing batch of operations
        otel.flush(timeout_seconds=5)
    """
    try:
        client = get_client()
        return client.flush(timeout_seconds=timeout_seconds)
    except RuntimeError:
        # Client not initialized
        logger.debug("Client not initialized, nothing to flush")
        return True
