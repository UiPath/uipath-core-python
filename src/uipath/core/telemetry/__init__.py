"""UiPath Core Telemetry - OpenTelemetry-based observability.

Provides lightweight, industry-standard telemetry for UiPath automations
with zero-overhead when disabled.

Quick Start:
    >>> from uipath.core.telemetry import (
    ...     get_telemetry_client,
    ...     TelemetryConfig,
    ...     ResourceAttr,
    ...     SpanType,
    ...     traced,
    ... )
    >>>
    >>> # Configure once at startup with resource attributes
    >>> config = TelemetryConfig(
    ...     resource_attributes=(
    ...         (ResourceAttr.ORG_ID, "org-123"),
    ...         (ResourceAttr.TENANT_ID, "tenant-456"),
    ...     ),
    ...     endpoint="https://telemetry.example.com"
    ... )
    >>> client = get_telemetry_client(config)
    >>>
    >>> # Execution-scoped tracing (recommended for workflows)
    >>> with client.start_as_current_span(
    ...     "workflow",
    ...     semantic_type=SpanType.AUTOMATION,
    ...     execution_id="exec-12345"  # All children inherit this ID
    ... ) as span:
    ...     process_invoice()  # Automatically tagged with execution.id
    ...     generate_report()  # All spans share execution context
    >>> client.flush()
    >>>
    >>> # Use decorator for automatic instrumentation
    >>> @traced(span_type="automation")
    >>> def process_invoice():
    ...     return {"status": "processed"}
    >>>
    >>> # Or manually set execution context (advanced usage)
    >>> from uipath.core.telemetry import set_execution_id
    >>> set_execution_id("exec-12345")
    >>> with client.start_as_current_span("custom_operation") as span:
    ...     span.set_attribute("key", "value")
"""

from .attributes import ResourceAttr, SpanAttr, SpanType
from .client import (
    TelemetryClient,
    get_telemetry_client,
    reset_telemetry_client,
)
from .config import TelemetryConfig
from .context import (
    clear_execution_id,
    get_execution_id,
    set_execution_id,
)
from .decorator import traced
from .observation import ObservationSpan

__all__ = [
    # Core classes
    "TelemetryConfig",
    "TelemetryClient",
    "ObservationSpan",
    # Public functions
    "get_telemetry_client",
    "reset_telemetry_client",
    "traced",
    # Context management
    "set_execution_id",
    "get_execution_id",
    "clear_execution_id",
    # Semantic conventions (enums)
    "ResourceAttr",
    "SpanAttr",
    "SpanType",
]

__version__ = "1.0.0"
