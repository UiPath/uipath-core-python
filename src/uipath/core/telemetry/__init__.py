"""UiPath Core Telemetry - OpenTelemetry-based observability.

Provides lightweight, industry-standard telemetry for UiPath automations
with zero-overhead when disabled.

Quick Start:
    >>> from uipath.core.telemetry import (
    ...     get_telemetry_client,
    ...     TelemetryConfig,
    ...     traced,
    ... )
    >>>
    >>> # Configure once at startup
    >>> config = TelemetryConfig(
    ...     org_id="org-123",
    ...     tenant_id="tenant-456",
    ...     endpoint="https://telemetry.example.com"
    ... )
    >>> client = get_telemetry_client(config)
    >>>
    >>> # Execution-scoped tracing (recommended for workflows)
    >>> with client.start_as_current_span(
    ...     "workflow",
    ...     semantic_type="automation",
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

from ._semantic_conventions import (
    SPAN_TYPE_ACTIVITY,
    SPAN_TYPE_AGENT,
    SPAN_TYPE_AUTOMATION,
    SPAN_TYPE_CHAIN,
    SPAN_TYPE_GENERATION,
    SPAN_TYPE_RETRIEVER,
    # Semantic span types
    SPAN_TYPE_SPAN,
    SPAN_TYPE_TOOL,
    SPAN_TYPE_WORKFLOW,
    UIPATH_EXECUTION_ID,
    UIPATH_FOLDER_KEY,
    # Span attributes
    UIPATH_JOB_ID,
    # Resource attributes
    UIPATH_ORG_ID,
    UIPATH_PROCESS_KEY,
    UIPATH_TENANT_ID,
    UIPATH_USER_ID,
)
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
    # Semantic conventions - Resource attributes
    "UIPATH_ORG_ID",
    "UIPATH_TENANT_ID",
    "UIPATH_USER_ID",
    # Semantic conventions - Span attributes
    "UIPATH_JOB_ID",
    "UIPATH_PROCESS_KEY",
    "UIPATH_FOLDER_KEY",
    "UIPATH_EXECUTION_ID",
    # Semantic conventions - Span types
    "SPAN_TYPE_SPAN",
    "SPAN_TYPE_GENERATION",
    "SPAN_TYPE_TOOL",
    "SPAN_TYPE_AUTOMATION",
    "SPAN_TYPE_ACTIVITY",
    "SPAN_TYPE_WORKFLOW",
    "SPAN_TYPE_AGENT",
    "SPAN_TYPE_CHAIN",
    "SPAN_TYPE_RETRIEVER",
]

__version__ = "1.0.0"
