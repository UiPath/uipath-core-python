"""Singleton telemetry client with OpenTelemetry integration.

Provides OpenTelemetry-based tracing with UiPath-specific metadata and
integration hooks for external tracing systems (LangGraph, LangChain, etc.).

Example:
    >>> from uipath.core.telemetry import get_telemetry_client, TelemetryConfig
    >>>
    >>> # Create client with config
    >>> config = TelemetryConfig(org_id="org-123", tenant_id="tenant-456")
    >>> client = get_telemetry_client(config)
    >>>
    >>> # Use client to create spans
    >>> with client.start_as_current_span("process", semantic_type="automation") as span:
    ...     span.set_attribute("status", "running")
    ...     result = do_work()
"""

import random
import warnings
from functools import lru_cache
from typing import Any, Callable, Optional, Union

from opentelemetry import context, trace
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

# Optional OTLP exporter (only needed for production endpoints)
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import-not-found]
        OTLPSpanExporter,
    )

    HAS_OTLP = True
except ImportError:
    HAS_OTLP = False
    OTLPSpanExporter = None

from opentelemetry.sdk.trace import (  # type: ignore[attr-defined]
    Resource,
    TracerProvider,
)
from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from .attributes import SpanAttr
from .config import TelemetryConfig
from .context import get_execution_id
from .observation import ObservationSpan


class TelemetryClient:
    """Singleton telemetry client. Access via get_telemetry_client().

    Provides OpenTelemetry-based tracing with UiPath-specific metadata and
    integration hooks for external tracing systems (LangGraph, LangChain, etc.).

    Example:
        >>> config = TelemetryConfig(org_id="org-123")
        >>> client = get_telemetry_client(config)
        >>> with client.start_as_current_span("process") as span:
        ...     span.set_attribute("status", "running")
    """

    def __init__(self, config: TelemetryConfig):
        """Initialize telemetry client.

        Args:
            config: Telemetry configuration

        Note:
            Do not call this directly. Use get_telemetry_client() instead.
        """
        self._config = config

        resource_attrs = {
            "service.name": config.service_name,
            "service.version": config.service_version or config.library_version,
            "telemetry.sdk.name": config.library_name,
            "telemetry.sdk.version": config.library_version,
        }

        if config.service_namespace:
            resource_attrs["service.namespace"] = config.service_namespace

        # Merge user-provided resource attributes (tuple of tuples)
        if config.resource_attributes:
            resource_attrs.update(dict(config.resource_attributes))

        resource = Resource.create(resource_attrs)  # type: ignore[arg-type]

        if config.endpoint:
            if not HAS_OTLP:
                raise ImportError(
                    "OTLP exporter not available. Install with: "
                    "pip install opentelemetry-exporter-otlp-proto-grpc"
                )
            # Convert tuple headers to dict for OTLP exporter
            headers_dict = dict(config.headers) if config.headers else {}
            exporter = OTLPSpanExporter(endpoint=config.endpoint, headers=headers_dict)
        else:
            # Default to Console for development (no external dependency)
            exporter = ConsoleSpanExporter()

        processor: Union[BatchSpanProcessor, SimpleSpanProcessor]
        if config.batch_export:
            processor = BatchSpanProcessor(
                exporter,
                max_queue_size=config.max_queue_size,
                export_timeout_millis=config.export_timeout_millis,
            )
        else:
            processor = SimpleSpanProcessor(exporter)

        current_provider = trace.get_tracer_provider()

        if isinstance(current_provider, SdkTracerProvider):
            self._tracer_provider = current_provider
            warnings.warn(
                "An existing OpenTelemetry TracerProvider was found. "
                "Resource attributes from TelemetryConfig will not be "
                "applied to the global provider. Spans will still be exported to your "
                "configured endpoint via the added processor.",
                UserWarning,
                stacklevel=3,
            )
        else:
            self._tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(self._tracer_provider)

        # Add our processor to the selected provider (works for both cases)
        self._tracer_provider.add_span_processor(processor)

        self._tracer = trace.get_tracer(
            config.library_name,
            config.library_version,
        )

        # Parent resolver hook (for external tracing system integration)
        # Enables LangGraph, LangChain, etc. to inject parent spans
        self._parent_resolver: Optional[Callable[[], Optional[trace.Span]]] = None

    def register_parent_resolver(
        self, resolver: Callable[[], Optional[trace.Span]]
    ) -> None:
        """Register external parent span resolver.

        This hook enables integration with external tracing systems (LangGraph,
        LangChain, etc.) that maintain their own span hierarchies. The resolver
        is called when creating new spans to determine the parent context.

        Primarily intended for use by official integration packages
        (uipath-agents-langgraph, etc.). Advanced users may provide custom
        resolvers for other tracing systems.

        Args:
            resolver: Callable returning an OTel Span or None. Called during
                     span creation to inject external parent spans.

        Example:
            >>> # In uipath-agents-langgraph package:
            >>> def langgraph_parent_resolver() -> Optional[Span]:
            ...     return get_langgraph_current_span()
            >>>
            >>> client = get_telemetry_client()
            >>> client.register_parent_resolver(langgraph_parent_resolver)
        """
        self._parent_resolver = resolver

    def unregister_parent_resolver(self) -> None:
        """Unregister external parent resolver (lifecycle control).

        Example:
            >>> client.unregister_parent_resolver()
        """
        self._parent_resolver = None

    def _resolve_parent_context(self) -> context.Context:
        """Internal: Resolve parent context (OTel or external).

        Resolution order:
        1. If external resolver registered, call it
        2. If external span returned and recording, use as parent
        3. Otherwise, use current OTel context

        Returns:
            Parent context for new span
        """
        if self._parent_resolver:
            try:
                external_span = self._parent_resolver()
                if external_span and external_span.is_recording():
                    return trace.set_span_in_context(external_span)
            except Exception as e:
                # Log warning but don't fail - fall through to OTel context
                warnings.warn(
                    f"Parent resolver failed: {e}. Using OTel context.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        return context.get_current()

    def start_as_current_span(
        self,
        name: str,
        semantic_type: str = "span",
        execution_id: Optional[str] = None,
        attributes: Optional[dict[str, Any]] = None,
        hide_input: bool = False,
        hide_output: bool = False,
        recording: bool = True,
    ) -> ObservationSpan:
        """Create span as context manager.

        Args:
            name: Span name (e.g., "process_invoice")
            semantic_type: Semantic span type (span, generation, automation, etc.)
            attributes: Additional span attributes (dynamic, high cardinality)
            hide_input: Hide input from telemetry (privacy)
            hide_output: Hide output from telemetry (privacy)
            recording: If False, create non-recording span that maintains trace
                      hierarchy without recording data (default True)
            execution_id: Optional execution ID to set in context for this span
                         and all children. If provided, automatically sets the
                         execution ID via set_execution_id(). If None, reads
                         execution ID from existing context (if set).

        Returns:
            ObservationSpan context manager

        Example:
            >>> # Basic usage
            >>> with client.start_as_current_span(
            ...     "process",
            ...     semantic_type="automation",
            ...     attributes={"uipath.job_id": "job-123"}
            ... ) as span:
            ...     result = do_work()
            ...     span.set_attribute("result_count", len(result))
            >>>
            >>> # Execution-scoped span (sets execution ID automatically)
            >>> with client.start_as_current_span(
            ...     "workflow",
            ...     semantic_type="automation",
            ...     execution_id="exec-123"
            ... ) as span:
            ...     process_invoice()  # Child spans inherit execution.id
            ...     generate_report()  # All spans get execution.id="exec-123"
            >>> client.flush()

        Note:
            The execution_id parameter is a convenience for the common pattern
            of execution-scoped tracing. It's equivalent to:

                set_execution_id("exec-123")
                with client.start_as_current_span("workflow"):
                    do_work()
        """
        if execution_id is not None:
            from .context import set_execution_id

            set_execution_id(execution_id)

        if not recording:
            # Try to reuse parent trace_id for hierarchy continuity
            current_span = trace.get_current_span()
            parent_span_ctx = current_span.get_span_context()

            if parent_span_ctx.is_valid and parent_span_ctx.trace_id != 0:
                trace_id = parent_span_ctx.trace_id
            else:
                trace_id = random.getrandbits(128)
                while trace_id == 0:
                    trace_id = random.getrandbits(128)

            span_id = random.getrandbits(64)
            while span_id == 0:
                span_id = random.getrandbits(64)

            # Create non-recording span with valid context
            # IMPORTANT: Use TraceFlags(0x01) to allow children to be recorded
            # The span itself is non-recording (no data collected), but children can be recorded
            non_recording_context = SpanContext(
                trace_id=trace_id,
                span_id=span_id,
                is_remote=False,
                trace_flags=TraceFlags(0x01),
            )
            otel_span = NonRecordingSpan(non_recording_context)
            span_cm = trace.use_span(otel_span, end_on_exit=False)

            return ObservationSpan(
                span_cm, hide_input=hide_input, hide_output=hide_output
            )

        parent_ctx = self._resolve_parent_context()

        span_attrs = attributes or {}
        span_attrs[SpanAttr.TYPE] = semantic_type

        # Add execution_id if set (automatic propagation via ContextVar)
        exec_id = get_execution_id()
        if exec_id:
            span_attrs[SpanAttr.EXECUTION_ID] = exec_id

        otel_span_cm = self._tracer.start_as_current_span(
            name,
            context=parent_ctx,
            attributes=span_attrs,
        )

        return ObservationSpan(
            otel_span_cm, hide_input=hide_input, hide_output=hide_output
        )

    def flush(self, timeout_seconds: int = 30) -> bool:
        """Flush pending spans to exporter.

        Args:
            timeout_seconds: Maximum time to wait for flush (default 30s)

        Returns:
            True if flush succeeded within timeout, False otherwise

        Example:
            >>> client.flush(timeout_seconds=10)
            True
        """
        return self._tracer_provider.force_flush(timeout_millis=timeout_seconds * 1000)

    def shutdown(self, timeout_seconds: int = 30) -> bool:
        """Shutdown telemetry client (flush and close).

        Args:
            timeout_seconds: Maximum time to wait for shutdown (default 30s)

        Returns:
            True if shutdown succeeded within timeout, False otherwise

        Example:
            >>> client.shutdown(timeout_seconds=10)
            True
        """
        self._tracer_provider.shutdown()
        return True  # Always return True for consistency with flush()


# Singleton pattern with warning on config mismatch
_default_config: Optional[TelemetryConfig] = None


@lru_cache(maxsize=None)  # Cache all unique configs (thread-safe)
def _get_client_cached(config: TelemetryConfig) -> TelemetryClient:
    """Internal: Create client for given config (cached).

    Args:
        config: Telemetry configuration

    Returns:
        TelemetryClient singleton
    """
    return TelemetryClient(config)


def get_telemetry_client(config: Optional[TelemetryConfig] = None) -> TelemetryClient:
    """Get singleton telemetry client.

    Args:
        config: Optional configuration. If None, uses default config with
                environment variable overrides.

    Returns:
        TelemetryClient singleton

    Warning:
        Calling with different config after initialization will emit a warning.
        Use reset_telemetry_client() first to change configuration.

    Example:
        >>> # First call - creates client with config
        >>> config = TelemetryConfig(org_id="org-123", tenant_id="tenant-456")
        >>> client = get_telemetry_client(config)
        >>>
        >>> # Subsequent calls - returns same client
        >>> client2 = get_telemetry_client()  # Same instance
        >>> assert client is client2
    """
    global _default_config

    if config is None:
        if _default_config is None:
            _default_config = TelemetryConfig()  # Uses env vars
        config = _default_config
    else:
        if _default_config is not None and config != _default_config:
            warnings.warn(
                "get_telemetry_client() called with different config. "
                "Previous config will be ignored. Use reset_telemetry_client() first.",
                UserWarning,
                stacklevel=2,
            )
        _default_config = config

    return _get_client_cached(config)


def reset_telemetry_client() -> None:
    """Reset singleton (for testing or reconfiguration).

    Clears cached client instance. Next call to get_telemetry_client()
    will create a new client with provided config.

    Example:
        >>> # Change config mid-process (testing only)
        >>> reset_telemetry_client()
        >>> new_config = TelemetryConfig(org_id="org-456")
        >>> client = get_telemetry_client(new_config)
    """
    global _default_config
    _default_config = None
    _get_client_cached.cache_clear()
