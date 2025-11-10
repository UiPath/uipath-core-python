"""Observation span wrapper with privacy controls and semantic methods.

Provides a user-friendly wrapper around OpenTelemetry Span with:
- Semantic update methods (set_attribute, fail, etc.)
- Input/output hiding (privacy controls)
- JSON serialization for complex objects
- Context manager protocol

Example:
    >>> from uipath.core.telemetry import get_telemetry_client
    >>>
    >>> client = get_telemetry_client()
    >>> with client.start_as_current_span(
    ...     "process",
    ...     semantic_type="automation",
    ...     hide_input=True
    ... ) as span:
    ...     span.set_attribute("status", "running")
    ...     result = do_work()
    ...     span.update_output(result)
"""

import json
from typing import Any, ContextManager, Optional

from opentelemetry.trace import Span, StatusCode


class ObservationSpan:
    """Wrapper around OpenTelemetry Span with semantic methods.

    Always wraps a ContextManager[Span] for consistent lifecycle management.
    Must be used as a context manager.

    Uses composition (not inheritance) to provide a simplified API
    while preserving full OTel Span functionality.

    Example:
        >>> client = get_telemetry_client()
        >>> with client.start_as_current_span("process") as span:
        ...     span.set_attribute("key", "value")
        ...     span.update_input({"param": "value"})
        ...     span.update_output({"result": "success"})
    """

    def __init__(
        self,
        span_cm: ContextManager[Span],
        hide_input: bool = False,
        hide_output: bool = False,
    ):
        """Initialize span wrapper.

        Args:
            span_cm: Context manager that yields an OpenTelemetry Span
            hide_input: Hide input from telemetry (privacy)
            hide_output: Hide output from telemetry (privacy)
        """
        self._span_cm = span_cm
        self._span: Optional[Span] = None  # Set in __enter__
        self._hide_input = hide_input
        self._hide_output = hide_output

    def __enter__(self) -> "ObservationSpan":
        """Enter context manager (activate span).

        Returns:
            Self for context manager protocol
        """
        self._span = self._span_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager (end span, record exceptions).

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised

        Returns:
            Optional[bool]: True to suppress exceptions, False/None otherwise
        """
        if exc_type is not None:
            self.fail(exc_val)

        return self._span_cm.__exit__(exc_type, exc_val, exc_tb)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute (key-value metadata).

        Args:
            key: Attribute key (e.g., "uipath.job_id")
            value: Attribute value (str, int, float, bool, or JSON-serializable)

        Raises:
            RuntimeError: If span not active (must use as context manager)

        Example:
            >>> with span:
            ...     span.set_attribute("uipath.job_id", "job-123")
            ...     span.set_attribute("retry_count", 3)
            ...     span.set_attribute("config", {"timeout": 30})
        """
        if self._span is None:
            raise RuntimeError(
                "Span not active. ObservationSpan must be used as a context manager:\n"
                "  with client.start_as_current_span(...) as span:\n"
                "      span.set_attribute(...)"
            )
        serialized = _serialize_value(value)
        self._span.set_attribute(key, serialized)

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        """Set multiple span attributes.

        Args:
            attributes: Dictionary of key-value pairs

        Raises:
            RuntimeError: If span not active (must use as context manager)

        Example:
            >>> with span:
            ...     span.set_attributes({
            ...         "uipath.job_id": "job-123",
            ...         "uipath.process_key": "InvoiceProcessing",
            ...         "retry_count": 3
            ...     })
        """
        for key, value in attributes.items():
            self.set_attribute(key, value)

    def update_input(self, input_data: Any) -> None:
        """Record input data (unless hide_input=True).

        Args:
            input_data: Function input or request data

        Raises:
            RuntimeError: If span not active (must use as context manager)

        Example:
            >>> with span:
            ...     span.update_input({"invoice_id": "INV-123", "amount": 1500})
        """
        if not self._hide_input:
            self.set_attribute("input", input_data)

    def update_output(self, output_data: Any) -> None:
        """Record output data (unless hide_output=True).

        Args:
            output_data: Function output or response data

        Raises:
            RuntimeError: If span not active (must use as context manager)

        Example:
            >>> with span:
            ...     span.update_output({"status": "processed", "result_id": "RES-456"})
        """
        if not self._hide_output:
            self.set_attribute("output", output_data)

    def fail(self, error: Exception) -> None:
        """Mark span as failed and record exception.

        Args:
            error: Exception that caused failure

        Raises:
            RuntimeError: If span not active (must use as context manager)

        Example:
            >>> with span:
            ...     try:
            ...         risky_operation()
            ...     except ValueError as e:
            ...         span.fail(e)
            ...         raise
        """
        if self._span is None:
            raise RuntimeError(
                "Span not active. ObservationSpan must be used as a context manager."
            )
        self._span.record_exception(error)
        self._span.set_status(StatusCode.ERROR, str(error))

    def end(self) -> None:
        """End span (if not using context manager).

        Note: Prefer using as context manager instead.

        Raises:
            RuntimeError: If span not active

        Example:
            >>> span = client.start_as_current_span("process")
            >>> span.__enter__()  # Manual activation
            >>> try:
            ...     do_work()
            ... finally:
            ...     span.end()
        """
        if self._span is None:
            raise RuntimeError(
                "Span not active. ObservationSpan must be used as a context manager."
            )
        self._span.end()

    @property
    def span_id(self) -> str:
        """Get span ID (for debugging).

        Returns:
            Span ID as hex string

        Raises:
            RuntimeError: If span not active

        Example:
            >>> with span:
            ...     print(f"Span ID: {span.span_id}")
            Span ID: 1234567890abcdef
        """
        if self._span is None:
            raise RuntimeError(
                "Span not active. ObservationSpan must be used as a context manager."
            )
        return format(self._span.get_span_context().span_id, "016x")

    @property
    def trace_id(self) -> str:
        """Get trace ID (for debugging).

        Returns:
            Trace ID as hex string

        Raises:
            RuntimeError: If span not active

        Example:
            >>> with span:
            ...     print(f"Trace ID: {span.trace_id}")
            Trace ID: 1234567890abcdef1234567890abcdef
        """
        if self._span is None:
            raise RuntimeError(
                "Span not active. ObservationSpan must be used as a context manager."
            )
        return format(self._span.get_span_context().trace_id, "032x")


def _safe_default_serializer(o: Any) -> str:
    """Return a safe string representation for non-serializable objects.

    Prevents accidental data leakage via __str__/__repr__ methods by
    returning only the type name for complex objects.

    Args:
        o: Object to serialize

    Returns:
        Safe string representation (type name only)

    Example:
        >>> class UserSession:
        ...     def __init__(self, password):
        ...         self.password = password
        >>> _safe_default_serializer(UserSession("secret"))
        '<UserSession object>'
    """
    return f"<{type(o).__name__} object>"


def _serialize_value(value: Any) -> Any:
    """Serialize value for span attribute.

    OpenTelemetry supports: str, bool, int, float, list[str|bool|int|float]
    Complex objects are JSON-serialized.

    Args:
        value: Value to serialize

    Returns:
        Serialized value (OTel-compatible type)

    Example:
        >>> _serialize_value("hello")
        'hello'
        >>> _serialize_value({"key": "value"})
        '{"key": "value"}'
        >>> _serialize_value([1, 2, 3])
        [1, 2, 3]
    """
    if isinstance(value, (str, bool, int, float)):
        return value

    if isinstance(value, list) and all(
        isinstance(v, (str, bool, int, float)) for v in value
    ):
        return value

    try:
        return json.dumps(value, default=_safe_default_serializer)
    except (TypeError, ValueError):
        return f"<unserializable type: {type(value).__name__}>"
