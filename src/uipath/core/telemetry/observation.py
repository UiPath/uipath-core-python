"""Observation wrapper for OpenTelemetry spans.

Minimal design: no processors, no kwargs, parser-driven metadata extraction.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .attributes import Attr
from ._utils import safe_json_dumps

if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


class ObservationSpan:
    """Observation wrapper for OpenTelemetry spans.

    Minimal design: Just wrap span and set kind. All metadata comes from
    parsers via update() method for guaranteed accuracy.

    Example:
        span = tracer.start_span("llm-call")
        obs = ObservationSpan(span, kind="generation")

        with obs:
            obs.record_input({"prompt": "Hello"})
            response = openai.chat.completions.create(...)
            obs.record_output(response)
            obs.update(response)  # Parsers extract model, tokens, etc.
    """

    def __init__(self, span: Span, kind: str | None = None):
        """Initialize observation with span.

        Args:
            span: OpenTelemetry span to wrap
            kind: Span kind for classification (generation, tool, agent, etc.)
                 No metadata is set from kwargs - use update() with parser instead.
        """
        self._span = span
        self._kind = kind
        self._context_token: object | None = None

        if kind:
            span.set_attribute(Attr.Common.OPENINFERENCE_SPAN_KIND, kind.upper())

    def __enter__(self) -> ObservationSpan:
        """Enter context manager and activate span.

        Returns:
            Self for use in with statement
        """
        ctx = trace.set_span_in_context(self._span)
        self._context_token = context_api.attach(ctx)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager, restore context, and end span.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        if self._context_token is not None:
            context_api.detach(self._context_token)  # type: ignore[arg-type]
        self._context_token = None

        if exc_val:
            self.record_exception(exc_val)
        self._span.end()

    def record_input(self, value: Any, hide: bool = False) -> ObservationSpan:
        """Record input value.

        Args:
            value: Input value to record
            hide: If True, record as "[REDACTED]" instead of actual value

        Returns:
            Self for method chaining
        """
        if hide:
            self.set_attribute(Attr.Common.INPUT_VALUE, "[REDACTED]")
            return self

        # Use shared serialization utility for consistency
        serialized = safe_json_dumps(value)
        self.set_attribute(Attr.Common.INPUT_VALUE, serialized)
        self.set_attribute(Attr.Common.INPUT_MIME_TYPE, "application/json")

        return self

    def record_output(self, value: Any, hide: bool = False) -> ObservationSpan:
        """Record output value.

        Args:
            value: Output value to record
            hide: If True, record as "[REDACTED]" instead of actual value

        Returns:
            Self for method chaining

        Note:
            For LLM responses, call update(response) after this to extract
            metadata like model, tokens, etc. via parsers.
        """
        if hide:
            self.set_attribute(Attr.Common.OUTPUT_VALUE, "[REDACTED]")
            return self

        # Use shared serialization utility for consistency
        serialized = safe_json_dumps(value)
        self.set_attribute(Attr.Common.OUTPUT_VALUE, serialized)
        self.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")

        return self

    def record_exception(
        self, exception: Exception, attributes: dict[str, Any] | None = None
    ) -> ObservationSpan:
        """Record exception on span.

        Args:
            exception: Exception to record
            attributes: Optional custom attributes for the exception event

        Returns:
            Self for method chaining
        """
        try:
            self._span.record_exception(exception, attributes=attributes)
            self._span.set_status(Status(StatusCode.ERROR, str(exception)))

            error_type = type(exception).__name__
            self.set_attribute(Attr.Error.TYPE, error_type)

            error_str = str(exception).lower()
            if "rate" in error_str or "429" in error_str:
                self.set_attribute(Attr.Error.RATE_LIMITED, True)
        except Exception as e:
            logger.error("Failed to record exception: %s", e, exc_info=True)

        return self

    def set_attribute(self, key: str, value: Any) -> ObservationSpan:
        """Set span attribute.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            Self for method chaining
        """
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)

            self._span.set_attribute(key, value)
        except Exception as e:
            logger.warning("Failed to set attribute %s: %s", key, e)

        return self

    def set_status(
        self,
        status: StatusCode,
        description: str | None = None,
    ) -> ObservationSpan:
        """Set span status.

        Args:
            status: Status code (OK, ERROR, UNSET)
            description: Optional status description

        Returns:
            Self for method chaining
        """
        self._span.set_status(Status(status, description))
        return self

    def set_status_ok(self) -> ObservationSpan:
        """Set span status to OK (successful completion).

        Returns:
            Self for method chaining
        """
        self._span.set_status(Status(StatusCode.OK))
        return self

    def update(self, provider_response: Any) -> ObservationSpan:
        """Update observation with provider response (parser-based extraction).

        This is the KEY method for accurate metadata. Parsers extract:
        - Model name (actual model used, not declared)
        - Token counts (prompt, completion, total)
        - Response IDs, finish reasons, tool calls
        - Any provider-specific metadata

        Args:
            provider_response: Response object from LLM provider
                             (OpenAI ChatCompletion, Anthropic Message, etc.)

        Returns:
            Self for method chaining

        Example:
            obs = ObservationSpan(span, kind="generation")
            response = openai.chat.completions.create(...)
            obs.record_output(response)
            obs.update(response)  # Extracts model, tokens, etc.
        """
        try:
            # NOTE: Parser registry was removed with integrations_full deprecation.
            # This method is now a no-op. For rich metadata extraction, use
            # integrations_openinference which delegates to OpenInference.
            logger.debug(
                "update() is deprecated - use integrations_openinference for automatic metadata extraction"
            )
        except Exception as e:
            logger.warning(
                "Failed to parse provider response (type=%s): %s",
                type(provider_response).__name__,
                e,
                exc_info=True,
            )
            self.set_attribute(Attr.Internal.PARSING_ERROR, str(e))
            self.set_attribute(
                Attr.Internal.PARSING_ERROR_TYPE,
                type(provider_response).__name__,
            )

        return self
