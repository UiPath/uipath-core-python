"""Observation classes for wrapping OpenTelemetry spans.

This module provides high-level observation classes that wrap OTel spans
and provide semantic methods for recording inputs, outputs, and errors.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)


class Observation:
    """Base observation class wrapping an OpenTelemetry span.

    Provides semantic methods for recording span attributes, status, and exceptions.
    """

    def __init__(self, span: Span) -> None:
        """Initialize observation with OTel span.

        Args:
            span: OpenTelemetry span to wrap
        """
        self._span = span
        self._context_token: object | None = None

    def __enter__(self) -> Observation:
        """Enter context manager and activate span.

        Returns:
            Self for use in with statement
        """
        # CRITICAL FIX: Activate span in OpenTelemetry context
        # This ensures child spans properly link to this parent
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
        # CRITICAL FIX: Restore previous context
        if self._context_token is not None:
            context_api.detach(self._context_token)
            self._context_token = None

        if exc_val:
            self.record_exception(exc_val)
            self.set_status(StatusCode.ERROR, str(exc_val))
        self._span.end()

    def set_attribute(self, key: str, value: Any) -> Observation:
        """Set a span attribute with privacy enforcement.

        Args:
            key: Attribute key
            value: Attribute value

        Returns:
            Self for method chaining
        """
        try:
            # CRITICAL FIX: Apply privacy sanitization before setting attribute
            sanitized_value = self._sanitize_value(key, value)

            # Serialize complex types to JSON
            if isinstance(sanitized_value, (dict, list)):
                sanitized_value = json.dumps(sanitized_value)
            self._span.set_attribute(key, sanitized_value)
        except Exception as e:
            logger.warning("Failed to set attribute %s: %s", key, e)
        return self

    def _sanitize_value(self, key: str, value: Any) -> Any:
        """Apply privacy rules to attribute value.

        Args:
            key: Attribute key
            value: Original value

        Returns:
            Sanitized value
        """
        # Get privacy config from client
        try:
            from .client import get_client

            privacy_config = get_client().get_privacy_config()
        except RuntimeError:
            # Client not initialized, skip privacy
            return value

        if not privacy_config:
            return value

        # Apply redaction rules
        if privacy_config.get("redact_inputs") and "input" in key.lower():
            return "[REDACTED]"
        if privacy_config.get("redact_outputs") and "output" in key.lower():
            return "[REDACTED]"

        # Apply truncation rules
        max_length = privacy_config.get("max_attribute_length", 10000)
        if isinstance(value, str) and len(value) > max_length:
            return value[:max_length] + "...[truncated]"

        return value

    def set_status(
        self,
        status: StatusCode,
        description: str | None = None,
    ) -> Observation:
        """Set span status.

        Args:
            status: Status code (OK, ERROR, UNSET)
            description: Optional status description

        Returns:
            Self for method chaining
        """
        self._span.set_status(Status(status, description))
        return self

    def record_exception(
        self,
        exception: Exception,
        attributes: dict[str, Any] | None = None,
    ) -> Observation:
        """Record an exception in the span.

        Args:
            exception: Exception to record
            attributes: Optional additional attributes

        Returns:
            Self for method chaining
        """
        self._span.record_exception(exception, attributes=attributes)
        return self

    def update(self, provider_response: Any) -> None:
        """Update observation with provider response.

        This is a placeholder that should be overridden by subclasses
        to implement smart parsing of provider-specific responses.

        Args:
            provider_response: Response from provider (e.g., OpenAI, Anthropic)
        """
        logger.debug(
            "update() called on base Observation - no parsing implemented for type: %s",
            type(provider_response).__name__,
        )


class GenerationObservation(Observation):
    """Observation for LLM generation spans.

    Provides smart update() method that parses provider responses (OpenAI, Anthropic)
    and extracts relevant attributes (model, tokens, messages, etc.).
    """

    def __init__(self, span: Span, model: str | None = None) -> None:
        """Initialize generation observation.

        Args:
            span: OpenTelemetry span
            model: Model identifier (e.g., "gpt-4")
        """
        super().__init__(span)
        self._model = model
        if model:
            self.set_attribute("gen_ai.request.model", model)

    def update(self, provider_response: Any) -> None:
        """Update with provider response using smart parsing.

        Attempts to parse OpenAI and Anthropic response formats automatically.

        Args:
            provider_response: Response from LLM provider
        """
        from .parsers.registry import parse_provider_response

        try:
            attributes = parse_provider_response(provider_response)
            for key, value in attributes.items():
                self.set_attribute(key, value)
            logger.debug("Successfully parsed provider response: %d attributes", len(attributes))
        except Exception as e:
            logger.warning(
                "Failed to parse provider response (type=%s): %s",
                type(provider_response).__name__,
                e,
                exc_info=True,  # HIGH FIX: Add stack trace for debugging
            )
            # HIGH FIX: Record parsing error on span for visibility
            self.set_attribute("otel.parsing_error", str(e))
            self.set_attribute("otel.parsing_error_type", type(provider_response).__name__)
            self.record_exception(e)


class ToolObservation(Observation):
    """Observation for tool/function call spans."""

    def __init__(self, span: Span, tool_name: str | None = None) -> None:
        """Initialize tool observation.

        Args:
            span: OpenTelemetry span
            tool_name: Tool name
        """
        super().__init__(span)
        if tool_name:
            self.set_attribute("tool.name", tool_name)


class AgentObservation(Observation):
    """Observation for agent operation spans."""

    def __init__(self, span: Span, agent_name: str | None = None) -> None:
        """Initialize agent observation.

        Args:
            span: OpenTelemetry span
            agent_name: Agent name
        """
        super().__init__(span)
        if agent_name:
            self.set_attribute("agent.name", agent_name)


class RetrieverObservation(Observation):
    """Observation for retrieval operation spans."""

    def __init__(self, span: Span) -> None:
        """Initialize retriever observation.

        Args:
            span: OpenTelemetry span
        """
        super().__init__(span)
        self.set_attribute("span.type", "retriever")


class EmbeddingObservation(Observation):
    """Observation for embedding generation spans."""

    def __init__(self, span: Span, model: str | None = None) -> None:
        """Initialize embedding observation.

        Args:
            span: OpenTelemetry span
            model: Embedding model name
        """
        super().__init__(span)
        if model:
            self.set_attribute("gen_ai.request.model", model)
        self.set_attribute("span.type", "embedding")


class WorkflowObservation(Observation):
    """Observation for workflow spans."""

    def __init__(self, span: Span) -> None:
        """Initialize workflow observation.

        Args:
            span: OpenTelemetry span
        """
        super().__init__(span)
        self.set_attribute("span.type", "workflow")


class ActivityObservation(Observation):
    """Observation for activity spans."""

    def __init__(self, span: Span) -> None:
        """Initialize activity observation.

        Args:
            span: OpenTelemetry span
        """
        super().__init__(span)
        self.set_attribute("span.type", "activity")
