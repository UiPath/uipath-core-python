"""Trace context manager for creating observation spans.

This module provides the Trace class which acts as a context manager for
creating traces and their child observation spans.

Note:
    To access the current trace, use the context manager variable:

        with telemetry.trace("workflow") as trace:
            # Use 'trace' directly to create child spans
            with trace.span("operation", kind="tool") as obs:
                ...

    For accessing the current span from OpenTelemetry:
        from opentelemetry import trace
        current_span = trace.get_current_span()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from .attributes import Attr
from .observation import ObservationSpan

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

logger = logging.getLogger(__name__)


class Trace:
    """Trace context manager for creating observation spans.

    Provides a generic span() method for creating observation spans.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        execution_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize trace context manager.

        Args:
            tracer: OpenTelemetry tracer
            name: Trace name
            execution_id: Execution ID for correlation
            metadata: Additional metadata attributes
        """
        self._tracer = tracer
        self._name = name
        self._execution_id = execution_id
        self._metadata = metadata or {}
        self._root_span: Span | None = None
        self._context_token: object | None = None

    def __enter__(self) -> Trace:
        """Enter trace context.

        Returns:
            Self for use in with statement
        """
        self._root_span = self._tracer.start_span(
            self._name,
            kind=SpanKind.INTERNAL,
        )

        if self._execution_id:
            self._root_span.set_attribute(Attr.UiPath.EXECUTION_ID, self._execution_id)

        for key, value in self._metadata.items():
            self._root_span.set_attribute(key, value)

        ctx = trace.set_span_in_context(self._root_span)
        self._context_token = context_api.attach(ctx)

        logger.debug(
            "Trace started: %s (execution_id=%s)", self._name, self._execution_id
        )
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit trace context and end root span.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        if self._context_token is not None:
            context_api.detach(self._context_token)  # type: ignore[arg-type]
        self._context_token = None

        if self._root_span is not None:
            if exc_val:
                self._root_span.record_exception(exc_val)
                self._root_span.set_status(trace.Status(trace.StatusCode.ERROR))
            self._root_span.end()
        logger.debug("Trace ended: %s", self._name)

    def span(
        self,
        name: str,
        kind: str,
    ) -> ObservationSpan:
        """Create an observation span of specified kind.

        Args:
            name: Span name
            kind: Span kind (generation, tool, agent, retriever, embedding, workflow, activity)

        Returns:
            ObservationSpan context manager

        Example:
            with telemetry.trace("workflow") as trace:
                # Create generation span
                with trace.span("llm-call", kind="generation") as obs:
                    obs.record_input({"prompt": "Hello"})
                    result = call_llm()
                    obs.record_output(result)
                    obs.update(result)

                with trace.span("search", kind="tool") as obs:
                    obs.record_input({"query": "..."})
                    results = search_tool()
                    obs.record_output(results)
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        return ObservationSpan(span, kind=kind)
