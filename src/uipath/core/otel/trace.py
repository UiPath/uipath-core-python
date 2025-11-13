"""Trace context manager for creating observation spans.

This module provides the Trace class which acts as a context manager for
creating traces and their child observation spans.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.trace import SpanKind

from .observation import (
    ActivityObservation,
    AgentObservation,
    EmbeddingObservation,
    GenerationObservation,
    RetrieverObservation,
    ToolObservation,
    WorkflowObservation,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer

logger = logging.getLogger(__name__)

# Ambient trace context for decorator access
_current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


class Trace:
    """Trace context manager for creating observation spans.

    Provides factory methods for creating semantic observation types and
    manages ambient trace context for decorator access.
    """

    def __init__(
        self,
        tracer: Tracer,
        name: str,
        execution_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize trace context manager.

        Args:
            tracer: OpenTelemetry tracer
            name: Trace name
            execution_id: Execution ID for correlation
            user_id: User ID for correlation
            metadata: Additional metadata attributes
        """
        self._tracer = tracer
        self._name = name
        self._execution_id = execution_id
        self._user_id = user_id
        self._metadata = metadata or {}
        self._root_span: Span | None = None
        self._context_token: object | None = None

    def __enter__(self) -> Trace:
        """Enter trace context and set as ambient trace.

        Returns:
            Self for use in with statement
        """
        # Create root span
        self._root_span = self._tracer.start_span(
            self._name,
            kind=SpanKind.INTERNAL,
        )

        # Set execution context attributes
        if self._execution_id:
            self._root_span.set_attribute("execution.id", self._execution_id)
        if self._user_id:
            self._root_span.set_attribute("user.id", self._user_id)

        # Set metadata attributes
        for key, value in self._metadata.items():
            self._root_span.set_attribute(key, value)

        # CRITICAL FIX: Activate span in OpenTelemetry context
        # This ensures proper parent-child relationships and async propagation
        ctx = trace.set_span_in_context(self._root_span)
        self._context_token = context_api.attach(ctx)

        # Set as ambient trace for decorator access
        _current_trace.set(self)

        logger.debug("Trace started: %s (execution_id=%s)", self._name, self._execution_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit trace context and end root span.

        Args:
            exc_type: Exception type if raised
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        # CRITICAL FIX: Restore previous context
        if self._context_token is not None:
            context_api.detach(self._context_token)
            self._context_token = None

        # Clear ambient trace
        _current_trace.set(None)

        # End root span
        if self._root_span:
            if exc_val:
                self._root_span.record_exception(exc_val)
                self._root_span.set_status(trace.Status(trace.StatusCode.ERROR))
            self._root_span.end()
            logger.debug("Trace ended: %s", self._name)

    def generation(
        self,
        name: str,
        model: str | None = None,
    ) -> GenerationObservation:
        """Create a generation observation span.

        Args:
            name: Span name
            model: Model identifier (e.g., "gpt-4")

        Returns:
            Generation observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        span.set_attribute("span.type", "generation")
        return GenerationObservation(span, model=model)

    def tool(self, name: str) -> ToolObservation:
        """Create a tool observation span.

        Args:
            name: Tool/function name

        Returns:
            Tool observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        span.set_attribute("span.type", "tool")
        return ToolObservation(span, tool_name=name)

    def agent(self, name: str) -> AgentObservation:
        """Create an agent observation span.

        Args:
            name: Agent name

        Returns:
            Agent observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        span.set_attribute("span.type", "agent")
        return AgentObservation(span, agent_name=name)

    def retriever(self, name: str) -> RetrieverObservation:
        """Create a retriever observation span.

        Args:
            name: Retriever name

        Returns:
            Retriever observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        return RetrieverObservation(span)

    def embedding(self, name: str, model: str | None = None) -> EmbeddingObservation:
        """Create an embedding observation span.

        Args:
            name: Span name
            model: Embedding model name

        Returns:
            Embedding observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        return EmbeddingObservation(span, model=model)

    def workflow(self, name: str) -> WorkflowObservation:
        """Create a workflow observation span.

        Args:
            name: Workflow name

        Returns:
            Workflow observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        return WorkflowObservation(span)

    def activity(self, name: str) -> ActivityObservation:
        """Create an activity observation span.

        Args:
            name: Activity name

        Returns:
            Activity observation context manager
        """
        span = self._tracer.start_span(name, kind=SpanKind.INTERNAL)
        return ActivityObservation(span)

    def get_url(self) -> str:
        """Get trace viewer URL.

        Returns:
            URL to view this trace in UiPath telemetry viewer
        """
        if not self._root_span:
            return ""

        span_context = self._root_span.get_span_context()
        trace_id = format(span_context.trace_id, "032x")
        return f"https://telemetry.uipath.com/trace/{trace_id}"


def get_current_trace() -> Trace | None:
    """Get current ambient trace from context.

    Returns:
        Current trace or None if not in trace context
    """
    return _current_trace.get()


def require_trace() -> Trace:
    """Get current trace or raise error if not in trace context.

    Returns:
        Current trace

    Raises:
        RuntimeError: If no active trace context
    """
    trace = get_current_trace()
    if trace is None:
        raise RuntimeError(
            "No active trace context. "
            "Use `with otel.trace(name) as trace:` before using decorators."
        )
    return trace
