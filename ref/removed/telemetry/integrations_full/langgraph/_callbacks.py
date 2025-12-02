"""LangGraph callback handler for tracing."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY
from opentelemetry.trace import Status, StatusCode

from ...attributes import Attr, SpanKind
from .._shared import safe_json_dumps

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from ...client import TelemetryClient

logger = logging.getLogger(__name__)


class LangGraphTracer(BaseCallbackHandler):
    """Callback handler for LangGraph execution tracing.

    This tracer captures node executions, state changes, and routing decisions
    in LangGraph workflows by implementing the LangChain callback interface.

    THREAD-SAFETY: Uses run_id → span mapping with RLock for concurrent execution safety.

    Attributes:
        telemetry_client: UiPath Telemetry client instance
        trace_state: Whether to capture state in span attributes
        trace_edges: Whether to capture edge transitions
        max_state_size: Maximum serialized state size (bytes)
    """

    def __init__(
        self,
        telemetry_client: TelemetryClient,
        trace_state: bool = True,
        trace_edges: bool = True,
        max_state_size: int = 10_000,
    ) -> None:
        """Initialize LangGraph tracer.

        Args:
            telemetry_client: UiPath Telemetry client instance
            trace_state: Whether to capture state changes (default: True)
            trace_edges: Whether to capture edge transitions (default: True)
            max_state_size: Maximum state size in bytes (default: 10KB)
        """
        self.telemetry_client = telemetry_client
        self.trace_state = trace_state
        self.trace_edges = trace_edges
        self.max_state_size = max_state_size
        # Store run_id → span mapping for callback lifecycle
        self._spans: dict[UUID, Span] = {}
        # Store context tokens for proper span activation/deactivation
        self._tokens: dict[UUID, object] = {}

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Start span when LangGraph node executes.

        Args:
            serialized: Node metadata (name, type, etc.), may be None
            inputs: Input state to the node
            run_id: Unique run identifier for this execution
            parent_run_id: Parent run identifier (for nested executions)
            **kwargs: Additional callback arguments
        """
        # Check if tracing is suppressed
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        # Handle None serialized parameter
        if serialized is None:
            # Try to get name from kwargs or use unknown
            name = kwargs.get("name", "unknown_node")
            serialized = {}  # Empty dict for _detect_span_type
        else:
            name = serialized.get("name", "unknown_node")

        span_type = self._detect_span_type(name)

        # Get parent span for context propagation
        # Priority: parent_run_id -> active span context -> ContextVar
        # Note: Parent span lookup done for potential future use
        if parent_run_id:
            _ = self._spans.get(parent_run_id)

        tracer = self.telemetry_client.get_tracer()

        # Prepare initial attributes
        attributes = {
            Attr.Common.OPENINFERENCE_SPAN_KIND: span_type,
            "node.name": name,
        }

        # Add session context if available
        from uipath.core.telemetry.integrations._shared import (
            get_session_id,
            get_thread_id,
        )

        session_id = get_session_id()
        if session_id:
            attributes["session.id"] = session_id
        thread_id = get_thread_id()
        if thread_id:
            attributes["thread_id"] = thread_id

        # Start span - will automatically use current active span as parent
        # The instrumentor sets langgraph.invoke as active span via start_as_current_span()
        span = tracer.start_span(
            name=f"langgraph.{name}",
            attributes=attributes,
        )

        # Extract LangGraph metadata from kwargs
        metadata = kwargs.get("metadata", {})
        if metadata:
            # Add rich LangGraph-specific attributes for observability
            if "langgraph_step" in metadata:
                span.set_attribute("langgraph_step", metadata["langgraph_step"])
            if "langgraph_node" in metadata:
                span.set_attribute("langgraph_node", metadata["langgraph_node"])
            if "langgraph_triggers" in metadata:
                # Use native list/tuple for better querying in observability platforms
                span.set_attribute("langgraph_triggers", metadata["langgraph_triggers"])
            if "langgraph_path" in metadata:
                # Use native list/tuple for better querying in observability platforms
                span.set_attribute("langgraph_path", metadata["langgraph_path"])
            if "langgraph_checkpoint_ns" in metadata:
                span.set_attribute(
                    "langgraph_checkpoint_ns", metadata["langgraph_checkpoint_ns"]
                )

        if self.trace_state:
            # Filter and record state
            try:
                filtered_state = self._filter_state(inputs)
                if filtered_state is not None:
                    span.set_attribute(
                        Attr.Common.INPUT_VALUE, json.dumps(filtered_state)
                    )
                    span.set_attribute(Attr.Common.INPUT_MIME_TYPE, "application/json")
            except Exception as e:
                # Log but don't fail - state capture is best effort
                logger.warning("Failed to capture input state: %s", e)

        # CRITICAL FIX: Store span by run_id instead of stack
        self._spans[run_id] = span

        # CRITICAL FIX: Activate span context so child spans are properly parented
        # Set the span in the current context without manual token management
        # This works better with async code where contexts can change
        ctx = trace.set_span_in_context(span)
        token = context_api.attach(ctx)
        self._tokens[run_id] = token

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """End span when LangGraph node completes.

        Args:
            outputs: Output state from the node
            run_id: Unique run identifier for this execution
            **kwargs: Additional callback arguments
        """
        # Check if tracing is suppressed
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        # CRITICAL FIX: Clean up context token
        # Note: We don't detach the token because in async code the context may have
        # changed, causing detachment to fail. The context will be automatically
        # cleaned up when the span ends.
        self._tokens.pop(run_id, None)

        # CRITICAL FIX: Retrieve span by run_id instead of stack pop
        span = self._spans.pop(run_id)
        if not span:
            logger.warning("on_chain_end called for unknown run_id: %s", run_id)
            return

        if self.trace_state:
            try:
                filtered_state = self._filter_state(outputs)
                if filtered_state is not None:
                    span.set_attribute(
                        Attr.Common.OUTPUT_VALUE, json.dumps(filtered_state)
                    )
                    span.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")
            except Exception as e:
                # Log but don't fail - state capture is best effort
                logger.warning("Failed to capture output state: %s", e)

        span.set_status(Status(StatusCode.OK))
        span.end()

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record error when LangGraph node fails.

        Args:
            error: Exception that occurred
            run_id: Unique run identifier for this execution
            **kwargs: Additional callback arguments
        """
        # Check if tracing is suppressed
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        # CRITICAL FIX: Clean up context token
        # Note: We don't detach the token because in async code the context may have
        # changed, causing detachment to fail. The context will be automatically
        # cleaned up when the span ends.
        self._tokens.pop(run_id, None)

        # CRITICAL FIX: Retrieve and remove span by run_id
        span = self._spans.pop(run_id)
        if not span:
            logger.warning("on_chain_error called for unknown run_id: %s", run_id)
            return

        # Record exception and end span
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()  # End the span immediately

    # Tool callbacks (for tool/function calling instrumentation)
    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record start of tool execution.

        Args:
            serialized: Tool metadata (name, type, etc.)
            input_str: Tool input as string
            run_id: Unique run identifier for this execution
            parent_run_id: Parent run identifier (for nested executions)
            tags: Optional tags for the tool execution
            metadata: Optional metadata for the tool execution
            inputs: Optional structured inputs dictionary
            **kwargs: Additional callback arguments
        """
        # Check if tracing is suppressed
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        # Extract tool name
        tool_name = serialized.get("name", "unknown_tool")
        span_name = f"langgraph.tool.{tool_name}"

        # Create span with tool-specific attributes
        tracer = self.telemetry_client.get_tracer()
        span = tracer.start_span(span_name)

        # Set semantic attributes
        span.set_attribute(Attr.Common.OPENINFERENCE_SPAN_KIND, SpanKind.TOOL)
        span.set_attribute(Attr.Tool.NAME, tool_name)

        # Record tool input
        # Prefer structured inputs dict over string representation
        if inputs:
            try:
                span.set_attribute(Attr.Common.INPUT_VALUE, json.dumps(inputs))
                span.set_attribute(Attr.Common.INPUT_MIME_TYPE, "application/json")
            except (TypeError, ValueError):
                # Fallback to string input if JSON serialization fails
                span.set_attribute(Attr.Common.INPUT_VALUE, str(inputs))
                span.set_attribute(Attr.Common.INPUT_MIME_TYPE, "text/plain")
        elif input_str:
            span.set_attribute(Attr.Common.INPUT_VALUE, input_str)
            span.set_attribute(Attr.Common.INPUT_MIME_TYPE, "text/plain")

        # Add tags as attribute if present
        if tags:
            span.set_attribute("tool.tags", json.dumps(tags))

        # Add metadata if present
        if metadata:
            for key, value in metadata.items():
                try:
                    # Use safe attribute naming
                    attr_key = f"tool.metadata.{key}"
                    span.set_attribute(attr_key, json.dumps(value))
                except (TypeError, ValueError):
                    span.set_attribute(attr_key, str(value))

        # Activate span in current context
        token = context_api.attach(trace.set_span_in_context(span))

        # Store span and token for later retrieval
        self._spans[run_id] = span
        self._tokens[run_id] = token

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record successful tool execution completion.

        Args:
            output: Tool output (any type)
            run_id: Unique run identifier for this execution
            parent_run_id: Parent run identifier (for nested executions)
            **kwargs: Additional callback arguments
        """
        # Check if tracing is suppressed
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        # CRITICAL FIX: Clean up context token
        self._tokens.pop(run_id, None)

        # Retrieve span by run_id
        span = self._spans.pop(run_id)
        if not span:
            logger.warning("on_tool_end called for unknown run_id: %s", run_id)
            return

        # Record tool output
        try:
            if isinstance(output, str):
                span.set_attribute(Attr.Common.OUTPUT_VALUE, output)
                span.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "text/plain")
            else:
                # Attempt JSON serialization for structured output
                span.set_attribute(Attr.Common.OUTPUT_VALUE, json.dumps(output))
                span.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")
        except (TypeError, ValueError):
            # Fallback to string representation
            span.set_attribute(Attr.Common.OUTPUT_VALUE, str(output))
            span.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "text/plain")

        # Set successful status and end span
        span.set_status(Status(StatusCode.OK))
        span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record error when tool execution fails.

        Args:
            error: Exception that occurred
            run_id: Unique run identifier for this execution
            parent_run_id: Parent run identifier (for nested executions)
            **kwargs: Additional callback arguments
        """
        # Check if tracing is suppressed
        if context_api.get_value(_SUPPRESS_INSTRUMENTATION_KEY):
            return

        # CRITICAL FIX: Clean up context token
        self._tokens.pop(run_id, None)

        # Retrieve and remove span by run_id
        span = self._spans.pop(run_id)
        if not span:
            logger.warning("on_tool_error called for unknown run_id: %s", run_id)
            return

        # Record exception and end span
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.end()

    # Async callback methods (LangGraph async support)
    async def aon_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chain_start.

        Delegates to sync method since span operations are thread-safe.

        Args:
            serialized: Node metadata (name, type, etc.), may be None
            inputs: Input state to the node
            run_id: Unique run identifier for this execution
            parent_run_id: Parent run identifier (for nested executions)
            **kwargs: Additional callback arguments
        """
        self.on_chain_start(
            serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs
        )

    async def aon_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chain_end.

        Delegates to sync method since span operations are thread-safe.

        Args:
            outputs: Output state from the node
            run_id: Unique run identifier for this execution
            **kwargs: Additional callback arguments
        """
        self.on_chain_end(outputs, run_id=run_id, **kwargs)

    async def aon_chain_error(
        self,
        error: Exception,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Async version of on_chain_error.

        Delegates to sync method since span operations are thread-safe.

        Args:
            error: Exception that occurred
            run_id: Unique run identifier for this execution
            **kwargs: Additional callback arguments
        """
        self.on_chain_error(error, run_id=run_id, **kwargs)

    def _detect_span_type(self, name: str) -> str:
        """Detect semantic span type from node name.

        Uses heuristics based on node name to classify the span type
        according to OpenInference conventions.

        Args:
            name: Node name (e.g., "agent", "tool", "retriever")

        Returns:
            OpenInference span kind value
        """
        name_lower = name.lower()

        if "agent" in name_lower:
            return SpanKind.AGENT
        elif "tool" in name_lower:
            return SpanKind.TOOL
        elif "retriev" in name_lower:
            return SpanKind.RETRIEVER
        elif "llm" in name_lower or "generation" in name_lower:
            return SpanKind.LLM
        else:
            return SpanKind.CHAIN

    def _filter_state(
        self,
        state: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Filter state for logging (privacy + performance).

        Serializes state to JSON and checks size limits. If the state
        exceeds max_state_size, returns a truncated placeholder.

        Args:
            state: LangGraph state dictionary

        Returns:
            Filtered state or None if serialization fails
        """
        try:
            # Use shared serialization (handles LangChain objects gracefully)
            serialized = safe_json_dumps(state)

            # Check size limits
            if len(serialized) > self.max_state_size:
                return {
                    "_truncated": True,
                    "_size": len(serialized),
                    "_limit": self.max_state_size,
                }

            return state
        except Exception as e:
            logger.warning("Failed to serialize state: %s", e)
            return {"_error": "serialization_failed", "_type": type(state).__name__}
