"""LangGraph augmentation for graph-specific features.

This module adds LangGraph-specific instrumentation on top of the base
LangChain instrumentation. It captures:
- Graph topology (nodes, edges, entry points)
- Checkpoints (save/load operations)
- State transitions (MessagesState deltas)
- Parallel execution patterns
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from opentelemetry.trace import SpanKind

from ...attributes import Attr
from .._shared import InstrumentationConfig, safe_json_dumps

if TYPE_CHECKING:
    from opentelemetry.trace import Span, Tracer


class LangGraphAugmentation:
    """Augmentation layer for LangGraph-specific tracing.

    Adds graph topology, checkpoint, and state transition tracking on top
    of the base LangChain callback instrumentation.

    Args:
        tracer: OpenTelemetry tracer instance
        config: Instrumentation configuration

    Examples:
        >>> from opentelemetry import trace
        >>> tracer = trace.get_tracer(__name__)
        >>> augmentation = LangGraphAugmentation(tracer)
        >>> augmentation.capture_graph_topology(app, "my-graph")
    """

    def __init__(
        self, tracer: Tracer, config: InstrumentationConfig | None = None
    ) -> None:
        """Initialize augmentation with tracer and config."""
        self._tracer = tracer
        self._config = config or InstrumentationConfig()

    def capture_graph_topology(self, graph_app: Any, graph_name: str) -> Span:
        """Capture graph topology as a span.

        Creates a span containing the graph structure including nodes,
        edges, entry point, and conditional edges.

        Args:
            graph_app: Compiled LangGraph application
            graph_name: Name for the graph span

        Returns:
            OpenTelemetry span with graph topology attributes
        """
        span = self._tracer.start_span(
            f"graph.topology.{graph_name}",
            kind=SpanKind.INTERNAL,
        )

        try:
            # Extract graph structure from compiled app
            if hasattr(graph_app, "get_graph"):
                graph = graph_app.get_graph()

                # Nodes
                if hasattr(graph, "nodes"):
                    nodes = [str(node) for node in graph.nodes]
                    span.set_attribute(
                        Attr.Graph.NODES,
                        safe_json_dumps(nodes, self._config.max_string_length),
                    )

                # Edges
                if hasattr(graph, "edges"):
                    edges = [
                        {"from": str(edge[0]), "to": str(edge[1])}
                        for edge in graph.edges
                    ]
                    span.set_attribute(
                        Attr.Graph.EDGES,
                        safe_json_dumps(edges, self._config.max_string_length),
                    )

                # Entry point
                if hasattr(graph, "entry_point"):
                    span.set_attribute(Attr.Graph.ENTRY_POINT, str(graph.entry_point))

                # Conditional edges
                if hasattr(graph, "conditional_edges"):
                    conditional = [
                        {"from": str(k), "condition": str(v)}
                        for k, v in graph.conditional_edges.items()
                    ]
                    span.set_attribute(
                        Attr.Graph.CONDITIONAL_EDGES,
                        safe_json_dumps(conditional, self._config.max_string_length),
                    )

        finally:
            span.end()

        return span

    def capture_checkpoint_save(
        self, checkpoint_id: str, checkpoint_data: dict[str, Any]
    ) -> Span:
        """Capture checkpoint save operation.

        Args:
            checkpoint_id: Unique checkpoint identifier
            checkpoint_data: Checkpoint data being saved

        Returns:
            OpenTelemetry span for checkpoint save
        """
        span = self._tracer.start_span(
            f"checkpoint.save.{checkpoint_id}",
            kind=SpanKind.INTERNAL,
        )

        try:
            span.set_attribute(Attr.Checkpoint.ID, checkpoint_id)

            # Extract metadata if available
            if "metadata" in checkpoint_data:
                metadata_str = safe_json_dumps(
                    checkpoint_data["metadata"], self._config.max_string_length
                )
                span.set_attribute(Attr.Checkpoint.METADATA, metadata_str)

            # Timestamp
            if "ts" in checkpoint_data or "timestamp" in checkpoint_data:
                ts = checkpoint_data.get("ts") or checkpoint_data.get("timestamp")
                span.set_attribute(Attr.Checkpoint.TIMESTAMP, str(ts))

        finally:
            span.end()

        return span

    def capture_checkpoint_load(self, checkpoint_id: str) -> Span:
        """Capture checkpoint load operation.

        Args:
            checkpoint_id: Unique checkpoint identifier

        Returns:
            OpenTelemetry span for checkpoint load
        """
        span = self._tracer.start_span(
            f"checkpoint.load.{checkpoint_id}",
            kind=SpanKind.INTERNAL,
        )

        try:
            span.set_attribute(Attr.Checkpoint.ID, checkpoint_id)

        finally:
            span.end()

        return span

    def capture_state_transition(
        self,
        before_state: dict[str, Any],
        after_state: dict[str, Any],
        node_name: str,
    ) -> Span:
        """Capture state transition during node execution.

        Analyzes MessagesState changes to track message additions/removals
        and iteration deltas.

        Args:
            before_state: State before node execution
            after_state: State after node execution
            node_name: Name of the executing node

        Returns:
            OpenTelemetry span for state transition
        """
        span = self._tracer.start_span(
            f"state.transition.{node_name}",
            kind=SpanKind.INTERNAL,
        )

        try:
            # Analyze message changes
            before_messages = before_state.get("messages", [])
            after_messages = after_state.get("messages", [])

            messages_added = len(after_messages) - len(before_messages)
            if messages_added > 0:
                span.set_attribute(Attr.State.MESSAGES_ADDED, messages_added)
            elif messages_added < 0:
                span.set_attribute(Attr.State.MESSAGES_REMOVED, abs(messages_added))

            span.set_attribute(Attr.State.MESSAGES_TOTAL, len(after_messages))

            # Iteration delta
            before_iteration = before_state.get("iteration", 0)
            after_iteration = after_state.get("iteration", 0)
            iteration_delta = after_iteration - before_iteration

            if iteration_delta != 0:
                span.set_attribute(Attr.State.ITERATION_DELTA, iteration_delta)

        finally:
            span.end()

        return span
