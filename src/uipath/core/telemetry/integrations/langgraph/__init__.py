"""LangGraph instrumentation for UiPath OpenTelemetry integration.

This module provides comprehensive instrumentation for LangGraph framework including:
- Base LangChain callback tracing (LLM calls, chains, tools, retrievers)
- Graph topology capture (nodes, edges, entry points, conditional edges)
- Checkpoint tracking (save/load operations)
- State transition monitoring (MessagesState deltas, iteration tracking)

The instrumentation combines LangChain's callback system with LangGraph-specific
augmentation to provide complete observability for graph-based AI workflows.

Examples:
    Basic usage with zero-code instrumentation:

    >>> from uipath.core.telemetry.integrations.langgraph import LangGraphInstrumentor
    >>> from opentelemetry import trace
    >>> from opentelemetry.sdk.trace import TracerProvider
    >>>
    >>> # Setup OpenTelemetry
    >>> provider = TracerProvider()
    >>> trace.set_tracer_provider(provider)
    >>>
    >>> # Instrument LangGraph
    >>> instrumentor = LangGraphInstrumentor()
    >>> instrumentor.instrument()
    >>>
    >>> # Use LangGraph normally - all operations are automatically traced
    >>> from langgraph.graph import StateGraph
    >>> workflow = StateGraph(AgentState)
    >>> # ... define workflow ...
    >>> app = workflow.compile()  # Topology automatically captured
    >>> result = app.invoke({"messages": [...]})  # All operations traced

    Advanced usage with custom configuration:

    >>> from uipath.core.telemetry.integrations._shared import InstrumentationConfig
    >>>
    >>> config = InstrumentationConfig(
    ...     capture_inputs=True,
    ...     capture_outputs=True,
    ...     max_string_length=2048,
    ...     hide_sensitive_data=True,
    ... )
    >>> instrumentor.instrument(config=config)
"""

from __future__ import annotations

__all__ = [
    "LangGraphInstrumentor",
    "LangGraphAugmentation",
]

from ._augmentation import LangGraphAugmentation
from .instrumentor import LangGraphInstrumentor
