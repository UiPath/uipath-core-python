"""UiPath LangGraph instrumentation using OpenInference patterns.

This module provides automatic instrumentation for LangGraph workflows using
OpenTelemetry conventions. It wraps StateGraph.compile() to inject tracing
callbacks that capture node executions, state changes, and routing decisions.

Example:
    >>> from uipath.core.telemetry.integrations.langgraph import instrument_langgraph
    >>> instrument_langgraph()
    >>>
    >>> # Normal LangGraph code - automatically traced
    >>> workflow = graph.compile()
    >>> workflow.invoke({"messages": ["test"]})
"""

from __future__ import annotations

from typing import Any

from .instrumentor import LangGraphInstrumentor

__all__ = [
    "instrument_langgraph",
    "uninstrument_langgraph",
    "LangGraphInstrumentor",
]

# Global instrumentor instance
_instrumentor: LangGraphInstrumentor | None = None


def instrument_langgraph(**kwargs: Any) -> None:
    """Instrument LangGraph with UiPath telemetry.

    Wraps StateGraph.compile() to inject LangGraphTracer callbacks automatically.
    This provides zero-code observability for all LangGraph workflows.

    Args:
        **kwargs: Additional instrumentation options:
            - tracer_provider: Optional OpenTelemetry TracerProvider (for testing)

    Example:
        >>> from uipath.core.telemetry.integrations.langgraph import instrument_langgraph
        >>> instrument_langgraph()
        >>>
        >>> # Compile graphs AFTER instrumentation for proper tracing
        >>> workflow = graph.compile()
        >>> workflow.invoke({"messages": ["test"]})
    """
    global _instrumentor

    if _instrumentor is not None:
        return  # Already instrumented

    _instrumentor = LangGraphInstrumentor()
    _instrumentor.instrument(**kwargs)


def uninstrument_langgraph() -> None:
    """Remove LangGraph instrumentation.

    Restores original StateGraph.compile() method.

    Example:
        >>> from uipath.core.telemetry.integrations.langgraph import uninstrument_langgraph
        >>> uninstrument_langgraph()
    """
    global _instrumentor

    if _instrumentor is None:
        return  # Not instrumented

    _instrumentor.uninstrument()
    _instrumentor = None
