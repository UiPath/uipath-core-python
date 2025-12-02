"""UiPath wrapper around OpenInference for LangChain/LangGraph instrumentation.

This package provides a thin wrapper around OpenInference that adds UiPath session
context (session_id, thread_id) to all spans.

**What OpenInference Provides (delegated):**
- Automatic LangChain and LangGraph instrumentation
- Rich metadata extraction (tokens, costs, model names)
- Node-level tracing for LangGraph
- OpenInference semantic conventions (llm.*, tool.*, embedding.*)

**What UiPath Adds:**
- Session context attributes (session_id, thread_id)
- Simple, self-contained implementation (~50 LOC)

Examples:
    Basic usage::

        from uipath.core.telemetry import init
        from uipath.core.telemetry.integrations_openinference import (
            instrument_langchain,
            set_session_context,
        )

        # Initialize telemetry
        init(enable_console_export=True)

        # Instrument LangChain/LangGraph
        instrument_langchain()

        # Set session context (optional)
        set_session_context(session_id="session-123", thread_id="thread-456")

        # Now all LangChain/LangGraph operations are automatically traced!
        from langgraph.graph import StateGraph

        builder = StateGraph(dict)
        builder.add_node("process", lambda x: {"result": x["a"] * x["b"]})
        graph = builder.compile()

        # Automatic tracing with both OpenInference and UiPath attributes!
        result = await graph.ainvoke({"a": 5, "b": 3})
"""

from typing import Any

from uipath.core.telemetry.context import (
    clear_session_context,
    set_session_context,
)

from ._instrumentor import UiPathLangChainInstrumentor

__all__ = [
    "instrument_langchain",
    "uninstrument_langchain",
    "set_session_context",
    "clear_session_context",
]

# Global instrumentor instance
_instrumentor: UiPathLangChainInstrumentor | None = None


def instrument_langchain(**kwargs: Any) -> None:
    """Instrument LangChain/LangGraph with OpenInference + UiPath features.

    Automatically traces LangChain and LangGraph operations with OpenInference
    attributes (llm.*, tool.*) plus UiPath attributes (session.id, thread.id).

    Args:
        **kwargs: Arguments passed to OpenInference LangChainInstrumentor
            (e.g., tracer_provider, skip_dep_check).

    Raises:
        ImportError: If openinference-instrumentation-langchain not installed.
        RuntimeError: If already instrumented.
    """
    global _instrumentor

    if _instrumentor is not None:
        raise RuntimeError(
            "LangChain/LangGraph already instrumented. "
            "Call uninstrument_langchain() first to re-instrument."
        )

    try:
        _instrumentor = UiPathLangChainInstrumentor()
        _instrumentor.instrument(**kwargs)
    except ImportError as e:
        raise ImportError(
            "openinference-instrumentation-langchain is not installed. "
            "Install with: pip install openinference-instrumentation-langchain>=0.1.55"
        ) from e


def uninstrument_langchain() -> None:
    """Remove LangChain/LangGraph instrumentation.

    Note: UiPathSpanProcessor remains attached (can't be removed from TracerProvider).

    Raises:
        RuntimeError: If not currently instrumented.
    """
    global _instrumentor

    if _instrumentor is None:
        raise RuntimeError(
            "LangChain/LangGraph not instrumented. "
            "Call instrument_langchain() first."
        )

    _instrumentor.uninstrument()
    _instrumentor = None
