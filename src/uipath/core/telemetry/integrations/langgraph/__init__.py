"""UiPath LangGraph instrumentation using OpenInference patterns.

LangGraph doesn't have a dedicated OpenInference instrumentor yet, so we
use the LangChain instrumentor (since LangGraph is built on LangChain)
plus custom callback injection for graph-level spans.
"""

from __future__ import annotations

import functools
import logging
import threading
from typing import Any, Callable

from opentelemetry import trace as trace_api
from opentelemetry.trace import Status, StatusCode, Tracer

from uipath.core.telemetry.client import get_client
from uipath.core.telemetry.integrations.langchain import instrument_langchain

__all__ = [
    "instrument_langgraph",
    "uninstrument_langgraph",
]

logger = logging.getLogger(__name__)

# Thread-safe global state
_lock = threading.Lock()
_original_compile: Callable[..., Any] | None = None
_instrumented = False
_cached_tracer: Tracer | None = None


def instrument_langgraph(**kwargs: Any) -> None:
    """Instrument LangGraph with UiPath telemetry.

    This instruments both:
    1. LangChain operations (via OpenInference LangChain instrumentor)
    2. Graph compilation and execution (via compile() wrapper)

    Args:
        **kwargs: Additional arguments (reserved for future use).

    Example:
        >>> from uipath.core.telemetry.integrations.langgraph import instrument_langgraph
        >>> instrument_langgraph()
        >>>
        >>> # Now all LangGraph workflows are traced
        >>> workflow.invoke({"messages": ["test"]})
    """
    global _original_compile, _instrumented, _cached_tracer

    with _lock:
        if _instrumented:
            return

        # Cache tracer with fallback for hot path
        try:
            _cached_tracer = get_client().get_tracer()
        except RuntimeError:
            logger.warning(
                "UiPath telemetry client not available, falling back to global tracer. "
                "Resource attributes (org_id, tenant_id, user_id) will not be available."
            )
            _cached_tracer = trace_api.get_tracer("uipath.langgraph")

        # Instrument LangChain (covers node operations)
        instrument_langchain(**kwargs)

        # Wrap StateGraph.compile() for graph-level spans
        from langgraph.graph.state import StateGraph

        _original_compile = StateGraph.compile

        @functools.wraps(_original_compile)
        def traced_compile(self: Any, *args: Any, **kwargs: Any) -> Any:
            """Wrap compile() to add graph execution tracing.

            Args:
                self: StateGraph instance.
                *args: Positional arguments passed to compile().
                **kwargs: Keyword arguments passed to compile().

            Returns:
                Compiled workflow with traced invoke/ainvoke methods.
            """
            compiled = _original_compile(self, *args, **kwargs)

            # Wrap invoke/ainvoke with parent spans
            original_invoke = compiled.invoke
            original_ainvoke = compiled.ainvoke

            @functools.wraps(original_invoke)
            def traced_invoke(*args: Any, **kwargs: Any) -> Any:
                """Traced invoke wrapper.

                Args:
                    *args: Positional arguments passed to invoke().
                    **kwargs: Keyword arguments passed to invoke().

                Returns:
                    Result from original invoke().

                Raises:
                    Exception: Re-raises any exception from invoke().
                """
                # Use cached tracer (safe, already initialized with fallback)
                with _cached_tracer.start_as_current_span(  # type: ignore[union-attr]
                    "langgraph.invoke",
                    attributes={"langgraph.graph": self.__class__.__name__},
                ) as span:
                    try:
                        result = original_invoke(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            @functools.wraps(original_ainvoke)
            async def traced_ainvoke(*args: Any, **kwargs: Any) -> Any:
                """Traced ainvoke wrapper.

                Args:
                    *args: Positional arguments passed to ainvoke().
                    **kwargs: Keyword arguments passed to ainvoke().

                Returns:
                    Result from original ainvoke().

                Raises:
                    Exception: Re-raises any exception from ainvoke().
                """
                # Use cached tracer (safe, already initialized with fallback)
                with _cached_tracer.start_as_current_span(  # type: ignore[union-attr]
                    "langgraph.ainvoke",
                    attributes={"langgraph.graph": self.__class__.__name__},
                ) as span:
                    try:
                        result = await original_ainvoke(*args, **kwargs)
                        span.set_status(Status(StatusCode.OK))
                        return result
                    except Exception as e:
                        span.record_exception(e)
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        raise

            compiled.invoke = traced_invoke  # type: ignore[method-assign]
            compiled.ainvoke = traced_ainvoke  # type: ignore[method-assign]
            return compiled

        StateGraph.compile = traced_compile  # type: ignore[method-assign]
        _instrumented = True


def uninstrument_langgraph() -> None:
    """Remove LangGraph instrumentation.

    Note: Already-compiled workflows retain their traced invoke/ainvoke methods.
    To fully uninstrument, discard compiled workflows and recompile after calling
    this function.

    Example:
        >>> from uipath.core.telemetry.integrations.langgraph import uninstrument_langgraph
        >>> uninstrument_langgraph()
    """
    global _original_compile, _instrumented, _cached_tracer

    with _lock:
        if not _instrumented or _original_compile is None:
            return

        from langgraph.graph.state import StateGraph

        StateGraph.compile = _original_compile  # type: ignore[method-assign]

        _instrumented = False
        _cached_tracer = None

    # Uninstrument LangChain since we depend on it
    from uipath.core.telemetry.integrations.langchain import uninstrument_langchain

    uninstrument_langchain()
