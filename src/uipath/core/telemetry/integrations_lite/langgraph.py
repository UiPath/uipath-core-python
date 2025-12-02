"""Lightweight LangGraph integration for telemetry.

This module provides minimal instrumentation for LangGraph by monkey-patching
the StateGraph.compile() method to add telemetry to invoke/ainvoke execution.
This follows the KISS/YAGNI principles - basic tracing without rich metadata.
"""

import functools
from typing import Any, Callable, Optional

from opentelemetry import trace as trace_api

from uipath.core.telemetry.attributes import Attr
from uipath.core.telemetry.integrations_full._shared import (
    get_session_id,
    get_thread_id,
    safe_json_dumps,
)
from uipath.core.telemetry.observation import ObservationSpan

_original_compile: Optional[Callable[..., Any]] = None
_is_instrumented: bool = False


def instrument_langgraph() -> None:
    """Instrument LangGraph by patching StateGraph.compile().

    This function monkey-patches the LangGraph StateGraph.compile() method
    to automatically wrap the compiled graph's invoke() and ainvoke() methods
    with telemetry spans. This enables automatic tracing of graph execution
    without code changes.

    Raises:
        ImportError: If langgraph package is not installed.
        RuntimeError: If already instrumented.

    Example:
        >>> from uipath.core.telemetry import init
        >>> from uipath.core.telemetry.integrations_lite import instrument_langgraph
        >>> from uipath.core.telemetry.integrations_full._shared import set_session_context
        >>>
        >>> init(enable_console_export=True)
        >>> instrument_langgraph()
        >>> set_session_context(session_id="my-session", thread_id="thread-1")
        >>>
        >>> # Now all LangGraph execution will be traced
        >>> from langgraph.graph import StateGraph
        >>> builder = StateGraph(dict)
        >>> # ... build graph ...
        >>> graph = builder.compile()
        >>> result = graph.invoke({"input": "test"})  # Automatically traced

    Note:
        Call this before compiling any LangGraph graphs. Graphs compiled
        before instrumentation will not be traced.

        The lite version only traces the top-level invoke/ainvoke calls.
        It does NOT instrument:
        - Individual node executions
        - Tool calls within nodes
        - LLM calls within nodes
        - Stream methods (stream, astream)
        - Batch methods (batch, abatch)

        Use the full LangGraph integration if you need detailed tracing.
    """
    global _original_compile, _is_instrumented

    if _is_instrumented:
        raise RuntimeError(
            "LangGraph is already instrumented. "
            "Call uninstrument_langgraph() first to re-instrument."
        )

    try:
        from langgraph.graph.state import StateGraph
    except ImportError as e:
        raise ImportError(
            "langgraph package not found. Install it with: pip install langgraph"
        ) from e

    _original_compile = StateGraph.compile

    @functools.wraps(_original_compile)
    def traced_compile(self: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrap compile() to add telemetry to invoke/ainvoke."""
        compiled = _original_compile(self, *args, **kwargs)

        tracer = trace_api.get_tracer("uipath.telemetry.lite.langgraph")

        original_invoke = compiled.invoke
        original_ainvoke = compiled.ainvoke

        @functools.wraps(original_invoke)
        def traced_invoke(*args: Any, **kwargs: Any) -> Any:
            """Traced version of invoke()."""
            with tracer.start_as_current_span("LangGraph") as span:
                obs = ObservationSpan(span, kind="CHAIN")

                if session_id := get_session_id():
                    obs.set_attribute(Attr.Common.SESSION_ID, session_id)
                if thread_id := get_thread_id():
                    obs.set_attribute("thread_id", thread_id)

                input_obj = args[0] if args else kwargs.get("input")
                if input_obj is not None:
                    input_value = safe_json_dumps(input_obj)
                    obs.set_attribute(Attr.Common.INPUT_VALUE, input_value)
                    obs.set_attribute(Attr.Common.INPUT_MIME_TYPE, "application/json")

                try:
                    result = original_invoke(*args, **kwargs)

                    output_value = safe_json_dumps(result)
                    obs.set_attribute(Attr.Common.OUTPUT_VALUE, output_value)
                    obs.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")
                    obs.set_status_ok()

                    return result
                except Exception as e:
                    obs.record_exception(e)
                    raise

        @functools.wraps(original_ainvoke)
        async def traced_ainvoke(*args: Any, **kwargs: Any) -> Any:
            """Traced version of ainvoke()."""
            with tracer.start_as_current_span("LangGraph") as span:
                obs = ObservationSpan(span, kind="CHAIN")

                if session_id := get_session_id():
                    obs.set_attribute(Attr.Common.SESSION_ID, session_id)
                if thread_id := get_thread_id():
                    obs.set_attribute("thread_id", thread_id)

                input_obj = args[0] if args else kwargs.get("input")
                if input_obj is not None:
                    input_value = safe_json_dumps(input_obj)
                    obs.set_attribute(Attr.Common.INPUT_VALUE, input_value)
                    obs.set_attribute(Attr.Common.INPUT_MIME_TYPE, "application/json")

                try:
                    result = await original_ainvoke(*args, **kwargs)

                    output_value = safe_json_dumps(result)
                    obs.set_attribute(Attr.Common.OUTPUT_VALUE, output_value)
                    obs.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")
                    obs.set_status_ok()

                    return result
                except Exception as e:
                    obs.record_exception(e)
                    raise

        compiled.invoke = traced_invoke  # type: ignore[method-assign]
        compiled.ainvoke = traced_ainvoke  # type: ignore[method-assign]

        return compiled

    StateGraph.compile = traced_compile  # type: ignore[method-assign]
    _is_instrumented = True


def uninstrument_langgraph() -> None:
    """Remove LangGraph instrumentation and restore original compile().

    This function restores the original StateGraph.compile() method,
    disabling UiPath telemetry integration.

    Raises:
        RuntimeError: If not currently instrumented.

    Example:
        >>> from uipath.core.telemetry.integrations_lite import (
        ...     instrument_langgraph,
        ...     uninstrument_langgraph
        ... )
        >>>
        >>> instrument_langgraph()
        >>> # ... use instrumented LangGraph ...
        >>> uninstrument_langgraph()  # Restore original behavior

    Note:
        This does not affect graphs that were already compiled while
        instrumented. Those graphs will continue to be traced until
        they are garbage collected.
    """
    global _original_compile, _is_instrumented

    if not _is_instrumented:
        raise RuntimeError(
            "LangGraph is not instrumented. Call instrument_langgraph() first."
        )

    try:
        from langgraph.graph.state import StateGraph
    except ImportError:
        pass
    else:
        if _original_compile is not None:
            StateGraph.compile = _original_compile  # type: ignore[method-assign]

    _original_compile = None
    _is_instrumented = False


def is_instrumented() -> bool:
    """Check if LangGraph is currently instrumented.

    Returns:
        True if LangGraph StateGraph.compile is patched, False otherwise.

    Example:
        >>> from uipath.core.telemetry.integrations_lite import (
        ...     instrument_langgraph,
        ...     is_instrumented
        ... )
        >>>
        >>> is_instrumented()
        False
        >>> instrument_langgraph()
        >>> is_instrumented()
        True
    """
    return _is_instrumented
