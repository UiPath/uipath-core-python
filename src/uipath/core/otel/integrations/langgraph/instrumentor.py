"""LangGraph instrumentor for automatic tracing."""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any, Collection

from .._base import UiPathInstrumentor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class LangGraphInstrumentor(UiPathInstrumentor):
    """Automatic instrumentation for LangGraph workflows.

    This instrumentor wraps LangGraph's StateGraph.compile() method to inject
    tracing callbacks automatically, enabling zero-code observability.

    Example:
        from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

        # Install once
        LangGraphInstrumentor().instrument()

        # Normal LangGraph code - automatically traced!
        from langgraph.graph import StateGraph

        graph = StateGraph(...)
        app = graph.compile()
        result = app.invoke({"messages": ["Hello"]})
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """Declare supported LangGraph versions.

        Returns:
            Collection of version specifiers
        """
        return ("langgraph >= 0.1.0",)

    def _instrument(self, **kwargs: Any) -> None:
        """Apply instrumentation by wrapping StateGraph.compile().

        This method monkey-patches LangGraph's compile() method to inject
        a LangGraphTracer callback automatically.

        Args:
            **kwargs: Additional instrumentation options
        """
        try:
            from langgraph.graph import StateGraph
        except ImportError:
            # LangGraph not installed - silently skip
            return

        otel_client = self._get_otel_client()

        # CRITICAL FIX: Guard against re-instrumentation
        # Check if StateGraph.compile has already been wrapped
        if hasattr(StateGraph.compile, "__wrapped_by_uipath__"):
            logger.warning(
                "LangGraph already instrumented by UiPath. "
                "Skipping re-instrumentation to avoid conflicts."
            )
            return

        # Store original compile method
        original_compile = StateGraph.compile

        def traced_compile(
            self: StateGraph,
            *args: Any,
            **compile_kwargs: Any,
        ) -> Any:
            """Wrapped compile that adds tracing callbacks."""
            # Call original compile
            app = original_compile(self, *args, **compile_kwargs)

            # Wrap invoke to inject callbacks
            original_invoke = app.invoke

            @functools.wraps(original_invoke)
            def traced_invoke(*args: Any, **kwargs: Any) -> Any:
                """Wrapped invoke that injects tracer and creates parent span.

                HIGH FIX: Uses *args, **kwargs to preserve original signature.
                CRITICAL FIX: Creates a parent span for the entire graph execution
                to ensure all callback-created spans are properly parented.
                """
                from ._callbacks import LangGraphTracer

                # CRITICAL FIX: Start parent span for entire graph execution
                # This ensures all node spans created by callbacks link properly
                tracer = otel_client.get_tracer()
                with tracer.start_as_current_span("langgraph.invoke"):
                    # CALLBACK PRESERVATION FIX: Handle all callback configurations
                    # Extract config (create copy to avoid mutating user's dict)
                    original_config = kwargs.get("config") or {}
                    config = original_config.copy() if isinstance(original_config, dict) else {}

                    # Collect user callbacks from multiple sources
                    user_callbacks = []

                    # Source 1: config["callbacks"]
                    if "callbacks" in config:
                        cb = config["callbacks"]
                        if isinstance(cb, list):
                            user_callbacks.extend(cb)
                        elif cb is not None:  # Single callback object
                            user_callbacks.append(cb)

                    # Source 2: Direct callbacks kwarg (less common but valid)
                    if "callbacks" in kwargs:
                        cb = kwargs["callbacks"]
                        if isinstance(cb, list):
                            user_callbacks.extend(cb)
                        elif cb is not None:
                            user_callbacks.append(cb)

                    # Create our tracer
                    callback_tracer = LangGraphTracer(
                        otel_client=otel_client,
                        trace_state=True,
                        trace_edges=True,
                    )

                    # Merge: user callbacks + our tracer
                    merged_callbacks = user_callbacks + [callback_tracer]

                    # Update config with merged callbacks
                    config["callbacks"] = merged_callbacks
                    kwargs["config"] = config

                    return original_invoke(*args, **kwargs)

            # Replace invoke method
            app.invoke = traced_invoke

            # HIGH FIX: Also wrap ainvoke for async support
            if hasattr(app, "ainvoke"):
                original_ainvoke = app.ainvoke

                @functools.wraps(original_ainvoke)
                async def traced_ainvoke(*args: Any, **kwargs: Any) -> Any:
                    """Wrapped ainvoke that injects tracer and creates parent span.

                    HIGH FIX: Async support for LangGraph workflows.
                    Uses *args, **kwargs to preserve original signature.
                    """
                    from ._callbacks import LangGraphTracer

                    # Start parent span for async graph execution
                    tracer = otel_client.get_tracer()
                    with tracer.start_as_current_span("langgraph.ainvoke"):
                        # CALLBACK PRESERVATION FIX: Handle all callback configurations
                        # Extract config (create copy to avoid mutating user's dict)
                        original_config = kwargs.get("config") or {}
                        config = original_config.copy() if isinstance(original_config, dict) else {}

                        # Collect user callbacks from multiple sources
                        user_callbacks = []

                        # Source 1: config["callbacks"]
                        if "callbacks" in config:
                            cb = config["callbacks"]
                            if isinstance(cb, list):
                                user_callbacks.extend(cb)
                            elif cb is not None:  # Single callback object
                                user_callbacks.append(cb)

                        # Source 2: Direct callbacks kwarg (less common but valid)
                        if "callbacks" in kwargs:
                            cb = kwargs["callbacks"]
                            if isinstance(cb, list):
                                user_callbacks.extend(cb)
                            elif cb is not None:
                                user_callbacks.append(cb)

                        # Create our tracer
                        callback_tracer = LangGraphTracer(
                            otel_client=otel_client,
                            trace_state=True,
                            trace_edges=True,
                        )

                        # Merge: user callbacks + our tracer
                        merged_callbacks = user_callbacks + [callback_tracer]

                        # Update config with merged callbacks
                        config["callbacks"] = merged_callbacks
                        kwargs["config"] = config

                        return await original_ainvoke(*args, **kwargs)

                app.ainvoke = traced_ainvoke

            return app

        # Apply monkey-patch
        StateGraph.compile = traced_compile  # type: ignore[method-assign]

        # CRITICAL FIX: Mark as instrumented to prevent re-instrumentation
        StateGraph.compile.__wrapped_by_uipath__ = True  # type: ignore[attr-defined]

        # Store original for uninstrumentation
        self._original_compile = original_compile

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation and restore original methods.

        Args:
            **kwargs: Additional uninstrumentation options
        """
        try:
            from langgraph.graph import StateGraph
        except ImportError:
            return

        if hasattr(self, "_original_compile"):
            StateGraph.compile = self._original_compile  # type: ignore[method-assign]

            # CRITICAL FIX: Remove instrumentation marker
            if hasattr(StateGraph.compile, "__wrapped_by_uipath__"):
                delattr(StateGraph.compile, "__wrapped_by_uipath__")
