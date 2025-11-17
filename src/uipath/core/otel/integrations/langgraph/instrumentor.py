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
        return ("langgraph >= 1.0.0",)

    def _instrument(self, **kwargs: Any) -> None:
        """Apply instrumentation by wrapping StateGraph.compile().

        This method monkey-patches LangGraph's compile() method to inject
        a LangGraphTracer callback automatically.

        Args:
            **kwargs: Additional instrumentation options including:
                - tracer_provider: Optional OpenTelemetry TracerProvider (for testing)
        """
        try:
            from langgraph.graph import StateGraph
        except ImportError:
            # LangGraph not installed - silently skip
            return

        # Get tracer provider from kwargs (for testing) or use global client
        from opentelemetry.trace import get_tracer

        tracer_provider = kwargs.get("tracer_provider")

        # Try to get otel_client, but fall back to tracer_provider if client not initialized
        try:
            otel_client = self._get_otel_client()
        except RuntimeError as e:
            # Client not initialized - use tracer_provider directly if provided
            if tracer_provider is None:
                raise RuntimeError(
                    "OTel client not initialized and no tracer_provider provided. "
                    "Either call otel.init() first or pass tracer_provider to instrument()."
                ) from e
            # Create a minimal client wrapper for test mode
            class _TestClientWrapper:
                """Minimal wrapper for test mode when tracer_provider is provided."""

                def __init__(self, provider: Any) -> None:
                    self._provider = provider
                    self._tracer = get_tracer(
                        instrumenting_module_name="uipath.core.otel.integrations.langgraph",
                        instrumenting_library_version="0.1.0",
                        tracer_provider=provider,
                    )

                def get_tracer(self) -> Any:
                    return self._tracer

            otel_client = _TestClientWrapper(tracer_provider)  # type: ignore[assignment]

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
            self: StateGraph,  # type: ignore[type-arg]
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
                from .._shared import get_session_id, get_thread_id
                from ._callbacks import LangGraphTracer

                tracer = otel_client.get_tracer()
                with tracer.start_as_current_span("langgraph.invoke") as span:
                    session_id = get_session_id()
                    thread_id = get_thread_id()
                    if session_id:
                        span.set_attribute("session.id", session_id)
                    if thread_id:
                        span.set_attribute("thread_id", thread_id)

                    original_config = kwargs.get("config") or {}
                    config = (
                        original_config.copy()
                        if isinstance(original_config, dict)
                        else {}
                    )

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

                    # Execute and set status on success
                    result = original_invoke(*args, **kwargs)
                    from opentelemetry.trace import Status, StatusCode

                    span.set_status(Status(StatusCode.OK))
                    return result

            # Replace invoke method
            app.invoke = traced_invoke  # type: ignore[method-assign]

            # HIGH FIX: Also wrap ainvoke for async support
            if hasattr(app, "ainvoke"):
                original_ainvoke = app.ainvoke

                @functools.wraps(original_ainvoke)
                async def traced_ainvoke(*args: Any, **kwargs: Any) -> Any:
                    """Wrapped ainvoke that injects tracer and creates parent span.

                    HIGH FIX: Async support for LangGraph workflows.
                    Uses *args, **kwargs to preserve original signature.
                    """
                    # Start parent span for async graph execution
                    from .._shared import get_session_id, get_thread_id
                    from ._callbacks import LangGraphTracer

                    tracer = otel_client.get_tracer()
                    with tracer.start_as_current_span("langgraph.ainvoke") as span:
                        # Add session context to root span for trace correlation
                        session_id = get_session_id()
                        thread_id = get_thread_id()
                        if session_id:
                            span.set_attribute("session.id", session_id)
                        if thread_id:
                            span.set_attribute("thread_id", thread_id)

                        # CALLBACK PRESERVATION FIX: Handle all callback configurations
                        # Extract config (create copy to avoid mutating user's dict)
                        original_config = kwargs.get("config") or {}
                        config = (
                            original_config.copy()
                            if isinstance(original_config, dict)
                            else {}
                        )

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

                        # Execute and set status on success
                        result = await original_ainvoke(*args, **kwargs)
                        from opentelemetry.trace import Status, StatusCode

                        span.set_status(Status(StatusCode.OK))
                        return result

                app.ainvoke = traced_ainvoke  # type: ignore[method-assign]

            return app

        # Apply monkey-patch
        StateGraph.compile = traced_compile  # type: ignore[assignment,method-assign]

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
