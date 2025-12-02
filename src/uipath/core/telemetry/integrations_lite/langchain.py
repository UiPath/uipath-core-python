"""Lightweight LangChain integration for telemetry.

This module provides minimal instrumentation for LangChain by patching the
LangSmith @traceable decorator to redirect to UiPath's @traced decorator.
This follows the KISS/YAGNI principles - basic tracing without rich metadata.
"""

from typing import Any, Callable, Optional, TypeVar, Union

from ._traced_adapter import adapt_to_traced

F = TypeVar("F", bound=Callable[..., Any])

_original_traceable: Optional[Callable[..., Any]] = None
_is_instrumented: bool = False


RUN_TYPE_TO_KIND_MAP = {
    "tool": "TOOL",
    "chain": "CHAIN",
    "llm": "LLM",
    "retriever": "RETRIEVER",
    "embedding": "EMBEDDING",
    "prompt": "CHAIN",
    "parser": "CHAIN",
}


def traceable_adapter(
    func: Optional[F] = None,
    *,
    run_type: Optional[str] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> Union[F, Callable[[F], F]]:
    """Adapter for LangSmith @traceable decorator.

    This function mimics the LangSmith @traceable decorator API but delegates
    to UiPath's @traced decorator. It maps LangSmith's run_type to OpenInference
    span kinds and ignores other LangSmith-specific parameters.

    Args:
        func: The function to decorate (when used without parentheses).
        run_type: LangSmith run type (tool, chain, llm, retriever, etc.).
            Maps to OpenInference span kind.
        name: Optional span name override.
        **kwargs: Additional LangSmith parameters (ignored in lite version).

    Returns:
        Decorated function if func is provided, otherwise returns decorator.

    Examples:
        >>> @traceable_adapter
        ... def my_chain():
        ...     pass

        >>> @traceable_adapter(run_type="tool", name="multiply")
        ... def multiply(a: int, b: int) -> int:
        ...     return a * b

    Note:
        The following LangSmith @traceable parameters are intentionally ignored
        in the lite implementation (YAGNI principle):
        - tags: Not mapped (use custom span attributes if needed)
        - metadata: Not mapped (use custom span attributes if needed)
        - client: Ignored (uses UiPath telemetry client)
        - reduce_fn: Ignored (not needed for basic tracing)
        - project_name: Ignored (use UiPath session context)
    """
    kind = RUN_TYPE_TO_KIND_MAP.get(run_type) if run_type else None

    return adapt_to_traced(
        func,
        name=name,
        kind=kind,
    )


def instrument_langchain() -> None:
    """Instrument LangChain by patching @traceable decorator.

    This function monkey-patches the LangSmith @traceable decorator to redirect
    to UiPath's telemetry system. This enables automatic tracing of all functions
    decorated with @traceable without code changes.

    Raises:
        ImportError: If langsmith package is not installed.
        RuntimeError: If already instrumented.

    Example:
        >>> from uipath.core.telemetry import init
        >>> from uipath.core.telemetry.integrations_lite import instrument_langchain
        >>>
        >>> init(enable_console_export=True)
        >>> instrument_langchain()
        >>>
        >>> # Now all @traceable decorators will use UiPath telemetry
        >>> from langsmith import traceable
        >>>
        >>> @traceable(run_type="tool")
        ... def my_tool():
        ...     return "result"

    Note:
        Call this before importing any LangChain code that uses @traceable.
        Calling it after @traceable has already been applied to functions
        will not affect those functions.
    """
    global _original_traceable, _is_instrumented

    if _is_instrumented:
        raise RuntimeError(
            "LangChain is already instrumented. "
            "Call uninstrument_langchain() first to re-instrument."
        )

    try:
        import langsmith
    except ImportError as e:
        raise ImportError(
            "langsmith package not found. Install it with: pip install langsmith"
        ) from e

    _original_traceable = langsmith.traceable
    langsmith.traceable = traceable_adapter  # type: ignore[assignment]
    _is_instrumented = True


def uninstrument_langchain() -> None:
    """Remove LangChain instrumentation and restore original @traceable.

    This function restores the original LangSmith @traceable decorator,
    disabling UiPath telemetry integration.

    Raises:
        RuntimeError: If not currently instrumented.

    Example:
        >>> from uipath.core.telemetry.integrations_lite import (
        ...     instrument_langchain,
        ...     uninstrument_langchain
        ... )
        >>>
        >>> instrument_langchain()
        >>> # ... use instrumented LangChain ...
        >>> uninstrument_langchain()  # Restore original behavior

    Note:
        This does not affect functions that were already decorated with
        @traceable while instrumented. Those functions will continue to
        use UiPath telemetry until the process is restarted or they are
        re-decorated.
    """
    global _original_traceable, _is_instrumented

    if not _is_instrumented:
        raise RuntimeError(
            "LangChain is not instrumented. Call instrument_langchain() first."
        )

    try:
        import langsmith
    except ImportError:
        pass
    else:
        if _original_traceable is not None:
            langsmith.traceable = _original_traceable

    _original_traceable = None
    _is_instrumented = False


def is_instrumented() -> bool:
    """Check if LangChain is currently instrumented.

    Returns:
        True if LangChain @traceable is patched, False otherwise.

    Example:
        >>> from uipath.core.telemetry.integrations_lite import (
        ...     instrument_langchain,
        ...     is_instrumented
        ... )
        >>>
        >>> is_instrumented()
        False
        >>> instrument_langchain()
        >>> is_instrumented()
        True
    """
    return _is_instrumented
