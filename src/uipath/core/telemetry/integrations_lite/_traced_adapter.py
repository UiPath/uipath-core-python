"""Core adapter for integrations_lite.

This module provides a lightweight adapter that converts integration-specific
decorator arguments to the standard @traced decorator format. This enables
reuse of the existing telemetry infrastructure without duplication.
"""

from typing import Any, Callable, Optional, TypeVar, Union

from uipath.core.telemetry.decorator import traced

F = TypeVar("F", bound=Callable[..., Any])


def adapt_to_traced(
    func: Optional[F] = None,
    *,
    name: Optional[str] = None,
    kind: Optional[str] = None,
    hide_input: bool = False,
    hide_output: bool = False,
    **kwargs: Any,
) -> Union[F, Callable[[F], F]]:
    """Adapt integration-specific arguments to @traced decorator.

    This function serves as a bridge between integration-specific decorator
    patterns (e.g., LangSmith's @traceable) and the standard @traced decorator.
    It normalizes arguments and delegates to the core telemetry implementation.

    Args:
        func: The function to decorate (when used without parentheses).
        name: Optional span name override. If not provided, uses function name.
        kind: Optional OpenInference span kind (e.g., "CHAIN", "TOOL", "LLM").
        hide_input: If True, prevents input from being recorded in span.
        hide_output: If True, prevents output from being recorded in span.
        **kwargs: Additional arguments (ignored for lite implementation).

    Returns:
        Decorated function if func is provided, otherwise returns decorator.

    Examples:
        >>> @adapt_to_traced
        ... def my_function():
        ...     pass

        >>> @adapt_to_traced(name="custom", kind="TOOL")
        ... def my_tool():
        ...     pass

    Note:
        This is the "lite" implementation following KISS/YAGNI principles.
        Extra arguments from integration-specific decorators are intentionally
        ignored unless they map to core @traced parameters.
    """
    traced_args: dict[str, Any] = {}

    if name is not None:
        traced_args["name"] = name
    if kind is not None:
        traced_args["kind"] = kind
    if hide_input:
        traced_args["hide_input"] = hide_input
    if hide_output:
        traced_args["hide_output"] = hide_output

    def decorator(f: F) -> F:
        return traced(**traced_args)(f)

    if func is None:
        return decorator
    else:
        return decorator(func)
