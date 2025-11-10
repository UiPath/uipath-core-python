"""@traced decorator for automatic function instrumentation.

Provides OpenTelemetry-based tracing with support for input/output processors,
privacy controls, and non-recording spans.

Example:
    >>> from uipath.core.telemetry import traced
    >>>
    >>> # Basic usage with privacy controls
    >>> @traced(span_type="automation", hide_input=True)
    >>> def process_invoice(invoice_data):
    ...     return {"status": "processed"}
    >>>
    >>> # With custom processors for PII scrubbing
    >>> def scrub_pii(data):
    ...     return {k: v for k, v in data.items() if k not in ['ssn', 'password']}
    >>>
    >>> @traced(input_processor=scrub_pii)
    >>> def handle_sensitive_data(user_data):
    ...     return {"processed": True}
    >>>
    >>> # Async functions and generators supported
    >>> @traced(span_type="generation")
    >>> async def call_llm(prompt):
    ...     return await llm.generate(prompt)
"""

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, cast

from .client import get_telemetry_client

if TYPE_CHECKING:
    from .observation import ObservationSpan

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_MAX_OUTPUT_ITEMS = 10000  # Memory limit for generator output buffering


def _default_input_processor(inputs: Any) -> dict[str, str]:
    """Redact all input data for privacy.

    Args:
        inputs: Input data to redact

    Returns:
        Redacted placeholder dictionary
    """
    return {"redacted": "Input data not logged for privacy/security"}


def _default_output_processor(outputs: Any) -> dict[str, str]:
    """Redact all output data for privacy.

    Args:
        outputs: Output data to redact

    Returns:
        Redacted placeholder dictionary
    """
    return {"redacted": "Output data not logged for privacy/security"}


def _process_function_input(
    span: "ObservationSpan",
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    processor: Optional[Callable[[dict[str, Any]], Any]],
    logger: logging.Logger,
) -> None:
    """Process and record function input (common logic for all wrappers).

    Applies input processor if provided, with graceful error handling.
    Captures input via signature inspection and records to span.

    Args:
        span: ObservationSpan to record input to
        func: Function being traced
        args: Positional arguments
        kwargs: Keyword arguments
        processor: Optional input processor function
        logger: Logger for error reporting
    """
    input_data = _capture_function_input(func, args, kwargs)
    if processor:
        try:
            processed_input = processor(input_data)
            span.update_input(processed_input)
        except Exception as e:
            logger.error(f"Input processor failed: {e}")
            span.set_attribute("input_processor_error", str(e))
    else:
        span.update_input(input_data)


def _process_function_output(
    span: "ObservationSpan",
    output_data: Any,
    processor: Optional[Callable[[Any], Any]],
    logger: logging.Logger,
) -> None:
    """Process and record function output (common logic for all wrappers).

    Applies output processor if provided, with graceful error handling.

    Args:
        span: ObservationSpan to record output to
        output_data: Function return value
        processor: Optional output processor function
        logger: Logger for error reporting
    """
    if processor:
        try:
            processed_output = processor(output_data)
            span.update_output(processed_output)
        except Exception as e:
            logger.error(f"Output processor failed: {e}")
            span.set_attribute("output_processor_error", str(e))
    else:
        span.update_output(output_data)


def _process_generator_output(
    span: "ObservationSpan",
    outputs: list[Any],
    item_count: int,
    processor: Optional[Callable[[list[Any]], Any]],
    logger: logging.Logger,
) -> None:
    """Process and record generator output (common logic for generator wrappers).

    Applies output processor to collected sample, or records metadata.

    Args:
        span: ObservationSpan to record output to
        outputs: Collected output items (up to MAX_OUTPUT_ITEMS)
        item_count: Total number of items yielded
        processor: Optional output processor function
        logger: Logger for error reporting
    """
    if processor:
        try:
            processed_output = processor(outputs)
            span.update_output(processed_output)
        except Exception as e:
            logger.error(f"Output processor failed: {e}")
            span.set_attribute("output_processor_error", str(e))
    else:
        span.update_output(
            {
                "items_yielded": item_count,
                "sample_items": outputs[:10],
                "truncated": item_count > _MAX_OUTPUT_ITEMS,
            }
        )


def _infer_span_type(func: Callable[..., Any]) -> str:
    """Infer span type from function type.

    Determines appropriate semantic span type based on function signature.

    Args:
        func: Function to infer type from

    Returns:
        Span type string (function_call_sync/async/generator_sync/async)
    """
    if inspect.iscoroutinefunction(func):
        return "function_call_async"
    elif inspect.isasyncgenfunction(func):
        return "function_call_generator_async"
    elif inspect.isgeneratorfunction(func):
        return "function_call_generator_sync"
    else:
        return "function_call_sync"


def traced(
    name: Optional[str] = None,
    run_type: Optional[str] = None,
    span_type: Optional[str] = None,
    input_processor: Optional[Callable[..., Any]] = None,
    output_processor: Optional[Callable[..., Any]] = None,
    hide_input: bool = False,
    hide_output: bool = False,
    recording: bool = True,
) -> Callable[[F], F]:
    """Decorator for automatic function instrumentation.

    Creates a span for the decorated function with support for input/output
    processors, privacy controls, and non-recording spans.

    Args:
        name: Span name (defaults to function name)
        run_type: Optional run type categorization (e.g., "llm", "tool", "chain")
        span_type: Semantic span type (span, generation, automation, etc.).
                  If None, defaults based on function type:
                  - Sync: "function_call_sync"
                  - Async: "function_call_async"
                  - Sync Generator: "function_call_generator_sync"
                  - Async Generator: "function_call_generator_async"
        input_processor: Optional function to process inputs before recording.
                        Signature: (input_dict) -> Any
        output_processor: Optional function to process outputs before recording.
                         Signature: (output_value) -> Any
        hide_input: Hide input from telemetry (privacy). Overrides input_processor.
        hide_output: Hide output from telemetry (privacy). Overrides output_processor.
        recording: If False, span not recorded but context propagates (default True)

    Returns:
        Decorated function

    Example:
        >>> # Privacy controls
        >>> @traced(span_type="automation", hide_input=True)
        >>> def process_invoice(invoice_data):
        ...     return {"status": "processed"}
        >>>
        >>> # Custom PII scrubbing processor
        >>> def scrub_pii(data):
        ...     return {k: v for k, v in data.items() if k not in ['ssn', 'password']}
        >>>
        >>> @traced(input_processor=scrub_pii)
        >>> def handle_user_data(user_info):
        ...     return {"processed": True}
        >>>
        >>> # Non-recording span (maintains hierarchy without recording)
        >>> @traced(recording=False)
        >>> def internal_helper():
        ...     return "result"

    Note:
        - Hide flags override processors (security-first precedence)
        - Processors are trusted code only (no sandboxing)
        - Processor errors are logged and gracefully degraded
        - Non-recording spans maintain trace hierarchy via valid SpanContext
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        effective_span_type = span_type or _infer_span_type(func)

        final_input_processor = input_processor
        final_output_processor = output_processor

        if hide_input:
            final_input_processor = _default_input_processor

        if hide_output:
            final_output_processor = _default_output_processor

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            client = get_telemetry_client()

            with client.start_as_current_span(
                span_name,
                semantic_type=effective_span_type,
                recording=recording,
            ) as span:
                if run_type is not None:
                    span.set_attribute("run_type", run_type)

                _process_function_input(
                    span, func, args, kwargs, final_input_processor, logger
                )

                try:
                    result = func(*args, **kwargs)

                    _process_function_output(
                        span, result, final_output_processor, logger
                    )

                    return result
                except Exception:
                    raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            client = get_telemetry_client()

            with client.start_as_current_span(
                span_name,
                semantic_type=effective_span_type,
                recording=recording,
            ) as span:
                if run_type is not None:
                    span.set_attribute("run_type", run_type)

                _process_function_input(
                    span, func, args, kwargs, final_input_processor, logger
                )

                try:
                    result = await func(*args, **kwargs)

                    _process_function_output(
                        span, result, final_output_processor, logger
                    )

                    return result
                except Exception:
                    raise

        @functools.wraps(func)
        def generator_wrapper(*args, **kwargs):
            client = get_telemetry_client()

            with client.start_as_current_span(
                span_name,
                semantic_type=effective_span_type,
                recording=recording,
            ) as span:
                if run_type is not None:
                    span.set_attribute("run_type", run_type)

                _process_function_input(
                    span, func, args, kwargs, final_input_processor, logger
                )

                outputs = []
                item_count = 0

                try:
                    for item in func(*args, **kwargs):
                        yield item
                        item_count += 1

                        if item_count <= _MAX_OUTPUT_ITEMS:
                            outputs.append(item)

                    _process_generator_output(
                        span, outputs, item_count, final_output_processor, logger
                    )
                except Exception:
                    raise

        @functools.wraps(func)
        async def async_generator_wrapper(*args, **kwargs):
            client = get_telemetry_client()

            with client.start_as_current_span(
                span_name,
                semantic_type=effective_span_type,
                recording=recording,
            ) as span:
                if run_type is not None:
                    span.set_attribute("run_type", run_type)

                _process_function_input(
                    span, func, args, kwargs, final_input_processor, logger
                )

                outputs = []
                item_count = 0

                try:
                    async for item in func(*args, **kwargs):
                        yield item
                        item_count += 1

                        if item_count <= _MAX_OUTPUT_ITEMS:
                            outputs.append(item)

                    _process_generator_output(
                        span, outputs, item_count, final_output_processor, logger
                    )
                except Exception:
                    raise

        if inspect.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        elif inspect.isgeneratorfunction(func):
            return cast(F, generator_wrapper)
        elif inspect.isasyncgenfunction(func):
            return cast(F, async_generator_wrapper)
        else:
            return cast(F, sync_wrapper)

    return decorator


def _capture_function_input(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Capture function input as dictionary.

    Maps positional and keyword arguments to parameter names using
    function signature inspection.

    Args:
        func: Function being called
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Dictionary mapping parameter names to values

    Example:
        >>> def my_func(a, b, c=3):
        ...     pass
        >>> _capture_function_input(my_func, (1, 2), {})
        {'a': 1, 'b': 2, 'c': 3}
    """
    try:
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)
    except Exception:
        return {
            "args": args,
            "kwargs": kwargs,
        }
