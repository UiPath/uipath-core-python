"""Unified decorator for function tracing."""

from __future__ import annotations

import functools
import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from opentelemetry.trace import SpanKind

from .observation import ObservationSpan

if TYPE_CHECKING:
    from .attributes import OpenInferenceSpanKindValues

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def traced(
    name: str | None = None,
    kind: OpenInferenceSpanKindValues | str | None = None,
    hide_input: bool = False,
    hide_output: bool = False,
    input_processor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    output_processor: Callable[[Any], Any] | None = None,
    recording: bool = True,
) -> Callable[[F], F]:
    """Unified decorator for function tracing with kind-specific behavior.

    Supports sync/async/generator functions with privacy controls and
    custom processors. Metadata is extracted via parsers using update().

    Args:
        name: Span name (defaults to function name)
        kind: Span kind (generation, tool, agent, retriever, etc.)
        hide_input: If True, redact all inputs
        hide_output: If True, redact all outputs
        input_processor: Custom input processor (called before recording)
        output_processor: Custom output processor (called before recording)
        recording: If False, create non-recording span (for hierarchy only)

    Returns:
        Decorated function

    Example:
        @traced(kind="generation")
        def extract_invoice(prompt: str) -> dict:
            response = openai.chat.completions.create(...)
            return response  # Metadata auto-extracted via parser

        @traced(kind="tool", hide_input=True)
        async def fetch_user_data(user_id: str) -> dict:
            return await db.get_user(user_id)
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        func_sig = inspect.signature(func)

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                from .client import get_client

                client = get_client()
                tracer = client.get_tracer()

                with tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.INTERNAL,
                    record_exception=recording,
                    set_status_on_exception=recording,
                ) as span:
                    if not recording or not span.is_recording():
                        return await func(*args, **kwargs)

                    obs = ObservationSpan(span, kind=kind)

                    try:
                        input_data = _extract_input(
                            func_sig, args, kwargs, input_processor
                        )
                        obs.record_input(input_data, hide=hide_input)

                        result = await func(*args, **kwargs)

                        output_data = _process_output(result, output_processor)
                        obs.record_output(output_data, hide=hide_output)

                        return result

                    except Exception as e:
                        obs.record_exception(e)
                        raise

            return async_wrapper  # type: ignore

        elif inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def generator_wrapper(*args: Any, **kwargs: Any) -> Any:
                from .client import get_client

                client = get_client()
                tracer = client.get_tracer()

                with tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.INTERNAL,
                    record_exception=recording,
                    set_status_on_exception=recording,
                ) as span:
                    if not recording or not span.is_recording():
                        yield from func(*args, **kwargs)
                        return

                    obs = ObservationSpan(span, kind=kind)

                    try:
                        input_data = _extract_input(
                            func_sig, args, kwargs, input_processor
                        )
                        obs.record_input(input_data, hide=hide_input)

                        output_buffer = []
                        MAX_OUTPUT_ITEMS = 10000

                        for item in func(*args, **kwargs):
                            if len(output_buffer) < MAX_OUTPUT_ITEMS:
                                output_buffer.append(item)
                            yield item

                        if output_buffer:
                            output_data = _process_output(
                                output_buffer, output_processor
                            )
                            obs.record_output(output_data, hide=hide_output)

                    except Exception as e:
                        obs.record_exception(e)
                        raise

            return generator_wrapper  # type: ignore

        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                from .client import get_client

                client = get_client()
                tracer = client.get_tracer()

                with tracer.start_as_current_span(
                    span_name,
                    kind=SpanKind.INTERNAL,
                    record_exception=recording,
                    set_status_on_exception=recording,
                ) as span:
                    if not recording or not span.is_recording():
                        return func(*args, **kwargs)

                    obs = ObservationSpan(span, kind=kind)

                    try:
                        input_data = _extract_input(
                            func_sig, args, kwargs, input_processor
                        )
                        obs.record_input(input_data, hide=hide_input)

                        result = func(*args, **kwargs)

                        output_data = _process_output(result, output_processor)
                        obs.record_output(output_data, hide=hide_output)

                        return result

                    except Exception as e:
                        obs.record_exception(e)
                        raise

            return sync_wrapper  # type: ignore

    return decorator


def _extract_input(
    sig: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    processor: Callable[[dict[str, Any]], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Extract input arguments with optional custom processing.

    Args:
        sig: Pre-computed function signature (from decoration time)
        args: Positional arguments
        kwargs: Keyword arguments
        processor: Optional custom processor

    Returns:
        Processed input dictionary
    """
    bound_args = sig.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()

    input_data = dict(bound_args.arguments)

    if processor:
        try:
            input_data = processor(input_data)
        except Exception as e:
            logger.warning("Input processor failed: %s", e)

    return input_data


def _process_output(
    value: Any,
    processor: Callable[[Any], Any] | None,
) -> Any:
    """Process output value with optional custom processor.

    Args:
        value: Output value
        processor: Optional custom processor

    Returns:
        Processed output value
    """
    if processor:
        try:
            return processor(value)
        except Exception as e:
            logger.warning("Output processor failed: %s", e)

    return value
