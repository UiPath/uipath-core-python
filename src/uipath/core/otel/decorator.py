"""Decorators for instrumenting functions with OpenTelemetry spans.

This module provides decorator functions for creating observation spans
with automatic provider response parsing (auto_update).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from functools import wraps
from typing import Any, Callable, TypeVar

from opentelemetry.trace import StatusCode

from .trace import require_trace

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def _create_observation_decorator(
    observation_factory: Callable[..., Any],
    auto_update_default: bool = False,
) -> Callable[..., Callable[[F], F]]:
    """Factory for creating observation decorators with reduced duplication.

    Args:
        observation_factory: Factory function that creates an observation from trace
        auto_update_default: Default value for auto_update parameter

    Returns:
        Decorator factory function
    """

    def decorator_factory(
        name: str | None = None,
        auto_update: bool | None = None,
        **factory_kwargs: Any,
    ) -> Callable[[F], F]:
        """Create decorator with specified parameters.

        Args:
            name: Span name (defaults to function name)
            auto_update: If True, automatically calls obs.update(return_value)
            **factory_kwargs: Additional kwargs for observation factory

        Returns:
            Decorated function
        """
        # Use default if not specified
        if auto_update is None:
            auto_update = auto_update_default

        def decorator(func: F) -> F:
            span_name = name or func.__name__

            # Handle async functions
            if asyncio.iscoroutinefunction(func):

                @wraps(func)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    trace = require_trace()
                    obs = observation_factory(trace, span_name, **factory_kwargs)

                    with obs:
                        try:
                            result = await func(*args, **kwargs)

                            # Auto-update if enabled and result is not None
                            if auto_update and result is not None:
                                _auto_update_with_guards(obs, result)

                            return result
                        except Exception as e:
                            obs.set_status(StatusCode.ERROR, str(e))
                            obs.record_exception(e)
                            raise

                return async_wrapper  # type: ignore[return-value]

            # Handle sync functions
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                trace = require_trace()
                obs = observation_factory(trace, span_name, **factory_kwargs)

                with obs:
                    try:
                        result = func(*args, **kwargs)

                        # Auto-update if enabled and result is not None
                        if auto_update and result is not None:
                            _auto_update_with_guards(obs, result)

                        return result
                    except Exception as e:
                        obs.set_status(StatusCode.ERROR, str(e))
                        obs.record_exception(e)
                        raise

            return sync_wrapper  # type: ignore[return-value]

        return decorator

    return decorator_factory


# P2 FIX: Refactored decorators using factory pattern to eliminate duplication
generation = _create_observation_decorator(
    observation_factory=lambda trace, name, **kw: trace.generation(
        name, model=kw.get("model")
    ),
    auto_update_default=True,  # Smart parsing for generation
)

tool = _create_observation_decorator(
    observation_factory=lambda trace, name, **kw: trace.tool(name),
    auto_update_default=False,  # No auto-update for generic tools
)

agent = _create_observation_decorator(
    observation_factory=lambda trace, name, **kw: trace.agent(name),
    auto_update_default=False,  # No auto-update for generic agents
)


def traced(
    span_type: str = "generation",
    auto_update: bool = False,
    **kwargs: Any,
) -> Callable[[F], F]:
    """Generic decorator for any span type (backwards compatible with V5).

    Args:
        span_type: Semantic span type
        auto_update: If True, automatically calls obs.update(return_value)
        **kwargs: Additional span attributes

    Returns:
        Decorated function

    Example:
        @otel.traced(span_type="retrieval")
        def search_documents(query: str):
            results = vector_store.search(query)
            return results
    """

    def decorator(func: F) -> F:
        span_name = kwargs.pop("name", func.__name__)

        # Route to specialized decorators if possible
        if span_type == "generation":
            return generation(
                name=span_name,
                model=kwargs.get("model"),
                auto_update=auto_update,
            )(func)
        elif span_type == "tool":
            return tool(name=span_name, auto_update=auto_update)(func)
        elif span_type == "agent":
            return agent(name=span_name, auto_update=auto_update)(func)

        # Generic implementation for other types
        # Handle async functions
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs_inner: Any) -> Any:
                trace = require_trace()
                # Use workflow for generic spans
                with trace.workflow(span_name) as obs:
                    obs.set_attribute("span.type", span_type)
                    for key, value in kwargs.items():
                        obs.set_attribute(key, value)

                    try:
                        result = await func(*args, **kwargs_inner)

                        if auto_update and result is not None:
                            _auto_update_with_guards(obs, result)

                        return result
                    except Exception as e:
                        obs.set_status(StatusCode.ERROR, str(e))
                        obs.record_exception(e)
                        raise

            return async_wrapper  # type: ignore[return-value]

        # Handle sync functions
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs_inner: Any) -> Any:
            trace = require_trace()
            # Use workflow for generic spans
            with trace.workflow(span_name) as obs:
                obs.set_attribute("span.type", span_type)
                for key, value in kwargs.items():
                    obs.set_attribute(key, value)

                try:
                    result = func(*args, **kwargs_inner)

                    if auto_update and result is not None:
                        _auto_update_with_guards(obs, result)

                    return result
                except Exception as e:
                    obs.set_status(StatusCode.ERROR, str(e))
                    obs.record_exception(e)
                    raise

        return sync_wrapper  # type: ignore[return-value]

    return decorator


def _auto_update_with_guards(obs: Any, result: Any) -> None:
    """Auto-update observation with safety guards.

    Args:
        obs: Observation instance
        result: Result value to update with
    """
    # Guard 1: Only if result is not None (already checked)
    # Guard 2: Check if result is a generator (not supported)
    if inspect.isgenerator(result) or inspect.isasyncgen(result):
        logger.warning(
            "Decorator auto_update does not support generators. "
            "Use context managers for streaming."
        )
        return

    # Guard 3: Exception-safe update
    try:
        obs.update(result)
    except Exception as e:
        logger.warning(
            "Auto-update failed for type %s: %s",
            type(result).__name__,
            e,
            exc_info=True,
        )
        # Don't fail user function - just skip update
