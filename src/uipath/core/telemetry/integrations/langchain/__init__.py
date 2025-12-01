"""UiPath LangChain instrumentation using OpenInference.

This module provides a thin wrapper around openinference-instrumentation-langchain
that automatically integrates with UiPath's telemetry client.
"""

from __future__ import annotations

import logging
from typing import Any

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api

from uipath.core.telemetry.client import get_client

__all__ = [
    "UiPathLangChainInstrumentor",
    "instrument_langchain",
    "uninstrument_langchain",
]

logger = logging.getLogger(__name__)

# Singleton instrumentor instance
_instrumentor_instance: UiPathLangChainInstrumentor | None = None


class UiPathLangChainInstrumentor(LangChainInstrumentor):
    """UiPath-enhanced LangChain instrumentor.

    Wraps OpenInference LangChainInstrumentor with automatic UiPath
    TracerProvider injection and resource attributes.
    """

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument LangChain with UiPath TracerProvider.

        Args:
            **kwargs: Additional arguments passed to parent instrumentor.
        """
        # Get UiPath telemetry client's TracerProvider
        try:
            client = get_client()
            tracer_provider = client.get_tracer_provider()
        except RuntimeError:
            # Fallback to global provider if no UiPath client configured
            logger.warning(
                "UiPath telemetry client not available, falling back to global TracerProvider. "
                "Resource attributes (org_id, tenant_id, user_id) will not be available."
            )
            from opentelemetry.sdk.trace import TracerProvider

            global_provider = trace_api.get_tracer_provider()
            # Check if it's an SDK TracerProvider
            if isinstance(global_provider, TracerProvider):
                tracer_provider = global_provider
            else:
                # Use the proxy provider as-is (type: ignore for abstract type)
                tracer_provider = global_provider  # type: ignore[assignment]

        # Instrument with UiPath TracerProvider
        super()._instrument(
            tracer_provider=tracer_provider,
            **kwargs,
        )


def instrument_langchain(**kwargs: Any) -> UiPathLangChainInstrumentor:
    """Instrument LangChain with UiPath telemetry.

    Args:
        **kwargs: Additional arguments passed to instrumentor.

    Returns:
        Instrumentor instance for manual control.

    Example:
        >>> from uipath.core.telemetry import get_telemetry_client
        >>> from uipath.core.telemetry.integrations.langchain import instrument_langchain
        >>>
        >>> client = get_telemetry_client()
        >>> instrument_langchain()
        >>>
        >>> # Now all LangChain operations are traced
        >>> chain.invoke({"input": "test"})
    """
    global _instrumentor_instance

    if _instrumentor_instance is None:
        _instrumentor_instance = UiPathLangChainInstrumentor()
        _instrumentor_instance.instrument(**kwargs)

    return _instrumentor_instance


def uninstrument_langchain() -> None:
    """Remove LangChain instrumentation.

    Example:
        >>> from uipath.core.telemetry.integrations.langchain import uninstrument_langchain
        >>> uninstrument_langchain()
    """
    global _instrumentor_instance

    if _instrumentor_instance is not None:
        _instrumentor_instance.uninstrument()
        _instrumentor_instance = None
