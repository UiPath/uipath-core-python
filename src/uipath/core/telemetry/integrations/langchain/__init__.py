"""LangChain instrumentation for UiPath OpenTelemetry integration.

This module provides automatic instrumentation for LangChain framework using
OpenTelemetry. It captures traces for all LangChain operations including:
- LLM calls (with input messages, output messages, and token counts)
- Chain executions (with inputs and outputs)
- Tool invocations (with parameters and results)
- Retriever queries (with documents and scores)

The instrumentation follows OpenInference semantic conventions for compatibility
with observability platforms like Arize Phoenix.

Examples:
    Basic usage with zero-code instrumentation:

    >>> from uipath.core.telemetry.integrations.langchain import LangChainInstrumentor
    >>> from opentelemetry import trace
    >>> from opentelemetry.sdk.trace import TracerProvider
    >>>
    >>> # Setup OpenTelemetry
    >>> provider = TracerProvider()
    >>> trace.set_tracer_provider(provider)
    >>>
    >>> # Instrument LangChain
    >>> instrumentor = LangChainInstrumentor()
    >>> instrumentor.instrument()
    >>>
    >>> # Use LangChain normally - all operations are automatically traced
    >>> from langchain_openai import ChatOpenAI
    >>> llm = ChatOpenAI()
    >>> result = llm.invoke("Hello!")

    Advanced usage with custom configuration:

    >>> from uipath.core.telemetry.integrations._shared import InstrumentationConfig
    >>>
    >>> config = InstrumentationConfig(
    ...     capture_inputs=True,
    ...     capture_outputs=True,
    ...     max_string_length=2048,
    ...     hide_sensitive_data=True,
    ... )
    >>> instrumentor.instrument(config=config)
"""

from __future__ import annotations

__all__ = [
    "LangChainInstrumentor",
    "UiPathTracer",
]

from ._instrumentor import LangChainInstrumentor
from ._tracer import UiPathTracer
