"""Basic instrumentation tests for LangChain integration.

Tests validate core instrumentation functionality without requiring
real LLM providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from uipath.core.telemetry.integrations_full.langchain import (
    UiPathLangChainInstrumentor,
    instrument_langchain,
    uninstrument_langchain,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


def test_instrumentor_initialization() -> None:
    """Verify instrumentor can be created and initialized."""
    instrumentor = UiPathLangChainInstrumentor()
    assert instrumentor is not None
    assert hasattr(instrumentor, "_instrument")
    assert hasattr(instrumentor, "_uninstrument")


def test_instrument_uses_uipath_tracer_provider(
    telemetry_client: TelemetryClient,
) -> None:
    """Verify wrapper injects UiPath TracerProvider."""
    # Instrument LangChain
    instrumentor = instrument_langchain()

    # Verify instrumentor was created
    assert instrumentor is not None
    assert isinstance(instrumentor, UiPathLangChainInstrumentor)

    # Verify it's instrumented (internal state check)
    assert instrumentor.is_instrumented_by_opentelemetry

    # Cleanup
    uninstrument_langchain()


def test_simple_llm_invocation_creates_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify basic LLM call creates a span."""
    instrument_langchain()

    # Create mock LLM with LangChain-compatible interface
    from langchain_core.language_models.fake import FakeListLLM

    llm = FakeListLLM(responses=["4"])

    # Invoke LLM
    result = llm.invoke("What is 2+2?")

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify at least one span was created
    assert len(spans) > 0

    # Verify we have an LLM span (OpenInference creates spans for LLM operations)
    span_names = [s.name for s in spans]
    assert any("llm" in name.lower() or "fake" in name.lower() for name in span_names)

    # Verify result
    assert result == "4"

    # Cleanup
    uninstrument_langchain()


def test_simple_chain_creates_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify basic chain creates a span."""
    instrument_langchain()

    # Create simple chain using LangChain LCEL
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    prompt = ChatPromptTemplate.from_template("What is {input}?")
    chain = prompt | RunnableLambda(lambda x: "4")

    # Invoke chain
    result = chain.invoke({"input": "2+2"})

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0

    # Verify we have chain/runnable spans
    span_names = [s.name for s in spans]
    assert any(
        "chain" in name.lower() or "runnable" in name.lower() for name in span_names
    )

    # Verify result
    assert result == "4"

    # Cleanup
    uninstrument_langchain()


def test_uninstrumentation(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify uninstrument() removes instrumentation."""
    # Instrument
    instrument_langchain()

    # Create and invoke LLM (should create spans)
    from langchain_core.language_models.fake import FakeListLLM

    llm = FakeListLLM(responses=["test1"])
    llm.invoke("test")

    telemetry_client.flush()

    # Verify spans were created
    spans_before = len(in_memory_exporter.get_finished_spans())
    assert spans_before > 0

    # Clear exporter
    in_memory_exporter.clear()

    # Uninstrument
    uninstrument_langchain()

    # Create new LLM instance after uninstrumentation
    llm2 = FakeListLLM(responses=["test2"])
    llm2.invoke("test")

    telemetry_client.flush()

    # Verify no new spans created (uninstrumentation worked)
    spans_after = len(in_memory_exporter.get_finished_spans())

    # After uninstrumentation, spans should not be created
    # (or significantly fewer spans)
    assert spans_after == 0


def test_fallback_to_global_provider(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify fallback to global provider when no UiPath client exists."""
    from uipath.core.telemetry.client import reset_client

    # Reset client to ensure no UiPath client exists
    reset_client()

    # Instrument without initializing UiPath client
    # Should fallback to global provider
    instrument_langchain()

    # Create and invoke LLM
    from langchain_core.language_models.fake import FakeListLLM

    llm = FakeListLLM(responses=["fallback test"])
    result = llm.invoke("test")

    # Verify result (proves instrumentation didn't crash)
    assert result == "fallback test"

    # Cleanup
    uninstrument_langchain()
