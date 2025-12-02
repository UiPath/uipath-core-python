"""Integration tests with LangChain/LangGraph."""

import pytest
from langgraph.graph import StateGraph
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.integrations_openinference import (
    clear_session_context,
    instrument_langchain,
    set_session_context,
    uninstrument_langchain,
)


@pytest.fixture
def instrumented_provider():
    """Create instrumented tracer provider."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrument_langchain(tracer_provider=provider)

    yield provider, exporter

    uninstrument_langchain()
    provider.shutdown()
    clear_session_context()


def test_langgraph_invoke_creates_spans(instrumented_provider):
    """Test that LangGraph invoke creates spans with UiPath attributes."""
    provider, exporter = instrumented_provider

    set_session_context(session_id="test-session")

    # Create a simple LangGraph
    def process_node(state: dict) -> dict:
        return {"result": state.get("value", 0) * 2}

    builder = StateGraph(dict)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    graph = builder.compile()

    # Execute
    result = graph.invoke({"value": 5})

    assert result["result"] == 10

    # Check spans
    spans = exporter.get_finished_spans()
    assert len(spans) > 0  # Should have at least one span from OpenInference

    # Check that at least one span has UiPath session attribute
    session_spans = [
        span
        for span in spans
        if span.attributes and span.attributes.get("session.id") == "test-session"
    ]
    assert len(session_spans) > 0, "Expected at least one span with session.id"


async def test_langgraph_ainvoke_creates_spans(instrumented_provider):
    """Test that LangGraph ainvoke creates spans with UiPath attributes."""
    provider, exporter = instrumented_provider

    set_session_context(session_id="async-session", thread_id="async-thread")

    # Create a simple async LangGraph
    async def async_process_node(state: dict) -> dict:
        return {"result": state.get("value", 0) * 3}

    builder = StateGraph(dict)
    builder.add_node("process", async_process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    graph = builder.compile()

    # Execute
    result = await graph.ainvoke({"value": 7})

    assert result["result"] == 21

    # Check spans
    spans = exporter.get_finished_spans()
    assert len(spans) > 0

    # Check that spans have UiPath attributes
    session_spans = [
        span
        for span in spans
        if span.attributes
        and span.attributes.get("session.id") == "async-session"
        and span.attributes.get("thread.id") == "async-thread"
    ]
    assert len(session_spans) > 0


def test_openinference_attributes_present(instrumented_provider):
    """Test that OpenInference attributes are present on spans."""
    provider, exporter = instrumented_provider

    def process_node(state: dict) -> dict:
        return {"result": state.get("value", 0) * 2}

    builder = StateGraph(dict)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    graph = builder.compile()
    graph.invoke({"value": 5})

    spans = exporter.get_finished_spans()
    assert len(spans) > 0

    # Check for OpenInference span kind attribute
    # OpenInference adds "openinference.span.kind" attribute
    openinference_spans = [
        span
        for span in spans
        if span.attributes and "openinference.span.kind" in span.attributes
    ]

    # We should have at least one span with OpenInference attributes
    assert len(openinference_spans) > 0


def test_combined_openinference_and_uipath_attributes(instrumented_provider):
    """Test that both OpenInference and UiPath attributes are present."""
    provider, exporter = instrumented_provider

    set_session_context(session_id="combined-session")

    def process_node(state: dict) -> dict:
        return {"result": state.get("value", 0) * 2}

    builder = StateGraph(dict)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    graph = builder.compile()
    graph.invoke({"value": 5})

    spans = exporter.get_finished_spans()
    assert len(spans) > 0

    # Find spans with both OpenInference and UiPath attributes
    combined_spans = [
        span
        for span in spans
        if span.attributes
        and "openinference.span.kind" in span.attributes
        and span.attributes.get("session.id") == "combined-session"
    ]

    # Should have at least one span with both types of attributes
    assert len(combined_spans) > 0
