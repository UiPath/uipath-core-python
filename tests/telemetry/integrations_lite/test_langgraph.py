"""Tests for integrations_lite LangGraph integration."""

from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.integrations_full._shared import (
    set_session_context,
)
from uipath.core.telemetry.integrations_lite.langgraph import (
    instrument_langgraph,
    is_instrumented,
    uninstrument_langgraph,
)

if TYPE_CHECKING:
    pass


def test_instrument_langgraph_patches_compile() -> None:
    """Test instrument_langgraph patches StateGraph.compile()."""
    pytest.importorskip("langgraph")

    assert not is_instrumented()

    instrument_langgraph()

    assert is_instrumented()

    uninstrument_langgraph()
    assert not is_instrumented()


def test_instrument_raises_if_already_instrumented() -> None:
    """Test instrument_langgraph raises if called twice."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    with pytest.raises(RuntimeError, match="already instrumented"):
        instrument_langgraph()

    uninstrument_langgraph()


def test_uninstrument_raises_if_not_instrumented() -> None:
    """Test uninstrument_langgraph raises if not instrumented."""
    with pytest.raises(RuntimeError, match="not instrumented"):
        uninstrument_langgraph()


def test_uninstrument_restores_original() -> None:
    """Test uninstrument_langgraph restores original compile()."""
    pytest.importorskip("langgraph")

    from langgraph.graph.state import StateGraph

    original = StateGraph.compile

    instrument_langgraph()
    assert StateGraph.compile != original

    uninstrument_langgraph()
    assert StateGraph.compile == original


def test_instrument_raises_without_langgraph() -> None:
    """Test instrument_langgraph raises if langgraph not installed."""
    import sys

    langgraph_module = sys.modules.get("langgraph")
    langgraph_graph = sys.modules.get("langgraph.graph")
    langgraph_state = sys.modules.get("langgraph.graph.state")

    sys.modules["langgraph"] = None  # type: ignore[assignment]
    sys.modules["langgraph.graph"] = None  # type: ignore[assignment]
    sys.modules["langgraph.graph.state"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(ImportError, match="langgraph package not found"):
            instrument_langgraph()
    finally:
        if langgraph_module:
            sys.modules["langgraph"] = langgraph_module
        if langgraph_graph:
            sys.modules["langgraph.graph"] = langgraph_graph
        if langgraph_state:
            sys.modules["langgraph.graph.state"] = langgraph_state


def test_invoke_creates_span(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test graph.invoke() creates telemetry span."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        def process_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": state.get("value", 0) * 2}

        builder = StateGraph(dict)
        builder.add_node("process", process_node)
        builder.set_entry_point("process")
        builder.set_finish_point("process")

        graph = builder.compile()
        result = graph.invoke({"value": 5})

        assert result["value"] == 10

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) >= 1

        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]
        assert langgraph_span.attributes.get("openinference.span.kind") == "CHAIN"

    finally:
        uninstrument_langgraph()


async def test_ainvoke_creates_span(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test graph.ainvoke() creates telemetry span."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        async def async_process_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": state.get("value", 0) * 3}

        builder = StateGraph(dict)
        builder.add_node("process", async_process_node)
        builder.set_entry_point("process")
        builder.set_finish_point("process")

        graph = builder.compile()
        result = await graph.ainvoke({"value": 5})

        assert result["value"] == 15

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) >= 1

        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]
        assert langgraph_span.attributes.get("openinference.span.kind") == "CHAIN"

    finally:
        uninstrument_langgraph()


def test_session_context_attached(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test session context is attached to spans."""
    pytest.importorskip("langgraph")

    instrument_langgraph()
    set_session_context(session_id="test-session", thread_id="test-thread")

    try:
        from langgraph.graph import StateGraph

        def process_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": 42}

        builder = StateGraph(dict)
        builder.add_node("process", process_node)
        builder.set_entry_point("process")
        builder.set_finish_point("process")

        graph = builder.compile()
        graph.invoke({})

        spans = in_memory_exporter.get_finished_spans()
        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]

        assert langgraph_span.attributes.get("session.id") == "test-session"
        assert langgraph_span.attributes.get("thread_id") == "test-thread"

    finally:
        uninstrument_langgraph()


def test_input_output_captured(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test input and output are captured in span attributes."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        def process_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"result": state.get("input", 0) + 10}

        builder = StateGraph(dict)
        builder.add_node("process", process_node)
        builder.set_entry_point("process")
        builder.set_finish_point("process")

        graph = builder.compile()
        result = graph.invoke({"input": 5})

        assert result["result"] == 15

        spans = in_memory_exporter.get_finished_spans()
        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]

        input_value = langgraph_span.attributes.get("input.value")
        output_value = langgraph_span.attributes.get("output.value")

        assert input_value is not None
        assert '"input": 5' in input_value or '"input":5' in input_value
        assert output_value is not None
        assert '"result": 15' in output_value or '"result":15' in output_value

    finally:
        uninstrument_langgraph()


def test_error_handling(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test errors in graph execution are recorded."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        def failing_node(state: dict[str, Any]) -> dict[str, Any]:
            raise ValueError("intentional error")

        builder = StateGraph(dict)
        builder.add_node("fail", failing_node)
        builder.set_entry_point("fail")
        builder.set_finish_point("fail")

        graph = builder.compile()

        with pytest.raises(ValueError, match="intentional error"):
            graph.invoke({})

        spans = in_memory_exporter.get_finished_spans()
        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]

        assert langgraph_span.status.status_code.name == "ERROR"

        events = list(langgraph_span.events)
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

    finally:
        uninstrument_langgraph()


def test_multiple_invocations(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test multiple graph invocations create separate spans."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        def process_node(state: dict[str, Any]) -> dict[str, Any]:
            return {"value": state.get("value", 0) + 1}

        builder = StateGraph(dict)
        builder.add_node("process", process_node)
        builder.set_entry_point("process")
        builder.set_finish_point("process")

        graph = builder.compile()

        graph.invoke({"value": 1})
        graph.invoke({"value": 2})
        graph.invoke({"value": 3})

        spans = in_memory_exporter.get_finished_spans()
        langgraph_spans = [s for s in spans if s.name == "LangGraph"]

        assert len(langgraph_spans) == 3

    finally:
        uninstrument_langgraph()
