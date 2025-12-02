"""Integration tests for integrations_lite combining multiple components."""

from typing import TYPE_CHECKING, Any

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.decorator import traced
from uipath.core.telemetry.integrations_full._shared import set_session_context
from uipath.core.telemetry.integrations_lite.langchain import (
    instrument_langchain,
    uninstrument_langchain,
)
from uipath.core.telemetry.integrations_lite.langgraph import (
    instrument_langgraph,
    uninstrument_langgraph,
)

if TYPE_CHECKING:
    pass


async def test_calculator_agent_e2e(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """End-to-end test of calculator agent with full tracing."""
    pytest.importorskip("langgraph")

    instrument_langgraph()
    set_session_context(session_id="calc-session", thread_id="calc-thread")

    try:
        from langgraph.graph import StateGraph

        @traced(name="calculate", kind="TOOL")
        async def calculate(state: dict[str, Any]) -> dict[str, Any]:
            a = state.get("a", 0)
            b = state.get("b", 0)
            op = state.get("operator", "+")

            if op == "+":
                result = a + b
            elif op == "*":
                result = a * b
            elif op == "-":
                result = a - b
            elif op == "/":
                result = a / b if b != 0 else 0
            else:
                result = 0

            return {"result": result}

        builder = StateGraph(dict)
        builder.add_node("calculate", calculate)
        builder.set_entry_point("calculate")
        builder.set_finish_point("calculate")

        graph = builder.compile()
        result = await graph.ainvoke({"a": 5, "b": 3, "operator": "*"})

        assert result["result"] == 15

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) >= 2

        calculate_span = [s for s in spans if s.name == "calculate"][0]
        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]

        assert calculate_span.attributes.get("openinference.span.kind") == "TOOL"
        assert langgraph_span.attributes.get("openinference.span.kind") == "CHAIN"
        assert langgraph_span.attributes.get("session.id") == "calc-session"
        assert langgraph_span.attributes.get("thread_id") == "calc-thread"

    finally:
        uninstrument_langgraph()


def test_langchain_and_langgraph_together(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test LangChain and LangGraph integrations work together."""
    pytest.importorskip("langchain")
    pytest.importorskip("langgraph")
    pytest.importorskip("langsmith")

    instrument_langchain()
    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph
        from langsmith import traceable

        @traceable(run_type="tool", name="preprocess")
        def preprocess(value: int) -> int:
            return value + 1

        @traceable(run_type="tool", name="postprocess")
        def postprocess(value: int) -> int:
            return value * 2

        def node_function(state: dict[str, Any]) -> dict[str, Any]:
            value = state.get("value", 0)
            value = preprocess(value)
            value = postprocess(value)
            return {"result": value}

        builder = StateGraph(dict)
        builder.add_node("process", node_function)
        builder.set_entry_point("process")
        builder.set_finish_point("process")

        graph = builder.compile()
        result = graph.invoke({"value": 5})

        assert result["result"] == 12

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) >= 3

        preprocess_span = [s for s in spans if s.name == "preprocess"][0]
        postprocess_span = [s for s in spans if s.name == "postprocess"][0]
        langgraph_span = [s for s in spans if s.name == "LangGraph"][0]

        assert preprocess_span.attributes.get("openinference.span.kind") == "TOOL"
        assert postprocess_span.attributes.get("openinference.span.kind") == "TOOL"
        assert langgraph_span.attributes.get("openinference.span.kind") == "CHAIN"

    finally:
        uninstrument_langchain()
        uninstrument_langgraph()


async def test_multi_turn_conversation(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test multi-turn conversation with session context."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        @traced(name="chat_turn", kind="CHAIN")
        async def chat_turn(state: dict[str, Any]) -> dict[str, Any]:
            message = state.get("message", "")
            history = state.get("history", [])
            history.append({"user": message})
            history.append({"assistant": f"Response to: {message}"})
            return {"history": history}

        builder = StateGraph(dict)
        builder.add_node("chat", chat_turn)
        builder.set_entry_point("chat")
        builder.set_finish_point("chat")

        graph = builder.compile()

        set_session_context(session_id="conv-123", thread_id="thread-1")
        result1 = await graph.ainvoke({"message": "Hello", "history": []})

        set_session_context(session_id="conv-123", thread_id="thread-1")
        result2 = await graph.ainvoke(
            {"message": "How are you?", "history": result1["history"]}
        )

        assert len(result2["history"]) == 4

        spans = in_memory_exporter.get_finished_spans()
        langgraph_spans = [s for s in spans if s.name == "LangGraph"]

        assert len(langgraph_spans) == 2

        for span in langgraph_spans:
            assert span.attributes.get("session.id") == "conv-123"
            assert span.attributes.get("thread_id") == "thread-1"

    finally:
        uninstrument_langgraph()


async def test_error_recovery(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test error handling and recovery in integrated scenario."""
    pytest.importorskip("langgraph")

    instrument_langgraph()

    try:
        from langgraph.graph import StateGraph

        attempt_count = 0

        @traced(name="fallible_operation", kind="TOOL")
        async def fallible_operation(state: dict[str, Any]) -> dict[str, Any]:
            nonlocal attempt_count
            attempt_count += 1

            if attempt_count == 1:
                raise ValueError("First attempt fails")

            return {"success": True, "attempts": attempt_count}

        builder = StateGraph(dict)
        builder.add_node("operation", fallible_operation)
        builder.set_entry_point("operation")
        builder.set_finish_point("operation")

        graph = builder.compile()

        with pytest.raises(ValueError, match="First attempt fails"):
            await graph.ainvoke({})

        result = await graph.ainvoke({})

        assert result["success"] is True
        assert result["attempts"] == 2

        spans = in_memory_exporter.get_finished_spans()
        operation_spans = [s for s in spans if s.name == "fallible_operation"]

        assert len(operation_spans) == 2

        failed_span = operation_spans[0]
        successful_span = operation_spans[1]

        assert failed_span.status.status_code.name == "ERROR"
        assert successful_span.status.status_code.name == "OK"

    finally:
        uninstrument_langgraph()
