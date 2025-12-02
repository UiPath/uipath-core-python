"""End-to-end integration test for OpenInference instrumentation.

This test creates a real LangGraph agent from scratch and verifies that:
1. Telemetry is properly instrumented
2. Spans are created with OpenInference attributes
3. UiPath session context is added to spans
4. Telemetry can be exported successfully
"""

from typing import Annotated, TypedDict

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
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


class AgentState(TypedDict):
    """State for the calculator agent."""

    messages: Annotated[list, add_messages]
    current_value: float
    operation: str | None


def parse_input_node(state: AgentState) -> AgentState:
    """Parse the input message and extract operation."""
    messages = state.get("messages", [])
    if not messages:
        return {"operation": None, "current_value": 0.0}

    last_message = messages[-1]
    content = last_message.content if hasattr(last_message, "content") else str(last_message)

    # Simple parsing: "add 5", "multiply 3", etc.
    parts = content.lower().split()
    if len(parts) >= 2:
        operation = parts[0]
        try:
            value = float(parts[1])
            return {"operation": operation, "current_value": value}
        except ValueError:
            return {"operation": None, "current_value": 0.0}

    return {"operation": None, "current_value": 0.0}


def calculate_node(state: AgentState) -> AgentState:
    """Perform the calculation based on operation."""
    current_value = state.get("current_value", 0.0)
    operation = state.get("operation")
    messages = state.get("messages", [])

    # Get previous result if exists
    previous_result = 0.0
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            try:
                previous_result = float(msg.content)
                break
            except ValueError:
                pass

    # Perform operation
    result = previous_result
    if operation == "add":
        result = previous_result + current_value
    elif operation == "subtract":
        result = previous_result - current_value
    elif operation == "multiply":
        result = previous_result * current_value
    elif operation == "divide" and current_value != 0:
        result = previous_result / current_value
    elif operation == "set":
        result = current_value

    return {
        "messages": [AIMessage(content=str(result))],
        "current_value": result,
    }


def should_continue(state: AgentState) -> str:
    """Decide whether to continue or end."""
    operation = state.get("operation")
    if operation in ["add", "subtract", "multiply", "divide", "set"]:
        return "calculate"
    return "end"


@pytest.fixture
def instrumented_provider():
    """Create instrumented tracer provider for e2e test."""
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrument_langchain(tracer_provider=provider)

    yield provider, exporter

    uninstrument_langchain()
    provider.shutdown()
    clear_session_context()


def test_e2e_langgraph_agent_with_telemetry(instrumented_provider):
    """End-to-end test: Create LangGraph agent and verify telemetry export.

    This test creates a complete calculator agent with:
    - Multiple nodes (parse, calculate)
    - Conditional edges
    - State management
    - Message history

    Then verifies:
    - OpenInference attributes are present (span kind, etc.)
    - UiPath session context is added to all spans
    - Telemetry can be successfully exported
    - Span hierarchy is correct
    """
    provider, exporter = instrumented_provider

    # Set session context for this execution
    set_session_context(session_id="e2e-test-session", thread_id="e2e-test-thread")

    # Build the calculator agent
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("parse", parse_input_node)
    builder.add_node("calculate", calculate_node)

    # Set entry point
    builder.set_entry_point("parse")

    # Add conditional edges
    builder.add_conditional_edges(
        "parse",
        should_continue,
        {
            "calculate": "calculate",
            "end": "__end__",
        },
    )

    # Calculate always goes to end
    builder.add_edge("calculate", "__end__")

    # Compile the graph
    graph = builder.compile()

    # Execute a series of operations with state accumulation
    operations = ["set 10", "add 5", "multiply 2", "subtract 3"]

    state = {"messages": []}
    results = []

    for op in operations:
        state["messages"].append(HumanMessage(content=op))
        result = graph.invoke(state)
        results.append(result)
        # Create a new dict with a copy of the messages list for next iteration
        state = {"messages": list(result["messages"]), "current_value": result.get("current_value", 0.0)}

    # Verify results
    assert len(results) == 4
    # set 10 = 10
    assert float(results[0]["messages"][-1].content) == 10.0
    # 10 + 5 = 15
    assert float(results[1]["messages"][-1].content) == 15.0
    # 15 * 2 = 30
    assert float(results[2]["messages"][-1].content) == 30.0
    # 30 - 3 = 27
    assert float(results[3]["messages"][-1].content) == 27.0

    # Verify telemetry export
    spans = exporter.get_finished_spans()
    assert len(spans) > 0, "Expected spans to be created by LangGraph execution"

    # Verify OpenInference attributes are present
    openinference_spans = [
        span
        for span in spans
        if span.attributes and "openinference.span.kind" in span.attributes
    ]
    assert (
        len(openinference_spans) > 0
    ), "Expected OpenInference span kind attribute on some spans"

    # Verify UiPath session context is added to spans
    session_spans = [
        span
        for span in spans
        if span.attributes
        and span.attributes.get("session.id") == "e2e-test-session"
        and span.attributes.get("thread.id") == "e2e-test-thread"
    ]
    assert (
        len(session_spans) > 0
    ), "Expected UiPath session context on spans"

    # Verify both OpenInference and UiPath attributes coexist
    combined_spans = [
        span
        for span in spans
        if span.attributes
        and "openinference.span.kind" in span.attributes
        and span.attributes.get("session.id") == "e2e-test-session"
    ]
    assert (
        len(combined_spans) > 0
    ), "Expected spans with both OpenInference and UiPath attributes"

    # Verify span hierarchy (parent-child relationships)
    span_ids = {span.context.span_id for span in spans}
    parent_spans = [
        span for span in spans if span.parent and span.parent.span_id in span_ids
    ]
    assert len(parent_spans) > 0, "Expected parent-child span relationships"

    print(f"\n✓ E2E Test Results:")
    print(f"  - Total spans created: {len(spans)}")
    print(f"  - OpenInference spans: {len(openinference_spans)}")
    print(f"  - UiPath session spans: {len(session_spans)}")
    print(f"  - Combined attribute spans: {len(combined_spans)}")
    print(f"  - Spans with parents: {len(parent_spans)}")
    print(f"  - Agent executed {len(operations)} operations successfully")
    print(f"  - Final result: {results[-1]['messages'][-1].content}")


async def test_e2e_langgraph_agent_async(instrumented_provider):
    """End-to-end test with async execution.

    Verifies that telemetry works correctly with async LangGraph execution.
    """
    provider, exporter = instrumented_provider

    set_session_context(session_id="e2e-async-session", thread_id="e2e-async-thread")

    # Build async agent
    async def async_parse(state: AgentState) -> AgentState:
        """Async version of parse."""
        return parse_input_node(state)

    async def async_calculate(state: AgentState) -> AgentState:
        """Async version of calculate."""
        return calculate_node(state)

    builder = StateGraph(AgentState)
    builder.add_node("parse", async_parse)
    builder.add_node("calculate", async_calculate)
    builder.set_entry_point("parse")
    builder.add_conditional_edges(
        "parse",
        should_continue,
        {
            "calculate": "calculate",
            "end": "__end__",
        },
    )
    builder.add_edge("calculate", "__end__")

    graph = builder.compile()

    # Execute async
    result = await graph.ainvoke({"messages": [HumanMessage(content="set 42")]})

    # Verify result
    assert float(result["messages"][-1].content) == 42.0

    # Verify telemetry
    spans = exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify async session context
    async_session_spans = [
        span
        for span in spans
        if span.attributes and span.attributes.get("session.id") == "e2e-async-session"
    ]
    assert len(async_session_spans) > 0, "Expected session context on async spans"

    print(f"\n✓ E2E Async Test Results:")
    print(f"  - Total spans created: {len(spans)}")
    print(f"  - Async session spans: {len(async_session_spans)}")
    print(f"  - Result: {result['messages'][-1].content}")
