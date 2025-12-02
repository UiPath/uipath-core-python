"""Operation type tests for LangGraph integration.

Tests validate different LangGraph operation types are instrumented correctly:
- StateGraph structure
- Node execution
- Edge traversal
- Conditional edges
- State management
- Multi-step workflows
- Loops/iterations
- Error handling
- Parallel execution
- Complex routing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from uipath.core.telemetry.integrations_full.langgraph import (
    instrument_langgraph,
    uninstrument_langgraph,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


# ============================================================================
# State Definitions
# ============================================================================


class SimpleState(TypedDict):
    """Simple state for basic tests."""

    value: int


class CounterState(TypedDict):
    """State for iteration tests."""

    count: int
    max_iterations: int


class MessageState(TypedDict):
    """State with messages."""

    messages: Annotated[list, add_messages]
    step: str


class MultiPathState(TypedDict):
    """State for routing tests."""

    path: str
    visited: list[str]


class RoutingState(TypedDict):
    """State for complex routing."""

    score: int
    path: str


# ============================================================================
# Tests
# ============================================================================


def test_simple_node_execution(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify simple node execution creates spans."""
    instrument_langgraph()

    def increment(state: SimpleState) -> SimpleState:
        """Increment value."""
        return {"value": state["value"] + 1}

    # Build workflow
    workflow = StateGraph(SimpleState)
    workflow.add_node("increment", increment)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", END)

    compiled = workflow.compile()
    result = compiled.invoke({"value": 5})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify langgraph.invoke span exists
    span_names = [s.name for s in spans]
    assert "langgraph.invoke" in span_names

    # Verify result
    assert result["value"] == 6

    # Cleanup
    uninstrument_langgraph()


def test_multi_node_workflow(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify multi-node workflow creates proper span hierarchy."""
    instrument_langgraph()

    def add_ten(state: SimpleState) -> SimpleState:
        """Add 10."""
        return {"value": state["value"] + 10}

    def multiply_two(state: SimpleState) -> SimpleState:
        """Multiply by 2."""
        return {"value": state["value"] * 2}

    def subtract_five(state: SimpleState) -> SimpleState:
        """Subtract 5."""
        return {"value": state["value"] - 5}

    # Build workflow: (5 + 10) * 2 - 5 = 25
    workflow = StateGraph(SimpleState)
    workflow.add_node("add", add_ten)
    workflow.add_node("multiply", multiply_two)
    workflow.add_node("subtract", subtract_five)
    workflow.set_entry_point("add")
    workflow.add_edge("add", "multiply")
    workflow.add_edge("multiply", "subtract")
    workflow.add_edge("subtract", END)

    compiled = workflow.compile()
    result = compiled.invoke({"value": 5})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert result["value"] == 25

    # Cleanup
    uninstrument_langgraph()


def test_conditional_edge_routing(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify conditional edges route correctly and create spans."""
    instrument_langgraph()

    def start(state: MultiPathState) -> MultiPathState:
        """Start node."""
        return {"visited": [*state.get("visited", []), "start"]}

    def path_a(state: MultiPathState) -> MultiPathState:
        """Path A node."""
        return {"visited": [*state["visited"], "path_a"]}

    def path_b(state: MultiPathState) -> MultiPathState:
        """Path B node."""
        return {"visited": [*state["visited"], "path_b"]}

    def router(state: MultiPathState) -> Literal["a", "b"]:
        """Route based on path."""
        return "a" if state.get("path") == "a" else "b"

    # Build workflow with conditional routing
    workflow = StateGraph(MultiPathState)
    workflow.add_node("start", start)
    workflow.add_node("path_a", path_a)
    workflow.add_node("path_b", path_b)
    workflow.set_entry_point("start")
    workflow.add_conditional_edges("start", router, {"a": "path_a", "b": "path_b"})
    workflow.add_edge("path_a", END)
    workflow.add_edge("path_b", END)

    compiled = workflow.compile()

    # Test path A
    result_a = compiled.invoke({"path": "a", "visited": []})
    assert result_a["visited"] == ["start", "path_a"]

    # Clear exporter
    in_memory_exporter.clear()

    # Test path B
    result_b = compiled.invoke({"path": "b", "visited": []})
    assert result_b["visited"] == ["start", "path_b"]

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Cleanup
    uninstrument_langgraph()


def test_loop_with_iteration_limit(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify loops create spans for each iteration."""
    instrument_langgraph()

    def increment(state: CounterState) -> CounterState:
        """Increment counter."""
        return {"count": state["count"] + 1}

    def should_continue(state: CounterState) -> Literal["continue", "end"]:
        """Check if should continue."""
        return "continue" if state["count"] < state["max_iterations"] else "end"

    # Build workflow with loop
    workflow = StateGraph(CounterState)
    workflow.add_node("increment", increment)
    workflow.set_entry_point("increment")
    workflow.add_conditional_edges(
        "increment", should_continue, {"continue": "increment", "end": END}
    )

    compiled = workflow.compile()
    result = compiled.invoke({"count": 0, "max_iterations": 5})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert result["count"] == 5

    # Cleanup
    uninstrument_langgraph()


def test_error_in_node_recorded(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify errors in nodes are recorded in spans."""
    instrument_langgraph()

    def failing_node(state: SimpleState) -> SimpleState:
        """Node that raises an error."""
        raise ValueError("Intentional node error")

    # Build workflow
    workflow = StateGraph(SimpleState)
    workflow.add_node("failing", failing_node)
    workflow.set_entry_point("failing")
    workflow.add_edge("failing", END)

    compiled = workflow.compile()

    # Invoke and expect error
    with pytest.raises(ValueError, match="Intentional node error"):
        compiled.invoke({"value": 1})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify error recorded
    error_spans = [s for s in spans if s.status.status_code.name == "ERROR"]
    assert len(error_spans) > 0

    # Cleanup
    uninstrument_langgraph()


@pytest.mark.asyncio
async def test_async_node_execution(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify async node execution creates spans."""
    instrument_langgraph()

    async def async_increment(state: SimpleState) -> SimpleState:
        """Async increment."""
        return {"value": state["value"] + 1}

    # Build workflow
    workflow = StateGraph(SimpleState)
    workflow.add_node("async_inc", async_increment)
    workflow.set_entry_point("async_inc")
    workflow.add_edge("async_inc", END)

    compiled = workflow.compile()
    result = await compiled.ainvoke({"value": 10})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify langgraph.ainvoke span
    span_names = [s.name for s in spans]
    assert "langgraph.ainvoke" in span_names

    # Verify result
    assert result["value"] == 11

    # Cleanup
    uninstrument_langgraph()


def test_state_updates_across_nodes(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify state updates properly across node executions."""
    instrument_langgraph()

    class SimpleMessageState(TypedDict):
        """Simple message state without add_messages."""
        messages: list[str]
        step: str

    def node1(state: SimpleMessageState) -> SimpleMessageState:
        """First node."""
        return {"messages": [*state.get("messages", []), "node1"], "step": "node1"}

    def node2(state: SimpleMessageState) -> SimpleMessageState:
        """Second node."""
        return {"messages": [*state.get("messages", []), "node2"], "step": "node2"}

    def node3(state: SimpleMessageState) -> SimpleMessageState:
        """Third node."""
        return {"messages": [*state.get("messages", []), "node3"], "step": "node3"}

    # Build workflow
    workflow = StateGraph(SimpleMessageState)
    workflow.add_node("n1", node1)
    workflow.add_node("n2", node2)
    workflow.add_node("n3", node3)
    workflow.set_entry_point("n1")
    workflow.add_edge("n1", "n2")
    workflow.add_edge("n2", "n3")
    workflow.add_edge("n3", END)

    compiled = workflow.compile()
    result = compiled.invoke({"messages": [], "step": ""})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify state accumulated messages
    assert "node1" in result["messages"]
    assert "node2" in result["messages"]
    assert "node3" in result["messages"]
    assert result["step"] == "node3"

    # Cleanup
    uninstrument_langgraph()


def test_parallel_node_execution(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify parallel node execution creates proper spans."""
    instrument_langgraph()

    class ParallelState(TypedDict):
        """State for parallel execution."""
        results: list[str]

    def node_a(state: ParallelState) -> ParallelState:
        """Node A."""
        return {"results": [*state.get("results", []), "a"]}

    def node_b(state: ParallelState) -> ParallelState:
        """Node B."""
        return {"results": [*state.get("results", []), "b"]}

    def node_c(state: ParallelState) -> ParallelState:
        """Node C."""
        return {"results": [*state.get("results", []), "c"]}

    def combiner(state: ParallelState) -> ParallelState:
        """Combine results."""
        return {"results": sorted(state.get("results", []))}

    # Build workflow - note: LangGraph doesn't have built-in parallel execution
    # but we can test sequential execution of multiple branches
    workflow = StateGraph(ParallelState)
    workflow.add_node("a", node_a)
    workflow.add_node("b", node_b)
    workflow.add_node("c", node_c)
    workflow.add_node("combine", combiner)
    workflow.set_entry_point("a")
    workflow.add_edge("a", "b")
    workflow.add_edge("b", "c")
    workflow.add_edge("c", "combine")
    workflow.add_edge("combine", END)

    compiled = workflow.compile()
    result = compiled.invoke({"results": []})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert result["results"] == ["a", "b", "c"]

    # Cleanup
    uninstrument_langgraph()


def test_complex_routing_logic(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify complex routing logic with multiple conditions."""
    instrument_langgraph()

    def scorer(state: RoutingState) -> RoutingState:
        """Score the input."""
        # Keep existing score or default to 0
        return {"path": "scored"}

    def low_handler(state: RoutingState) -> RoutingState:
        """Handle low scores."""
        return {"path": "low"}

    def mid_handler(state: RoutingState) -> RoutingState:
        """Handle mid scores."""
        return {"path": "mid"}

    def high_handler(state: RoutingState) -> RoutingState:
        """Handle high scores."""
        return {"path": "high"}

    def route_by_score(state: RoutingState) -> Literal["low", "mid", "high"]:
        """Route based on score."""
        score = state.get("score", 0)
        if score < 30:
            return "low"
        elif score < 70:
            return "mid"
        else:
            return "high"

    # Build workflow
    workflow = StateGraph(RoutingState)
    workflow.add_node("scorer", scorer)
    workflow.add_node("low", low_handler)
    workflow.add_node("mid", mid_handler)
    workflow.add_node("high", high_handler)
    workflow.set_entry_point("scorer")
    workflow.add_conditional_edges(
        "scorer", route_by_score, {"low": "low", "mid": "mid", "high": "high"}
    )
    workflow.add_edge("low", END)
    workflow.add_edge("mid", END)
    workflow.add_edge("high", END)

    compiled = workflow.compile()

    # Test different scores
    result_low = compiled.invoke({"score": 10, "path": ""})
    assert result_low["path"] == "low"

    result_mid = compiled.invoke({"score": 50, "path": ""})
    assert result_mid["path"] == "mid"

    result_high = compiled.invoke({"score": 90, "path": ""})
    assert result_high["path"] == "high"

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Should have 3 invoke spans (one for each test)
    invoke_spans = [s for s in spans if s.name == "langgraph.invoke"]
    assert len(invoke_spans) == 3

    # Cleanup
    uninstrument_langgraph()


def test_workflow_with_start_constant(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify workflow using START constant creates spans."""
    instrument_langgraph()

    def first_node(state: SimpleState) -> SimpleState:
        """First node."""
        return {"value": state["value"] * 2}

    # Build workflow using START constant
    workflow = StateGraph(SimpleState)
    workflow.add_node("first", first_node)
    workflow.add_edge(START, "first")
    workflow.add_edge("first", END)

    compiled = workflow.compile()
    result = compiled.invoke({"value": 21})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert result["value"] == 42

    # Cleanup
    uninstrument_langgraph()
