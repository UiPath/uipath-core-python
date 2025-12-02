"""Shared fixtures for LangGraph integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypedDict

import pytest
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


# ============================================================================
# State Definitions (module-level for LangGraph type introspection)
# ============================================================================


class SimpleState(TypedDict):
    """Simple state for basic workflows."""

    value: int


class MessageState(TypedDict):
    """State with messages for agent-like workflows."""

    messages: Annotated[list, add_messages]
    iteration: int


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def telemetry_client(in_memory_exporter: InMemorySpanExporter) -> TelemetryClient:
    """Create TelemetryClient with in-memory exporter.

    Args:
        in_memory_exporter: In-memory exporter from parent conftest

    Returns:
        Configured TelemetryClient instance
    """
    from uipath.core.telemetry import init

    # Initialize with console export (uses in-memory via global provider)
    client = init(enable_console_export=True)
    return client


@pytest.fixture
def simple_workflow() -> StateGraph:
    """Create simple LangGraph workflow for testing.

    Returns:
        StateGraph (NOT compiled - compile after instrumentation)
    """

    def increment_node(state: SimpleState) -> SimpleState:
        """Increment value by 1."""
        return {"value": state["value"] + 1}

    def double_node(state: SimpleState) -> SimpleState:
        """Double the value."""
        return {"value": state["value"] * 2}

    # Build graph but DON'T compile yet
    workflow = StateGraph(SimpleState)
    workflow.add_node("increment", increment_node)
    workflow.add_node("double", double_node)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", "double")
    workflow.add_edge("double", END)

    return workflow


@pytest.fixture
def conditional_workflow() -> StateGraph:
    """Create conditional workflow for testing routing.

    Returns:
        StateGraph (NOT compiled - compile after instrumentation)
    """

    def start_node(state: MessageState) -> MessageState:
        """Start node."""
        return {
            "messages": ["Start"],
            "iteration": state.get("iteration", 0) + 1,
        }

    def process_node(state: MessageState) -> MessageState:
        """Process node."""
        return {
            "messages": [f"Processed: iteration {state['iteration']}"],
            "iteration": state["iteration"] + 1,
        }

    def should_continue(state: MessageState) -> str:
        """Router function."""
        if state["iteration"] >= 3:
            return "end"
        return "continue"

    # Build graph but DON'T compile yet
    workflow = StateGraph(MessageState)
    workflow.add_node("start", start_node)
    workflow.add_node("process", process_node)
    workflow.set_entry_point("start")
    workflow.add_conditional_edges(
        "start", should_continue, {"continue": "process", "end": END}
    )
    workflow.add_edge("process", "start")

    return workflow
