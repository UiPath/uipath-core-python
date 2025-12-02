"""Basic instrumentation tests for LangGraph integration.

Tests validate core instrumentation functionality for LangGraph workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from uipath.core.telemetry.integrations_full.langgraph import (
    instrument_langgraph,
    uninstrument_langgraph,
)

if TYPE_CHECKING:
    from langgraph.graph.state import StateGraph
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


def test_instrumentor_initialization() -> None:
    """Verify LangGraph instrumentor can be initialized."""
    # Instrument
    instrument_langgraph()

    # Verify module state
    from uipath.core.telemetry.integrations_full.langgraph import _instrumented

    assert _instrumented is True

    # Cleanup
    uninstrument_langgraph()


def test_simple_workflow_creates_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify basic workflow creates spans."""
    instrument_langgraph()

    # Compile after instrumentation
    compiled = simple_workflow.compile()

    # Execute workflow
    result = compiled.invoke({"value": 5})

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0

    # Verify we have a langgraph.invoke parent span
    span_names = [s.name for s in spans]
    assert "langgraph.invoke" in span_names

    # Verify result (5 + 1 = 6, then 6 * 2 = 12)
    assert result["value"] == 12

    # Cleanup
    uninstrument_langgraph()


@pytest.mark.asyncio
async def test_async_workflow_creates_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify async workflow creates spans."""
    instrument_langgraph()

    # Compile after instrumentation
    compiled = simple_workflow.compile()

    # Execute workflow asynchronously
    result = await compiled.ainvoke({"value": 3})

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0

    # Verify we have a langgraph.ainvoke parent span
    span_names = [s.name for s in spans]
    assert "langgraph.ainvoke" in span_names

    # Verify result (3 + 1 = 4, then 4 * 2 = 8)
    assert result["value"] == 8

    # Cleanup
    uninstrument_langgraph()


def test_workflow_with_conditional_edges(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    conditional_workflow: StateGraph,
) -> None:
    """Verify workflow with conditional edges creates proper spans."""
    instrument_langgraph()

    # Compile after instrumentation
    compiled = conditional_workflow.compile()

    # Execute conditional workflow
    result = compiled.invoke({"messages": [], "iteration": 0})

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0

    # Verify parent span exists
    span_names = [s.name for s in spans]
    assert "langgraph.invoke" in span_names

    # Verify workflow executed to completion
    assert result["iteration"] >= 3

    # Cleanup
    uninstrument_langgraph()


def test_uninstrumentation(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify uninstrument() properly restores original behavior."""
    # Instrument
    instrument_langgraph()

    # Compile after instrumentation
    compiled = simple_workflow.compile()

    # Execute workflow (should create spans)
    result1 = compiled.invoke({"value": 1})
    telemetry_client.flush()

    # Verify spans were created
    spans_before = len(in_memory_exporter.get_finished_spans())
    assert spans_before > 0
    assert result1["value"] == 4  # (1 + 1) * 2

    # Clear exporter
    in_memory_exporter.clear()

    # Uninstrument
    uninstrument_langgraph()

    # Create new workflow after uninstrumentation
    from tests.telemetry.integrations_full.langgraph.conftest import SimpleState

    from langgraph.graph import END, StateGraph

    def increment(state: SimpleState) -> SimpleState:
        return {"value": state["value"] + 1}

    new_workflow = StateGraph(SimpleState)
    new_workflow.add_node("inc", increment)
    new_workflow.set_entry_point("inc")
    new_workflow.add_edge("inc", END)
    compiled_new = new_workflow.compile()

    # Execute new workflow (should NOT create langgraph.invoke spans)
    result2 = compiled_new.invoke({"value": 10})
    telemetry_client.flush()

    # Get spans after uninstrumentation
    spans_after = in_memory_exporter.get_finished_spans()

    # Verify workflow still works
    assert result2["value"] == 11

    # After uninstrumentation, no langgraph.invoke spans should be created
    span_names_after = [s.name for s in spans_after]
    assert "langgraph.invoke" not in span_names_after


def test_multiple_invocations(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify multiple workflow invocations create separate span hierarchies."""
    instrument_langgraph()

    # Compile after instrumentation
    compiled = simple_workflow.compile()

    # Execute workflow multiple times
    result1 = compiled.invoke({"value": 1})
    result2 = compiled.invoke({"value": 2})
    result3 = compiled.invoke({"value": 3})

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0

    # Count langgraph.invoke parent spans (should have 3)
    invoke_spans = [s for s in spans if s.name == "langgraph.invoke"]
    assert len(invoke_spans) == 3

    # Verify results
    assert result1["value"] == 4  # (1 + 1) * 2
    assert result2["value"] == 6  # (2 + 1) * 2
    assert result3["value"] == 8  # (3 + 1) * 2

    # Cleanup
    uninstrument_langgraph()
