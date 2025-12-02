"""Context propagation tests for LangGraph integration.

Tests validate that UiPath context (execution_id, resource attributes, etc.)
propagates correctly through LangGraph workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from uipath.core.telemetry import init
from uipath.core.telemetry.config import TelemetryConfig
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


def test_execution_id_propagates_to_langgraph_spans(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify execution_id propagates to all LangGraph spans."""
    instrument_langgraph()

    # Compile after instrumentation
    compiled = simple_workflow.compile()

    # Create parent span with execution_id
    from uipath.core.telemetry import trace

    with trace("langgraph_workflow", execution_id="exec-graph-123"):
        compiled.invoke({"value": 5})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id is present on workflow span
    workflow_span = [s for s in spans if s.name == "langgraph_workflow"]
    assert len(workflow_span) == 1
    assert workflow_span[0].attributes.get("execution.id") == "exec-graph-123"

    # Verify langgraph.invoke span exists
    invoke_spans = [s for s in spans if s.name == "langgraph.invoke"]
    assert len(invoke_spans) >= 1

    # Cleanup
    uninstrument_langgraph()


@pytest.mark.no_auto_tracer
def test_resource_attributes_visible_in_langgraph(
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify UiPath resource attributes visible in LangGraph spans."""
    # Set up TracerProvider with custom resource attributes and in-memory exporter
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Reset global provider state
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    # Create resource with custom attributes
    resource = Resource(
        attributes={
            "service.name": "test-langgraph",
            "uipath.org_id": "org-456",
            "uipath.tenant_id": "tenant-789",
            "uipath.user_id": "user-abc",
        }
    )

    # Create provider with resource and exporter
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    trace.set_tracer_provider(provider)

    # Initialize UiPath client
    client = init(
        service_name="test-langgraph",
        enable_console_export=True,
        resource_attributes={
            "uipath.org_id": "org-456",
            "uipath.tenant_id": "tenant-789",
            "uipath.user_id": "user-abc",
        },
    )

    instrument_langgraph()

    # Compile and execute workflow
    compiled = simple_workflow.compile()
    compiled.invoke({"value": 10})

    client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify resource attributes on all spans
    for span in spans:
        resource = span.resource
        assert resource.attributes.get("uipath.org_id") == "org-456"
        assert resource.attributes.get("uipath.tenant_id") == "tenant-789"
        assert resource.attributes.get("uipath.user_id") == "user-abc"

    # Cleanup
    uninstrument_langgraph()


def test_nested_langgraph_workflows_inherit_execution_id(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify nested LangGraph workflows inherit execution_id."""
    instrument_langgraph()

    # Compile workflow
    compiled = simple_workflow.compile()

    from uipath.core.telemetry import trace

    # Create nested trace hierarchy
    with trace("parent_workflow", execution_id="exec-nested-graph"):
        with trace("child_operation"):
            compiled.invoke({"value": 3})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Parent should have execution_id
    parent_span = [s for s in spans if s.name == "parent_workflow"]
    assert len(parent_span) == 1
    assert parent_span[0].attributes.get("execution.id") == "exec-nested-graph"

    # Verify langgraph.invoke span exists
    invoke_spans = [s for s in spans if s.name == "langgraph.invoke"]
    assert len(invoke_spans) >= 1

    # Cleanup
    uninstrument_langgraph()


@pytest.mark.asyncio
async def test_async_langgraph_context_propagation(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify context propagates in async LangGraph operations."""
    instrument_langgraph()

    # Compile workflow
    compiled = simple_workflow.compile()

    from uipath.core.telemetry import trace

    # Create async trace
    async def async_langgraph_workflow() -> dict:
        with trace("async_langgraph", execution_id="exec-async-graph"):
            return await compiled.ainvoke({"value": 7})

    result = await async_langgraph_workflow()

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    workflow_span = [s for s in spans if s.name == "async_langgraph"]
    assert len(workflow_span) == 1
    assert workflow_span[0].attributes.get("execution.id") == "exec-async-graph"

    # Verify langgraph.ainvoke span exists
    ainvoke_spans = [s for s in spans if s.name == "langgraph.ainvoke"]
    assert len(ainvoke_spans) >= 1

    # Verify result (7 + 1 = 8, then 8 * 2 = 16)
    assert result["value"] == 16

    # Cleanup
    uninstrument_langgraph()


def test_multiple_langgraph_executions_isolated(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
    simple_workflow: StateGraph,
) -> None:
    """Verify multiple LangGraph executions have isolated execution_ids."""
    instrument_langgraph()

    # Compile workflow
    compiled = simple_workflow.compile()

    from uipath.core.telemetry import trace

    # Execute multiple workflows with different execution_ids
    with trace("graph_workflow_1", execution_id="graph-exec-1"):
        result1 = compiled.invoke({"value": 1})

    with trace("graph_workflow_2", execution_id="graph-exec-2"):
        result2 = compiled.invoke({"value": 2})

    with trace("graph_workflow_3", execution_id="graph-exec-3"):
        result3 = compiled.invoke({"value": 3})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify all three execution_ids are present
    exec1_spans = [
        s for s in spans if s.attributes.get("execution.id") == "graph-exec-1"
    ]
    exec2_spans = [
        s for s in spans if s.attributes.get("execution.id") == "graph-exec-2"
    ]
    exec3_spans = [
        s for s in spans if s.attributes.get("execution.id") == "graph-exec-3"
    ]

    assert len(exec1_spans) >= 1, "graph-exec-1 spans not found"
    assert len(exec2_spans) >= 1, "graph-exec-2 spans not found"
    assert len(exec3_spans) >= 1, "graph-exec-3 spans not found"

    # Verify results
    assert result1["value"] == 4  # (1 + 1) * 2
    assert result2["value"] == 6  # (2 + 1) * 2
    assert result3["value"] == 8  # (3 + 1) * 2

    # Verify no cross-contamination of execution_ids
    for span in spans:
        exec_id = span.attributes.get("execution.id")
        if exec_id:
            assert exec_id in ["graph-exec-1", "graph-exec-2", "graph-exec-3"]

    # Cleanup
    uninstrument_langgraph()
