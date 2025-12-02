"""Tests for LangGraph callback handler span hierarchy and metadata.

These tests verify critical requirements identified in consensus analysis:
1. Function spans must be children of their node spans (not siblings)
2. Session context must propagate to all node spans
3. Metadata should use native types (lists) not JSON strings
4. Node spans should capture input/output values
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from uipath.core.telemetry.integrations_full._shared import set_session_context
from uipath.core.telemetry.integrations_full.langgraph._callbacks import LangGraphTracer

if TYPE_CHECKING:
    from uipath.core.telemetry.client import TelemetryClient


@pytest.fixture
def tracer(telemetry_client: TelemetryClient) -> LangGraphTracer:
    """Create LangGraphTracer instance."""
    return LangGraphTracer(
        telemetry_client=telemetry_client,
        trace_state=True,
        trace_edges=True,
    )


@pytest.fixture
def setup_session_context():
    """Set up session context for tests."""
    set_session_context(session_id="test-session", thread_id="test-thread")
    yield
    # Cleanup
    from uipath.core.telemetry.integrations_full._shared import clear_session_context

    clear_session_context()


def test_function_spans_are_children_of_node_spans(
    tracer: LangGraphTracer,
    exporter: InMemorySpanExporter,
    telemetry_client: TelemetryClient,
    setup_session_context,
):
    """CRITICAL: Function spans must be children of their node spans, not siblings.

    Expected hierarchy:
        langgraph.calculate (node)
        └── calculate (function) ← must be child of node

    Current bug: Function span is sibling to node span (both share same parent).
    """
    from uuid import uuid4

    # Simulate node start (langgraph.calculate)
    node_run_id = uuid4()
    tracer.on_chain_start(
        serialized={"name": "calculate"},
        inputs={"a": 10, "b": 5},
        run_id=node_run_id,
        parent_run_id=None,
        metadata={
            "langgraph_step": 1,
            "langgraph_node": "calculate",
            "langgraph_triggers": ["branch:to:calculate"],
            "langgraph_path": ["__pregel_pull", "calculate"],
            "langgraph_checkpoint_ns": "calculate:123",
        },
    )

    # Simulate function call inside node (calculate function)
    function_run_id = uuid4()
    tracer.on_chain_start(
        serialized={"name": "calculate_impl"},
        inputs={"x": 10, "y": 5},
        run_id=function_run_id,
        parent_run_id=node_run_id,  # Function should be child of node
    )

    # End function
    tracer.on_chain_end(
        outputs={"result": 15},
        run_id=function_run_id,
    )

    # End node
    tracer.on_chain_end(
        outputs={"result": 15},
        run_id=node_run_id,
    )

    telemetry_client.flush()
    spans = exporter.get_finished_spans()

    # Find the spans
    node_span = next((s for s in spans if s.name == "langgraph.calculate"), None)
    function_span = next(
        (s for s in spans if s.name == "langgraph.calculate_impl"), None
    )

    assert node_span is not None, "Node span not found"
    assert function_span is not None, "Function span not found"

    # CRITICAL ASSERTION: Function span's parent MUST be the node span
    node_span_id = format(node_span.context.span_id, "016x")
    function_parent_id = (
        format(function_span.parent.span_id, "016x") if function_span.parent else None
    )

    assert function_parent_id == node_span_id, (
        f"Function span must be child of node span!\n"
        f"Expected parent: {node_span_id}\n"
        f"Actual parent: {function_parent_id}\n"
        f"This indicates the node span is not properly activated as the active context."
    )


def test_session_context_propagates_to_node_spans(
    tracer: LangGraphTracer,
    exporter: InMemorySpanExporter,
    telemetry_client: TelemetryClient,
    setup_session_context,
):
    """CRITICAL: session.id and thread_id must appear on all node spans.

    Current bug: Session context only on root span, missing from node spans.
    Reference implementation shows session.id on calculate node span.
    """
    from uuid import uuid4

    # Simulate node execution
    node_run_id = uuid4()
    tracer.on_chain_start(
        serialized={"name": "calculate"},
        inputs={"a": 10},
        run_id=node_run_id,
        metadata={
            "langgraph_step": 1,
            "langgraph_node": "calculate",
        },
    )

    tracer.on_chain_end(
        outputs={"result": 10},
        run_id=node_run_id,
    )

    telemetry_client.flush()
    spans = exporter.get_finished_spans()

    node_span = next((s for s in spans if s.name == "langgraph.calculate"), None)
    assert node_span is not None, "Node span not found"

    attrs = dict(node_span.attributes) if node_span.attributes else {}

    # CRITICAL ASSERTIONS: Session context must be on node spans
    assert "session.id" in attrs, (
        "session.id missing from node span! "
        "Session context must propagate to all node spans for trace correlation."
    )
    assert attrs.get("session.id") == "test-session", (
        f"session.id has wrong value: {attrs.get('session.id')}"
    )

    assert "thread_id" in attrs, (
        "thread_id missing from node span! "
        "Thread context must propagate to all node spans."
    )
    assert attrs.get("thread_id") == "test-thread", (
        f"thread_id has wrong value: {attrs.get('thread_id')}"
    )


def test_langgraph_metadata_uses_native_types_not_json_strings(
    tracer: LangGraphTracer,
    exporter: InMemorySpanExporter,
    telemetry_client: TelemetryClient,
):
    """MEDIUM: langgraph_triggers and langgraph_path should be lists, not JSON strings.

    Current bug: Stored as JSON strings like '["branch:to:calculate"]'
    Should be: Native OpenTelemetry list attributes for better querying
    """
    from uuid import uuid4

    node_run_id = uuid4()
    tracer.on_chain_start(
        serialized={"name": "calculate"},
        inputs={},
        run_id=node_run_id,
        metadata={
            "langgraph_triggers": ["branch:to:calculate", "start"],
            "langgraph_path": ["__pregel_pull", "calculate"],
        },
    )

    tracer.on_chain_end(outputs={}, run_id=node_run_id)
    telemetry_client.flush()

    spans = exporter.get_finished_spans()
    node_span = next((s for s in spans if s.name == "langgraph.calculate"), None)
    assert node_span is not None

    attrs = dict(node_span.attributes) if node_span.attributes else {}

    # Check langgraph_triggers
    triggers = attrs.get("langgraph_triggers")
    assert triggers is not None, "langgraph_triggers missing"

    # MEDIUM ASSERTION: Should be a list, not a JSON string
    if isinstance(triggers, str):
        pytest.fail(
            f"langgraph_triggers is a JSON string: {triggers}\n"
            f"Should be a native list for better querying in observability platforms.\n"
            f"OpenTelemetry supports array attributes - use those instead of JSON serialization."
        )

    assert isinstance(triggers, (list, tuple)), (
        f"langgraph_triggers should be a list/tuple, got {type(triggers)}"
    )
    # Telemetry may convert lists to tuples - both are acceptable
    assert list(triggers) == ["branch:to:calculate", "start"], (
        f"Wrong triggers value: {triggers}"
    )

    # Check langgraph_path
    path = attrs.get("langgraph_path")
    assert path is not None, "langgraph_path missing"

    if isinstance(path, str):
        pytest.fail(
            f"langgraph_path is a JSON string: {path}\n"
            f"Should be a native list for better querying."
        )

    assert isinstance(path, (list, tuple)), (
        f"langgraph_path should be a list/tuple, got {type(path)}"
    )
    # Telemetry may convert lists to tuples - both are acceptable
    assert list(path) == ["__pregel_pull", "calculate"], f"Wrong path value: {path}"


def test_node_spans_capture_input_and_output_values(
    tracer: LangGraphTracer,
    exporter: InMemorySpanExporter,
    telemetry_client: TelemetryClient,
):
    """MEDIUM: Node spans should capture input.value and output.value.

    Current gap: Node spans missing I/O that reference implementation includes.
    """
    from uuid import uuid4

    node_run_id = uuid4()
    input_state = {"a": 10, "b": 5, "operator": "*"}
    output_state = {"result": 50}

    tracer.on_chain_start(
        serialized={"name": "calculate"},
        inputs=input_state,
        run_id=node_run_id,
        metadata={"langgraph_node": "calculate"},
    )

    tracer.on_chain_end(
        outputs=output_state,
        run_id=node_run_id,
    )

    telemetry_client.flush()
    spans = exporter.get_finished_spans()

    node_span = next((s for s in spans if s.name == "langgraph.calculate"), None)
    assert node_span is not None

    attrs = dict(node_span.attributes) if node_span.attributes else {}

    # MEDIUM ASSERTIONS: Node spans should have I/O
    assert "input.value" in attrs, (
        "input.value missing from node span! "
        "Reference implementation shows this on calculate node."
    )

    assert "output.value" in attrs, (
        "output.value missing from node span! "
        "Node spans should capture output for observability."
    )

    # Verify values are correct (they may be JSON-serialized)
    input_value = attrs.get("input.value")
    if isinstance(input_value, str):
        input_value = json.loads(input_value)
    assert input_value == input_state, f"Wrong input value: {input_value}"

    output_value = attrs.get("output.value")
    if isinstance(output_value, str):
        output_value = json.loads(output_value)
    assert output_value == output_state, f"Wrong output value: {output_value}"


def test_successful_spans_have_ok_status(
    tracer: LangGraphTracer,
    exporter: InMemorySpanExporter,
    telemetry_client: TelemetryClient,
):
    """Successful node spans should have OK status, not UNSET."""
    from uuid import uuid4

    node_run_id = uuid4()
    tracer.on_chain_start(
        serialized={"name": "calculate"},
        inputs={},
        run_id=node_run_id,
    )

    tracer.on_chain_end(outputs={}, run_id=node_run_id)
    telemetry_client.flush()

    spans = exporter.get_finished_spans()
    node_span = next((s for s in spans if s.name == "langgraph.calculate"), None)
    assert node_span is not None

    assert node_span.status.status_code == StatusCode.OK, (
        f"Node span should have OK status, got {node_span.status.status_code}. "
        f"This aligns with OTLP conventions and reference implementation."
    )
