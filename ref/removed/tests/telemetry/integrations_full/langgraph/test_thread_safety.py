"""Tests for LangGraph thread safety verification.

This module tests that concurrent instrumentation calls are thread-safe,
addressing the critical thread safety bug identified in the code review where
global variables were accessed without lock protection.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Annotated

import pytest
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from uipath.core.telemetry import init
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.integrations_full.langgraph import (
    instrument_langgraph,
    uninstrument_langgraph,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter

    from uipath.core.telemetry.client import TelemetryClient


# State definition for test workflows
class SimpleState(TypedDict):
    """Simple state for thread safety tests."""

    messages: Annotated[list[BaseMessage], add_messages]
    count: int


def increment_node(state: SimpleState) -> SimpleState:
    """Simple node that increments count."""
    return {"count": state.get("count", 0) + 1, "messages": []}


def test_concurrent_instrumentation_is_thread_safe(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that multiple threads calling instrument_langgraph() concurrently is safe.

    This verifies the fix for the race condition where global variables
    (_original_compile, _instrumented) were accessed without lock protection.
    """
    # Reset instrumentation state
    uninstrument_langgraph()

    # Track successes and failures
    results: list[bool] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def instrument_in_thread() -> None:
        """Instrument in a thread and track result."""
        try:
            instrument_langgraph()
            with lock:
                results.append(True)
        except Exception as e:
            with lock:
                errors.append(e)
                results.append(False)

    # Launch 10 concurrent threads trying to instrument
    threads = [threading.Thread(target=instrument_in_thread) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent instrumentation failed with errors: {errors}"
    assert all(results), "All instrumentation calls should succeed"

    # Verify instrumentation actually works after concurrent setup
    workflow = StateGraph(SimpleState)
    workflow.add_node("increment", increment_node)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", END)

    compiled = workflow.compile()
    result = compiled.invoke({"count": 0, "messages": []})

    telemetry_client.flush()
    spans = in_memory_exporter.get_finished_spans()

    # Should have spans (instrumentation successful)
    assert len(spans) > 0, "Instrumentation should work after concurrent setup"
    assert result["count"] == 1

    # Cleanup
    uninstrument_langgraph()


def test_concurrent_uninstrumentation_is_thread_safe(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that multiple threads calling uninstrument_langgraph() concurrently is safe."""
    # Setup: Instrument first
    instrument_langgraph()

    # Track successes and failures
    results: list[bool] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def uninstrument_in_thread() -> None:
        """Uninstrument in a thread and track result."""
        try:
            uninstrument_langgraph()
            with lock:
                results.append(True)
        except Exception as e:
            with lock:
                errors.append(e)
                results.append(False)

    # Launch 10 concurrent threads trying to uninstrument
    threads = [threading.Thread(target=uninstrument_in_thread) for _ in range(10)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent uninstrumentation failed with errors: {errors}"
    assert all(results), "All uninstrumentation calls should succeed"


def test_concurrent_workflow_execution_is_safe(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that multiple threads executing workflows concurrently is safe.

    Verifies that the cached tracer and instrumentation state work correctly
    under concurrent execution.
    """
    # Setup
    instrument_langgraph()

    workflow = StateGraph(SimpleState)
    workflow.add_node("increment", increment_node)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", END)
    compiled = workflow.compile()

    # Track results
    results: list[int] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def execute_workflow(initial_count: int) -> None:
        """Execute workflow in a thread."""
        try:
            result = compiled.invoke({"count": initial_count, "messages": []})
            with lock:
                results.append(result["count"])
        except Exception as e:
            with lock:
                errors.append(e)

    # Launch 20 concurrent workflow executions
    threads = [threading.Thread(target=execute_workflow, args=(i,)) for i in range(20)]

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    # Verify no errors occurred
    assert len(errors) == 0, f"Concurrent execution failed with errors: {errors}"
    assert len(results) == 20, "All executions should complete"

    # Verify each result is correct (input + 1)
    for i, result in enumerate(sorted(results)):
        expected = i + 1
        assert result == expected, f"Result {result} should be {expected}"

    # Verify spans were created
    telemetry_client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0, "Should have spans from concurrent executions"

    # Cleanup
    uninstrument_langgraph()


@pytest.mark.no_auto_tracer
def test_race_condition_detection() -> None:
    """Test that detects if race condition protection is removed.

    This test attempts to trigger a race condition by having multiple threads
    compete to instrument simultaneously. With proper locking, all should succeed.
    Without locking, we'd expect failures or corruption.
    """
    # Reset state completely
    uninstrument_langgraph()

    # Track if any thread saw inconsistent state
    inconsistencies: list[str] = []
    lock = threading.Lock()

    def attempt_instrumentation(thread_id: int) -> None:
        """Attempt instrumentation and check for inconsistencies."""
        try:
            # Uninstrument to reset
            uninstrument_langgraph()

            # Small delay to increase race condition likelihood
            import time

            time.sleep(0.001)

            # Instrument
            instrument_langgraph()

            # Verify we can compile (checks that _original_compile is valid)
            from langgraph.graph.state import StateGraph

            workflow = StateGraph(SimpleState)
            workflow.add_node("test", increment_node)
            workflow.set_entry_point("test")
            workflow.add_edge("test", END)

            try:
                compiled = workflow.compile()
                # Quick execution to verify it works
                compiled.invoke({"count": 0, "messages": []})
            except Exception as e:
                with lock:
                    inconsistencies.append(f"Thread {thread_id}: Compile/execute failed: {e}")

        except Exception as e:
            with lock:
                inconsistencies.append(f"Thread {thread_id}: Instrumentation failed: {e}")

    # Launch threads
    threads = [threading.Thread(target=attempt_instrumentation, args=(i,)) for i in range(5)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # With proper locking, should have no inconsistencies
    assert len(inconsistencies) == 0, f"Race conditions detected: {inconsistencies}"

    # Cleanup
    uninstrument_langgraph()


def test_instrumentation_persists_across_executions(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that instrumentation persists correctly across multiple executions.

    Once instrumented, multiple workflows can be compiled and executed without
    issues.
    """
    # Instrument once
    uninstrument_langgraph()
    instrument_langgraph()

    workflow = StateGraph(SimpleState)
    workflow.add_node("increment", increment_node)
    workflow.set_entry_point("increment")
    workflow.add_edge("increment", END)

    # Compile and execute multiple times
    for i in range(5):
        compiled = workflow.compile()
        result = compiled.invoke({"count": i, "messages": []})
        assert result["count"] == i + 1

    # Verify spans were created
    telemetry_client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0, "Should have spans from multiple executions"

    # Cleanup
    uninstrument_langgraph()
