"""Integration tests for end-to-end workflows."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from uipath.core.otel.client import OTelClient, OTelConfig, init_client
from uipath.core.otel.decorator import generation, tool
from uipath.core.otel.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_end_to_end_trace(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test complete trace with nested observations.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute - create nested observations
    with Trace(tracer, "workflow", execution_id="exec-123") as trace_ctx:
        with trace_ctx.agent("planner") as agent_obs:
            agent_obs.set_attribute("plan", "Execute steps")

        with trace_ctx.generation("llm-call", model="gpt-4") as gen_obs:
            gen_obs.set_attribute("prompt", "Hello")
            gen_obs.set_attribute("response", "Hi there!")

        with trace_ctx.tool("calculator") as tool_obs:
            tool_obs.set_attribute("operation", "add")
            tool_obs.set_attribute("result", 5)

    # Verify hierarchy
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 4  # 1 root + 3 observations

    # Find spans
    workflow = next(s for s in spans if s.name == "workflow")
    planner = next(s for s in spans if s.name == "planner")
    llm = next(s for s in spans if s.name == "llm-call")
    calc = next(s for s in spans if s.name == "calculator")

    # Verify parent-child relationships
    assert planner.parent.span_id == workflow.context.span_id
    assert llm.parent.span_id == workflow.context.span_id
    assert calc.parent.span_id == workflow.context.span_id

    # Verify all have execution_id
    assert workflow.attributes.get("execution.id") == "exec-123"


def test_decorator_integration(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test multiple decorators creating nested hierarchy.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @tool()
    def fetch_data(query: str) -> dict[str, str]:
        """Fetch data tool.

        Args:
            query: Search query

        Returns:
            Data
        """
        return {"data": f"results for {query}"}

    @generation()
    def analyze_data(data: dict[str, str]) -> str:
        """Analyze data with LLM.

        Args:
            data: Input data

        Returns:
            Analysis
        """
        # Call nested tool
        additional = fetch_data("additional query")
        return f"Analysis: {data} + {additional}"

    # Execute
    with Trace(tracer, "workflow"):
        result = analyze_data({"initial": "data"})

    # Verify
    assert "Analysis:" in result

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 3  # workflow + analyze_data + fetch_data

    # Verify hierarchy
    workflow = next(s for s in spans if s.name == "workflow")
    analyze = next(s for s in spans if s.name == "analyze_data")
    fetch = next(s for s in spans if s.name == "fetch_data")

    assert analyze.parent.span_id == workflow.context.span_id
    assert fetch.parent.span_id == analyze.context.span_id


def test_privacy_integration(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test privacy enforcement in real workflow.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup with privacy config
    config = OTelConfig(
        mode="dev",
        privacy={
            "redact_inputs": True,
            "redact_outputs": True,
        },
    )
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
    def llm_call(user_input: str) -> dict[str, str]:
        """LLM call.

        Args:
            user_input: User input

        Returns:
            Response
        """
        return {"output_text": "response"}

    # Execute
    with Trace(tracer, "workflow") as trace_ctx:
        # Manual observation with privacy
        with trace_ctx.generation("manual-gen") as obs:
            obs.set_attribute("input_data", "sensitive")
            obs.set_attribute("output_data", "secret")

        # Decorated function (note: decorator doesn't auto-set input/output attrs)
        llm_call("user query")

    # Verify privacy applied
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    manual_span = next(s for s in spans if s.name == "manual-gen")

    assert manual_span.attributes.get("input_data") == "[REDACTED]"
    assert manual_span.attributes.get("output_data") == "[REDACTED]"


async def test_async_workflow(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test async workflow with proper context propagation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
    async def async_llm(prompt: str) -> str:
        """Async LLM call.

        Args:
            prompt: Prompt

        Returns:
            Response
        """
        await asyncio.sleep(0.01)
        return f"Response to: {prompt}"

    @tool()
    async def async_tool(input_val: str) -> str:
        """Async tool.

        Args:
            input_val: Input

        Returns:
            Result
        """
        await asyncio.sleep(0.01)
        return f"Processed: {input_val}"

    # Execute
    with Trace(tracer, "async-workflow"):
        result1 = await async_llm("test1")
        result2 = await async_tool("test2")

    # Verify
    assert "Response to: test1" in result1
    assert "Processed: test2" in result2

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 3  # workflow + 2 operations

    # Verify hierarchy
    workflow = next(s for s in spans if s.name == "async-workflow")
    llm_span = next(s for s in spans if s.name == "async_llm")
    tool_span = next(s for s in spans if s.name == "async_tool")

    assert llm_span.parent.span_id == workflow.context.span_id
    assert tool_span.parent.span_id == workflow.context.span_id


def test_multi_trace_isolation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test multiple traces remain isolated.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute two separate traces
    with Trace(tracer, "trace1", execution_id="exec1"):
        with Trace(tracer, "trace1-child"):
            pass

    with Trace(tracer, "trace2", execution_id="exec2"):
        with Trace(tracer, "trace2-child"):
            pass

    # Verify isolation
    client.flush()
    spans = in_memory_exporter.get_finished_spans()

    trace1 = next(s for s in spans if s.name == "trace1")
    trace1_child = next(s for s in spans if s.name == "trace1-child")
    trace2 = next(s for s in spans if s.name == "trace2")
    trace2_child = next(s for s in spans if s.name == "trace2-child")

    # Verify execution IDs
    assert trace1.attributes.get("execution.id") == "exec1"
    assert trace2.attributes.get("execution.id") == "exec2"

    # Verify children belong to correct parents
    assert trace1_child.parent.span_id == trace1.context.span_id
    assert trace2_child.parent.span_id == trace2.context.span_id


def test_singleton_client_integration(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test singleton client works across multiple traces.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Initialize client via init_client
    config = OTelConfig(mode="dev")
    client = init_client(config)
    tracer = client.get_tracer()

    # Create multiple traces using same client
    with Trace(tracer, "trace1"):
        pass

    with Trace(tracer, "trace2"):
        pass

    # Verify both exported
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 2
    assert any(s.name == "trace1" for s in spans)
    assert any(s.name == "trace2" for s in spans)


def test_complex_nested_workflow(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test complex workflow with deep nesting.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @tool()
    def sub_tool(data: str) -> str:
        """Sub tool.

        Args:
            data: Input

        Returns:
            Result
        """
        return f"processed-{data}"

    @generation()
    def main_llm(prompt: str) -> str:
        """Main LLM.

        Args:
            prompt: Prompt

        Returns:
            Response
        """
        # Call tool within generation
        result = sub_tool(prompt)
        return f"llm-response-{result}"

    # Execute
    with Trace(tracer, "workflow", execution_id="complex-123") as trace_ctx:
        # Manual observation
        with trace_ctx.agent("orchestrator"):
            # Call decorated function from within manual observation
            response = main_llm("test")

    # Verify complex hierarchy
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 4  # workflow + orchestrator + main_llm + sub_tool

    workflow = next(s for s in spans if s.name == "workflow")
    orchestrator = next(s for s in spans if s.name == "orchestrator")
    main = next(s for s in spans if s.name == "main_llm")
    sub = next(s for s in spans if s.name == "sub_tool")

    # Verify hierarchy
    assert orchestrator.parent.span_id == workflow.context.span_id
    assert main.parent.span_id == orchestrator.context.span_id
    assert sub.parent.span_id == main.context.span_id


def test_error_propagation_integration(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test errors propagate through hierarchy.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @tool()
    def failing_tool() -> None:
        """Tool that fails.

        Raises:
            RuntimeError: Test error
        """
        raise RuntimeError("tool failed")

    @generation()
    def llm_with_tool() -> None:
        """LLM that calls failing tool.

        Raises:
            RuntimeError: Propagated from tool
        """
        failing_tool()

    # Execute - error should propagate
    import pytest

    with pytest.raises(RuntimeError, match="tool failed"):
        with Trace(tracer, "workflow"):
            llm_with_tool()

    # Verify all spans recorded errors
    client.flush()
    spans = in_memory_exporter.get_finished_spans()

    tool_span = next(s for s in spans if s.name == "failing_tool")
    llm_span = next(s for s in spans if s.name == "llm_with_tool")

    # Both should have ERROR status and exception events
    from opentelemetry.trace import StatusCode

    assert tool_span.status.status_code == StatusCode.ERROR
    assert llm_span.status.status_code == StatusCode.ERROR
    assert len(tool_span.events) > 0
    assert len(llm_span.events) > 0
