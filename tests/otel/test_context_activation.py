"""Tests for span context activation (CRITICAL fix validation)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from opentelemetry import trace as trace_api

from uipath.core.otel.client import init_client
from uipath.core.otel.config import TelemetryConfig
from uipath.core.otel.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_trace_root_span_activation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test that root span is activated in OpenTelemetry context.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "test-trace"):
        # Verify span is active in OTel context
        current_span = trace_api.get_current_span()
        assert current_span is not None
        assert current_span.is_recording()
        assert current_span.get_span_context().is_valid

    # Verify context restored (no active span)
    after_span = trace_api.get_current_span()
    assert not after_span.is_recording()

    # Verify span exported
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test-trace"


def test_observation_span_activation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test that observation spans are activated and create parent-child relationships.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "root") as trace_ctx:
        root_span = trace_api.get_current_span()
        root_span_id = root_span.get_span_context().span_id

        with trace_ctx.span("child", kind="generation"):
            child_span = trace_api.get_current_span()
            assert child_span.is_recording()

            # Verify parent-child relationship
            child_context = child_span.get_span_context()
            child_span_id = child_context.span_id
            assert child_span_id != root_span_id  # Different spans

        # After observation exits, root should be active again
        restored_span = trace_api.get_current_span()
        assert restored_span.get_span_context().span_id == root_span_id

    # Verify spans exported with correct hierarchy
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 2

    # Find root and child spans
    root = next(s for s in spans if s.name == "root")
    child = next(s for s in spans if s.name == "child")

    # Verify parent-child relationship in exported spans
    assert child.parent is not None
    assert child.parent.span_id == root.context.span_id


def test_nested_observations(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test nested observations maintain proper hierarchy.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute - create 3 levels of nesting
    with Trace(tracer, "root") as trace_ctx:
        with trace_ctx.span("level1", kind="workflow"):
            with trace_ctx.span("level2", kind="tool"):
                with trace_ctx.span("level3", kind="agent"):
                    # Innermost span should be active
                    current = trace_api.get_current_span()
                    assert current.is_recording()

    # Verify hierarchy
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 4

    # Build hierarchy map
    span_map = {s.name: s for s in spans}
    root = span_map["root"]
    level1 = span_map["level1"]
    level2 = span_map["level2"]
    level3 = span_map["level3"]

    # Verify parent-child relationships
    assert level1.parent is not None
    assert level1.parent.span_id == root.context.span_id

    assert level2.parent is not None
    assert level2.parent.span_id == level1.context.span_id

    assert level3.parent is not None
    assert level3.parent.span_id == level2.context.span_id


async def test_async_context_propagation(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test async context propagation across await boundaries.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    async def async_operation(trace_ctx: Trace, name: str) -> None:
        """Async function that creates observation.

        Args:
            trace_ctx: Trace context
            name: Operation name
        """
        with trace_ctx.span(name, kind="generation"):
            # Simulate async work
            await asyncio.sleep(0.01)
            current = trace_api.get_current_span()
            assert current.is_recording()

    # Execute
    with Trace(tracer, "async-root") as trace_ctx:
        # Launch concurrent async operations
        await asyncio.gather(
            async_operation(trace_ctx, "async-op-1"),
            async_operation(trace_ctx, "async-op-2"),
            async_operation(trace_ctx, "async-op-3"),
        )

    # Verify all spans exported
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 4  # 1 root + 3 operations

    # Verify all async operations have same root parent
    root = next(s for s in spans if s.name == "async-root")
    async_spans = [s for s in spans if s.name.startswith("async-op")]
    assert len(async_spans) == 3

    for span in async_spans:
        assert span.parent is not None
        assert span.parent.span_id == root.context.span_id


def test_decorator_context_activation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator properly activates spans for nesting.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    from uipath.core.otel.decorator import traced

    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "decorator-test") as trace_ctx:

        @traced(kind="generation")
        def decorated_func() -> dict[str, str]:
            """Decorated function for testing.

            Returns:
                Test response dict
            """
            # Inside decorated function, create manual observation using trace_ctx
            with trace_ctx.span("manual-tool", kind="tool"):
                pass
            return {"response": "test"}

        decorated_func()

    # Verify hierarchy
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 3  # root + decorated_func + manual-tool

    # Verify decorator span is child of root
    root = next(s for s in spans if s.name == "decorator-test")
    decorator_span = next(s for s in spans if s.name == "decorated_func")
    manual_tool = next(s for s in spans if s.name == "manual-tool")

    assert decorator_span.parent is not None
    assert decorator_span.parent.span_id == root.context.span_id

    # Verify manual tool is child of decorator span
    assert manual_tool.parent is not None
    assert manual_tool.parent.span_id == decorator_span.context.span_id
