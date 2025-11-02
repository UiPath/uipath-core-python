"""Test span nesting behavior for traced decorators."""

import pytest

from tests.conftest import SpanCapture


def test_simple_sync_nesting(span_capture: SpanCapture):
    """Test that nested sync functions create proper parent-child relationships."""
    from uipath.core.tracing.decorators import traced

    @traced(name="outer")
    def outer_function():
        return inner_function()

    @traced(name="inner")
    def inner_function():
        return "result"

    result = outer_function()

    assert result == "result"

    spans = span_capture.get_spans()
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

    # Find spans by name
    inner_span = next(s for s in spans if s.name == "inner")
    outer_span = next(s for s in spans if s.name == "outer")

    # Verify parent-child relationship
    assert inner_span.parent is not None, "Inner span should have a parent"
    assert inner_span.parent.span_id == outer_span.context.span_id, (
        "Inner span's parent should be outer span"
    )

    # Verify they're in the same trace
    assert inner_span.context.trace_id == outer_span.context.trace_id, (
        "Spans should be in the same trace"
    )

    span_capture.print_hierarchy()


def test_deep_sync_nesting(span_capture: SpanCapture):
    """Test deeply nested sync functions."""
    from uipath.core.tracing.decorators import traced

    @traced(name="level1")
    def level1():
        return level2()

    @traced(name="level2")
    def level2():
        return level3()

    @traced(name="level3")
    def level3():
        return "deep_result"

    result = level1()

    assert result == "deep_result"

    spans = span_capture.get_spans()
    assert len(spans) == 3, f"Expected 3 spans, got {len(spans)}"

    # Find spans
    level1_span = next(s for s in spans if s.name == "level1")
    level2_span = next(s for s in spans if s.name == "level2")
    level3_span = next(s for s in spans if s.name == "level3")

    # Verify chain: level1 -> level2 -> level3
    assert level1_span.parent is None, "Level1 should be root"
    assert level2_span.parent.span_id == level1_span.context.span_id
    assert level3_span.parent.span_id == level2_span.context.span_id

    span_capture.print_hierarchy()


@pytest.mark.asyncio
async def test_async_nesting(span_capture: SpanCapture):
    """Test that nested async functions create proper parent-child relationships."""
    from uipath.core.tracing.decorators import traced

    @traced(name="async_outer")
    async def async_outer():
        return await async_inner()

    @traced(name="async_inner")
    async def async_inner():
        return "async_result"

    result = await async_outer()

    assert result == "async_result"

    spans = span_capture.get_spans()
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"

    inner_span = next(s for s in spans if s.name == "async_inner")
    outer_span = next(s for s in spans if s.name == "async_outer")

    assert inner_span.parent is not None
    assert inner_span.parent.span_id == outer_span.context.span_id

    span_capture.print_hierarchy()


@pytest.mark.asyncio
async def test_mixed_sync_async_nesting(span_capture: SpanCapture):
    """Test mixing sync and async traced functions."""
    from uipath.core.tracing.decorators import traced

    @traced(name="async_root")
    async def async_root():
        return await async_child()

    @traced(name="async_child")
    async def async_child():
        return sync_child()

    @traced(name="sync_child")
    def sync_child():
        return "mixed_result"

    result = await async_root()

    assert result == "mixed_result"

    spans = span_capture.get_spans()
    assert len(spans) == 3

    # Verify hierarchy
    async_root_span = next(s for s in spans if s.name == "async_root")
    async_child_span = next(s for s in spans if s.name == "async_child")
    sync_child_span = next(s for s in spans if s.name == "sync_child")

    assert async_child_span.parent.span_id == async_root_span.context.span_id
    assert sync_child_span.parent.span_id == async_child_span.context.span_id

    span_capture.print_hierarchy()


def test_multiple_calls_same_function(span_capture: SpanCapture):
    """Test that multiple calls to the same function create separate spans."""
    from uipath.core.tracing.decorators import traced

    @traced(name="called_multiple_times")
    def reusable_function(value):
        return value * 2

    @traced(name="caller")
    def caller():
        result1 = reusable_function(5)
        result2 = reusable_function(10)
        return result1 + result2

    result = caller()

    assert result == 30

    spans = span_capture.get_spans()
    assert len(spans) == 3, f"Expected 3 spans (1 caller + 2 calls), got {len(spans)}"

    # Find the caller span
    caller_span = next(s for s in spans if s.name == "caller")

    # Both reusable_function calls should be children of caller
    reusable_spans = [s for s in spans if s.name == "called_multiple_times"]
    assert len(reusable_spans) == 2

    for reusable_span in reusable_spans:
        assert reusable_span.parent.span_id == caller_span.context.span_id

    span_capture.print_hierarchy()


def test_sibling_functions(span_capture: SpanCapture):
    """Test that sibling function calls are handled correctly."""
    from uipath.core.tracing.decorators import traced

    @traced(name="sibling1")
    def sibling1():
        return "s1"

    @traced(name="sibling2")
    def sibling2():
        return "s2"

    @traced(name="parent")
    def parent():
        r1 = sibling1()
        r2 = sibling2()
        return r1 + r2

    result = parent()

    assert result == "s1s2"

    spans = span_capture.get_spans()
    assert len(spans) == 3

    parent_span = next(s for s in spans if s.name == "parent")
    sibling1_span = next(s for s in spans if s.name == "sibling1")
    sibling2_span = next(s for s in spans if s.name == "sibling2")

    # Both siblings should have the same parent
    assert sibling1_span.parent.span_id == parent_span.context.span_id
    assert sibling2_span.parent.span_id == parent_span.context.span_id

    span_capture.print_hierarchy()


def test_generator_nesting(span_capture: SpanCapture):
    """Test that generator functions maintain proper span nesting."""
    from uipath.core.tracing.decorators import traced

    @traced(name="generator_parent")
    def generator_parent():
        results = list(generator_child())
        return sum(results)

    @traced(name="generator_child")
    def generator_child():
        for i in range(3):
            yield i * 2

    result = generator_parent()

    assert result == 6  # 0 + 2 + 4

    spans = span_capture.get_spans()
    assert len(spans) == 2

    parent_span = next(s for s in spans if s.name == "generator_parent")
    child_span = next(s for s in spans if s.name == "generator_child")

    assert child_span.parent.span_id == parent_span.context.span_id

    span_capture.print_hierarchy()


@pytest.mark.asyncio
async def test_async_generator_nesting(span_capture: SpanCapture):
    """Test async generator nesting."""
    from uipath.core.tracing.decorators import traced

    @traced(name="async_gen_parent")
    async def async_gen_parent():
        results = []
        async for item in async_gen_child():
            results.append(item)
        return sum(results)

    @traced(name="async_gen_child")
    async def async_gen_child():
        for i in range(3):
            yield i * 3

    result = await async_gen_parent()

    assert result == 9  # 0 + 3 + 6

    spans = span_capture.get_spans()
    assert len(spans) == 2

    parent_span = next(s for s in spans if s.name == "async_gen_parent")
    child_span = next(s for s in spans if s.name == "async_gen_child")

    assert child_span.parent.span_id == parent_span.context.span_id

    span_capture.print_hierarchy()


def test_non_recording_blocks_children(span_capture: SpanCapture):
    """Test that recording=False on parent prevents children from being recorded."""
    from uipath.core.tracing.decorators import traced

    @traced(name="non_recording_parent", recording=False)
    def non_recording_parent():
        return recording_child()

    @traced(name="recording_child")
    def recording_child():
        return "result"

    result = non_recording_parent()

    assert result == "result"

    spans = span_capture.get_spans()
    # When parent has recording=False, children are also not recorded due to ParentBased sampler
    assert len(spans) == 0, (
        f"Expected 0 spans, but got {len(spans)}: {[s.name for s in spans]}"
    )

    span_capture.print_hierarchy()
