"""Tests for integrations_lite LangChain integration."""

from typing import TYPE_CHECKING

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.integrations_lite.langchain import (
    instrument_langchain,
    is_instrumented,
    traceable_adapter,
    uninstrument_langchain,
)

if TYPE_CHECKING:
    pass


def test_traceable_adapter_without_parentheses(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test traceable_adapter works without parentheses."""

    @traceable_adapter
    def my_function(x: int) -> int:
        return x * 2

    result = my_function(5)
    assert result == 10

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my_function"


def test_traceable_adapter_with_parentheses(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test traceable_adapter works with empty parentheses."""

    @traceable_adapter()
    def my_function(x: int) -> int:
        return x * 3

    result = my_function(5)
    assert result == 15

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my_function"


def test_traceable_adapter_run_type_mapping(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test run_type is correctly mapped to OpenInference kind."""
    test_cases = [
        ("tool", "TOOL"),
        ("chain", "CHAIN"),
        ("llm", "LLM"),
        ("retriever", "RETRIEVER"),
        ("embedding", "EMBEDDING"),
        ("prompt", "CHAIN"),
        ("parser", "CHAIN"),
    ]

    for run_type, expected_kind in test_cases:

        @traceable_adapter(run_type=run_type)
        def my_function() -> str:
            return "test"

        result = my_function()
        assert result == "test"

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) > 0
        last_span = spans[-1]
        assert last_span.attributes.get("openinference.span.kind") == expected_kind

        in_memory_exporter.clear()


def test_traceable_adapter_custom_name(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test traceable_adapter respects custom name parameter."""

    @traceable_adapter(name="custom_tool_name")
    def my_tool() -> str:
        return "result"

    result = my_tool()
    assert result == "result"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_tool_name"


def test_traceable_adapter_ignores_langsmith_params(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test traceable_adapter ignores LangSmith-specific parameters."""

    @traceable_adapter(
        run_type="tool",
        name="test",
        tags=["ignored"],
        metadata={"also": "ignored"},
        project_name="ignored_project",
    )
    def my_function() -> str:
        return "ok"

    result = my_function()
    assert result == "ok"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"
    assert spans[0].attributes.get("openinference.span.kind") == "TOOL"


def test_instrument_langchain_patches_decorator() -> None:
    """Test instrument_langchain patches the @traceable decorator."""
    pytest.importorskip("langsmith")

    assert not is_instrumented()

    instrument_langchain()

    assert is_instrumented()

    import langsmith

    assert langsmith.traceable == traceable_adapter

    uninstrument_langchain()
    assert not is_instrumented()


def test_instrument_langchain_raises_if_already_instrumented() -> None:
    """Test instrument_langchain raises if called twice."""
    pytest.importorskip("langsmith")

    instrument_langchain()

    with pytest.raises(RuntimeError, match="already instrumented"):
        instrument_langchain()

    uninstrument_langchain()


def test_uninstrument_langchain_raises_if_not_instrumented() -> None:
    """Test uninstrument_langchain raises if not instrumented."""
    with pytest.raises(RuntimeError, match="not instrumented"):
        uninstrument_langchain()


def test_uninstrument_langchain_restores_original() -> None:
    """Test uninstrument_langchain restores original @traceable."""
    pytest.importorskip("langsmith")

    import langsmith

    original = langsmith.traceable

    instrument_langchain()
    assert langsmith.traceable != original

    uninstrument_langchain()
    assert langsmith.traceable == original


def test_instrument_raises_without_langsmith() -> None:
    """Test instrument_langchain raises if langsmith not installed."""
    import sys

    langsmith_module = sys.modules.get("langsmith")
    if langsmith_module:
        sys.modules["langsmith"] = None  # type: ignore[assignment]

    try:
        with pytest.raises(ImportError, match="langsmith package not found"):
            instrument_langchain()
    finally:
        if langsmith_module:
            sys.modules["langsmith"] = langsmith_module


def test_integration_with_langsmith_traceable(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test integration with real LangSmith @traceable decorator."""
    pytest.importorskip("langsmith")

    instrument_langchain()

    try:
        from langsmith import traceable

        @traceable(run_type="tool", name="multiply")
        def multiply(a: int, b: int) -> int:
            return a * b

        result = multiply(5, 3)
        assert result == 15

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "multiply"
        assert spans[0].attributes.get("openinference.span.kind") == "TOOL"

    finally:
        uninstrument_langchain()


async def test_integration_with_async_function(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test integration with async functions."""
    pytest.importorskip("langsmith")

    instrument_langchain()

    try:
        from langsmith import traceable

        @traceable(run_type="chain", name="async_chain")
        async def async_chain(x: int) -> int:
            return x * 2

        result = await async_chain(7)
        assert result == 14

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "async_chain"
        assert spans[0].attributes.get("openinference.span.kind") == "CHAIN"

    finally:
        uninstrument_langchain()


def test_nested_traceable_calls(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test nested @traceable decorated functions create span hierarchy."""
    pytest.importorskip("langsmith")

    instrument_langchain()

    try:
        from langsmith import traceable

        @traceable(run_type="tool", name="inner_tool")
        def inner_tool(x: int) -> int:
            return x + 1

        @traceable(run_type="chain", name="outer_chain")
        def outer_chain(x: int) -> int:
            return inner_tool(x) * 2

        result = outer_chain(5)
        assert result == 12

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 2

        inner_span = spans[0]
        outer_span = spans[1]

        assert inner_span.name == "inner_tool"
        assert outer_span.name == "outer_chain"

        assert inner_span.parent is not None
        assert inner_span.parent.span_id == outer_span.context.span_id

    finally:
        uninstrument_langchain()


def test_error_handling_in_traceable(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test errors in @traceable functions are recorded."""
    pytest.importorskip("langsmith")

    instrument_langchain()

    try:
        from langsmith import traceable

        @traceable(run_type="tool", name="failing_tool")
        def failing_tool() -> None:
            raise ValueError("intentional error")

        with pytest.raises(ValueError, match="intentional error"):
            failing_tool()

        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "failing_tool"
        assert span.status.status_code.name == "ERROR"

        events = list(span.events)
        assert len(events) >= 1
        exception_events = [e for e in events if e.name == "exception"]
        assert len(exception_events) >= 1

    finally:
        uninstrument_langchain()
