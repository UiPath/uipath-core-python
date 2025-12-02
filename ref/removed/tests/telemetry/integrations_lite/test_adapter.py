"""Tests for integrations_lite core adapter."""

from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from uipath.core.telemetry.integrations_lite._traced_adapter import adapt_to_traced

if TYPE_CHECKING:
    pass


def test_adapter_without_parentheses(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter works as @adapt_to_traced without parentheses."""

    @adapt_to_traced
    def my_function(x: int) -> int:
        return x * 2

    result = my_function(5)
    assert result == 10

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my_function"


def test_adapter_with_parentheses(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter works as @adapt_to_traced() with parentheses."""

    @adapt_to_traced()
    def my_function(x: int) -> int:
        return x * 3

    result = my_function(5)
    assert result == 15

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "my_function"


def test_adapter_with_custom_name(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter accepts custom span name."""

    @adapt_to_traced(name="custom_name")
    def my_function() -> str:
        return "test"

    result = my_function()
    assert result == "test"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "custom_name"


def test_adapter_with_kind(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter accepts OpenInference kind parameter."""

    @adapt_to_traced(kind="TOOL")
    def my_tool() -> str:
        return "tool_result"

    result = my_tool()
    assert result == "tool_result"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("openinference.span.kind") == "TOOL"


def test_adapter_with_hide_input(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter respects hide_input flag."""

    @adapt_to_traced(hide_input=True)
    def my_function(secret: str) -> str:
        return secret.upper()

    result = my_function("password123")
    assert result == "PASSWORD123"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    input_value = spans[0].attributes.get("input.value")
    assert input_value == "[REDACTED]"


def test_adapter_with_hide_output(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter respects hide_output flag."""

    @adapt_to_traced(hide_output=True)
    def my_function() -> str:
        return "secret_data"

    result = my_function()
    assert result == "secret_data"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    output_value = spans[0].attributes.get("output.value")
    assert output_value == "[REDACTED]"


def test_adapter_with_all_parameters(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter with all supported parameters."""

    @adapt_to_traced(
        name="full_test",
        kind="CHAIN",
        hide_input=False,
        hide_output=False,
    )
    def my_function(x: int) -> int:
        return x + 1

    result = my_function(10)
    assert result == 11

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "full_test"
    assert spans[0].attributes.get("openinference.span.kind") == "CHAIN"


def test_adapter_ignores_unknown_kwargs(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter ignores unknown kwargs (YAGNI principle)."""

    @adapt_to_traced(
        name="test",
        unknown_param="ignored",
        another_param=123,
    )
    def my_function() -> str:
        return "ok"

    result = my_function()
    assert result == "ok"

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "test"


async def test_adapter_with_async_function(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test adapter works with async functions."""

    @adapt_to_traced(name="async_test")
    async def async_function(x: int) -> int:
        return x * 2

    result = await async_function(7)
    assert result == 14

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "async_test"
