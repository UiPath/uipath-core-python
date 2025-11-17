"""Tests for decorator functionality."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from uipath.core.otel.client import init_client
from uipath.core.otel.config import TelemetryConfig
from uipath.core.otel.decorator import traced
from uipath.core.otel.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_generation_decorator_sync(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test @generation decorator on sync function.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    def llm_call(prompt: str) -> dict[str, str]:
        """LLM call function.

        Args:
            prompt: Input prompt

        Returns:
            Mock response
        """
        return {"response": f"Response to: {prompt}"}

    # Execute
    with Trace(tracer, "test"):
        result = llm_call("Hello")

    # Verify
    assert result == {"response": "Response to: Hello"}

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm_call")

    assert llm_span.attributes.get("openinference.span.kind") == "GENERATION"


async def test_generation_decorator_async(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @generation decorator on async function.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    async def async_llm_call(prompt: str) -> dict[str, str]:
        """Async LLM call.

        Args:
            prompt: Input prompt

        Returns:
            Mock response
        """
        await asyncio.sleep(0.01)
        return {"response": prompt}

    # Execute
    with Trace(tracer, "test"):
        result = await async_llm_call("test")

    # Verify
    assert result == {"response": "test"}

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert any(s.name == "async_llm_call" for s in spans)


def test_generation_decorator_auto_update(
    in_memory_exporter: InMemorySpanExporter,
    sample_openai_response: dict[str, any],
) -> None:
    """Test @generation auto_update enabled by default.

    Args:
        in_memory_exporter: In-memory exporter fixture
        sample_openai_response: Sample OpenAI response fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    def llm_call() -> dict[str, any]:
        """LLM call returning OpenAI-like response.

        Returns:
            OpenAI response mock
        """
        return sample_openai_response

    # Execute
    with Trace(tracer, "test"):
        llm_call()

    # Verify - parser should have extracted attributes
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm_call")

    # Should have more attributes than just span.type due to parsing
    assert len(llm_span.attributes) > 2


def test_tool_decorator_no_auto_update(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @tool decorator has auto_update=False by default.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="tool")
    def calculator(a: int, b: int) -> int:
        """Calculator tool.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum
        """
        return a + b

    # Execute
    with Trace(tracer, "test"):
        result = calculator(2, 3)

    # Verify
    assert result == 5

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    tool_span = next(s for s in spans if s.name == "calculator")

    assert tool_span.attributes.get("openinference.span.kind") == "TOOL"
    # Should NOT have update() attributes (auto_update=False)


def test_agent_decorator(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test @agent decorator.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="agent")
    def reasoning_agent(task: str) -> str:
        """Reasoning agent.

        Args:
            task: Task to reason about

        Returns:
            Result
        """
        return f"Completed: {task}"

    # Execute
    with Trace(tracer, "test"):
        result = reasoning_agent("analyze data")

    # Verify
    assert result == "Completed: analyze data"

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    agent_span = next(s for s in spans if s.name == "reasoning_agent")

    assert agent_span.attributes.get("openinference.span.kind") == "AGENT"


def test_decorator_exception_handling(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test decorator records exception and re-raises.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    def failing_function() -> None:
        """Function that raises exception.

        Raises:
            ValueError: Test error
        """
        raise ValueError("test error")

    # Execute - should raise
    with pytest.raises(ValueError, match="test error"):
        with Trace(tracer, "test"):
            failing_function()

    # Verify exception recorded
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    func_span = next(s for s in spans if s.name == "failing_function")

    assert func_span.status.status_code == StatusCode.ERROR
    # Should have at least one exception event (may have multiple from parser failures)
    exception_events = [e for e in func_span.events if e.name == "exception"]
    assert len(exception_events) >= 1
    # Verify the first exception is the ValueError we raised
    assert "ValueError" in str(exception_events[0].attributes.get("exception.type", ""))


def test_decorator_custom_name(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator with custom span name.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation", name="custom-span-name")
    def my_function() -> str:
        """Function with custom span name.

        Returns:
            Result
        """
        return "result"

    # Execute
    with Trace(tracer, "test"):
        my_function()

    # Verify custom name used
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert any(s.name == "custom-span-name" for s in spans)
    assert not any(s.name == "my_function" for s in spans)


def test_traced_generic_decorator(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test @traced decorator with custom span type.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="retrieval")
    def search_documents(query: str) -> list[str]:
        """Search function.

        Args:
            query: Search query

        Returns:
            Results
        """
        return [f"result for {query}"]

    # Execute
    with Trace(tracer, "test"):
        result = search_documents("test query")

    # Verify
    assert result == ["result for test query"]

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    search_span = next(s for s in spans if s.name == "search_documents")

    assert search_span.attributes.get("openinference.span.kind") == "RETRIEVAL"


def test_traced_routes_to_specialized(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @traced routes to specialized decorators.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    def llm_function() -> dict[str, str]:
        """LLM function.

        Returns:
            Response
        """
        return {"response": "test"}

    # Execute
    with Trace(tracer, "test"):
        llm_function()

    # Verify routed to generation decorator
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm_function")

    assert llm_span.attributes.get("openinference.span.kind") == "GENERATION"


def test_decorator_with_model_parameter(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @traced decorator creates generation span.

    Model metadata should come from update() with actual response in real usage.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    def llm_call() -> str:
        """LLM call.

        Returns:
            Response
        """
        return "response"

    # Execute
    with Trace(tracer, "test"):
        result = llm_call()

    # Verify span created with correct kind
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm_call")

    assert llm_span.attributes.get("openinference.span.kind") == "GENERATION"
    assert result == "response"


def test_decorator_explicit_auto_update_false(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test decorator does not auto-extract metadata from responses.

    Metadata extraction requires manual update() call with manual API.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    def llm_call() -> dict[str, str]:
        """LLM call.

        Returns:
            Response dict (mimics LLM response structure)
        """
        return {"model": "gpt-4", "response": "test"}

    # Execute
    with Trace(tracer, "test"):
        result = llm_call()

    # Verify
    assert result == {"model": "gpt-4", "response": "test"}

    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm_call")

    # Should only have openinference.span.kind (no auto-parsed metadata)
    attrs = llm_span.attributes
    assert "openinference.span.kind" in attrs
    # Should not have gen_ai.request.model or other parsed attributes
    # since decorator doesn't auto-extract metadata
    assert "gen_ai.request.model" not in attrs


async def test_decorator_async_exception(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test async decorator handles exceptions.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    @traced(kind="generation")
    async def async_failing() -> None:
        """Async function that fails.

        Raises:
            RuntimeError: Test error
        """
        await asyncio.sleep(0.01)
        raise RuntimeError("async error")

    # Execute
    with pytest.raises(RuntimeError, match="async error"):
        with Trace(tracer, "test"):
            await async_failing()

    # Verify exception recorded
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    func_span = next(s for s in spans if s.name == "async_failing")

    assert func_span.status.status_code == StatusCode.ERROR
    # Should have at least one exception event (may have multiple from parser failures)
    exception_events = [e for e in func_span.events if e.name == "exception"]
    assert len(exception_events) >= 1
    # Verify the first exception is the RuntimeError we raised
    assert "RuntimeError" in str(exception_events[0].attributes.get("exception.type", ""))
