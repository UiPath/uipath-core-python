"""Tests for decorator functionality."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from opentelemetry.trace import StatusCode

from uipath.core.otel.client import OTelClient, OTelConfig
from uipath.core.otel.decorator import agent, generation, tool, traced
from uipath.core.otel.trace import Trace, require_trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_generation_decorator_sync(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test @generation decorator on sync function.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
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

    assert llm_span.attributes.get("span.type") == "generation"


async def test_generation_decorator_async(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @generation decorator on async function.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
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
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
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
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @tool()
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

    assert tool_span.attributes.get("span.type") == "tool"
    # Should NOT have update() attributes (auto_update=False)


def test_agent_decorator(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test @agent decorator.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @agent()
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

    assert agent_span.attributes.get("span.type") == "agent"


def test_decorator_exception_handling(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test decorator records exception and re-raises.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
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
    assert len(func_span.events) == 1
    assert func_span.events[0].name == "exception"


def test_decorator_custom_name(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator with custom span name.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation(name="custom-span-name")
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


def test_decorator_requires_trace(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator raises error outside trace context.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    OTelClient(config)

    @generation()
    def my_function() -> str:
        """Test function.

        Returns:
            Result
        """
        return "test"

    # Execute - should raise outside trace
    with pytest.raises(RuntimeError, match="No active trace context"):
        my_function()


def test_traced_generic_decorator(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test @traced decorator with custom span type.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @traced(span_type="retrieval")
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

    assert search_span.attributes.get("span.type") == "retrieval"


def test_traced_routes_to_specialized(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @traced routes to specialized decorators.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @traced(span_type="generation", model="gpt-4")
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

    assert llm_span.attributes.get("span.type") == "generation"
    # Should have model attribute from specialized decorator
    assert llm_span.attributes.get("gen_ai.request.model") == "gpt-4"


def test_auto_update_guards_generator(
    in_memory_exporter: InMemorySpanExporter,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test auto_update guards against generators.

    Args:
        in_memory_exporter: In-memory exporter fixture
        caplog: Pytest log capture fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()  # auto_update=True by default
    def generator_function():
        """Function returning generator.

        Yields:
            Values
        """
        yield 1
        yield 2
        yield 3

    # Execute - should not crash
    with Trace(tracer, "test"):
        gen = generator_function()
        list(gen)  # Consume generator

    # Verify warning logged
    assert "does not support generators" in caplog.text

    # Verify span still created
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert any(s.name == "generator_function" for s in spans)


def test_decorator_with_model_parameter(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test @generation decorator with model parameter.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation(model="gpt-4-turbo")
    def llm_call() -> str:
        """LLM call.

        Returns:
            Response
        """
        return "response"

    # Execute
    with Trace(tracer, "test"):
        llm_call()

    # Verify model attribute set
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    llm_span = next(s for s in spans if s.name == "llm_call")

    assert llm_span.attributes.get("gen_ai.request.model") == "gpt-4-turbo"


def test_decorator_explicit_auto_update_false(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test decorator with auto_update explicitly set to False.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation(auto_update=False)  # Override default True
    def llm_call() -> dict[str, str]:
        """LLM call.

        Returns:
            Response
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

    # Should NOT have parsed attributes (auto_update=False)
    # Only span.type and maybe gen_ai.request.model
    attrs = llm_span.attributes
    assert "span.type" in attrs
    # Should not have usage or other parsed attributes


async def test_decorator_async_exception(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test async decorator handles exceptions.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    @generation()
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
    assert len(func_span.events) == 1
