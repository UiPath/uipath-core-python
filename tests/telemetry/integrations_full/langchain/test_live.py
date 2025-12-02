"""Live integration tests with real LLM providers.

These tests use real API keys from the environment to validate end-to-end
integration with actual LLM providers:
- OpenAI (GPT models)
- Anthropic (Claude models)

Tests are skipped if API keys are not available.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from uipath.core.telemetry import trace
from uipath.core.telemetry.integrations_full.langchain import (
    instrument_langchain,
    uninstrument_langchain,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


# Skip markers
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


@requires_openai
def test_live_openai_chat(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live OpenAI chat with execution_id tracking."""
    instrument_langchain()

    from langchain_openai import ChatOpenAI

    # Create OpenAI chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Execute with execution_id
    with trace("live_openai_test", execution_id="live-openai-123"):
        result = llm.invoke("Say 'test successful' and nothing else")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-openai-123"]
    assert len(exec_spans) >= 1

    # Verify OpenAI span exists
    span_names = [s.name for s in spans]
    assert any("openai" in name.lower() or "chat" in name.lower() for name in span_names)

    # Verify result
    assert result.content is not None
    assert len(result.content) > 0

    # Cleanup
    uninstrument_langchain()


@requires_anthropic
def test_live_anthropic_chat(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live Anthropic chat with execution_id tracking."""
    instrument_langchain()

    from langchain_anthropic import ChatAnthropic

    # Create Anthropic chat model
    llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)

    # Execute with execution_id
    with trace("live_anthropic_test", execution_id="live-anthropic-456"):
        result = llm.invoke("Say 'test successful' and nothing else")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-anthropic-456"]
    assert len(exec_spans) >= 1

    # Verify Anthropic span exists
    span_names = [s.name for s in spans]
    assert any("anthropic" in name.lower() or "chat" in name.lower() for name in span_names)

    # Verify result
    assert result.content is not None
    assert len(result.content) > 0

    # Cleanup
    uninstrument_langchain()


@requires_openai
def test_live_openai_chain(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live OpenAI chain with multiple steps."""
    instrument_langchain()

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    # Create chain
    prompt = ChatPromptTemplate.from_template("What is {number} + {number}? Just give the number.")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    chain = prompt | llm | parser

    # Execute with execution_id
    with trace("live_chain_test", execution_id="live-chain-789"):
        result = chain.invoke({"number": 5})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-chain-789"]
    assert len(exec_spans) >= 1

    # Verify multiple spans (prompt, llm, parser)
    assert len(spans) >= 3

    # Verify result
    assert result is not None

    # Cleanup
    uninstrument_langchain()


@requires_openai
def test_live_openai_streaming(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live OpenAI streaming with telemetry."""
    instrument_langchain()

    from langchain_openai import ChatOpenAI

    # Create OpenAI chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

    # Stream with execution_id
    with trace("live_streaming_test", execution_id="live-stream-999"):
        chunks = []
        for chunk in llm.stream("Count from 1 to 3, separated by spaces"):
            chunks.append(chunk.content)

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-stream-999"]
    assert len(exec_spans) >= 1

    # Verify streaming worked
    assert len(chunks) > 0

    # Cleanup
    uninstrument_langchain()


@requires_openai
@pytest.mark.asyncio
async def test_live_openai_async(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live OpenAI async operations with telemetry."""
    instrument_langchain()

    from langchain_openai import ChatOpenAI

    # Create OpenAI chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Async invoke with execution_id
    with trace("live_async_test", execution_id="live-async-111"):
        result = await llm.ainvoke("Say 'async test successful' and nothing else")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-async-111"]
    assert len(exec_spans) >= 1

    # Verify result
    assert result.content is not None
    assert len(result.content) > 0

    # Cleanup
    uninstrument_langchain()


@requires_openai
def test_live_openai_with_tools(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live OpenAI with tool calling."""
    instrument_langchain()

    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    def get_weather(location: str) -> str:
        """Get the weather for a location."""
        return f"The weather in {location} is sunny and 72°F"

    # Create OpenAI chat model with tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools([get_weather])

    # Execute with execution_id
    with trace("live_tools_test", execution_id="live-tools-222"):
        result = llm_with_tools.invoke("What's the weather in San Francisco?")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-tools-222"]
    assert len(exec_spans) >= 1

    # Verify result has tool calls
    assert result is not None

    # Cleanup
    uninstrument_langchain()


@requires_openai
def test_live_openai_batch(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live OpenAI batch operations."""
    instrument_langchain()

    from langchain_openai import ChatOpenAI

    # Create OpenAI chat model
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Batch invoke with execution_id
    with trace("live_batch_test", execution_id="live-batch-333"):
        results = llm.batch([
            "Say 'one'",
            "Say 'two'",
            "Say 'three'",
        ])

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-batch-333"]
    assert len(exec_spans) >= 1

    # Verify results
    assert len(results) == 3
    for result in results:
        assert result.content is not None

    # Cleanup
    uninstrument_langchain()
