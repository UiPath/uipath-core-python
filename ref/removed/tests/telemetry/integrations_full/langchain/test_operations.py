"""Operation type tests for LangChain integration.

Tests validate different LangChain operation types are instrumented correctly:
- LLM operations
- Chain operations
- Tool usage
- Retriever operations
- Token tracking
- Error handling
- Streaming
- Batch operations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from uipath.core.telemetry.integrations_full.langchain import (
    instrument_langchain,
    uninstrument_langchain,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


def test_llm_operation_creates_llm_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify LLM operations create LLM-type spans."""
    instrument_langchain()

    from langchain_core.language_models.fake import FakeListLLM

    llm = FakeListLLM(responses=["The answer is 42"])
    result = llm.invoke("What is the answer?")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify LLM span exists
    span_names = [s.name for s in spans]
    assert any("llm" in name.lower() or "fake" in name.lower() for name in span_names)

    # Verify result
    assert result == "The answer is 42"

    # Cleanup
    uninstrument_langchain()


def test_chain_operation_creates_chain_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify chain operations create chain-type spans."""
    instrument_langchain()

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    # Create LCEL chain
    prompt = ChatPromptTemplate.from_template("Answer: {question}")
    chain = prompt | RunnableLambda(lambda x: "42")

    result = chain.invoke({"question": "What is the answer?"})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify chain/runnable spans exist
    span_names = [s.name for s in spans]
    assert any(
        "chain" in name.lower() or "runnable" in name.lower() for name in span_names
    )

    # Verify result
    assert result == "42"

    # Cleanup
    uninstrument_langchain()


def test_tool_operation_creates_tool_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify tool operations create tool-type spans."""
    instrument_langchain()

    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Calculate a math expression."""
        return str(eval(expression))

    # Invoke tool
    result = calculator.invoke({"expression": "2+2"})

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify tool span exists
    span_names = [s.name for s in spans]
    assert any("tool" in name.lower() or "calculator" in name.lower() for name in span_names)

    # Verify result
    assert result == "4"

    # Cleanup
    uninstrument_langchain()


def test_retriever_operation_creates_retriever_span(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify retriever operations create retriever-type spans."""
    instrument_langchain()

    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class FakeRetriever(BaseRetriever):
        """Fake retriever for testing."""

        def _get_relevant_documents(self, query: str) -> list[Document]:
            """Return fake documents."""
            return [
                Document(page_content="Doc 1", metadata={"id": 1}),
                Document(page_content="Doc 2", metadata={"id": 2}),
            ]

    retriever = FakeRetriever()
    docs = retriever.invoke("test query")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify retriever span exists
    span_names = [s.name for s in spans]
    assert any("retriever" in name.lower() or "fake" in name.lower() for name in span_names)

    # Verify result
    assert len(docs) == 2

    # Cleanup
    uninstrument_langchain()


def test_sequential_chain_creates_hierarchy(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify sequential chains create proper span hierarchies."""
    instrument_langchain()

    from langchain_core.runnables import RunnableLambda

    # Create sequential chain
    step1 = RunnableLambda(lambda x: x + " step1")
    step2 = RunnableLambda(lambda x: x + " step2")
    step3 = RunnableLambda(lambda x: x + " step3")

    chain = step1 | step2 | step3
    result = chain.invoke("start")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify multiple spans (one per step + parent)
    assert len(spans) >= 3

    # Verify result
    assert result == "start step1 step2 step3"

    # Cleanup
    uninstrument_langchain()


def test_error_in_chain_recorded(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify errors in chains are recorded in spans."""
    instrument_langchain()

    from langchain_core.runnables import RunnableLambda

    def failing_step(x: str) -> str:
        """Step that raises an error."""
        raise ValueError("Intentional error for testing")

    chain = RunnableLambda(failing_step)

    # Invoke and expect error
    with pytest.raises(ValueError, match="Intentional error"):
        chain.invoke("test")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify error recorded in span
    error_spans = [s for s in spans if s.status.status_code.name == "ERROR"]
    assert len(error_spans) > 0

    # Cleanup
    uninstrument_langchain()


@pytest.mark.asyncio
async def test_async_chain_operations(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify async chain operations are instrumented."""
    instrument_langchain()

    from langchain_core.runnables import RunnableLambda

    async def async_step(x: str) -> str:
        """Async step."""
        return x + " async"

    chain = RunnableLambda(async_step)
    result = await chain.ainvoke("test")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert result == "test async"

    # Cleanup
    uninstrument_langchain()


def test_batch_operations(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify batch operations create separate spans."""
    instrument_langchain()

    from langchain_core.runnables import RunnableLambda

    chain = RunnableLambda(lambda x: x.upper())

    # Batch invoke
    results = chain.batch(["test1", "test2", "test3"])

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify results
    assert results == ["TEST1", "TEST2", "TEST3"]

    # Cleanup
    uninstrument_langchain()


def test_streaming_operations(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify streaming operations are instrumented."""
    instrument_langchain()

    from langchain_core.language_models.fake import FakeStreamingListLLM

    llm = FakeStreamingListLLM(responses=["streaming response"])

    # Stream
    chunks = []
    for chunk in llm.stream("test"):
        chunks.append(chunk)

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert len(chunks) > 0

    # Cleanup
    uninstrument_langchain()


def test_parallel_chain_execution(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify parallel chain execution creates proper spans."""
    instrument_langchain()

    from langchain_core.runnables import RunnableParallel

    # Create parallel chains
    parallel = RunnableParallel(
        {
            "upper": lambda x: x.upper(),
            "lower": lambda x: x.lower(),
            "length": lambda x: len(x),
        }
    )

    result = parallel.invoke("TeSt")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify result
    assert result["upper"] == "TEST"
    assert result["lower"] == "test"
    assert result["length"] == 4

    # Cleanup
    uninstrument_langchain()
