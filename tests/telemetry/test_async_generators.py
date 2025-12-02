"""Tests for async generator tracing support."""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

import pytest

from uipath.core.telemetry import get_client, traced
from uipath.core.telemetry.attributes import Attr
from uipath.core.telemetry.client import init_client
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )


class TestAsyncGeneratorBasic:
    """Test basic async generator tracing functionality."""

    @pytest.mark.asyncio
    async def test_simple_async_generator(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test tracing simple async generator function."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation")
        async def stream_data() -> AsyncIterator[int]:
            for i in range(5):
                yield i

        items = []
        async for item in stream_data():
            items.append(item)

        assert items == [0, 1, 2, 3, 4]

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "stream_data"
        assert span.attributes.get(Attr.Common.OPENINFERENCE_SPAN_KIND) == "GENERATION"

    @pytest.mark.asyncio
    async def test_async_generator_with_arguments(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test async generator with input arguments."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation")
        async def stream_range(start: int, end: int) -> AsyncIterator[int]:
            for i in range(start, end):
                yield i

        items = []
        async for item in stream_range(10, 15):
            items.append(item)

        assert items == [10, 11, 12, 13, 14]

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "stream_range"

    @pytest.mark.asyncio
    async def test_async_generator_custom_name(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test async generator with custom span name."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(name="custom_stream", kind="generation")
        async def stream_data() -> AsyncIterator[str]:
            for word in ["hello", "world"]:
                yield word

        items = []
        async for item in stream_data():
            items.append(item)

        assert items == ["hello", "world"]

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].name == "custom_stream"


class TestAsyncGeneratorPrivacy:
    """Test privacy controls for async generators."""

    @pytest.mark.asyncio
    async def test_hide_input(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test that hide_input redacts input data."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation", hide_input=True)
        async def stream_sensitive(api_key: str) -> AsyncIterator[str]:
            yield "result1"
            yield "result2"

        items = []
        async for item in stream_sensitive("secret-key-123"):
            items.append(item)

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        input_value = span.attributes.get(Attr.Common.INPUT_VALUE)
        assert input_value == "[REDACTED]"
        assert "secret-key-123" not in str(span.attributes)

    @pytest.mark.asyncio
    async def test_hide_output(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test that hide_output redacts output data."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation", hide_output=True)
        async def stream_private() -> AsyncIterator[str]:
            yield "private_data_1"
            yield "private_data_2"

        items = []
        async for item in stream_private():
            items.append(item)

        assert len(items) == 2

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        output_value = span.attributes.get(Attr.Common.OUTPUT_VALUE)
        assert output_value == "[REDACTED]"
        assert "private_data" not in str(span.attributes)

    @pytest.mark.asyncio
    async def test_hide_both(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test that both input and output can be hidden."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="tool", hide_input=True, hide_output=True)
        async def secure_stream(password: str) -> AsyncIterator[dict]:
            yield {"token": "secret"}

        items = []
        async for item in secure_stream("password123"):
            items.append(item)

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.attributes.get(Attr.Common.INPUT_VALUE) == "[REDACTED]"
        assert span.attributes.get(Attr.Common.OUTPUT_VALUE) == "[REDACTED]"


class TestAsyncGeneratorErrors:
    """Test error handling for async generators."""

    @pytest.mark.asyncio
    async def test_exception_recorded(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test that exceptions are recorded in spans."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation")
        async def failing_stream() -> AsyncIterator[int]:
            yield 1
            yield 2
            raise ValueError("Stream error")

        with pytest.raises(ValueError, match="Stream error"):
            async for _ in failing_stream():
                pass

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        events = span.events
        assert any("exception" in event.name.lower() for event in events)

    @pytest.mark.asyncio
    async def test_partial_consumption_error(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test error when async generator is partially consumed."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation")
        async def stream_data() -> AsyncIterator[int]:
            for i in range(10):
                if i == 5:
                    raise RuntimeError("Error at item 5")
                yield i

        items = []
        with pytest.raises(RuntimeError, match="Error at item 5"):
            async for item in stream_data():
                items.append(item)

        assert items == [0, 1, 2, 3, 4]

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1


class TestAsyncGeneratorNonRecording:
    """Test non-recording spans for async generators."""

    @pytest.mark.asyncio
    async def test_non_recording_span(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test that non-recording spans execute without errors."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation", recording=False)
        async def stream_data() -> AsyncIterator[str]:
            yield "item1"
            yield "item2"

        items = []
        async for item in stream_data():
            items.append(item)

        # Verify function executes correctly
        assert items == ["item1", "item2"]


class TestAsyncGeneratorStreaming:
    """Test streaming behavior of async generators."""

    @pytest.mark.asyncio
    async def test_streaming_llm_tokens(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test realistic LLM token streaming scenario."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation", hide_output=True)
        async def stream_llm_response(prompt: str) -> AsyncIterator[str]:
            """Simulate streaming LLM response."""
            tokens = ["Hello", " world", "!", " How", " are", " you", "?"]
            for token in tokens:
                yield token

        full_response = ""
        async for token in stream_llm_response("Say hello"):
            full_response += token

        assert full_response == "Hello world! How are you?"

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "stream_llm_response"
        assert span.attributes.get(Attr.Common.OPENINFERENCE_SPAN_KIND) == "GENERATION"
        assert span.attributes.get(Attr.Common.OUTPUT_VALUE) == "[REDACTED]"

    @pytest.mark.asyncio
    async def test_empty_async_generator(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test async generator that yields no items."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="generation")
        async def empty_stream() -> AsyncIterator[str]:
            return
            yield  # Make it a generator

        items = []
        async for item in empty_stream():
            items.append(item)

        assert items == []

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 1

        span = spans[0]
        assert span.name == "empty_stream"


class TestAsyncGeneratorIntegration:
    """Integration tests for async generators."""

    @pytest.mark.asyncio
    async def test_nested_async_generators(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test nested async generator spans."""
        config = TelemetryConfig(enable_console_export=True)
        init_client(config)

        @traced(kind="tool")
        async def fetch_chunks(source: str) -> AsyncIterator[str]:
            for i in range(3):
                yield f"{source}_{i}"

        @traced(kind="generation")
        async def process_stream(source: str) -> AsyncIterator[str]:
            async for chunk in fetch_chunks(source):
                yield chunk.upper()

        items = []
        async for item in process_stream("data"):
            items.append(item)

        assert items == ["DATA_0", "DATA_1", "DATA_2"]

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()
        assert len(spans) == 2

        parent_span = next((s for s in spans if s.name == "process_stream"), None)
        child_span = next((s for s in spans if s.name == "fetch_chunks"), None)

        assert parent_span is not None
        assert child_span is not None
        assert child_span.parent.span_id == parent_span.context.span_id

    @pytest.mark.asyncio
    async def test_async_generator_with_trace_context(self, in_memory_exporter: InMemorySpanExporter) -> None:
        """Test async generator within trace context."""
        config = TelemetryConfig(enable_console_export=True)
        client = init_client(config)
        tracer = client.get_tracer()

        with Trace(tracer, "workflow"):

            @traced(kind="generation")
            async def stream_in_execution() -> AsyncIterator[int]:
                for i in range(5):
                    yield i

            items = []
            async for item in stream_in_execution():
                items.append(item)

        assert items == [0, 1, 2, 3, 4]

        get_client().flush()
        spans = in_memory_exporter.get_finished_spans()

        # Verify both workflow and stream spans exist
        assert len(spans) >= 2

        stream_span = next((s for s in spans if s.name == "stream_in_execution"), None)
        assert stream_span is not None
