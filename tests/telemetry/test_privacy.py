"""Tests for simplified privacy enforcement via hide flags."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uipath.core.telemetry import traced
from uipath.core.telemetry.client import init_client
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_record_input_hide_flag(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test record_input with hide=True redacts input.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "privacy-test") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.record_input({"sensitive": "data"}, hide=True)
            obs.record_output({"result": "ok"}, hide=False)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Input should be redacted
    assert gen_span.attributes.get("input.value") == "[REDACTED]"

    # Output should be visible
    assert '"result"' in gen_span.attributes.get("output.value", "")


def test_record_output_hide_flag(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test record_output with hide=True redacts output.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "privacy-test") as trace_ctx:
        with trace_ctx.span("test", kind="generation") as obs:
            obs.record_input({"query": "search"}, hide=False)
            obs.record_output({"pii": "sensitive"}, hide=True)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Input should be visible
    assert '"query"' in gen_span.attributes.get("input.value", "")

    # Output should be redacted
    assert gen_span.attributes.get("output.value") == "[REDACTED]"


def test_both_hide_flags(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test hiding both input and output.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "privacy-test") as trace_ctx:
        with trace_ctx.span("test", kind="tool") as obs:
            obs.record_input({"card": "1234-5678"}, hide=True)
            obs.record_output({"transaction": "approved"}, hide=True)

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Both should be redacted
    assert gen_span.attributes.get("input.value") == "[REDACTED]"
    assert gen_span.attributes.get("output.value") == "[REDACTED]"


def test_decorator_hide_input(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator hide_input flag.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    init_client(config)

    # Execute
    @traced(kind="tool", hide_input=True)
    def authenticate(api_key: str) -> dict:
        return {"authenticated": True}

    authenticate("secret-key-123")

    # Verify
    spans = in_memory_exporter.get_finished_spans()
    auth_span = next(s for s in spans if s.name == "authenticate")

    # Input should be redacted
    assert auth_span.attributes.get("input.value") == "[REDACTED]"

    # Output should be visible
    assert '"authenticated"' in auth_span.attributes.get("output.value", "")


def test_decorator_hide_output(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator hide_output flag.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    init_client(config)

    # Execute
    @traced(kind="generation", hide_output=True)
    def extract_pii(document: str) -> dict:
        return {"name": "John Doe", "ssn": "123-45-6789"}

    extract_pii("document-id-456")

    # Verify
    spans = in_memory_exporter.get_finished_spans()
    extract_span = next(s for s in spans if s.name == "extract_pii")

    # Input should be visible
    assert '"document"' in extract_span.attributes.get("input.value", "")

    # Output should be redacted
    assert extract_span.attributes.get("output.value") == "[REDACTED]"


def test_decorator_hide_both(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test decorator hiding both input and output.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    init_client(config)

    # Execute
    @traced(kind="tool", hide_input=True, hide_output=True)
    def process_payment(card_number: str, amount: float) -> dict:
        return {"transaction_id": "txn_123", "status": "approved"}

    process_payment("4532-1111-2222-3333", 99.99)

    # Verify
    spans = in_memory_exporter.get_finished_spans()
    payment_span = next(s for s in spans if s.name == "process_payment")

    # Both should be redacted
    assert payment_span.attributes.get("input.value") == "[REDACTED]"
    assert payment_span.attributes.get("output.value") == "[REDACTED]"


def test_no_hide_flags_default(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test that by default (no hide flags), everything is visible.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "no-privacy-test") as trace_ctx:
        with trace_ctx.span("test", kind="tool") as obs:
            obs.record_input({"data": "visible input"})
            obs.record_output({"result": "visible output"})

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Both should be visible
    assert '"data"' in gen_span.attributes.get("input.value", "")
    assert '"result"' in gen_span.attributes.get("output.value", "")
