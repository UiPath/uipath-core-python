"""Tests for privacy enforcement (CRITICAL fix validation)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from uipath.core.otel.client import OTelClient, OTelConfig
from uipath.core.otel.trace import Trace

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


def test_input_redaction(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test input attributes are redacted when privacy config enabled.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup with privacy config
    config = OTelConfig(
        mode="dev",
        privacy={"redact_inputs": True},
    )
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "privacy-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("user_input", "sensitive data")
            obs.set_attribute("input_text", "secret information")
            obs.set_attribute("output_text", "should not be redacted")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Input attributes should be redacted
    assert gen_span.attributes.get("user_input") == "[REDACTED]"
    assert gen_span.attributes.get("input_text") == "[REDACTED]"

    # Output attributes should NOT be redacted
    assert gen_span.attributes.get("output_text") == "should not be redacted"


def test_output_redaction(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test output attributes are redacted when privacy config enabled.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup with privacy config
    config = OTelConfig(
        mode="dev",
        privacy={"redact_outputs": True},
    )
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "privacy-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("model_output", "sensitive response")
            obs.set_attribute("output_data", "secret result")
            obs.set_attribute("input_data", "should not be redacted")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Output attributes should be redacted
    assert gen_span.attributes.get("model_output") == "[REDACTED]"
    assert gen_span.attributes.get("output_data") == "[REDACTED]"

    # Input attributes should NOT be redacted
    assert gen_span.attributes.get("input_data") == "should not be redacted"


def test_truncation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test long attribute values are truncated.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup with truncation config
    config = OTelConfig(
        mode="dev",
        privacy={"max_attribute_length": 100},
    )
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute - set attribute with 500 characters
    long_value = "x" * 500

    with Trace(tracer, "truncation-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("long_attribute", long_value)
            obs.set_attribute("short_attribute", "short")

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Long attribute should be truncated
    long_attr = gen_span.attributes.get("long_attribute")
    assert long_attr is not None
    assert len(long_attr) == 100 + len("...[truncated]")
    assert long_attr.endswith("...[truncated]")
    assert long_attr.startswith("x" * 100)

    # Short attribute should not be truncated
    assert gen_span.attributes.get("short_attribute") == "short"


def test_privacy_config_integration(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test privacy config is properly integrated from client to observation.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    privacy_config = {
        "redact_inputs": True,
        "redact_outputs": True,
        "max_attribute_length": 50,
    }
    config = OTelConfig(mode="dev", privacy=privacy_config)
    client = OTelClient(config)

    # Verify client stores privacy config
    assert client.get_privacy_config() == privacy_config

    # Verify observation uses client's privacy config
    tracer = client.get_tracer()
    with Trace(tracer, "integration-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("input_data", "should be redacted")
            obs.set_attribute("output_data", "also redacted")
            obs.set_attribute("long_attr", "y" * 100)

    # Verify all privacy rules applied
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    assert gen_span.attributes.get("input_data") == "[REDACTED]"
    assert gen_span.attributes.get("output_data") == "[REDACTED]"

    long_attr = gen_span.attributes.get("long_attr")
    assert long_attr is not None
    assert len(long_attr) == 50 + len("...[truncated]")


def test_no_privacy_config(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test attributes not redacted when no privacy config.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup without privacy config
    config = OTelConfig(mode="dev")
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "no-privacy-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("input_data", "visible input")
            obs.set_attribute("output_data", "visible output")
            obs.set_attribute("long_attr", "x" * 500)

    # Verify nothing redacted or truncated
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    assert gen_span.attributes.get("input_data") == "visible input"
    assert gen_span.attributes.get("output_data") == "visible output"
    assert gen_span.attributes.get("long_attr") == "x" * 500


def test_case_insensitive_privacy(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test privacy rules are case-insensitive.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(
        mode="dev",
        privacy={"redact_inputs": True, "redact_outputs": True},
    )
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute - test various case combinations
    with Trace(tracer, "case-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("INPUT", "test1")
            obs.set_attribute("input", "test2")
            obs.set_attribute("Input", "test3")
            obs.set_attribute("user_input", "test4")
            obs.set_attribute("input_data", "test5")
            obs.set_attribute("OUTPUT", "test6")
            obs.set_attribute("output", "test7")
            obs.set_attribute("Output", "test8")
            obs.set_attribute("model_output", "test9")
            obs.set_attribute("output_text", "test10")

    # Verify all variations redacted
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # All input variations should be redacted
    assert gen_span.attributes.get("INPUT") == "[REDACTED]"
    assert gen_span.attributes.get("input") == "[REDACTED]"
    assert gen_span.attributes.get("Input") == "[REDACTED]"
    assert gen_span.attributes.get("user_input") == "[REDACTED]"
    assert gen_span.attributes.get("input_data") == "[REDACTED]"

    # All output variations should be redacted
    assert gen_span.attributes.get("OUTPUT") == "[REDACTED]"
    assert gen_span.attributes.get("output") == "[REDACTED]"
    assert gen_span.attributes.get("Output") == "[REDACTED]"
    assert gen_span.attributes.get("model_output") == "[REDACTED]"
    assert gen_span.attributes.get("output_text") == "[REDACTED]"


def test_privacy_with_dict_serialization(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test privacy applies before JSON serialization of dicts.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup
    config = OTelConfig(
        mode="dev",
        privacy={"redact_inputs": True},
    )
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute - set dict value (will be JSON serialized)
    with Trace(tracer, "dict-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("input_dict", {"key": "value"})
            obs.set_attribute("other_dict", {"key": "value"})

    # Verify
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    # Input dict should be redacted (before serialization)
    assert gen_span.attributes.get("input_dict") == "[REDACTED]"

    # Other dict should be serialized normally
    assert gen_span.attributes.get("other_dict") == '{"key": "value"}'


def test_privacy_empty_config(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test empty privacy config dict doesn't break anything.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Setup with empty privacy config
    config = OTelConfig(mode="dev", privacy={})
    client = OTelClient(config)
    tracer = client.get_tracer()

    # Execute
    with Trace(tracer, "empty-config-test") as trace_ctx:
        with trace_ctx.generation("test") as obs:
            obs.set_attribute("input_data", "visible")
            obs.set_attribute("output_data", "visible")

    # Verify nothing redacted
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    gen_span = next(s for s in spans if s.name == "test")

    assert gen_span.attributes.get("input_data") == "visible"
    assert gen_span.attributes.get("output_data") == "visible"
