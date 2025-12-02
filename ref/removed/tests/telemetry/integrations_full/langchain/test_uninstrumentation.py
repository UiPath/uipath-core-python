"""Tests for LangChain uninstrumentation verification.

This module tests that uninstrumentation actually removes instrumentation,
addressing the critical bug identified in the code review where creating a new
instrumentor instance in uninstrument_langchain() made it a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake import FakeListLLM

from uipath.core.telemetry import init
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.integrations_full.langchain import (
    instrument_langchain,
    uninstrument_langchain,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter

    from uipath.core.telemetry.client import TelemetryClient


def test_uninstrumentation_actually_removes_instrumentation(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that uninstrument_langchain() actually removes instrumentation.

    This test verifies the fix for the critical bug where uninstrument_langchain()
    created a new instrumentor instance instead of reusing the tracked one.
    """
    # Setup
    llm = FakeListLLM(responses=["test response"])

    # Phase 1: Instrument and verify spans are created
    instrument_langchain()
    result1 = llm.invoke("test input")
    telemetry_client.flush()

    spans_instrumented = in_memory_exporter.get_finished_spans()
    assert len(spans_instrumented) > 0, "Should have spans when instrumented"
    assert result1 == "test response"

    # Clear spans for next phase
    in_memory_exporter.clear()

    # Phase 2: Uninstrument and verify spans are NOT created
    uninstrument_langchain()
    result2 = llm.invoke("test input 2")
    telemetry_client.flush()

    spans_uninstrumented = in_memory_exporter.get_finished_spans()
    assert len(spans_uninstrumented) == 0, "Should have NO spans after uninstrumentation"
    assert result2 == "test response"


def test_reinstrumentation_after_uninstrumentation(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that instrumentation can be re-enabled after uninstrumentation.

    Verifies the complete lifecycle: instrument -> uninstrument -> re-instrument.
    """
    llm = FakeListLLM(responses=["response1", "response2", "response3"])

    # Phase 1: Initial instrumentation
    instrument_langchain()
    llm.invoke("input1")
    telemetry_client.flush()
    spans_phase1 = len(in_memory_exporter.get_finished_spans())
    assert spans_phase1 > 0, "Phase 1: Should have spans"
    in_memory_exporter.clear()

    # Phase 2: Uninstrumentation
    uninstrument_langchain()
    llm.invoke("input2")
    telemetry_client.flush()
    spans_phase2 = len(in_memory_exporter.get_finished_spans())
    assert spans_phase2 == 0, "Phase 2: Should have NO spans after uninstrumentation"
    in_memory_exporter.clear()

    # Phase 3: Re-instrumentation
    instrument_langchain()
    llm.invoke("input3")
    telemetry_client.flush()
    spans_phase3 = len(in_memory_exporter.get_finished_spans())
    assert spans_phase3 > 0, "Phase 3: Should have spans after re-instrumentation"


def test_multiple_uninstrumentation_calls_are_safe(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that calling uninstrument_langchain() multiple times is safe.

    Should not raise errors even when called multiple times.
    """
    # Reset state first
    uninstrument_langchain()
    in_memory_exporter.clear()

    llm = FakeListLLM(responses=["test", "test2"])

    # Instrument
    instrument_langchain()
    llm.invoke("test")
    telemetry_client.flush()
    assert len(in_memory_exporter.get_finished_spans()) > 0
    in_memory_exporter.clear()

    # Uninstrument multiple times - should not raise
    uninstrument_langchain()
    uninstrument_langchain()
    uninstrument_langchain()

    # Verify still uninstrumented
    llm.invoke("test2")
    telemetry_client.flush()
    assert len(in_memory_exporter.get_finished_spans()) == 0


def test_uninstrumentation_without_prior_instrumentation(
    telemetry_client: TelemetryClient,
) -> None:
    """Test that uninstrument_langchain() is safe to call without prior instrumentation.

    Should not raise errors when called before instrument_langchain().
    """
    # Should not raise
    uninstrument_langchain()
    uninstrument_langchain()


def test_instrumentation_idempotency(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test that calling instrument_langchain() multiple times is idempotent.

    Multiple calls should not create multiple instrumentors or cause issues.
    """
    # Reset first
    uninstrument_langchain()
    in_memory_exporter.clear()

    llm = FakeListLLM(responses=["test1", "test2", "test3"])

    # Instrument multiple times - should be safe
    instrument_langchain()
    instrument_langchain()
    instrument_langchain()

    # Should still work correctly
    llm.invoke("test")
    telemetry_client.flush()

    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Cleanup
    uninstrument_langchain()


@pytest.mark.no_auto_tracer
def test_warning_logged_when_falling_back_to_global_provider() -> None:
    """Test that a warning is logged when falling back to global TracerProvider.

    Verifies that users are informed when they're not using UiPath client.
    """
    from unittest.mock import patch

    # Reset state
    uninstrument_langchain()

    # Capture log messages
    with patch("uipath.core.telemetry.integrations_full.langchain.logger") as mock_logger:
        with patch("uipath.core.telemetry.integrations_full.langchain.get_client") as mock_get_client:
            mock_get_client.side_effect = RuntimeError("Client not initialized")

            # Instrument (should trigger fallback and log warning)
            instrument_langchain()

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "UiPath telemetry client not available" in warning_msg
            assert "falling back" in warning_msg.lower()
            assert "Resource attributes" in warning_msg

    # Cleanup
    uninstrument_langchain()
