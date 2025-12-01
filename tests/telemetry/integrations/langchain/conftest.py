"""Shared fixtures for LangChain integration tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


@pytest.fixture
def telemetry_client(in_memory_exporter: InMemorySpanExporter) -> TelemetryClient:
    """Create TelemetryClient with in-memory exporter.

    Args:
        in_memory_exporter: In-memory exporter from parent conftest

    Returns:
        Configured TelemetryClient instance
    """
    from uipath.core.telemetry import init

    # Initialize with console export (uses in-memory via global provider)
    client = init(enable_console_export=True)
    return client


@pytest.fixture
def mock_llm() -> Mock:
    """Create mock LLM for testing.

    Returns:
        Mock LLM with invoke method
    """
    llm = Mock()
    llm.invoke = Mock(return_value="Mock response")
    llm.ainvoke = Mock(return_value="Mock async response")
    return llm


@pytest.fixture
def mock_chain() -> Mock:
    """Create mock chain for testing.

    Returns:
        Mock chain with invoke method
    """
    chain = Mock()
    chain.invoke = Mock(return_value={"output": "Mock chain output"})
    return chain
