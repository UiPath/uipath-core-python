"""Test utilities for telemetry validation.

Provides helper functions for testing telemetry functionality.
These utilities are NOT part of the production API.
"""

from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )


def get_execution_spans(
    exporter: "InMemorySpanExporter",
    execution_id: str,
) -> List["ReadableSpan"]:
    """Filter spans by execution ID (testing only).

    Helper function to retrieve spans for a specific execution ID from
    an InMemorySpanExporter. This is the testing equivalent of the old
    UiPathTraceManager.get_execution_spans() method.

    Args:
        exporter: InMemorySpanExporter instance from test setup
        execution_id: Execution ID to filter by

    Returns:
        List of spans with matching execution.id attribute

    Example:
        >>> from opentelemetry.sdk.trace.export import InMemorySpanExporter
        >>> from uipath.core.telemetry import get_telemetry_client, TelemetryConfig
        >>> from tests.telemetry.utils import get_execution_spans
        >>>
        >>> # Test setup
        >>> exporter = InMemorySpanExporter()
        >>> config = TelemetryConfig(endpoint=None)  # Console exporter
        >>> client = get_telemetry_client(config)
        >>>
        >>> # Execute code
        >>> with client.start_as_current_span(
        ...     "workflow",
        ...     execution_id="exec-123"
        ... ):
        ...     do_work()
        >>> client.flush()
        >>>
        >>> # Validate spans
        >>> spans = get_execution_spans(exporter, "exec-123")
        >>> assert len(spans) > 0
        >>> assert all(
        ...     s.attributes.get("execution.id") == "exec-123"
        ...     for s in spans
        ... )

    Note:
        This function is for testing only and should not be used in
        production code. Production code should export spans to a real
        backend (OTLP, Console, etc.).
    """
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get("execution.id") == execution_id
    ]


def get_spans_by_type(
    exporter: "InMemorySpanExporter",
    span_type: str,
) -> List["ReadableSpan"]:
    """Filter spans by semantic type (testing only).

    Args:
        exporter: InMemorySpanExporter instance
        span_type: Semantic span type (e.g., "automation", "generation")

    Returns:
        List of spans with matching span_type attribute

    Example:
        >>> from tests.telemetry.utils import get_spans_by_type
        >>> spans = get_spans_by_type(exporter, "automation")
        >>> assert all(
        ...     s.attributes.get("span_type") == "automation"
        ...     for s in spans
        ... )
    """
    return [
        span
        for span in exporter.get_finished_spans()
        if span.attributes and span.attributes.get("span_type") == span_type
    ]


def assert_span_hierarchy(
    spans: List["ReadableSpan"],
    expected_structure: Dict[str, List[str]],
) -> None:
    """Assert that spans form expected parent-child hierarchy.

    Validates that spans are correctly nested according to the expected
    structure. Useful for testing trace hierarchy correctness.

    Args:
        spans: List of spans to validate
        expected_structure: Dict mapping span names to expected children

    Raises:
        AssertionError: If hierarchy doesn't match expected structure

    Example:
        >>> from tests.telemetry.utils import assert_span_hierarchy
        >>> spans = exporter.get_finished_spans()
        >>> assert_span_hierarchy(
        ...     spans,
        ...     {
        ...         "workflow": ["process_invoice", "generate_report"],
        ...         "process_invoice": ["validate_data", "save_to_db"],
        ...     }
        ... )
    """
    # Build parent-child relationships
    children_by_parent: Dict[str, List[str]] = {}

    for span in spans:
        parent_id = span.parent.span_id if span.parent else None
        if parent_id:
            parent_span = next(
                (s for s in spans if s.context.span_id == parent_id), None
            )
            if parent_span:
                children_by_parent.setdefault(parent_span.name, []).append(span.name)

    # Validate structure
    for parent_name, expected_children in expected_structure.items():
        actual_children = children_by_parent.get(parent_name, [])
        assert set(actual_children) == set(expected_children), (
            f"Parent '{parent_name}' expected children {expected_children}, "
            f"got {actual_children}"
        )


def get_span_by_name(
    exporter: "InMemorySpanExporter",
    name: str,
) -> "ReadableSpan":
    """Get first span matching name (testing only).

    Args:
        exporter: InMemorySpanExporter instance
        name: Span name to search for

    Returns:
        First span with matching name

    Raises:
        ValueError: If no span with given name found

    Example:
        >>> from tests.telemetry.utils import get_span_by_name
        >>> span = get_span_by_name(exporter, "workflow")
        >>> assert span.name == "workflow"
    """
    for span in exporter.get_finished_spans():
        if span.name == name:
            return span
    raise ValueError(f"No span found with name: {name}")
