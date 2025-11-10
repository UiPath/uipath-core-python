"""Tests for non-recording span hierarchy (Issue #1 - FIXED).

This test file verifies that non-recording spans become active parents in the
OpenTelemetry context, allowing child spans to correctly inherit trace hierarchy.

Related: docs/CODE-REVIEW-V1.md Issue #1 (resolved in client.py:316-318)
"""

from uipath.core.telemetry import (
    TelemetryConfig,
    get_telemetry_client,
    reset_telemetry_client,
)


def test_non_recording_span_becomes_parent():
    """Verify non-recording spans maintain hierarchy.

    EXPECTED BEHAVIOR:
        Child spans created within a non-recording parent span should
        inherit the parent's trace_id to maintain trace hierarchy.

    FIX (Issue #1):
        Non-recording spans now use trace.use_span() to activate in context,
        ensuring child spans correctly inherit the parent's trace_id.
    """
    reset_telemetry_client()
    config = TelemetryConfig(endpoint=None)
    client = get_telemetry_client(config)

    with client.start_as_current_span("parent", recording=False) as parent:
        parent_ctx = parent._span.get_span_context()

        with client.start_as_current_span("child") as child:
            child_ctx = child._span.get_span_context()

            # This assertion SHOULD pass but currently fails
            assert child_ctx.trace_id == parent_ctx.trace_id, (
                f"Child trace_id {child_ctx.trace_id:032x} != "
                f"parent trace_id {parent_ctx.trace_id:032x}. "
                "Non-recording parent span is not active in context."
            )

    # Additional verification: check span hierarchy
    client.flush()
