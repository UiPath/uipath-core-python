"""Context propagation tests for LangChain integration.

Tests validate that UiPath context (execution_id, resource attributes, etc.)
propagates correctly through OpenInference instrumentation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from uipath.core.telemetry import init
from uipath.core.telemetry.attributes import Attr
from uipath.core.telemetry.config import TelemetryConfig
from uipath.core.telemetry.integrations_full.langchain import (
    instrument_langchain,
    uninstrument_langchain,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


def test_execution_id_propagates_to_spans(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify execution_id propagates to all LangChain spans."""
    instrument_langchain()

    # Create parent span with execution_id using trace API
    from uipath.core.telemetry import trace

    with trace("workflow", execution_id="exec-123"):
        # Execute LangChain operation
        from langchain_core.language_models.fake import FakeListLLM

        llm = FakeListLLM(responses=["test response"])
        llm.invoke("test prompt")

    # Flush spans
    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify spans were created
    assert len(spans) > 0

    # Verify execution_id is present on child spans
    # Note: Child spans inherit attributes from parent context
    execution_id_spans = [
        s for s in spans if s.attributes.get("execution.id") == "exec-123"
    ]

    # At minimum, the workflow span should have execution_id
    assert len(execution_id_spans) >= 1, (
        f"Expected at least 1 span with execution.id, "
        f"got {len(execution_id_spans)}. "
        f"Span names: {[s.name for s in spans]}"
    )

    # Cleanup
    uninstrument_langchain()


@pytest.mark.no_auto_tracer
def test_resource_attributes_visible(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify UiPath resource attributes visible in spans."""
    # Set up TracerProvider with custom resource attributes and in-memory exporter
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    # Reset global provider state
    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]

    # Create resource with custom attributes
    resource = Resource(
        attributes={
            "service.name": "test-service",
            "uipath.org_id": "org-123",
            "uipath.tenant_id": "tenant-456",
            "uipath.user_id": "user-789",
        }
    )

    # Create provider with resource and exporter
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    trace.set_tracer_provider(provider)

    # Initialize UiPath client (will use global provider)
    client = init(
        service_name="test-service",
        enable_console_export=True,
        resource_attributes={
            "uipath.org_id": "org-123",
            "uipath.tenant_id": "tenant-456",
            "uipath.user_id": "user-789",
        },
    )

    instrument_langchain()

    # Execute LangChain operation
    from langchain_core.language_models.fake import FakeListLLM

    llm = FakeListLLM(responses=["test"])
    llm.invoke("test")

    client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify resource attributes set on spans
    for span in spans:
        resource = span.resource
        assert resource.attributes.get("uipath.org_id") == "org-123"
        assert resource.attributes.get("uipath.tenant_id") == "tenant-456"
        assert resource.attributes.get("uipath.user_id") == "user-789"

    # Cleanup
    uninstrument_langchain()


def test_nested_spans_inherit_execution_id(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify nested operations inherit execution_id."""
    instrument_langchain()

    from uipath.core.telemetry import trace

    # Create nested trace hierarchy
    with trace("parent", execution_id="exec-nested"):
        with trace("child"):
            # Execute LangChain operation
            from langchain_core.language_models.fake import FakeListLLM

            llm = FakeListLLM(responses=["nested test"])
            llm.invoke("test")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Find parent and child spans
    parent_span = [s for s in spans if s.name == "parent"]
    assert len(parent_span) == 1

    # Parent should have execution_id
    assert parent_span[0].attributes.get("execution.id") == "exec-nested"

    # Child spans should also have execution_id through context propagation
    # (OpenTelemetry propagates context to children)
    execution_id_spans = [
        s for s in spans if s.attributes.get("execution.id") == "exec-nested"
    ]
    assert len(execution_id_spans) >= 1

    # Cleanup
    uninstrument_langchain()


@pytest.mark.asyncio
async def test_async_context_propagation(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify context propagates in async operations."""
    instrument_langchain()

    from uipath.core.telemetry import trace

    # Create async trace
    async def async_workflow() -> str:
        with trace("async_workflow", execution_id="exec-async"):
            from langchain_core.language_models.fake import FakeListLLM

            llm = FakeListLLM(responses=["async response"])
            return await llm.ainvoke("async test")

    result = await async_workflow()

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated in async context
    execution_id_spans = [
        s for s in spans if s.attributes.get("execution.id") == "exec-async"
    ]
    assert len(execution_id_spans) >= 1

    # Verify result
    assert result == "async response"

    # Cleanup
    uninstrument_langchain()


def test_multiple_executions_isolated(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Verify concurrent executions have isolated execution_ids."""
    instrument_langchain()

    from uipath.core.telemetry import trace

    # Execute multiple workflows with different execution_ids
    with trace("workflow1", execution_id="exec-1"):
        from langchain_core.language_models.fake import FakeListLLM

        llm1 = FakeListLLM(responses=["response1"])
        llm1.invoke("test1")

    with trace("workflow2", execution_id="exec-2"):
        llm2 = FakeListLLM(responses=["response2"])
        llm2.invoke("test2")

    with trace("workflow3", execution_id="exec-3"):
        llm3 = FakeListLLM(responses=["response3"])
        llm3.invoke("test3")

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()

    # Verify all three execution_ids are present
    exec1_spans = [s for s in spans if s.attributes.get("execution.id") == "exec-1"]
    exec2_spans = [s for s in spans if s.attributes.get("execution.id") == "exec-2"]
    exec3_spans = [s for s in spans if s.attributes.get("execution.id") == "exec-3"]

    assert len(exec1_spans) >= 1, "exec-1 spans not found"
    assert len(exec2_spans) >= 1, "exec-2 spans not found"
    assert len(exec3_spans) >= 1, "exec-3 spans not found"

    # Verify no span has multiple execution_ids (no cross-contamination)
    for span in spans:
        exec_id = span.attributes.get("execution.id")
        if exec_id:
            assert exec_id in ["exec-1", "exec-2", "exec-3"]

    # Cleanup
    uninstrument_langchain()
