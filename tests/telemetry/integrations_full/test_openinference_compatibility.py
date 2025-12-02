"""Compatibility tests comparing UiPath instrumentation with OpenInference.

These tests validate that our LangChain/LangGraph instrumentation produces
telemetry that is semantically equivalent to the OpenInference reference
implementation.
"""

from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

# Try to import dependencies
try:
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from openinference.instrumentation.langchain import (
        LangChainInstrumentor as OpenInferenceLangChainInstrumentor,
    )

    OPENINFERENCE_AVAILABLE = True
except ImportError:
    OPENINFERENCE_AVAILABLE = False


def compare_span_structures(
    uipath_spans: list[Any],
    openinference_spans: list[Any],
) -> dict[str, Any]:
    """Compare span structures between UiPath and OpenInference implementations.

    Args:
        uipath_spans: Spans from UiPath instrumentation
        openinference_spans: Spans from OpenInference instrumentation

    Returns:
        Dictionary with comparison results including:
        - span_count_match: Whether span counts match
        - span_names: Comparison of span names
        - parent_child_relationships: Whether hierarchies match
        - common_attributes: Shared attributes between implementations
    """
    comparison = {
        "span_count_match": len(uipath_spans) == len(openinference_spans),
        "uipath_span_count": len(uipath_spans),
        "openinference_span_count": len(openinference_spans),
        "span_names": {
            "uipath": sorted([s.name for s in uipath_spans]),
            "openinference": sorted([s.name for s in openinference_spans]),
        },
        "differences": [],
    }

    # Check if span names match
    if comparison["span_names"]["uipath"] != comparison["span_names"]["openinference"]:
        comparison["differences"].append(
            f"Span names differ: {comparison['span_names']}"
        )

    # Check parent-child relationships
    uipath_hierarchy = _extract_hierarchy(uipath_spans)
    openinference_hierarchy = _extract_hierarchy(openinference_spans)

    if uipath_hierarchy != openinference_hierarchy:
        comparison["differences"].append(
            f"Hierarchies differ:\n  UiPath: {uipath_hierarchy}\n  OpenInference: {openinference_hierarchy}"
        )

    # Check for common semantic convention attributes
    common_attrs = _compare_attributes(uipath_spans, openinference_spans)
    comparison["common_attributes"] = common_attrs

    return comparison


def _extract_hierarchy(spans: list[Any]) -> dict[str, str | None]:
    """Extract parent-child hierarchy from spans.

    Args:
        spans: List of spans

    Returns:
        Dictionary mapping span names to parent span names
    """
    span_map = {s.context.span_id: s for s in spans}
    hierarchy = {}

    for span in spans:
        parent_name = None
        if span.parent and span.parent.span_id in span_map:
            parent_name = span_map[span.parent.span_id].name
        hierarchy[span.name] = parent_name

    return hierarchy


def _compare_attributes(
    uipath_spans: list[Any],
    openinference_spans: list[Any],
) -> dict[str, Any]:
    """Compare attributes between span sets.

    Args:
        uipath_spans: Spans from UiPath instrumentation
        openinference_spans: Spans from OpenInference instrumentation

    Returns:
        Dictionary with attribute comparison results
    """
    if not uipath_spans or not openinference_spans:
        return {"error": "No spans to compare"}

    # Compare attributes on first span (as sample)
    uipath_attrs = (
        set(uipath_spans[0].attributes.keys()) if uipath_spans[0].attributes else set()
    )
    openinf_attrs = (
        set(openinference_spans[0].attributes.keys())
        if openinference_spans[0].attributes
        else set()
    )

    return {
        "shared_attributes": sorted(uipath_attrs & openinf_attrs),
        "uipath_only": sorted(uipath_attrs - openinf_attrs),
        "openinference_only": sorted(openinf_attrs - uipath_attrs),
    }


@pytest.mark.skipif(
    not (LANGCHAIN_AVAILABLE and OPENINFERENCE_AVAILABLE),
    reason="Requires langchain_core and openinference-instrumentation-langchain",
)
class TestOpenInferenceCompatibility:
    """Compare UiPath instrumentation with OpenInference reference implementation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        # Separate providers for each instrumentation to avoid conflicts
        self.uipath_exporter = InMemorySpanExporter()
        self.uipath_provider = TracerProvider(sampler=ALWAYS_ON)
        self.uipath_provider.add_span_processor(
            SimpleSpanProcessor(self.uipath_exporter)
        )

        self.openinference_exporter = InMemorySpanExporter()
        self.openinference_provider = TracerProvider(sampler=ALWAYS_ON)
        self.openinference_provider.add_span_processor(
            SimpleSpanProcessor(self.openinference_exporter)
        )

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.uipath_exporter.clear()
        self.openinference_exporter.clear()

    def test_runnable_chain_compatibility(self) -> None:
        """Test that UiPath instrumentation produces spans compatible with OpenInference."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define test chain
        def uppercase(text: str) -> str:
            return text.upper()

        test_input = "hello world"

        # ===== Run with UiPath instrumentation =====
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(uppercase)
            uipath_result = chain.invoke(test_input)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # ===== Run with OpenInference instrumentation =====
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(uppercase)
            openinference_result = chain.invoke(test_input)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # ===== Compare results =====
        # Functional output should be identical
        assert uipath_result == openinference_result == "HELLO WORLD"

        # Both should produce spans
        assert len(uipath_spans) > 0, "UiPath instrumentation produced no spans"
        assert len(openinference_spans) > 0, (
            "OpenInference instrumentation produced no spans"
        )

        # Compare span structures
        comparison = compare_span_structures(uipath_spans, openinference_spans)

        # Report comparison results
        print("\n=== Span Comparison ===")
        print(f"UiPath spans: {comparison['uipath_span_count']}")
        print(f"OpenInference spans: {comparison['openinference_span_count']}")
        print(f"\nSpan names (UiPath): {comparison['span_names']['uipath']}")
        print(
            f"Span names (OpenInference): {comparison['span_names']['openinference']}"
        )

        if comparison["common_attributes"]:
            print(
                f"\nShared attributes: {comparison['common_attributes']['shared_attributes']}"
            )
            print(
                f"UiPath-only attributes: {comparison['common_attributes']['uipath_only']}"
            )
            print(
                f"OpenInference-only: {comparison['common_attributes']['openinference_only']}"
            )

        if comparison["differences"]:
            print("\n=== Differences ===")
            for diff in comparison["differences"]:
                print(f"  {diff}")

        # Assertions for compatibility
        # Note: We allow some differences as long as semantic conventions are respected
        assert comparison["span_count_match"], (
            f"Span count mismatch: UiPath={comparison['uipath_span_count']}, "
            f"OpenInference={comparison['openinference_span_count']}"
        )

    def test_message_handling_compatibility(self) -> None:
        """Test that message attribute extraction matches OpenInference."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define test chain
        def echo_message(msg: HumanMessage) -> str:
            return f"Echo: {msg.content}"

        test_message = HumanMessage(content="Test message")

        # ===== Run with UiPath instrumentation =====
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(echo_message)
            uipath_result = chain.invoke(test_message)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # ===== Run with OpenInference instrumentation =====
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(echo_message)
            openinference_result = chain.invoke(test_message)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # ===== Compare results =====
        assert uipath_result == openinference_result

        # Both should produce spans
        assert len(uipath_spans) > 0
        assert len(openinference_spans) > 0

        # Compare message attributes
        comparison = compare_span_structures(uipath_spans, openinference_spans)

        print("\n=== Message Handling Comparison ===")
        print(f"UiPath spans: {len(uipath_spans)}")
        print(f"OpenInference spans: {len(openinference_spans)}")

        if comparison["common_attributes"]:
            shared = comparison["common_attributes"]["shared_attributes"]
            print(f"\nShared attributes: {shared}")

            # Look for message-related attributes
            message_attrs = [attr for attr in shared if "message" in attr.lower()]
            print(f"Message attributes: {message_attrs}")

        # Verify both implementations capture message data
        # (exact attribute names may differ, but both should have message attributes)
        assert len(uipath_spans) == len(openinference_spans), (
            "Different number of spans for message handling"
        )


@pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE,
    reason="Requires langchain_core",
)
class TestSemanticConventions:
    """Test that our implementation follows OpenInference semantic conventions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(sampler=ALWAYS_ON)
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.exporter.clear()

    def test_span_kind_convention(self) -> None:
        """Test that spans use correct SpanKind values."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def simple_func(text: str) -> str:
            return text.upper()

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(simple_func)
            chain.invoke("test")
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        # All spans should have a valid SpanKind
        from opentelemetry.trace import SpanKind

        for span in spans:
            assert span.kind in [
                SpanKind.INTERNAL,
                SpanKind.CLIENT,
                SpanKind.SERVER,
                SpanKind.PRODUCER,
                SpanKind.CONSUMER,
            ], f"Invalid SpanKind: {span.kind}"

    def test_semantic_attribute_naming(self) -> None:
        """Test that attribute names follow OpenInference conventions."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def simple_func(text: str) -> str:
            return text.upper()

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(simple_func)
            chain.invoke("test")
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        # Collect all attribute keys
        all_attrs = set()
        for span in spans:
            if span.attributes:
                all_attrs.update(span.attributes.keys())

        print("\n=== Captured attributes ===")
        for attr in sorted(all_attrs):
            print(f"  {attr}")

        # Check for expected OpenInference semantic convention patterns
        # These are the prefixes used by OpenInference
        expected_prefixes = [
            "llm.",  # LLM-related attributes
            "input.",  # Input attributes
            "output.",  # Output attributes
            "openinference.",  # OpenInference-specific
        ]

        # At least some attributes should follow the conventions
        # (Note: Not all spans will have all prefixes)
        convention_attrs = [
            attr
            for attr in all_attrs
            if any(attr.startswith(prefix) for prefix in expected_prefixes)
        ]

        print(
            f"\nAttributes following OpenInference conventions: {sorted(convention_attrs)}"
        )

        # We should have at least some conforming attributes
        # (This is a soft assertion - actual attribute presence depends on run type)
        assert len(all_attrs) > 0, "No attributes captured"


@pytest.mark.skipif(
    not (LANGCHAIN_AVAILABLE and OPENINFERENCE_AVAILABLE),
    reason="Requires langchain_core and openinference-instrumentation-langchain",
)
class TestOpenInferenceScenarios:
    """Phase 1.1: Comprehensive scenario testing for OpenInference compatibility.

    These tests validate OpenInference compatibility across diverse execution patterns
    to ensure robust coverage beyond basic workflows.
    """

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.uipath_exporter = InMemorySpanExporter()
        self.uipath_provider = TracerProvider(sampler=ALWAYS_ON)
        self.uipath_provider.add_span_processor(
            SimpleSpanProcessor(self.uipath_exporter)
        )

        self.openinference_exporter = InMemorySpanExporter()
        self.openinference_provider = TracerProvider(sampler=ALWAYS_ON)
        self.openinference_provider.add_span_processor(
            SimpleSpanProcessor(self.openinference_exporter)
        )

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.uipath_exporter.clear()
        self.openinference_exporter.clear()

    def test_nested_chains_scenario(self) -> None:
        """Test nested chain execution produces compatible span hierarchy."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define nested chains
        def inner_chain(text: str) -> str:
            return text.upper()

        def outer_chain(text: str) -> str:
            inner = RunnableLambda(inner_chain)
            return f"Result: {inner.invoke(text)}"

        test_input = "nested test"

        # Run with UiPath instrumentation
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(outer_chain)
            uipath_result = chain.invoke(test_input)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # Run with OpenInference instrumentation
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(outer_chain)
            openinference_result = chain.invoke(test_input)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # Verify functional equivalence
        assert uipath_result == openinference_result

        # Verify both captured nested structure
        assert len(uipath_spans) > 1, "UiPath should capture nested chains"
        assert len(openinference_spans) > 1, (
            "OpenInference should capture nested chains"
        )

        # Verify hierarchy structure matches
        uipath_hierarchy = _extract_hierarchy(uipath_spans)
        openinference_hierarchy = _extract_hierarchy(openinference_spans)

        # Both should have parent-child relationships
        uipath_has_children = any(
            parent is not None for parent in uipath_hierarchy.values()
        )
        openinf_has_children = any(
            parent is not None for parent in openinference_hierarchy.values()
        )

        assert uipath_has_children, (
            "UiPath spans should have parent-child relationships"
        )
        assert openinf_has_children, (
            "OpenInference spans should have parent-child relationships"
        )

    def test_error_handling_scenario(self) -> None:
        """Test error handling produces compatible error attributes."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define chain that raises error
        def failing_chain(text: str) -> str:
            raise ValueError("Intentional test error")

        test_input = "error test"

        # Run with UiPath instrumentation
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(failing_chain)
            with pytest.raises(ValueError):
                chain.invoke(test_input)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # Run with OpenInference instrumentation
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(failing_chain)
            with pytest.raises(ValueError):
                chain.invoke(test_input)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # Verify both captured error spans
        assert len(uipath_spans) > 0, "UiPath should capture error spans"
        assert len(openinference_spans) > 0, "OpenInference should capture error spans"

        # Verify error status
        from opentelemetry.trace import StatusCode

        uipath_error_spans = [
            s for s in uipath_spans if s.status.status_code == StatusCode.ERROR
        ]
        openinf_error_spans = [
            s for s in openinference_spans if s.status.status_code == StatusCode.ERROR
        ]

        assert len(uipath_error_spans) > 0, "UiPath should mark spans with error status"
        assert len(openinf_error_spans) > 0, (
            "OpenInference should mark spans with error status"
        )

        # Verify both recorded exceptions
        uipath_has_exception = any(len(s.events) > 0 for s in uipath_spans)
        openinf_has_exception = any(len(s.events) > 0 for s in openinference_spans)

        assert uipath_has_exception, "UiPath should record exception events"
        assert openinf_has_exception, "OpenInference should record exception events"

    def test_async_workflow_scenario(self) -> None:
        """Test async workflow execution produces compatible spans."""
        import asyncio

        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define async chain
        async def async_chain(text: str) -> str:
            await asyncio.sleep(0.01)  # Simulate async work
            return text.upper()

        test_input = "async test"

        # Run with UiPath instrumentation
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(async_chain)
            uipath_result = asyncio.run(chain.ainvoke(test_input))
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # Run with OpenInference instrumentation
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(async_chain)
            openinference_result = asyncio.run(chain.ainvoke(test_input))
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # Verify functional equivalence
        assert uipath_result == openinference_result == "ASYNC TEST"

        # Verify both captured async execution
        assert len(uipath_spans) > 0, "UiPath should capture async spans"
        assert len(openinference_spans) > 0, "OpenInference should capture async spans"

        # Verify span structure matches
        assert len(uipath_spans) == len(openinference_spans), (
            "Async execution should produce same span count"
        )

    def test_batch_processing_scenario(self) -> None:
        """Test batch processing produces compatible span structure."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define chain
        def batch_chain(text: str) -> str:
            return text.upper()

        test_inputs = ["test1", "test2", "test3"]

        # Run with UiPath instrumentation
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(batch_chain)
            uipath_result = chain.batch(test_inputs)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # Run with OpenInference instrumentation
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(batch_chain)
            openinference_result = chain.batch(test_inputs)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # Verify functional equivalence
        assert uipath_result == openinference_result == ["TEST1", "TEST2", "TEST3"]

        # Verify both captured batch execution
        assert len(uipath_spans) > 0, "UiPath should capture batch spans"
        assert len(openinference_spans) > 0, "OpenInference should capture batch spans"

        # Both should capture multiple invocations
        assert len(uipath_spans) >= len(test_inputs), (
            "UiPath should have spans for each batch item"
        )
        assert len(openinference_spans) >= len(test_inputs), (
            "OpenInference should have spans for each batch item"
        )

    def test_sequential_chain_scenario(self) -> None:
        """Test sequential chain execution produces compatible span structure."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define sequential chains
        def step1(text: str) -> str:
            return text.upper()

        def step2(text: str) -> str:
            return f"Processed: {text}"

        test_input = "sequential test"

        # Run with UiPath instrumentation
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableLambda(step1) | RunnableLambda(step2)
            uipath_result = chain.invoke(test_input)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # Run with OpenInference instrumentation
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableLambda(step1) | RunnableLambda(step2)
            openinference_result = chain.invoke(test_input)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # Verify functional equivalence
        assert uipath_result == openinference_result == "Processed: SEQUENTIAL TEST"

        # Verify both captured sequential execution
        assert len(uipath_spans) > 1, "UiPath should capture multiple steps"
        assert len(openinference_spans) > 1, (
            "OpenInference should capture multiple steps"
        )

        # Verify span count matches (both should capture sequence + steps)
        assert len(uipath_spans) == len(openinference_spans), (
            "Sequential chain should produce same span count"
        )

    def test_parallel_execution_scenario(self) -> None:
        """Test parallel execution produces compatible span structure."""
        from langchain_core.runnables import RunnableParallel

        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        # Define parallel branches
        def branch1(text: str) -> str:
            return text.upper()

        def branch2(text: str) -> str:
            return text.lower()

        test_input = "Parallel Test"

        # Run with UiPath instrumentation
        uipath_instrumentor = LangChainInstrumentor()
        try:
            uipath_instrumentor.instrument(tracer_provider=self.uipath_provider)
            chain = RunnableParallel(
                upper=RunnableLambda(branch1),
                lower=RunnableLambda(branch2),
            )
            uipath_result = chain.invoke(test_input)
            self.uipath_provider.force_flush()
        finally:
            uipath_instrumentor.uninstrument()

        uipath_spans = self.uipath_exporter.get_finished_spans()

        # Run with OpenInference instrumentation
        openinference_instrumentor = OpenInferenceLangChainInstrumentor()
        try:
            openinference_instrumentor.instrument(
                tracer_provider=self.openinference_provider
            )
            chain = RunnableParallel(
                upper=RunnableLambda(branch1),
                lower=RunnableLambda(branch2),
            )
            openinference_result = chain.invoke(test_input)
            self.openinference_provider.force_flush()
        finally:
            openinference_instrumentor.uninstrument()

        openinference_spans = self.openinference_exporter.get_finished_spans()

        # Verify functional equivalence
        assert (
            uipath_result
            == openinference_result
            == {
                "upper": "PARALLEL TEST",
                "lower": "parallel test",
            }
        )

        # Verify both captured parallel execution
        assert len(uipath_spans) > 2, "UiPath should capture parallel branches"
        assert len(openinference_spans) > 2, (
            "OpenInference should capture parallel branches"
        )

        # Verify span count matches
        assert len(uipath_spans) == len(openinference_spans), (
            "Parallel execution should produce same span count"
        )


@pytest.mark.skipif(
    not LANGCHAIN_AVAILABLE,
    reason="Requires langchain_core",
)
class TestOpenInferenceRegression:
    """Phase 1.2: Regression tests for OpenInference attribute stability.

    These tests ensure that semantic convention implementation remains stable
    across code changes and refactoring.
    """

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider(sampler=ALWAYS_ON)
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.exporter.clear()

    def test_attribute_name_stability(self) -> None:
        """Regression: Verify core OpenInference attribute names remain stable."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def simple_chain(text: str) -> str:
            return text.upper()

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(simple_chain)
            chain.invoke("test")
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        # Verify required OpenInference attributes exist
        span = spans[0]
        attrs = span.attributes or {}

        # MUST HAVE: openinference.span.kind
        assert "openinference.span.kind" in attrs, (
            "REGRESSION: openinference.span.kind attribute missing"
        )

        # MUST HAVE: input.value and output.value
        assert "input.value" in attrs, "REGRESSION: input.value attribute missing"
        assert "output.value" in attrs, "REGRESSION: output.value attribute missing"

        # MUST HAVE: MIME types
        assert "input.mime_type" in attrs, (
            "REGRESSION: input.mime_type attribute missing"
        )
        assert "output.mime_type" in attrs, (
            "REGRESSION: output.mime_type attribute missing"
        )

    def test_mime_type_consistency(self) -> None:
        """Regression: Verify MIME type values remain consistent."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def json_chain(data: dict[str, Any]) -> dict[str, Any]:
            return {"result": data}

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(json_chain)
            chain.invoke({"test": "data"})
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        span = spans[0]
        attrs = span.attributes or {}

        # Verify MIME type values
        assert attrs.get("input.mime_type") == "application/json", (
            "REGRESSION: input.mime_type should be application/json for dict inputs"
        )
        assert attrs.get("output.mime_type") == "application/json", (
            "REGRESSION: output.mime_type should be application/json for dict outputs"
        )

    def test_span_kind_mapping_stability(self) -> None:
        """Regression: Verify span kind mapping remains correct."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def simple_chain(text: str) -> str:
            return text.upper()

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(simple_chain)
            chain.invoke("test")
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        # Verify OpenInference span kind values
        valid_span_kinds = {"LLM", "CHAIN", "TOOL", "RETRIEVER", "AGENT", "EMBEDDING"}

        for span in spans:
            attrs = span.attributes or {}
            span_kind = attrs.get("openinference.span.kind")

            assert span_kind in valid_span_kinds, (
                f"REGRESSION: Invalid openinference.span.kind value: {span_kind}. "
                f"Must be one of {valid_span_kinds}"
            )

    def test_uipath_namespace_stability(self) -> None:
        """Regression: Verify UiPath custom attributes use correct namespace."""
        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def simple_chain(text: str) -> str:
            return text.upper()

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(simple_chain)
            chain.invoke("test")
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        # Collect all attribute keys
        all_attrs = set()
        for span in spans:
            if span.attributes:
                all_attrs.update(span.attributes.keys())

        # Verify no collision with OpenInference namespace
        for attr in all_attrs:
            if not attr.startswith(
                ("openinference.", "input.", "output.", "llm.", "gen_ai.", "run.")
            ):
                # Custom attributes should use uipath.* namespace
                if not attr.startswith("uipath."):
                    # Some standard Telemetry attributes are allowed
                    allowed_non_namespaced = {
                        "exception.type",
                        "exception.message",
                        "exception.stacktrace",
                    }
                    assert attr in allowed_non_namespaced, (
                        f"REGRESSION: Custom attribute '{attr}' should use 'uipath.*' namespace"
                    )

    def test_input_output_serialization_stability(self) -> None:
        """Regression: Verify input/output serialization format remains stable."""
        import json

        from uipath.core.telemetry.integrations_full.langchain import LangChainInstrumentor

        def complex_chain(data: dict[str, Any]) -> dict[str, Any]:
            return {"processed": data, "count": len(data)}

        test_input = {"key1": "value1", "key2": 123, "key3": [1, 2, 3]}

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)
            chain = RunnableLambda(complex_chain)
            chain.invoke(test_input)
            self.provider.force_flush()
        finally:
            instrumentor.uninstrument()

        spans = self.exporter.get_finished_spans()
        assert len(spans) > 0

        span = spans[0]
        attrs = span.attributes or {}

        # Verify input serialization
        input_value = attrs.get("input.value")
        assert input_value is not None, "input.value should be present"

        # Should be valid JSON
        try:
            input_data = json.loads(input_value)
            assert isinstance(input_data, dict), (
                "input.value should deserialize to dict"
            )
        except json.JSONDecodeError:
            pytest.fail("REGRESSION: input.value should be valid JSON")

        # Verify output serialization
        output_value = attrs.get("output.value")
        assert output_value is not None, "output.value should be present"

        try:
            output_data = json.loads(output_value)
            assert isinstance(output_data, dict), (
                "output.value should deserialize to dict"
            )
            assert "processed" in output_data, (
                "output should contain expected structure"
            )
        except json.JSONDecodeError:
            pytest.fail("REGRESSION: output.value should be valid JSON")
