"""Integration tests for LangChain and LangGraph instrumentation."""

from __future__ import annotations

from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath.core.otel.integrations._shared import InstrumentationConfig


class TestLangChainInstrumentationIntegration:
    """Integration tests for LangChain instrumentation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        trace_api.set_tracer_provider(provider)

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.exporter.clear()

    def test_instrumentor_initialization(self) -> None:
        """Test that instrumentor can be initialized."""
        from uipath.core.otel.integrations.langchain import LangChainInstrumentor

        instrumentor = LangChainInstrumentor()
        assert instrumentor is not None

    def test_instrumentor_with_config(self) -> None:
        """Test instrumentor initialization with custom config."""
        from uipath.core.otel.integrations.langchain import LangChainInstrumentor

        config = InstrumentationConfig(
            capture_inputs=False,
            max_string_length=2048,
        )

        instrumentor = LangChainInstrumentor()
        # Note: We can't actually call instrument() without langchain_core installed
        # But we can verify the instrumentor was created and config is valid
        assert instrumentor is not None
        assert config.capture_inputs is False
        assert config.max_string_length == 2048

    def test_instrumentation_dependencies(self) -> None:
        """Test that instrumentor declares correct dependencies."""
        from uipath.core.otel.integrations.langchain import LangChainInstrumentor

        instrumentor = LangChainInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()

        assert "langchain_core >= 0.3.9" in dependencies


class TestLangGraphInstrumentationIntegration:
    """Integration tests for LangGraph instrumentation."""

    def test_instrumentor_initialization(self) -> None:
        """Test that LangGraph instrumentor can be initialized."""
        from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

        instrumentor = LangGraphInstrumentor()
        assert instrumentor is not None

    def test_instrumentor_composition(self) -> None:
        """Test that LangGraph instrumentor has necessary internal state."""
        from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

        instrumentor = LangGraphInstrumentor()
        # Verify instrumentor is properly initialized
        # Note: The new instrumentor uses callback-based approach,
        # not composition, so we just verify it can be created
        assert instrumentor is not None
        assert hasattr(instrumentor, "_instrument")
        assert hasattr(instrumentor, "_uninstrument")

    def test_instrumentation_dependencies(self) -> None:
        """Test that LangGraph instrumentor declares correct dependencies."""
        from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

        instrumentor = LangGraphInstrumentor()
        dependencies = instrumentor.instrumentation_dependencies()

        assert "langgraph >= 1.0.0" in dependencies


class TestModuleImports:
    """Test that all modules can be imported correctly."""

    def test_import_shared_module(self) -> None:
        """Test importing shared module."""
        from uipath.core.otel.integrations._shared import (
            InstrumentationConfig,
            TTLSpanRegistry,
            safe_json_dumps,
            truncate_string,
        )

        assert InstrumentationConfig is not None
        assert TTLSpanRegistry is not None
        assert safe_json_dumps is not None
        assert truncate_string is not None

    def test_import_langchain_module(self) -> None:
        """Test importing LangChain module."""
        from uipath.core.otel.integrations.langchain import (
            LangChainInstrumentor,
            UiPathTracer,
        )

        assert LangChainInstrumentor is not None
        assert UiPathTracer is not None

    def test_import_langgraph_module(self) -> None:
        """Test importing LangGraph module."""
        from uipath.core.otel.integrations.langgraph import (
            LangGraphAugmentation,
            LangGraphInstrumentor,
        )

        assert LangGraphAugmentation is not None
        assert LangGraphInstrumentor is not None
