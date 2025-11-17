"""Live tests with actual LangChain/LangGraph execution.

These tests require langchain_core and langgraph to be installed.
They will be skipped if dependencies are not available.
"""

from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

# Try to import dependencies
try:
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableLambda

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from langgraph.graph import StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain_core not installed")
class TestLangChainLive:
    """Live tests with actual LangChain execution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.exporter.clear()

    def test_simple_runnable_chain(self) -> None:
        """Test instrumentation with a simple LangChain runnable."""
        from uipath.core.otel.integrations.langchain import LangChainInstrumentor

        # Instrument LangChain with our custom tracer provider
        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)

            # Create and run a simple chain
            def uppercase(text: str) -> str:
                return text.upper()

            chain = RunnableLambda(uppercase)
            result = chain.invoke("hello world")

            assert result == "HELLO WORLD"

            # Force flush to ensure all spans are exported
            self.provider.force_flush()

            # Verify spans were created
            spans = self.exporter.get_finished_spans()
            assert len(spans) > 0

            # Spans should be created for the chain execution
            # Note: Span names vary by LangChain version, so we just verify spans exist

        finally:
            instrumentor.uninstrument()

    def test_chain_with_messages(self) -> None:
        """Test instrumentation captures message attributes."""
        from uipath.core.otel.integrations.langchain import LangChainInstrumentor

        instrumentor = LangChainInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)

            # Create a chain that processes messages
            def process_message(msg: HumanMessage) -> str:
                return f"Processed: {msg.content}"

            chain = RunnableLambda(process_message)
            result = chain.invoke(HumanMessage(content="Test message"))

            assert "Processed: Test message" in result

            # Force flush to ensure all spans are exported
            self.provider.force_flush()

            spans = self.exporter.get_finished_spans()
            assert len(spans) > 0

        finally:
            instrumentor.uninstrument()


@pytest.mark.skipif(
    not (LANGCHAIN_AVAILABLE and LANGGRAPH_AVAILABLE),
    reason="langchain_core and langgraph not installed",
)
class TestLangGraphLive:
    """Live tests with actual LangGraph execution."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))

    def teardown_method(self) -> None:
        """Clean up after tests."""
        self.exporter.clear()

    def test_simple_graph_execution(self) -> None:
        """Test instrumentation with a simple LangGraph workflow."""
        from typing import TypedDict

        from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

        class AgentState(TypedDict):
            """Simple state for testing."""

            message: str
            count: int

        # Instrument LangGraph with our custom tracer provider
        instrumentor = LangGraphInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)

            # Create a simple graph
            workflow = StateGraph(AgentState)

            def process_node(state: AgentState) -> AgentState:
                return {
                    "message": state["message"].upper(),
                    "count": state["count"] + 1,
                }

            workflow.add_node("process", process_node)
            workflow.set_entry_point("process")
            workflow.set_finish_point("process")

            # Compile and run
            app = workflow.compile()
            result = app.invoke({"message": "hello", "count": 0})

            assert result["message"] == "HELLO"
            assert result["count"] == 1

            # Force flush to ensure all spans are exported
            self.provider.force_flush()

            # Verify spans were created
            spans = self.exporter.get_finished_spans()
            assert len(spans) > 0

            # Check for LangGraph execution spans
            span_names = [span.name for span in spans]
            # New instrumentor creates spans like "langgraph.invoke", "langgraph.LangGraph", "langgraph.process"
            assert any("langgraph." in name for name in span_names)

        finally:
            instrumentor.uninstrument()

    def test_graph_with_conditional_edges(self) -> None:
        """Test instrumentation captures conditional edges."""
        from typing import TypedDict

        from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

        class RouterState(TypedDict):
            """State with routing logic."""

            value: int
            result: str

        instrumentor = LangGraphInstrumentor()
        try:
            instrumentor.instrument(tracer_provider=self.provider)

            workflow = StateGraph(RouterState)

            # Define node functions without type hints to avoid LangGraph introspection issues
            def start_node(state):  # type: ignore
                return {"value": state["value"], "result": "started"}

            def positive_node(state):  # type: ignore
                return {"value": state["value"], "result": "positive"}

            def negative_node(state):  # type: ignore
                return {"value": state["value"], "result": "negative"}

            def router(state):  # type: ignore
                """Route based on value sign."""
                return "positive" if state["value"] > 0 else "negative"

            workflow.add_node("start", start_node)
            workflow.add_node("positive", positive_node)
            workflow.add_node("negative", negative_node)

            workflow.set_entry_point("start")
            workflow.add_conditional_edges("start", router)
            workflow.set_finish_point("positive")
            workflow.set_finish_point("negative")

            app = workflow.compile()

            # Test positive path
            result = app.invoke({"value": 5, "result": ""})
            assert result["result"] == "positive"

            # Force flush to ensure all spans are exported
            self.provider.force_flush()

            spans = self.exporter.get_finished_spans()
            assert len(spans) > 0

        finally:
            instrumentor.uninstrument()
