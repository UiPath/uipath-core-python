"""Live integration tests for LangGraph with real LLM providers.

These tests use real API keys to validate end-to-end LangGraph workflows
with actual LLM providers.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Annotated, Any, Literal, TypedDict

import pytest
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from uipath.core.telemetry import trace
from uipath.core.telemetry.integrations_full.langgraph import (
    instrument_langgraph,
    uninstrument_langgraph,
)

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from uipath.core.telemetry.client import TelemetryClient


# Skip markers
requires_openai = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)

requires_anthropic = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


# ============================================================================
# State Definitions
# ============================================================================


class AgentState(TypedDict):
    """State for agent workflows."""

    messages: Annotated[list[BaseMessage], add_messages]
    next_action: str


class MultiStepState(TypedDict):
    """State for multi-step workflows."""

    input: str
    step1_output: str
    step2_output: str
    final_output: str


class RoutingState(TypedDict):
    """State for routing tests."""

    question: str
    category: str
    answer: str


class IterativeState(TypedDict):
    """State for iterative refinement."""

    query: str
    draft: str
    iteration: int
    max_iterations: int


# ============================================================================
# Tests
# ============================================================================


@requires_openai
def test_live_langgraph_simple_workflow(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live LangGraph workflow with OpenAI."""
    instrument_langgraph()

    from langchain_openai import ChatOpenAI

    def llm_node(state: AgentState) -> AgentState:
        """Node that calls LLM."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response], "next_action": "end"}

    # Build workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", llm_node)
    workflow.set_entry_point("llm")
    workflow.add_edge("llm", END)

    compiled = workflow.compile()

    # Execute with execution_id
    with trace("live_langgraph_test", execution_id="live-graph-001"):
        result = compiled.invoke({
            "messages": [("user", "Say 'workflow successful' and nothing else")],
            "next_action": "",
        })

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-graph-001"]
    assert len(exec_spans) >= 1

    # Verify langgraph.invoke span
    span_names = [s.name for s in spans]
    assert "langgraph.invoke" in span_names

    # Verify result
    assert len(result["messages"]) > 0

    # Cleanup
    uninstrument_langgraph()


@requires_openai
def test_live_langgraph_multi_step_workflow(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live LangGraph multi-step workflow."""
    instrument_langgraph()

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def analyze_input(state: MultiStepState) -> MultiStepState:
        """Analyze the input."""
        result = llm.invoke(f"Analyze this input: {state['input']}. Be brief.")
        return {"step1_output": result.content}

    def generate_response(state: MultiStepState) -> MultiStepState:
        """Generate response based on analysis."""
        result = llm.invoke(f"Based on this analysis: {state['step1_output']}, provide a brief response.")
        return {"step2_output": result.content}

    def finalize(state: MultiStepState) -> MultiStepState:
        """Finalize the output."""
        return {"final_output": f"Complete: {state['step2_output']}"}

    # Build workflow
    workflow = StateGraph(MultiStepState)
    workflow.add_node("analyze", analyze_input)
    workflow.add_node("generate", generate_response)
    workflow.add_node("finalize", finalize)
    workflow.set_entry_point("analyze")
    workflow.add_edge("analyze", "generate")
    workflow.add_edge("generate", "finalize")
    workflow.add_edge("finalize", END)

    compiled = workflow.compile()

    # Execute with execution_id
    with trace("live_multi_step", execution_id="live-graph-002"):
        result = compiled.invoke({
            "input": "test workflow",
            "step1_output": "",
            "step2_output": "",
            "final_output": "",
        })

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-graph-002"]
    assert len(exec_spans) >= 1

    # Verify result
    assert result["final_output"] is not None
    assert "Complete:" in result["final_output"]

    # Cleanup
    uninstrument_langgraph()


@requires_openai
def test_live_langgraph_conditional_routing(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live LangGraph with conditional routing."""
    instrument_langgraph()

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def categorize(state: RoutingState) -> RoutingState:
        """Categorize the question."""
        result = llm.invoke(
            f"Is this question about math or general knowledge? Answer with just 'math' or 'general': {state['question']}"
        )
        category = "math" if "math" in result.content.lower() else "general"
        return {"category": category}

    def math_handler(state: RoutingState) -> RoutingState:
        """Handle math questions."""
        result = llm.invoke(f"Answer this math question briefly: {state['question']}")
        return {"answer": result.content}

    def general_handler(state: RoutingState) -> RoutingState:
        """Handle general questions."""
        result = llm.invoke(f"Answer this general question briefly: {state['question']}")
        return {"answer": result.content}

    def route(state: RoutingState) -> Literal["math", "general"]:
        """Route based on category."""
        return "math" if state.get("category") == "math" else "general"

    # Build workflow
    workflow = StateGraph(RoutingState)
    workflow.add_node("categorize", categorize)
    workflow.add_node("math", math_handler)
    workflow.add_node("general", general_handler)
    workflow.set_entry_point("categorize")
    workflow.add_conditional_edges(
        "categorize",
        route,
        {"math": "math", "general": "general"},
    )
    workflow.add_edge("math", END)
    workflow.add_edge("general", END)

    compiled = workflow.compile()

    # Execute with execution_id
    with trace("live_routing", execution_id="live-graph-003"):
        result = compiled.invoke({
            "question": "What is 2 + 2?",
            "category": "",
            "answer": "",
        })

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-graph-003"]
    assert len(exec_spans) >= 1

    # Verify result
    assert result["answer"] is not None
    assert len(result["answer"]) > 0

    # Cleanup
    uninstrument_langgraph()


@requires_anthropic
def test_live_langgraph_anthropic(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live LangGraph with Anthropic Claude."""
    instrument_langgraph()

    from langchain_anthropic import ChatAnthropic

    def claude_node(state: AgentState) -> AgentState:
        """Node that calls Claude."""
        llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response], "next_action": "end"}

    # Build workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("claude", claude_node)
    workflow.set_entry_point("claude")
    workflow.add_edge("claude", END)

    compiled = workflow.compile()

    # Execute with execution_id
    with trace("live_anthropic_graph", execution_id="live-graph-004"):
        result = compiled.invoke({
            "messages": [("user", "Say 'Claude workflow successful' and nothing else")],
            "next_action": "",
        })

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-graph-004"]
    assert len(exec_spans) >= 1

    # Verify result
    assert len(result["messages"]) > 0

    # Cleanup
    uninstrument_langgraph()


@requires_openai
@pytest.mark.asyncio
async def test_live_langgraph_async(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live LangGraph async execution."""
    instrument_langgraph()

    from langchain_openai import ChatOpenAI

    async def async_llm_node(state: AgentState) -> AgentState:
        """Async node that calls LLM."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        messages = state["messages"]
        response = await llm.ainvoke(messages)
        return {"messages": [response], "next_action": "end"}

    # Build workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("llm", async_llm_node)
    workflow.set_entry_point("llm")
    workflow.add_edge("llm", END)

    compiled = workflow.compile()

    # Execute with execution_id
    with trace("live_async_graph", execution_id="live-graph-005"):
        result = await compiled.ainvoke({
            "messages": [("user", "Say 'async workflow successful' and nothing else")],
            "next_action": "",
        })

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-graph-005"]
    assert len(exec_spans) >= 1

    # Verify langgraph.ainvoke span
    span_names = [s.name for s in spans]
    assert "langgraph.ainvoke" in span_names

    # Verify result
    assert len(result["messages"]) > 0

    # Cleanup
    uninstrument_langgraph()


@requires_openai
def test_live_langgraph_iterative_workflow(
    telemetry_client: TelemetryClient,
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Test live LangGraph with iterative refinement."""
    instrument_langgraph()

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def generate_draft(state: IterativeState) -> IterativeState:
        """Generate initial draft."""
        if state["iteration"] == 0:
            result = llm.invoke(f"Write a brief answer to: {state['query']}")
        else:
            result = llm.invoke(f"Improve this answer: {state['draft']}")
        return {
            "draft": result.content,
            "iteration": state["iteration"] + 1,
        }

    def should_continue(state: IterativeState) -> Literal["continue", "end"]:
        """Check if should continue iterating."""
        return "continue" if state["iteration"] < state["max_iterations"] else "end"

    # Build workflow
    workflow = StateGraph(IterativeState)
    workflow.add_node("generate", generate_draft)
    workflow.set_entry_point("generate")
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {"continue": "generate", "end": END},
    )

    compiled = workflow.compile()

    # Execute with execution_id
    with trace("live_iterative", execution_id="live-graph-006"):
        result = compiled.invoke({
            "query": "What is AI?",
            "draft": "",
            "iteration": 0,
            "max_iterations": 2,
        })

    telemetry_client.flush()

    # Get captured spans
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) > 0

    # Verify execution_id propagated
    exec_spans = [s for s in spans if s.attributes.get("execution.id") == "live-graph-006"]
    assert len(exec_spans) >= 1

    # Verify result
    assert result["draft"] is not None
    assert result["iteration"] == 2

    # Cleanup
    uninstrument_langgraph()
