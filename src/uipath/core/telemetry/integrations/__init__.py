"""UiPath OpenTelemetry framework integrations.

This package provides automatic instrumentation for popular AI/LLM frameworks
using the OpenTelemetry BaseInstrumentor pattern.

Available integrations:
- langchain: LangChain instrumentation (OpenInference-based)
- langgraph: LangGraph workflow instrumentation (OpenInference-based)
"""

from ._base import UiPathInstrumentor
from .langchain import (
    UiPathLangChainInstrumentor,
    instrument_langchain,
    uninstrument_langchain,
)
from .langgraph import instrument_langgraph, uninstrument_langgraph

__all__ = [
    "UiPathInstrumentor",
    "UiPathLangChainInstrumentor",
    "instrument_langchain",
    "uninstrument_langchain",
    "instrument_langgraph",
    "uninstrument_langgraph",
]
