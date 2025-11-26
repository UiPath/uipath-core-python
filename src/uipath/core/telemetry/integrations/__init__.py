"""UiPath OpenTelemetry framework integrations.

This package provides automatic instrumentation for popular AI/LLM frameworks
using the OpenTelemetry BaseInstrumentor pattern.

Available integrations:
- langgraph: LangGraph workflow instrumentation
"""

from ._base import UiPathInstrumentor

__all__ = ["UiPathInstrumentor"]
