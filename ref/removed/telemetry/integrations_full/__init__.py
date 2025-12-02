"""DEPRECATED: UiPath OpenTelemetry framework integrations (Legacy Implementation).

.. deprecated:: 2025-12-01
   This module is deprecated and will be removed in a future version.
   Please use ``uipath.core.telemetry.integrations_openinference`` instead.

   Migration guide:
   - Old: from uipath.core.telemetry.integrations_full.langchain import instrument_langchain
   - New: from uipath.core.telemetry.integrations_openinference import instrument_langchain

   See: src/uipath/core/telemetry/integrations_openinference/README.md

LEGACY IMPLEMENTATION - NOT RECOMMENDED FOR NEW CODE

This package provides automatic instrumentation for popular AI/LLM frameworks
using custom tracers and extractors (~1,800 LOC). This implementation is being
replaced by integrations_openinference which delegates to OpenInference (~92 LOC).

Available integrations:
- langchain: LangChain instrumentation (custom implementation)
- langgraph: LangGraph workflow instrumentation (custom implementation)
"""

import warnings

warnings.warn(
    "integrations_full is deprecated. "
    "Use integrations_openinference instead for a simpler, OpenInference-based implementation. "
    "See src/uipath/core/telemetry/integrations_openinference/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

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
