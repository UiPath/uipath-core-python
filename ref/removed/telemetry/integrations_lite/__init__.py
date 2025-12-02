"""DEPRECATED: Lightweight telemetry integrations for LangChain and LangGraph (Legacy).

.. deprecated:: 2025-12-01
   This module is deprecated and will be removed in a future version.
   Please use ``uipath.core.telemetry.integrations_openinference`` instead.

   Migration guide:
   - Old: from uipath.core.telemetry.integrations_lite import instrument_langchain
   - New: from uipath.core.telemetry.integrations_openinference import instrument_langchain

   See: src/uipath/core/telemetry/integrations_openinference/README.md

LEGACY IMPLEMENTATION - NOT RECOMMENDED FOR NEW CODE

This module provides minimal, easy-to-use instrumentation for LangChain and
LangGraph following KISS/YAGNI/DRY principles. However, integrations_openinference
provides the same benefits with automatic rich metadata via OpenInference delegation.

Example:
    >>> from uipath.core.telemetry import init
    >>> from uipath.core.telemetry.integrations_lite import (
    ...     instrument_langchain,
    ...     instrument_langgraph,
    ... )
    >>> from uipath.core.telemetry.integrations_full._shared import set_session_context
    >>>
    >>> # Initialize telemetry
    >>> init(enable_console_export=True)
    >>>
    >>> # Instrument integrations
    >>> instrument_langchain()
    >>> instrument_langgraph()
    >>>
    >>> # Set session context (optional)
    >>> set_session_context(session_id="my-session", thread_id="thread-1")
    >>>
    >>> # Now all @traceable and LangGraph execution is automatically traced
    >>> from langsmith import traceable
    >>> from langgraph.graph import StateGraph
    >>>
    >>> @traceable(run_type="tool")
    >>> def my_tool():
    ...     return "result"
    >>>
    >>> # Build and use LangGraph...

For session context management, import from the shared module:
    >>> from uipath.core.telemetry.integrations_full._shared import (
    ...     set_session_context,
    ...     clear_session_context,
    ...     get_session_id,
    ...     get_thread_id,
    ... )

Trade-offs vs Full Integration:
    Lite version (~430 LOC):
    - ✅ Basic tracing (name, duration, I/O)
    - ✅ Session/thread context
    - ✅ Simple maintenance
    - ❌ No rich metadata (model names, tokens, etc.)
    - ❌ No LLM response parsing
    - ❌ No custom extractors

    Full version (~2,366 LOC):
    - ✅ All lite features
    - ✅ Rich metadata extraction
    - ✅ LLM response parsing (OpenAI, Anthropic)
    - ✅ Custom callback handlers
    - ✅ Production-grade observability
"""

import warnings

warnings.warn(
    "integrations_lite is deprecated. "
    "Use integrations_openinference instead for automatic rich metadata via OpenInference. "
    "See src/uipath/core/telemetry/integrations_openinference/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)

from .langchain import (
    instrument_langchain,
    uninstrument_langchain,
)
from .langchain import (
    is_instrumented as is_langchain_instrumented,
)
from .langgraph import (
    instrument_langgraph,
    uninstrument_langgraph,
)
from .langgraph import (
    is_instrumented as is_langgraph_instrumented,
)

__all__ = [
    "instrument_langchain",
    "uninstrument_langchain",
    "is_langchain_instrumented",
    "instrument_langgraph",
    "uninstrument_langgraph",
    "is_langgraph_instrumented",
]
