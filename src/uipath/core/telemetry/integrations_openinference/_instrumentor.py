"""Wrapper around OpenInference LangChainInstrumentor with UiPath features."""

from typing import Any

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from ._span_processor import UiPathSpanProcessor


class UiPathLangChainInstrumentor:
    """Wrapper around OpenInference LangChainInstrumentor.

    Adds UiPath session context (session_id, thread_id) to all spans via
    UiPathSpanProcessor. Delegates instrumentation to OpenInference.
    """

    def __init__(self) -> None:
        """Initialize the UiPath LangChain instrumentor."""
        self._openinference_instrumentor = LangChainInstrumentor()

    def instrument(self, **kwargs: Any) -> None:
        """Instrument LangChain/LangGraph with OpenInference + UiPath features.

        Adds UiPathSpanProcessor to the TracerProvider, then instruments with OpenInference.

        Args:
            **kwargs: Passed to OpenInference LangChainInstrumentor
                (e.g., tracer_provider, skip_dep_check).

        Raises:
            ImportError: If openinference-instrumentation-langchain not installed.
            RuntimeError: If already instrumented.
        """
        # Get or create tracer provider
        tracer_provider = kwargs.get("tracer_provider")
        if tracer_provider is None:
            tracer_provider = trace.get_tracer_provider()

        # Add our custom span processor to inject UiPath attributes
        if isinstance(tracer_provider, TracerProvider):
            span_processor = UiPathSpanProcessor()
            tracer_provider.add_span_processor(span_processor)

        # Instrument with OpenInference
        # This handles all the LangChain/LangGraph patching
        self._openinference_instrumentor.instrument(**kwargs)

    def uninstrument(self) -> None:
        """Remove LangChain/LangGraph instrumentation.

        Note: UiPathSpanProcessor remains attached to TracerProvider (safe).

        Raises:
            RuntimeError: If not currently instrumented.
        """
        self._openinference_instrumentor.uninstrument()
