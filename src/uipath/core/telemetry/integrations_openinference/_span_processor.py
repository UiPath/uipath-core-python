"""Custom span processor for UiPath attributes."""

from typing import Optional

from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from uipath.core.telemetry.attributes import Attr
from uipath.core.telemetry.context import get_session_id, get_thread_id


class UiPathSpanProcessor(SpanProcessor):
    """Span processor that adds UiPath session attributes to OpenInference spans.

    This processor enriches spans created by OpenInference with UiPath-specific
    session context (session_id, thread_id) from ContextVars.

    Examples:
        Add to tracer provider::

            from opentelemetry.sdk.trace import TracerProvider
            from uipath.core.telemetry.integrations_openinference._span_processor import (
                UiPathSpanProcessor
            )

            provider = TracerProvider()
            provider.add_span_processor(UiPathSpanProcessor())
    """

    def on_start(
        self,
        span: Span,
        parent_context: Optional[object] = None,
    ) -> None:
        """Called when span starts - add UiPath session attributes.

        Args:
            span: The span being started.
            parent_context: Parent context (optional, unused).
        """
        session_id = get_session_id()
        if session_id:
            span.set_attribute(Attr.Common.SESSION_ID, session_id)

        thread_id = get_thread_id()
        if thread_id:
            span.set_attribute(Attr.Common.THREAD_ID, thread_id)

    def on_end(self, span: ReadableSpan) -> None:
        """Required by SpanProcessor interface. No action needed on span end."""
        pass

    def shutdown(self) -> None:
        """Required by SpanProcessor interface. No cleanup needed."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Required by SpanProcessor interface. No buffering, always returns True."""
        return True
