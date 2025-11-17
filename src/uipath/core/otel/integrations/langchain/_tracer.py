"""UiPath tracer for LangChain framework instrumentation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tracers.base import BaseTracer
from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from ...attributes import Attr
from ...attributes import SpanKind as OpenInferenceSpanKind
from .._shared import InstrumentationConfig, TTLSpanRegistry
from ._extractors import (
    extract_chain_attributes,
    extract_llm_attributes,
    extract_retriever_attributes,
    extract_tool_attributes,
)

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run
    from opentelemetry.trace import Span, Tracer


class UiPathTracer(BaseTracer):
    """OpenTelemetry tracer for LangChain instrumentation.

    Extends LangChain's BaseTracer to create OpenTelemetry spans for all
    LangChain operations. Automatically captures inputs, outputs, and metadata
    according to OpenInference semantic conventions.

    Args:
        tracer: OpenTelemetry tracer instance
        config: Instrumentation configuration

    Examples:
        >>> from opentelemetry import trace
        >>> tracer = trace.get_tracer(__name__)
        >>> config = InstrumentationConfig()
        >>> uipath_tracer = UiPathTracer(tracer, config)
    """

    def __init__(self, tracer: Tracer, config: InstrumentationConfig | None = None) -> None:
        """Initialize UiPath tracer with OpenTelemetry tracer and config."""
        super().__init__()
        self._tracer = tracer
        self._config = config or InstrumentationConfig()
        self._span_registry = TTLSpanRegistry(
            ttl_seconds=self._config.ttl_seconds,
            max_size=self._config.max_registry_size,
        )

    def _persist_run(self, run: Run) -> None:
        """Persist run (no-op for OpenTelemetry tracer).

        BaseTracer requires this method, but we don't need to persist runs
        since OpenTelemetry handles span export.

        Args:
            run: LangChain Run object
        """
        pass

    def _on_run_create(self, run: Run) -> None:
        """Handle run creation by starting OpenTelemetry span.

        Args:
            run: LangChain Run object being created
        """
        # Determine span kind based on run type
        span_kind = self._get_span_kind(run)

        # Get parent span if exists
        parent_span = None
        if run.parent_run_id:
            parent_span = self._span_registry.get(str(run.parent_run_id))

        # Start new span
        span_name = self._get_span_name(run)
        if parent_span:
            ctx = trace.set_span_in_context(parent_span)
            span = self._tracer.start_span(span_name, kind=span_kind, context=ctx)
        else:
            span = self._tracer.start_span(span_name, kind=span_kind)

        # Register span for parent lookup
        self._span_registry.register(str(run.id), span)

        # Set common attributes
        span.set_attribute(Attr.Run.ID, str(run.id))
        span.set_attribute(Attr.Run.TYPE, run.run_type)
        if run.parent_run_id:
            span.set_attribute(Attr.Run.PARENT_ID, str(run.parent_run_id))

        # Set OpenInference span kind (MUST HAVE for ecosystem compatibility)
        openinference_kind = self._get_openinference_kind(run)
        span.set_attribute(Attr.Common.OPENINFERENCE_SPAN_KIND, openinference_kind)

        # Extract type-specific attributes
        self._extract_attributes(span, run)

    def _on_run_update(self, run: Run) -> None:
        """Handle run completion by ending OpenTelemetry span.

        Args:
            run: LangChain Run object being completed
        """
        span = self._span_registry.get(str(run.id))
        if not span:
            return

        # Record outputs if available
        if run.outputs and self._config.capture_outputs:
            self._record_outputs(span, run)

        # Record error if present
        if run.error:
            span.set_status(Status(StatusCode.ERROR, run.error))
            span.record_exception(Exception(run.error))
        else:
            span.set_status(Status(StatusCode.OK))

        # End span
        span.end()

    def _get_span_kind(self, run: Run) -> SpanKind:
        """Determine OpenTelemetry span kind from run type.

        Args:
            run: LangChain Run object

        Returns:
            Appropriate SpanKind for the run type
        """
        # All LangChain operations are internal processing
        return SpanKind.INTERNAL

    def _get_span_name(self, run: Run) -> str:
        """Generate span name from run information.

        Args:
            run: LangChain Run object

        Returns:
            Descriptive span name
        """
        if run.name:
            return run.name
        return f"{run.run_type}_{run.id}"

    def _get_openinference_kind(self, run: Run) -> str:
        """Determine OpenInference span kind from run type.

        Args:
            run: LangChain Run object

        Returns:
            OpenInference span kind (LLM, CHAIN, TOOL, RETRIEVER, AGENT)
        """
        if run.run_type == "llm":
            return OpenInferenceSpanKind.LLM
        elif run.run_type == "chain":
            return OpenInferenceSpanKind.CHAIN
        elif run.run_type == "tool":
            return OpenInferenceSpanKind.TOOL
        elif run.run_type == "retriever":
            return OpenInferenceSpanKind.RETRIEVER
        elif run.run_type == "agent":
            return OpenInferenceSpanKind.AGENT
        else:
            # Default to CHAIN for unknown types
            return OpenInferenceSpanKind.CHAIN

    def _extract_attributes(self, span: Span, run: Run) -> None:
        """Extract and set type-specific attributes on span.

        Args:
            span: OpenTelemetry span
            run: LangChain Run object
        """
        if run.run_type == "llm":
            attributes = extract_llm_attributes(run, self._config)
            for key, value in attributes.items():
                span.set_attribute(key, value)
        elif run.run_type == "chain":
            attributes = extract_chain_attributes(run, self._config)
            for key, value in attributes.items():
                span.set_attribute(key, value)
        elif run.run_type == "tool":
            attributes = extract_tool_attributes(run, self._config)
            for key, value in attributes.items():
                span.set_attribute(key, value)
        elif run.run_type == "retriever":
            attributes = extract_retriever_attributes(run, self._config)
            for key, value in attributes.items():
                span.set_attribute(key, value)

    def _record_outputs(self, span: Span, run: Run) -> None:
        """Record output attributes on span.

        Args:
            span: OpenTelemetry span
            run: LangChain Run object with outputs
        """
        if run.run_type == "llm":
            # LLM outputs handled by extract_llm_attributes during update
            attributes = extract_llm_attributes(run, self._config)
            for key, value in attributes.items():
                # Set output-related attributes
                if (
                    key.startswith(Attr.LLM.OUTPUT_MESSAGES)
                    or key.startswith("llm.token_count")
                    or key == Attr.Common.OUTPUT_VALUE
                    or key == Attr.Common.OUTPUT_MIME_TYPE
                    or key.startswith("gen_ai.")
                ):
                    span.set_attribute(key, value)
        elif run.run_type == "chain":
            from .._shared import safe_json_dumps, truncate_string

            output_str = safe_json_dumps(run.outputs)
            if len(output_str) > self._config.max_string_length:
                output_str = truncate_string(output_str, self._config.max_string_length)
            # OpenInference output.value
            span.set_attribute(Attr.Common.OUTPUT_VALUE, output_str)
            span.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")
        elif run.run_type == "tool":
            from .._shared import safe_json_dumps, truncate_string

            output_str = safe_json_dumps(run.outputs)
            if len(output_str) > self._config.max_string_length:
                output_str = truncate_string(output_str, self._config.max_string_length)
            # OpenInference output.value
            span.set_attribute(Attr.Common.OUTPUT_VALUE, output_str)
            span.set_attribute(Attr.Common.OUTPUT_MIME_TYPE, "application/json")
        elif run.run_type == "retriever":
            # Retriever outputs handled by extract_retriever_attributes
            pass
