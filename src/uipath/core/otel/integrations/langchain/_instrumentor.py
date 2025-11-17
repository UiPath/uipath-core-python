"""LangChain instrumentor for automatic tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from .._shared import InstrumentationConfig
from ._tracer import UiPathTracer

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider


class LangChainInstrumentor(BaseInstrumentor):
    """Instrumentor for LangChain framework.

    Automatically instruments LangChain by injecting UiPathTracer into
    BaseCallbackManager. This provides zero-code instrumentation for all
    LangChain operations including chains, LLMs, tools, and retrievers.

    Examples:
        >>> from openinference.instrumentation.langchain import LangChainInstrumentor
        >>> instrumentor = LangChainInstrumentor()
        >>> instrumentor.instrument()
    """

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return list of packages that this instrumentor depends on.

        Returns:
            Collection of package names
        """
        return ["langchain_core >= 0.3.9"]

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument LangChain by monkey-patching BaseCallbackManager.

        Args:
            **kwargs: Instrumentation arguments including:
                - tracer_provider: OpenTelemetry TracerProvider
                - config: InstrumentationConfig instance
        """
        tracer_provider: TracerProvider | None = kwargs.get("tracer_provider")
        config: InstrumentationConfig | None = kwargs.get("config")

        # Get or create tracer
        tracer = get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="0.1.0",
            tracer_provider=tracer_provider,
        )

        # Create UiPath tracer
        self._uipath_tracer = UiPathTracer(tracer, config)

        # Monkey-patch BaseCallbackManager.__init__
        wrap_function_wrapper(
            module="langchain_core.callbacks.manager",
            name="BaseCallbackManager.__init__",
            wrapper=self._wrap_callback_manager_init,
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation from LangChain.

        Note: This is a no-op since wrapt doesn't support unwrapping.
        Requires process restart to fully uninstrument.

        Args:
            **kwargs: Uninstrumentation arguments
        """
        # wrapt doesn't support unwrapping, so this is a no-op
        # User must restart process to uninstrument
        pass

    def _wrap_callback_manager_init(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Wrapper for BaseCallbackManager.__init__ to inject UiPathTracer.

        Args:
            wrapped: Original __init__ method
            instance: BaseCallbackManager instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Result of original __init__
        """
        # Call original __init__
        result = wrapped(*args, **kwargs)

        # Inject UiPathTracer into handlers if not already present
        if hasattr(instance, "handlers"):
            # Check if our tracer is already in handlers
            has_uipath_tracer = any(
                isinstance(handler, UiPathTracer) for handler in instance.handlers
            )

            if not has_uipath_tracer:
                instance.handlers.append(self._uipath_tracer)

        return result
