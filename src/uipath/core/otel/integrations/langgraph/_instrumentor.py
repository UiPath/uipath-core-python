"""LangGraph instrumentor for automatic tracing with graph features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.trace import get_tracer
from wrapt import wrap_function_wrapper

from .._shared import InstrumentationConfig
from ..langchain import LangChainInstrumentor
from ._augmentation import LangGraphAugmentation

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider


class LangGraphInstrumentor(BaseInstrumentor):
    """Instrumentor for LangGraph framework.

    Provides comprehensive instrumentation for LangGraph including:
    - Base LangChain callback tracing (via LangChainInstrumentor)
    - Graph topology capture
    - Checkpoint tracking
    - State transition monitoring

    Examples:
        Basic usage:

        >>> instrumentor = LangGraphInstrumentor()
        >>> instrumentor.instrument()

        With custom configuration:

        >>> config = InstrumentationConfig(capture_inputs=True)
        >>> instrumentor.instrument(config=config)
    """

    def __init__(self) -> None:
        """Initialize LangGraph instrumentor."""
        super().__init__()
        self._langchain_instrumentor: LangChainInstrumentor | None = None
        self._augmentation: LangGraphAugmentation | None = None

    def instrumentation_dependencies(self) -> Collection[str]:
        """Return list of packages that this instrumentor depends on.

        Returns:
            Collection of package names
        """
        return ["langgraph >= 0.2.0", "langchain_core >= 0.3.9"]

    def _instrument(self, **kwargs: Any) -> None:
        """Instrument LangGraph with base LangChain tracing and augmentation.

        Args:
            **kwargs: Instrumentation arguments including:
                - tracer_provider: OpenTelemetry TracerProvider
                - config: InstrumentationConfig instance
        """
        tracer_provider: TracerProvider | None = kwargs.get("tracer_provider")
        config: InstrumentationConfig | None = kwargs.get("config")

        # First, instrument LangChain (provides base callback tracing)
        self._langchain_instrumentor = LangChainInstrumentor()
        self._langchain_instrumentor.instrument(
            tracer_provider=tracer_provider,
            config=config,
        )

        # Get tracer for augmentation
        tracer = get_tracer(
            instrumenting_module_name=__name__,
            instrumenting_library_version="0.1.0",
            tracer_provider=tracer_provider,
        )

        # Create augmentation layer
        self._augmentation = LangGraphAugmentation(tracer, config)

        # Monkey-patch StateGraph.compile to capture topology
        wrap_function_wrapper(
            module="langgraph.graph.state",
            name="StateGraph.compile",
            wrapper=self._wrap_compile,
        )

        # Monkey-patch checkpoint save/load if available
        try:
            wrap_function_wrapper(
                module="langgraph.checkpoint.base",
                name="BaseCheckpointSaver.put",
                wrapper=self._wrap_checkpoint_save,
            )
            wrap_function_wrapper(
                module="langgraph.checkpoint.base",
                name="BaseCheckpointSaver.get",
                wrapper=self._wrap_checkpoint_load,
            )
        except (ImportError, AttributeError):
            # Checkpoint support is optional
            pass

    def _uninstrument(self, **kwargs: Any) -> None:
        """Remove instrumentation from LangGraph.

        Args:
            **kwargs: Uninstrumentation arguments
        """
        # Uninstrument LangChain
        if self._langchain_instrumentor:
            self._langchain_instrumentor.uninstrument()
            self._langchain_instrumentor = None

        # Note: wrapt doesn't support unwrapping, so graph augmentation
        # patches remain until process restart
        self._augmentation = None

    def _wrap_compile(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Wrapper for StateGraph.compile to capture topology.

        Args:
            wrapped: Original compile method
            instance: StateGraph instance
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Compiled graph application
        """
        # Call original compile
        result = wrapped(*args, **kwargs)

        # Capture topology using augmentation
        if self._augmentation:
            graph_name = getattr(instance, "name", "unknown")
            self._augmentation.capture_graph_topology(result, graph_name)

        return result

    def _wrap_checkpoint_save(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Wrapper for checkpoint save to capture checkpoint operations.

        Args:
            wrapped: Original put method
            instance: BaseCheckpointSaver instance
            args: Positional arguments (checkpoint data)
            kwargs: Keyword arguments

        Returns:
            Result of checkpoint save
        """
        # Extract checkpoint ID and data from args
        checkpoint_id = None
        checkpoint_data = {}

        if len(args) > 0:
            checkpoint_data = args[0] if isinstance(args[0], dict) else {}
            checkpoint_id = checkpoint_data.get("id", "unknown")

        # Call original save
        result = wrapped(*args, **kwargs)

        # Capture checkpoint save
        if self._augmentation and checkpoint_id:
            self._augmentation.capture_checkpoint_save(checkpoint_id, checkpoint_data)

        return result

    def _wrap_checkpoint_load(
        self,
        wrapped: Any,
        instance: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Wrapper for checkpoint load to capture checkpoint operations.

        Args:
            wrapped: Original get method
            instance: BaseCheckpointSaver instance
            args: Positional arguments (checkpoint ID)
            kwargs: Keyword arguments

        Returns:
            Loaded checkpoint data
        """
        # Extract checkpoint ID from args
        checkpoint_id = args[0] if len(args) > 0 else kwargs.get("checkpoint_id", "unknown")

        # Capture checkpoint load
        if self._augmentation:
            self._augmentation.capture_checkpoint_load(str(checkpoint_id))

        # Call original load
        return wrapped(*args, **kwargs)
