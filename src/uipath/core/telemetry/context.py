"""Execution context propagation via ContextVar.

Provides async-safe propagation of execution_id across function calls
without explicit parameter threading. This enables automatic correlation
of all operations within a single workflow execution.

Example:
    >>> from uipath.core.telemetry import set_execution_id, get_telemetry_client
    >>>
    >>> # Set once at workflow start
    >>> set_execution_id("workflow-run-12345")
    >>>
    >>> # All spans automatically include execution.id attribute
    >>> client = get_telemetry_client()
    >>> with client.start_as_current_span("process"):
    >>>     pass  # execution.id="workflow-run-12345" added automatically
"""

from contextvars import ContextVar
from typing import Optional

# Thread-safe, async-safe context variable
_execution_id: ContextVar[Optional[str]] = ContextVar("execution_id", default=None)


def set_execution_id(execution_id: str) -> None:
    """Set execution ID for current context.

    The execution ID is automatically propagated to all child spans created
    within this context, enabling correlation of all operations within a
    single workflow execution.

    Args:
        execution_id: Unique identifier for this execution (e.g., workflow run ID)

    Example:
        >>> from uipath.core.telemetry import set_execution_id, get_telemetry_client
        >>>
        >>> # Set once at workflow start
        >>> set_execution_id("workflow-run-12345")
        >>>
        >>> # All spans automatically include execution.id attribute
        >>> client = get_telemetry_client()
        >>> with client.start_as_current_span("process"):
        >>>     pass  # execution.id="workflow-run-12345" added automatically
    """
    _execution_id.set(execution_id)


def get_execution_id() -> Optional[str]:
    """Get execution ID from current context.

    Returns:
        Current execution ID, or None if not set

    Example:
        >>> set_execution_id("exec-123")
        >>> assert get_execution_id() == "exec-123"
    """
    return _execution_id.get()


def clear_execution_id() -> None:
    """Clear execution ID from current context.

    Useful for cleanup in long-running processes or when execution scope ends.

    Example:
        >>> set_execution_id("exec-123")
        >>> clear_execution_id()
        >>> assert get_execution_id() is None
    """
    _execution_id.set(None)
