"""Session context management for OpenInference instrumentation.

Provides thread-safe session and thread ID tracking using ContextVars.
These IDs are automatically added to spans by UiPathSpanProcessor.
"""

from __future__ import annotations

from contextvars import ContextVar

# Thread-safe context variables
_session_id: ContextVar[str | None] = ContextVar("uipath_session_id", default=None)
_thread_id: ContextVar[str | None] = ContextVar("uipath_thread_id", default=None)


def set_session_context(session_id: str, thread_id: str | None = None) -> None:
    """Set session context for all spans in this execution.

    Session and thread IDs are automatically added to all spans created by
    instrumented LangChain/LangGraph operations.

    Args:
        session_id: Unique session identifier.
        thread_id: Optional thread/conversation identifier.

    Examples:
        Set session context::

            from uipath.core.telemetry.integrations_openinference import (
                set_session_context,
            )

            set_session_context(session_id="session-123", thread_id="thread-456")
    """
    _session_id.set(session_id)
    if thread_id:
        _thread_id.set(thread_id)


def get_session_id() -> str | None:
    """Get current session ID from context.

    Returns:
        Session ID if set, None otherwise.
    """
    return _session_id.get()


def get_thread_id() -> str | None:
    """Get current thread ID from context.

    Returns:
        Thread ID if set, None otherwise.
    """
    return _thread_id.get()


def clear_session_context() -> None:
    """Clear session context.

    Examples:
        Clear context::

            from uipath.core.telemetry.integrations_openinference import (
                clear_session_context,
            )

            clear_session_context()
    """
    _session_id.set(None)
    _thread_id.set(None)
