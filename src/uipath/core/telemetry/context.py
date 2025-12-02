"""UiPath telemetry context management.

Provides thread-safe context tracking using ContextVars for:
- Session ID: Unique identifier for a user session
- Thread ID: Unique identifier for a conversation/thread within a session

These context values are automatically added to spans by instrumentation layers.
"""

from __future__ import annotations

from contextvars import ContextVar

# Thread-safe context variables for UiPath identifiers
_session_id: ContextVar[str | None] = ContextVar("uipath_session_id", default=None)
_thread_id: ContextVar[str | None] = ContextVar("uipath_thread_id", default=None)


def set_session_context(session_id: str, thread_id: str | None = None) -> None:
    """Set UiPath session context for all spans in this execution context.

    Session and thread IDs are automatically propagated to all spans created
    within the current async context (async-safe via ContextVar).

    Args:
        session_id: Unique session identifier.
        thread_id: Optional thread/conversation identifier.

    Examples:
        Set session context::

            from uipath.core.telemetry import set_session_context

            set_session_context(session_id="session-123", thread_id="thread-456")

            # All spans created after this will have session.id and thread.id attributes
    """
    _session_id.set(session_id)
    if thread_id:
        _thread_id.set(thread_id)


def get_session_id() -> str | None:
    """Get current session ID from context.

    Returns:
        Session ID if set, None otherwise.

    Examples:
        >>> from uipath.core.telemetry.context import get_session_id
        >>> get_session_id()
        'session-123'
    """
    return _session_id.get()


def get_thread_id() -> str | None:
    """Get current thread ID from context.

    Returns:
        Thread ID if set, None otherwise.

    Examples:
        >>> from uipath.core.telemetry.context import get_thread_id
        >>> get_thread_id()
        'thread-456'
    """
    return _thread_id.get()


def clear_session_context() -> None:
    """Clear session and thread context.

    Resets both session_id and thread_id to None.

    Examples:
        Clear context::

            from uipath.core.telemetry import clear_session_context

            clear_session_context()
    """
    _session_id.set(None)
    _thread_id.set(None)
