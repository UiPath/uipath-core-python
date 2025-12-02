"""Session context management for UiPath OpenTelemetry integrations.

Provides thread-safe session and thread ID tracking using ContextVars.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

# Thread-safe context variables
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_thread_id: ContextVar[Optional[str]] = ContextVar("thread_id", default=None)


def set_session_context(session_id: str, thread_id: Optional[str] = None) -> None:
    """Set session context for all spans in this execution.

    Args:
        session_id: Unique session identifier
        thread_id: Optional thread/conversation identifier
    """
    _session_id.set(session_id)
    if thread_id:
        _thread_id.set(thread_id)


def get_session_id() -> Optional[str]:
    """Get current session ID from context.

    Returns:
        Session ID if set, None otherwise
    """
    return _session_id.get()


def get_thread_id() -> Optional[str]:
    """Get current thread ID from context.

    Returns:
        Thread ID if set, None otherwise
    """
    return _thread_id.get()


def clear_session_context() -> None:
    """Clear session context."""
    _session_id.set(None)
    _thread_id.set(None)
