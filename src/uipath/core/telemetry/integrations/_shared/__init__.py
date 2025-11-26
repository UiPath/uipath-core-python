"""Shared utilities for UiPath OpenTelemetry integrations.

This module provides common functionality used across multiple integrations
to avoid code duplication and maintain consistency.
"""

from __future__ import annotations

from ._config import InstrumentationConfig
from ._serialization import safe_json_dumps, truncate_string
from ._session_context import (
    clear_session_context,
    get_session_id,
    get_thread_id,
    set_session_context,
)
from ._ttl_registry import TTLSpanRegistry

__all__ = [
    # Config
    "InstrumentationConfig",
    # Serialization
    "safe_json_dumps",
    "truncate_string",
    # Session context
    "set_session_context",
    "get_session_id",
    "get_thread_id",
    "clear_session_context",
    # Registry
    "TTLSpanRegistry",
]
