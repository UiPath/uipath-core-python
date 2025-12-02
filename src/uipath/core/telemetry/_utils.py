"""Internal utility functions for telemetry."""

from __future__ import annotations

import json
from typing import Any


def safe_json_dumps(obj: Any, max_length: int | None = None) -> str:
    """Safely serialize object to JSON string with error handling.

    Args:
        obj: Object to serialize
        max_length: Optional maximum length for result string

    Returns:
        JSON string or fallback representation

    Examples:
        >>> safe_json_dumps({"key": "value"})
        '{"key": "value"}'
        >>> safe_json_dumps({"key": "value"}, max_length=10)
        '{"key": "v'
    """
    try:
        result = json.dumps(obj, ensure_ascii=False)
        if max_length is not None:
            result = _truncate_string(result, max_length)
        return result
    except (TypeError, ValueError):
        # Fallback to repr for non-serializable objects
        result = repr(obj)
        if max_length is not None:
            result = _truncate_string(result, max_length)
        return result


def _truncate_string(s: str, max_length: int) -> str:
    """Truncate string to maximum length with ellipsis indicator.

    Args:
        s: String to truncate
        max_length: Maximum length (must be >= 3 for ellipsis)

    Returns:
        Truncated string with "..." suffix if truncated

    Raises:
        ValueError: If max_length < 3

    Examples:
        >>> _truncate_string("hello world", 8)
        'hello...'
        >>> _truncate_string("hi", 10)
        'hi'
    """
    if max_length < 3:
        raise ValueError("max_length must be >= 3 to allow for ellipsis")

    if len(s) <= max_length:
        return s

    return s[: max_length - 3] + "..."
