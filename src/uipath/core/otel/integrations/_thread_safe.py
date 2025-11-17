"""Thread-safe data structures for concurrent span management.

This module provides thread-safe dictionary implementation used for managing
spans by run_id in concurrent LangChain/LangGraph executions.
"""

from __future__ import annotations

from threading import RLock
from typing import Dict, Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class DictWithLock(Generic[K, V]):
    """Thread-safe dictionary wrapper with reentrant locking.

    Used for managing spans by run_id in concurrent LangChain executions.
    Uses RLock (reentrant lock) to prevent deadlocks in nested access patterns.

    Example:
        >>> from uuid import UUID
        >>> from opentelemetry.trace import Span
        >>> spans = DictWithLock[UUID, Span]()
        >>> spans[run_id] = span  # Thread-safe
        >>> span = spans.get(run_id)  # Thread-safe
        >>> spans.pop(run_id)  # Thread-safe
    """

    def __init__(self) -> None:
        """Initialize thread-safe dictionary with empty state."""
        self._dict: Dict[K, V] = {}
        self._lock = RLock()  # Reentrant lock for nested access

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value by key (thread-safe).

        Args:
            key: Dictionary key
            default: Default value if key not found

        Returns:
            Value associated with key, or default if not found
        """
        with self._lock:
            return self._dict.get(key, default)

    def __setitem__(self, key: K, value: V) -> None:
        """Set value by key (thread-safe).

        Args:
            key: Dictionary key
            value: Value to set
        """
        with self._lock:
            self._dict[key] = value

    def __getitem__(self, key: K) -> V:
        """Get value by key, raise KeyError if missing (thread-safe).

        Args:
            key: Dictionary key

        Returns:
            Value associated with key

        Raises:
            KeyError: If key not found in dictionary
        """
        with self._lock:
            return self._dict[key]

    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Remove and return value by key (thread-safe).

        Args:
            key: Dictionary key
            default: Default value if key not found

        Returns:
            Value associated with key, or default if not found
        """
        with self._lock:
            return self._dict.pop(key, default)

    def __contains__(self, key: K) -> bool:
        """Check if key exists (thread-safe).

        Args:
            key: Dictionary key

        Returns:
            True if key exists in dictionary, False otherwise
        """
        with self._lock:
            return key in self._dict

    def clear(self) -> None:
        """Clear all entries (thread-safe)."""
        with self._lock:
            self._dict.clear()

    def __len__(self) -> int:
        """Get number of entries (thread-safe).

        Returns:
            Number of key-value pairs in dictionary
        """
        with self._lock:
            return len(self._dict)

    def keys(self):
        """Get dictionary keys (thread-safe).

        Returns:
            Snapshot of dictionary keys at call time
        """
        with self._lock:
            return list(self._dict.keys())

    def values(self):
        """Get dictionary values (thread-safe).

        Returns:
            Snapshot of dictionary values at call time
        """
        with self._lock:
            return list(self._dict.values())

    def items(self):
        """Get dictionary items (thread-safe).

        Returns:
            Snapshot of dictionary items at call time
        """
        with self._lock:
            return list(self._dict.items())
