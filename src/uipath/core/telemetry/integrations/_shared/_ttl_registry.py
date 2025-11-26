"""Time-to-live span registry for memory-safe span tracking."""

from __future__ import annotations

import time
from collections import OrderedDict
from threading import Lock
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLSpanRegistry(Generic[K, V]):
    """Registry for tracking spans with automatic expiration.

    Prevents memory leaks by automatically removing old entries after TTL expires.
    Uses OrderedDict for efficient cleanup of expired entries.
    Thread-safe for concurrent access.

    Args:
        ttl_seconds: Time-to-live for entries in seconds (default: 3600)
        max_size: Maximum number of entries before cleanup (default: 10000)

    Examples:
        >>> registry = TTLSpanRegistry(ttl_seconds=60)
        >>> registry.register("span-123", span_object)
        >>> span = registry.get("span-123")
        >>> registry.cleanup()  # Remove expired entries
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 10000) -> None:
        """Initialize registry with TTL and size limits."""
        self._ttl_seconds = ttl_seconds
        self._max_size = max_size
        self._registry: OrderedDict[K, tuple[V, float]] = OrderedDict()
        self._lock = Lock()
        self._register_count = 0

    def register(self, span_id: K, span: V) -> None:
        """Register span with current timestamp.

        Args:
            span_id: Unique identifier for span
            span: Span object to register
        """
        with self._lock:
            current_time = time.time()
            self._registry[span_id] = (span, current_time)
            self._register_count += 1

            # Automatic cleanup when size limit reached or periodically
            if len(self._registry) > self._max_size or self._register_count % 1000 == 0:
                self._cleanup_unlocked()

    def get(self, span_id: K) -> V | None:
        """Get span by ID if not expired.

        Args:
            span_id: Unique identifier for span

        Returns:
            Span object if found and not expired, None otherwise
        """
        with self._lock:
            entry = self._registry.get(span_id)
            if entry is None:
                return None

            span, timestamp = entry
            if self._is_expired(timestamp):
                del self._registry[span_id]
                return None

            return span

    def cleanup(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            return self._cleanup_unlocked()

    def _cleanup_unlocked(self) -> int:
        """Internal cleanup that assumes lock is held.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, timestamp) in self._registry.items()
            if current_time - timestamp > self._ttl_seconds
        ]

        for key in expired_keys:
            del self._registry[key]

        return len(expired_keys)

    def pop(self, span_id: K, default: V | None = None) -> V | None:
        """Remove and return span by ID.

        Args:
            span_id: Unique identifier for span
            default: Default value if span_id not found

        Returns:
            Span object if found, default otherwise
        """
        with self._lock:
            entry = self._registry.pop(span_id, None)
            if entry is None:
                return default
            span, _ = entry
            return span

    def __setitem__(self, span_id: K, span: V) -> None:
        """Set span by ID (dict-like interface).

        Args:
            span_id: Unique identifier for span
            span: Span object to register
        """
        self.register(span_id, span)

    def __getitem__(self, span_id: K) -> V:
        """Get span by ID (dict-like interface).

        Args:
            span_id: Unique identifier for span

        Returns:
            Span object if found and not expired

        Raises:
            KeyError: If span_id not found or expired
        """
        result = self.get(span_id)
        if result is None:
            raise KeyError(span_id)
        return result

    def clear(self) -> None:
        """Remove all entries from registry."""
        with self._lock:
            self._registry.clear()

    def _is_expired(self, timestamp: float) -> bool:
        """Check if timestamp is expired based on TTL.

        Args:
            timestamp: Unix timestamp to check

        Returns:
            True if expired, False otherwise
        """
        return time.time() - timestamp > self._ttl_seconds

    def __len__(self) -> int:
        """Return number of entries in registry."""
        with self._lock:
            return len(self._registry)
