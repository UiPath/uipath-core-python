"""Tests for TTLSpanRegistry."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from uipath.core.otel.integrations._shared._ttl_registry import TTLSpanRegistry

if TYPE_CHECKING:
    pass


class TestTTLSpanRegistry:
    """Test cases for TTLSpanRegistry."""

    def test_register_and_get(self) -> None:
        """Test basic register and get operations."""
        registry = TTLSpanRegistry(ttl_seconds=60)
        span = "test_span"

        registry.register("span-1", span)
        retrieved = registry.get("span-1")

        assert retrieved == span

    def test_get_nonexistent_span(self) -> None:
        """Test getting a span that doesn't exist."""
        registry = TTLSpanRegistry()
        result = registry.get("nonexistent")

        assert result is None

    def test_ttl_expiration(self) -> None:
        """Test that expired spans are removed."""
        registry = TTLSpanRegistry(ttl_seconds=1)
        span = "test_span"

        registry.register("span-1", span)
        time.sleep(1.1)  # Wait for TTL to expire

        result = registry.get("span-1")
        assert result is None

    def test_cleanup_removes_expired(self) -> None:
        """Test that cleanup removes expired entries."""
        registry = TTLSpanRegistry(ttl_seconds=1)

        registry.register("span-1", "span1")
        registry.register("span-2", "span2")
        time.sleep(1.1)
        registry.register("span-3", "span3")

        removed_count = registry.cleanup()

        assert removed_count == 2
        assert len(registry) == 1
        assert registry.get("span-3") is not None

    def test_max_size_triggers_cleanup(self) -> None:
        """Test that exceeding max_size triggers cleanup."""
        registry = TTLSpanRegistry(ttl_seconds=1, max_size=5)

        # Register 5 spans
        for i in range(5):
            registry.register(f"span-{i}", f"span{i}")

        assert len(registry) == 5

        time.sleep(1.1)  # Make them expire

        # Register 6th span, should trigger cleanup
        registry.register("span-5", "span5")

        # Old spans should be cleaned up, only new one remains
        assert len(registry) <= 1

    def test_periodic_cleanup_trigger(self) -> None:
        """Test that cleanup triggers every 1000 registrations."""
        registry = TTLSpanRegistry(ttl_seconds=1, max_size=10000)

        # Register some spans and let them expire
        for i in range(10):
            registry.register(f"old-span-{i}", f"span{i}")

        time.sleep(1.1)

        # Register 1000 more to trigger periodic cleanup
        for i in range(1000):
            registry.register(f"new-span-{i}", f"span{i}")

        # Old expired spans should have been cleaned up
        assert registry.get("old-span-0") is None

    def test_clear(self) -> None:
        """Test that clear removes all entries."""
        registry = TTLSpanRegistry()

        for i in range(10):
            registry.register(f"span-{i}", f"span{i}")

        assert len(registry) == 10

        registry.clear()

        assert len(registry) == 0

    def test_len(self) -> None:
        """Test __len__ method."""
        registry = TTLSpanRegistry()

        assert len(registry) == 0

        registry.register("span-1", "span1")
        registry.register("span-2", "span2")

        assert len(registry) == 2

    def test_thread_safety_concurrent_register(self) -> None:
        """Test thread safety with concurrent register operations."""
        registry = TTLSpanRegistry()
        num_threads = 10
        spans_per_thread = 100

        def register_spans(thread_id: int) -> int:
            for i in range(spans_per_thread):
                registry.register(f"thread-{thread_id}-span-{i}", f"span{i}")
            return thread_id

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(register_spans, i) for i in range(num_threads)]
            for future in as_completed(futures):
                future.result()

        # All spans should be registered
        assert len(registry) == num_threads * spans_per_thread

    def test_thread_safety_concurrent_get(self) -> None:
        """Test thread safety with concurrent get operations."""
        registry = TTLSpanRegistry()

        # Pre-populate registry
        for i in range(100):
            registry.register(f"span-{i}", f"span{i}")

        def get_spans(thread_id: int) -> list[str | None]:
            results = []
            for i in range(100):
                results.append(registry.get(f"span-{i}"))
            return results

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_spans, i) for i in range(10)]
            for future in as_completed(futures):
                results = future.result()
                # All gets should succeed
                assert all(r is not None for r in results)

    def test_thread_safety_mixed_operations(self) -> None:
        """Test thread safety with mixed register/get/cleanup operations."""
        registry = TTLSpanRegistry(ttl_seconds=2)

        def mixed_operations(thread_id: int) -> int:
            # Register some spans
            for i in range(50):
                registry.register(f"thread-{thread_id}-span-{i}", f"span{i}")

            # Get some spans
            for i in range(25):
                registry.get(f"thread-{thread_id}-span-{i}")

            # Cleanup
            registry.cleanup()

            return thread_id

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(mixed_operations, i) for i in range(5)]
            for future in as_completed(futures):
                future.result()

        # Registry should be in a consistent state
        assert len(registry) >= 0  # No crashes or corrupted state

    def test_thread_safety_cleanup_during_iteration(self) -> None:
        """Test that cleanup doesn't cause iteration errors during concurrent access."""
        registry = TTLSpanRegistry(ttl_seconds=1, max_size=100)

        # Pre-populate with expired spans
        for i in range(50):
            registry.register(f"old-span-{i}", f"span{i}")

        time.sleep(1.1)

        def cleanup_thread() -> int:
            return registry.cleanup()

        def register_thread(thread_id: int) -> None:
            for i in range(60):  # Exceeds max_size to trigger cleanup
                registry.register(f"new-thread-{thread_id}-span-{i}", f"span{i}")

        with ThreadPoolExecutor(max_workers=5) as executor:
            # Start cleanup and register threads concurrently
            cleanup_future = executor.submit(cleanup_thread)
            register_futures = [executor.submit(register_thread, i) for i in range(4)]

            cleanup_future.result()
            for future in as_completed(register_futures):
                future.result()

        # Should complete without RuntimeError: dictionary changed size during iteration
        assert True
