"""Tests for thread safety (CRITICAL fix validation)."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

import pytest

from uipath.core.otel.client import TelemetryClient, get_client, init_client
from uipath.core.otel.config import TelemetryConfig

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.export import InMemorySpanExporter


@pytest.mark.no_auto_tracer
def test_concurrent_init_client(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test concurrent init_client() calls create only one instance.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Reset global state for this test
    from opentelemetry import trace

    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    clients: list[TelemetryClient] = []
    exceptions: list[Exception] = []

    def init_worker() -> None:
        """Worker thread that calls init_client.

        Raises:
            Exception: Any exception from init_client
        """
        try:
            config = TelemetryConfig(enable_console_export=True)
            client = init_client(config)
            clients.append(client)
        except Exception as e:
            exceptions.append(e)

    # Launch 10 threads simultaneously
    threads = [threading.Thread(target=init_worker) for _ in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify no exceptions
    assert len(exceptions) == 0, f"Exceptions raised: {exceptions}"

    # Verify all threads got a client
    assert len(clients) == 10

    # Verify all clients are the same instance (singleton)
    first_client = clients[0]
    for client in clients[1:]:
        assert client is first_client


def test_init_client_with_shutdown(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test reset_client() and reinitializing works safely.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    from uipath.core.otel.client import reset_client

    # Create first client
    config1 = TelemetryConfig(enable_console_export=True)
    client1 = init_client(config1)

    # Verify first client works
    tracer1 = client1.get_tracer()
    assert tracer1 is not None

    # Reset and create second client
    reset_client()
    config2 = TelemetryConfig(enable_console_export=True)
    client2 = init_client(config2)

    # Verify second client is different instance
    assert client2 is not client1

    # Verify get_client() returns second client
    current = get_client()
    assert current is client2


def test_concurrent_reset_client() -> None:
    """Test concurrent reset_client() calls don't cause issues."""
    from uipath.core.otel.client import reset_client

    # Initialize client first
    config = TelemetryConfig(enable_console_export=True)
    init_client(config)

    exceptions: list[Exception] = []

    def reset_worker() -> None:
        """Worker thread that calls reset_client.

        Raises:
            Exception: Any exception from reset_client
        """
        try:
            reset_client()
        except Exception as e:
            exceptions.append(e)

    # Launch 5 threads simultaneously
    threads = [threading.Thread(target=reset_worker) for _ in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify no exceptions
    assert len(exceptions) == 0

    # Verify client is reset
    with pytest.raises(RuntimeError, match="not initialized"):
        get_client()


def test_concurrent_span_creation(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test concurrent span creation from multiple threads.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    from uipath.core.otel.trace import Trace

    # Initialize client
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    exceptions: list[Exception] = []
    created_spans: list[str] = []

    def span_worker(worker_id: int) -> None:
        """Worker thread that creates spans.

        Args:
            worker_id: Worker identifier

        Raises:
            Exception: Any exception during span creation
        """
        try:
            with Trace(tracer, f"thread-{worker_id}") as trace_ctx:
                with trace_ctx.span(f"op-{worker_id}", kind="generation"):
                    # Simulate work
                    time.sleep(0.01)
                    created_spans.append(f"thread-{worker_id}")
        except Exception as e:
            exceptions.append(e)

    # Launch 10 worker threads
    threads = [threading.Thread(target=span_worker, args=(i,)) for i in range(10)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify no exceptions
    assert len(exceptions) == 0

    # Verify all workers created spans
    assert len(created_spans) == 10

    # Verify all spans exported
    client.flush()
    spans = in_memory_exporter.get_finished_spans()
    assert len(spans) == 20  # 10 root + 10 operations


@pytest.mark.no_auto_tracer
def test_race_condition_init_and_get(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test race condition between init_client() and get_client().

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Reset global state for this test
    from opentelemetry import trace

    trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER_SET_ONCE._done = False  # type: ignore[attr-defined]
    clients: list[TelemetryClient | None] = []
    exceptions: list[Exception] = []

    def init_worker() -> None:
        """Worker that initializes client.

        Raises:
            Exception: Any exception during init
        """
        try:
            config = TelemetryConfig(enable_console_export=True)
            client = init_client(config)
            clients.append(client)
        except Exception as e:
            exceptions.append(e)

    def get_worker() -> None:
        """Worker that gets client.

        Raises:
            Exception: Any exception during get
        """
        try:
            # May fail if called before init, that's okay
            client = get_client()
            clients.append(client)
        except RuntimeError:
            # Expected if get called before init
            clients.append(None)
        except Exception as e:
            exceptions.append(e)

    # Launch mixed init and get threads
    threads = [
        threading.Thread(target=init_worker),
        threading.Thread(target=get_worker),
        threading.Thread(target=get_worker),
        threading.Thread(target=init_worker),
        threading.Thread(target=get_worker),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify no unexpected exceptions (RuntimeError is expected)
    assert len(exceptions) == 0

    # Verify all non-None clients are same instance
    non_none_clients = [c for c in clients if c is not None]
    assert len(non_none_clients) > 0
    first = non_none_clients[0]
    for client in non_none_clients[1:]:
        assert client is first


def test_concurrent_flush(in_memory_exporter: InMemorySpanExporter) -> None:
    """Test concurrent flush() calls are safe.

    Args:
        in_memory_exporter: In-memory exporter fixture
    """
    # Initialize client and create some spans
    config = TelemetryConfig(enable_console_export=True)
    client = init_client(config)
    tracer = client.get_tracer()

    from uipath.core.otel.trace import Trace

    with Trace(tracer, "test"):
        pass

    exceptions: list[Exception] = []

    def flush_worker() -> None:
        """Worker that flushes client.

        Raises:
            Exception: Any exception during flush
        """
        try:
            client.flush()
        except Exception as e:
            exceptions.append(e)

    # Launch 5 threads calling flush simultaneously
    threads = [threading.Thread(target=flush_worker) for _ in range(5)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Verify no exceptions
    assert len(exceptions) == 0
