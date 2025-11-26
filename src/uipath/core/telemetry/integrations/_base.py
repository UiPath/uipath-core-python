"""Base instrumentor class for UiPath framework integrations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from opentelemetry.instrumentation.instrumentor import (  # type: ignore[attr-defined]
    BaseInstrumentor,
)

if TYPE_CHECKING:
    from ..client import TelemetryClient


class UiPathInstrumentor(BaseInstrumentor):
    """Base class for all UiPath framework instrumentors.

    Provides access to UiPath Telemetry client and shared utilities.

    Subclasses must implement:
    - instrumentation_dependencies(): Return supported framework versions
    - _instrument(**kwargs): Apply framework-specific patching
    - _uninstrument(**kwargs): Remove instrumentation and restore originals
    """

    def __init__(self) -> None:
        """Initialize instrumentor."""
        super().__init__()
        self._telemetry_client: TelemetryClient | None = None

    def _get_telemetry_client(self) -> TelemetryClient:
        """Get UiPath Telemetry client.

        Returns:
            TelemetryClient instance

        Raises:
            RuntimeError: If client not initialized
        """
        if self._telemetry_client is None:
            from ..client import get_client

            self._telemetry_client = get_client()
        return self._telemetry_client
