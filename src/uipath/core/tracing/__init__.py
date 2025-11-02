"""OpenTelemetry tracing module with UiPath integration.

This module provides decorators and utilities for instrumenting Python functions
with OpenTelemetry tracing, including custom processors for UiPath execution tracking.
"""

from uipath.core.tracing.context import UiPathTraceContext
from uipath.core.tracing.decorators import traced
from uipath.core.tracing.manager import UiPathTracingManager

__all__ = [
    "traced",
    "UiPathTraceContext",
    "UiPathTracingManager",
]
