"""UiPath Core Package."""

from uipath.core.tracing.decorators import traced
from uipath.core.tracing.manager import UiPathTracingManager

__all__ = [
    "traced",
    "UiPathTracingManager",
]
