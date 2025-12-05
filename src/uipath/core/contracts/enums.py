"""UiPath common enum."""

from enum import Enum


class InternalTriggerMarker(str, Enum):
    """UiPath internal trigger markers."""

    NO_CONTENT = "NO_CONTENT"
