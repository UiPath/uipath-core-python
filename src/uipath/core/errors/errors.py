"""UiPath common exceptions."""

from enum import Enum


class ErrorCategory(str, Enum):
    """Categories of UiPath errors."""

    DEPLOYMENT = "Deployment"
    SYSTEM = "System"
    UNKNOWN = "Unknown"
    USER = "User"


class UiPathFaultedTriggerError(Exception):
    """UiPath resume trigger error."""

    category: ErrorCategory
    message: str


class UiPathPendingTriggerError(UiPathFaultedTriggerError):
    """Custom resume trigger error for pending triggers."""

    pass
