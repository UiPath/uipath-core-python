"""UiPath common exceptions."""


class UiPathFaultedTriggerError(Exception):
    """UiPath resume trigger error."""

    message: str


class UiPathPendingTriggerError(UiPathFaultedTriggerError):
    """Custom resume trigger error for pending triggers."""

    pass
