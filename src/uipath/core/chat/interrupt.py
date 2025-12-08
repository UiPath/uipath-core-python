"""Interrupt events for human-in-the-loop patterns."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class UiPathConversationInterruptStartEvent(BaseModel):
    """Signals the start of an interrupt - a pause point where the agent needs external input."""

    type: str
    value: Any

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class UiPathConversationInterruptEndEvent(BaseModel):
    """Signals the interrupt end event with the provided value."""

    # Can be any type
    model_config = ConfigDict(
        validate_by_name=True, validate_by_alias=True, extra="allow"
    )


class UiPathConversationInterruptEvent(BaseModel):
    """Encapsulates interrupt-related events within a message."""

    interrupt_id: str = Field(..., alias="interruptId")
    start: UiPathConversationInterruptStartEvent | None = Field(
        None, alias="startInterrupt"
    )
    end: Any | None = Field(None, alias="endInterrupt")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
