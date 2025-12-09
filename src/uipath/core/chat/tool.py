"""Tool call events."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .content import InlineOrExternal
from .error import UiPathConversationErrorEvent


class UiPathConversationToolCallResult(BaseModel):
    """Represents the result of a tool call execution."""

    timestamp: str | None = None
    value: InlineOrExternal | None = None
    is_error: bool | None = Field(None, alias="isError")
    cancelled: bool | None = None

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class UiPathConversationToolCall(BaseModel):
    """Represents a call to an external tool or function within a message."""

    tool_call_id: str = Field(..., alias="toolCallId")
    name: str
    input: InlineOrExternal | None = None
    timestamp: str | None = None
    result: UiPathConversationToolCallResult | None = None

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class UiPathConversationToolCallStartEvent(BaseModel):
    """Signals the start of a tool call."""

    tool_name: str = Field(..., alias="toolName")
    timestamp: str | None = None
    input: InlineOrExternal | None = None
    metadata: dict[str, Any] | None = Field(None, alias="metaData")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class UiPathConversationToolCallEndEvent(BaseModel):
    """Signals the end of a tool call."""

    timestamp: str | None = None
    output: Any = None
    is_error: bool | None = Field(None, alias="isError")
    cancelled: bool | None = None
    metadata: dict[str, Any] | None = Field(None, alias="metaData")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)


class UiPathConversationToolCallEvent(BaseModel):
    """Encapsulates the data related to a tool call event."""

    tool_call_id: str = Field(..., alias="toolCallId")
    start: UiPathConversationToolCallStartEvent | None = Field(
        None, alias="startToolCall"
    )
    end: UiPathConversationToolCallEndEvent | None = Field(None, alias="endToolCall")
    meta_event: dict[str, Any] | None = Field(None, alias="metaEvent")
    error: UiPathConversationErrorEvent | None = Field(None, alias="toolCallError")

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
