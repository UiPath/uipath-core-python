"""The top-level event type representing an event in a conversation.

This is the root container for all other event subtypes (conversation start,
exchanges, messages, content, citations, tool calls, and async streams).
"""

from pydantic import BaseModel, ConfigDict, Field

from .async_stream import UiPathConversationAsyncInputStreamEvent
from .conversation import (
    UiPathConversationEndEvent,
    UiPathConversationStartedEvent,
    UiPathConversationStartEvent,
)
from .exchange import UiPathConversationExchangeEvent
from .meta import UiPathConversationMetaEvent
from .tool import UiPathConversationToolCallEvent


class UiPathConversationEvent(BaseModel):
    """The top-level event type representing an event in a conversation.

    This is the root container for all other event subtypes (conversation start,
    exchanges, messages, content, citations, tool calls, and async streams).
    """

    conversation_id: str = Field(
        ...,
        alias="conversationId",
        description="A globally unique identifier for conversation to which the other sub-event and data properties apply.",
    )
    start: UiPathConversationStartEvent | None = Field(
        None,
        description="Signals the start of an event stream concerning a conversation. This event does NOT necessarily mean this is a brand new conversation. It may be a continuation of an existing conversation.",
    )
    started: UiPathConversationStartedEvent | None = Field(
        None, description="Signals the acceptance of the start of a conversation."
    )
    end: UiPathConversationEndEvent | None = Field(
        None,
        description="Signals the end of a conversation event stream. This does NOT mean the conversation is over. A new event stream for the conversation could be started in the future.",
    )
    exchange: UiPathConversationExchangeEvent | None = Field(
        None,
        description="Encapsulates sub-events related to an exchange within a conversation.",
    )
    async_input_stream: UiPathConversationAsyncInputStreamEvent | None = Field(
        None,
        alias="asyncInputStream",
        description="Encapsulates sub-events related to an asynchronous input stream.",
    )
    async_tool_call: UiPathConversationToolCallEvent | None = Field(
        None,
        alias="asyncToolCall",
        description="Optional async tool call sub-event. This feature is not supported by all LLMs. Most tool calls are scoped to a message, and use the toolCall and toolResult properties defined by the ConversationMessage type.",
    )
    meta_event: UiPathConversationMetaEvent | None = Field(
        None,
        alias="metaEvent",
        description="Allows additional events to be sent in the context of the enclosing event stream.",
    )

    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)
