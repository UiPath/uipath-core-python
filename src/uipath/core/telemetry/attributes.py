"""Telemetry attribute enums for type-safe span and resource attributes.

Defines standard attribute keys for UiPath automation workflows
and semantic span types for observability.

Example:
    >>> from uipath.core.telemetry import ResourceAttr, SpanAttr, SpanType
    >>> span.set_attribute(ResourceAttr.JOB_ID, "job-123")
    >>> span.set_attribute(SpanAttr.EXECUTION_ID, "exec-123")
"""

from enum import Enum

__all__ = ("ResourceAttr", "SpanAttr", "SpanType")


class ResourceAttr(str, Enum):
    """Resource-level attributes (static, process-wide metadata).

    These attributes describe the entity producing telemetry and are
    typically set once during client initialization.
    """

    # Organization & tenant context
    ORG_ID = "uipath.org_id"
    TENANT_ID = "uipath.tenant_id"
    USER_ID = "uipath.user_id"

    # Job context (can be resource or span-level depending on usage)
    JOB_ID = "uipath.job_id"
    PROCESS_KEY = "uipath.process_key"
    FOLDER_KEY = "uipath.folder_key"


class SpanAttr(str, Enum):
    """Span-level attributes (dynamic, per-operation metadata).

    These attributes describe specific operations and can vary per span.
    """

    EXECUTION_ID = "execution.id"
    TYPE = "span.type"


class SpanType(str, Enum):
    """Semantic span types for observability categorization."""

    SPAN = "span"
    GENERATION = "generation"
    TOOL = "tool"
    AUTOMATION = "automation"
    ACTIVITY = "activity"
    WORKFLOW = "workflow"
    AGENT = "agent"
    CHAIN = "chain"
    RETRIEVER = "retriever"
