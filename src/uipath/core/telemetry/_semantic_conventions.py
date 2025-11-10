"""UiPath-specific semantic conventions for span attributes.

Defines standard attribute keys for UiPath automation workflows
and semantic span types for observability.

Example:
    >>> from uipath.core.telemetry._semantic_conventions import (
    ...     UIPATH_JOB_ID,
    ...     SPAN_TYPE_AUTOMATION
    ... )
    >>> span.set_attribute(UIPATH_JOB_ID, "job-123")
"""

# UiPath Resource Attributes (static, low cardinality)
# These are set once per client at initialization
UIPATH_ORG_ID = "uipath.org_id"
UIPATH_TENANT_ID = "uipath.tenant_id"
UIPATH_USER_ID = "uipath.user_id"

# UiPath Span Attributes (dynamic, high cardinality)
# These can vary per span and represent business context
UIPATH_JOB_ID = "uipath.job_id"
UIPATH_PROCESS_KEY = "uipath.process_key"
UIPATH_FOLDER_KEY = "uipath.folder_key"
UIPATH_EXECUTION_ID = "execution.id"

# Span metadata attributes
SPAN_TYPE_ATTRIBUTE = "span.type"

# Semantic span types (UiPath-specific + LLM standard)
# Used via semantic_type parameter in start_as_current_span()
SPAN_TYPE_SPAN = "span"  # Default generic span
SPAN_TYPE_GENERATION = "generation"  # LLM call
SPAN_TYPE_TOOL = "tool"  # Tool execution
SPAN_TYPE_AUTOMATION = "automation"  # UiPath workflow
SPAN_TYPE_ACTIVITY = "activity"  # UiPath activity
SPAN_TYPE_WORKFLOW = "workflow"  # UiPath workflow (alias)
SPAN_TYPE_AGENT = "agent"  # AI agent
SPAN_TYPE_CHAIN = "chain"  # LangChain-style chain
SPAN_TYPE_RETRIEVER = "retriever"  # RAG retrieval
