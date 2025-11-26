"""Semantic conventions for OpenTelemetry attributes.

This module provides a unified, domain-organized structure for all semantic
conventions used in the telemetry/ module. Attributes are organized by domain for
clarity and discoverability.

Architecture:
    - Attr.Common.*: Shared attributes (input/output, openinference)
    - Attr.LLM.*: LLM and generation-related attributes
    - Attr.Message.*: Message-level attributes for chat completions
    - Attr.Tool.*: Tool/function call attributes
    - Attr.Retriever.*: Retriever and RAG attributes
    - Attr.Document.*: Document attributes (for retrieval results)
    - Attr.Reranker.*: Reranker attributes
    - Attr.Embedding.*: Embedding attributes
    - Attr.Agent.*: Agent attributes
    - Attr.Graph.*: Graph execution attributes (LangGraph)
    - Attr.Checkpoint.*: Checkpoint attributes (LangGraph persistence)
    - Attr.State.*: State tracking attributes (LangGraph)
    - Attr.Run.*: LangChain/LangGraph run attributes
    - Attr.UiPath.*: UiPath-specific attributes
    - Attr.Error.*: Error attributes
    - Attr.Internal.*: Internal/custom attributes

Usage:
    from uipath.core.telemetry.attributes import Attr, SpanKind

    # Use domain-organized attributes
    span.set_attribute(Attr.Common.INPUT_VALUE, data)
    span.set_attribute(Attr.LLM.MODEL_NAME, "gpt-4")
    span.set_attribute(Attr.UiPath.EXECUTION_ID, exec_id)
    span.set_attribute(Attr.Common.OPENINFERENCE_SPAN_KIND, SpanKind.LLM)

References:
    - OpenInference Semantic Conventions: https://github.com/Arize-ai/openinference
    - OpenTelemetry Semantic Conventions: https://opentelemetry.io/docs/specs/semconv/
"""

from __future__ import annotations


class Attr:
    """Root semantic conventions namespace.

    Organized by domain for clarity and discoverability.
    """

    class Common:
        """Common attributes used across all span types."""

        OPENINFERENCE_SPAN_KIND = "openinference.span.kind"

        INPUT_VALUE = "input.value"
        INPUT_MIME_TYPE = "input.mime_type"
        OUTPUT_VALUE = "output.value"
        OUTPUT_MIME_TYPE = "output.mime_type"

        SESSION_ID = "session.id"
        USER_ID = "user.id"
        METADATA = "metadata"

    class LLM:
        """LLM and generation-related attributes."""

        MODEL_NAME = "llm.model_name"
        PROVIDER = "llm.provider"
        INVOCATION_PARAMETERS = "llm.invocation_parameters"

        INPUT_MESSAGES = "llm.input_messages"
        OUTPUT_MESSAGES = "llm.output_messages"

        TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
        TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
        TOKEN_COUNT_TOTAL = "llm.token_count.total"

        GEN_AI_SYSTEM = "gen_ai.system"
        GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
        GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
        GEN_AI_PROMPT = "gen_ai.prompt"
        GEN_AI_COMPLETION = "gen_ai.completion"
        GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
        GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
        GEN_AI_REQUEST_TEMPERATURE = "gen_ai.request.temperature"
        GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"

    class Message:
        """Message-level attributes for chat completions."""

        ROLE = "message.role"
        CONTENT = "message.content"
        NAME = "message.name"
        FUNCTION_CALL_NAME = "message.function_call.name"
        FUNCTION_CALL_ARGUMENTS_JSON = "message.function_call.arguments_json"
        TOOL_CALLS = "message.tool_calls"

    class Tool:
        """Tool/function call attributes."""

        NAME = "tool.name"
        DESCRIPTION = "tool.description"
        PARAMETERS = "tool.parameters"

        CALL_ID = "id"
        CALL_FUNCTION_NAME = "function.name"
        CALL_FUNCTION_ARGUMENTS_JSON = "function.arguments_json"

    class Retriever:
        """Retriever and RAG attributes."""

        TOP_K = "retriever.top_k"
        THRESHOLD = "retriever.threshold"
        DOCUMENT_COUNT = "retriever.document_count"
        DOCUMENTS = "retrieval.documents"
        QUERY = "retrieval.query"

    class Document:
        """Document attributes (for retrieval results)."""

        ID = "document.id"
        CONTENT = "document.content"
        METADATA = "document.metadata"
        SCORE = "document.score"

    class Reranker:
        """Reranker attributes."""

        MODEL = "reranker.model"
        TOP_K = "reranker.top_k"
        INPUT_DOCUMENT_COUNT = "reranker.input_document_count"
        OUTPUT_DOCUMENT_COUNT = "reranker.output_document_count"

    class Embedding:
        """Embedding attributes."""

        DIMENSIONS = "embedding.dimensions"
        VECTOR_DIMENSION = "embedding.vector_dimension"

    class Agent:
        """Agent attributes."""

        NAME = "agent.name"

    class Graph:
        """Graph execution attributes (LangGraph)."""

        NODES = "graph.nodes"
        EDGES = "graph.edges"
        ENTRY_POINT = "graph.entry_point"
        CONDITIONAL_EDGES = "graph.conditional_edges"

        NODE_ID = "graph.node.id"
        NODE_NAME = "graph.node.name"
        NODE_PARENT_ID = "graph.node.parent_id"

    class Checkpoint:
        """Checkpoint attributes (LangGraph persistence)."""

        ID = "checkpoint.id"
        TIMESTAMP = "checkpoint.timestamp"
        METADATA = "checkpoint.metadata"

    class State:
        """State tracking attributes (LangGraph)."""

        MESSAGES_ADDED = "state.messages.added"
        MESSAGES_REMOVED = "state.messages.removed"
        MESSAGES_TOTAL = "state.messages.total"
        ITERATION_DELTA = "state.iteration.delta"

    class Run:
        """LangChain/LangGraph run attributes."""

        ID = "run.id"
        TYPE = "run.type"
        PARENT_ID = "run.parent_id"

    class UiPath:
        """UiPath-specific attributes."""

        EXECUTION_ID = "execution.id"
        JOB_ID = "uipath.job_id"
        FOLDER_KEY = "uipath.folder_key"
        PROCESS_KEY = "uipath.process_key"
        WORKFLOW_ID = "uipath.workflow_id"
        TENANT_ID = "uipath.tenant_id"
        ORG_ID = "uipath.org_id"

        ACTIVITY_NAME = "activity.name"
        ACTIVITY_PACKAGE = "activity.package"
        ACTIVITY_VERSION = "activity.version"

        WORKFLOW_NAME = "workflow.name"
        WORKFLOW_VERSION = "workflow.version"

    class Error:
        """Error attributes."""

        TYPE = "error.type"
        RATE_LIMITED = "error.rate_limited"

    class Internal:
        """Internal/custom attributes."""

        PARSING_ERROR = "telemetry.parsing_error"
        PARSING_ERROR_TYPE = "telemetry.parsing_error_type"


class SpanKind:
    """OpenInference span kind enumeration.

    These values are used with Attr.Common.OPENINFERENCE_SPAN_KIND to classify
    spans according to the OpenInference semantic conventions.
    """

    CHAIN = "CHAIN"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    EMBEDDING = "EMBEDDING"
    TOOL = "TOOL"
    AGENT = "AGENT"
    RERANKER = "RERANKER"
