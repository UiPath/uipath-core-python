"""Attribute extractors for LangChain Run objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...attributes import Attr
from .._shared import (
    InstrumentationConfig,
    safe_json_dumps,
    truncate_string,
)

if TYPE_CHECKING:
    from langchain_core.tracers.schemas import Run


def extract_llm_attributes(run: Run, config: InstrumentationConfig) -> dict[str, Any]:
    """Extract LLM-specific attributes from Run object.

    Args:
        run: LangChain Run object for LLM execution
        config: Instrumentation configuration

    Returns:
        Dictionary of attribute name to value mappings
    """
    attributes: dict[str, Any] = {}

    # Extract invocation parameters
    invocation_params = {}
    model_name = None
    if hasattr(run, "extra") and run.extra:
        invocation_params = run.extra.get("invocation_params", {})

        # Model name (OpenInference)
        if "model_name" in invocation_params:
            model_name = invocation_params["model_name"]
            attributes[Attr.LLM.MODEL_NAME] = model_name
        elif "model" in invocation_params:
            model_name = invocation_params["model"]
            attributes[Attr.LLM.MODEL_NAME] = model_name

        # Provider (OpenInference)
        # Try to extract provider from invocation_params or model name
        provider = invocation_params.get("_type")
        if not provider and model_name:
            # Infer provider from model name
            if "gpt" in model_name.lower() or "openai" in model_name.lower():
                provider = "openai"
            elif "claude" in model_name.lower():
                provider = "anthropic"
            elif "gemini" in model_name.lower():
                provider = "google"
        if provider:
            attributes[Attr.LLM.PROVIDER] = provider
            # OTel GenAI system attribute
            attributes[Attr.LLM.GEN_AI_SYSTEM] = provider

        # OTel GenAI request model
        if model_name:
            attributes[Attr.LLM.GEN_AI_REQUEST_MODEL] = model_name

        # Temperature (OTel GenAI)
        if "temperature" in invocation_params:
            attributes[Attr.LLM.GEN_AI_REQUEST_TEMPERATURE] = invocation_params[
                "temperature"
            ]

        # Max tokens (OTel GenAI)
        if "max_tokens" in invocation_params:
            attributes[Attr.LLM.GEN_AI_REQUEST_MAX_TOKENS] = invocation_params[
                "max_tokens"
            ]

        # Invocation parameters (OpenInference)
        if invocation_params and config.capture_inputs:
            params_str = safe_json_dumps(invocation_params, config.max_string_length)
            attributes[Attr.LLM.INVOCATION_PARAMETERS] = params_str

    # Input messages
    if run.inputs and config.capture_inputs:
        messages = run.inputs.get("messages", [])
        if messages:
            # OpenInference structured messages
            _extract_messages(messages, attributes, is_input=True, config=config)

            # OpenInference input.value
            input_str = safe_json_dumps(messages, config.max_string_length)
            attributes[Attr.Common.INPUT_VALUE] = input_str
            attributes[Attr.Common.INPUT_MIME_TYPE] = "application/json"

            # OTel GenAI prompt
            attributes[Attr.LLM.GEN_AI_PROMPT] = input_str

    # Output messages and token counts
    if run.outputs and config.capture_outputs:
        # Extract output messages
        generations = run.outputs.get("generations", [])
        if generations and len(generations) > 0:
            _extract_output_generations(generations, attributes, config)

            # OpenInference output.value
            output_str = safe_json_dumps(generations, config.max_string_length)
            attributes[Attr.Common.OUTPUT_VALUE] = output_str
            attributes[Attr.Common.OUTPUT_MIME_TYPE] = "application/json"

            # OTel GenAI completion
            attributes[Attr.LLM.GEN_AI_COMPLETION] = output_str

        # Extract token usage
        llm_output = run.outputs.get("llm_output", {})
        if llm_output:
            _extract_token_counts(llm_output, attributes)

            # OTel GenAI response model (may differ from request)
            response_model = llm_output.get("model_name")
            if response_model:
                attributes[Attr.LLM.GEN_AI_RESPONSE_MODEL] = response_model

    return attributes


def extract_chain_attributes(run: Run, config: InstrumentationConfig) -> dict[str, Any]:
    """Extract chain-specific attributes from Run object.

    Args:
        run: LangChain Run object for chain execution
        config: Instrumentation configuration

    Returns:
        Dictionary of attribute name to value mappings
    """
    attributes: dict[str, Any] = {}

    # Input (OpenInference)
    if run.inputs and config.capture_inputs:
        input_str = safe_json_dumps(run.inputs)
        if len(input_str) > config.max_string_length:
            input_str = truncate_string(input_str, config.max_string_length)
        attributes[Attr.Common.INPUT_VALUE] = input_str
        attributes[Attr.Common.INPUT_MIME_TYPE] = "application/json"

    # Output (will be set during _on_run_update)
    # We don't set it here because outputs aren't available until run completes

    return attributes


def extract_tool_attributes(run: Run, config: InstrumentationConfig) -> dict[str, Any]:
    """Extract tool-specific attributes from Run object.

    Args:
        run: LangChain Run object for tool execution
        config: Instrumentation configuration

    Returns:
        Dictionary of attribute name to value mappings
    """
    attributes: dict[str, Any] = {}

    # Tool name (OpenInference)
    if run.name:
        attributes[Attr.Tool.NAME] = run.name

    # Tool description (OpenInference)
    if hasattr(run, "extra") and run.extra:
        description = run.extra.get("description")
        if description:
            attributes[Attr.Tool.DESCRIPTION] = description

    # Tool input parameters (OpenInference)
    if run.inputs and config.capture_inputs:
        params_str = safe_json_dumps(run.inputs)
        if len(params_str) > config.max_string_length:
            params_str = truncate_string(params_str, config.max_string_length)
        attributes[Attr.Tool.PARAMETERS] = params_str

        # OpenInference input.value (general compatibility)
        attributes[Attr.Common.INPUT_VALUE] = params_str
        attributes[Attr.Common.INPUT_MIME_TYPE] = "application/json"

    return attributes


def extract_retriever_attributes(run: Run, config: InstrumentationConfig) -> dict[str, Any]:
    """Extract retriever-specific attributes from Run object.

    Args:
        run: LangChain Run object for retriever execution
        config: Instrumentation configuration

    Returns:
        Dictionary of attribute name to value mappings
    """
    attributes: dict[str, Any] = {}

    # Query (OpenInference)
    if run.inputs and config.capture_inputs:
        query = run.inputs.get("query") or run.inputs.get("input")
        if query:
            if isinstance(query, str):
                query_str = truncate_string(query, config.max_string_length)
                attributes[Attr.Retriever.QUERY] = query_str
                # OpenInference input.value
                attributes[Attr.Common.INPUT_VALUE] = query_str
                attributes[Attr.Common.INPUT_MIME_TYPE] = "text/plain"
            else:
                query_str = safe_json_dumps(query)
                query_str = truncate_string(query_str, config.max_string_length)
                attributes[Attr.Retriever.QUERY] = query_str
                # OpenInference input.value
                attributes[Attr.Common.INPUT_VALUE] = query_str
                attributes[Attr.Common.INPUT_MIME_TYPE] = "application/json"

    # Retrieved documents (from outputs)
    if run.outputs and config.capture_outputs:
        documents = run.outputs.get("documents", [])
        if documents:
            _extract_documents(documents, attributes, config)

            # OpenInference output.value
            output_str = safe_json_dumps(documents, config.max_string_length)
            attributes[Attr.Common.OUTPUT_VALUE] = output_str
            attributes[Attr.Common.OUTPUT_MIME_TYPE] = "application/json"

    return attributes


def _extract_messages(
    messages: list[Any],
    attributes: dict[str, Any],
    is_input: bool,
    config: InstrumentationConfig,
) -> None:
    """Extract message attributes from message list.

    Args:
        messages: List of message objects
        attributes: Dictionary to populate with attributes
        is_input: Whether these are input messages (vs output)
        config: Instrumentation configuration
    """
    max_messages = min(len(messages), config.max_array_items)

    for i in range(max_messages):
        msg = messages[i]

        # Type guard: Try importing LangChain message types
        try:
            from langchain_core.messages import BaseMessage

            if isinstance(msg, BaseMessage):
                # Handle LangChain message objects
                role = getattr(msg, "type", "unknown")
                content = getattr(msg, "content", "")
            elif isinstance(msg, dict):
                # Handle dict messages
                role = msg.get("role") or msg.get("type", "unknown")
                content = msg.get("content", "")
            else:
                # Skip unexpected types
                continue
        except ImportError:
            # Fallback if langchain_core not available
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type", "unknown")
                content = msg.get("content", "")
            else:
                # Skip unexpected types
                continue

        # Set role attribute
        if is_input:
            role_key = f"llm.input_messages.{i}.{Attr.Message.ROLE}"
        else:
            role_key = f"llm.output_messages.{i}.{Attr.Message.ROLE}"
        attributes[role_key] = role

        # Set content attribute
        if isinstance(content, str):
            content_str = truncate_string(content, config.max_string_length)
        else:
            content_str = safe_json_dumps(content, config.max_string_length)

        if is_input:
            content_key = f"llm.input_messages.{i}.{Attr.Message.CONTENT}"
        else:
            content_key = f"llm.output_messages.{i}.{Attr.Message.CONTENT}"
        attributes[content_key] = content_str


def _extract_output_generations(
    generations: list[Any], attributes: dict[str, Any], config: InstrumentationConfig
) -> None:
    """Extract output message attributes from LLM generations.

    Args:
        generations: List of generation lists
        attributes: Dictionary to populate with attributes
        config: Instrumentation configuration
    """
    # Flatten generations (usually list of lists)
    all_generations = []
    for gen_list in generations:
        if isinstance(gen_list, list):
            all_generations.extend(gen_list)
        else:
            all_generations.append(gen_list)

    max_generations = min(len(all_generations), config.max_array_items)

    for i in range(max_generations):
        gen = all_generations[i]

        # Type guard: Extract message from generation
        try:
            from langchain_core.outputs import Generation

            if isinstance(gen, Generation):
                # Handle LangChain Generation objects
                message = getattr(gen, "message", None)
                text = getattr(gen, "text", "")
            elif isinstance(gen, dict):
                # Handle dict generations
                message = gen.get("message", {})
                text = gen.get("text", "")
            else:
                # Skip unexpected types
                continue
        except ImportError:
            # Fallback if langchain_core not available
            if isinstance(gen, dict):
                message = gen.get("message", {})
                text = gen.get("text", "")
            else:
                # Skip unexpected types
                continue

        # If we have a structured message, extract it
        if message:
            if isinstance(message, dict):
                role = message.get("role") or message.get("type", "assistant")
                content = message.get("content", text)
            else:
                role = getattr(message, "type", "assistant")
                content = getattr(message, "content", text)

            # Set attributes
            role_key = f"llm.output_messages.{i}.{Attr.Message.ROLE}"
            attributes[role_key] = role

            content_str = truncate_string(str(content), config.max_string_length)
            content_key = f"llm.output_messages.{i}.{Attr.Message.CONTENT}"
            attributes[content_key] = content_str
        elif text:
            # Fallback to text if no structured message
            role_key = f"llm.output_messages.{i}.{Attr.Message.ROLE}"
            attributes[role_key] = "assistant"

            content_str = truncate_string(text, config.max_string_length)
            content_key = f"llm.output_messages.{i}.{Attr.Message.CONTENT}"
            attributes[content_key] = content_str


def _extract_token_counts(llm_output: dict[str, Any], attributes: dict[str, Any]) -> None:
    """Extract token count attributes from LLM output.

    Args:
        llm_output: LLM output dictionary containing token usage
        attributes: Dictionary to populate with attributes
    """
    token_usage = llm_output.get("token_usage", {})
    if not token_usage:
        return

    # Prompt tokens (OpenInference)
    if "prompt_tokens" in token_usage:
        prompt_tokens = token_usage["prompt_tokens"]
        attributes[Attr.LLM.TOKEN_COUNT_PROMPT] = prompt_tokens
        # OTel GenAI input tokens
        attributes[Attr.LLM.GEN_AI_USAGE_INPUT_TOKENS] = prompt_tokens

    # Completion tokens (OpenInference)
    if "completion_tokens" in token_usage:
        completion_tokens = token_usage["completion_tokens"]
        attributes[Attr.LLM.TOKEN_COUNT_COMPLETION] = completion_tokens
        # OTel GenAI output tokens
        attributes[Attr.LLM.GEN_AI_USAGE_OUTPUT_TOKENS] = completion_tokens

    # Total tokens (OpenInference)
    if "total_tokens" in token_usage:
        attributes[Attr.LLM.TOKEN_COUNT_TOTAL] = token_usage["total_tokens"]


def _extract_documents(
    documents: list[Any], attributes: dict[str, Any], config: InstrumentationConfig
) -> None:
    """Extract document attributes from retrieved documents.

    Args:
        documents: List of document objects
        attributes: Dictionary to populate with attributes
        config: Instrumentation configuration
    """
    max_docs = min(len(documents), config.max_array_items)

    for i in range(max_docs):
        doc = documents[i]

        # Type guard: Handle both dict and object documents
        try:
            from langchain_core.documents import Document

            if isinstance(doc, Document):
                # Handle LangChain Document objects
                doc_id = getattr(doc, "id", None)
                content = getattr(doc, "page_content", "")
                metadata = getattr(doc, "metadata", {})
                score = getattr(doc, "score", None)
            elif isinstance(doc, dict):
                # Handle dict documents
                doc_id = doc.get("id")
                content = doc.get("page_content") or doc.get("content", "")
                metadata = doc.get("metadata", {})
                score = doc.get("score")
            else:
                # Skip unexpected types
                continue
        except ImportError:
            # Fallback if langchain_core not available
            if isinstance(doc, dict):
                doc_id = doc.get("id")
                content = doc.get("page_content") or doc.get("content", "")
                metadata = doc.get("metadata", {})
                score = doc.get("score")
            else:
                # Skip unexpected types
                continue

        # Document ID
        if doc_id:
            id_key = f"retrieval.documents.{i}.{Attr.Document.ID}"
            attributes[id_key] = str(doc_id)

        # Document content
        if content:
            content_str = truncate_string(str(content), config.max_string_length)
            content_key = f"retrieval.documents.{i}.{Attr.Document.CONTENT}"
            attributes[content_key] = content_str

        # Document metadata
        if metadata:
            metadata_str = safe_json_dumps(metadata, config.max_string_length)
            metadata_key = f"retrieval.documents.{i}.{Attr.Document.METADATA}"
            attributes[metadata_key] = metadata_str

        # Document score
        if score is not None:
            score_key = f"retrieval.documents.{i}.{Attr.Document.SCORE}"
            attributes[score_key] = float(score)
