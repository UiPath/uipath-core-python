"""Tests for LangChain attribute extractors."""

from __future__ import annotations

from unittest.mock import Mock

from uipath.core.telemetry.attributes import Attr
from uipath.core.telemetry.integrations._shared import InstrumentationConfig
from uipath.core.telemetry.integrations.langchain._extractors import (
    extract_chain_attributes,
    extract_llm_attributes,
    extract_retriever_attributes,
    extract_tool_attributes,
)


class TestExtractLLMAttributes:
    """Test cases for extract_llm_attributes."""

    def test_extract_model_name(self) -> None:
        """Test extraction of model name."""
        run = Mock()
        run.extra = {"invocation_params": {"model_name": "gpt-4"}}
        run.inputs = {}
        run.outputs = None

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        assert attributes[Attr.LLM.MODEL_NAME] == "gpt-4"

    def test_extract_model_fallback(self) -> None:
        """Test extraction of model name with fallback to 'model' key."""
        run = Mock()
        run.extra = {"invocation_params": {"model": "gpt-3.5-turbo"}}
        run.inputs = {}
        run.outputs = None

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        assert attributes[Attr.LLM.MODEL_NAME] == "gpt-3.5-turbo"

    def test_extract_invocation_parameters(self) -> None:
        """Test extraction of invocation parameters."""
        run = Mock()
        run.extra = {
            "invocation_params": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        }
        run.inputs = {}
        run.outputs = None

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        params = attributes[Attr.LLM.INVOCATION_PARAMETERS]
        assert "temperature" in params
        assert "max_tokens" in params

    def test_extract_input_messages_dict(self) -> None:
        """Test extraction of input messages as dicts."""
        run = Mock()
        run.extra = {}
        run.inputs = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"type": "assistant", "content": "Hi there!"},
            ]
        }
        run.outputs = None

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        # Check first message
        role_key = f"llm.input_messages.0.{Attr.Message.ROLE}"
        content_key = f"llm.input_messages.0.{Attr.Message.CONTENT}"
        assert attributes[role_key] == "user"
        assert attributes[content_key] == "Hello"

        # Check second message
        role_key = f"llm.input_messages.1.{Attr.Message.ROLE}"
        content_key = f"llm.input_messages.1.{Attr.Message.CONTENT}"
        assert attributes[role_key] == "assistant"
        assert attributes[content_key] == "Hi there!"

    def test_extract_token_counts(self) -> None:
        """Test extraction of token usage."""
        run = Mock()
        run.extra = {}
        run.inputs = {}
        run.outputs = {
            "llm_output": {
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            }
        }

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        # OpenInference attributes
        assert attributes[Attr.LLM.TOKEN_COUNT_PROMPT] == 10
        assert attributes[Attr.LLM.TOKEN_COUNT_COMPLETION] == 20
        assert attributes[Attr.LLM.TOKEN_COUNT_TOTAL] == 30

        # Telemetry GenAI attributes
        assert attributes[Attr.LLM.GEN_AI_USAGE_INPUT_TOKENS] == 10
        assert attributes[Attr.LLM.GEN_AI_USAGE_OUTPUT_TOKENS] == 20

    def test_extract_provider_from_model_name(self) -> None:
        """Test provider inference from model name."""
        test_cases = [
            ("gpt-4", "openai"),
            ("gpt-3.5-turbo", "openai"),
            ("claude-2", "anthropic"),
            ("gemini-pro", "google"),
        ]

        for model_name, expected_provider in test_cases:
            run = Mock()
            run.extra = {"invocation_params": {"model_name": model_name}}
            run.inputs = {}
            run.outputs = None

            config = InstrumentationConfig()
            attributes = extract_llm_attributes(run, config)

            assert attributes[Attr.LLM.PROVIDER] == expected_provider
            assert attributes[Attr.LLM.GEN_AI_SYSTEM] == expected_provider

    def test_extract_gen_ai_attributes(self) -> None:
        """Test extraction of Telemetry GenAI semantic convention attributes."""
        run = Mock()
        run.extra = {
            "invocation_params": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 100,
            }
        }
        run.inputs = {}
        run.outputs = None

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        assert attributes[Attr.LLM.GEN_AI_REQUEST_MODEL] == "gpt-4"
        assert attributes[Attr.LLM.GEN_AI_REQUEST_TEMPERATURE] == 0.7
        assert attributes[Attr.LLM.GEN_AI_REQUEST_MAX_TOKENS] == 100

    def test_extract_input_output_values(self) -> None:
        """Test extraction of OpenInference input.value and output.value."""
        run = Mock()
        run.extra = {"invocation_params": {"model_name": "gpt-4"}}
        run.inputs = {"messages": [{"role": "user", "content": "Hello"}]}
        run.outputs = {
            "generations": [
                [{"text": "Hi there!", "message": {"content": "Hi there!"}}]
            ]
        }

        config = InstrumentationConfig()
        attributes = extract_llm_attributes(run, config)

        # Input value and MIME type
        assert Attr.Common.INPUT_VALUE in attributes
        assert attributes[Attr.Common.INPUT_MIME_TYPE] == "application/json"

        # Output value and MIME type
        assert Attr.Common.OUTPUT_VALUE in attributes
        assert attributes[Attr.Common.OUTPUT_MIME_TYPE] == "application/json"

        # Telemetry GenAI prompt and completion
        assert Attr.LLM.GEN_AI_PROMPT in attributes
        assert Attr.LLM.GEN_AI_COMPLETION in attributes

    def test_capture_inputs_false(self) -> None:
        """Test that inputs are not captured when config.capture_inputs is False."""
        run = Mock()
        run.extra = {"invocation_params": {"model_name": "gpt-4"}}
        run.inputs = {"messages": [{"role": "user", "content": "Hello"}]}
        run.outputs = None

        config = InstrumentationConfig(capture_inputs=False)
        attributes = extract_llm_attributes(run, config)

        # Model name should still be extracted
        assert attributes.get(Attr.LLM.MODEL_NAME) == "gpt-4"

        # But messages should not
        message_role_key = f"llm.input_messages.0.{Attr.Message.ROLE}"
        assert message_role_key not in attributes

    def test_max_array_items_limit(self) -> None:
        """Test that max_array_items limits message extraction."""
        run = Mock()
        run.extra = {}
        run.inputs = {
            "messages": [{"role": "user", "content": f"Message {i}"} for i in range(10)]
        }
        run.outputs = None

        config = InstrumentationConfig(max_array_items=3)
        attributes = extract_llm_attributes(run, config)

        # Only first 3 messages should be extracted
        third_message = f"llm.input_messages.2.{Attr.Message.ROLE}"
        fourth_message = f"llm.input_messages.3.{Attr.Message.ROLE}"

        assert third_message in attributes
        assert fourth_message not in attributes


class TestExtractChainAttributes:
    """Test cases for extract_chain_attributes."""

    def test_extract_chain_input(self) -> None:
        """Test extraction of chain input (OpenInference input.value)."""
        run = Mock()
        run.inputs = {
            "query": "What is the capital of France?",
            "context": ["doc1", "doc2"],
        }

        config = InstrumentationConfig()
        attributes = extract_chain_attributes(run, config)

        # OpenInference input.value
        input_str = attributes[Attr.Common.INPUT_VALUE]
        assert "query" in input_str
        assert "What is the capital of France?" in input_str

        # OpenInference input.mime_type
        assert attributes[Attr.Common.INPUT_MIME_TYPE] == "application/json"

    def test_capture_inputs_false(self) -> None:
        """Test that inputs are not captured when config.capture_inputs is False."""
        run = Mock()
        run.inputs = {"query": "test"}

        config = InstrumentationConfig(capture_inputs=False)
        attributes = extract_chain_attributes(run, config)

        assert Attr.Common.INPUT_VALUE not in attributes

    def test_truncation(self) -> None:
        """Test that long inputs are truncated."""
        run = Mock()
        run.inputs = {"data": "a" * 10000}

        config = InstrumentationConfig(max_string_length=100)
        attributes = extract_chain_attributes(run, config)

        input_str = attributes[Attr.Common.INPUT_VALUE]
        assert len(input_str) == 100
        assert input_str.endswith("...")


class TestExtractToolAttributes:
    """Test cases for extract_tool_attributes."""

    def test_extract_tool_name(self) -> None:
        """Test extraction of tool name."""
        run = Mock()
        run.name = "calculator"
        run.extra = {}
        run.inputs = {}

        config = InstrumentationConfig()
        attributes = extract_tool_attributes(run, config)

        assert attributes[Attr.Tool.NAME] == "calculator"

    def test_extract_tool_description(self) -> None:
        """Test extraction of tool description."""
        run = Mock()
        run.name = "calculator"
        run.extra = {"description": "A tool for performing calculations"}
        run.inputs = {}

        config = InstrumentationConfig()
        attributes = extract_tool_attributes(run, config)

        assert attributes[Attr.Tool.DESCRIPTION] == "A tool for performing calculations"

    def test_extract_tool_parameters(self) -> None:
        """Test extraction of tool parameters."""
        run = Mock()
        run.name = "calculator"
        run.extra = {}
        run.inputs = {"expression": "2 + 2", "precision": 2}

        config = InstrumentationConfig()
        attributes = extract_tool_attributes(run, config)

        params = attributes[Attr.Tool.PARAMETERS]
        assert "expression" in params
        assert "precision" in params

        # OpenInference input.value
        assert Attr.Common.INPUT_VALUE in attributes
        assert attributes[Attr.Common.INPUT_MIME_TYPE] == "application/json"


class TestExtractRetrieverAttributes:
    """Test cases for extract_retriever_attributes."""

    def test_extract_retrieval_query(self) -> None:
        """Test extraction of retrieval query."""
        run = Mock()
        run.inputs = {"query": "What is OpenTelemetry?"}
        run.outputs = {}

        config = InstrumentationConfig()
        attributes = extract_retriever_attributes(run, config)

        assert attributes[Attr.Retriever.QUERY] == "What is OpenTelemetry?"

        # OpenInference input.value
        assert attributes[Attr.Common.INPUT_VALUE] == "What is OpenTelemetry?"
        assert attributes[Attr.Common.INPUT_MIME_TYPE] == "text/plain"

    def test_extract_query_from_input_key(self) -> None:
        """Test extraction using 'input' key fallback."""
        run = Mock()
        run.inputs = {"input": "fallback query"}
        run.outputs = {}

        config = InstrumentationConfig()
        attributes = extract_retriever_attributes(run, config)

        assert attributes[Attr.Retriever.QUERY] == "fallback query"

    def test_extract_documents(self) -> None:
        """Test extraction of retrieved documents."""
        run = Mock()
        run.inputs = {"query": "test"}
        run.outputs = {
            "documents": [
                {
                    "id": "doc1",
                    "page_content": "Document content 1",
                    "metadata": {"source": "file1.txt"},
                    "score": 0.95,
                },
                {
                    "page_content": "Document content 2",
                    "metadata": {"source": "file2.txt"},
                },
            ]
        }

        config = InstrumentationConfig()
        attributes = extract_retriever_attributes(run, config)

        # Check first document
        doc0_id = f"retrieval.documents.0.{Attr.Document.ID}"
        doc0_content = f"retrieval.documents.0.{Attr.Document.CONTENT}"
        doc0_score = f"retrieval.documents.0.{Attr.Document.SCORE}"

        assert attributes[doc0_id] == "doc1"
        assert "Document content 1" in attributes[doc0_content]
        assert attributes[doc0_score] == 0.95

        # Check second document (no ID or score)
        doc1_content = f"retrieval.documents.1.{Attr.Document.CONTENT}"
        assert "Document content 2" in attributes[doc1_content]

        # OpenInference output.value
        assert Attr.Common.OUTPUT_VALUE in attributes
        assert attributes[Attr.Common.OUTPUT_MIME_TYPE] == "application/json"

    def test_max_documents_limit(self) -> None:
        """Test that max_array_items limits document extraction."""
        run = Mock()
        run.inputs = {"query": "test"}
        run.outputs = {"documents": [{"page_content": f"Doc {i}"} for i in range(10)]}

        config = InstrumentationConfig(max_array_items=3)
        attributes = extract_retriever_attributes(run, config)

        # Only first 3 documents should be extracted
        doc2_content = f"retrieval.documents.2.{Attr.Document.CONTENT}"
        doc3_content = f"retrieval.documents.3.{Attr.Document.CONTENT}"

        assert doc2_content in attributes
        assert doc3_content not in attributes
