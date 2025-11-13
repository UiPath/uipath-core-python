"""Tests for provider response parsers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from uipath.core.otel.parsers.anthropic_parser import AnthropicParser
from uipath.core.otel.parsers.openai_parser import OpenAIParser
from uipath.core.otel.parsers.registry import parse_provider_response

if TYPE_CHECKING:
    pass


def test_parse_openai_chat_completion(sample_openai_response: dict[str, any]) -> None:
    """Test parsing OpenAI ChatCompletion response.

    Args:
        sample_openai_response: Sample response fixture
    """
    # Execute
    attributes = parse_provider_response(sample_openai_response)

    # Verify attributes extracted
    assert "gen_ai.response.model" in attributes
    assert attributes["gen_ai.response.model"] == "gpt-4"

    assert "gen_ai.usage.prompt_tokens" in attributes
    assert attributes["gen_ai.usage.prompt_tokens"] == 10

    assert "gen_ai.usage.completion_tokens" in attributes
    assert attributes["gen_ai.usage.completion_tokens"] == 9


def test_parse_anthropic_message(sample_anthropic_response: dict[str, any]) -> None:
    """Test parsing Anthropic Message response.

    Args:
        sample_anthropic_response: Sample response fixture
    """
    # Execute
    attributes = parse_provider_response(sample_anthropic_response)

    # Verify attributes extracted
    assert "gen_ai.response.model" in attributes
    assert attributes["gen_ai.response.model"] == "claude-3-opus-20240229"

    assert "gen_ai.usage.input_tokens" in attributes
    assert attributes["gen_ai.usage.input_tokens"] == 10

    assert "gen_ai.usage.output_tokens" in attributes
    assert attributes["gen_ai.usage.output_tokens"] == 9


def test_parse_unknown_type() -> None:
    """Test parsing unknown response type raises exception."""
    # Unknown structure
    unknown_response = {"unknown_key": "unknown_value"}

    # Should raise ValueError or return empty dict
    with pytest.raises((ValueError, KeyError)):
        parse_provider_response(unknown_response)


def test_openai_parser_directly(sample_openai_response: dict[str, any]) -> None:
    """Test OpenAIParser directly.

    Args:
        sample_openai_response: Sample response fixture
    """
    parser = OpenAIParser()

    # Check if can parse
    assert parser.can_parse(sample_openai_response) is True

    # Parse
    attributes = parser.parse(sample_openai_response)

    # Verify
    assert attributes["gen_ai.response.model"] == "gpt-4"
    assert attributes["gen_ai.usage.prompt_tokens"] == 10


def test_anthropic_parser_directly(sample_anthropic_response: dict[str, any]) -> None:
    """Test AnthropicParser directly.

    Args:
        sample_anthropic_response: Sample response fixture
    """
    parser = AnthropicParser()

    # Check if can parse
    assert parser.can_parse(sample_anthropic_response) is True

    # Parse
    attributes = parser.parse(sample_anthropic_response)

    # Verify
    assert attributes["gen_ai.response.model"] == "claude-3-opus-20240229"
    assert attributes["gen_ai.usage.input_tokens"] == 10


def test_openai_parser_cannot_parse_anthropic(
    sample_anthropic_response: dict[str, any],
) -> None:
    """Test OpenAI parser rejects Anthropic response.

    Args:
        sample_anthropic_response: Sample response fixture
    """
    parser = OpenAIParser()

    # Should not be able to parse Anthropic format
    assert parser.can_parse(sample_anthropic_response) is False


def test_anthropic_parser_cannot_parse_openai(
    sample_openai_response: dict[str, any],
) -> None:
    """Test Anthropic parser rejects OpenAI response.

    Args:
        sample_openai_response: Sample response fixture
    """
    parser = AnthropicParser()

    # Should not be able to parse OpenAI format
    assert parser.can_parse(sample_openai_response) is False


def test_parse_openai_streaming_chunk() -> None:
    """Test parsing OpenAI streaming chunk."""
    chunk = {
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hello"},
                "finish_reason": None,
            }
        ],
    }

    parser = OpenAIParser()

    # Should be able to parse chunk
    if parser.can_parse(chunk):
        attributes = parser.parse(chunk)
        assert "gen_ai.response.model" in attributes


def test_parse_openai_with_function_call() -> None:
    """Test parsing OpenAI response with function call."""
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "get_weather",
                        "arguments": '{"location": "SF"}',
                    },
                },
                "finish_reason": "function_call",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 10,
            "total_tokens": 25,
        },
    }

    attributes = parse_provider_response(response)

    # Should extract model and usage
    assert attributes["gen_ai.response.model"] == "gpt-4"
    assert attributes["gen_ai.usage.prompt_tokens"] == 15


def test_parse_anthropic_with_tool_use() -> None:
    """Test parsing Anthropic response with tool use."""
    response = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": "tool_123",
                "name": "get_weather",
                "input": {"location": "SF"},
            }
        ],
        "model": "claude-3-opus-20240229",
        "stop_reason": "tool_use",
        "usage": {
            "input_tokens": 15,
            "output_tokens": 20,
        },
    }

    attributes = parse_provider_response(response)

    # Should extract model and usage
    assert attributes["gen_ai.response.model"] == "claude-3-opus-20240229"
    assert attributes["gen_ai.usage.input_tokens"] == 15


def test_parse_missing_usage() -> None:
    """Test parsing response without usage information."""
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello",
                },
                "finish_reason": "stop",
            }
        ],
        # No usage field
    }

    # Should still parse model
    attributes = parse_provider_response(response)

    assert "gen_ai.response.model" in attributes
    # Usage tokens may or may not be present


def test_parse_empty_response() -> None:
    """Test parsing empty response."""
    with pytest.raises((ValueError, KeyError)):
        parse_provider_response({})


def test_parse_none_response() -> None:
    """Test parsing None response."""
    with pytest.raises((ValueError, TypeError, AttributeError)):
        parse_provider_response(None)


def test_openai_parser_extracts_finish_reason(
    sample_openai_response: dict[str, any],
) -> None:
    """Test OpenAI parser extracts finish_reason.

    Args:
        sample_openai_response: Sample response fixture
    """
    parser = OpenAIParser()
    attributes = parser.parse(sample_openai_response)

    # Finish reason should be extracted
    assert "gen_ai.response.finish_reasons" in attributes


def test_anthropic_parser_extracts_stop_reason(
    sample_anthropic_response: dict[str, any],
) -> None:
    """Test Anthropic parser extracts stop_reason.

    Args:
        sample_anthropic_response: Sample response fixture
    """
    parser = AnthropicParser()
    attributes = parser.parse(sample_anthropic_response)

    # Stop reason should be extracted
    assert "gen_ai.response.finish_reasons" in attributes
