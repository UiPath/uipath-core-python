"""OpenAI response parser for extracting GenAI semantic convention attributes."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_openai_response(response: Any) -> dict[str, Any]:
    """Parse OpenAI ChatCompletion response and extract attributes.

    Extracts attributes following OpenTelemetry GenAI semantic conventions:
    - gen_ai.response.model
    - gen_ai.response.id
    - gen_ai.response.finish_reasons
    - gen_ai.usage.input_tokens
    - gen_ai.usage.output_tokens
    - gen_ai.response.output_text (first choice content)

    Args:
        response: OpenAI ChatCompletion response

    Returns:
        Dictionary of extracted attributes
    """
    attributes: dict[str, Any] = {}

    try:
        # Model
        if hasattr(response, "model"):
            attributes["gen_ai.response.model"] = response.model

        # Response ID
        if hasattr(response, "id"):
            attributes["gen_ai.response.id"] = response.id

        # Finish reasons
        if hasattr(response, "choices") and response.choices:
            finish_reasons = [choice.finish_reason for choice in response.choices]
            attributes["gen_ai.response.finish_reasons"] = json.dumps(finish_reasons)

            # Extract first choice content
            first_choice = response.choices[0]
            if hasattr(first_choice, "message"):
                message = first_choice.message
                if hasattr(message, "content") and message.content:
                    attributes["gen_ai.response.output_text"] = message.content

                # Tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    tool_call_data = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ]
                    attributes["gen_ai.response.tool_calls"] = json.dumps(tool_call_data)

        # Usage/tokens
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                attributes["gen_ai.usage.input_tokens"] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                attributes["gen_ai.usage.output_tokens"] = usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                attributes["gen_ai.usage.total_tokens"] = usage.total_tokens

        logger.debug("Parsed OpenAI response: %d attributes", len(attributes))

    except Exception as e:
        logger.warning("Error parsing OpenAI response: %s", e, exc_info=True)

    return attributes
