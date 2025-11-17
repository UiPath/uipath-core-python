"""Anthropic response parser for extracting GenAI semantic convention attributes."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def parse_anthropic_response(response: Any) -> dict[str, Any]:
    """Parse Anthropic Message response and extract attributes.

    Extracts attributes following OpenTelemetry GenAI semantic conventions:
    - gen_ai.response.model
    - gen_ai.response.id
    - gen_ai.response.finish_reasons (stop_reason)
    - gen_ai.usage.input_tokens
    - gen_ai.usage.output_tokens
    - gen_ai.response.output_text (concatenated content)

    Args:
        response: Anthropic Message response

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

        # Stop reason (finish reason)
        if hasattr(response, "stop_reason") and response.stop_reason:
            attributes["gen_ai.response.finish_reasons"] = json.dumps([response.stop_reason])

        # Content
        if hasattr(response, "content") and response.content:
            # Concatenate all text content blocks
            text_parts = []
            for block in response.content:
                if hasattr(block, "type") and block.type == "text":
                    if hasattr(block, "text"):
                        text_parts.append(block.text)

            if text_parts:
                attributes["gen_ai.response.output_text"] = "".join(text_parts)

        # Usage/tokens
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "input_tokens"):
                attributes["gen_ai.usage.input_tokens"] = usage.input_tokens
            if hasattr(usage, "output_tokens"):
                attributes["gen_ai.usage.output_tokens"] = usage.output_tokens
            # Calculate total
            if "gen_ai.usage.input_tokens" in attributes and "gen_ai.usage.output_tokens" in attributes:
                attributes["gen_ai.usage.total_tokens"] = (
                    attributes["gen_ai.usage.input_tokens"]
                    + attributes["gen_ai.usage.output_tokens"]
                )

        # Role
        if hasattr(response, "role"):
            attributes["gen_ai.response.role"] = response.role

        logger.debug("Parsed Anthropic response: %d attributes", len(attributes))

    except Exception as e:
        logger.warning("Error parsing Anthropic response: %s", e, exc_info=True)

    return attributes
