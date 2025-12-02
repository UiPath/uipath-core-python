"""OpenAI integration for UiPath Telemetry.

This integration provides response parsing for OpenAI SDK responses.
"""

from ._parser import parse_openai_response

__all__ = ["parse_openai_response"]
