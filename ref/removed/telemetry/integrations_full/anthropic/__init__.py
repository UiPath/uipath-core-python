"""Anthropic integration for UiPath Telemetry.

This integration provides response parsing for Anthropic SDK responses.
"""

from ._parser import parse_anthropic_response

__all__ = ["parse_anthropic_response"]
