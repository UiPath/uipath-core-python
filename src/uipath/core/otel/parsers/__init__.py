"""Provider response parsers for extracting OpenTelemetry attributes.

This package provides parsers for common LLM providers (OpenAI, Anthropic)
that extract standard GenAI semantic convention attributes.
"""

from .registry import parse_provider_response, register_parser

__all__ = ["parse_provider_response", "register_parser"]
