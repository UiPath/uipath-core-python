"""Parser registry for provider response parsing.

This module provides a registry for parser functions that extract attributes
from provider responses (OpenAI, Anthropic, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type for parser functions
ParserFunc = Callable[[Any], dict[str, Any]]

# Registry mapping type names to parser functions
_PARSER_REGISTRY: dict[str, ParserFunc] = {}


def register_parser(type_name: str, parser: ParserFunc) -> None:
    """Register a parser for a specific provider response type.

    Args:
        type_name: Fully qualified type name (e.g., "openai.ChatCompletion")
        parser: Parser function that extracts attributes
    """
    _PARSER_REGISTRY[type_name] = parser
    logger.debug("Registered parser for type: %s", type_name)


def parse_provider_response(response: Any) -> dict[str, Any]:
    """Parse provider response and extract attributes.

    Attempts to match response type against registered parsers.

    Args:
        response: Provider response object

    Returns:
        Dictionary of extracted attributes (GenAI semantic conventions)

    Raises:
        ValueError: If no parser found for response type
    """
    # Get fully qualified type name
    type_name = f"{response.__class__.__module__}.{response.__class__.__name__}"

    # Try exact match
    if type_name in _PARSER_REGISTRY:
        parser = _PARSER_REGISTRY[type_name]
        logger.debug("Found parser for type: %s", type_name)
        return parser(response)

    # Try simple class name match (fallback)
    simple_name = response.__class__.__name__
    for registered_type, parser in _PARSER_REGISTRY.items():
        if registered_type.endswith(f".{simple_name}"):
            logger.debug(
                "Found parser for type via simple name: %s -> %s",
                simple_name,
                registered_type,
            )
            return parser(response)

    # No parser found
    raise ValueError(
        f"No parser registered for type: {type_name} "
        f"(simple name: {simple_name})"
    )


def can_parse(response: Any) -> bool:
    """Check if response type has a registered parser.

    Args:
        response: Provider response object

    Returns:
        True if parser is available
    """
    type_name = f"{response.__class__.__module__}.{response.__class__.__name__}"
    if type_name in _PARSER_REGISTRY:
        return True

    # Check simple name match
    simple_name = response.__class__.__name__
    for registered_type in _PARSER_REGISTRY:
        if registered_type.endswith(f".{simple_name}"):
            return True

    return False


# Auto-register parsers on import
def _register_default_parsers() -> None:
    """Register default parsers for common providers."""
    try:
        from .openai_parser import parse_openai_response

        register_parser("openai.ChatCompletion", parse_openai_response)
        register_parser("openai.types.chat.ChatCompletion", parse_openai_response)
    except ImportError:
        logger.debug("OpenAI parser not available (openai not installed)")

    try:
        from .anthropic_parser import parse_anthropic_response

        register_parser("anthropic.Message", parse_anthropic_response)
        register_parser("anthropic.types.Message", parse_anthropic_response)
    except ImportError:
        logger.debug("Anthropic parser not available (anthropic not installed)")


# Register on module import
_register_default_parsers()
