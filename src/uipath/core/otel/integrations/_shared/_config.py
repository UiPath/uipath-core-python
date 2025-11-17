"""Configuration for LangChain and LangGraph instrumentation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentationConfig:
    """Configuration for instrumentation behavior.

    Args:
        capture_inputs: Whether to capture input data (default: True)
        capture_outputs: Whether to capture output data (default: True)
        max_string_length: Maximum length for string attributes (default: 4096)
        max_array_items: Maximum number of items in arrays (default: 100)
        ttl_seconds: TTL for span registry in seconds (default: 3600)
        max_registry_size: Maximum span registry size (default: 10000)

    Examples:
        >>> config = InstrumentationConfig()
        >>> config.capture_inputs
        True
        >>> config = InstrumentationConfig(max_string_length=2048)
        >>> config.max_string_length
        2048
    """

    capture_inputs: bool = True
    capture_outputs: bool = True
    max_string_length: int = 4096
    max_array_items: int = 100
    ttl_seconds: int = 3600
    max_registry_size: int = 10000

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_string_length < 1:
            raise ValueError("max_string_length must be >= 1")
        if self.max_array_items < 1:
            raise ValueError("max_array_items must be >= 1")
        if self.ttl_seconds < 1:
            raise ValueError("ttl_seconds must be >= 1")
        if self.max_registry_size < 1:
            raise ValueError("max_registry_size must be >= 1")
