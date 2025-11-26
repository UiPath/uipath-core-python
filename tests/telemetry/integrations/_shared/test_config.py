"""Tests for InstrumentationConfig."""

from __future__ import annotations

import pytest

from uipath.core.telemetry.integrations._shared._config import InstrumentationConfig


class TestInstrumentationConfig:
    """Test cases for InstrumentationConfig."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        config = InstrumentationConfig()

        assert config.capture_inputs is True
        assert config.capture_outputs is True
        assert config.max_string_length == 4096
        assert config.max_array_items == 100
        assert config.ttl_seconds == 3600
        assert config.max_registry_size == 10000

    def test_custom_values(self) -> None:
        """Test creation with custom values."""
        config = InstrumentationConfig(
            capture_inputs=False,
            capture_outputs=False,
            max_string_length=2048,
            max_array_items=50,
            ttl_seconds=1800,
            max_registry_size=5000,
        )

        assert config.capture_inputs is False
        assert config.capture_outputs is False
        assert config.max_string_length == 2048
        assert config.max_array_items == 50
        assert config.ttl_seconds == 1800
        assert config.max_registry_size == 5000

    def test_immutability(self) -> None:
        """Test that config is frozen and immutable."""
        config = InstrumentationConfig()

        # FrozenInstanceError from dataclasses or AttributeError from frozen classes
        with pytest.raises((AttributeError, TypeError)):
            config.max_string_length = 1024  # type: ignore[misc]

    def test_validation_max_string_length(self) -> None:
        """Test validation of max_string_length."""
        with pytest.raises(ValueError, match="max_string_length must be >= 1"):
            InstrumentationConfig(max_string_length=0)

        with pytest.raises(ValueError, match="max_string_length must be >= 1"):
            InstrumentationConfig(max_string_length=-1)

    def test_validation_max_array_items(self) -> None:
        """Test validation of max_array_items."""
        with pytest.raises(ValueError, match="max_array_items must be >= 1"):
            InstrumentationConfig(max_array_items=0)

        with pytest.raises(ValueError, match="max_array_items must be >= 1"):
            InstrumentationConfig(max_array_items=-1)

    def test_validation_ttl_seconds(self) -> None:
        """Test validation of ttl_seconds."""
        with pytest.raises(ValueError, match="ttl_seconds must be >= 1"):
            InstrumentationConfig(ttl_seconds=0)

        with pytest.raises(ValueError, match="ttl_seconds must be >= 1"):
            InstrumentationConfig(ttl_seconds=-1)

    def test_validation_max_registry_size(self) -> None:
        """Test validation of max_registry_size."""
        with pytest.raises(ValueError, match="max_registry_size must be >= 1"):
            InstrumentationConfig(max_registry_size=0)

        with pytest.raises(ValueError, match="max_registry_size must be >= 1"):
            InstrumentationConfig(max_registry_size=-1)

    def test_min_valid_values(self) -> None:
        """Test that minimum valid values (1) are accepted."""
        config = InstrumentationConfig(
            max_string_length=1,
            max_array_items=1,
            ttl_seconds=1,
            max_registry_size=1,
        )

        assert config.max_string_length == 1
        assert config.max_array_items == 1
        assert config.ttl_seconds == 1
        assert config.max_registry_size == 1
