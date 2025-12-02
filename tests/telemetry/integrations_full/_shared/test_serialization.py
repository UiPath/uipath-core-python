"""Tests for serialization utilities."""

from __future__ import annotations

import pytest

from uipath.core.telemetry.integrations_full._shared._serialization import (
    safe_json_dumps,
    truncate_string,
)


class TestSafeJsonDumps:
    """Test cases for safe_json_dumps function."""

    def test_simple_dict(self) -> None:
        """Test serialization of simple dictionary."""
        obj = {"key": "value", "number": 42}
        result = safe_json_dumps(obj)

        assert '"key": "value"' in result
        assert '"number": 42' in result

    def test_non_serializable_object(self) -> None:
        """Test fallback to repr for non-serializable objects."""

        class CustomClass:
            def __repr__(self) -> str:
                return "CustomClass(test=123)"

        obj = CustomClass()
        result = safe_json_dumps(obj)

        assert "CustomClass(test=123)" in result

    def test_with_max_length(self) -> None:
        """Test max_length truncation."""
        obj = {"key": "very long value that should be truncated"}
        result = safe_json_dumps(obj, max_length=20)

        assert len(result) == 20
        assert result.endswith("...")

    def test_unicode_characters(self) -> None:
        """Test handling of Unicode characters."""
        obj = {"greeting": "Hello 世界", "emoji": "🚀"}
        result = safe_json_dumps(obj)

        assert "世界" in result
        assert "🚀" in result

    def test_nested_structures(self) -> None:
        """Test serialization of nested data structures."""
        obj = {
            "level1": {
                "level2": {"level3": ["a", "b", "c"]},
                "items": [1, 2, 3],
            }
        }
        result = safe_json_dumps(obj)

        assert "level1" in result
        assert "level2" in result
        assert "level3" in result

    def test_none_max_length(self) -> None:
        """Test that None max_length doesn't truncate."""
        long_str = "a" * 10000
        obj = {"data": long_str}
        result = safe_json_dumps(obj, max_length=None)

        assert len(result) > 10000
        assert not result.endswith("...")


class TestTruncateString:
    """Test cases for truncate_string function."""

    def test_no_truncation_needed(self) -> None:
        """Test that short strings are not truncated."""
        s = "short"
        result = truncate_string(s, 10)

        assert result == "short"

    def test_truncation_with_ellipsis(self) -> None:
        """Test truncation adds ellipsis."""
        s = "this is a long string that needs truncation"
        result = truncate_string(s, 20)

        assert len(result) == 20
        assert result.endswith("...")
        assert result == "this is a long st..."

    def test_exact_length(self) -> None:
        """Test string exactly at max_length."""
        s = "exactly20characters!"
        result = truncate_string(s, 20)

        assert result == s
        assert len(result) == 20

    def test_min_length_validation(self) -> None:
        """Test that max_length must be >= 3."""
        with pytest.raises(ValueError, match="max_length must be >= 3"):
            truncate_string("test", 2)

    def test_min_length_exactly_three(self) -> None:
        """Test truncation with minimum length of 3."""
        s = "testing"
        result = truncate_string(s, 3)

        assert result == "..."
        assert len(result) == 3

    def test_unicode_truncation(self) -> None:
        """Test truncation with Unicode characters."""
        s = "Hello 世界 this is a long string"
        result = truncate_string(s, 15)

        assert len(result) == 15
        assert result.endswith("...")

    def test_empty_string(self) -> None:
        """Test truncation of empty string."""
        s = ""
        result = truncate_string(s, 10)

        assert result == ""
