"""Tests for telemetry value serialization (ObservationSpan._serialize_value)."""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from uipath.core.telemetry import TelemetryClient
from uipath.core.telemetry.observation import _serialize_value

if TYPE_CHECKING:
    pass


def test_serialize_native_types():
    """Test serialization of OpenTelemetry native types."""
    # Strings
    assert _serialize_value("hello") == "hello"
    assert _serialize_value("") == ""

    # Numbers
    assert _serialize_value(42) == 42
    assert _serialize_value(3.14) == 3.14
    assert _serialize_value(0) == 0
    assert _serialize_value(-100) == -100

    # Booleans
    assert _serialize_value(True) is True
    assert _serialize_value(False) is False


def test_serialize_list_of_native_types():
    """Test serialization of lists containing only native types."""
    # List of strings
    result = _serialize_value(["a", "b", "c"])
    assert result == ["a", "b", "c"]

    # List of ints
    result = _serialize_value([1, 2, 3])
    assert result == [1, 2, 3]

    # List of floats
    result = _serialize_value([1.5, 2.5, 3.5])
    assert result == [1.5, 2.5, 3.5]

    # List of bools
    result = _serialize_value([True, False, True])
    assert result == [True, False, True]

    # Mixed list of native types
    result = _serialize_value([1, "hello", True, 3.14])
    assert result == [1, "hello", True, 3.14]


def test_serialize_dict():
    """Test JSON serialization of dictionaries."""
    data = {"key1": "value1", "key2": 42, "key3": True}
    result = _serialize_value(data)

    # Should be JSON string
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["key1"] == "value1"
    assert parsed["key2"] == 42
    assert parsed["key3"] is True


def test_serialize_nested_dict():
    """Test JSON serialization of nested dictionaries."""
    data = {
        "outer": {
            "inner": {"value": 123},
            "list": [1, 2, 3],
        }
    }
    result = _serialize_value(data)

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed["outer"]["inner"]["value"] == 123
    assert parsed["outer"]["list"] == [1, 2, 3]


def test_serialize_list_with_mixed_types():
    """Test JSON serialization of lists containing complex objects."""
    # List with dict inside - should JSON serialize
    data = [1, {"key": "value"}, "string"]
    result = _serialize_value(data)

    assert isinstance(result, str)
    parsed = json.loads(result)
    assert parsed == [1, {"key": "value"}, "string"]


def test_serialize_dataclass():
    """Test JSON serialization with dataclass support."""

    @dataclass
    class Config:
        timeout: int
        retry: bool

    config = Config(timeout=30, retry=True)
    result = _serialize_value(config)

    # Should serialize to JSON string (using __dict__ or similar)
    assert isinstance(result, str)
    # Should either be JSON or safe type name
    if result.startswith("{"):
        parsed = json.loads(result)
        assert parsed["timeout"] == 30
        assert parsed["retry"] is True
    else:
        # Safe fallback
        assert "Config" in result


def test_serialize_datetime():
    """Test serialization of datetime objects."""
    dt = datetime(2025, 1, 10, 12, 30, 45)
    result = _serialize_value(dt)

    # Should be JSON string (ISO format) or safe type name
    assert isinstance(result, str)


def test_serialize_custom_object():
    """Test safe serialization of custom objects."""

    class CustomObject:
        def __init__(self, secret):
            self.secret = secret

    obj = CustomObject("password123")
    result = _serialize_value(obj)

    # Should return safe type name, NOT the secret value
    assert isinstance(result, str)
    assert "CustomObject" in result
    assert "password123" not in result  # Privacy check


def test_serialize_none():
    """Test serialization of None value."""
    result = _serialize_value(None)

    # Should serialize to JSON "null"
    assert isinstance(result, str)
    assert result == "null"


def test_serialize_circular_reference():
    """Test safe handling of circular references."""
    data = {"key": "value"}
    data["self"] = data  # Circular reference

    result = _serialize_value(data)

    # Should not raise exception, return safe fallback
    assert isinstance(result, str)


def test_serialize_unserializable_object():
    """Test safe handling of completely unserializable objects."""

    class UnserializableObject:
        def __init__(self):
            self.file = open(__file__, "r")  # File handle can't be serialized

    obj = UnserializableObject()
    result = _serialize_value(obj)
    obj.file.close()

    # Should return safe type name
    assert isinstance(result, str)
    assert "UnserializableObject" in result


def test_serialization_integration_with_span(
    telemetry_client: TelemetryClient,
    memory_exporter: InMemorySpanExporter,
):
    """Test that serialization works end-to-end with spans."""

    @dataclass
    class Request:
        method: str
        url: str
        timeout: int

    request = Request(method="GET", url="https://api.example.com", timeout=30)

    with telemetry_client.start_as_current_span("test_span") as span:
        span.set_attribute("request", request)
        span.set_attribute("tags", ["api", "http"])
        span.set_attribute("count", 42)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span_data = spans[0]

    # Request should be JSON-serialized (or safe fallback)
    request_attr = span_data.attributes["request"]
    assert isinstance(request_attr, str)

    # Tags should be list (or tuple)
    tags_attr = span_data.attributes["tags"]
    assert list(tags_attr) == ["api", "http"]

    # Count should be int
    assert span_data.attributes["count"] == 42


def test_serialize_empty_collections():
    """Test serialization of empty lists and dicts."""
    # Empty list
    assert _serialize_value([]) == []

    # Empty dict
    result = _serialize_value({})
    assert isinstance(result, str)
    assert result == "{}"


def test_serialize_large_object():
    """Test serialization of reasonably large objects."""
    large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
    result = _serialize_value(large_dict)

    # Should serialize without error
    assert isinstance(result, str)
    parsed = json.loads(result)
    assert len(parsed) == 100
    assert parsed["key_0"] == "value_0"
    assert parsed["key_99"] == "value_99"
