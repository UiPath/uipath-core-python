"""Helper utilities for the tracing module."""

import inspect
import json
import random
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Mapping
from zoneinfo import ZoneInfo

from pydantic import BaseModel


def get_supported_params(
    tracer_impl: Callable[..., Any],
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    """Extract the parameters supported by the tracer implementation."""
    try:
        sig = inspect.signature(tracer_impl)
    except (TypeError, ValueError):
        # If we can't inspect, pass all parameters and let the function handle it
        return dict(params)

    supported: Dict[str, Any] = {}
    for name, value in params.items():
        if value is not None and name in sig.parameters:
            supported[name] = value
    return supported


def _simple_serialize_defaults(obj):
    # Handle Pydantic BaseModel instances
    if hasattr(obj, "model_dump") and not isinstance(obj, type):
        return obj.model_dump(exclude_none=True, mode="json")

    # Handle classes - convert to schema representation
    if isinstance(obj, type) and issubclass(obj, BaseModel):
        return {
            "__class__": obj.__name__,
            "__module__": obj.__module__,
            "schema": obj.model_json_schema(),
        }
    if hasattr(obj, "dict") and not isinstance(obj, type):
        return obj.dict()
    if hasattr(obj, "to_dict") and not isinstance(obj, type):
        return obj.to_dict()

    # Handle dataclasses
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)

    # Handle enums
    if isinstance(obj, Enum):
        return _simple_serialize_defaults(obj.value)

    if isinstance(obj, (set, tuple)):
        if hasattr(obj, "_asdict") and callable(obj._asdict):
            return obj._asdict()
        return list(obj)

    if isinstance(obj, datetime):
        return obj.isoformat()

    if isinstance(obj, (timezone, ZoneInfo)):
        return obj.tzname(None)

    # Allow JSON-serializable primitives to pass through unchanged
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    return str(obj)


def span_id_to_uuid4(span_id: int) -> uuid.UUID:
    """Convert a 64-bit span ID to a valid UUID4 format.

    Creates a UUID where:
    - The 64 least significant bits contain the span ID
    - The UUID version (bits 48-51) is set to 4
    - The UUID variant (bits 64-65) is set to binary 10
    """
    # Generate deterministic high bits using the span_id as seed
    temp_random = random.Random(span_id)
    high_bits = temp_random.getrandbits(64)

    # Combine high bits and span ID into a 128-bit integer
    combined = (high_bits << 64) | span_id

    # Set version to 4 (UUID4)
    combined = (combined & ~(0xF << 76)) | (0x4 << 76)

    # Set variant to binary 10
    combined = (combined & ~(0x3 << 62)) | (2 << 62)

    # Convert to hex string in UUID format
    hex_str = format(combined, "032x")
    return uuid.UUID(hex_str)


def trace_id_to_uuid4(trace_id: int) -> uuid.UUID:
    """Convert a 128-bit trace ID to a valid UUID4 format.

    Modifies the trace ID to conform to UUID4 requirements:
    - The UUID version (bits 48-51) is set to 4
    - The UUID variant (bits 64-65) is set to binary 10
    """
    # Set version to 4 (UUID4)
    uuid_int = (trace_id & ~(0xF << 76)) | (0x4 << 76)

    # Set variant to binary 10
    uuid_int = (uuid_int & ~(0x3 << 62)) | (2 << 62)

    # Convert to hex string in UUID format
    hex_str = format(uuid_int, "032x")
    return uuid.UUID(hex_str)


def format_args_for_trace_json(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> str:
    """Return a JSON string of inputs from the function signature."""
    result = format_args_for_trace(signature, *args, **kwargs)
    return json.dumps(result, default=_simple_serialize_defaults)


def format_object_for_trace_json(
    input_object: Any,
) -> str:
    """Return a JSON string of inputs from the function signature."""
    return json.dumps(input_object, default=_simple_serialize_defaults)


def format_args_for_trace(
    signature: inspect.Signature, *args: Any, **kwargs: Any
) -> Dict[str, Any]:
    try:
        """Return a dictionary of inputs from the function signature."""
        # Create a parameter mapping by partially binding the arguments

        parameter_binding = signature.bind_partial(*args, **kwargs)

        # Fill in default values for any unspecified parameters
        parameter_binding.apply_defaults()

        # Extract the input parameters, skipping special Python parameters
        result = {}
        for name, value in parameter_binding.arguments.items():
            # Skip class and instance references
            if name in ("self", "cls"):
                continue

            # Handle **kwargs parameters specially
            param_info = signature.parameters.get(name)
            if param_info and param_info.kind == inspect.Parameter.VAR_KEYWORD:
                # Flatten nested kwargs directly into the result
                if isinstance(value, dict):
                    result.update(value)
            else:
                # Regular parameter
                result[name] = value

        return result
    except Exception:
        return {"args": args, "kwargs": kwargs}
