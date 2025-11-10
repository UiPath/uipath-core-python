"""Telemetry configuration with auto-detection and environment variable support.

This module provides immutable configuration for the telemetry client with:
- Automatic library version detection via importlib.metadata
- Environment variable overrides (12-factor app pattern)
- Validation for configuration parameters
- Frozen dataclass for hashability (singleton pattern support)
"""

import importlib.metadata
import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TelemetryConfig:
    """Immutable telemetry configuration.

    Configuration precedence (highest to lowest):
    1. Explicit constructor parameters
    2. Environment variables (UIPATH_*)
    3. Default values

    Example:
        >>> # Explicit configuration
        >>> config = TelemetryConfig(
        ...     org_id="org-123",
        ...     tenant_id="tenant-456",
        ...     endpoint="https://telemetry.example.com"
        ... )
        >>>
        >>> # Environment variable configuration
        >>> # export UIPATH_ORG_ID=org-123
        >>> # export UIPATH_TENANT_ID=tenant-456
        >>> config = TelemetryConfig()  # Auto-loads from env
    """

    # UiPath-specific metadata (Resource Attributes - low cardinality, static)
    org_id: Optional[str] = None  # Env: UIPATH_ORG_ID
    tenant_id: Optional[str] = None  # Env: UIPATH_TENANT_ID
    user_id: Optional[str] = None  # Env: UIPATH_USER_ID

    # Library metadata (auto-detected)
    library_name: str = "uipath-core"
    library_version: Optional[str] = None  # Auto-detect if None

    # OTLP exporter configuration
    endpoint: Optional[str] = None  # Env: UIPATH_TELEMETRY_ENDPOINT
    headers: Optional[Tuple[Tuple[str, str], ...]] = (
        None  # Hashable headers (tuple of tuples)
    )

    # Batching configuration
    batch_export: bool = True
    max_queue_size: int = 2048
    export_timeout_millis: int = 30000

    # Sampling (0.0 = disabled, 1.0 = 100%)
    sample_rate: float = 1.0  # Env: UIPATH_TELEMETRY_SAMPLE_RATE

    # Service metadata
    service_name: str = "uipath-core"  # Env: UIPATH_SERVICE_NAME
    service_namespace: Optional[str] = None  # Env: UIPATH_SERVICE_NAMESPACE
    service_version: Optional[str] = None  # Env: UIPATH_SERVICE_VERSION

    def __post_init__(self):
        """Auto-detect library version and apply environment variable overrides.

        This method runs after dataclass initialization to:
        1. Auto-detect library version via importlib.metadata
        2. Apply environment variable overrides for unset fields
        3. Validate configuration parameters
        """
        if self.library_version is None:
            try:
                version = importlib.metadata.version(self.library_name)
                object.__setattr__(self, "library_version", version)
            except importlib.metadata.PackageNotFoundError:
                object.__setattr__(self, "library_version", "unknown")

        none_default_fields = {
            "org_id": ("UIPATH_ORG_ID", None),
            "tenant_id": ("UIPATH_TENANT_ID", None),
            "user_id": ("UIPATH_USER_ID", None),
            "endpoint": ("UIPATH_TELEMETRY_ENDPOINT", None),
            "service_namespace": ("UIPATH_SERVICE_NAMESPACE", None),
            "service_version": ("UIPATH_SERVICE_VERSION", None),
        }

        non_none_default_fields = {
            "service_name": ("UIPATH_SERVICE_NAME", "uipath-core"),
        }

        for field_name, (env_var, _) in none_default_fields.items():
            env_value = os.getenv(env_var)
            if env_value is not None and getattr(self, field_name) is None:
                object.__setattr__(self, field_name, env_value)

        for field_name, (env_var, default_value) in non_none_default_fields.items():
            env_value = os.getenv(env_var)
            if env_value is not None and getattr(self, field_name) == default_value:
                object.__setattr__(self, field_name, env_value)

        sample_rate_env = os.getenv("UIPATH_TELEMETRY_SAMPLE_RATE")
        if sample_rate_env is not None:
            try:
                sample_rate = float(sample_rate_env)
                if 0.0 <= sample_rate <= 1.0:
                    object.__setattr__(self, "sample_rate", sample_rate)
            except ValueError:
                pass

        if not (0.0 <= self.sample_rate <= 1.0):
            raise ValueError(
                f"sample_rate must be between 0.0 and 1.0, got {self.sample_rate}"
            )
