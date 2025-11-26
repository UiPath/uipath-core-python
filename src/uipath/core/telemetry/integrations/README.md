# UiPath OTel Framework Integrations

This package provides automatic instrumentation for AI/LLM frameworks using the OpenTelemetry `BaseInstrumentor` pattern.

## Available Integrations

### LangGraph

Automatic instrumentation for LangGraph workflows using callback-based tracing.

```python
from uipath.core.otel.integrations.langgraph import LangGraphInstrumentor

# Install once
LangGraphInstrumentor().instrument()

# Normal LangGraph code - automatically traced!
app = graph.compile()
result = app.invoke({"messages": ["Hello"]})
```

## Creating New Integrations

To add a new framework instrumentor:

1. Create a new subdirectory: `integrations/myframework/`
2. Implement `MyFrameworkInstrumentor(UiPathInstrumentor)`:
   - `instrumentation_dependencies()`: Declare supported versions
   - `_instrument()`: Apply monkey-patching
   - `_uninstrument()`: Restore originals
3. Add tests in `tests/otel/integrations/myframework/`
4. Update this README

### Example Structure

```
integrations/
├── __init__.py
├── _base.py                 # UiPathInstrumentor base class
├── README.md                # This file
└── myframework/
    ├── __init__.py
    ├── instrumentor.py      # MyFrameworkInstrumentor
    ├── _callbacks.py        # Framework-specific callbacks (if needed)
    └── README.md            # Usage documentation
```

### Example Implementat

ion

```python
from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from ..._base import UiPathInstrumentor

class MyFrameworkInstrumentor(UiPathInstrumentor):
    \"\"\"Automatic instrumentation for MyFramework.\"\"\"

    def instrumentation_dependencies(self):
        return ("myframework >= 1.0.0",)

    def _instrument(self, **kwargs):
        otel_client = self._get_otel_client()
        # Apply monkey-patching here
        pass

    def _uninstrument(self, **kwargs):
        # Restore original methods
        pass
```

## Best Practices

1. **Respect privacy**: Apply state filtering and size limits
2. **Version gating**: Check framework versions before patching
3. **Error handling**: Never let instrumentation break user code
4. **Testing**: Comprehensive tests with multiple framework versions
5. **Context propagation**: Use OpenTelemetry's standard context API for parent-child relationships
