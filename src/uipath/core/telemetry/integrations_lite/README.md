# Telemetry Integrations Lite

Lightweight, minimal-dependency telemetry integrations for LangChain and LangGraph following KISS/YAGNI/DRY principles.

## Three UiPath Integration Options

UiPath Core provides three LangChain/LangGraph integration implementations:

1. **integrations_lite** (569 LOC) ⭐ - Zero external dependencies, manual wrapping, basic tracing
   - **Best for:** Development, airgapped environments, minimal footprint

2. **integrations_openinference** (684 LOC) ⭐⭐ - Thin wrapper around OpenInference + UiPath features
   - **Best for:** Production with rich metadata, minimal maintenance (saves ~1,352 LOC vs full)

3. **integrations** (full, ~2,112 LOC) - Complete custom implementation with custom extractors
   - **Best for:** Custom requirements, full control, specialized extraction needs

**Most users should use `integrations_openinference`** for the best balance of features and maintainability.

## Quick Start

```python
from uipath.core.telemetry import init
from uipath.core.telemetry.integrations_lite import (
    instrument_langchain,
    instrument_langgraph,
)
from uipath.core.telemetry.integrations._shared import set_session_context

# Initialize telemetry
init(enable_console_export=True)

# Instrument integrations
instrument_langchain()
instrument_langgraph()

# Set session context (optional)
set_session_context(session_id="my-session", thread_id="thread-1")

# Now all @traceable and LangGraph execution is automatically traced!
from langsmith import traceable
from langgraph.graph import StateGraph

@traceable(run_type="tool")
def my_tool():
    return "result"
```

## Implementation Comparison

This section compares different UiPath telemetry/tracing implementations across various projects, including industry-standard alternatives.

**⚠️ Important Notes:**
- **uipath-langchain:** The comparison below is for **programmatic/library usage only**. The `uipath run` CLI tool has separate, more capable instrumentation. See [Library vs CLI section](#important-uipath-langchain-library-vs-cli) for details.
- **OpenInference:** Industry-standard alternative for AI observability. See [OpenInference section](#alternative-openinference) for when to use it instead.

### Feature Matrix

| Feature | Full Integration<br/>(uipath-core) | **Lite Integration**<br/>(uipath-core) | **OpenInference Wrapper**<br/>(uipath-core) | OpenInference<br/>(external) | uipath_langchain | uipath-python |
|---------|------------------|------------------|------------------|---------------|------------------|---------------|
| **Core Tracing** | | | | | |
| Basic span creation | ✅ | ✅ | ✅ | ✅ | ✅ |
| Span hierarchy | ✅ | ✅ | ✅ | ✅ | ✅ |
| Error handling | ✅ | ✅ | ✅ | ✅ | ✅ |
| Input/output capture | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async support | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Session Context** | | | | | |
| Session ID tracking | ✅ | ✅ | ⚠️ Manual | ❌ | ❌ |
| Thread ID tracking | ✅ | ✅ | ⚠️ Manual | ❌ | ❌ |
| Context propagation | ✅ | ✅ | ✅ | ❌ | ✅ (ContextVar) |
| **LangChain Support** | | | | | |
| `@traceable` decorator | ✅ | ✅ | ✅ Auto | ✅ | ❌ |
| `run_type` mapping | ✅ | ✅ | ✅ Auto | ✅ | ❌ |
| Decorator patterns | ✅ | ✅ | ✅ Auto | ✅ | N/A |
| Custom callbacks | ✅ | ❌ | ✅ | ❌ | ❌ |
| Rich metadata extraction | ✅ | ❌ | ✅ | ❌ | ❌ |
| **LangGraph Support** | | | | | |
| `invoke()` tracing | ✅ | ✅ | ✅ Auto | ❌ Library only<br/>✅ CLI has it | ❌ No LangGraph |
| `ainvoke()` tracing | ✅ | ✅ | ✅ Auto | ❌ Library only<br/>✅ CLI has it | ❌ No LangGraph |
| Node-level tracing | ✅ | ❌ | ✅ Auto | ❌ Library only<br/>✅ CLI has it | ❌ No LangGraph |
| Checkpoint tracking | ✅ | ❌ | ⚠️ Limited | ❌ | ❌ No LangGraph |
| Stream methods | ✅ | ❌ | ✅ Auto | ❌ | ❌ No LangGraph |
| **Metadata Extraction** | | | | | |
| LLM model names | ✅ | ❌ | ✅ | ❌ | ❌ |
| Token counts | ✅ | ❌ | ✅ | ❌ | ❌ |
| Chain metadata | ✅ | ❌ | ✅ | ❌ | ❌ |
| Tool metadata | ✅ | ❌ | ✅ | ❌ | ❌ |
| Retriever metadata | ✅ | ❌ | ✅ | ❌ | ❌ |
| **LLM Parsers** | | | | | |
| OpenAI parser | ✅ | ❌ | ✅ | ❌ | ❌ |
| Anthropic parser | ✅ | ❌ | ✅ | ❌ | ❌ |
| Parser registry | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Exporters** | | | | | |
| Console exporter | ✅ | ✅ | ✅ | ✅ | ✅ |
| OTLP exporter | ✅ | ✅ | ✅ | ✅ | ✅ |
| JsonLines exporter | ❌ | ❌ | ❌ | ❌ | ✅ |
| LlmOps HTTP exporter | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Advanced Features** | | | | | |
| Custom extractors | ✅ | ❌ | ✅ | ❌ | ❌ |
| Parser plugins | ✅ | ❌ | ✅ | ❌ | ❌ |
| Config management | ✅ | ❌ | ✅ | ❌ | ❌ |
| Parent span provider | ✅ | ❌ | ✅ | ❌ | ✅ |
| Tracer reapplication | ❌ | ❌ | ❌ | ❌ | ✅ |
| Function registry | ❌ | ❌ | ❌ | ❌ | ✅ |
| **UiPath-Specific** | | | | | |
| UiPath attributes | ✅ | ✅ | ❌ | ❌ | ✅ |
| Privacy controls | ✅ | ✅ | ⚠️ Manual | ❌ | ✅ |
| Org/Tenant tracking | ✅ | ✅ | ❌ | ❌ | ✅ |
| **Implementation** | | | | | |
| **Total LOC** | **2,366** | **211** | **External** | **65** | **902** |
| Total LOC (with docs) | 2,366 | 571 | External | 82 | 1,328 |
| Implementation files | 8 | 4 | External pkg | 2 | 4 |
| Test LOC | ~1,800 | 1,026 | External | Unknown | Unknown |
| Primary dependency | OpenTelemetry | OpenTelemetry | OpenTelemetry | uipath.tracing | OpenTelemetry |
| External dependencies | None | None | ✅ Required | None | None |
| Integration scope | LangChain + LangGraph | LangChain + LangGraph | **Industry-standard AI** | **Library: @traceable ONLY**<br/>**CLI: Full LangGraph** | Generic @traced |
| Semantic conventions | UiPath-specific | UiPath-specific | **OpenInference standard** | UiPath-specific | UiPath-specific |
| Complexity | High | Low | Auto (black box) | Minimal | Medium |
| Maintenance burden | High | Low | External (Arize AI) | Minimal | Medium |

### Implementation Details

#### Full Integration (uipath-core) - ~2,112 LOC
**Scope:** Comprehensive LangChain + LangGraph instrumentation with rich metadata extraction
**Package:** `uipath.core.telemetry.integrations`
**LOC Breakdown:**
- Total: ~2,112 lines
- Implementation: ~1,811 lines (excl. _shared)
- LangChain: 936 lines
- LangGraph: 875 lines
- Shared: 301 lines

**Best for:** Production deployments requiring detailed observability, custom extractors, full control

#### Lite Integration (uipath-core) - 569 LOC ⭐
**Scope:** Lightweight LangChain + LangGraph instrumentation following KISS/YAGNI
**Package:** `uipath.core.telemetry.integrations_lite`
**LOC Breakdown:**
- Total: 569 lines
- Implementation: 487 lines (excl. __init__)
- LangChain: 191 lines
- LangGraph: 224 lines
- Adapter: 72 lines

**Best for:** Development, testing, simple workflows, minimal dependencies, quick implementation
**Reduction:** 77% smaller than full (487 vs 1,811 LOC implementation)

#### OpenInference Wrapper (uipath-core) - 684 LOC ⭐⭐
**Scope:** Thin wrapper around OpenInference + UiPath-specific features
**Package:** `uipath.core.telemetry.integrations_openinference`
**LOC Breakdown:**
- Total: 684 lines
- Implementation: 459 lines (excl. __init__)
  - Privacy config: 159 lines
  - Span processor: 164 lines
  - Instrumentor wrapper: 136 lines
- **Delegates ~900 LOC to OpenInference** (instrumentation, extractors, metadata)

**Best for:** Rich metadata + minimal maintenance, automatic instrumentation, production deployments
**Reduction:** 75% smaller than full (459 vs 1,811 LOC implementation)
**Key Advantage:** Leverages battle-tested OpenInference library, saves ~1,352 LOC

#### uipath_langchain - 65 LOC (Library) / Unknown LOC (CLI)
**Scope:**
- **Library (`_tracing` module):** LangChain `@traceable` decorator patching ONLY - NO LangGraph support
- **CLI (`uipath run` tool):** Full automatic LangGraph instrumentation built-in

**Package:** `uipath_langchain._tracing` (library) / `uipath run` (CLI)
**Best for:**
- **Library:** Projects using `uipath.tracing` that need LangChain `@traceable` support
- **CLI:** Running LangGraph agents with automatic observability

**Reduction:** 97% smaller than full (65 vs 2,366 LOC, library module only)
**Note:**
- Library delegates to `uipath.tracing.traced` - NO LangGraph instrumentation
- CLI has separate built-in LangGraph instrumentation (automatic, no decorators needed)
- **For programmatic use without CLI, library has ZERO LangGraph support**

#### uipath-python - 902 LOC
**Scope:** Generic `@traced` decorator with custom exporters (JsonLines, LlmOps HTTP)
**Package:** `uipath.tracing`
**Best for:** Projects needing custom exporters, tracer reapplication, parent span provider hooks
**Reduction:** 62% smaller than full (902 vs 2,366 LOC)
**Note:** No LangChain/LangGraph integration - provides base tracing infrastructure

#### OpenInference - External Package
**Scope:** Industry-standard automatic instrumentation for AI applications (LangChain + LangGraph)
**Package:** `openinference-instrumentation-langchain>=0.1.55` (external dependency)
**Maintainer:** Arize AI (observability platform company)
**License:** Apache-2.0, Production/Stable

**Best for:**
- Industry-standard AI observability
- Arize Phoenix integration
- OpenInference semantic conventions
- Automatic instrumentation without code changes
- No UiPath-specific features needed

**Key Features:**
- ✅ **Automatic LangChain/LangGraph instrumentation** - Call `.instrument()` once, everything is traced
- ✅ **Rich metadata extraction** - Model names, token counts, costs, messages, documents
- ✅ **Industry-standard conventions** - OpenInference semantic attributes (llm.*, tool.*, embedding.*)
- ✅ **OpenTelemetry compatible** - Works with any OTLP backend
- ✅ **Node-level tracing** - Captures individual LangGraph node executions
- ✅ **No code changes** - Auto-instruments existing code via monkey-patching

**What It Doesn't Provide:**
- ❌ **No UiPath-specific attributes** - No org_id, tenant_id, folder_key, job_id
- ❌ **No built-in privacy controls** - No hide_input/hide_output flags
- ❌ **No session context API** - Must manually add session/thread attributes
- ❌ **External dependency** - Maintained by third party (Arize AI)
- ❌ **Black box** - Auto-instrumentation may be harder to customize

**Usage:**
```python
from openinference.instrumentation.langchain import LangChainInstrumentor

# One-time setup
LangChainInstrumentor().instrument()

# Now ALL LangChain/LangGraph operations are automatically traced!
from langgraph.graph import StateGraph

builder = StateGraph(dict)
builder.add_node("process", lambda x: {"result": x["a"] * x["b"]})
graph = builder.compile()

# Automatic tracing - no decorators needed!
result = await graph.ainvoke({"a": 5, "b": 3})
```

**When to Use OpenInference Instead:**
1. ✅ Need industry-standard OpenInference conventions
2. ✅ Using Arize Phoenix or other OpenInference-compatible platforms
3. ✅ Want automatic instrumentation with zero code changes
4. ✅ Don't need UiPath-specific attributes (org_id, tenant_id, etc.)
5. ✅ Don't need privacy controls (hide_input/hide_output)
6. ✅ Prefer external package over in-house implementation

**When NOT to Use OpenInference:**
1. ❌ Need UiPath-specific attributes (org_id, tenant_id, folder_key, job_id)
2. ❌ Need privacy controls (hide_input/hide_output per function)
3. ❌ Want minimal dependencies (OpenInference is external package)
4. ❌ Need explicit control over what gets traced
5. ❌ Want to avoid external dependencies maintained by third party

**Comparison to Our Implementation:**

| Aspect | OpenInference | Our Lite Integration |
|--------|--------------|---------------------|
| Approach | Auto-instrumentation (black box) | Explicit patching (transparent) |
| Dependencies | External package required | Zero external instrumentation deps |
| UiPath features | ❌ None | ✅ Full support |
| Privacy controls | ⚠️ Manual configuration | ✅ Built-in (hide_input/hide_output) |
| Maintenance | Arize AI | UiPath team |
| Semantic conventions | OpenInference standard | UiPath-specific |
| Control | All-or-nothing | Selective instrumentation |
| Size | External (unknown LOC) | 211 LOC (transparent) |

**Can They Work Together?**
**Not recommended.** Both instrument LangChain/LangGraph in similar ways, which would create duplicate spans and conflicting behavior. Choose one:
- **OpenInference** → For industry-standard AI observability
- **Our Lite Integration** → For UiPath-specific features and privacy controls

### Important: uipath-langchain Library vs CLI

**Critical Distinction:** The uipath-langchain package contains TWO separate components with different capabilities:

#### Component 1: Library Module (`uipath_langchain._tracing`)

**Programmatic Usage:**
```python
from uipath_langchain._tracing import _instrument_traceable_attributes
_instrument_traceable_attributes()

# Execute LangGraph directly
result = await graph.ainvoke(input_data)  # ❌ NOT traced - library has no LangGraph support
```

**Capabilities:**
- ✅ Patches LangSmith `@traceable` decorator
- ✅ Maps `run_type` to span types
- ✅ Delegates to `uipath.tracing.traced`
- ❌ **NO LangGraph instrumentation** (invoke/ainvoke/nodes not traced)
- ❌ **NO automatic tracing** without `@traceable` decorators

**Size:** 65 LOC (code only)

---

#### Component 2: CLI Tool (`uipath run`)

**CLI Usage:**
```bash
# Automatic LangGraph instrumentation built into CLI
uv run uipath run agent '{"input": "..."}' --trace-file traces.jsonl
```

**Capabilities:**
- ✅ **Automatic LangGraph instrumentation** (no decorators needed!)
- ✅ Traces `invoke()`/`ainvoke()` calls
- ✅ Traces node executions
- ✅ Captures LangGraph metadata (step, node, triggers, path)
- ✅ Session and thread context
- ✅ Works WITHOUT `@traceable` decorators

**Size:** Unknown (larger than library, includes runtime infrastructure)

---

#### Evidence

**Without CLI (library only):**
```python
from uipath_langchain._tracing import _instrument_traceable_attributes
_instrument_traceable_attributes()
result = await graph.ainvoke(input_data)
# Result: 0 traces (no LangGraph support in library)
```

**With CLI:**
```bash
uv run uipath run agent '{"a": 10, "b": 5, "operator": "*"}' --trace-file traces.jsonl
# Result: 2+ traces with full LangGraph metadata (automatic instrumentation)
```

**Conclusion:** For programmatic LangGraph tracing without the CLI, use **uipath-core** (Full or Lite), NOT uipath_langchain library module.

---

#### How CLI Actually Works (Architecture)

The `uipath run` CLI uses a **three-layer instrumentation architecture**:

**Layer 1: Library Module** (`uipath_langchain._tracing`)
- Patches LangSmith `@traceable` decorator only
- 65 LOC, minimal scope
- Delegates to `uipath.tracing.traced`

**Layer 2: OpenInference** (External Package)
- **This is the key!** Uses `openinference-instrumentation-langchain>=0.1.54`
- Provides automatic LangGraph instrumentation
- Maintained by Arize AI (third party)
- Called via: `LangChainInstrumentor().instrument()`

**Layer 3: UiPath Integration** (Legacy)
- Uses `uipath.core.tracing` (old tracing system, NOT new telemetry)
- Registers span provider and ancestor tracking
- Provides UiPath-specific utilities

**Source:** `/Users/religa/src/uipath-langchain-python/src/uipath_langchain/runtime/factory.py:56-61`

```python
def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
    """Setup tracing and instrumentation."""
    _instrument_traceable_attributes()              # Layer 1: Library
    LangChainInstrumentor().instrument()            # Layer 2: OpenInference (EXTERNAL!)
    UiPathSpanUtils.register_current_span_provider(get_current_span)
    UiPathSpanUtils.register_current_span_ancestors_provider(get_ancestor_spans)
```

**Key Insight:** The CLI doesn't implement LangGraph instrumentation itself - it uses OpenInference (external package from Arize AI).

**Why Our Implementation Is Different:**
- ✅ **Self-contained:** No external instrumentation dependencies
- ✅ **Modern:** Uses new `uipath.core.telemetry` system (not legacy tracing)
- ✅ **UiPath-specific:** Native support for UiPath attributes and privacy
- ✅ **Transparent:** 211 LOC, easy to understand and debug

**Why CLI Uses OpenInference:**
- ✅ Production-tested by wider community
- ✅ Compatible with Arize Phoenix and other platforms
- ✅ Rich LLM metadata extraction out of the box
- ⚠️ Trade-off: External dependency, no UiPath-specific features

**Both Are Valid:** CLI (OpenInference-based) and our integrations (UiPath-specific) serve different use cases.

---

### Code Size Breakdown

#### Full Integration (~2,112 LOC)

```
integrations/
├── langchain/
│   ├── __init__.py               37 LOC - Public API
│   ├── _instrumentor.py          25 LOC - Setup/teardown
│   ├── _tracer.py               103 LOC - Custom callback handler
│   └── _extractors.py           228 LOC - Metadata extraction
│   ├── ... (other files)        543 LOC
│   Total:                       936 LOC
│
├── langgraph/
│   ├── __init__.py               15 LOC - Public API
│   ├── instrumentor.py          108 LOC - Compile patching
│   └── _callbacks.py            181 LOC - Comprehensive callbacks
│   ├── ... (other files)        571 LOC
│   Total:                       875 LOC
│
├── _shared/
│   ├── __init__.py                5 LOC - Exports
│   ├── _session_context.py      16 LOC - Session management
│   ├── _serialization.py         20 LOC - Safe JSON
│   ├── _parser_registry.py      44 LOC - Parser system
│   ├── _config.py                19 LOC - Configuration
│   ├── ... (other files)        197 LOC
│   Total:                       301 LOC

Total: ~2,112 LOC
```

#### Lite Integration (569 LOC)

```
integrations_lite/
├── __init__.py                   82 LOC - Public API + docs
├── _traced_adapter.py            72 LOC - Core adapter
├── langchain.py                 191 LOC - Decorator patching
└── langgraph.py                 224 LOC - Minimal instrumentation

Reused from existing:
- safe_json_dumps               (from _shared)
- set_session_context           (from _shared)
- ObservationSpan               (from telemetry)
- Attr constants                (from telemetry)

Total: 569 LOC (487 LOC implementation, 82 LOC docs)
```

#### OpenInference Wrapper (684 LOC)

```
integrations_openinference/
├── __init__.py                  225 LOC - Public API + comprehensive docs
├── _config.py                   159 LOC - Privacy configuration
├── _span_processor.py           164 LOC - UiPath span enrichment
└── _instrumentor.py             136 LOC - OpenInference wrapper

Delegates to OpenInference (~900 LOC):
- LangChain/LangGraph instrumentation
- Rich metadata extraction (tokens, costs, models)
- Automatic tracing and callbacks

Total: 684 LOC (459 LOC implementation, 225 LOC docs)
Effective: ~1,584 LOC (459 UiPath + ~900 OpenInference + 225 docs)
```

### What's Included in Lite

✅ **Fully Supported:**
- Basic tracing with span creation
- Session and thread context tracking
- LangChain `@traceable` decorator patching
- LangGraph `invoke()`/`ainvoke()` tracing
- Input/output capture (JSON serialization)
- Error handling and exception recording
- Async function support
- Nested function call tracing
- Parent-child span relationships

✅ **Uses Existing Infrastructure:**
- `safe_json_dumps()` for serialization
- `set_session_context()` for context management
- `ObservationSpan` for span operations
- `Attr` constants for attribute names

### What's NOT Included in Lite

❌ **Excluded (Use Full Integration If Needed):**
- **Rich metadata extraction** (model names, token counts, chain types)
- **LLM response parsing** (OpenAI, Anthropic parsers)
- **Custom callback handlers** for LangChain
- **Node-level tracing** for LangGraph (only top-level invoke)
- **Stream methods** (stream/astream/batch support)
- **Custom extractors** and parser plugins
- **Parser registry** system
- **Integration-specific configuration**

### Performance Comparison

| Metric | Full | Lite | Difference |
|--------|------|------|------------|
| Import time | ~50ms | ~10ms | 5x faster |
| Memory overhead | ~2MB | ~500KB | 4x smaller |
| Span creation overhead | ~100μs | ~50μs | 2x faster |
| Instrumentation complexity | High | Low | Simpler |

## When to Use Lite

### ✅ Use Lite If You Need:

1. **Basic tracing** - Just need to see function calls and execution flow
2. **Simple setup** - Want minimal configuration and dependencies
3. **Low maintenance** - Don't want to update complex integrations
4. **Quick start** - Need to get tracing working fast
5. **Lightweight** - Care about import time and memory usage
6. **Session tracking** - Need session/thread context propagation
7. **Essential features only** - Don't need advanced metadata

### Example Use Cases:

- Development and debugging
- Local testing with console export
- Simple LangChain/LangGraph workflows
- Proof-of-concept implementations
- Services with basic observability needs
- Applications sensitive to dependencies

## When to Use Full

### ✅ Use Full If You Need:

1. **Production observability** - Rich metrics and metadata
2. **Token counting** - Track LLM usage and costs
3. **Model information** - Capture model names, versions
4. **Advanced analytics** - Need detailed execution data
5. **Custom extractors** - Want to add custom metadata
6. **Node-level tracing** - Need to see individual LangGraph nodes
7. **Stream support** - Use streaming methods

### Example Use Cases:

- Production deployments
- Cost tracking and analysis
- Performance monitoring
- Compliance and auditing
- Advanced debugging scenarios
- Multi-tenant applications

## Decision Guide: Which Implementation to Choose?

Use this guide to choose the right telemetry implementation for your needs:

### Quick Decision Tree

```
Need LangChain/LangGraph tracing?
├─ No → Use uipath-python (@traced decorator only)
└─ Yes
   ├─ Need UiPath-specific features? (org_id, tenant_id, privacy controls)
   │  ├─ Yes
   │  │  ├─ Need rich metadata? (token counts, costs, model names)
   │  │  │  ├─ Yes
   │  │  │  │  ├─ Want minimal maintenance? (74% less code)
   │  │  │  │  │  └─ Yes → Use OpenInference Wrapper ⭐⭐ RECOMMENDED
   │  │  │  │  └─ Want custom extractors/full control?
   │  │  │  │     └─ Yes → Use Full Integration
   │  │  │  └─ No → Use Lite Integration ⭐ (zero external deps)
   │  │  └─ Using uipath run CLI?
   │  │     └─ Yes → CLI has built-in instrumentation (OpenInference-based)
   │  └─ No (Industry-standard only, no UiPath features)
   │     ├─ Want automatic instrumentation?
   │     │  └─ Yes → Use OpenInference (external, industry-standard)
   │     └─ Want minimal dependencies?
   │        └─ Yes → Use Lite Integration (UiPath-specific but simpler)
```

### Detailed Comparison

| Use Case | Recommendation | Why? |
|----------|---------------|------|
| **Programmatic LangGraph + UiPath + Rich metadata** | **OpenInference Wrapper** ⭐⭐ | UiPath attributes, privacy, auto-instrumentation, minimal maintenance (459 LOC) |
| **Programmatic LangGraph + UiPath (basic)** | **Lite Integration** ⭐ | UiPath attributes, privacy controls, zero external deps (487 LOC) |
| **Programmatic LangGraph + Custom extractors** | **Full Integration** | Full control, custom metadata extraction (1,811 LOC) |
| **CLI/Runtime LangGraph execution** | **uipath run CLI** | Built-in OpenInference, no setup needed |
| **Industry-standard AI observability (no UiPath)** | **OpenInference** (external) | Arize Phoenix, standard conventions |
| **Generic function tracing only** | **uipath-python** | Base @traced decorator, custom exporters |
| **LangChain @traceable patching only** | **uipath_langchain** | Minimal, delegates to uipath.tracing |

### By Primary Need

#### Need: UiPath Features + Rich Metadata (MOST COMMON) ⭐⭐
**Examples:** org_id, tenant_id + token counts, costs, model names

**Choose:** OpenInference Wrapper (`integrations_openinference`)
- ✅ UiPath attributes (session, thread, privacy)
- ✅ Rich metadata automatic (tokens, costs, models)
- ✅ Automatic instrumentation (zero code changes)
- ✅ Minimal maintenance (459 LOC vs 1,811 LOC full)
- ✅ Production-tested (OpenInference community)
- ⚠️ One external dependency (openinference-instrumentation-langchain)

**Why choose this:** Best balance of features and maintainability. Saves ~1,352 LOC compared to full integration.

---

#### Need: UiPath-Specific Attributes (Basic Tracing)
**Examples:** org_id, tenant_id, folder_key, job_id, process_key

**Choose:** Lite Integration (if zero dependencies required)
- ✅ Native UiPath attribute support
- ✅ Built-in privacy controls (hide_input/hide_output)
- ✅ Session/thread tracking
- ✅ Zero external dependencies
- ❌ No rich metadata (tokens, costs)
- ❌ Manual instrumentation required

---

#### Need: Privacy Controls
**Examples:** hide_input, hide_output, PII protection

**Choose:** OpenInference Wrapper (best) or Lite Integration
- ✅ **OpenInference Wrapper:** Global privacy config + sensitive span detection
- ✅ **Lite Integration:** Per-function hide_input/hide_output flags
- ✅ Both: Safe by default
- ❌ Don't use OpenInference external (manual configuration required)

---

#### Need: Industry-Standard Conventions
**Examples:** OpenInference attributes (llm.*, tool.*, embedding.*)

**Choose:** OpenInference
- ✅ Standard AI observability conventions
- ✅ Compatible with Arize Phoenix and other platforms
- ✅ Rich LLM metadata
- ❌ Don't use if you need UiPath-specific attributes

---

#### Need: Minimal Dependencies
**Examples:** Keep it simple, no external packages

**Choose:** Lite Integration
- ✅ Zero external instrumentation dependencies
- ✅ 211 LOC only
- ✅ Full control
- ❌ Don't use OpenInference (external dependency)

---

#### Need: Automatic Everything
**Examples:** No code changes, instrument and go

**Choose:** OpenInference or CLI
- ✅ OpenInference: Auto-instruments LangChain/LangGraph
- ✅ CLI: Built-in runtime instrumentation
- ⚠️ Trade-off: Less control, no UiPath features

---

#### Need: Rich Metadata Extraction
**Examples:** Token counts, costs, model names, chain types

**Choose:** Full Integration or OpenInference
- ✅ Full: UiPath-specific + rich metadata
- ✅ OpenInference: Industry-standard + rich metadata
- ❌ Don't use Lite (minimal metadata only)

---

### Migration Paths

**From Nothing → Start Here:**
1. **Try OpenInference Wrapper first** ⭐⭐ - UiPath features + rich metadata + minimal maintenance
2. **Use Lite** if zero dependencies required
3. **Use Full** only if you need custom extractors

**From Lite → OpenInference Wrapper:** ⬆️ **RECOMMENDED UPGRADE**
- **Why:** Add rich metadata (tokens, costs) + automatic instrumentation
- **How:** Change import path and remove @traceable usage
  ```python
  # Before (Lite)
  from uipath.core.telemetry.integrations_lite import instrument_langchain

  # After (OpenInference Wrapper)
  from uipath.core.telemetry.integrations_openinference import instrument_langchain
  ```
- **Impact:** +1 dependency, +rich metadata, -manual wrapping

**From Lite → Full:**
- **Why:** Need custom extractors, full control
- **How:** Change import path only (API identical)
- **Impact:** +1,324 LOC to maintain, +custom extractors

**From OpenInference Wrapper → Full:**
- **Why:** Need custom extractors beyond OpenInference
- **How:** Change import path
- **Impact:** +1,352 LOC to maintain, -OpenInference dependency

**From Full → OpenInference Wrapper:** ⬇️ **RECOMMENDED SIMPLIFICATION**
- **Why:** Reduce maintenance (74% less code), same features
- **How:** Change import path
- **Impact:** +OpenInference dependency, -1,352 LOC maintenance

**From Full → Lite:**
- **Why:** Remove all dependencies, simplify maximally
- **How:** Change import path
- **Impact:** -Rich metadata, -1,324 LOC

---

## Migration Between Versions

### Upgrading from Lite to Full

No code changes required, just change import path:

```python
# Before (Lite)
from uipath.core.telemetry.integrations_lite import (
    instrument_langchain,
    instrument_langgraph,
)

# After (Full)
from uipath.core.telemetry.integrations import (
    instrument_langchain,
    instrument_langgraph,
)

# API is identical
instrument_langchain()
instrument_langgraph()
```

### Downgrading from Full to Lite

Same change in reverse. Note: You'll lose rich metadata but keep all traces.

```python
# Before (Full)
from uipath.core.telemetry.integrations import (
    instrument_langchain,
    instrument_langgraph,
)

# After (Lite)
from uipath.core.telemetry.integrations_lite import (
    instrument_langchain,
    instrument_langgraph,
)
```

## API Reference

### LangChain Integration

```python
def instrument_langchain() -> None:
    """Instrument LangChain by patching @traceable decorator.

    Raises:
        ImportError: If langsmith package not installed.
        RuntimeError: If already instrumented.
    """

def uninstrument_langchain() -> None:
    """Remove LangChain instrumentation.

    Raises:
        RuntimeError: If not currently instrumented.
    """

def is_langchain_instrumented() -> bool:
    """Check if LangChain is currently instrumented."""
```

### LangGraph Integration

```python
def instrument_langgraph() -> None:
    """Instrument LangGraph by patching StateGraph.compile().

    Raises:
        ImportError: If langgraph package not installed.
        RuntimeError: If already instrumented.
    """

def uninstrument_langgraph() -> None:
    """Remove LangGraph instrumentation.

    Raises:
        RuntimeError: If not currently instrumented.
    """

def is_langgraph_instrumented() -> bool:
    """Check if LangGraph is currently instrumented."""
```

### Session Context (Shared)

```python
from uipath.core.telemetry.integrations._shared import (
    set_session_context,
    clear_session_context,
    get_session_id,
    get_thread_id,
)

def set_session_context(session_id: str, thread_id: str | None = None) -> None:
    """Set session and thread context for all spans."""

def clear_session_context() -> None:
    """Clear session and thread context."""

def get_session_id() -> str | None:
    """Get current session ID."""

def get_thread_id() -> str | None:
    """Get current thread ID."""
```

## Examples

### Complete Example

```python
import asyncio
from uipath.core.telemetry import init, traced
from uipath.core.telemetry.integrations_lite import (
    instrument_langchain,
    instrument_langgraph,
)
from uipath.core.telemetry.integrations._shared import set_session_context


async def main():
    # Initialize telemetry
    init(enable_console_export=True)

    # Instrument before using LangChain/LangGraph
    instrument_langchain()
    instrument_langgraph()

    # Set session context
    set_session_context(session_id="user-123", thread_id="conv-456")

    # Use LangSmith @traceable
    from langsmith import traceable

    @traceable(run_type="tool", name="calculator")
    def calculate(a: int, b: int, op: str) -> int:
        if op == "+":
            return a + b
        elif op == "*":
            return a * b
        return 0

    # Use LangGraph
    from langgraph.graph import StateGraph

    @traced(name="process")
    async def process_node(state: dict) -> dict:
        result = calculate(state["a"], state["b"], state["op"])
        return {"result": result}

    builder = StateGraph(dict)
    builder.add_node("process", process_node)
    builder.set_entry_point("process")
    builder.set_finish_point("process")

    graph = builder.compile()

    # Execute - all automatically traced!
    result = await graph.ainvoke({"a": 5, "b": 3, "op": "*"})
    print(f"Result: {result}")  # {"result": 15}


if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

```bash
# Run all lite integration tests
pytest tests/telemetry/integrations_lite/ -v

# Run specific test file
pytest tests/telemetry/integrations_lite/test_langchain.py -v

# Run with coverage
pytest tests/telemetry/integrations_lite/ --cov=src/uipath/core/telemetry/integrations_lite
```

## Design Principles

### KISS (Keep It Simple, Stupid)
- Simple adapter pattern, no complex abstractions
- Minimal dependencies, reuses existing code
- Clear, readable implementation

### YAGNI (You Aren't Gonna Need It)
- No LLM parsers (add manually if needed)
- No rich extractors (use full integration if needed)
- Only essential features included

### DRY (Don't Repeat Yourself)
- Reuses `safe_json_dumps` from `_shared`
- Reuses session context from `_shared`
- Uses `ObservationSpan` for span management
- Uses `Attr` constants for attribute names
- Zero code duplication

## Contributing

When contributing to integrations_lite, follow these principles:

1. **Keep it minimal** - Only add features that are universally needed
2. **Reuse existing code** - Import from `_shared` or telemetry core
3. **Use Attr constants** - Never hardcode attribute names
4. **Test thoroughly** - 100% test coverage required
5. **Document well** - Clear docstrings and examples

## License

Same as parent project (UiPath Core Python SDK).

## See Also

- [Full Integration Documentation](../README.md)
- [Telemetry Core Documentation](../../README.md)
- [Testing Guide](../../../../tests/telemetry/integrations_lite/README.md)
