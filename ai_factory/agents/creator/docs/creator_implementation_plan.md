### Creator Implementation Plan (Deterministic, LLM-free)

This plan describes how to implement a deterministic “Creator” that generates runnable agents (and Azure Function wrappers) from YAML recipes, without using any LLMs. It integrates with Redis Pub/Sub for recipe ingestion and supports writing outputs to local filesystem or Azure Blob Storage. The generated agents reuse `core/CoreAgent` and `core/tools`.

---

### 1) Objectives

- Deterministic agent creation from a YAML recipe (no LLM).
- Outputs:
  - `agent.py` constructing a `CoreAgent` using recipe’s `system_prompt` and `tools`.
  - `function/__init__.py` and `function/function.json` for Azure Function HTTP trigger.
  - `recipe.yaml` copy for traceability.
- Delivery targets: local filesystem and Azure Blob.
- Pub/Sub: Creator listens on a Redis channel for incoming recipes, writes outputs, and publishes status.
- Integrate seamlessly with existing repo (imports from `core.core_agent` and `core.tools`).

---

### 2) High-Level Architecture

- `creator/schema.py`: Pydantic model and validation for recipes.
- `creator/registry.py`: Deterministic tool registry (maps recipe tool keys to Python classes).
- `creator/templates/`: Jinja2 templates for `agent.py`, `function/__init__.py`, `function/function.json`.
- `creator/generator.py`: Template rendering from a validated recipe.
- `creator/storage.py`: Writers for local filesystem and Azure Blob.
- `creator/service.py`: Redis subscriber that receives recipes (YAML/URL), orchestrates generation, persists outputs, and publishes result.
- `creator/cli.py`: CLI to generate artifacts from a local YAML recipe.

---

### 3) Data Model (Recipe)

- Minimal schema (extendable):

```yaml
name: EchoAgent                  # required, used for folder and class names
description: Basit echo ajanı    # optional
system_prompt: |                 # required
  You are an assistant that echoes everything.
tools:                           # optional; values must exist in TOOL_REGISTRY
  - python_executor
  - file_writer
memory:                          # optional
  enabled: false
azure:                           # optional
  route: /api/echo               # function.json route
  auth_level: function           # anonymous | function | admin
pubsub:                          # optional (for generated agent runtime)
  enable: false
  sub_channels: []
  pub_channel: null
```

- Pydantic model (conceptual):

```python
class Recipe(BaseModel):
  name: str
  description: str | None = None
  system_prompt: str
  tools: list[str] = []
  memory: dict[str, Any] = Field(default_factory=lambda: {"enabled": False})
  azure: dict[str, Any] = Field(default_factory=lambda: {"route": None, "auth_level": "function"})
  pubsub: dict[str, Any] = Field(default_factory=dict)
```

- Name sanitization rules:
  - Class/Folder name derived from `name`: allow [A-Za-z0-9_], first letter uppercase, collapse spaces/hyphens to underscores.

---

### 4) Tool Registry (Deterministic)

- A strict allowlist of tools:

```python
TOOL_REGISTRY = {
  "python_executor": {"module": "core.tools", "class": "PythonExecutorTool"},
  "file_writer":     {"module": "core.tools", "class": "FileWriterTool"},
  "file_reader":     {"module": "core.tools", "class": "FileReaderTool"},
  "directory_list":  {"module": "core.tools", "class": "DirectoryListTool"},
}
```

- Unknown tool keys → validation error (fail fast).

---

### 5) Templates (Jinja2)

- agent.py.j2

```python
from ai_factory.agents.core import CoreAgent
from ai_factory.agents.core import AgentConfig

{ %
for imp in tool_imports %}from {{imp.module}} import {{imp.class_name}}
{ % endfor %}

SYSTEM_PROMPT = """{{ system_prompt }}"""


class {{class_name}}(CoreAgent):
    def __init__(self):
        tools = [
            { % for t in tools %}            {{t}}(),
        { % endfor %}]
        cfg = AgentConfig(
            name="{{ class_name }}",
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            enable_memory={{memory_enabled | lower}}
        { % if pubsub_enabled %},
        enable_pubsub = True,
        pubsub_sub_channels = {{pubsub_sub_channels}},
        pubsub_pub_channel = {{pubsub_pub_channel | tojson}}
        { % endif %}
        )
        super().__init__(cfg)
```

- function/__init__.py.j2

```python
import azure.functions as func
from ai.{{ class_name }}.agent import {{ class_name }}

agent = {{ class_name }}()

def main(req: func.HttpRequest) -> func.HttpResponse:
    q = req.params.get('q')
    if not q:
        body = req.get_json(silent=True) or {}
        q = body.get('q') or body.get('input')
    if not q:
        return func.HttpResponse("Missing q or input", status_code=400)
    result = agent.invoke(q)
    content = result["messages"][-1].content if result and "messages" in result else ""
    return func.HttpResponse(content, status_code=200)
```

- function/function.json.j2

```json
{
  "bindings": [
    {"authLevel": "{{ auth_level }}", "type": "httpTrigger", "direction": "in", "name": "req", "methods": ["get", "post"], "route": "{{ route }}"},
    {"type": "http", "direction": "out", "name": "$return"}
  ]
}
```

---

### 6) Generation Flow

- Input → `Recipe` (validate)
- Compute:
  - class_name from name (sanitize)
  - tool imports (from TOOL_REGISTRY)
  - Jinja2 context:
    - system_prompt, class_name
    - tool_imports + tools (class names)
    - memory_enabled, pubsub flags
    - azure.route (defaults to `/<lower-name>`) and azure.auth_level (defaults to `function`)
- Render templates to strings
- In-memory file map:
  - `agent.py`, `function/__init__.py`, `function/function.json`, `recipe.yaml` (original)

---

### 7) Storage

- Local filesystem:
  - Output base: `ai/<name>/` (configurable via `--out` or `CREATOR_OUT_BASE`)
  - Create directory, write files, ensure idempotency:
    - If exists without `--overwrite`, fail with clear error
- Azure Blob (optional):
  - Container: `agents` (default, configurable)
  - Path: `<name>/agent.py`, `<name>/function/__init__.py`, `<name>/function/function.json`, `<name>/recipe.yaml`
  - Requires `azure-storage-blob` and credentials via `AZURE_STORAGE_CONNECTION_STRING` or account/key pair

---

### 8) CLI

- Command:
  - `python -m creator.cli --recipe recipes/ai.yaml --dest local --out ai`
  - `--dest blob --container agents`
  - `--overwrite` (optional)
- Behavior:
  - Load YAML, validate
  - Generate files in-memory (generator)
  - Write via chosen storage
  - Print exact output paths and a success summary

---

### 9) Pub/Sub Service

- Subscriber:
  - Channel: `creator:recipes:in`
- Message format (two options):

```json
{
  "recipe_yaml": "<full YAML string>",
  "dest": "local|blob",
  "container": "agents",
  "path": "ai",
  "overwrite": false
}
```

```json
{
  "recipe_url": "https://.../ai.yaml",
  "dest": "blob",
  "container": "agents",
  "path": "ai"
}
```

- Flow:
  - Parse message (YAML inline or fetch by URL)
  - Validate → Generate → Persist
  - Publish result to `creator:recipes:out`:

```json
{
  "status": "ok",
  "name": "EchoAgent",
  "dest": "local",
  "paths": ["ai/EchoAgent/agent.py", "ai/EchoAgent/function/__init__.py", ...]
}
```

- Errors → `{"status":"error","message":"...","details":{...}}`

---

### 10) Determinism & Safety

- No LLM calls; template substitution only.
- Strict tool allowlist (TOOL_REGISTRY).
- Fail fast on unknown tool keys or schema violations.
- Sanitize class/folder names.
- Stable code generation (same recipe → identical outputs).
- No execution of user-provided code; YAML fields are declarative only.

---

### 11) Error Handling

- CLI:
  - Clear messages for missing fields, unknown tools, invalid auth levels, path conflicts.
- Service:
  - Log errors and publish failure payload to `creator:recipes:out`.
- Storage:
  - Validate credentials; for Blob, create container if needed (or instruct user).
  - Respect `--overwrite` flag for local writes.

---

### 12) Testing

- Unit tests:
  - Recipe schema validation (success/failure cases).
  - Tool registry resolution (valid/invalid).
  - Template rendering (golden files) — compare to expected outputs.
- Integration tests:
  - Local storage — verify exact file structure & contents.
  - Pub/Sub end-to-end using a test Redis — publish recipe, wait, validate outputs.
- Optional:
  - Mock Azure Blob uploads (e.g., using emulator or mock client).

---

### 13) Deployment & Ops

- The generated agents import from `core.core_agent` and `core.tools`. Ensure these are present in the deployment repo/image.
- Azure Functions:
  - Per-agent: use generated `function/` folder as a Function.
  - Orchestrator: separate Function that routes by agent name; optionally call `agent_registry`.
- FastAPI:
  - Expose `/invoke?agent=...` endpoint backed by `agent_registry`.
- Kubernetes:
  - Prefer FastAPI container for multi-agent hosting with HPA/KEDA.
  - For Functions, use containerized Functions runtime with KEDA if needed.

---

### 14) Observability

- Creator:
  - Logs (receive → validate → generate → write → publish result).
  - Optional: publish metrics to `creator:metrics`.
- Generated agents:
  - Can enable LangSmith via env (`LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`).
  - Pub/Sub (if enabled in recipe) provides additional traceability.

---

### 15) Security

- Do not interpolate arbitrary Python into templates beyond the allowlist mappings.
- Validate Azure `auth_level` is one of `anonymous|function|admin`.
- Sanitized agent names to prevent path traversal or invalid module/class names.
- For Blob, restrict containers/paths to a safe prefix (e.g., `agents/<sanitized_name>/...`).

---

### 16) Roadmap

- GitHub Integration:
  - After generation, open a PR to a target repo (with commit message referencing recipe).
- Extended Tooling:
  - Add more tools via explicit registry entries only.
- Versioning:
  - Add `version` to recipe; include `recipe.yaml` in output for reproducibility.
- Idempotent Updates:
  - `--overwrite` or versioned folders (e.g., `<name>_<timestamp>`).

---

### 17) Milestones & Timeline

- Day 1:
  - Implement `schema.py`, `registry.py`, `templates/`
- Day 2:
  - Implement `generator.py`, `storage.py`, minimal unit tests
- Day 3:
  - Implement `cli.py`, `service.py` (Redis subscriber), integration tests (local)
- Day 4:
  - Extended tests, Blob integration, documentation, quickstart demos

---

### 18) Quickstart

- CLI (local):
  - `python -m creator.cli --recipe recipes/ai.yaml --dest local --out ai`
- Service:
  - `export REDIS_URL=redis://:password@localhost:6379/0`
  - `python -m creator.service`
  - Publish:
    - `redis-cli -u $REDIS_URL PUBLISH creator:recipes:in '{"recipe_yaml":"<full YAML>","dest":"local"}'`
    - or with URL: `{"recipe_url":"https://…/ai.yaml","dest":"blob","container":"agents"}`

This plan yields a fully deterministic, production-ready “Creator” that generates deployable agents (and Azure Functions) from YAML, integrates with Pub/Sub, and fits cleanly into the existing CoreAgent ecosystem.



