### Creator (Deterministic) - Agent Generator from YAML Recipe

Purpose
- Deterministically generate a runnable agent (and Azure Function wrapper) from a YAML recipe, without any LLM.
- Integrate with Redis Pub/Sub to receive recipes, and write outputs to Azure Blob Storage or local filesystem.
- Reuse `core/CoreAgent` and existing tools. The generated agent only imports from the repo.

Key Requirements
- No LLM usage; fully deterministic.
- Input is a YAML recipe with at least: `name`, `system_prompt`, `tools`, optional `azure` config.
- Output is a folder named by `name`, containing:
  - `agent.py` (imports `core.core_agent.CoreAgent`, assembles `AgentConfig` from recipe)
  - `function/__init__.py` and `function.json` (Azure Function HTTP trigger that invokes agent)
  - copy of the original `recipe.yaml`
- Delivery targets:
  - Local filesystem (solutions repo root): `<name>/...`
  - Azure Blob Storage container: `agents/<name>/...`
- Pub/Sub support: listens for recipes via Redis channel; generates output upon receipt.

YAML Recipe Schema
```yaml
name: EchoAgent                  # required, used as class and folder name
description: Basit echo ajanı    # optional
system_prompt: |                 # required
  You are an assistant that echoes everything.
tools:                           # optional, subset of supported tools
  - python_executor
  - file_writer
memory:                          # optional
  enabled: false
azure:                           # optional
  route: /api/echo               # HTTP route for function.json
  auth_level: function           # anonymous | function | admin
pubsub:                          # optional (for the generated agent)
  enable: false
  sub_channels: []
  pub_channel: null
```

Supported Tool Keys (deterministic registry)
- `python_executor` → `core.tools.PythonExecutorTool`
- `file_writer` → `core.tools.FileWriterTool`
- `file_reader` → `core.tools.FileReaderTool`
- `directory_list` → `core.tools.DirectoryListTool`

Message Format over Redis (Pub/Sub)
- Subscribe channel (creator): `creator:recipes:in`
- Optional response channel: `creator:recipes:out` (status, errors)
- Payload options:
```json
{"recipe_yaml": "<full YAML string>", "dest": "blob|local", "container": "agents", "path": "."}
```
or
```json
{"recipe_url": "https://.../ai.yaml", "dest": "blob|local"}
```

Output Artifacts (per name X)
- Local (solutions repo): `X/agent.py`, `X/function/__init__.py`, `X/function/function.json`, `X/recipe.yaml`
- Blob:  `agents/X/agent.py`, `agents/X/function/__init__.py`, `agents/X/function/function.json`, `agents/X/recipe.yaml`

Environment Variables
- Redis: `REDIS_URL=redis://:password@host:6379/0`
- Azure Blob:
  - `AZURE_STORAGE_CONNECTION_STRING` or `AZURE_STORAGE_ACCOUNT`, `AZURE_STORAGE_KEY`
  - `BLOB_CONTAINER=agents` (default)
- Creator options:
  - `CREATOR_DEST=local|blob` (default local)
  - `CREATOR_OUT_BASE=ai` (local output base)

Deterministic Code Generation (Templates)
- Template engine: Jinja2
- Templates (`creator/templates/`):
  - `agent.py.j2`:
    ```python
    from ai_factory.agents.core.core_agent import CoreAgent
    from ai_factory.agents.core.config import AgentConfig
    {% for imp in tool_imports %}from {{ imp.module }} import {{ imp.class_name }}
    {% endfor %}

    SYSTEM_PROMPT = """{{ system_prompt }}"""

    class {{ class_name }}(CoreAgent):
        def __init__(self):
            tools = [
            {% for t in tools %}    {{ t }}(),
            {% endfor %}]
            cfg = AgentConfig(
                name="{{ class_name }}",
                system_prompt=SYSTEM_PROMPT,
                tools=tools,
                enable_memory={{ memory_enabled | lower }}{% if pubsub_enabled %},
                enable_pubsub=True,
                pubsub_sub_channels={{ pubsub_sub_channels }},
                pubsub_pub_channel={{ pubsub_pub_channel | tojson }}{% endif %}
            )
            super().__init__(cfg)
    ```
  - `function/__init__.py.j2`:
    ```python
    import azure.functions as func
    from {{ class_name }}.agent import {{ class_name }}
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
  - `function/function.json.j2`:
    ```json
    {
      "bindings": [
        {"authLevel": "{{ auth_level }}", "type": "httpTrigger", "direction": "in", "name": "req", "methods": ["get", "post"], "route": "{{ route }}"},
        {"type": "http", "direction": "out", "name": "$return"}
      ]
    }
    ```

Module Structure (Implementation)
- `creator/`
  - `service.py`: Redis subscriber (listens `creator:recipes:in`), dispatch to generator
  - `schema.py`: Pydantic `Recipe` model and validation
  - `generator.py`: Renders templates from recipe, returns in-memory files
  - `storage.py`: Writes files to local FS or Azure Blob
  - `templates/`: Jinja2 templates (above)
  - `cli.py`: CLI entry `python -m creator --recipe recipes/ai.yaml --dest local`

Schema (Pydantic)
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

Tool Registry (Python)
```python
TOOL_REGISTRY = {
  "python_executor": {"module": "core.tools", "class": "PythonExecutorTool"},
  "file_writer":     {"module": "core.tools", "class": "FileWriterTool"},
  "file_reader":     {"module": "core.tools", "class": "FileReaderTool"},
  "directory_list":  {"module": "core.tools", "class": "DirectoryListTool"},
}
```

Redis Flow (Creator Service)
- Sub to `creator:recipes:in`.
- On message:
  1) Extract `recipe_yaml` or fetch `recipe_url` → parse/validate
  2) Generate files via `generator.py`
  3) Persist via `storage.py` to selected destination
  4) Publish status to `creator:recipes:out`:
     `{ "status": "ok", "name": "...", "paths": [...], "dest": "local|blob" }`

Azure Blob Writing
- Use `azure-storage-blob`
- Container default `agents`; path `<name>/...`
- Create if not exists; upload `agent.py`, `function/__init__.py`, `function/function.json`, `recipe.yaml`

CLI Usage Examples
- Local:
  - `python -m creator.cli --recipe recipes/ai.yaml --dest local --out .`
- Blob:
  - `python -m creator.cli --recipe recipes/ai.yaml --dest blob --container agents`

Determinism Rules
- No dynamic code from LLMs; only template substitution.
- Class name = sanitized `name` (alnum + underscores), first letter uppercase.
- Tools resolved strictly from `TOOL_REGISTRY`. Unknown keys → error.
- Routes default to `/<lower-name>` if not provided.
- Conflicts: existing folder/object path → `--overwrite` flag required; otherwise fail.

Testing Plan
- Unit tests:
  - YAML validation (missing fields, unknown tools)
  - Template rendering outputs expected code (golden files)
  - Local writer creates correct structure
- Integration tests:
  - Pub/Sub end-to-end (publish recipe, observe files written)
  - Optionally mock Azure Blob to verify uploads

Observability
- Creator logs each step and publishes a compact result on `creator:recipes:out`.
- LangSmith not required for creator (no LLM), but downstream agents can enable it from env.

Security & Hygiene
- Sanitize `name` for folder/class (allow [A-Za-z0-9_], strip others)
- Do not execute arbitrary code from recipes; only declarative fields allowed.
- Validate auth level for Azure Functions: one of `anonymous|function|admin`.

Roadmap
- Optional: GitHub integration to open a PR with generated files
- Optional: Additional tool registries (db, http tools) behind explicit allowlist

