Architecture and Deployment Options
===================================

Goals
-----
- Deterministic agent creation from YAML recipe(s)
- Reuse `core/CoreAgent` and shared tools
- Three deployment options:
  1) Per-agent Azure Function (HTTP trigger) – one Function per agent
  2) LangGraph Orchestrator – single Function that routes to multiple agents
  3) FastAPI service – single web app serving multiple agents
- Support Kubernetes deployment (esp. FastAPI), and optionally Azure Functions containers

Monorepo Folder Structure
-------------------------

```
agents/
  core/                      # Core agent engine, managers, demos
  agent/                     # (Optional) prebuilt agents (coder/executor/..)
  recipes/                   # Agent recipes (YAML)
    ai.yaml                  # Example recipe to drive creator

  deploy/
    shared/
      agent_registry.py      # Central registry/factories for agents used by deployments

    azure_functions/
      per_agent/
        template/
          __init__.py        # Template Function handler (HTTP)
          function.json
          README.md
      orchestrator/
        __init__.py          # Single Function that routes to agents via agent_registry
        function.json
        README.md

    fastapi/
      app/
        main.py              # FastAPI app exposing /invoke?agent=... and /health
      README.md

  k8s/
    fastapi-deployment.yaml  # K8s Deployment for FastAPI
    fastapi-service.yaml     # K8s Service for FastAPI

  docs/
    arc.md                   # This file
```

Deployment Options
------------------
1) Azure Functions – Per Agent
   - Each agent has its own Function (HTTP trigger). CI can stamp out a new folder from the template and wire to `agent_registry`.
   - Pros: Clear isolation, independent scaling.
   - Cons: Many Functions to manage.

2) Azure Functions – Orchestrator
   - A single Function that reads `agent` from query/body and dispatches to the matching agent in `agent_registry`.
   - Pros: One entrypoint, simple to manage.
   - Cons: Single scale unit; routing logic lives in app.

3) FastAPI Service
   - A single web app exposing `/invoke?agent=...`, `/agents`, `/health`.
   - Pros: Native HTTP app, easy to run in Kubernetes with HPA/KEDA; simpler local dev.
   - Cons: You manage server lifecycle (unlike Functions platform).

Kubernetes Notes
----------------
- FastAPI option ships with K8s manifests (Deployment + Service). Add Ingress, HPA, and secrets as needed.
- For Azure Functions in K8s, you can containerize Functions and run with KEDA/Functions runtime; treat similarly to any container workload.

Agent Registry
--------------
- `deploy/shared/agent_registry.py` centralizes agent factories. Both Functions and FastAPI import it for consistent behavior.
- You can map agent names to prompts, tools, and memory options here.

Creator Flow (future)
---------------------
- A YAML recipe (`recipes/ai.yaml`) is parsed by a `creator` script/agent to stamp out:
  - A new agent entry in `agent_registry`
  - Optional Azure Function folder (per-agent) or orchestrator route
  - Optional FastAPI route/metadata



