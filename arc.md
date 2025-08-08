# Agent Platform – Architecture Guide (Nested in `agents_functions/`)

## 1. Goal

This guide defines a **self‑contained agent platform** that lives inside a wider application repo under `agents_functions/`. It delivers two execution models—(1) a single LangGraph orchestrator Function and (2) a multi‑Function Azure Service Bus pipeline—while sharing the same agent code base. The document also explains how this sub‑module integrates with the **parent project’s existing Docker Compose and CI/CD**.

---

## 2. Top‑Level Folder Layout

> Everything below resides **inside** the folder `agents_functions/` at the root of the parent repository.

```text
parent‑repo
│
├─ ai/
│   │
│   ├─ core/                    # Common infrastructure code
│   │   └─ core_agent.py        # Abstract BaseAgent + helpers
│   │
│   ├─ agents/                  # All concrete agents
│   │   ├─ scoper/scoper_agent.py
│   │   ├─ creator/creator_agent.py
│   │   ├─ publisher/publisher_agent.py
│   │   └─ curator/curator_agent.py
│   │   └─ */docs/README.md     # Per‑agent docs
│   │
│   ├─ docs/                    # Global docs & ADRs for the agent platform
│   │   ├─ architecture.md
│   │   └─ contributing.md
│   │
│   ├─ infra/                   # Bicep + azd
│   │   ├─ main.bicep  (RG, KV, Insights)
│   │   ├─ storage.bicep
│   │   ├─ servicebus.bicep
│   │   └─ functionapps.bicep   # 5 Function Apps
│   │
│   ├─ solutions/
│   │   ├─ langgraph_orchestrator/   # Solution A
│   │   │   ├─ orchestrator/agent_graph.py
│   │   │   ├─ orchestrator/handlers.py
│   │   │   └─ orchestrator/function_app.py
│   │   │   ├─ host.json
│   │   │   └─ requirements.txt
│   │   └─ sb_functions/             # Solution B
│   │       ├─ scoper_function/...
│   │       ├─ creator_function/...
│   │       ├─ publisher_function/...
│   │       └─ curator_function/...
│   │       ├─ host.json
│   │       └─ requirements.txt
│   │
│   ├─ demos/                  # End‑to‑end runnable examples per agent
│   │   ├─ scoper/
│   │   ├─ creator/
│   │   ├─ publisher/
│   │   └─ curator/
│   │
│   ├─ scripts/
│   │   ├─ run_local.sh        # Start both solutions locally
│   │   ├─ deploy_local.sh     # `azd up` + slot swap (dev)
│   │   └─ seed_test_data.py   # Push sample messages
│   │
│   ├─ tests/
│   │   ├─ unit/
│   │   └─ contract/
│   │
│   └─ .github/workflows/      # Scoped CI/CD for the sub‑module
│       ├─ ci.yml
│       └─ cd.yml
│
└─ docker‑compose.yml          # Parent‑level compose, see §13
```

---

## 3. Shared Components

| Directory | Content                                       | Purpose                        |
| --------- | --------------------------------------------- | ------------------------------ |
| `core/`   | `core_agent.py`, `AgentContext`, OTEL helpers | **DRY, single point of truth** |
| `agents/` | Concrete agents that extend `CoreAgent`       | Imported by both solutions     |

---

## 4. Solution A – LangGraph Orchestrator

* **One** Python Azure Function App (HTTP trigger).
* `agent_graph.py` builds the LangGraph DAG and wires agents.
* Intended for fast POCs or local development with Docker Compose (see §13).

### Runtime Flow

1. HTTP POST `/api/orchestrate` (user message).
2. LangGraph workflow → `Scoper → Creator → Publisher → Curator`.
3. Final response returned as HTTP 200.

---

## 5. Solution B – Service Bus Pipeline

* Four Azure Function Apps (Scoper, Creator, Publisher, Curator) running independently.
* Message flow via **queues** except the last hop which uses a **topic + subscriptions** for audit fan‑out.

| Queue/Topic                | Producer  | Consumer                                   | Reason         |
| -------------------------- | --------- | ------------------------------------------ | -------------- |
| `scoper-out`               | Scoper    | Creator                                    | point‑to‑point |
| `creator-out`              | Creator   | Publisher                                  | ″              |
| `publisher-events` (topic) | Publisher | Curator (sub `main`) & Audit (sub `audit`) | fan‑out        |

---

## 6. Scripts

| Script              | Purpose          | Key Steps                                          |
| ------------------- | ---------------- | -------------------------------------------------- |
| `run_local.sh`      | Local dev runner | `func start` SB functions + `uvicorn` orchestrator |
| `deploy_local.sh`   | Dev‑env deploy   | `azd env new dev` → `azd up`                       |
| `seed_test_data.py` | Demo data        | Sends test messages via Service Bus SDK            |

---

## 7. CI / CD (Scoped)

### CI (`agents_functions/.github/workflows/ci.yml`)

```yaml
on:
  push:
    paths:
      - 'agents_functions/**'
```

* Matrix test for Python 3.11.
* Lint → pytest → upload zip artefacts.

### CD (`agents_functions/.github/workflows/cd.yml`)

```yaml
on:
  push:
    branches: [main]
    paths: ['agents_functions/**']
```

* `azure/login` + `azure/azd-action` → `azd up` (prod).
* Both solutions deploy in parallel; slot swap ensures zero downtime.

> The parent project’s root workflow remains unchanged. It simply calls these child workflows if it needs holistic checks.

---

## 8. Local Development

* **Dev Container** includes Python 3.11, Azure CLI, Functions Core Tools.
* `run_local.sh` spins up Azurite (if available) for local SB/Blob emulation.
* Secrets stored in `.env` at `agents_functions/`.

---

## 9. Infrastructure (Bicep)

* All resources are prefixed with `agents‑fx‑<env>` to avoid collision with root app.
* `functionapps.bicep` provisions 5 Function Apps on Flex Consumption.
* Key Vault + Managed Identities for secretless connectivity.

---

## 10. Security & Config

* Managed Identity → Service Bus / Storage with RBAC.
* Key Vault for LangChain or OpenAI keys.
* `host.json` fine‑tunes `maxConcurrentCalls` + `prefetchCount`.

---

## 11. Observability

* OpenTelemetry instrumentation → Azure Monitor.
* Alert on DLQs per subscription (`publisher-events/$DeadLetterQueue`).

---

## 12. Growth Path

1. Add agent folder under `agents/`, extend DAG or add new queue.
2. CI path filter auto‑detects and redeploys.
3. Promote to Premium plan if throughput demands.

---

## 13. Docker Compose Integration (Parent App)

The parent project already ships a `docker-compose.yml`. Extend it with an **orchestrator** service so local developers can run the LangGraph solution without Azure:

```yaml
aff-orchestrator:
  image: mcr.microsoft.com/azure-functions/python:4-python3.11
  volumes:
    - ./agents_functions/solutions/langgraph_orchestrator:/home/site/wwwroot
  environment:
    - AzureWebJobsScriptRoot=/home/site/wwwroot
    - FUNCTIONS_WORKER_RUNTIME=python
    - LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
    - SERVICE_BUS_CONNECTION=Endpoint=sb://broker:5672/...
  ports:
    - "7072:80"
  depends_on:
    - broker   # e.g., a local RabbitMQ or none if using Azurite SB emulator
```

* **Why only the orchestrator?** Multi‑Function pipeline requires Azure Service Bus triggers which are not available locally without heavy setup. For e2e tests you can still run `func start` via `run_local.sh`.
* Compose override files (`docker-compose.override.yml`) can tailor dev‑container envs.

The core application’s existing pipelines stay intact; they merely mount `agents_functions/` for code scanning or delegate to its dedicated workflows.

---

🎯 **Result:** A neatly encapsulated agent subsystem that plugs into an existing app’s Docker Compose and CI/CD while staying fully autonomous for build & deploy.
