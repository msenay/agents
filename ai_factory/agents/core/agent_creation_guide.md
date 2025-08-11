## Core Agent Creation Guide

### 🚀 Quick Start

Creating a Core Agent is very simple! There are 3 basic methods:

### 1) Simplest Agent

```python
from ai_factory.agents.core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

agent = CoreAgent(AgentConfig(
    name="MyAgent",
    model=model,
    system_prompt="You are a helpful assistant."
))

response = agent.invoke("Hello!")
```

### 2) Agent with Tools

```python
from ai_factory.agents.core import CoreAgent, AgentConfig
from langchain_core.tools import tool


# Define custom tool
@tool
def calculator(expression: str) -> str:
    """Performs mathematical operations"""
    try:
        return str(eval(expression))
    except:
        return "Error: Invalid operation"


# Config
config = AgentConfig(
    name="MathAgent",
    model=model,
    system_prompt="You are a mathematics assistant.",
    tools=[calculator]  # Add tools
)

agent = CoreAgent(config)
```

### 3) Agent with Memory

```python
config = AgentConfig(
    name="MemoryAgent",
    model=model,
    system_prompt="You are an assistant with memory.",
    
    # Memory settings
    enable_memory=True,
    memory_backend="inmemory",  # or "redis", "postgres"
    memory_types=["short_term", "long_term"]
)

agent = CoreAgent(config)

# Thread memory usage
response = agent.invoke(
    "My name is Alice",
    config={"configurable": {"thread_id": "user_123"}}
)
```

### 📋 Configuration Overview (Use only what you need)

Below is a simple map of the most important options. Each flag is independent unless noted.

- Basic
  - name: agent identifier (string)
  - model: LLM instance (ChatOpenAI/AzureChatOpenAI, can be None)
  - system_prompt: system role text
  - tools: list of LangChain tools
  - description: human-friendly summary

- Memory (enable_memory=True)
  - memory_types: ["short_term", "long_term", "session", "semantic"]
  - memory_backend: "inmemory" | "redis" | "postgres"
  - redis_url / postgres_url: connection strings for selected backend
  - session_id: required when using "session"
  - embedding_model, embedding_dims: for semantic memory
  - enable_ttl, default_ttl_minutes, refresh_on_read: Redis TTL controls
  - enable_message_trimming, max_tokens, trim_strategy, start_on, end_on: context window control
  - enable_summarization, summarization_trigger_tokens, max_summary_tokens: auto summarization (uses model)
  - enable_memory_tools, memory_namespace_store: exposes memory tools to the agent

- Integrations
  - enable_mcp, mcp_servers: Model Context Protocol servers
  - enable_langsmith, langsmith_*: LangSmith tracing defaults
  - enable_langfuse, langfuse_*: Langfuse tracing

- Pub/Sub (Redis)
  - enable_pubsub: True to auto-run Redis subscriber in background
  - pubsub_sub_channels: list of input channels
  - pubsub_pub_channel: single output channel
  - pubsub_daemon_thread: run subscriber as daemon

- Multi-agent patterns (choose at most one)
  - enable_supervisor | enable_swarm | enable_handoff
  - agents: dict of name -> CoreAgent
  - default_active_agent, handoff_agents

- Performance & Safety
  - enable_rate_limiting, requests_per_second, check_every_n_seconds, max_bucket_size, custom_rate_limiter
  - enable_human_feedback, interrupt_before, interrupt_after
  - enable_evaluation, evaluation_metrics

- Extensibility
  - response_format: pydantic model for structured outputs
  - enable_streaming: True to stream chunks
  - pre_model_hook / post_model_hook
  - enable_subgraphs, subgraph_configs

### Basic Parameters
```python
config = AgentConfig(
    # Required
    name="AgentName",           # Agent name
    model=model,                # LLM model
    
    # Optional
    system_prompt="...",        # System prompt
    tools=[],                   # Tool list
    description="..."           # Agent description
)
```

### Memory Parameters
```python
config = AgentConfig(
    # Enable memory
    enable_memory=True,
    
    # Choose backend (one of)
    memory_backend="inmemory",  # "redis", "postgres"
    
    # Memory types (choose what you need)
    memory_types=["short_term", "long_term", "session", "semantic"],
    
    # Backend URLs (only if needed)
    redis_url="redis://localhost:6379",
    postgres_url="postgresql://user:pass@localhost:5432/db"
)
```

### Advanced Features
```python
config = AgentConfig(
    # Rate limiting
    enable_rate_limiting=True,
    requests_per_second=2.0,
    
    # Human feedback
    enable_human_feedback=True,
    interrupt_before=["tool_call"],
    
    # Streaming
    enable_streaming=True,
    
    # Message trimming
    enable_message_trimming=True,
    max_tokens=4000
)
```

### 🎯 Practical Examples

#### 1) Chatbot Agent
```python
def create_chatbot():
    return CoreAgent(AgentConfig(
        name="Chatbot",
        model=ChatOpenAI(model="gpt-4o-mini"),
        system_prompt="You are a friendly and helpful chatbot.",
        enable_memory=True,
        memory_backend="inmemory"
    ))

chatbot = create_chatbot()
response = chatbot.invoke("How are you?")
```

#### 2) Coder Agent

```python
from ai_factory.agents.core import create_python_coding_tools


def create_coder():
    return CoreAgent(AgentConfig(
        name="PythonCoder",
        model=ChatOpenAI(model="gpt-4"),
        system_prompt="You are an expert Python developer.",
        tools=create_python_coding_tools(),
        enable_memory=True,
        memory_types=["short_term", "long_term"]
    ))


coder = create_coder()
response = coder.invoke("Write a Fibonacci function")
```

#### 3) Research Agent
```python
from langchain_community.tools import TavilySearchResults

def create_researcher():
    search_tool = TavilySearchResults()
    
    return CoreAgent(AgentConfig(
        name="Researcher",
        model=ChatOpenAI(model="gpt-4"),
        system_prompt="You are an assistant that conducts detailed research.",
        tools=[search_tool],
        enable_memory=True,
        memory_types=["short_term", "semantic"],  # Semantic search
        memory_backend="postgres",  # with pgvector
        embedding_model="openai:text-embedding-3-small"
    ))
```

#### 4) Pub/Sub Agent (Redis)
```python
from ai_factory.agents.core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

agent = CoreAgent(AgentConfig(
    name="PubSubAgent",
    model=ChatOpenAI(model="gpt-4o-mini"),  # or None to run tool-only
    system_prompt="You are a helpful assistant.",
    enable_pubsub=True,
    pubsub_sub_channels=["agent:in"],
    pubsub_pub_channel="agent:out"
))

# Publish with redis-cli:
# redis-cli -u $REDIS_URL PUBLISH agent:in '{"input":"Hello"}'
```

#### 5) Multi-Agent System
```python
# Create agents
coder = create_coder()
tester = create_tester()

# Supervisor config
supervisor_config = AgentConfig(
    name="Supervisor",
    model=ChatOpenAI(model="gpt-4"),
    enable_supervisor=True,
    agents={
        "coder": coder,
        "tester": tester
    }
)

supervisor = CoreAgent(supervisor_config)
```

### 💡 Tips

#### 1) Start Minimal
```python
# ❌ Complex
config = AgentConfig(
    name="MyAgent",
    model=model,
    enable_memory=True,
    memory_backend="redis",
    redis_url="...",
    enable_rate_limiting=True,
    requests_per_second=5,
    enable_evaluation=True,
    # ... 20 more parameters
)

# ✅ Start simple
config = AgentConfig(
    name="MyAgent",
    model=model,
    system_prompt="You are a helpful assistant."
)
```

#### 2) Factory Pattern
```python
class AgentFactory:
    @staticmethod
    def create_chatbot(name: str = "Chatbot"):
        return CoreAgent(AgentConfig(
            name=name,
            model=ChatOpenAI(model="gpt-4o-mini"),
            system_prompt="You are a friendly chatbot.",
            enable_memory=True
        ))
    
    @staticmethod
    def create_coder(name: str = "Coder"):
        return CoreAgent(AgentConfig(
            name=name,
            model=ChatOpenAI(model="gpt-4"),
            system_prompt="You are a Python expert.",
            tools=create_python_coding_tools()
        ))

# Usage
chatbot = AgentFactory.create_chatbot()
coder = AgentFactory.create_coder()
```

#### 3) Load Config from JSON
```python
import json

def create_agent_from_json(json_path: str):
    with open(json_path) as f:
        config_dict = json.load(f)
    
    # Create model separately
    model = ChatOpenAI(model=config_dict.pop("model_name", "gpt-4o-mini"))
    
    config = AgentConfig(
        model=model,
        **config_dict
    )
    
    return CoreAgent(config)

# config.json
{
    "name": "MyAgent",
    "system_prompt": "You are an assistant.",
    "enable_memory": true,
    "memory_backend": "inmemory"
}
```

### 🎓 Summary

1. **Start simple**: Start with just `name`, `model`, `system_prompt`
2. **Add as needed**: Memory, tools, rate limiting etc. only if necessary
3. **Config is not complex**: Use only the parameters you need
4. **Factory pattern**: Write reusable agent creators
5. **Test**: Test with inmemory backend first, then move to production

### More Examples

Check `core/test_core/simple_agent_creators.py` for 20+ ready-to-use agent creator examples!