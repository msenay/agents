# Core Agent Creation Guide

## 🚀 Quick Start

Creating a Core Agent is very simple! There are 3 basic methods:

## 1. Simplest Method - Minimal Agent

```python
from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

# Model
model = ChatOpenAI(model="gpt-4o-mini")

# Config
config = AgentConfig(
    name="MyAgent",
    model=model,
    system_prompt="You are a helpful assistant."
)

# Create agent
agent = CoreAgent(config)

# Use
response = agent.invoke("Hello!")
```

## 2. Agent with Tools

```python
from core import CoreAgent, AgentConfig
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

## 3. Agent with Memory

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

## 📋 Config Parameters (Only Use What You Need!)

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

## 🎯 Practical Examples

### 1. Chatbot Agent
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

### 2. Coder Agent
```python
from core.tools import create_python_coding_tools

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

### 3. Research Agent
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

### 4. Multi-Agent System
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

## 💡 Tips

### 1. Use Minimal Config to Start
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

### 2. Use Factory Pattern
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

### 3. Load Config from JSON
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

## 🎓 Summary

1. **Start simple**: Start with just `name`, `model`, `system_prompt`
2. **Add as needed**: Memory, tools, rate limiting etc. only if necessary
3. **Config is not complex**: Use only the parameters you need
4. **Factory pattern**: Write reusable agent creators
5. **Test**: Test with inmemory backend first, then move to production

## More Examples

Check `core/test_core/simple_agent_creators.py` for 20+ ready-to-use agent creator examples!