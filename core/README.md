# 🤖 Core Agent Framework

> **Enterprise-grade LangGraph-based agent framework for building sophisticated AI systems**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Creating Agents](#creating-agents)
- [Configuration Examples](#configuration-examples)
- [Specialized Agents](#specialized-agents)
- [Architecture](#architecture)
- [Advanced Features](#advanced-features)
- [Best Practices](#best-practices)
- [Testing](#testing)

## 🎯 Overview

The **Core Agent Framework** is a production-ready foundation for building AI agents with LangGraph. It provides a comprehensive set of features including memory management, multi-agent coordination, tool integration, and evaluation capabilities - all with a clean, extensible architecture.

### Why Core Agent Framework?

- **🏗️ Built on LangGraph** - Leverages the power of graph-based agent workflows
- **🧩 Modular Design** - Enable only the features you need
- **🚀 Production Ready** - Battle-tested with comprehensive test coverage
- **⚡ High Performance** - Optimized for speed with built-in rate limiting
- **🔧 Extensible** - Easy to add custom tools, memory backends, and patterns
- **📊 Observable** - Built-in monitoring, evaluation, and debugging tools

## ✨ Key Features

### Core Capabilities
- **State Management** - Robust conversation state handling
- **Tool Integration** - Seamless integration with LangChain tools
- **Memory Systems** - Multiple backends (Redis, PostgreSQL, MongoDB)
- **Multi-Agent Patterns** - Supervisor, Swarm, and Handoff patterns
- **Evaluation Framework** - Built-in performance evaluation
- **MCP Support** - Model Context Protocol integration
- **Rate Limiting** - Protect against API limits
- **Streaming** - Real-time response streaming

### Memory Features
- **Short-term Memory** - Conversation context
- **Long-term Memory** - Persistent knowledge storage
- **Session Memory** - Shared memory between agents
- **Semantic Memory** - Vector-based similarity search
- **TTL Support** - Automatic memory expiration

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd workspace

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

#### Option 1: Use Pre-built Specialized Agents

```python
# Use the CoderAgent for development tasks
from agent.coder import CoderAgent
coder = CoderAgent()
response = coder.chat("Create a simple data processing agent")

# Use the TesterAgent for testing
from agent.tester import TesterAgent
tester = TesterAgent()
response = tester.chat("Write tests for a user authentication function")

# Use the ExecutorAgent for code execution
from agent.executor import ExecutorAgent
executor = ExecutorAgent()
response = executor.chat("Run a simple Python calculation")
```

#### Option 2: Create Custom Agents with CoreAgent

```python
from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

# Create a simple agent
config = AgentConfig(
    name="Assistant",
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a helpful AI assistant."
)

agent = CoreAgent(config)
response = agent.invoke("Hello! How can you help me?")
print(response)
```

## 🛠️ Creating Agents

### Step 1: Define Your Agent's Purpose

Every agent should have a clear, well-defined purpose:

```python
from core import AgentConfig

# Define agent configuration
config = AgentConfig(
    name="DataAnalyst",
    description="Analyzes data and provides insights",
    system_prompt="""You are an expert data analyst. 
    You help users understand their data through analysis, 
    visualization recommendations, and insights."""
)
```

### Step 2: Add Tools

Equip your agent with the tools it needs:

```python
from core.tools import PythonExecutorTool, FileReaderTool

config = AgentConfig(
    name="DataAnalyst",
    tools=[
        PythonExecutorTool(),    # Execute Python code
        FileReaderTool(),        # Read data files
    ]
)
```

### Step 3: Configure Memory (Optional)

Add memory for context retention:

```python
config = AgentConfig(
    name="DataAnalyst",
    enable_memory=True,
    memory_types=["short_term", "long_term"],
    memory_backend="redis",
    redis_url="redis://localhost:6379"
)
```

### Step 4: Initialize and Use

```python
from core import CoreAgent

# Create the agent
agent = CoreAgent(config)

# Use the agent
response = agent.invoke("Analyze the sales data from last quarter")
print(response)

# Check agent status
status = agent.get_status()
print(f"Agent: {status['name']} - Features: {status['features']}")
```

## 📊 Configuration Examples

### 🧑‍💻 Coder Agent Configuration

A specialized agent for code generation and analysis:

```python
from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI
from core.tools import PythonExecutorTool, FileWriterTool

coder_config = AgentConfig(
    name="CoderAgent",
    description="Expert software developer specializing in Python",
    
    # Model configuration
    model=AzureChatOpenAI(
        deployment_name="gpt-4",
        temperature=0.1,  # Low temperature for consistent code
        max_tokens=4000
    ),
    
    # System prompt
    system_prompt="""You are an expert software developer.
    You write clean, efficient, and well-documented code.
    You follow best practices and design patterns.
    You always include error handling and tests.""",
    
    # Tools
    tools=[
        PythonExecutorTool(),
        FileWriterTool(),
    ],
    
    # Memory configuration
    enable_memory=True,
    memory_types=["short_term", "long_term"],
    memory_backend="redis",
    session_id="coding_session",
    
    # Performance
    enable_rate_limiting=True,
    requests_per_second=5.0,
    
    # Features
    enable_message_trimming=True,
    max_tokens=8000,
    enable_streaming=True
)

coder = CoreAgent(coder_config)
```

### 🧪 Unit Tester Agent Configuration

An agent specialized in writing and executing tests:

```python
from agent.tester import TesterAgent

# The TesterAgent is now a specialized agent with comprehensive testing tools
# It includes tools for test generation, fixtures, mocking, and coverage analysis
tester = TesterAgent()

# Or create with custom configuration
unittester_config = AgentConfig(
    name="TesterAgent",
    description="Expert in test-driven development and quality assurance",
    
    # Model configuration
    model=ChatOpenAI(
        model="gpt-4",
        temperature=0.2,  # Slightly higher for creative test cases
    ),
    
    # System prompt
    system_prompt="""You are a testing expert specializing in:
    - Writing comprehensive unit tests
    - Creating test fixtures and mocks
    - Identifying edge cases
    - Ensuring high code coverage
    - Using pytest best practices
    Always write tests that are maintainable and clear.""",
    
    # Tools
    tools=[
        PythonExecutorTool(sandbox_mode=True),  # Safe execution
        FileReaderTool(),
        FileWriterTool(),
    ],
    
    # Memory for test patterns
    enable_memory=True,
    memory_types=["long_term"],  # Remember successful test patterns
    memory_backend="postgres",
    postgres_url="postgresql://localhost/testdb",
    
    # Evaluation
    enable_evaluation=True,
    evaluation_metrics=["test_coverage", "test_quality"],
    
    # Safety
    enable_human_feedback=True,
    interrupt_before=["file_write"]  # Review before writing test files
)

tester = CoreAgent(unittester_config)
```

### ⚙️ Executor Agent Configuration

An agent for safely executing and monitoring code:

```python
from agent.executor import ExecutorAgent

# The ExecutorAgent is now a specialized agent with built-in tools
# It provides safe code execution with comprehensive test running capabilities
executor = ExecutorAgent()

# Or create with custom configuration
executor_config = AgentConfig(
    name="ExecutorAgent",
    description="Safe code execution and monitoring specialist",
    
    # Model configuration
    model=ChatOpenAI(
        model="gpt-3.5-turbo",  # Faster model for execution tasks
        temperature=0.0,  # Deterministic execution
    ),
    
    # System prompt
    system_prompt="""You are a code execution specialist.
    Your responsibilities:
    - Safely execute user code
    - Monitor resource usage
    - Capture and report outputs
    - Handle errors gracefully
    - Enforce security constraints
    Never execute unsafe or malicious code.""",
    
    # Tools with restrictions
    tools=[
        PythonExecutorTool(
            sandbox_mode=True,
            timeout=30,  # 30 second timeout
            memory_limit="512MB",
            allowed_modules=["numpy", "pandas", "matplotlib"]
        ),
    ],
    
    # Minimal memory (execution focused)
    enable_memory=True,
    memory_types=["short_term"],
    memory_backend="inmemory",
    
    # Strict rate limiting
    enable_rate_limiting=True,
    requests_per_second=1.0,  # Prevent abuse
    
    # Monitoring
    enable_evaluation=True,
    evaluation_metrics=["execution_safety", "resource_usage"],
    
    # Human oversight for sensitive operations
    enable_human_feedback=True,
    interrupt_before=["code_execution"],
    interrupt_after=["error_detection"]
)

executor = CoreAgent(executor_config)
```

### 🤝 Multi-Agent System Configuration

Combining multiple agents into a coordinated system:

```python
# Create specialized agents
coder = CoreAgent(coder_config)
tester = CoreAgent(unittester_config)
executor = CoreAgent(executor_config)

# Supervisor configuration
supervisor_config = AgentConfig(
    name="DevelopmentSupervisor",
    description="Coordinates the software development workflow",
    
    model=ChatOpenAI(model="gpt-4", temperature=0.3),
    
    system_prompt="""You coordinate a development team consisting of:
    - Coder: Writes implementation code
    - Tester: Creates and runs tests
    - Executor: Safely executes and validates code
    
    Delegate tasks appropriately and ensure quality.""",
    
    # Enable supervisor pattern
    enable_supervisor=True,
    agents={
        "coder": coder,
        "tester": tester,
        "executor": executor
    },
    
    # Workflow memory
    enable_memory=True,
    memory_types=["session", "long_term"],
    memory_backend="redis",
    session_id="dev_workflow",
    
    # Monitoring
    enable_evaluation=True,
    evaluation_metrics=["task_completion", "code_quality"]
)

supervisor = CoreAgent(supervisor_config)

# Use the system
result = supervisor.coordinate_task(
    "Create a Python function to calculate fibonacci numbers with tests"
)
```

## 🎯 Specialized Agents

The framework includes several pre-built specialized agents that extend the CoreAgent with domain-specific tools and capabilities:

### 🚀 **CoderAgent**
A comprehensive agent for generating, analyzing, and optimizing LangGraph agents.

```python
from agent.coder import CoderAgent

# Initialize with 12 specialized development tools
coder = CoderAgent(session_id="dev_session")

# Generate a new agent
result = coder.generate_agent(
    template_type="simple",
    agent_name="DataProcessor",
    purpose="Process and analyze data",
    use_our_core=True  # Use CoreAgent infrastructure
)
```

**Key Features:**
- 12 specialized tools for agent development
- Supports both standalone LangGraph and CoreAgent modes
- Intelligent tool chaining for complex workflows
- Auto-generates tests, docs, and deployment configs

### 🧪 **TesterAgent**
Expert in generating comprehensive unit tests with pytest.

```python
from agent.tester import TesterAgent

# Initialize with testing-specific tools
tester = TesterAgent()

# Generate tests for existing code
response = tester.chat("""
Generate comprehensive unit tests for this function:

def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
""")
```

**Key Features:**
- Generates pytest-compatible test suites
- Creates fixtures and mocks automatically
- Identifies edge cases and error conditions
- Follows testing best practices

### ⚙️ **ExecutorAgent**
Safely executes code and runs tests with comprehensive monitoring.

```python
from agent.executor import ExecutorAgent

# Initialize with execution tools
executor = ExecutorAgent()

# Execute code safely
response = executor.chat("""
Execute this code and show me the output:

import numpy as np
data = np.random.randn(1000)
print(f"Mean: {data.mean():.2f}")
print(f"Std: {data.std():.2f}")
""")
```

**Key Features:**
- Safe code execution with sandboxing
- Resource monitoring and limits
- Test execution with coverage reports
- Error handling and recovery

### 🎭 **OrchestratorAgent**
Coordinates multiple agents for complete development workflows.

```python
from agent.orchestrator import OrchestratorAgent

# Initialize with supervisor pattern (default)
orchestrator = OrchestratorAgent()

# Orchestrate a complete workflow
result = orchestrator.orchestrate(
    "Create a REST API with authentication and tests",
    workflow_type="full_development"
)

# Or use different coordination patterns
orchestrator_swarm = OrchestratorAgent(coordination_pattern="swarm")  # Parallel
orchestrator_pipeline = OrchestratorAgent(coordination_pattern="pipeline")  # Sequential
orchestrator_adaptive = OrchestratorAgent(coordination_pattern="adaptive")  # Auto-select
```

**Key Features:**
- 4 coordination patterns (Supervisor, Swarm, Pipeline, Adaptive)
- Manages CoderAgent, TesterAgent, and ExecutorAgent in harmony
- Built-in workflow templates
- Quality control between steps
- Progress monitoring and reporting

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                     CoreAgent                           │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │    State     │  │    Graph     │  │   Manager    │ │
│  │  Management  │  │  Compiler    │  │ Coordinator  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────┤
│                      Managers                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Memory     │  │   Rate       │  │  Supervisor  │ │
│  │   Manager    │  │  Limiter     │  │   Manager    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │     MCP      │  │  Subgraph    │  │ Evaluation   │ │
│  │   Manager    │  │   Manager    │  │   Manager    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### Manager Responsibilities

- **MemoryManager**: Handles all memory operations across different backends
- **RateLimiterManager**: Prevents API rate limit violations
- **SupervisorManager**: Coordinates multi-agent workflows
- **MCPManager**: Integrates Model Context Protocol servers
- **SubgraphManager**: Manages reusable graph components
- **EvaluationManager**: Evaluates agent performance

## 🚀 Advanced Features

### Custom Tools

Create specialized tools for your agents:

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class DataAnalysisTool(BaseTool):
    name = "analyze_data"
    description = "Performs statistical analysis on datasets"
    
    class InputSchema(BaseModel):
        data_path: str = Field(description="Path to data file")
        analysis_type: str = Field(description="Type of analysis")
    
    args_schema = InputSchema
    
    def _run(self, data_path: str, analysis_type: str) -> str:
        # Implementation here
        return f"Analysis results for {data_path}"

# Add to agent
config.tools.append(DataAnalysisTool())
```

### Memory Patterns

```python
# Semantic memory for similarity search
config = AgentConfig(
    enable_memory=True,
    memory_types=["semantic"],
    embedding_model="openai:text-embedding-3-small",
    memory_backend="postgres",  # Supports vector operations
)

# TTL-based memory for temporary data
config = AgentConfig(
    enable_memory=True,
    memory_backend="redis",
    enable_ttl=True,
    default_ttl_minutes=60,  # Expire after 1 hour
)
```

### Evaluation and Monitoring

```python
# Enable comprehensive evaluation
config = AgentConfig(
    enable_evaluation=True,
    evaluation_metrics=["accuracy", "relevance", "safety", "performance"],
    custom_evaluators=[MyCustomEvaluator()],
)

# Get evaluation results
agent = CoreAgent(config)
response = agent.invoke("Generate a sorting algorithm")
scores = agent.evaluate_last_response()
print(f"Evaluation scores: {scores}")
```

## 💡 Best Practices

### 1. **Agent Design**
- Keep agents focused on a single responsibility
- Use clear, specific system prompts
- Choose appropriate models for the task

### 2. **Memory Management**
- Use `inmemory` for development, Redis/PostgreSQL for production
- Enable TTL for temporary data to prevent memory bloat
- Use semantic memory for knowledge-based agents

### 3. **Tool Safety**
- Always use sandbox mode for code execution tools
- Implement timeouts and resource limits
- Validate tool inputs and outputs

### 4. **Performance**
- Enable rate limiting for all production agents
- Use streaming for long responses
- Implement message trimming for long conversations

### 5. **Multi-Agent Systems**
- Use supervisor pattern for complex workflows
- Keep inter-agent communication clear and structured
- Monitor agent coordination with evaluation metrics

### 6. **Testing**
- Test each agent configuration thoroughly
- Use mock tools for unit testing
- Implement integration tests for multi-agent systems

## 🧪 Testing

### Run Tests

```bash
# Run all tests
python -m pytest core/test_core/

# Run specific test suite
python core/test_core/test_core_agent_comprehensive.py

# Run with coverage
python -m pytest --cov=core core/test_core/
```

### Test Your Agents

```python
# Example test for a custom agent
import unittest
from core import CoreAgent, AgentConfig

class TestMyAgent(unittest.TestCase):
    def setUp(self):
        self.config = AgentConfig(
            name="TestAgent",
            model=MockLLM(),  # Use mock for testing
            tools=[MockTool()],
        )
        self.agent = CoreAgent(self.config)
    
    def test_agent_response(self):
        response = self.agent.invoke("Test query")
        self.assertIsNotNone(response)
        self.assertIn("expected_content", response)
    
    def test_agent_status(self):
        status = self.agent.get_status()
        self.assertEqual(status['name'], "TestAgent")
        self.assertEqual(status['tools_count'], 1)
```

## 📚 Documentation

For more detailed documentation:

- [Configuration Guide](docs/configuration.md)
- [Memory Systems](docs/memory.md)
- [Multi-Agent Patterns](docs/multi-agent.md)
- [Tool Development](docs/tools.md)
- [API Reference](docs/api.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Ready to build intelligent agents? Get started with the examples above!** 🚀