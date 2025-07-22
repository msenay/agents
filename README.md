# 🤖 AI Agent Development Platform

> **Enterprise-grade platform for building, testing, and deploying AI agents with LangGraph**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Latest-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 🎯 Overview

This platform provides a comprehensive ecosystem for developing AI agents using LangGraph. It includes:

- **Core Agent Framework** - A robust foundation for building custom agents
- **Specialized Agents** - Pre-built agents for common tasks (code generation, testing, execution)
- **Docker Support** - Easy deployment with Docker Compose
- **Comprehensive Testing** - Built-in test suites and validation tools

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker & Docker Compose (optional, for containerized deployment)
- Azure OpenAI API access (or OpenAI API)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd workspace

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export OPENAI_API_KEY="your-api-key"
```

### Basic Usage

#### Option 1: Use Pre-built Agents

```python
# Code Generation
from agent.coder import CoderAgent
coder = CoderAgent()
result = coder.generate_agent(
    template_type="simple",
    agent_name="DataProcessor",
    purpose="Process CSV files and generate reports"
)

# Test Generation
from agent.tester import TesterAgent
tester = TesterAgent()
response = tester.chat("Generate tests for my authentication module")

# Code Execution
from agent.executor import ExecutorAgent
executor = ExecutorAgent()
response = executor.chat("Run pytest on my test suite")
```

#### Option 2: Build Custom Agents

```python
from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

config = AgentConfig(
    name="MyCustomAgent",
    model=ChatOpenAI(model="gpt-4"),
    system_prompt="You are a specialized assistant for data analysis.",
    tools=[...],  # Add your tools
    enable_memory=True
)

agent = CoreAgent(config)
response = agent.invoke("Analyze this dataset...")
```

## 📦 Project Structure

```
workspace/
├── core/                  # Core Agent Framework
│   ├── agents.py         # CoreAgent implementation
│   ├── config.py         # Configuration classes
│   ├── managers/         # Feature managers (memory, rate limiting, etc.)
│   └── README.md         # Framework documentation
├── agent/                # Specialized Agents
│   ├── coder/           # CoderAgent - generates agents
│   ├── tester/          # TesterAgent - generates tests
│   └── executor/        # ExecutorAgent - executes code safely
├── tests/               # Test suites
├── docker-compose.yml   # Docker deployment
└── requirements.txt     # Python dependencies
```

## 🎯 Specialized Agents

### 🚀 CoderAgent
**Purpose:** Generate, analyze, and optimize LangGraph agents

- 12 specialized tools for agent development
- Supports standalone and Core Agent modes
- Auto-generates tests and documentation
- [Full Documentation](agent/coder/README_coder.md)

### 🧪 TesterAgent
**Purpose:** Generate comprehensive unit tests

- Creates pytest-compatible test suites
- Automatic fixture and mock generation
- Edge case identification
- Coverage analysis
- [Full Documentation](agent/tester/README_tester.md)

### ⚙️ ExecutorAgent
**Purpose:** Safely execute code and run tests

- Sandboxed code execution
- Resource monitoring and limits
- Test execution with reports
- Error handling and recovery
- [Full Documentation](agent/executor/README_executor.md)

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build and run all services
docker-compose up --build

# Or run specific services
docker-compose up redis postgres
```

### Services Included

- **Core Agent API** - REST API for agent interactions
- **Redis** - For memory and caching
- **PostgreSQL** - For persistent storage
- **MongoDB** - Alternative storage option

## 🛠️ Configuration

### Environment Variables

```bash
# Azure OpenAI (recommended)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
OPENAI_API_KEY=your-api-key
OPENAI_API_VERSION=2023-12-01-preview

# Or standard OpenAI
OPENAI_API_KEY=your-openai-key

# Memory Backends (optional)
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost/agentdb
MONGODB_URL=mongodb://localhost:27017/agents
```

### Agent Configuration

```python
from core import AgentConfig

config = AgentConfig(
    # Basic settings
    name="MyAgent",
    description="Agent purpose",
    
    # Model configuration
    model=ChatOpenAI(model="gpt-4", temperature=0.7),
    
    # Features
    enable_memory=True,
    memory_backend="redis",
    enable_rate_limiting=True,
    enable_streaming=True,
    
    # Tools
    tools=[...],
)
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest

# Run core framework tests
python -m pytest core/test_core/

# Run with coverage
python -m pytest --cov=core --cov=agent

# Run specific test
python core/test_core/test_core_agent_comprehensive.py
```

## 📚 Documentation

- [Core Agent Framework Documentation](core/README.md)
- [CoderAgent Documentation](agent/coder/README_coder.md)
- [TesterAgent Documentation](agent/tester/README_tester.md)
- [ExecutorAgent Documentation](agent/executor/README_executor.md)
- [Configuration Guide](core/README.md#configuration-examples)
- [Multi-Agent Patterns](core/README.md#multi-agent-system-configuration)

## 🚀 Example: Complete Development Workflow

```python
from agent.coder import CoderAgent
from agent.tester import TesterAgent
from agent.executor import ExecutorAgent

# 1. Generate an agent
coder = CoderAgent()
result = coder.chat(
    "Create a web scraping agent that monitors product prices "
    "and sends alerts when prices drop"
)

# 2. Generate tests for it
tester = TesterAgent()
tests = tester.chat(f"Generate comprehensive tests for this code:\n{result}")

# 3. Execute the tests
executor = ExecutorAgent()
test_results = executor.chat(f"Run these tests:\n{tests}")

print("✅ Agent created, tested, and validated!")
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on [LangGraph](https://langchain-ai.github.io/langgraph/) by LangChain
- Powered by Azure OpenAI GPT-4
- Inspired by modern agent architectures

---

**Ready to build intelligent agents? Get started with the examples above!** 🚀