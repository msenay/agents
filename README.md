# CoreAgent Framework

A comprehensive LangGraph-based agent framework that provides a foundation for creating specialized AI agents with optional advanced features.

## 🌟 Features

The CoreAgent framework includes all the features mentioned in the LangGraph documentation:

### Core Capabilities
- **Subgraph Encapsulation**: Reusable core components for building complex agent workflows
- **Persistent Memory**: Multi-session long memory with RedisSaver support
- **SupervisorGraph**: Hierarchical multi-agent orchestration (sequential/DAG, retry, checkpoint)
- **Prebuilt Components**: Ready-to-use LangGraph components for rapid development
- **Memory Management**: Both short-term and long-term memory capabilities
- **Human-in-the-Loop**: Asynchronous approval, correction, and intervention
- **Streaming Support**: Real-time streaming of agent states and outputs
- **Agent Evaluation**: Built-in performance evaluation utilities

### LangGraph Ecosystem Support
- `langgraph-prebuilt`: Prebuilt components to create agents ✅ **Included**
- `langgraph-supervisor`: Tools for building supervisor agents ✅ **Supported**
- `langgraph-swarm`: Tools for building swarm multi-agent systems ✅ **Supported**
- `langchain-mcp-adapters`: Interfaces to MCP servers for tool and resource integration ✅ **Supported**
- `langmem`: Agent memory management (short-term and long-term) ✅ **Supported**
- `agentevals`: Utilities to evaluate agent performance ✅ **Supported**

## 🚀 Quick Start

### Installation

```bash
# Install core dependencies
pip install -r requirements.txt

# For full functionality, install optional packages:
pip install -U langgraph-supervisor langgraph-swarm langchain-mcp-adapters langmem agentevals
```

### Basic Usage

```python
from core_agent import create_basic_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Define tools
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    return str(eval(expression))

# Create a basic agent
model = ChatOpenAI("gpt-4")
agent = create_basic_agent(model=model, tools=[calculate])

# Use the agent
result = agent.invoke("What is 15 * 7?")
print(result)
```

### Advanced Configuration

```python
from core_agent import CoreAgent, AgentConfig

# Custom configuration
config = AgentConfig(
    name="MySpecializedAgent",
    model=ChatOpenAI("gpt-4"),
    tools=[calculate],
    enable_memory=True,
    memory_type="redis",
    redis_url="redis://localhost:6379",
    enable_supervisor=True,
    enable_evaluation=True,
    enable_streaming=True,
    enable_human_feedback=True,
    system_prompt="You are a specialized assistant for mathematical tasks."
)

agent = CoreAgent(config)
```

## 📋 Configuration Options

The `AgentConfig` class provides comprehensive configuration options:

### Core Settings
- `name`: Agent name
- `description`: Agent description  
- `model`: Language model to use
- `system_prompt`: System prompt for the agent

### Tools and Capabilities
- `tools`: List of tools available to the agent
- `tool_calling_enabled`: Enable/disable tool calling
- `pre_model_hook`: Function called before model invocation
- `post_model_hook`: Function called after model invocation
- `response_format`: Pydantic model for structured responses

### Memory Settings
- `enable_memory`: Enable memory functionality
- `memory_type`: "memory", "redis", or "both"
- `redis_url`: Redis connection URL for persistent memory

### Advanced Features
- `enable_supervisor`: Enable supervisor capabilities
- `enable_swarm`: Enable swarm multi-agent features
- `enable_mcp`: Enable MCP server integration
- `enable_evaluation`: Enable agent evaluation

### Human-in-the-Loop
- `enable_human_feedback`: Enable human feedback
- `interrupt_before`: List of nodes to interrupt before
- `interrupt_after`: List of nodes to interrupt after

### Other Options
- `enable_streaming`: Enable response streaming
- `enable_subgraphs`: Enable subgraph functionality
- `evaluation_metrics`: List of evaluation metrics to use

## 🎯 Creating Specialized Agents

The framework is designed to let you create specialized agents by extending the `CoreAgent` class:

```python
class CodeReviewAgent(CoreAgent):
    def __init__(self, model):
        config = AgentConfig(
            name="CodeReviewAgent",
            model=model,
            system_prompt="You are an expert code reviewer...",
            tools=[analyze_code, search_documentation, run_tests],
            response_format=CodeAnalysisResult,
            evaluation_metrics=["technical_accuracy", "helpfulness"]
        )
        super().__init__(config)
    
    def review_code(self, code: str, language: str = "python"):
        """Specialized method for code review"""
        return self.invoke(f"Review this {language} code: {code}")
```

## 🏗️ Architecture Overview

### Core Components

1. **CoreAgent**: Main agent class with all functionality
2. **AgentConfig**: Configuration dataclass for agent settings
3. **CoreAgentState**: Pydantic model defining agent state
4. **SubgraphManager**: Manages reusable subgraph components
5. **MemoryManager**: Handles short-term and long-term memory
6. **SupervisorManager**: Coordinates multi-agent workflows
7. **EvaluationManager**: Handles agent performance evaluation

### Factory Functions

- `create_basic_agent()`: Creates a simple agent with minimal config
- `create_advanced_agent()`: Creates an agent with enhanced capabilities  
- `create_supervisor_agent()`: Creates a supervisor for multi-agent coordination

## 🔧 Examples

### Memory Management
```python
# Store and retrieve from memory
agent.store_memory("user_preferences", {"language": "Turkish"})
preferences = agent.retrieve_memory("user_preferences")
```

### Subgraph Usage
```python
# Create and register reusable subgraphs
tool_subgraph = agent.subgraph_manager.create_tool_subgraph(tools)
agent.add_subgraph("tool_execution", tool_subgraph)
```

### Multi-Agent Coordination

The framework supports three multi-agent patterns:

#### 1. Supervisor Pattern (Central Coordination)
```python
# Create specialized agents
flight_agent = FlightAgent(model)
hotel_agent = HotelAgent(model)

# Create supervisor
agents = {"flight": flight_agent, "hotel": hotel_agent}
supervisor = create_supervisor_agent(model, agents)

# Coordinate tasks
result = supervisor.coordinate_task("Book a flight and hotel")
```

#### 2. Swarm Pattern (Dynamic Handoffs)
```python
# Create swarm system
agents = {"flight": flight_agent, "hotel": hotel_agent}
swarm = create_swarm_agent(model, agents, default_active_agent="flight")

# Dynamic coordination
result = swarm.coordinate_task("Plan my trip")
```

#### 3. Handoff Pattern (Manual Transfers)
```python
# Create handoff system
agents = {"flight": flight_agent, "hotel": hotel_agent}
handoff = create_handoff_agent(model, agents, default_active_agent="flight")

# Manual transfers between agents
result = handoff.coordinate_task("Help with travel planning")
```

### Streaming Responses
```python
# Stream agent responses
for chunk in agent.stream("Analyze this data"):
    print(chunk)
```

### Human-in-the-Loop
```python
# Configure interruption points for human feedback
config = AgentConfig(
    enable_human_feedback=True,
    interrupt_before=["execute_tools"],
    interrupt_after=["generate_response"]
)
```

## 📁 Project Structure

```
├── core_agent.py              # Main CoreAgent framework
├── examples/
│   ├── basic_usage.py         # Basic usage examples
│   └── specialized_agents.py  # Specialized agent examples
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## 🧪 Testing

```bash
# Run examples
python examples/basic_usage.py
python examples/specialized_agents.py

# Run with actual model (requires API key)
export OPENAI_API_KEY="your-key-here"
python examples/basic_usage.py
```

## 🛠️ Development

### Optional Dependencies

The framework gracefully handles missing optional dependencies. Install only what you need:

```bash
# For supervisor functionality
pip install langgraph-supervisor

# For swarm multi-agent systems  
pip install langgraph-swarm

# For MCP server integration
pip install langchain-mcp-adapters

# For advanced memory management
pip install langmem

# For agent evaluation
pip install agentevals

# For Redis persistent memory
pip install redis
```

### Extending the Framework

1. **Create Specialized Agents**: Extend `CoreAgent` with custom configurations
2. **Add Custom Tools**: Create `@tool` decorated functions for your use case
3. **Define Response Formats**: Use Pydantic models for structured outputs
4. **Implement Custom Hooks**: Add pre/post model processing logic
5. **Create Subgraphs**: Build reusable workflow components

## 📚 Documentation

### Key Classes and Methods

#### CoreAgent
- `invoke(input_data)`: Synchronous agent execution
- `ainvoke(input_data)`: Asynchronous agent execution  
- `stream(input_data)`: Stream responses
- `add_subgraph(name, subgraph)`: Register reusable subgraphs
- `store_memory(key, value)`: Store information in memory
- `retrieve_memory(key)`: Retrieve from memory
- `get_status()`: Get agent capabilities and status

#### AgentConfig
Complete configuration options for customizing agent behavior, tools, memory, and advanced features.

#### Factory Functions
- `create_basic_agent(model, tools)`: Quick agent creation
- `create_advanced_agent(model, **options)`: Feature-rich agent
- `create_supervisor_agent(model, agents)`: Supervisor pattern coordination
- `create_swarm_agent(model, agents, default_agent)`: Swarm pattern coordination  
- `create_handoff_agent(model, agents, default_agent)`: Handoff pattern coordination

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your enhancements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

Built on top of the excellent [LangGraph](https://github.com/langchain-ai/langgraph) framework by LangChain AI.

---

**Note**: Bu CoreAgent framework, dokümandaki tüm özellikleri opsiyonel olarak içerir. Farklı prompt'lar, tool'lar ve konfigürasyonlarla yeni agenlar yaratabilirsiniz. Her özellik isteğe bağlı olarak aktif edilebilir ve framework eksik bağımlılıkları zarif bir şekilde handle eder.
