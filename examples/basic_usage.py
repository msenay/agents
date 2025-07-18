"""
Basic Usage Examples for CoreAgent Framework

This file demonstrates how to use the CoreAgent to create different types of agents
with various configurations and capabilities.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import CoreAgent, AgentConfig, create_basic_agent, create_advanced_agent
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel


# Mock model for demonstration (replace with actual model)
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        return "Mock response from the model"
    
    def _llm_type(self):
        return "mock"


# Example tools
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except:
        return "Invalid expression"


@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Mock search results for: {query}"


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Mock weather data for {location}: Sunny, 25°C"


# Example response format
class TaskResult(BaseModel):
    result: str
    confidence: float
    sources: list = []


def example_1_basic_agent():
    """Example 1: Create a basic agent with minimal configuration"""
    print("\n=== Example 1: Basic Agent ===")
    
    model = MockChatModel()
    tools = [calculate, search_web]
    
    # Create basic agent using factory function
    agent = create_basic_agent(model=model, tools=tools)
    
    # Get agent status
    status = agent.get_status()
    print(f"Agent Name: {status['name']}")
    print(f"Features: {status['features']}")
    
    # Use the agent
    try:
        result = agent.invoke("Calculate 15 * 7")
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")


def example_2_custom_config_agent():
    """Example 2: Create an agent with custom configuration"""
    print("\n=== Example 2: Custom Configuration Agent ===")
    
    # Custom hooks
    def pre_model_hook(state):
        print("Pre-model hook: Processing input...")
        return state
        
    def post_model_hook(state):
        print("Post-model hook: Processing output...")
        return state
    
    # Create custom configuration
    config = AgentConfig(
        name="CustomAgent",
        description="An agent with custom configuration",
        model=MockChatModel(),
        tools=[calculate, search_web, get_weather],
        system_prompt="You are a helpful assistant with access to various tools.",
        enable_memory=True,
        memory_type="memory",
        enable_streaming=True,
        enable_evaluation=False,  # Disabled for this example
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        response_format=TaskResult
    )
    
    # Create agent with custom config
    agent = CoreAgent(config)
    
    # Test the agent
    print(f"Agent: {agent.config.name}")
    print(f"Description: {agent.config.description}")
    print(f"Available tools: {len(agent.config.tools)}")


def example_3_memory_management():
    """Example 3: Demonstrate memory management capabilities"""
    print("\n=== Example 3: Memory Management ===")
    
    config = AgentConfig(
        name="MemoryAgent",
        model=MockChatModel(),
        tools=[search_web],
        enable_memory=True,
        memory_type="memory"
    )
    
    agent = CoreAgent(config)
    
    # Store information in memory
    agent.store_memory("user_preferences", {"language": "English", "style": "formal"})
    agent.store_memory("conversation_history", ["Previous conversation context"])
    
    # Retrieve from memory
    preferences = agent.retrieve_memory("user_preferences")
    history = agent.retrieve_memory("conversation_history")
    
    print(f"User preferences: {preferences}")
    print(f"Conversation history: {history}")


def example_4_subgraph_management():
    """Example 4: Demonstrate subgraph encapsulation"""
    print("\n=== Example 4: Subgraph Management ===")
    
    config = AgentConfig(
        name="SubgraphAgent",
        model=MockChatModel(),
        enable_subgraphs=True
    )
    
    agent = CoreAgent(config)
    
    # Create a tool subgraph
    tools = [calculate, get_weather]
    tool_subgraph = agent.subgraph_manager.create_tool_subgraph(tools)
    
    # Register the subgraph
    agent.add_subgraph("tool_execution", tool_subgraph)
    
    # Get the subgraph
    retrieved_subgraph = agent.get_subgraph("tool_execution")
    
    print(f"Registered subgraphs: {list(agent.subgraph_manager.subgraphs.keys())}")
    print(f"Tool subgraph available: {retrieved_subgraph is not None}")


def example_5_advanced_features():
    """Example 5: Create an advanced agent with multiple features"""
    print("\n=== Example 5: Advanced Agent Features ===")
    
    # Create advanced agent with multiple features enabled
    agent = create_advanced_agent(
        model=MockChatModel(),
        tools=[calculate, search_web, get_weather],
        enable_redis=False,  # Would require Redis server
        enable_supervisor=False,  # Would require langgraph-supervisor
        enable_evaluation=False   # Would require agentevals
    )
    
    status = agent.get_status()
    print(f"Agent: {status['name']}")
    print(f"Features enabled: {status['features']}")
    
    # Save configuration
    agent.save_config("advanced_agent_config.json")
    print("Configuration saved to advanced_agent_config.json")
    
    # Load configuration (example)
    try:
        loaded_config = CoreAgent.load_config("advanced_agent_config.json")
        print(f"Loaded config name: {loaded_config.name}")
    except Exception as e:
        print(f"Error loading config: {e}")


def example_6_streaming():
    """Example 6: Demonstrate streaming capabilities"""
    print("\n=== Example 6: Streaming ===")
    
    config = AgentConfig(
        name="StreamingAgent",
        model=MockChatModel(),
        tools=[search_web],
        enable_streaming=True
    )
    
    agent = CoreAgent(config)
    
    print("Streaming response:")
    try:
        for chunk in agent.stream("What's the weather like?"):
            print(f"Chunk: {chunk}")
    except Exception as e:
        print(f"Streaming error: {e}")


def example_7_human_in_the_loop():
    """Example 7: Human-in-the-loop configuration"""
    print("\n=== Example 7: Human-in-the-Loop ===")
    
    config = AgentConfig(
        name="HITLAgent",
        model=MockChatModel(),
        tools=[calculate],
        enable_human_feedback=True,
        interrupt_before=["execute_tools"],  # Interrupt before tool execution
        interrupt_after=["generate_response"]  # Interrupt after response generation
    )
    
    agent = CoreAgent(config)
    
    print(f"Human feedback enabled: {agent.config.enable_human_feedback}")
    print(f"Interrupt points: {agent.config.interrupt_before + agent.config.interrupt_after}")


def run_all_examples():
    """Run all examples"""
    print("CoreAgent Framework - Usage Examples")
    print("=" * 50)
    
    example_1_basic_agent()
    example_2_custom_config_agent()
    example_3_memory_management()
    example_4_subgraph_management()
    example_5_advanced_features()
    example_6_streaming()
    example_7_human_in_the_loop()
    
    print("\n" + "=" * 50)
    print("All examples completed!")


if __name__ == "__main__":
    run_all_examples()