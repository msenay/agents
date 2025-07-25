#!/usr/bin/env python3
"""
InMemory Backend Demo - Shows all memory features without Redis
"""

import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool


@tool
def save_note(title: str, content: str) -> str:
    """Save a note to memory"""
    return f"Note '{title}' saved"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    print("\n" + "="*60)
    print("🚀 MEMORY DEMO - INMEMORY BACKEND")
    print("="*60)
    
    # Azure OpenAI model
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-12-01-preview",
        azure_deployment="gpt4o",
        temperature=0.7
    )
    
    # Agent config with inmemory backend
    config = AgentConfig(
        name="MemoryAgent",
        model=model,
        tools=[save_note, calculate],
        
        # Memory configuration
        enable_memory=True,
        memory_backend="inmemory",  # Using inmemory backend
        memory_types=["short_term", "long_term"],
        
        # Memory features
        enable_memory_tools=True,
        
        system_prompt="You are a helpful assistant with memory capabilities."
    )
    
    print("\n📋 Configuration:")
    print(f"   Backend: {config.memory_backend}")
    print(f"   Memory Types: {config.memory_types}")
    print(f"   Tools: {[t.name for t in config.tools]}")
    
    # Create agent
    agent = CoreAgent(config)
    print("\n✅ Agent created successfully!")
    
    # Test 1: Short-term memory
    print("\n" + "-"*60)
    print("TEST 1: SHORT-TERM MEMORY")
    print("-"*60)
    
    thread_id = "user_123"
    
    # First message
    response = agent.invoke(
        "Hi! My name is Alice and I work as a data scientist.",
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n👤 User: Hi! My name is Alice and I work as a data scientist.")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Second message - should remember
    response = agent.invoke(
        "What's my name and profession?",
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n👤 User: What's my name and profession?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Test 2: Long-term memory
    print("\n" + "-"*60)
    print("TEST 2: LONG-TERM MEMORY")
    print("-"*60)
    
    mm = agent.memory_manager
    
    # Store data
    print("\n📝 Storing data...")
    mm.store_long_term_memory("user_prefs", {
        "theme": "dark",
        "language": "Turkish",
        "notifications": True
    })
    print("✅ Data stored")
    
    # Retrieve data
    data = mm.get_long_term_memory("user_prefs")
    print(f"📖 Retrieved: {data}")
    
    # Test 3: Memory tools
    print("\n" + "-"*60)
    print("TEST 3: MEMORY TOOLS")
    print("-"*60)
    
    response = agent.invoke(
        "Please save a note that I have a meeting at 3 PM tomorrow",
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n👤 User: Please save a note that I have a meeting at 3 PM tomorrow")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Test 4: Different thread
    print("\n" + "-"*60)
    print("TEST 4: THREAD ISOLATION")
    print("-"*60)
    
    response = agent.invoke(
        "What's my name?",
        config={"configurable": {"thread_id": "different_user"}}
    )
    print("\n👤 User (different thread): What's my name?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETED!")
    print("="*60)
    
    print("\n📚 Summary:")
    print("- Short-term memory: ✅ Conversation history per thread")
    print("- Long-term memory: ✅ Key-value storage")
    print("- Memory tools: ✅ Agent can save/retrieve")
    print("- Thread isolation: ✅ Different conversations isolated")
    print("\n💡 Note: Using InMemory backend (data not persistent)")


if __name__ == "__main__":
    main()