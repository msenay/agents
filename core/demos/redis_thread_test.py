#!/usr/bin/env python3
"""
Detailed thread isolation test for Redis memory
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI

REDIS_URL = "redis://:redis_password@localhost:6379"


def test_thread_isolation():
    """Test thread isolation in detail"""
    
    print("\n" + "="*60)
    print("🧵 THREAD ISOLATION TEST")
    print("="*60)
    
    # Create model
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-12-01-preview",
        azure_deployment="gpt4o",
        temperature=0
    )
    
    # Create agent
    config = AgentConfig(
        name="ThreadTestAgent",
        model=model,
        enable_memory=True,
        memory_backend="redis",
        redis_url=REDIS_URL,
        memory_types=["short_term"],
        system_prompt="You are a helpful assistant. Only answer based on information from the current conversation."
    )
    
    agent = CoreAgent(config)
    print("✅ Agent created\n")
    
    # Thread 1: Alice
    print("--- Thread 1: user_alice ---")
    response = agent.invoke(
        "Hi! My name is Alice and I love Python.",
        config={"configurable": {"thread_id": "user_alice"}}
    )
    print(f"👤 Alice: Hi! My name is Alice and I love Python.")
    print(f"🤖 Agent: {response['messages'][-1].content}\n")
    
    # Thread 2: Bob
    print("--- Thread 2: user_bob ---")
    response = agent.invoke(
        "Hi! My name is Bob and I love Java.",
        config={"configurable": {"thread_id": "user_bob"}}
    )
    print(f"👤 Bob: Hi! My name is Bob and I love Java.")
    print(f"🤖 Agent: {response['messages'][-1].content}\n")
    
    # Test Thread 1 memory
    print("--- Back to Thread 1: user_alice ---")
    response = agent.invoke(
        "What's my name and what do I love?",
        config={"configurable": {"thread_id": "user_alice"}}
    )
    print(f"👤 Alice: What's my name and what do I love?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Check if it mentions Alice and Python
    content = response['messages'][-1].content.lower()
    if "alice" in content and "python" in content:
        print("✅ Thread 1 memory works!\n")
    else:
        print("❌ Thread 1 memory failed!\n")
    
    # Test Thread 2 memory
    print("--- Back to Thread 2: user_bob ---")
    response = agent.invoke(
        "What's my name and what do I love?",
        config={"configurable": {"thread_id": "user_bob"}}
    )
    print(f"👤 Bob: What's my name and what do I love?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Check if it mentions Bob and Java
    content = response['messages'][-1].content.lower()
    if "bob" in content and "java" in content:
        print("✅ Thread 2 memory works!\n")
    else:
        print("❌ Thread 2 memory failed!\n")
    
    # Test new thread (should not know anyone)
    print("--- New Thread: new_user ---")
    response = agent.invoke(
        "Do you know my name?",
        config={"configurable": {"thread_id": "new_user"}}
    )
    print(f"👤 New User: Do you know my name?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    content = response['messages'][-1].content.lower()
    if "alice" not in content and "bob" not in content:
        print("✅ Thread isolation works!\n")
    else:
        print("❌ Thread isolation failed!\n")
    
    print("="*60)
    print("📊 Summary:")
    print("- Each thread should have its own conversation history")
    print("- New threads should not know previous users")
    print("="*60)


if __name__ == "__main__":
    test_thread_isolation()