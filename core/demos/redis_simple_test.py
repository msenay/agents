#!/usr/bin/env python3
"""
Simple Redis test without memory tools
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


def test_simple_redis():
    print("\n" + "="*60)
    print("🚀 SIMPLE REDIS TEST")
    print("="*60)
    
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-12-01-preview",
        azure_deployment="gpt4o",
        temperature=0.7
    )
    
    # Simple config - no memory tools
    config = AgentConfig(
        name="SimpleRedisAgent",
        model=model,
        enable_memory=True,
        memory_backend="redis",
        redis_url=REDIS_URL,
        memory_types=["short_term", "long_term"],
        enable_memory_tools=False,  # Disable memory tools
        system_prompt="You are a helpful assistant."
    )
    
    try:
        agent = CoreAgent(config)
        print("✅ Agent created")
        
        # Test 1: Conversation
        print("\n--- Test 1: Conversation ---")
        response = agent.invoke(
            "Hi, my name is Bob",
            config={"configurable": {"thread_id": "test1"}}
        )
        print(f"User: Hi, my name is Bob")
        print(f"Agent: {response['messages'][-1].content}")
        
        response = agent.invoke(
            "What's my name?",
            config={"configurable": {"thread_id": "test1"}}
        )
        print(f"\nUser: What's my name?")
        print(f"Agent: {response['messages'][-1].content}")
        
        # Test 2: Direct memory access
        print("\n--- Test 2: Direct Memory ---")
        mm = agent.memory_manager
        
        # Store
        mm.store_long_term_memory("test_key", {"data": "test_value"})
        print("✅ Stored data")
        
        # Retrieve
        data = mm.get_long_term_memory("test_key")
        print(f"✅ Retrieved: {data}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_redis()