#!/usr/bin/env python3
"""
Debug thread_id handling in Redis memory
"""

import os
import sys
import redis

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI

REDIS_URL = "redis://:redis_password@localhost:6379"


def debug_thread_handling():
    """Debug how thread_id is handled"""
    
    print("\n" + "="*60)
    print("🐛 THREAD_ID DEBUG TEST")
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
        name="DebugAgent",
        model=model,
        enable_memory=True,
        memory_backend="redis",
        redis_url=REDIS_URL,
        memory_types=["short_term"],
        system_prompt="You are a debug assistant. Always state the exact information given in the conversation."
    )
    
    agent = CoreAgent(config)
    print("✅ Agent created\n")
    
    # Clear Redis to start fresh
    r = redis.from_url(REDIS_URL)
    print("🧹 Clearing all Redis data...")
    r.flushdb()
    print("✅ Redis cleared\n")
    
    # Test 1: Store in thread1
    print("--- Test 1: Store in thread1 ---")
    config1 = {"configurable": {"thread_id": "thread1"}}
    response = agent.invoke("My secret code is ABC123", config=config1)
    print(f"Config: {config1}")
    print(f"👤 User: My secret code is ABC123")
    print(f"🤖 Agent: {response['messages'][-1].content}\n")
    
    # Test 2: Store in thread2
    print("--- Test 2: Store in thread2 ---")
    config2 = {"configurable": {"thread_id": "thread2"}}
    response = agent.invoke("My secret code is XYZ789", config=config2)
    print(f"Config: {config2}")
    print(f"👤 User: My secret code is XYZ789")
    print(f"🤖 Agent: {response['messages'][-1].content}\n")
    
    # Test 3: Check thread1 memory
    print("--- Test 3: Retrieve from thread1 ---")
    response = agent.invoke("What is my secret code?", config=config1)
    print(f"Config: {config1}")
    print(f"👤 User: What is my secret code?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    if "ABC123" in response['messages'][-1].content:
        print("✅ Correct! Thread1 remembers ABC123\n")
    else:
        print("❌ Wrong! Thread1 should remember ABC123\n")
    
    # Test 4: Check thread2 memory
    print("--- Test 4: Retrieve from thread2 ---")
    response = agent.invoke("What is my secret code?", config=config2)
    print(f"Config: {config2}")
    print(f"👤 User: What is my secret code?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    if "XYZ789" in response['messages'][-1].content:
        print("✅ Correct! Thread2 remembers XYZ789\n")
    else:
        print("❌ Wrong! Thread2 should remember XYZ789\n")
    
    # Check Redis keys
    print("--- Redis Key Analysis ---")
    keys = r.keys("*")
    print(f"Total keys in Redis: {len(keys)}")
    
    checkpoint_keys = [k for k in keys if b'checkpoint:' in k]
    print(f"Checkpoint keys: {len(checkpoint_keys)}")
    
    for key in checkpoint_keys[:5]:  # Show first 5
        key_str = key.decode()
        print(f"  - {key_str}")
        if "thread1" in key_str or "thread2" in key_str:
            print("    ✅ Contains thread_id")
        else:
            print("    ❌ No thread_id found!")
    
    print("\n" + "="*60)
    print("💡 If threads are mixing, check:")
    print("1. Is thread_id being passed to checkpointer?")
    print("2. Are Redis keys using thread_id?")
    print("3. Is config propagating correctly?")
    print("="*60)


if __name__ == "__main__":
    debug_thread_handling()