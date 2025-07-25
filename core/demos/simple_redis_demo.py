#!/usr/bin/env python3
"""
Simple Redis Memory Demo - Basic functionality test
"""

import os
import sys
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables
os.environ["OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"

from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI

# Redis URL
REDIS_URL = "redis://:redis_password@localhost:6379"


def test_redis_connection():
    """Test basic Redis connection"""
    try:
        import redis
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
        
        # Clear any existing data
        print("🧹 Clearing Redis data...")
        r.flushdb()
        print("✅ Redis cleared")
        
        return True
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("🚀 SIMPLE REDIS DEMO")
    print("="*60)
    
    if not test_redis_connection():
        print("\nPlease start Redis with: docker-compose up redis")
        return
    
    # Use a simple model
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.environ["OPENAI_API_KEY"]
    )
    
    # Simple config - just short-term memory
    print("\n📋 Creating agent with Redis short-term memory...")
    config = AgentConfig(
        name="SimpleRedisAgent",
        model=model,
        enable_memory=True,
        memory_backend="redis",
        redis_url=REDIS_URL,
        memory_types=["short_term"],
        system_prompt="You are a helpful assistant with Redis memory."
    )
    
    try:
        agent = CoreAgent(config)
        print("✅ Agent created!")
    except Exception as e:
        print(f"❌ Failed to create agent: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test conversation
    print("\n" + "-"*60)
    print("TESTING CONVERSATION MEMORY")
    print("-"*60)
    
    thread_id = "test_user"
    
    # Message 1
    print("\n👤 User: My name is Bob and I like pizza.")
    response = agent.invoke(
        "My name is Bob and I like pizza.",
        config={"configurable": {"thread_id": thread_id}}
    )
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Message 2
    print("\n👤 User: What's my name and what do I like?")
    response = agent.invoke(
        "What's my name and what do I like?",
        config={"configurable": {"thread_id": thread_id}}
    )
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Test with different thread
    print("\n👤 User (different thread): What's my name?")
    response = agent.invoke(
        "What's my name?",
        config={"configurable": {"thread_id": "other_user"}}
    )
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()