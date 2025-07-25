#!/usr/bin/env python3
"""
Redis Memory Demo - Tests all Redis features with CoreAgent
Using updated LangGraph 0.5.4
"""

import os
import sys
import time
import redis

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_password@localhost:6379")


def check_redis():
    """Check Redis connection"""
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
        
        # Clear any existing data for clean test
        print("🧹 Clearing Redis data...")
        r.flushdb()
        print("✅ Redis cleared for fresh start")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis error: {e}")
        print("\n🔧 Solutions:")
        print("   1. Start Redis: docker-compose up redis")
        print("   2. Check password in docker-compose.yml")
        return False


@tool
def save_note(title: str, content: str) -> str:
    """Save a note to memory"""
    return f"Note '{title}' saved successfully"


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def run_redis_demo():
    """Main demo function"""
    print("\n" + "="*60)
    print("🚀 REDIS MEMORY DEMO - LangGraph 0.5.4")
    print("="*60)
    
    # Check Redis
    if not check_redis():
        return
    
    # Initialize model
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-12-01-preview",
        azure_deployment="gpt4o",
        temperature=0.7
    )
    
    # Create agent with Redis memory
    config = AgentConfig(
        name="RedisAgent",
        model=model,
        tools=[save_note, calculate],
        
        # Memory configuration
        enable_memory=True,
        memory_backend="redis",
        redis_url=REDIS_URL,
        memory_types=["short_term", "long_term"],
        
        # Memory features
        enable_memory_tools=True,
        
        # System prompt
        system_prompt="You are a helpful assistant with Redis memory capabilities."
    )
    
    print("\n📋 Configuration:")
    print(f"   Backend: {config.memory_backend}")
    print(f"   Memory Types: {config.memory_types}")
    print(f"   Tools: {[t.name for t in config.tools]}")
    
    # Create agent
    try:
        agent = CoreAgent(config)
        print("\n✅ Agent created successfully!")
        print("   Note: New LangGraph versions might handle indexes automatically")
    except Exception as e:
        print(f"\n❌ Failed to create agent: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check if Redis Stack is running (not just Redis)")
        print("   2. Try: docker run -d -p 6379:6379 redis/redis-stack:latest")
        print("   3. Or use InMemory backend for testing")
        return
    
    # Test 1: Short-term memory (conversation)
    print("\n" + "-"*60)
    print("TEST 1: SHORT-TERM MEMORY (Conversation)")
    print("-"*60)
    
    thread_id = "user_123"
    
    try:
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
        
    except Exception as e:
        print(f"❌ Short-term memory error: {e}")
        print("   This might be due to missing Redis indexes")
    
    # Test 2: Long-term memory (persistent storage)
    print("\n" + "-"*60)
    print("TEST 2: LONG-TERM MEMORY (Persistent Storage)")
    print("-"*60)
    
    try:
        mm = agent.memory_manager
        
        # Store data
        print("\n📝 Storing user preferences...")
        mm.store_long_term_memory("alice_preferences", {
            "favorite_color": "blue",
            "favorite_language": "Python",
            "interests": ["machine learning", "data visualization"]
        })
        print("✅ Preferences stored")
        
        # Retrieve data
        prefs = mm.get_long_term_memory("alice_preferences")
        print(f"\n📖 Retrieved: {prefs}")
        
    except Exception as e:
        print(f"❌ Long-term memory error: {e}")
    
    # Test 3: Memory tools
    print("\n" + "-"*60)
    print("TEST 3: MEMORY TOOLS")
    print("-"*60)
    
    try:
        response = agent.invoke(
            "Please save a note that I have a meeting tomorrow at 2 PM with the marketing team",
            config={"configurable": {"thread_id": thread_id}}
        )
        print("\n👤 User: Please save a note about tomorrow's meeting")
        print(f"🤖 Agent: {response['messages'][-1].content}")
        
    except Exception as e:
        print(f"❌ Memory tools error: {e}")
    
    # Test 4: Thread isolation
    print("\n" + "-"*60)
    print("TEST 4: THREAD ISOLATION")
    print("-"*60)
    
    try:
        response = agent.invoke(
            "What's my name?",
            config={"configurable": {"thread_id": "different_user"}}
        )
        print("\n👤 User (different thread): What's my name?")
        print(f"🤖 Agent: {response['messages'][-1].content}")
        
    except Exception as e:
        print(f"❌ Thread isolation error: {e}")
    
    print("\n" + "="*60)
    print("📊 DEMO SUMMARY")
    print("="*60)
    
    print("\n✅ Tests attempted:")
    print("1. Short-term memory - Conversation history per thread")
    print("2. Long-term memory - Persistent key-value storage")
    print("3. Memory tools - Agent can save/retrieve data")
    print("4. Thread isolation - Different conversations isolated")
    
    print("\n💡 Notes:")
    print("- LangGraph 0.5.4 should handle Redis better")
    print("- If errors occur, indexes might still be needed")
    print("- Consider using PostgreSQL or InMemory for stability")


if __name__ == "__main__":
    run_redis_demo()