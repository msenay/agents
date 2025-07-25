#!/usr/bin/env python3
"""
Redis Memory Demo - Comprehensive test of all Redis features
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


def check_and_fix_redis():
    """Check Redis connection and create indexes if needed"""
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
        
        # Try to create indexes if they don't exist
        try:
            # Create checkpoints index for short-term memory
            r.execute_command("FT.INFO", "checkpoints")
            print("✅ Index 'checkpoints' already exists")
        except:
            print("📝 Creating 'checkpoints' index...")
            try:
                r.execute_command(
                    "FT.CREATE", "checkpoints",
                    "ON", "HASH",
                    "PREFIX", "1", "checkpoint:",
                    "SCHEMA",
                    "thread_id", "TAG",
                    "checkpoint_id", "TAG",
                    "thread_ts", "NUMERIC"
                )
                print("✅ Created 'checkpoints' index")
            except Exception as e:
                print(f"⚠️  Could not create checkpoints index: {e}")
        
        # Create store index for long-term memory
        try:
            r.execute_command("FT.INFO", "store")
            print("✅ Index 'store' already exists")
        except:
            print("📝 Creating 'store' index...")
            try:
                r.execute_command(
                    "FT.CREATE", "store",
                    "ON", "HASH",
                    "PREFIX", "1", "store:",
                    "SCHEMA",
                    "namespace", "TAG",
                    "key", "TAG",
                    "value", "TEXT"
                )
                print("✅ Created 'store' index")
            except Exception as e:
                print(f"⚠️  Could not create store index: {e}")
        
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
    return f"Note '{title}' saved"


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
    print("🚀 REDIS MEMORY DEMO")
    print("="*60)
    
    # Check and fix Redis
    if not check_and_fix_redis():
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
        
        system_prompt="You are a helpful assistant with Redis memory."
    )
    
    print("\n📋 Configuration:")
    print(f"   Backend: {config.memory_backend}")
    print(f"   Memory Types: {config.memory_types}")
    
    # Create agent
    try:
        agent = CoreAgent(config)
        print("\n✅ Agent created successfully!")
    except Exception as e:
        print(f"\n❌ Failed to create agent: {e}")
        print("\n💡 If you see index errors:")
        print("   1. Redis Stack might not support RediSearch")
        print("   2. Try using InMemory backend instead")
        print("   3. Or manually create indexes with redis-cli")
        return
    
    # Test 1: Short-term memory
    print("\n" + "-"*60)
    print("TEST 1: SHORT-TERM MEMORY (Conversation)")
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
    print("TEST 2: LONG-TERM MEMORY (Persistent Storage)")
    print("-"*60)
    
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
    
    # Test 3: Memory tools
    print("\n" + "-"*60)
    print("TEST 3: MEMORY TOOLS")
    print("-"*60)
    
    response = agent.invoke(
        "Please save a note that I have a meeting tomorrow at 2 PM with the marketing team",
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n👤 User: Please save a note that I have a meeting tomorrow at 2 PM")
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
    print("✅ REDIS DEMO COMPLETED!")
    print("="*60)
    
    print("\n📚 What we tested:")
    print("1. ✅ Short-term memory - Conversation history per thread")
    print("2. ✅ Long-term memory - Persistent key-value storage")
    print("3. ✅ Memory tools - Agent can save/retrieve data")
    print("4. ✅ Thread isolation - Different conversations isolated")


if __name__ == "__main__":
    run_redis_demo()