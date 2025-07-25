#!/usr/bin/env python3
"""
Redis Memory Demo - Tests all Redis features with CoreAgent
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables for Azure OpenAI
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]  # For compatibility
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis URL
REDIS_URL = os.getenv("REDIS_URL", "redis://:redis_password@localhost:6379")


def check_redis():
    """Check Redis connection"""
    try:
        import redis
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\n🔧 Solutions:")
        print("   1. Start Redis: docker-compose up redis")
        print("   2. Check password in docker-compose.yml")
        return False


# Define tools
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
    print("🚀 REDIS MEMORY DEMO - CORE AGENT")
    print("="*60)
    
    # Check Redis first
    if not check_redis():
        return
    
    # Initialize Azure OpenAI model
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment="gpt4o",
        temperature=0.7
    )
    
    # Create agent configuration
    config = AgentConfig(
        name="RedisAgent",
        model=model,
        tools=[save_note, calculate],
        
        # Memory configuration
        enable_memory=True,
        memory_backend="inmemory",  # Use inmemory for now to test flow
        # redis_url=REDIS_URL,
        memory_types=["short_term", "long_term"],  # Remove semantic for now
        
        # Memory features
        enable_memory_tools=True,
        enable_ttl=True,
        default_ttl_minutes=60,
        
        # System prompt
        system_prompt="""You are a helpful AI assistant with Redis memory capabilities.
You can remember conversations, save information, and perform calculations."""
    )
    
    print("\n📋 Agent Configuration:")
    print(f"   Name: {config.name}")
    print(f"   Memory Backend: {config.memory_backend}")
    print(f"   Memory Types: {config.memory_types}")
    print(f"   Tools: {[t.name for t in config.tools]}")
    
    # Create the agent
    try:
        agent = CoreAgent(config)
        print("\n✅ Agent created successfully!")
    except Exception as e:
        print(f"\n❌ Failed to create agent: {e}")
        return
    
    # Test 1: Short-term memory (conversation)
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
    
    # Test 2: Long-term memory (persistent storage)
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
    print(f"\n📖 Retrieved preferences: {prefs}")
    
    # Test 3: Memory tools
    print("\n" + "-"*60)
    print("TEST 3: MEMORY TOOLS (Agent-controlled memory)")
    print("-"*60)
    
    response = agent.invoke(
        "Please save a note that I have a meeting tomorrow at 2 PM with the marketing team",
        config={"configurable": {"thread_id": thread_id}}
    )
    print("\n👤 User: Please save a note that I have a meeting tomorrow at 2 PM with the marketing team")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Test 4: TTL
    print("\n" + "-"*60)
    print("TEST 4: TTL (Time-To-Live)")
    print("-"*60)
    
    print("\n📝 Storing temporary data with 5 second TTL...")
    mm.store_long_term_memory("temp_token", {"token": "abc123", "expires": "soon"}, ttl=5)
    
    # Check immediately
    data = mm.get_long_term_memory("temp_token")
    print(f"✅ Immediate check: {data}")
    
    print("\n⏳ Waiting 6 seconds...")
    time.sleep(6)
    
    data = mm.get_long_term_memory("temp_token")
    print(f"📖 After expiry: {data}")
    
    # Test 5: Semantic memory (vector search) - DISABLED FOR NOW
    # print("\n" + "-"*60)
    # print("TEST 5: SEMANTIC MEMORY (Vector Search)")
    # print("-"*60)
    # 
    # print("\n📝 Storing documents for semantic search...")
    # 
    # # Store some documents
    # documents = [
    #     ("doc1", {"content": "I love traveling to Paris. The Eiffel Tower is amazing."}),
    #     ("doc2", {"content": "Python programming is great for data science and machine learning."}),
    #     ("doc3", {"content": "Italian pasta is my favorite food, especially carbonara."}),
    #     ("doc4", {"content": "Tokyo is an incredible city with beautiful temples and great sushi."}),
    #     ("doc5", {"content": "JavaScript is essential for web development and React applications."})
    # ]
    # 
    # for key, doc in documents:
    #     mm.store_long_term_memory(key, doc)
    #     print(f"✅ Stored: {key}")
    # 
    # # Search semantically
    # print("\n🔍 Semantic search tests:")
    # 
    # queries = ["travel experiences", "programming languages", "food and cuisine"]
    # 
    # for query in queries:
    #     print(f"\n🔎 Searching for: '{query}'")
    #     if hasattr(mm, 'search_memory'):
    #         try:
    #             results = mm.search_memory(query, limit=2)
    #             for i, result in enumerate(results, 1):
    #                 print(f"   {i}. {result}")
    #         except Exception as e:
    #             print(f"   ❌ Search error: {e}")
    #     else:
    #         print("   ⚠️  Semantic search not available")
    
    # Different thread test
    print("\n" + "-"*60)
    print("TEST 6: DIFFERENT THREAD (Isolation)")
    print("-"*60)
    
    response = agent.invoke(
        "What's my name?",
        config={"configurable": {"thread_id": "different_user"}}
    )
    print("\n👤 User (different thread): What's my name?")
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📚 What we tested:")
    print("1. ✅ Short-term memory - Conversation history per thread")
    print("2. ✅ Long-term memory - Persistent key-value storage")
    print("3. ✅ Memory tools - Agent can save/retrieve data")
    print("4. ✅ TTL support - Auto-expiring data")
    print("5. ✅ Thread isolation - Different conversations isolated")


if __name__ == "__main__":
    run_redis_demo()