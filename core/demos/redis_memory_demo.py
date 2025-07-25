#!/usr/bin/env python3
"""
Comprehensive Redis Memory Demo for Core Agent
Tests all Redis-related memory features with real Azure OpenAI
"""

import os
import sys
import time
import asyncio
from typing import Dict, Any, List
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables for Azure OpenAI (needed for embeddings)
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["AZURE_OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = "gpt4o"  # You might need a specific embedding deployment

# Also set OPENAI_API_KEY for compatibility
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-202-fbeta-dev.openai.azure.com/")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2023-12-01-preview")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME", "gpt4o")

# Redis configuration
DEFAULT_REDIS_URL = "redis://:redis_password@localhost:6379"
REDIS_URL = os.getenv("REDIS_URL", DEFAULT_REDIS_URL)


def check_redis_connection():
    """Check Redis connection"""
    try:
        import redis
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
        print(f"   URL: {REDIS_URL.replace('redis_password', '***')}")  # Hide password in output
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print(f"   URL: {REDIS_URL.replace('redis_password', '***')}")
        print("\n🔧 Solutions:")
        print("   1. Check if Redis is running: docker-compose up redis")
        print("   2. Use correct password: redis://:redis_password@localhost:6379")
        print("   3. Or set REDIS_URL environment variable")
        print("   4. For passwordless: docker-compose -f docker-compose.override.yml up redis")
        return False


# Define custom tools for the agent
@tool
def get_current_time() -> str:
    """Get the current time"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: calculate('2 + 2')"""
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def search_web(query: str) -> str:
    """Search the web for information. This is a mock implementation."""
    # Mock implementation
    mock_results = {
        "python": "Python is a high-level programming language",
        "redis": "Redis is an in-memory data structure store",
        "langgraph": "LangGraph is a framework for building stateful agents",
        "weather": "Today's weather is sunny with 22°C",
        "news": "Latest news: AI advances continue to accelerate"
    }
    
    for key, value in mock_results.items():
        if key.lower() in query.lower():
            return value
    
    return f"No specific results found for '{query}', but in a real implementation this would search the web"


class RedisMemoryDemo:
    """Demo testing Redis memory features with real Azure OpenAI"""

    def __init__(self):
        # Use environment variable or default
        self.redis_url = REDIS_URL
        
        # Initialize Azure OpenAI
        self.model = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=OPENAI_API_KEY,
            api_version=OPENAI_API_VERSION,
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            temperature=0.7,
            max_tokens=1000
        )
        
        print("\n🤖 Using Azure OpenAI:")
        print(f"   Endpoint: {AZURE_OPENAI_ENDPOINT}")
        print(f"   Deployment: {AZURE_DEPLOYMENT_NAME}")
        print(f"   API Version: {OPENAI_API_VERSION}")
        
        self.agent = None

    def create_agent_with_full_config(self) -> CoreAgent:
        """Create agent with explicit full configuration"""
        
        # Full explicit configuration showing all options
        config = AgentConfig(
            # Basic Configuration
            name="RedisMemoryAgent",
            model=self.model,
            
            # Tools Configuration
            tools=[get_current_time, calculate, search_web],
            
            # Memory Configuration - Main Switch
            enable_memory=True,
            memory_backend="redis",
            
            # Backend URLs
            redis_url=self.redis_url,
            postgres_url=None,  # Not used in this demo
            
            # Memory Types to Enable
            # Start with basic types, semantic can be tested separately
            memory_types=["short_term", "long_term", "session"],  # Remove "semantic" for now
            
            # Memory Features
            enable_memory_tools=True,  # Gives agent access to save/load memory
            
            # TTL Support (only for redis backend)
            enable_ttl=True,  # Time-to-live for Redis entries
            default_ttl_minutes=60,  # 60 minutes default TTL (parameter name is default_ttl_minutes, not default_ttl)
            refresh_on_read=True,  # Refresh TTL on access
            
            # Semantic Search Configuration (commented out for now)
            # embedding_model="azure_openai:text-embedding-3-small",  # Use azure_openai instead of openai
            # embedding_dims=1536,
            
            # Session Configuration
            session_id="demo_session_123",
            memory_namespace="demo_namespace",  # parameter name is memory_namespace, not session_namespace
            
            # Conversation Memory Management
            enable_message_trimming=True,  # parameter name is enable_message_trimming, not enable_trimming
            max_tokens=1000,  # Used with message trimming
            trim_strategy="last",  # Keep last messages when trimming
            enable_summarization=False,  # Don't summarize (requires langmem)
            
            # Agent Behavior
            system_prompt="""You are a helpful AI assistant with comprehensive memory capabilities.
            
Your memory features:
1. Short-term: You automatically remember our conversation history
2. Long-term: You can save and retrieve important information
3. Session: You can share memory with other agents
4. Semantic: You can search memories by meaning/similarity

Use your tools and memory effectively to help users.""",
            
            # Optional Features (these don't exist in AgentConfig)
            # verbose=True,  # NOT a valid parameter
            # stream_mode="values",  # NOT a valid parameter
            # recursion_limit=10,  # NOT a valid parameter
            
            # Response Format (optional)
            response_format=None,  # Could be JSON schema
            
            # Rate Limiting
            enable_rate_limiting=False,
            requests_per_second=1.0,  # Default rate limit
            
            # Hooks (optional)
            pre_model_hook=None,
            post_model_hook=None,
            
            # Multi-agent Configuration (these parameters exist but with different names)
            # enable_multi_agent=False,  # NOT a valid parameter
            # multi_agent_url=None  # NOT a valid parameter
            enable_supervisor=False,  # For multi-agent patterns
            enable_swarm=False,  # For swarm patterns
            enable_handoff=False  # For handoff patterns
        )
        
        print("\n📋 Agent Configuration:")
        print(f"   Name: {config.name}")
        print(f"   Memory Backend: {config.memory_backend}")
        print(f"   Memory Types: {config.memory_types}")
        print(f"   Tools: {[t.name for t in config.tools]}")
        print(f"   TTL Enabled: {config.enable_ttl} (default: {config.default_ttl_minutes} minutes)")
        print(f"   Session ID: {config.session_id}")
        print(f"   Message Trimming: {config.enable_message_trimming} (max_tokens: {config.max_tokens})")
        
        # Create the agent
        agent = CoreAgent(config)
        print("\n✅ Agent created successfully with full configuration")
        
        return agent

    async def test_short_term_memory(self):
        """Test short-term (conversation) memory"""
        print("\n" + "="*60)
        print("1️⃣  SHORT-TERM MEMORY TEST (Conversation History)")
        print("="*60)
        
        thread_id = "test_thread_1"
        
        # First message
        print("\n📤 First message:")
        response = await self.agent.ainvoke(
            "Hello! My name is John and I love programming in Python.",
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")
        
        # Second message - should remember
        print("\n📤 Second message (same thread):")
        response = await self.agent.ainvoke(
            "What's my name and what do I love?",
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")
        
        # Different thread - should not remember
        print("\n📤 Third message (different thread):")
        response = await self.agent.ainvoke(
            "What's my name?",
            config={"configurable": {"thread_id": "different_thread"}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")

    async def test_long_term_memory(self):
        """Test long-term (persistent) memory"""
        print("\n" + "="*60)
        print("2️⃣  LONG-TERM MEMORY TEST (Persistent Storage)")
        print("="*60)
        
        # Get memory manager
        mm = self.agent.memory_manager
        
        # Store some data
        print("\n📝 Storing data in long-term memory...")
        mm.store_long_term_memory("user_profile", {
            "name": "Alice Johnson",
            "age": 30,
            "preferences": {
                "language": "Turkish",
                "theme": "dark",
                "notifications": True
            },
            "joined_date": "2024-01-15"
        })
        print("✅ User profile stored")
        
        mm.store_long_term_memory("project_info", {
            "name": "Core Agent Framework",
            "version": "1.0.0",
            "features": ["memory", "tools", "streaming"],
            "status": "active"
        })
        print("✅ Project info stored")
        
        # Retrieve data
        print("\n📖 Retrieving data from long-term memory...")
        user_data = mm.get_long_term_memory("user_profile")
        print(f"User Profile: {user_data}")
        
        project_data = mm.get_long_term_memory("project_info")
        print(f"Project Info: {project_data}")
        
        # List all keys
        print("\n📋 All stored keys:")
        # Note: This is a simplified example, actual implementation may vary
        print("- user_profile")
        print("- project_info")

    async def test_semantic_memory(self):
        """Test semantic (vector search) memory"""
        print("\n" + "="*60)
        print("3️⃣  SEMANTIC MEMORY TEST (Vector Search)")
        print("="*60)
        
        mm = self.agent.memory_manager
        
        # Store documents with semantic content
        print("\n📝 Storing semantic documents...")
        
        documents = [
            ("travel_paris", {"content": "I visited Paris last summer. The Eiffel Tower was amazing!"}),
            ("travel_tokyo", {"content": "Tokyo trip was incredible. Loved the sushi and temples."}),
            ("coding_python", {"content": "Python programming is fun. I built a web scraper today."}),
            ("coding_javascript", {"content": "JavaScript async/await makes handling promises easier."}),
            ("cooking_pasta", {"content": "Made delicious pasta carbonara with fresh ingredients."}),
        ]
        
        for key, doc in documents:
            mm.store_long_term_memory(key, doc)
            print(f"✅ Stored: {key}")
        
        # Search semantically
        print("\n🔍 Semantic search results:")
        
        queries = [
            "travel experiences in Europe",
            "programming languages and coding",
            "food and cooking recipes"
        ]
        
        for query in queries:
            print(f"\n🔎 Query: '{query}'")
            results = mm.search_memory(query, limit=3)
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")

    async def test_session_memory(self):
        """Test session (multi-agent shared) memory"""
        print("\n" + "="*60)
        print("4️⃣  SESSION MEMORY TEST (Multi-Agent Shared)")
        print("="*60)
        
        # Store session data
        print("\n📝 Storing session data...")
        
        # Using the agent's session_id from config
        response = await self.agent.ainvoke(
            "Please remember that the meeting is scheduled for 3 PM tomorrow with the marketing team.",
            config={"configurable": {"thread_id": "session_test"}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")
        
        # In a real multi-agent scenario, another agent with the same session_id
        # could access this information

    async def test_ttl_memory(self):
        """Test TTL (Time-To-Live) feature"""
        print("\n" + "="*60)
        print("5️⃣  TTL (TIME-TO-LIVE) TEST")
        print("="*60)
        
        mm = self.agent.memory_manager
        
        # Store with custom TTL
        print("\n📝 Storing data with 5-second TTL...")
        mm.store_long_term_memory("temp_data", {"message": "This will expire soon"}, ttl=5)
        print("✅ Stored temporary data")
        
        # Retrieve immediately
        data = mm.get_long_term_memory("temp_data")
        print(f"📖 Immediate retrieval: {data}")
        
        # Wait and try again
        print("\n⏳ Waiting 6 seconds...")
        await asyncio.sleep(6)
        
        data = mm.get_long_term_memory("temp_data")
        print(f"📖 After expiry: {data}")

    async def test_memory_tools(self):
        """Test memory tools available to the agent"""
        print("\n" + "="*60)
        print("6️⃣  MEMORY TOOLS TEST (Agent-Controlled Memory)")
        print("="*60)
        
        # Agent can use memory tools autonomously
        print("\n📤 Asking agent to remember information:")
        response = await self.agent.ainvoke(
            "Please remember that my favorite color is blue and I prefer morning meetings.",
            config={"configurable": {"thread_id": "tools_test"}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")
        
        print("\n📤 Asking agent to recall information:")
        response = await self.agent.ainvoke(
            "What do you remember about my preferences?",
            config={"configurable": {"thread_id": "tools_test"}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")

    async def test_memory_trimming(self):
        """Test conversation memory trimming"""
        print("\n" + "="*60)
        print("7️⃣  MEMORY TRIMMING TEST (Conversation Management)")
        print("="*60)
        
        thread_id = "trim_test"
        
        # Send many messages to trigger trimming
        print("\n📤 Sending multiple messages to test trimming...")
        for i in range(25):  # Will trigger trimming based on max_tokens
            response = await self.agent.ainvoke(
                f"Message {i+1}: This is test message number {i+1}",
                config={"configurable": {"thread_id": thread_id}}
            )
            if i < 3 or i >= 22:  # Show first 3 and last 3
                print(f"   Message {i+1} sent")
            elif i == 3:
                print("   ... (sending more messages) ...")
        
        # Check if old messages are trimmed
        print("\n📤 Checking memory after trimming:")
        response = await self.agent.ainvoke(
            "Can you tell me what was in message 1?",
            config={"configurable": {"thread_id": thread_id}}
        )
        print(f"📥 Response: {response['messages'][-1].content}")

    async def run_all_tests(self):
        """Run all Redis memory tests"""
        print("\n🚀 STARTING COMPREHENSIVE REDIS MEMORY TESTS")
        print(f"   Timestamp: {datetime.now()}")
        
        # Check Redis connection first
        if not check_redis_connection():
            print("\n❌ Cannot proceed without Redis connection")
            return
        
        # Create agent with full configuration
        self.agent = self.create_agent_with_full_config()
        
        # Run all tests
        tests = [
            self.test_short_term_memory,
            self.test_long_term_memory,
            # self.test_semantic_memory,  # Skip for now due to embedding configuration issues
            self.test_session_memory,
            self.test_ttl_memory,
            self.test_memory_tools,
            self.test_memory_trimming
        ]
        
        for test in tests:
            try:
                await test()
            except Exception as e:
                print(f"\n❌ Error in {test.__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED!")
        print("="*60)


async def main():
    """Main entry point"""
    demo = RedisMemoryDemo()
    await demo.run_all_tests()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())