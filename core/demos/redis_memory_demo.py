#!/usr/bin/env python3
"""
Comprehensive Redis Memory Demo for Core Agent
Tests all memory type combinations
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set Azure OpenAI credentials
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from core import CoreAgent, AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
import redis

# Redis connection
REDIS_URL = os.environ.get("REDIS_URL", "redis://:redis_password@localhost:6379")


def check_redis():
    """Check Redis connection"""
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("\n💡 Make sure Redis is running:")
        print("   docker-compose up -d redis")
        return False


def clear_redis_data():
    """Clear all Redis data for fresh start"""
    try:
        r = redis.from_url(REDIS_URL)
        r.flushdb()
        print("🧹 Redis data cleared")
    except:
        pass


def test_memory_combination(memory_types, test_name):
    """Test a specific memory combination"""
    print("\n" + "="*80)
    print(f"🧪 TESTING: {test_name}")
    print(f"   Memory Types: {memory_types}")
    print("="*80)
    
    # Create model
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-12-01-preview",
        azure_deployment="gpt4o",
        temperature=0.7
    )
    
    # Configure based on memory types
    config_params = {
        "name": f"RedisAgent_{test_name}",
        "model": model,
        "enable_memory": True,
        "memory_backend": "redis",
        "redis_url": REDIS_URL,
        "memory_types": memory_types,
        "enable_memory_tools": "long_term" in memory_types,
        "system_prompt": "You are a helpful AI assistant with memory capabilities."
    }
    
    # Add embedding config if needed
    if "semantic" in memory_types:
        config_params.update({
            "embedding_model": "openai:text-embedding-3-small",
            "embedding_dims": 1536
        })
    
    try:
        # Create agent
        config = AgentConfig(**config_params)
        agent = CoreAgent(config)
        print("✅ Agent created successfully")
        
        # Test based on memory types
        results = {}
        
        # Test 1: Short-term memory (if enabled)
        if "short_term" in memory_types:
            print("\n--- Testing SHORT-TERM Memory ---")
            
            # First message
            response = agent.invoke(
                "Hi! My name is TestUser and I love Python programming.",
                config={"configurable": {"thread_id": f"test_{test_name}"}}
            )
            print("👤 User: Hi! My name is TestUser and I love Python programming.")
            print(f"🤖 Agent: {response['messages'][-1].content}")
            
            # Test memory recall
            response = agent.invoke(
                "What do you remember about me?",
                config={"configurable": {"thread_id": f"test_{test_name}"}}
            )
            print("\n👤 User: What do you remember about me?")
            print(f"🤖 Agent: {response['messages'][-1].content}")
            
            # Check if it remembers
            content = response['messages'][-1].content.lower()
            if "testuser" in content or "python" in content:
                results["short_term"] = "✅ Working - Remembers conversation"
            else:
                results["short_term"] = "❌ Failed - Doesn't remember"
        
        # Test 2: Long-term memory (if enabled)
        if "long_term" in memory_types:
            print("\n--- Testing LONG-TERM Memory ---")
            
            # Store data directly
            if hasattr(agent.memory_manager, 'store'):
                agent.memory_manager.store.put(
                    namespace="user_data",
                    key="preferences",
                    value={"theme": "dark", "language": "Python", "level": "expert"}
                )
                print("📝 Stored user preferences")
                
                # Retrieve data
                data = agent.memory_manager.store.get(
                    namespace="user_data",
                    key="preferences"
                )
                print(f"📖 Retrieved: {data.value if data else 'None'}")
                
                if data and data.value.get("language") == "Python":
                    results["long_term"] = "✅ Working - Can store/retrieve data"
                else:
                    results["long_term"] = "❌ Failed - Storage issue"
            else:
                results["long_term"] = "⚠️  No store available"
        
        # Test 3: Semantic/Embedding memory (if enabled)
        if "semantic" in memory_types:
            print("\n--- Testing SEMANTIC Memory ---")
            
            # Store some documents
            if hasattr(agent.memory_manager, 'store'):
                try:
                    # Store related documents
                    docs = [
                        ("doc1", "Python is a great programming language for data science and machine learning."),
                        ("doc2", "JavaScript is popular for web development and frontend frameworks."),
                        ("doc3", "Machine learning models can be trained using Python libraries like TensorFlow.")
                    ]
                    
                    for doc_id, content in docs:
                        agent.memory_manager.store.put(
                            namespace="knowledge_base",
                            key=doc_id,
                            value={"content": content}
                        )
                    print("📚 Stored 3 documents for semantic search")
                    
                    # Search semantically
                    search_results = agent.memory_manager.store.search(
                        namespace="knowledge_base",
                        query="What programming language is good for AI?",
                        limit=2
                    )
                    
                    if search_results:
                        print(f"🔍 Found {len(search_results)} relevant documents")
                        results["semantic"] = "✅ Working - Semantic search successful"
                    else:
                        results["semantic"] = "❌ Failed - No search results"
                        
                except Exception as e:
                    results["semantic"] = f"❌ Error - {str(e)[:50]}"
            else:
                results["semantic"] = "⚠️  No store available"
        
        # Test 4: Combined functionality (if multiple types)
        if len(memory_types) > 1:
            print("\n--- Testing COMBINED Memory ---")
            
            # Use conversation with memory tools
            if "short_term" in memory_types and "long_term" in memory_types:
                response = agent.invoke(
                    "Please remember that my favorite color is blue",
                    config={"configurable": {"thread_id": f"test_{test_name}_combined"}}
                )
                print("👤 User: Please remember that my favorite color is blue")
                print(f"🤖 Agent: {response['messages'][-1].content}")
                
                results["combined"] = "✅ Multiple memory types active"
        
        # Summary
        print("\n" + "-"*80)
        print("📊 Results:")
        for key, value in results.items():
            print(f"   {key}: {value}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


def run_all_tests():
    """Run all memory combination tests"""
    print("\n" + "="*80)
    print("🚀 REDIS MEMORY DEMO - ALL COMBINATIONS")
    print("="*80)
    
    if not check_redis():
        return
    
    # Test combinations
    test_configs = [
        (["short_term"], "Short-term Only"),
        (["long_term"], "Long-term Only"),
        (["semantic"], "Semantic Only"),
        (["short_term", "long_term"], "Short + Long"),
        (["short_term", "long_term", "semantic"], "Short + Long + Semantic"),
    ]
    
    all_results = {}
    
    for memory_types, test_name in test_configs:
        # Clear Redis between tests for isolation
        clear_redis_data()
        time.sleep(1)  # Give Redis time to clear
        
        results = test_memory_combination(memory_types, test_name)
        all_results[test_name] = results
        
        # Small delay between tests
        time.sleep(2)
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY - ALL TESTS")
    print("="*80)
    
    for test_name, results in all_results.items():
        print(f"\n{test_name}:")
        if isinstance(results, dict) and "error" not in results:
            for key, value in results.items():
                print(f"  - {key}: {value}")
        else:
            print(f"  - Status: {results}")
    
    print("\n" + "="*80)
    print("✨ Demo completed!")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()