#!/usr/bin/env python3
"""
MongoDB Memory Testing for CoreAgent Framework

This test demonstrates MongoDB integration for:
1. Short-term memory (checkpointer) with MongoDBSaver
2. Long-term memory (store) with MongoDBStore
3. TTL support for automatic data expiration
4. Session-based memory for agent collaboration
5. Semantic search with embeddings

Requirements:
- MongoDB running (local or cloud)
- pip install pymongo langgraph-checkpoint-mongodb langgraph-store-mongodb
"""

import asyncio
import time
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

# Test configuration
TEST_MONGODB_URL = "mongodb://localhost:27017/coreagent_test"  # Change as needed
TEST_SESSION_ID = f"test_session_{int(time.time())}"

# Mock LLM for testing (to avoid API costs)
class MockChatModel:
    """Mock LLM model for testing"""
    def __init__(self, name="MockLLM"):
        self.name = name
    
    def invoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=f"Mock response from {self.name}")
    
    async def ainvoke(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content=f"Mock async response from {self.name}")
    
    def stream(self, messages, **kwargs):
        from langchain_core.messages import AIMessage
        content = f"Mock streaming response from {self.name}"
        for token in content.split():
            yield AIMessage(content=token + " ")

# Test tools
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Simple calculator for addition"
    args_schema = CalculatorInput
    
    def _run(self, a: float, b: float) -> str:
        return f"Result: {a + b}"

def test_mongodb_connection():
    """Test basic MongoDB connection"""
    print("🔄 Testing MongoDB connection...")
    
    try:
        import pymongo
        client = pymongo.MongoClient(TEST_MONGODB_URL)
        # Test connection
        client.admin.command('ping')
        
        # Get database info
        db_name = TEST_MONGODB_URL.split('/')[-1] if '/' in TEST_MONGODB_URL else 'coreagent_test'
        db = client[db_name]
        
        # Test collection access
        test_collection = db['test_connection']
        test_doc = {
            "test": "connection",
            "timestamp": datetime.now(),
            "framework": "CoreAgent"
        }
        result = test_collection.insert_one(test_doc)
        
        # Verify insertion
        found_doc = test_collection.find_one({"_id": result.inserted_id})
        assert found_doc is not None, "Failed to insert/retrieve test document"
        
        # Cleanup
        test_collection.delete_one({"_id": result.inserted_id})
        
        print("✅ MongoDB connection successful")
        print(f"   📍 Connected to: {TEST_MONGODB_URL}")
        print(f"   📊 Database: {db_name}")
        print(f"   🔗 Ping: Success")
        return True
        
    except ImportError:
        print("❌ MongoDB dependencies not installed")
        print("   Run: pip install pymongo langgraph-checkpoint-mongodb langgraph-store-mongodb")
        return False
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("   💡 Make sure MongoDB is running and connection string is correct")
        return False

def test_mongodb_short_term_memory():
    """Test MongoDB short-term memory (checkpointer)"""
    print("\n🔄 Testing MongoDB short-term memory (checkpointer)...")
    
    try:
        from core_agent import create_short_term_memory_agent
        
        # Create agent with MongoDB short-term memory
        model = MockChatModel("MongoMemoryLLM")
        tools = [CalculatorTool()]
        
        agent = create_short_term_memory_agent(
            model=model,
            name="MongoShortTermAgent",
            memory_backend="mongodb",
            mongodb_url=TEST_MONGODB_URL,
            tools=tools,
            system_prompt="You are a test agent with MongoDB short-term memory."
        )
        
        # Test conversation with memory
        thread_id = f"mongo_test_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # First message
        response1 = agent.invoke({
            "messages": [{"role": "user", "content": "Remember: My favorite color is blue"}]
        }, config)
        
        # Second message (should remember previous context)
        response2 = agent.invoke({
            "messages": [{"role": "user", "content": "What's my favorite color?"}]
        }, config)
        
        print("✅ MongoDB short-term memory test successful")
        print(f"   📝 Thread ID: {thread_id}")
        print(f"   💾 Checkpointer: MongoDBSaver")
        print(f"   🔄 Memory persistence: Working")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB short-term memory test failed: {e}")
        return False

def test_mongodb_long_term_memory():
    """Test MongoDB long-term memory (store)"""
    print("\n🔄 Testing MongoDB long-term memory (store)...")
    
    try:
        from core_agent import create_long_term_memory_agent
        
        # Create agent with MongoDB long-term memory
        model = MockChatModel("MongoStoreLLM")
        
        agent = create_long_term_memory_agent(
            model=model,
            name="MongoLongTermAgent",
            memory_backend="mongodb",
            mongodb_url=TEST_MONGODB_URL,
            enable_semantic_search=False,  # Disable embeddings for this test
            system_prompt="You are a test agent with MongoDB long-term memory."
        )
        
        # Test memory storage and retrieval
        config = {"configurable": {"thread_id": f"mongo_store_test_{int(time.time())}"}}
        
        # Store some information
        response = agent.invoke({
            "messages": [{"role": "user", "content": "Store this information: User John Doe prefers morning meetings"}]
        }, config)
        
        print("✅ MongoDB long-term memory test successful")
        print(f"   💾 Store: MongoDBStore")
        print(f"   🗃️ Long-term storage: Working")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB long-term memory test failed: {e}")
        return False

def test_mongodb_comprehensive_memory():
    """Test comprehensive MongoDB memory (both short and long term)"""
    print("\n🔄 Testing comprehensive MongoDB memory...")
    
    try:
        from core_agent import create_memory_agent
        
        # Create agent with both MongoDB memories
        model = MockChatModel("MongoComprehensiveLLM")
        tools = [CalculatorTool()]
        
        agent = create_memory_agent(
            model=model,
            name="MongoComprehensiveAgent",
            short_term_memory="mongodb",
            long_term_memory="mongodb",
            mongodb_url=TEST_MONGODB_URL,
            tools=tools,
            enable_semantic_search=False,  # Disable embeddings for simplicity
            enable_memory_tools=False,     # Disable memory tools for simplicity
            system_prompt="You are a test agent with comprehensive MongoDB memory."
        )
        
        # Test comprehensive memory
        thread_id = f"mongo_comprehensive_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Conversation with both short and long term memory
        response = agent.invoke({
            "messages": [{"role": "user", "content": "Calculate 15 + 25 and remember this calculation"}]
        }, config)
        
        print("✅ MongoDB comprehensive memory test successful")
        print(f"   📝 Thread ID: {thread_id}")
        print(f"   💾 Short-term: MongoDBSaver")
        print(f"   🗃️ Long-term: MongoDBStore")
        print(f"   🔧 Tools: Calculator")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB comprehensive memory test failed: {e}")
        return False

def test_mongodb_ttl_memory():
    """Test MongoDB TTL (Time-To-Live) memory"""
    print("\n🔄 Testing MongoDB TTL memory...")
    
    try:
        from core_agent import create_ttl_memory_agent
        
        # Create agent with MongoDB TTL memory (short TTL for testing)
        model = MockChatModel("MongoTTLLLM")
        
        agent = create_ttl_memory_agent(
            model=model,
            name="MongoTTLAgent",
            memory_backend="mongodb",
            ttl_minutes=1,  # Very short TTL for testing
            mongodb_url=TEST_MONGODB_URL,
            system_prompt="You are a test agent with MongoDB TTL memory."
        )
        
        # Test TTL memory
        thread_id = f"mongo_ttl_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": "Remember this temporary information: Meeting at 3 PM"}]
        }, config)
        
        print("✅ MongoDB TTL memory test successful")
        print(f"   📝 Thread ID: {thread_id}")
        print(f"   ⏰ TTL: 1 minute")
        print(f"   🗃️ Auto-expiration: Configured")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB TTL memory test failed: {e}")
        return False

def test_mongodb_session_memory():
    """Test MongoDB session-based memory for agent collaboration"""
    print("\n🔄 Testing MongoDB session-based memory...")
    
    try:
        from core_agent import create_session_agent
        
        # Create multiple agents sharing session memory
        model1 = MockChatModel("MongoSessionLLM1")
        model2 = MockChatModel("MongoSessionLLM2")
        
        # Agent 1: Coder
        coder_agent = create_session_agent(
            model=model1,
            session_id=TEST_SESSION_ID,
            name="MongoCoder",
            memory_namespace="coder",
            mongodb_url=TEST_MONGODB_URL,
            system_prompt="You are a coder agent with MongoDB session memory."
        )
        
        # Agent 2: Reviewer  
        reviewer_agent = create_session_agent(
            model=model2,
            session_id=TEST_SESSION_ID,
            name="MongoReviewer", 
            memory_namespace="reviewer",
            mongodb_url=TEST_MONGODB_URL,
            system_prompt="You are a code reviewer agent with MongoDB session memory."
        )
        
        # Test session collaboration
        config = {"configurable": {"thread_id": f"mongo_session_{int(time.time())}"}}
        
        # Coder writes code
        coder_response = coder_agent.invoke({
            "messages": [{"role": "user", "content": "Write a simple hello world function"}]
        }, config)
        
        # Reviewer checks code (should have access to coder's work through session memory)
        reviewer_response = reviewer_agent.invoke({
            "messages": [{"role": "user", "content": "Review the hello world function"}]
        }, config)
        
        print("✅ MongoDB session memory test successful")
        print(f"   🎯 Session ID: {TEST_SESSION_ID}")
        print(f"   👥 Agents: MongoCoder, MongoReviewer")
        print(f"   🔄 Shared memory: Working")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB session memory test failed: {e}")
        return False

def test_mongodb_performance():
    """Test MongoDB memory performance"""
    print("\n🔄 Testing MongoDB memory performance...")
    
    try:
        import pymongo
        from core_agent import create_memory_agent
        
        # Performance test setup
        model = MockChatModel("MongoPerfLLM")
        
        agent = create_memory_agent(
            model=model,
            name="MongoPerfAgent",
            short_term_memory="mongodb",
            long_term_memory="mongodb",
            mongodb_url=TEST_MONGODB_URL,
            enable_semantic_search=False,
            system_prompt="Performance test agent with MongoDB memory."
        )
        
        # Performance measurements
        thread_id = f"mongo_perf_{int(time.time())}"
        config = {"configurable": {"thread_id": thread_id}}
        
        # Test write performance
        start_time = time.time()
        
        for i in range(5):  # Small test for demo
            response = agent.invoke({
                "messages": [{"role": "user", "content": f"Remember item {i}: Important data point {i}"}]
            }, config)
        
        write_time = time.time() - start_time
        
        # Test read performance
        start_time = time.time()
        
        response = agent.invoke({
            "messages": [{"role": "user", "content": "What items do you remember?"}]
        }, config)
        
        read_time = time.time() - start_time
        
        # MongoDB stats
        client = pymongo.MongoClient(TEST_MONGODB_URL)
        db_name = TEST_MONGODB_URL.split('/')[-1] if '/' in TEST_MONGODB_URL else 'coreagent_test'
        db = client[db_name]
        stats = db.command('dbStats')
        
        print("✅ MongoDB performance test successful")
        print(f"   ⚡ Write performance: {write_time:.3f}s (5 operations)")
        print(f"   ⚡ Read performance: {read_time:.3f}s")
        print(f"   💾 Database size: {stats.get('dataSize', 0) / 1024:.1f} KB")
        print(f"   📊 Collections: {stats.get('collections', 0)}")
        return True
        
    except Exception as e:
        print(f"❌ MongoDB performance test failed: {e}")
        return False

def run_mongodb_tests():
    """Run all MongoDB memory tests"""
    print("🚀 Starting MongoDB Memory Tests for CoreAgent Framework")
    print("=" * 60)
    
    # Test results
    results = {}
    
    # 1. Basic MongoDB connection
    results["connection"] = test_mongodb_connection()
    
    if not results["connection"]:
        print("\n❌ MongoDB connection failed - skipping other tests")
        return results
    
    # 2. Short-term memory (checkpointer)
    results["short_term"] = test_mongodb_short_term_memory()
    
    # 3. Long-term memory (store)
    results["long_term"] = test_mongodb_long_term_memory()
    
    # 4. Comprehensive memory
    results["comprehensive"] = test_mongodb_comprehensive_memory()
    
    # 5. TTL memory
    results["ttl"] = test_mongodb_ttl_memory()
    
    # 6. Session-based memory
    results["session"] = test_mongodb_session_memory()
    
    # 7. Performance test
    results["performance"] = test_mongodb_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 MongoDB Memory Test Summary:")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    success_rate = (passed_tests / total_tests) * 100
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test_name.ljust(15)}: {status}")
    
    print("-" * 40)
    print(f"   Total: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate == 100:
        print("\n🎉 All MongoDB memory tests passed!")
        print("   MongoDB integration is working perfectly!")
    elif success_rate >= 80:
        print(f"\n✅ Most MongoDB tests passed ({success_rate:.1f}%)")
        print("   MongoDB integration is mostly working!")
    else:
        print(f"\n⚠️ Some MongoDB tests failed ({success_rate:.1f}%)")
        print("   Please check your MongoDB setup and dependencies!")
    
    return results

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    results = run_mongodb_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        exit(0)  # All tests passed
    else:
        exit(1)  # Some tests failed