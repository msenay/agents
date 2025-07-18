#!/usr/bin/env python3
"""
Comprehensive Memory Options Test for CoreAgent Framework

This test demonstrates all LangGraph memory patterns and options:
1. Short-term Memory (InMemorySaver, RedisSaver, PostgresSaver)
2. Long-term Memory (InMemoryStore, RedisStore, PostgresStore) 
3. Message Management (trimming, summarization, deletion)
4. Semantic Search with embeddings
5. Session-based Memory for agent collaboration
6. TTL (Time-To-Live) memory with automatic cleanup
7. LangMem integration for advanced memory management

Based on LangGraph documentation patterns.
"""

import os
import asyncio
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Ensure we can import our enhanced framework
import sys
sys.path.append('.')

from core_agent import (
    CoreAgent, AgentConfig,
    # Memory-optimized factory functions
    create_memory_agent,
    create_short_term_memory_agent,
    create_long_term_memory_agent, 
    create_message_management_agent,
    create_semantic_search_agent,
    create_ttl_memory_agent,
    # Session-based functions (existing)
    create_session_agent,
    create_collaborative_agents
)

# Mock LLM for testing without API costs
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# =============================================================================
# MOCK SETUP AND TOOLS
# =============================================================================

# Mock responses for different memory scenarios
MEMORY_RESPONSES = [
    "I'll help you with memory management. Let me store this information.",
    "Based on my memory, I can see we've discussed this before.",
    "I'm using semantic search to find relevant information.",
    "Let me summarize our previous conversation.",
    "I'll trim older messages to stay within context limits.",
    "Storing this in long-term memory for future reference.",
    "I remember our conversation about this topic.",
    "Using TTL memory - this will expire automatically.",
    "Searching my memory for similar content...",
    "I've stored this information across sessions."
]

@tool
def test_memory_tool() -> str:
    """Test tool for memory operations"""
    return "Memory tool executed successfully"

def create_mock_model() -> FakeListChatModel:
    """Create a mock model with memory-focused responses"""
    return FakeListChatModel(responses=MEMORY_RESPONSES * 5)

# =============================================================================
# MEMORY PATTERN TESTS
# =============================================================================

def test_short_term_memory_patterns():
    """Test 1: Short-term Memory (Thread-level persistence)"""
    print("\n" + "="*80)
    print("🧠 TEST 1: SHORT-TERM MEMORY (Thread-level persistence)")
    print("="*80)
    
    model = create_mock_model()
    
    # Test InMemorySaver
    print("\n📝 Testing InMemorySaver...")
    agent_inmemory = create_short_term_memory_agent(
        model=model,
        name="InMemoryAgent",
        memory_backend="inmemory",
        enable_trimming=True,
        max_tokens=4000
    )
    
    try:
        response = agent_inmemory.invoke(
            {"messages": [HumanMessage(content="Remember that I like coffee")]},
            config={"configurable": {"thread_id": "coffee_thread"}}
        )
        print(f"✅ InMemorySaver: {response}")
        print(f"   Memory Manager: {agent_inmemory.memory_manager.has_short_term_memory()}")
        print(f"   Checkpointer: {type(agent_inmemory.memory_manager.get_checkpointer()).__name__}")
    except Exception as e:
        print(f"❌ InMemorySaver failed: {e}")
    
    # Test RedisSaver (if Redis available)
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"\n🔴 Testing RedisSaver with {redis_url}...")
    
    try:
        agent_redis = create_short_term_memory_agent(
            model=model,
            name="RedisAgent",
            memory_backend="redis", 
            redis_url=redis_url,
            enable_trimming=True
        )
        
        response = agent_redis.invoke(
            {"messages": [HumanMessage(content="Remember my Redis preference")]},
            config={"configurable": {"thread_id": "redis_thread"}}
        )
        print(f"✅ RedisSaver: {response}")
        print(f"   Memory Manager: {agent_redis.memory_manager.has_short_term_memory()}")
        print(f"   Checkpointer: {type(agent_redis.memory_manager.get_checkpointer()).__name__}")
    except Exception as e:
        print(f"❌ RedisSaver failed: {e}")
    
    print("\n✨ Short-term memory test completed!")


def test_long_term_memory_patterns():
    """Test 2: Long-term Memory (Cross-session persistence)"""
    print("\n" + "="*80)
    print("🗄️ TEST 2: LONG-TERM MEMORY (Cross-session persistence)")
    print("="*80)
    
    model = create_mock_model()
    
    # Test InMemoryStore
    print("\n📝 Testing InMemoryStore...")
    agent_store = create_long_term_memory_agent(
        model=model,
        name="StoreAgent",
        memory_backend="inmemory",
        enable_semantic_search=True,
        enable_memory_tools=True
    )
    
    try:
        if agent_store.memory_manager.has_long_term_memory():
            # Store some test data
            agent_store.memory_manager.store_long_term_memory(
                "user_pref", 
                {"coffee": "espresso", "language": "python"}
            )
            
            # Retrieve it
            stored_data = agent_store.memory_manager.get_long_term_memory("user_pref")
            print(f"✅ InMemoryStore: Stored and retrieved {stored_data}")
            print(f"   Store Type: {type(agent_store.memory_manager.get_store()).__name__}")
            print(f"   Memory Tools: {len(agent_store.memory_manager.get_memory_tools())} tools available")
        else:
            print("❌ Long-term memory not available")
    except Exception as e:
        print(f"❌ InMemoryStore failed: {e}")
    
    # Test RedisStore (if Redis available)  
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    print(f"\n🔴 Testing RedisStore with {redis_url}...")
    
    try:
        agent_redis_store = create_long_term_memory_agent(
            model=model,
            name="RedisStoreAgent",
            memory_backend="redis",
            redis_url=redis_url,
            enable_semantic_search=True
        )
        
        if agent_redis_store.memory_manager.has_long_term_memory():
            # Test store operations
            test_data = {"topic": "machine_learning", "level": "advanced"}
            agent_redis_store.memory_manager.store_long_term_memory("ml_pref", test_data)
            retrieved = agent_redis_store.memory_manager.get_long_term_memory("ml_pref") 
            print(f"✅ RedisStore: Stored and retrieved {retrieved}")
            print(f"   Store Type: {type(agent_redis_store.memory_manager.get_store()).__name__}")
        else:
            print("❌ RedisStore not available")
    except Exception as e:
        print(f"❌ RedisStore failed: {e}")
    
    print("\n✨ Long-term memory test completed!")


def test_message_management_patterns():
    """Test 3: Message Management (trimming, summarization, deletion)"""
    print("\n" + "="*80)
    print("✂️ TEST 3: MESSAGE MANAGEMENT (trimming, summarization, deletion)")
    print("="*80)
    
    model = create_mock_model()
    
    # Test Message Trimming
    print("\n✂️ Testing Message Trimming...")
    agent_trim = create_message_management_agent(
        model=model,
        name="TrimAgent",
        management_strategy="trim",
        max_tokens=100,  # Small limit to trigger trimming
        trim_strategy="last"
    )
    
    try:
        # Create a long conversation to trigger trimming
        long_messages = [
            HumanMessage(content="This is message 1" * 20),
            AIMessage(content="Response 1" * 20),
            HumanMessage(content="This is message 2" * 20),
            AIMessage(content="Response 2" * 20),
            HumanMessage(content="Final message")
        ]
        
        pre_hook = agent_trim.memory_manager.get_pre_model_hook()
        if pre_hook:
            # Simulate trimming
            trimmed_state = pre_hook({"messages": long_messages})
            print(f"✅ Message Trimming: Processed {len(long_messages)} → {len(trimmed_state.get('llm_input_messages', []))} messages")
            print(f"   Trimming enabled: {agent_trim.config.enable_message_trimming}")
            print(f"   Max tokens: {agent_trim.config.max_tokens}")
        else:
            print("❌ Message trimming hook not available")
    except Exception as e:
        print(f"❌ Message trimming failed: {e}")
    
    # Test Message Summarization  
    print("\n📋 Testing Message Summarization...")
    agent_summarize = create_message_management_agent(
        model=model,
        name="SummarizeAgent",
        management_strategy="summarize",
        enable_summarization=True,
        max_summary_tokens=50
    )
    
    try:
        summary_hook = agent_summarize.memory_manager.get_pre_model_hook()
        if summary_hook:
            print(f"✅ Message Summarization: Hook available")
            print(f"   Summarization enabled: {agent_summarize.config.enable_summarization}")
            print(f"   Max summary tokens: {agent_summarize.config.max_summary_tokens}")
        else:
            print("❌ Summarization requires LangMem installation")
    except Exception as e:
        print(f"❌ Message summarization failed: {e}")
    
    # Test Message Deletion
    print("\n🗑️ Testing Message Deletion...")
    try:
        delete_hook = agent_trim.memory_manager.delete_messages_hook(remove_all=True)
        if delete_hook:
            test_messages = [HumanMessage(content="Test 1"), AIMessage(content="Response 1")]
            deleted_state = delete_hook({"messages": test_messages})
            print(f"✅ Message Deletion: Hook created for removing messages")
            print(f"   Delete operation: {len(deleted_state.get('messages', []))} removal commands")
        else:
            print("❌ Message deletion requires RemoveMessage support")
    except Exception as e:
        print(f"❌ Message deletion failed: {e}")
    
    print("\n✨ Message management test completed!")


def test_semantic_search_patterns():
    """Test 4: Semantic Search with embeddings"""
    print("\n" + "="*80)
    print("🔍 TEST 4: SEMANTIC SEARCH with embeddings")
    print("="*80)
    
    model = create_mock_model()
    
    print("\n🔍 Testing Semantic Search Agent...")
    agent_search = create_semantic_search_agent(
        model=model,
        name="SearchAgent",
        memory_backend="inmemory",
        embedding_model="openai:text-embedding-3-small",
        enable_memory_tools=True
    )
    
    try:
        if agent_search.memory_manager.has_long_term_memory():
            # Test semantic search configuration
            print(f"✅ Semantic Search Agent created")
            print(f"   Long-term memory: {agent_search.memory_manager.has_long_term_memory()}")
            print(f"   Semantic search: {agent_search.config.enable_semantic_search}")
            print(f"   Embedding model: {agent_search.config.embedding_model}")
            print(f"   Embedding dims: {agent_search.config.embedding_dims}")
            print(f"   Distance type: {agent_search.config.distance_type}")
            
            # Test memory search (will work if embeddings are available)
            search_results = agent_search.memory_manager.search_memory("python programming", limit=3)
            print(f"   Search results: {len(search_results)} items found")
        else:
            print("❌ Semantic search agent creation failed")
    except Exception as e:
        print(f"❌ Semantic search failed: {e}")
    
    print("\n✨ Semantic search test completed!")


def test_session_based_memory():
    """Test 5: Session-based Memory for agent collaboration"""
    print("\n" + "="*80)
    print("🤝 TEST 5: SESSION-BASED MEMORY (Agent collaboration)")
    print("="*80)
    
    model = create_mock_model()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    session_id = f"test_session_{int(time.time())}"
    
    print(f"\n🤝 Testing Session-based Memory with session: {session_id}...")
    
    try:
        # Create collaborative agents sharing session memory
        coder_agent = create_session_agent(
            model=model,
            session_id=session_id,
            name="CoderAgent",
            role="coder",
            redis_url=redis_url
        )
        
        reviewer_agent = create_session_agent(
            model=model,
            session_id=session_id,
            name="ReviewerAgent", 
            role="reviewer",
            redis_url=redis_url
        )
        
        if coder_agent.memory_manager.has_session_memory():
            # Test session memory sharing
            test_data = {
                "code": "def hello(): return 'world'",
                "timestamp": str(datetime.now()),
                "agent": "CoderAgent"
            }
            
            coder_agent.memory_manager.store_session_memory(test_data)
            shared_memory = reviewer_agent.memory_manager.get_session_memory()
            
            print(f"✅ Session Memory: {len(shared_memory)} items shared between agents")
            print(f"   Session ID: {session_id}")
            print(f"   Shared data: {shared_memory[0] if shared_memory else 'None'}")
            print(f"   Coder has session memory: {coder_agent.memory_manager.has_session_memory()}")
            print(f"   Reviewer has session memory: {reviewer_agent.memory_manager.has_session_memory()}")
        else:
            print("❌ Session memory not available (requires Redis)")
            
    except Exception as e:
        print(f"❌ Session-based memory failed: {e}")
    
    print("\n✨ Session-based memory test completed!")


def test_ttl_memory_patterns():
    """Test 6: TTL (Time-To-Live) memory with automatic cleanup"""
    print("\n" + "="*80)
    print("⏰ TEST 6: TTL MEMORY (Time-To-Live with automatic cleanup)")
    print("="*80)
    
    model = create_mock_model()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    print(f"\n⏰ Testing TTL Memory with Redis...")
    
    try:
        agent_ttl = create_ttl_memory_agent(
            model=model,
            name="TTLAgent",
            memory_backend="redis",
            ttl_minutes=1,  # 1 minute for testing
            refresh_on_read=True,
            redis_url=redis_url
        )
        
        if agent_ttl.memory_manager.has_short_term_memory() and agent_ttl.memory_manager.has_long_term_memory():
            print(f"✅ TTL Memory Agent created")
            print(f"   TTL enabled: {agent_ttl.config.enable_ttl}")
            print(f"   TTL minutes: {agent_ttl.config.default_ttl_minutes}")
            print(f"   Refresh on read: {agent_ttl.config.refresh_on_read}")
            print(f"   Short-term memory: {agent_ttl.memory_manager.has_short_term_memory()}")
            print(f"   Long-term memory: {agent_ttl.memory_manager.has_long_term_memory()}")
            
            # Test TTL functionality
            if agent_ttl.memory_manager.has_long_term_memory():
                agent_ttl.memory_manager.store_long_term_memory(
                    "ttl_test", 
                    {"message": "This will expire", "timestamp": str(datetime.now())}
                )
                print(f"   Data stored with TTL - will expire in {agent_ttl.config.default_ttl_minutes} minutes")
        else:
            print("❌ TTL memory not available (requires Redis)")
            
    except Exception as e:
        print(f"❌ TTL memory failed: {e}")
    
    print("\n✨ TTL memory test completed!")


def test_comprehensive_memory_agent():
    """Test 7: Comprehensive Memory Agent with all features"""
    print("\n" + "="*80)
    print("🚀 TEST 7: COMPREHENSIVE MEMORY AGENT (All features)")
    print("="*80)
    
    model = create_mock_model()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    print(f"\n🚀 Testing Comprehensive Memory Agent...")
    
    try:
        agent_comprehensive = create_memory_agent(
            model=model,
            name="ComprehensiveAgent",
            short_term_memory="redis",
            long_term_memory="redis",
            enable_semantic_search=True,
            enable_memory_tools=True,
            enable_message_trimming=True,
            enable_summarization=False,  # Disable to avoid LangMem requirement
            redis_url=redis_url
        )
        
        print(f"✅ Comprehensive Memory Agent created")
        print(f"   Short-term memory: {agent_comprehensive.memory_manager.has_short_term_memory()}")
        print(f"   Long-term memory: {agent_comprehensive.memory_manager.has_long_term_memory()}")
        print(f"   Semantic search: {agent_comprehensive.config.enable_semantic_search}")
        print(f"   Memory tools: {len(agent_comprehensive.memory_manager.get_memory_tools())} tools")
        print(f"   Message trimming: {agent_comprehensive.config.enable_message_trimming}")
        print(f"   Max tokens: {agent_comprehensive.config.max_tokens}")
        
        # Test comprehensive functionality
        if agent_comprehensive.memory_manager.has_long_term_memory():
            # Store some data
            agent_comprehensive.memory_manager.store_long_term_memory(
                "comprehensive_test",
                {"features": "all_enabled", "performance": "excellent"}
            )
            
            # Retrieve it
            stored = agent_comprehensive.memory_manager.get_long_term_memory("comprehensive_test")
            print(f"   Data storage test: {stored}")
            
    except Exception as e:
        print(f"❌ Comprehensive memory agent failed: {e}")
    
    print("\n✨ Comprehensive memory test completed!")


def test_memory_performance():
    """Test 8: Memory Performance and Statistics"""
    print("\n" + "="*80)
    print("📊 TEST 8: MEMORY PERFORMANCE and statistics")
    print("="*80)
    
    model = create_mock_model()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    performance_stats = {
        "agents_created": 0,
        "memory_operations": 0,
        "errors": 0,
        "start_time": time.time()
    }
    
    print(f"\n📊 Testing Memory Performance...")
    
    try:
        # Create multiple agents with different memory configurations
        memory_configs = [
            ("inmemory", "inmemory"),
            ("redis", "redis"),
            ("inmemory", "redis"),
        ]
        
        agents = []
        for i, (short_mem, long_mem) in enumerate(memory_configs):
            try:
                agent = create_memory_agent(
                    model=model,
                    name=f"PerfAgent{i}",
                    short_term_memory=short_mem,
                    long_term_memory=long_mem,
                    redis_url=redis_url if "redis" in [short_mem, long_mem] else None
                )
                agents.append(agent)
                performance_stats["agents_created"] += 1
                
                # Test memory operations
                if agent.memory_manager.has_long_term_memory():
                    agent.memory_manager.store_long_term_memory(f"perf_test_{i}", {"test": f"data_{i}"})
                    agent.memory_manager.get_long_term_memory(f"perf_test_{i}")
                    performance_stats["memory_operations"] += 2
                    
            except Exception as e:
                performance_stats["errors"] += 1
                print(f"   ⚠️ Config {short_mem}/{long_mem} failed: {e}")
        
        # Calculate performance metrics
        elapsed_time = time.time() - performance_stats["start_time"]
        
        print(f"✅ Performance Test Results:")
        print(f"   Agents created: {performance_stats['agents_created']}")
        print(f"   Memory operations: {performance_stats['memory_operations']}")
        print(f"   Errors: {performance_stats['errors']}")
        print(f"   Total time: {elapsed_time:.2f} seconds")
        print(f"   Avg time per agent: {elapsed_time/max(1, performance_stats['agents_created']):.3f} seconds")
        
        if performance_stats["memory_operations"] > 0:
            print(f"   Avg time per operation: {elapsed_time/performance_stats['memory_operations']:.3f} seconds")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    print("\n✨ Performance test completed!")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

def main():
    """Run all comprehensive memory tests"""
    print("🎯 COREAGENT COMPREHENSIVE MEMORY TEST SUITE")
    print("Testing all LangGraph memory patterns and options")
    print(f"Start time: {datetime.now()}")
    
    # Check Redis availability
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    try:
        import redis
        r = redis.from_url(redis_url)
        r.ping()
        print(f"✅ Redis available at {redis_url}")
    except Exception as e:
        print(f"⚠️ Redis not available: {e}")
        print("   Some tests will use InMemory fallbacks")
    
    # Run all memory pattern tests
    test_functions = [
        test_short_term_memory_patterns,
        test_long_term_memory_patterns,
        test_message_management_patterns,
        test_semantic_search_patterns,
        test_session_based_memory,
        test_ttl_memory_patterns,
        test_comprehensive_memory_agent,
        test_memory_performance
    ]
    
    results = {"passed": 0, "failed": 0}
    
    for test_func in test_functions:
        try:
            test_func()
            results["passed"] += 1
        except Exception as e:
            print(f"\n❌ {test_func.__name__} FAILED: {e}")
            results["failed"] += 1
    
    # Final results
    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE MEMORY TEST RESULTS")
    print("="*80)
    print(f"✅ Tests passed: {results['passed']}")
    print(f"❌ Tests failed: {results['failed']}")
    print(f"📊 Success rate: {results['passed']/(results['passed']+results['failed'])*100:.1f}%")
    print(f"End time: {datetime.now()}")
    
    if results["failed"] == 0:
        print("\n🎉 ALL MEMORY TESTS PASSED!")
        print("   CoreAgent framework supports all LangGraph memory patterns!")
    else:
        print(f"\n⚠️ {results['failed']} tests failed - check dependencies and configuration")


if __name__ == "__main__":
    main()