"""
Redis Memory Test for CoreAgent Framework
=========================================

This test validates Redis-based memory functionality in a Docker environment.
Tests include:
1. Basic Redis connectivity
2. Agent memory persistence
3. Multi-agent memory isolation
4. Memory retrieval and conversation continuity
5. Error handling and recovery
"""

import asyncio
import os
import time
import redis
from typing import Dict, Any

# Set Azure OpenAI environment
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACOGgIx4"

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from core_agent import (
    CoreAgent, AgentConfig,
    create_simple_agent, create_advanced_agent, create_memory_agent
)


# =============================================================================
# REDIS CONNECTION TESTING
# =============================================================================

def test_redis_connection():
    """Test basic Redis connectivity"""
    print("🔗 TESTING REDIS CONNECTION")
    print("=" * 40)
    
    try:
        # Get Redis URL from environment
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        print(f"Redis URL: {redis_url}")
        
        # Create Redis client
        r = redis.from_url(redis_url)
        
        # Test connection
        r.ping()
        print("✅ Redis connection successful!")
        
        # Test basic operations
        r.set("test_key", "test_value")
        value = r.get("test_key")
        print(f"✅ Redis set/get test: {value.decode()}")
        
        # Get Redis info
        info = r.info()
        print(f"✅ Redis version: {info.get('redis_version')}")
        print(f"✅ Connected clients: {info.get('connected_clients')}")
        print(f"✅ Used memory: {info.get('used_memory_human')}")
        
        # Clean up test key
        r.delete("test_key")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False


# =============================================================================
# TOOLS FOR MEMORY TESTING
# =============================================================================

@tool
def remember_important_info(info: str) -> str:
    """Store important information for later recall"""
    return f"Stored important info: {info[:50]}..."

@tool
def recall_previous_info(query: str) -> str:
    """Recall previously stored information"""
    return f"Recalling info about: {query}"

@tool
def analyze_user_behavior(behavior: str) -> str:
    """Analyze and remember user behavior patterns"""
    return f"Analyzed behavior: {behavior}"


# =============================================================================
# AGENT CREATION WITH REDIS MEMORY
# =============================================================================

def create_redis_llm():
    """Create LLM for Redis memory testing"""
    return AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=1500,
        model_name="gpt-4"
    )


def create_redis_memory_agents():
    """Create agents with Redis memory enabled"""
    
    llm = create_redis_llm()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Agent 1: Personal Assistant with Redis memory
    personal_agent = create_memory_agent(
        model=llm,
        name="Personal Assistant",
        tools=[remember_important_info, recall_previous_info],
        system_prompt="""You are a personal assistant with persistent memory. 
        
🧠 MEMORY CAPABILITIES:
- Remember user preferences, habits, and important information
- Recall previous conversations and context
- Learn from interactions to provide better assistance
- Maintain continuity across sessions

📝 YOUR APPROACH:
- Always reference relevant past information when available
- Store important details for future reference
- Provide personalized assistance based on memory
- Acknowledge when you remember something from before

Be helpful, remembering, and continuously learning!""",
        memory_type="redis",
        redis_url=redis_url,
        enable_evaluation=False
    )
    
    # Agent 2: Data Analyst with Redis memory
    analyst_agent = create_memory_agent(
        model=llm,
        name="Data Analyst",
        tools=[analyze_user_behavior, remember_important_info],
        system_prompt="""You are a data analyst with persistent memory for patterns.
        
📊 ANALYTICAL MEMORY:
- Remember data patterns and trends over time
- Store analytical insights and findings
- Track user behavior and preferences
- Build knowledge base of analytical results

🔍 YOUR APPROACH:
- Reference previous analyses when relevant
- Build on past insights and findings
- Identify patterns across sessions
- Provide data-driven recommendations

Be analytical, thorough, and memory-driven!""",
        memory_type="redis",
        redis_url=redis_url,
        enable_evaluation=False
    )
    
    # Agent 3: Simple Agent without memory (control group)
    simple_agent = create_simple_agent(
        model=llm,
        name="Simple Agent",
        system_prompt="You are a helpful assistant without persistent memory.",
        enable_memory=False
    )
    
    return {
        "personal": personal_agent,
        "analyst": analyst_agent,
        "simple": simple_agent
    }


# =============================================================================
# MEMORY PERSISTENCE TESTING
# =============================================================================

async def test_memory_persistence():
    """Test if memory persists across agent sessions"""
    
    print("\n🧠 TESTING MEMORY PERSISTENCE")
    print("=" * 40)
    
    try:
        agents = create_redis_memory_agents()
        personal_agent = agents["personal"]
        
        # Session 1: Store some information
        print("\n📝 SESSION 1: Storing information")
        session1_messages = [
            "My name is John and I work as a software engineer.",
            "I prefer tea over coffee in the mornings.",
            "I have a meeting with the team every Tuesday at 2 PM.",
            "I'm currently working on a machine learning project."
        ]
        
        session1_responses = []
        for message in session1_messages:
            print(f"User: {message}")
            response = await personal_agent.ainvoke(message)
            response_text = str(response)[:200]
            print(f"Agent: {response_text}...")
            session1_responses.append(response_text)
            await asyncio.sleep(1)  # Small delay
        
        print("✅ Session 1 completed - Information stored")
        
        # Wait a moment to simulate session gap
        await asyncio.sleep(2)
        
        # Session 2: Test memory recall
        print("\n🔍 SESSION 2: Testing memory recall")
        recall_messages = [
            "What's my name and profession?",
            "What do I prefer to drink in the morning?",
            "When is my team meeting?",
            "What project am I working on?"
        ]
        
        session2_responses = []
        for message in recall_messages:
            print(f"User: {message}")
            response = await personal_agent.ainvoke(message)
            response_text = str(response)[:200]
            print(f"Agent: {response_text}...")
            session2_responses.append(response_text)
            await asyncio.sleep(1)
        
        print("✅ Session 2 completed - Memory recall tested")
        
        # Analyze memory effectiveness
        memory_keywords = ["John", "engineer", "tea", "Tuesday", "machine learning"]
        memory_matches = 0
        
        for response in session2_responses:
            for keyword in memory_keywords:
                if keyword.lower() in response.lower():
                    memory_matches += 1
                    break
        
        memory_effectiveness = (memory_matches / len(recall_messages)) * 100
        print(f"\n📊 MEMORY ANALYSIS:")
        print(f"Memory effectiveness: {memory_effectiveness:.1f}%")
        print(f"Successful recalls: {memory_matches}/{len(recall_messages)}")
        
        return {
            "session1_responses": session1_responses,
            "session2_responses": session2_responses,
            "memory_effectiveness": memory_effectiveness,
            "success": memory_effectiveness > 50
        }
        
    except Exception as e:
        print(f"❌ Memory persistence test failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# MULTI-AGENT MEMORY ISOLATION TESTING
# =============================================================================

async def test_memory_isolation():
    """Test that different agents have isolated memories"""
    
    print("\n🔒 TESTING MEMORY ISOLATION")
    print("=" * 40)
    
    try:
        agents = create_redis_memory_agents()
        personal_agent = agents["personal"]
        analyst_agent = agents["analyst"]
        
        # Store different information in each agent
        print("\n📝 STORING AGENT-SPECIFIC INFORMATION")
        
        # Personal agent stores user preferences
        personal_info = "I love Italian food and prefer working from home."
        print(f"Personal Agent - User: {personal_info}")
        personal_response = await personal_agent.ainvoke(personal_info)
        print(f"Personal Agent: {str(personal_response)[:150]}...")
        
        await asyncio.sleep(1)
        
        # Analyst agent stores different data
        analyst_info = "User clicks on product recommendations 80% of the time."
        print(f"\nAnalyst Agent - User: {analyst_info}")
        analyst_response = await analyst_agent.ainvoke(analyst_info)
        print(f"Analyst Agent: {str(analyst_response)[:150]}...")
        
        await asyncio.sleep(2)
        
        # Test cross-contamination
        print("\n🔍 TESTING MEMORY ISOLATION")
        
        # Ask personal agent about analyst data (should not know)
        cross_test1 = "What percentage of time do I click on product recommendations?"
        print(f"Personal Agent - User: {cross_test1}")
        cross_response1 = await personal_agent.ainvoke(cross_test1)
        print(f"Personal Agent: {str(cross_response1)[:150]}...")
        
        await asyncio.sleep(1)
        
        # Ask analyst agent about personal data (should not know)
        cross_test2 = "What type of food do I love?"
        print(f"Analyst Agent - User: {cross_test2}")
        cross_response2 = await analyst_agent.ainvoke(cross_test2)
        print(f"Analyst Agent: {str(cross_response2)[:150]}...")
        
        # Check for memory leakage
        personal_leak = "italian" in str(cross_response2).lower() or "home" in str(cross_response2).lower()
        analyst_leak = "80%" in str(cross_response1) or "click" in str(cross_response1).lower()
        
        isolation_success = not (personal_leak or analyst_leak)
        
        print(f"\n📊 ISOLATION ANALYSIS:")
        print(f"Personal data leaked to analyst: {'❌ Yes' if personal_leak else '✅ No'}")
        print(f"Analyst data leaked to personal: {'❌ Yes' if analyst_leak else '✅ No'}")
        print(f"Memory isolation: {'✅ Success' if isolation_success else '❌ Failed'}")
        
        return {
            "isolation_success": isolation_success,
            "personal_leak": personal_leak,
            "analyst_leak": analyst_leak
        }
        
    except Exception as e:
        print(f"❌ Memory isolation test failed: {e}")
        return {"isolation_success": False, "error": str(e)}


# =============================================================================
# REDIS PERFORMANCE TESTING
# =============================================================================

async def test_redis_performance():
    """Test Redis performance with memory operations"""
    
    print("\n⚡ TESTING REDIS PERFORMANCE")
    print("=" * 40)
    
    try:
        agents = create_redis_memory_agents()
        agent = agents["personal"]
        
        # Performance test parameters
        num_messages = 5
        performance_data = []
        
        print(f"\n📊 PERFORMANCE TEST: {num_messages} messages")
        
        for i in range(num_messages):
            start_time = time.time()
            
            message = f"Remember this test fact #{i + 1}: Performance testing is important for validation."
            response = await agent.ainvoke(message)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            performance_data.append(response_time)
            print(f"Message {i + 1}: {response_time:.2f}s")
            
            await asyncio.sleep(0.5)  # Small delay between messages
        
        # Calculate performance metrics
        avg_response_time = sum(performance_data) / len(performance_data)
        min_response_time = min(performance_data)
        max_response_time = max(performance_data)
        
        print(f"\n📈 PERFORMANCE RESULTS:")
        print(f"Average response time: {avg_response_time:.2f}s")
        print(f"Minimum response time: {min_response_time:.2f}s")
        print(f"Maximum response time: {max_response_time:.2f}s")
        
        # Performance threshold (should be reasonable)
        performance_acceptable = avg_response_time < 30.0  # 30 seconds threshold
        
        print(f"Performance acceptable: {'✅ Yes' if performance_acceptable else '❌ No'}")
        
        return {
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "performance_acceptable": performance_acceptable
        }
        
    except Exception as e:
        print(f"❌ Redis performance test failed: {e}")
        return {"performance_acceptable": False, "error": str(e)}


# =============================================================================
# REDIS CLEANUP AND MONITORING
# =============================================================================

def test_redis_cleanup():
    """Test Redis cleanup and monitoring capabilities"""
    
    print("\n🧹 TESTING REDIS CLEANUP")
    print("=" * 40)
    
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url)
        
        # Get memory usage before
        info_before = r.info()
        memory_before = info_before.get('used_memory', 0)
        keys_before = len(r.keys("*"))
        
        print(f"Before cleanup:")
        print(f"  Memory used: {info_before.get('used_memory_human', 'N/A')}")
        print(f"  Total keys: {keys_before}")
        
        # Cleanup test keys (be careful not to delete real data)
        test_pattern = "test_*"
        test_keys = r.keys(test_pattern)
        if test_keys:
            deleted = r.delete(*test_keys)
            print(f"  Deleted {deleted} test keys")
        
        # Get memory usage after
        info_after = r.info()
        memory_after = info_after.get('used_memory', 0)
        keys_after = len(r.keys("*"))
        
        print(f"\nAfter cleanup:")
        print(f"  Memory used: {info_after.get('used_memory_human', 'N/A')}")
        print(f"  Total keys: {keys_after}")
        print(f"  Memory saved: {memory_before - memory_after} bytes")
        
        return {
            "cleanup_success": True,
            "keys_before": keys_before,
            "keys_after": keys_after,
            "memory_saved": memory_before - memory_after
        }
        
    except Exception as e:
        print(f"❌ Redis cleanup test failed: {e}")
        return {"cleanup_success": False, "error": str(e)}


# =============================================================================
# COMPREHENSIVE TEST RUNNER
# =============================================================================

async def run_redis_memory_tests():
    """Run comprehensive Redis memory tests"""
    
    print("🚀 COREAGENT REDIS MEMORY TEST SUITE")
    print("=" * 50)
    print("Testing Redis-based memory functionality in Docker environment")
    print("=" * 50)
    
    # Test results storage
    test_results = {}
    
    # 1. Test Redis connection
    redis_connection_ok = test_redis_connection()
    test_results["redis_connection"] = redis_connection_ok
    
    if not redis_connection_ok:
        print("\n❌ CRITICAL: Redis connection failed - aborting tests")
        return test_results
    
    # 2. Test memory persistence
    persistence_results = await test_memory_persistence()
    test_results["memory_persistence"] = persistence_results
    
    # 3. Test memory isolation
    isolation_results = await test_memory_isolation()
    test_results["memory_isolation"] = isolation_results
    
    # 4. Test Redis performance
    performance_results = await test_redis_performance()
    test_results["redis_performance"] = performance_results
    
    # 5. Test Redis cleanup
    cleanup_results = test_redis_cleanup()
    test_results["redis_cleanup"] = cleanup_results
    
    # Generate final report
    print("\n📊 FINAL TEST REPORT")
    print("=" * 50)
    
    total_tests = 5
    passed_tests = 0
    
    if test_results.get("redis_connection"):
        print("✅ Redis Connection: PASSED")
        passed_tests += 1
    else:
        print("❌ Redis Connection: FAILED")
    
    if test_results.get("memory_persistence", {}).get("success"):
        print("✅ Memory Persistence: PASSED")
        passed_tests += 1
    else:
        print("❌ Memory Persistence: FAILED")
    
    if test_results.get("memory_isolation", {}).get("isolation_success"):
        print("✅ Memory Isolation: PASSED")
        passed_tests += 1
    else:
        print("❌ Memory Isolation: FAILED")
    
    if test_results.get("redis_performance", {}).get("performance_acceptable"):
        print("✅ Redis Performance: PASSED")
        passed_tests += 1
    else:
        print("❌ Redis Performance: FAILED")
    
    if test_results.get("redis_cleanup", {}).get("cleanup_success"):
        print("✅ Redis Cleanup: PASSED")
        passed_tests += 1
    else:
        print("❌ Redis Cleanup: FAILED")
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n🎯 OVERALL RESULTS:")
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 REDIS MEMORY SYSTEM: SUCCESS!")
    elif success_rate >= 60:
        print("⚠️  REDIS MEMORY SYSTEM: PARTIAL SUCCESS")
    else:
        print("❌ REDIS MEMORY SYSTEM: NEEDS IMPROVEMENT")
    
    return test_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main test execution function"""
    
    try:
        # Wait for Redis to be ready
        print("⏳ Waiting for Redis to be ready...")
        await asyncio.sleep(5)
        
        # Run the comprehensive test suite
        results = await run_redis_memory_tests()
        
        # Save results to file for Docker logs
        with open("/app/redis_test_results.txt", "w") as f:
            f.write("COREAGENT REDIS MEMORY TEST RESULTS\n")
            f.write("="*40 + "\n")
            for test_name, result in results.items():
                f.write(f"{test_name}: {result}\n")
        
        print(f"\n📄 Test results saved to: /app/redis_test_results.txt")
        
        return results
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())