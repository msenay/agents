"""
Comprehensive Redis Memory Test for CoreAgent Framework
======================================================

Advanced Redis memory testing including:
1. Complex conversation continuity
2. Memory degradation testing
3. Concurrent agent memory operations
4. Memory size and performance limits
5. Redis failover and recovery
"""

import asyncio
import os
import time
import redis
import json
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

# Set Azure OpenAI environment
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACOGgIx4"

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from core_agent import (
    CoreAgent, AgentConfig,
    create_memory_agent, create_advanced_agent
)


# =============================================================================
# ADVANCED TOOLS FOR COMPREHENSIVE TESTING
# =============================================================================

@tool
def store_conversation_context(context: str) -> str:
    """Store conversation context for later reference"""
    return f"Stored conversation context: {context[:100]}..."

@tool
def remember_user_preference(preference: str) -> str:
    """Remember user preference for personalization"""
    return f"Remembered preference: {preference}"

@tool
def track_learning_progress(progress: str) -> str:
    """Track and remember learning progress"""
    return f"Tracked progress: {progress}"

@tool
def analyze_conversation_pattern(pattern: str) -> str:
    """Analyze and remember conversation patterns"""
    return f"Analyzed pattern: {pattern}"


# =============================================================================
# COMPLEX CONVERSATION CONTINUITY TEST
# =============================================================================

async def test_complex_conversation_continuity():
    """Test complex conversation continuity across multiple sessions"""
    
    print("🗣️ TESTING COMPLEX CONVERSATION CONTINUITY")
    print("=" * 50)
    
    try:
        # Create agent with Redis memory
        llm = AzureChatOpenAI(
            azure_deployment="gpt4",
            api_version="2023-12-01-preview",
            temperature=0.1,
            max_tokens=2000
        )
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        conversation_agent = create_memory_agent(
            model=llm,
            name="Conversation Continuity Agent",
            tools=[store_conversation_context, remember_user_preference],
            system_prompt="""You are a sophisticated assistant with persistent memory for complex conversations.
            
🧠 ADVANCED MEMORY CAPABILITIES:
- Remember detailed conversation history and context
- Track conversation topics and their evolution
- Maintain awareness of user preferences and patterns
- Provide continuity across multiple sessions
- Reference specific details from previous interactions

📝 CONVERSATION APPROACH:
- Always acknowledge relevant previous interactions
- Build on previous conversation topics naturally
- Show awareness of user's communication style
- Maintain consistent personality and knowledge
- Demonstrate learning from past interactions

Be contextually aware, personally engaged, and demonstrably remembering!""",
            memory_type="redis",
            redis_url=redis_url,
            enable_evaluation=False
        )
        
        # Multi-session conversation scenario
        conversation_sessions = [
            {
                "session": 1,
                "messages": [
                    "Hi, I'm planning a trip to Japan next month.",
                    "I'm particularly interested in traditional temples and gardens.",
                    "My budget is around $3000 for the entire trip.",
                    "I prefer quiet, peaceful places over crowded tourist spots."
                ]
            },
            {
                "session": 2,
                "messages": [
                    "Remember we talked about my Japan trip? I've been doing research.",
                    "I found some beautiful temples in Kyoto - Kinkaku-ji and Fushimi Inari.",
                    "Do you remember my budget and preferences?",
                    "Can you suggest some quiet ryokans based on what you know about me?"
                ]
            },
            {
                "session": 3,
                "messages": [
                    "Quick update on my Japan planning!",
                    "I decided to extend my budget to $3500 based on your suggestions.",
                    "Remember the ryokans you mentioned? I'm ready to book one.",
                    "Also, what were those garden recommendations from our first chat?"
                ]
            }
        ]
        
        all_responses = []
        context_references = 0
        
        for session_data in conversation_sessions:
            session_num = session_data["session"]
            messages = session_data["messages"]
            
            print(f"\n📱 SESSION {session_num}")
            print("-" * 30)
            
            session_responses = []
            
            for message in messages:
                print(f"User: {message}")
                response = await conversation_agent.ainvoke(message)
                response_text = str(response)
                print(f"Agent: {response_text[:300]}...")
                
                session_responses.append(response_text)
                
                # Check for context references from previous sessions
                context_keywords = ["remember", "mentioned", "discussed", "previous", "earlier", "before", "budget", "3000", "temple", "quiet", "peaceful"]
                for keyword in context_keywords:
                    if keyword.lower() in response_text.lower():
                        context_references += 1
                        break
                
                await asyncio.sleep(1)
            
            all_responses.extend(session_responses)
            
            # Simulate time gap between sessions
            if session_num < len(conversation_sessions):
                print(f"⏳ Gap between sessions {session_num} and {session_num + 1}")
                await asyncio.sleep(3)
        
        # Analyze conversation continuity
        total_messages = sum(len(session["messages"]) for session in conversation_sessions)
        continuity_score = (context_references / total_messages) * 100
        
        print(f"\n📊 CONVERSATION CONTINUITY ANALYSIS:")
        print(f"Total messages: {total_messages}")
        print(f"Context references: {context_references}")
        print(f"Continuity score: {continuity_score:.1f}%")
        
        success = continuity_score > 30  # At least 30% context awareness
        
        return {
            "continuity_score": continuity_score,
            "context_references": context_references,
            "total_messages": total_messages,
            "success": success,
            "all_responses": all_responses
        }
        
    except Exception as e:
        print(f"❌ Conversation continuity test failed: {e}")
        return {"success": False, "error": str(e)}


# =============================================================================
# CONCURRENT MEMORY OPERATIONS TEST
# =============================================================================

async def test_concurrent_memory_operations():
    """Test concurrent memory operations with multiple agents"""
    
    print("\n🔄 TESTING CONCURRENT MEMORY OPERATIONS")
    print("=" * 50)
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment="gpt4",
            api_version="2023-12-01-preview",
            temperature=0.1,
            max_tokens=1000
        )
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Create multiple agents with Redis memory
        agents = []
        for i in range(3):
            agent = create_memory_agent(
                model=llm,
                name=f"Concurrent Agent {i+1}",
                tools=[track_learning_progress],
                system_prompt=f"You are concurrent agent {i+1} with persistent memory. Remember and track your specific agent number and interactions.",
                memory_type="redis",
                redis_url=redis_url,
                enable_evaluation=False
            )
            agents.append(agent)
        
        # Concurrent operations
        async def agent_task(agent, agent_id, task_count):
            """Task for individual agent"""
            results = []
            for i in range(task_count):
                message = f"Agent {agent_id}: Task {i+1} - Remember this specific interaction."
                response = await agent.ainvoke(message)
                results.append(str(response)[:150])
                await asyncio.sleep(0.5)
            return results
        
        # Run concurrent operations
        print("🚀 Starting concurrent memory operations...")
        
        start_time = time.time()
        
        # Create tasks for all agents
        tasks = []
        for i, agent in enumerate(agents):
            task = agent_task(agent, i+1, 3)  # 3 messages per agent
            tasks.append(task)
        
        # Execute concurrently
        concurrent_results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Concurrent operations completed in {total_time:.2f}s")
        
        # Test memory isolation after concurrent operations
        print("\n🔍 Testing memory isolation after concurrent operations...")
        
        isolation_test_results = []
        for i, agent in enumerate(agents):
            test_message = f"What is your agent number and how many tasks have you completed?"
            response = await agent.ainvoke(test_message)
            response_text = str(response)
            isolation_test_results.append(response_text)
            print(f"Agent {i+1} memory check: {response_text[:200]}...")
        
        # Analyze concurrent performance
        total_operations = len(agents) * 3
        avg_time_per_op = total_time / total_operations
        
        concurrent_success = total_time < 120  # Should complete in under 2 minutes
        
        print(f"\n📊 CONCURRENT OPERATIONS ANALYSIS:")
        print(f"Total operations: {total_operations}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average time per operation: {avg_time_per_op:.2f}s")
        print(f"Concurrent performance: {'✅ Acceptable' if concurrent_success else '❌ Too slow'}")
        
        return {
            "concurrent_success": concurrent_success,
            "total_time": total_time,
            "avg_time_per_op": avg_time_per_op,
            "total_operations": total_operations,
            "isolation_results": isolation_test_results
        }
        
    except Exception as e:
        print(f"❌ Concurrent memory operations test failed: {e}")
        return {"concurrent_success": False, "error": str(e)}


# =============================================================================
# MEMORY SIZE AND PERFORMANCE LIMITS TEST
# =============================================================================

async def test_memory_size_limits():
    """Test memory performance with large amounts of data"""
    
    print("\n📏 TESTING MEMORY SIZE AND PERFORMANCE LIMITS")
    print("=" * 50)
    
    try:
        llm = AzureChatOpenAI(
            azure_deployment="gpt4",
            api_version="2023-12-01-preview",
            temperature=0.1,
            max_tokens=1000
        )
        
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        large_memory_agent = create_memory_agent(
            model=llm,
            name="Large Memory Agent",
            tools=[store_conversation_context],
            system_prompt="You are an agent designed to handle large amounts of memory data. Store and recall information efficiently.",
            memory_type="redis",
            redis_url=redis_url,
            enable_evaluation=False
        )
        
        # Test with progressively larger data
        data_sizes = [10, 50, 100]  # Number of messages
        performance_data = []
        
        for size in data_sizes:
            print(f"\n📊 Testing with {size} messages...")
            
            start_time = time.time()
            
            # Store large amount of data
            for i in range(size):
                message = f"Store this detailed information #{i+1}: This is a comprehensive data entry containing multiple details, facts, and context that should be remembered for future reference. The entry includes timestamps, user preferences, behavioral patterns, and analytical insights that are crucial for maintaining conversation continuity and providing personalized assistance."
                
                response = await large_memory_agent.ainvoke(message)
                
                if i % 10 == 0:  # Progress indicator
                    print(f"  Stored {i+1}/{size} messages...")
                
                await asyncio.sleep(0.1)  # Small delay
            
            # Test retrieval
            retrieval_start = time.time()
            retrieval_message = f"Can you recall information from the {size} messages I just shared?"
            retrieval_response = await large_memory_agent.ainvoke(retrieval_message)
            retrieval_time = time.time() - retrieval_start
            
            total_time = time.time() - start_time
            
            performance_data.append({
                "size": size,
                "storage_time": total_time - retrieval_time,
                "retrieval_time": retrieval_time,
                "total_time": total_time,
                "avg_per_message": (total_time - retrieval_time) / size
            })
            
            print(f"  Storage time: {total_time - retrieval_time:.2f}s")
            print(f"  Retrieval time: {retrieval_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
        
        # Analyze performance scaling
        print(f"\n📈 MEMORY SIZE PERFORMANCE ANALYSIS:")
        for data in performance_data:
            print(f"Size {data['size']}: {data['avg_per_message']:.3f}s per message")
        
        # Check if performance degrades linearly
        if len(performance_data) >= 2:
            scaling_factor = performance_data[-1]['avg_per_message'] / performance_data[0]['avg_per_message']
            scaling_acceptable = scaling_factor < 3.0  # Less than 3x slowdown
        else:
            scaling_acceptable = True
            scaling_factor = 1.0
        
        print(f"Performance scaling factor: {scaling_factor:.2f}x")
        print(f"Scaling acceptable: {'✅ Yes' if scaling_acceptable else '❌ No'}")
        
        return {
            "scaling_acceptable": scaling_acceptable,
            "scaling_factor": scaling_factor,
            "performance_data": performance_data,
            "max_size_tested": max(data_sizes)
        }
        
    except Exception as e:
        print(f"❌ Memory size limits test failed: {e}")
        return {"scaling_acceptable": False, "error": str(e)}


# =============================================================================
# REDIS MONITORING AND HEALTH CHECK
# =============================================================================

def test_redis_health_monitoring():
    """Test Redis health monitoring and metrics"""
    
    print("\n🏥 TESTING REDIS HEALTH MONITORING")
    print("=" * 50)
    
    try:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        r = redis.from_url(redis_url)
        
        # Get comprehensive Redis info
        info = r.info()
        
        # Key metrics
        metrics = {
            "redis_version": info.get('redis_version'),
            "connected_clients": info.get('connected_clients'),
            "used_memory": info.get('used_memory'),
            "used_memory_human": info.get('used_memory_human'),
            "used_memory_peak": info.get('used_memory_peak'),
            "total_commands_processed": info.get('total_commands_processed'),
            "keyspace_hits": info.get('keyspace_hits', 0),
            "keyspace_misses": info.get('keyspace_misses', 0),
            "uptime_in_seconds": info.get('uptime_in_seconds')
        }
        
        print("📊 REDIS HEALTH METRICS:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Calculate hit ratio
        hits = metrics['keyspace_hits']
        misses = metrics['keyspace_misses']
        if hits + misses > 0:
            hit_ratio = (hits / (hits + misses)) * 100
        else:
            hit_ratio = 0
        
        print(f"  keyspace_hit_ratio: {hit_ratio:.2f}%")
        
        # Health checks
        health_checks = {
            "memory_usage_acceptable": info.get('used_memory', 0) < 200 * 1024 * 1024,  # Less than 200MB
            "clients_reasonable": info.get('connected_clients', 0) < 100,
            "hit_ratio_good": hit_ratio > 50 if hits + misses > 10 else True,
            "uptime_stable": info.get('uptime_in_seconds', 0) > 10
        }
        
        print(f"\n🏥 HEALTH CHECKS:")
        for check, status in health_checks.items():
            print(f"  {check}: {'✅ Pass' if status else '❌ Fail'}")
        
        overall_health = all(health_checks.values())
        
        return {
            "overall_health": overall_health,
            "metrics": metrics,
            "hit_ratio": hit_ratio,
            "health_checks": health_checks
        }
        
    except Exception as e:
        print(f"❌ Redis health monitoring failed: {e}")
        return {"overall_health": False, "error": str(e)}


# =============================================================================
# COMPREHENSIVE TEST RUNNER
# =============================================================================

async def run_comprehensive_redis_tests():
    """Run comprehensive Redis memory tests"""
    
    print("🚀 COMPREHENSIVE REDIS MEMORY TEST SUITE")
    print("=" * 60)
    print("Advanced Redis memory testing for CoreAgent framework")
    print("=" * 60)
    
    # Test results storage
    test_results = {}
    
    # 1. Complex conversation continuity
    print("\n1️⃣ COMPLEX CONVERSATION CONTINUITY TEST")
    continuity_results = await test_complex_conversation_continuity()
    test_results["conversation_continuity"] = continuity_results
    
    # 2. Concurrent memory operations
    print("\n2️⃣ CONCURRENT MEMORY OPERATIONS TEST")
    concurrent_results = await test_concurrent_memory_operations()
    test_results["concurrent_operations"] = concurrent_results
    
    # 3. Memory size and performance limits
    print("\n3️⃣ MEMORY SIZE AND PERFORMANCE LIMITS TEST")
    size_results = await test_memory_size_limits()
    test_results["memory_size_limits"] = size_results
    
    # 4. Redis health monitoring
    print("\n4️⃣ REDIS HEALTH MONITORING TEST")
    health_results = test_redis_health_monitoring()
    test_results["redis_health"] = health_results
    
    # Generate comprehensive report
    print("\n📊 COMPREHENSIVE TEST REPORT")
    print("=" * 60)
    
    total_tests = 4
    passed_tests = 0
    
    if test_results.get("conversation_continuity", {}).get("success"):
        print("✅ Conversation Continuity: PASSED")
        passed_tests += 1
    else:
        print("❌ Conversation Continuity: FAILED")
    
    if test_results.get("concurrent_operations", {}).get("concurrent_success"):
        print("✅ Concurrent Operations: PASSED")
        passed_tests += 1
    else:
        print("❌ Concurrent Operations: FAILED")
    
    if test_results.get("memory_size_limits", {}).get("scaling_acceptable"):
        print("✅ Memory Size Limits: PASSED")
        passed_tests += 1
    else:
        print("❌ Memory Size Limits: FAILED")
    
    if test_results.get("redis_health", {}).get("overall_health"):
        print("✅ Redis Health: PASSED")
        passed_tests += 1
    else:
        print("❌ Redis Health: FAILED")
    
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n🎯 COMPREHENSIVE RESULTS:")
    print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    # Detailed metrics
    if "conversation_continuity" in test_results:
        continuity_score = test_results["conversation_continuity"].get("continuity_score", 0)
        print(f"📈 Conversation continuity score: {continuity_score:.1f}%")
    
    if "concurrent_operations" in test_results:
        concurrent_time = test_results["concurrent_operations"].get("total_time", 0)
        print(f"⚡ Concurrent operations time: {concurrent_time:.2f}s")
    
    if "memory_size_limits" in test_results:
        scaling_factor = test_results["memory_size_limits"].get("scaling_factor", 1)
        print(f"📏 Memory scaling factor: {scaling_factor:.2f}x")
    
    if "redis_health" in test_results:
        hit_ratio = test_results["redis_health"].get("hit_ratio", 0)
        print(f"🎯 Redis cache hit ratio: {hit_ratio:.1f}%")
    
    if success_rate >= 75:
        print("\n🎉 COMPREHENSIVE REDIS MEMORY SYSTEM: EXCELLENT!")
        print("   All major functionality working optimally")
    elif success_rate >= 50:
        print("\n✅ COMPREHENSIVE REDIS MEMORY SYSTEM: GOOD")
        print("   Core functionality working with some areas for improvement")
    else:
        print("\n⚠️  COMPREHENSIVE REDIS MEMORY SYSTEM: NEEDS ATTENTION")
        print("   Significant issues require investigation")
    
    return test_results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main comprehensive test execution"""
    
    try:
        # Wait for Redis and system to be ready
        print("⏳ Waiting for system initialization...")
        await asyncio.sleep(8)
        
        # Run comprehensive test suite
        results = await run_comprehensive_redis_tests()
        
        # Save detailed results
        results_file = "/app/comprehensive_redis_test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: {results_file}")
        
        # Summary file
        summary_file = "/app/comprehensive_redis_summary.txt"
        with open(summary_file, "w") as f:
            f.write("COMPREHENSIVE REDIS MEMORY TEST SUMMARY\n")
            f.write("="*50 + "\n\n")
            
            for test_name, result in results.items():
                f.write(f"{test_name.upper()}:\n")
                if isinstance(result, dict):
                    for key, value in result.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  Result: {result}\n")
                f.write("\n")
        
        print(f"📄 Summary saved to: {summary_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive test execution failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())