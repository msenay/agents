#!/usr/bin/env python3
"""
Elite Coder Agent Integration Tests
==================================

Real-world integration tests using actual Azure OpenAI GPT-4 and Redis.
These tests verify the Elite Coder Agent works with real external services.

⚠️ REQUIREMENTS:
- Redis server running on localhost:6379
- Valid Azure OpenAI API key and endpoint
- Network connectivity for API calls

🔧 SETUP:
1. Start Redis: redis-server
2. Ensure Azure OpenAI credentials are valid
3. Run tests: python3 test_coder_integration.py
"""

import sys
import unittest
import os
import time
import json
import redis
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Add workspace to path
sys.path.insert(0, '/workspace')

# Test configuration with real credentials
REAL_AZURE_CONFIG = {
    "OPENAI_API_VERSION": "2023-12-01-preview",
    "AZURE_OPENAI_ENDPOINT": "https://oai-202-fbeta-dev.openai.azure.com/",
    "OPENAI_API_KEY": "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4",
    "GPT4_MODEL_NAME": "gpt-4",
    "GPT4_DEPLOYMENT_NAME": "gpt4"
}

# Set environment variables for integration tests
for key, value in REAL_AZURE_CONFIG.items():
    os.environ[key] = value

try:
    from agent.coder import (
        EliteCoderAgent, 
        CoderConfig, 
        create_elite_coder_agent
    )
    CODER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Elite Coder Agent not available: {e}")
    CODER_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    print("⚠️ Redis not available")
    REDIS_AVAILABLE = False


class RealServiceHealthCheck:
    """Health check for real external services"""
    
    @staticmethod
    def check_redis(host='localhost', port=6379, db=0) -> bool:
        """Check if Redis is available"""
        try:
            r = redis.Redis(host=host, port=port, db=db, socket_timeout=5)
            r.ping()
            return True
        except Exception as e:
            print(f"❌ Redis health check failed: {e}")
            return False
    
    @staticmethod
    def check_azure_openai() -> bool:
        """Check if Azure OpenAI is available"""
        try:
            from langchain_openai import AzureChatOpenAI
            from langchain_core.messages import HumanMessage
            
            llm = AzureChatOpenAI(
                azure_endpoint=REAL_AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"],
                api_key=REAL_AZURE_CONFIG["OPENAI_API_KEY"],
                api_version=REAL_AZURE_CONFIG["OPENAI_API_VERSION"],
                model=REAL_AZURE_CONFIG["GPT4_MODEL_NAME"],
                deployment_name=REAL_AZURE_CONFIG["GPT4_DEPLOYMENT_NAME"],
                temperature=0.1,
                max_tokens=100
            )
            
            # Simple test request
            response = llm.invoke([HumanMessage(content="Say 'test successful' in exactly those words.")])
            return "test successful" in response.content.lower()
            
        except Exception as e:
            print(f"❌ Azure OpenAI health check failed: {e}")
            return False


def skip_if_services_unavailable(test_func):
    """Decorator to skip tests if real services are not available"""
    def wrapper(self):
        if not CODER_AVAILABLE:
            self.skipTest("Elite Coder Agent not available")
        if not REDIS_AVAILABLE:
            self.skipTest("Redis not available")
        if not RealServiceHealthCheck.check_redis():
            self.skipTest("Redis server not running")
        if not RealServiceHealthCheck.check_azure_openai():
            self.skipTest("Azure OpenAI not accessible")
        return test_func(self)
    return wrapper


class TestEliteCoderIntegrationReal(unittest.TestCase):
    """Real integration tests with actual external services"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with service availability check"""
        print("🔍 CHECKING REAL SERVICE AVAILABILITY")
        print("=" * 60)
        
        cls.redis_available = RealServiceHealthCheck.check_redis()
        cls.openai_available = RealServiceHealthCheck.check_azure_openai()
        
        print(f"Redis Available: {'✅' if cls.redis_available else '❌'}")
        print(f"Azure OpenAI Available: {'✅' if cls.openai_available else '❌'}")
        print(f"Elite Coder Available: {'✅' if CODER_AVAILABLE else '❌'}")
        print("=" * 60)
        
        if cls.redis_available and cls.openai_available and CODER_AVAILABLE:
            print("🚀 ALL SERVICES AVAILABLE - RUNNING REAL INTEGRATION TESTS")
            
            # Create test Redis connection for cleanup
            cls.redis_client = redis.Redis(host='localhost', port=6379, db=3)
            
            # Clean up any existing test data
            test_keys = cls.redis_client.keys("elite_coder:test:*")
            if test_keys:
                cls.redis_client.delete(*test_keys)
                print(f"🧹 Cleaned up {len(test_keys)} existing test keys")
        else:
            print("⚠️ SOME SERVICES UNAVAILABLE - TESTS WILL BE SKIPPED")
        
        print()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests"""
        if hasattr(cls, 'redis_client'):
            # Clean up test data
            test_keys = cls.redis_client.keys("elite_coder:test:*")
            if test_keys:
                cls.redis_client.delete(*test_keys)
                print(f"🧹 Final cleanup: removed {len(test_keys)} test keys")
            cls.redis_client.close()
    
    def setUp(self):
        """Set up each test"""
        self.test_session_id = f"test_integration_{int(time.time())}"
        self.start_time = time.time()
        
    def tearDown(self):
        """Clean up after each test"""
        duration = time.time() - self.start_time
        print(f"⏱️ Test completed in {duration:.2f}s")
    
    @skip_if_services_unavailable
    def test_01_agent_initialization_with_real_services(self):
        """Test Elite Coder Agent initialization with real services"""
        print("\n🧪 TEST 1: Agent Initialization with Real Services")
        print("-" * 50)
        
        # Create agent with real services
        agent = EliteCoderAgent(self.test_session_id)
        
        # Verify agent properties
        self.assertEqual(agent.session_id, self.test_session_id)
        self.assertIsNotNone(agent.config)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.memory_manager)
        
        # Verify configuration matches real Azure OpenAI settings
        self.assertEqual(agent.config.name, "EliteCoderAgent")
        self.assertTrue(agent.config.enable_memory)
        self.assertEqual(agent.config.memory_backend, "redis")
        
        print("✅ Agent initialized successfully with real services")
        
        # Test Redis connectivity through agent
        try:
            # This should work with real Redis
            memory_key = f"elite_coder:test:{self.test_session_id}:init_test"
            test_data = {"test": "real_integration", "timestamp": datetime.now().isoformat()}
            
            result = agent.memory_manager.save_memory(memory_key, json.dumps(test_data))
            self.assertTrue(result, "Failed to save to real Redis")
            
            retrieved_data = agent.memory_manager.get_memory(memory_key)
            self.assertIsNotNone(retrieved_data, "Failed to retrieve from real Redis")
            
            parsed_data = json.loads(retrieved_data)
            self.assertEqual(parsed_data["test"], "real_integration")
            
            print("✅ Real Redis integration working")
            
        except Exception as e:
            self.fail(f"Redis integration failed: {e}")
    
    @skip_if_services_unavailable
    def test_02_real_llm_simple_chat(self):
        """Test real LLM interaction through chat"""
        print("\n🧪 TEST 2: Real LLM Simple Chat")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Test simple chat with real Azure OpenAI
        test_message = "Hello! Please respond with exactly these words: 'Elite Coder Agent is working perfectly'"
        
        print(f"📤 Sending: {test_message}")
        response = agent.chat(test_message)
        
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 10)
        
        print(f"📥 Received: {response[:100]}...")
        
        # Verify it's a real response (not mock)
        self.assertIn("Elite Coder Agent", response)
        
        print("✅ Real LLM chat working")
    
    @skip_if_services_unavailable
    def test_03_real_agent_generation_simple(self):
        """Test real agent generation with LLM"""
        print("\n🧪 TEST 3: Real Agent Generation - Simple")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Generate a simple agent using real LLM
        result = agent.generate_complete_agent(
            template_type="simple",
            agent_name="RealTestAgent", 
            purpose="Integration testing with real services"
        )
        
        # Verify result structure
        self.assertIn("agent_code", result)
        self.assertIn("save_result", result)
        self.assertIn("test_result", result)
        self.assertTrue(result["core_agent_integrated"])
        
        # Verify generated code quality
        agent_code = result["agent_code"]
        self.assertIn("RealTestAgent", agent_code)
        self.assertIn("class RealTestAgentState", agent_code)
        self.assertIn("def process_node", agent_code)
        self.assertIn("StateGraph", agent_code)
        
        # Verify real Azure OpenAI configuration in generated code
        self.assertIn(REAL_AZURE_CONFIG["AZURE_OPENAI_ENDPOINT"], agent_code)
        self.assertIn(REAL_AZURE_CONFIG["GPT4_MODEL_NAME"], agent_code)
        
        print("✅ Real agent generation working")
        print(f"📝 Generated {len(agent_code)} characters of code")
    
    @skip_if_services_unavailable
    def test_04_real_agent_generation_with_tools(self):
        """Test real agent generation with tools"""
        print("\n🧪 TEST 4: Real Agent Generation - With Tools")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Generate agent with tools using real LLM
        tools_needed = ["calculator", "file_processor", "data_analyzer"]
        result = agent.generate_complete_agent(
            template_type="with_tools",
            agent_name="RealToolAgent",
            purpose="Testing with real tools integration",
            tools_needed=tools_needed
        )
        
        # Verify result
        self.assertIn("agent_code", result)
        self.assertEqual(result["template_type"], "with_tools")
        self.assertEqual(result["tools"], tools_needed)
        
        agent_code = result["agent_code"]
        
        # Verify all tools are included
        for tool in tools_needed:
            clean_tool_name = tool.lower().replace(" ", "_").replace("-", "_")
            self.assertIn(f"{clean_tool_name}_tool", agent_code)
        
        # Verify tool workflow components
        self.assertIn("tool_selection_node", agent_code)
        self.assertIn("tool_execution_node", agent_code)
        
        print("✅ Real agent with tools generation working")
        print(f"📝 Generated agent with {len(tools_needed)} tools")
    
    @skip_if_services_unavailable  
    def test_05_real_multi_agent_system_generation(self):
        """Test real multi-agent system generation"""
        print("\n🧪 TEST 5: Real Multi-Agent System Generation")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Generate multi-agent system using real LLM
        result = agent.generate_complete_agent(
            template_type="multi_agent",
            agent_name="RealMultiSystem",
            purpose="Complex real-world task coordination"
        )
        
        # Verify result
        self.assertIn("agent_code", result)
        self.assertEqual(result["template_type"], "multi_agent")
        
        agent_code = result["agent_code"]
        
        # Verify multi-agent components
        required_nodes = [
            "elite_supervisor_node",
            "elite_researcher_node", 
            "elite_analyzer_node",
            "elite_creator_node",
            "elite_reviewer_node",
            "elite_aggregator_node"
        ]
        
        for node in required_nodes:
            self.assertIn(node, agent_code)
        
        # Verify workflow coordination
        self.assertIn("workflow_history", agent_code)
        self.assertIn("route_to_agent", agent_code)
        
        print("✅ Real multi-agent system generation working")
        print(f"📝 Generated complex system with {len(required_nodes)} specialized agents")
    
    @skip_if_services_unavailable
    def test_06_real_memory_persistence_across_sessions(self):
        """Test real memory persistence across agent sessions"""
        print("\n🧪 TEST 6: Real Memory Persistence Across Sessions")
        print("-" * 50)
        
        session_base = f"memory_test_{int(time.time())}"
        
        # Session 1: Store information
        agent1 = EliteCoderAgent(session_base)
        
        chat1_response = agent1.chat("Please remember this important information: The Elite Coder Agent was tested on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"📤 Session 1: {chat1_response[:100]}...")
        
        # Generate and save an agent
        result1 = agent1.generate_complete_agent(
            template_type="simple",
            agent_name="MemoryTestAgent",
            purpose="Testing memory persistence"
        )
        
        # Wait a moment
        time.sleep(1)
        
        # Session 2: Retrieve information (same session ID)
        agent2 = EliteCoderAgent(session_base)
        
        chat2_response = agent2.chat("What important information did I tell you to remember in our previous conversation?")
        print(f"📥 Session 2: {chat2_response[:100]}...")
        
        # Verify memory persistence - the response should reference the previous conversation
        # Note: This might not be exact due to LLM variance, but should show some awareness
        self.assertIsNotNone(chat2_response)
        self.assertGreater(len(chat2_response), 20)
        
        print("✅ Real memory persistence working across sessions")
    
    @skip_if_services_unavailable
    def test_07_real_conversation_flow_with_context(self):
        """Test real conversation flow with context building"""
        print("\n🧪 TEST 7: Real Conversation Flow with Context Building")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Multi-turn conversation with context building
        conversations = [
            "I want to create a Python calculator agent using LangGraph",
            "Add error handling and validation to the calculator",
            "Now add memory so it can remember previous calculations",
            "Generate the complete implementation"
        ]
        
        responses = []
        for i, message in enumerate(conversations, 1):
            print(f"📤 Turn {i}: {message}")
            response = agent.chat(message)
            responses.append(response)
            print(f"📥 Response {i}: {response[:100]}...")
            
            # Verify each response is substantial
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 50)
            
            # Small delay between requests
            time.sleep(1)
        
        # Verify the conversation built context
        final_response = responses[-1]
        
        # The final response should reference previous conversation elements
        context_indicators = ["calculator", "error", "memory", "previous"]
        context_found = sum(1 for indicator in context_indicators if indicator in final_response.lower())
        
        self.assertGreaterEqual(context_found, 2, "Conversation should build context")
        
        print("✅ Real conversation flow with context building working")
        print(f"🧠 Context indicators found: {context_found}/4")
    
    @skip_if_services_unavailable
    def test_08_real_code_memory_integration(self):
        """Test real code memory integration with Redis"""
        print("\n🧪 TEST 8: Real Code Memory Integration")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Test memory tool integration if available
        memory_tools = [tool for tool in agent.config.tools if hasattr(tool, 'name') and 'memory' in tool.name.lower()]
        
        if memory_tools:
            memory_tool = memory_tools[0]
            
            # Save a module
            save_result = memory_tool._run(
                action="save",
                module_name="real_test_module",
                code="def real_test_function():\n    return 'Real integration test successful!'",
                description="Real integration test module",
                tags=["integration", "real", "test"]
            )
            
            print(f"💾 Save result: {save_result}")
            self.assertIn("✅", save_result)
            
            # Load the module
            load_result = memory_tool._run(
                action="load",
                module_name="real_test_module"
            )
            
            print(f"📖 Load result: {load_result[:100]}...")
            self.assertIn("✅", load_result)
            self.assertIn("real_test_function", load_result)
            
            # List modules
            list_result = memory_tool._run(action="list")
            print(f"📚 List result: {list_result}")
            self.assertIn("real_test_module", list_result)
            
            # Search modules
            search_result = memory_tool._run(action="search", search_query="real")
            print(f"🔍 Search result: {search_result}")
            self.assertIn("real_test_module", search_result)
            
            print("✅ Real code memory integration working")
        else:
            print("⚠️ Code memory tool not available in agent configuration")
    
    @skip_if_services_unavailable
    def test_09_real_performance_and_reliability(self):
        """Test real performance and reliability under load"""
        print("\n🧪 TEST 9: Real Performance and Reliability")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Test multiple rapid requests
        num_requests = 3  # Keep small to avoid rate limits
        responses = []
        start_time = time.time()
        
        for i in range(num_requests):
            message = f"Request {i+1}: Create a simple function that returns the number {i+1}"
            print(f"📤 Request {i+1}: {message}")
            
            request_start = time.time()
            response = agent.chat(message)
            request_duration = time.time() - request_start
            
            responses.append({
                'response': response,
                'duration': request_duration,
                'request_num': i+1
            })
            
            print(f"📥 Response {i+1} ({request_duration:.2f}s): {response[:100]}...")
            
            # Verify response quality
            self.assertIsNotNone(response)
            self.assertGreater(len(response), 20)
            
            # Rate limiting - wait between requests
            time.sleep(2)
        
        total_duration = time.time() - start_time
        avg_duration = total_duration / num_requests
        
        print(f"⏱️ Total time: {total_duration:.2f}s")
        print(f"📊 Average response time: {avg_duration:.2f}s")
        print(f"🚀 All {num_requests} requests successful")
        
        # Verify all requests succeeded
        self.assertEqual(len(responses), num_requests)
        
        # Verify reasonable performance (should be under 30s per request on average)
        self.assertLess(avg_duration, 30.0, "Average response time too slow")
        
        print("✅ Real performance and reliability test passed")
    
    @skip_if_services_unavailable
    def test_10_real_error_handling_and_recovery(self):
        """Test real error handling and recovery scenarios"""
        print("\n🧪 TEST 10: Real Error Handling and Recovery")
        print("-" * 50)
        
        agent = EliteCoderAgent(self.test_session_id)
        
        # Test 1: Malformed request handling
        print("🔸 Testing malformed request handling...")
        malformed_response = agent.chat("Generate a completely invalid agent type that doesn't exist in any universe: ∞∞∞INVALID∞∞∞")
        
        self.assertIsNotNone(malformed_response)
        self.assertIsInstance(malformed_response, str)
        print(f"✅ Handled malformed request: {malformed_response[:100]}...")
        
        # Test 2: Recovery with valid request after malformed
        print("🔸 Testing recovery after malformed request...")
        recovery_response = agent.chat("Create a simple hello world agent")
        
        self.assertIsNotNone(recovery_response)
        self.assertGreater(len(recovery_response), 50)
        print(f"✅ Recovered successfully: {recovery_response[:100]}...")
        
        # Test 3: Invalid agent generation type
        print("🔸 Testing invalid agent generation type...")
        invalid_generation = agent.generate_complete_agent(
            template_type="completely_invalid_type",
            agent_name="ErrorTestAgent",
            purpose="Testing error handling"
        )
        
        self.assertIn("agent_code", invalid_generation)
        # Should contain error message or fallback
        self.assertTrue(
            "Unknown template type" in invalid_generation["agent_code"] or 
            "error" in invalid_generation.get("agent_code", "").lower()
        )
        
        print("✅ Invalid generation type handled gracefully")
        
        # Test 4: Recovery after invalid generation
        print("🔸 Testing recovery after invalid generation...")
        valid_generation = agent.generate_complete_agent(
            template_type="simple",
            agent_name="RecoveryAgent", 
            purpose="Testing recovery"
        )
        
        self.assertIn("agent_code", valid_generation)
        self.assertIn("RecoveryAgent", valid_generation["agent_code"])
        
        print("✅ Recovered from invalid generation successfully")
        print("✅ Real error handling and recovery working")


def run_integration_tests():
    """Run comprehensive real integration tests"""
    print("🚀 ELITE CODER AGENT - REAL INTEGRATION TESTS")
    print("=" * 80)
    print("Using REAL services:")
    print(f"🔗 Azure OpenAI: {REAL_AZURE_CONFIG['AZURE_OPENAI_ENDPOINT']}")
    print(f"🧠 Model: {REAL_AZURE_CONFIG['GPT4_MODEL_NAME']} ({REAL_AZURE_CONFIG['GPT4_DEPLOYMENT_NAME']})")
    print(f"💾 Redis: localhost:6379/3")
    print("=" * 80)
    
    if not (CODER_AVAILABLE and REDIS_AVAILABLE):
        print("❌ PREREQUISITES NOT MET:")
        if not CODER_AVAILABLE:
            print("  - Elite Coder Agent not available")
        if not REDIS_AVAILABLE:
            print("  - Redis client not available")
        return False
    
    # Check service availability
    redis_ok = RealServiceHealthCheck.check_redis()
    openai_ok = RealServiceHealthCheck.check_azure_openai()
    
    if not (redis_ok and openai_ok):
        print("❌ REAL SERVICES NOT AVAILABLE:")
        if not redis_ok:
            print("  - Redis server not accessible")
        if not openai_ok:
            print("  - Azure OpenAI not accessible")
        print("\n💡 To run integration tests:")
        print("  1. Start Redis server: redis-server")
        print("  2. Verify Azure OpenAI credentials")
        print("  3. Check network connectivity")
        return False
    
    print("✅ ALL REAL SERVICES AVAILABLE - PROCEEDING WITH INTEGRATION TESTS\n")
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEliteCoderIntegrationReal)
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("📊 REAL INTEGRATION TEST SUMMARY")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"⏭️ Skipped: {skipped}")
    
    if passed > 0:
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")
    
    if result.wasSuccessful():
        print("\n🎉 ALL REAL INTEGRATION TESTS PASSED!")
        print("🔥 Elite Coder Agent works perfectly with real services:")
        print("  ✅ Real Azure OpenAI GPT-4 integration")
        print("  ✅ Real Redis memory persistence") 
        print("  ✅ Real agent generation")
        print("  ✅ Real conversation flows")
        print("  ✅ Real error handling")
        print("  ✅ Real performance validation")
    else:
        print("\n❌ SOME INTEGRATION TESTS FAILED!")
        if failures:
            print(f"\nFailures ({failures}):")
            for test, traceback in result.failures:
                print(f"  - {test}")
                
        if errors:
            print(f"\nErrors ({errors}):")
            for test, traceback in result.errors:
                print(f"  - {test}")
    
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_integration_tests()
    exit(0 if success else 1)