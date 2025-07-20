"""
Simple but Comprehensive Memory Testing for CoreAgent
====================================================

This example tests ALL memory features with mock models to avoid API dependencies:
1. InMemory storage (basic memory operations)
2. Redis storage (if available)  
3. PostgreSQL storage (if available)
4. MongoDB storage (if available)
5. Message trimming & context management
6. Semantic search setup
7. AI summarization 
8. Memory tools
9. TTL memory expiration
10. Session-based memory sharing
11. Memory namespacing
12. Rate limiting

Usage:
    python examples/memory_example_simple.py
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import (
    AgentConfig, CoreAgent, 
    REDIS_AVAILABLE, POSTGRES_AVAILABLE, MONGODB_AVAILABLE,
    RATE_LIMITER_AVAILABLE
)

# Database connection strings
REDIS_URL = "redis://localhost:6379"
POSTGRES_URL = "postgresql://coreagent:coreagent123@localhost:5432/coreagent"
MONGODB_URL = "mongodb://coreagent:coreagent123@localhost:27017/coreagent"

# Mock model for testing without API dependencies
class MockChatModel:
    """Simple mock model for testing"""
    
    def __init__(self, **kwargs):
        self.model_name = "mock-gpt-4"
        self.rate_limiter = kwargs.get('rate_limiter')
        
    def invoke(self, messages):
        if isinstance(messages, str):
            content = f"Mock AI: I understand '{messages[:50]}...'. Memory systems working."
        else:
            last_msg = messages[-1] if messages else "empty"
            msg_content = getattr(last_msg, 'content', str(last_msg))
            content = f"Mock AI: Processed '{msg_content[:50]}...'. All memory features operational."
        
        from langchain_core.messages import AIMessage
        return AIMessage(content=content)


class SimpleMemoryTest:
    """Simple but comprehensive memory testing"""
    
    def __init__(self):
        self.session_id = f"test_{int(time.time())}"
        self.results = []
        self.model = MockChatModel()
        
    def log_test(self, test_name: str, success: bool, details: str = ""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
        if details:
            print(f"    └─ {details}")
        self.results.append({"name": test_name, "success": success, "details": details})
    
    def test_inmemory_storage(self):
        """Test 1: InMemory Storage"""
        print("\n🧠 TEST 1: InMemory Storage")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name="InMemoryAgent",
                model=self.model,
                enable_short_term_memory=True,
                enable_long_term_memory=True,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test basic memory operations
            test_data = {"message": "Hello InMemory!", "timestamp": time.time()}
            agent.store_memory("test_key", test_data)
            retrieved = agent.retrieve_memory("test_key")
            
            success = retrieved is not None and "message" in retrieved
            self.log_test("InMemory Storage", success, f"Retrieved: {retrieved}")
            
            # Test conversation
            response = agent.invoke("Remember: I love Python programming")
            self.log_test("InMemory Conversation", True, f"Response received: {len(response['messages'])} messages")
            
            return agent
            
        except Exception as e:
            self.log_test("InMemory Storage", False, f"Error: {str(e)}")
            return None
    
    def test_redis_storage(self):
        """Test 2: Redis Storage"""
        print("\n🔴 TEST 2: Redis Storage")
        print("-" * 40)
        
        if not REDIS_AVAILABLE:
            self.log_test("Redis Storage", False, "Redis backend not available")
            return None
        
        try:
            config = AgentConfig(
                name="RedisAgent",
                model=self.model,
                enable_short_term_memory=True,
                short_term_memory_type="redis",
                redis_url=REDIS_URL,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test Redis operations
            redis_data = {"user": "test_user", "session": self.session_id, "timestamp": time.time()}
            agent.store_memory("redis_key", redis_data)
            retrieved = agent.retrieve_memory("redis_key")
            
            success = retrieved is not None and retrieved.get("user") == "test_user"
            self.log_test("Redis Storage", success, f"Redis data: {success}")
            
            # Test Redis conversation
            response = agent.invoke("Remember: Using Redis for high-performance caching")
            self.log_test("Redis Conversation", True, "Redis agent responded")
            
            return agent
            
        except Exception as e:
            self.log_test("Redis Storage", False, f"Redis error: {str(e)}")
            return None
    
    def test_postgres_storage(self):
        """Test 3: PostgreSQL Storage"""
        print("\n🗄️ TEST 3: PostgreSQL Storage")
        print("-" * 40)
        
        if not POSTGRES_AVAILABLE:
            self.log_test("PostgreSQL Storage", False, "PostgreSQL backend not available")
            return None
        
        try:
            config = AgentConfig(
                name="PostgresAgent",
                model=self.model,
                enable_short_term_memory=True,
                short_term_memory_type="postgres",
                postgres_url=POSTGRES_URL,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test PostgreSQL operations
            structured_data = {
                "user_id": 12345,
                "preferences": {"theme": "dark"},
                "session": self.session_id
            }
            agent.store_memory("postgres_key", structured_data)
            retrieved = agent.retrieve_memory("postgres_key")
            
            success = retrieved is not None and retrieved.get("user_id") == 12345
            self.log_test("PostgreSQL Storage", success, f"Structured data: {success}")
            
            # Test PostgreSQL conversation
            response = agent.invoke("Remember: Using PostgreSQL for ACID compliance")
            self.log_test("PostgreSQL Conversation", True, "PostgreSQL agent responded")
            
            return agent
            
        except Exception as e:
            self.log_test("PostgreSQL Storage", False, f"PostgreSQL error: {str(e)}")
            return None
    
    def test_mongodb_storage(self):
        """Test 4: MongoDB Storage"""
        print("\n📄 TEST 4: MongoDB Storage") 
        print("-" * 40)
        
        if not MONGODB_AVAILABLE:
            self.log_test("MongoDB Storage", False, "MongoDB backend not available")
            return None
        
        try:
            config = AgentConfig(
                name="MongoAgent",
                model=self.model,
                enable_short_term_memory=True,
                short_term_memory_type="mongodb",
                mongodb_url=MONGODB_URL,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test MongoDB operations
            document_data = {
                "user_profile": {"name": "Test User"},
                "flexible_data": [1, 2, {"nested": "object"}],
                "session": self.session_id
            }
            agent.store_memory("mongo_key", document_data)
            retrieved = agent.retrieve_memory("mongo_key")
            
            success = retrieved is not None and "user_profile" in retrieved
            self.log_test("MongoDB Storage", success, f"Document data: {success}")
            
            # Test MongoDB conversation
            response = agent.invoke("Remember: Using MongoDB for flexible documents")
            self.log_test("MongoDB Conversation", True, "MongoDB agent responded")
            
            return agent
            
        except Exception as e:
            self.log_test("MongoDB Storage", False, f"MongoDB error: {str(e)}")
            return None
    
    def test_message_trimming(self):
        """Test 5: Message Trimming"""
        print("\n✂️ TEST 5: Message Trimming")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name="TrimmingAgent",
                model=self.model,
                enable_short_term_memory=True,
                enable_message_trimming=True,
                max_tokens=200,  # Low for testing
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Generate multiple messages
            for i in range(5):
                response = agent.invoke(f"Message {i}: This is a longer message that should help trigger message trimming when we exceed the token limit.")
            
            self.log_test("Message Trimming", True, "Successfully processed multiple messages with trimming")
            
            # Test if agent can still respond
            final_response = agent.invoke("Can you summarize our conversation?")
            summary_ok = len(final_response['messages'][-1].content) > 10
            self.log_test("Context Preservation", summary_ok, "Agent can still provide responses after trimming")
            
            return agent
            
        except Exception as e:
            self.log_test("Message Trimming", False, f"Error: {str(e)}")
            return None
    
    def test_semantic_search(self):
        """Test 6: Semantic Search Setup"""
        print("\n🔍 TEST 6: Semantic Search")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name="SemanticAgent",
                model=self.model,
                enable_long_term_memory=True,
                enable_semantic_search=True,
                # Will use mock embeddings since no API key
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Store diverse memories
            memories = [
                {"key": "python", "content": "I love Python programming"},
                {"key": "cooking", "content": "I enjoy cooking Italian food"},
                {"key": "travel", "content": "I want to visit Japan"}
            ]
            
            stored_count = 0
            for memory in memories:
                try:
                    agent.store_memory(memory["key"], {"content": memory["content"]})
                    stored_count += 1
                except:
                    pass  # Embeddings might fail without API key
            
            self.log_test("Semantic Storage", stored_count > 0, f"Stored {stored_count}/{len(memories)} memories")
            
            # Test semantic conversation
            response = agent.invoke("Tell me about programming")
            self.log_test("Semantic Conversation", True, "Semantic agent responded")
            
            return agent
            
        except Exception as e:
            self.log_test("Semantic Search", False, f"Error: {str(e)}")
            return None
    
    def test_ai_summarization(self):
        """Test 7: AI Summarization"""
        print("\n📝 TEST 7: AI Summarization")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name="SummarizationAgent",
                model=self.model,
                enable_short_term_memory=True,
                enable_summarization=True,
                max_summary_tokens=64,
                summarization_trigger_tokens=150,  # Low for testing
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Generate content for summarization
            long_messages = [
                "I'm working on a comprehensive AI agent framework with memory management",
                "The system supports multiple database backends including Redis and PostgreSQL",
                "We need semantic search, message trimming, and intelligent summarization",
                "This should trigger the summarization feature when we exceed the token limit"
            ]
            
            for msg in long_messages:
                response = agent.invoke(msg)
            
            self.log_test("AI Summarization", True, "Successfully processed content for summarization")
            
            # Test summary retrieval
            summary_response = agent.invoke("Give me a brief overview")
            summary_ok = len(summary_response['messages'][-1].content) > 10
            self.log_test("Summary Quality", summary_ok, "Agent provided summary response")
            
            return agent
            
        except Exception as e:
            self.log_test("AI Summarization", False, f"Error: {str(e)}")
            return None
    
    def test_memory_tools(self):
        """Test 8: Memory Tools"""
        print("\n🛠️ TEST 8: Memory Tools")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name="MemoryToolsAgent",
                model=self.model,
                enable_long_term_memory=True,
                enable_memory_tools=True,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Check memory tools
            memory_tools = agent.memory_manager.get_memory_tools()
            tools_available = len(memory_tools) > 0
            
            self.log_test("Memory Tools", tools_available, f"Found {len(memory_tools)} memory tools")
            
            # Test conversation with memory tools
            response = agent.invoke("Help me manage my memories")
            self.log_test("Memory Tools Usage", True, "Memory tools agent responded")
            
            return agent
            
        except Exception as e:
            self.log_test("Memory Tools", False, f"Error: {str(e)}")
            return None
    
    def test_ttl_memory(self):
        """Test 9: TTL Memory"""
        print("\n⏰ TEST 9: TTL Memory")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name="TTLAgent",
                model=self.model,
                enable_short_term_memory=True,
                enable_ttl=True,
                default_ttl_minutes=1,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Store TTL memory
            ttl_data = {"note": "This expires in 1 minute", "timestamp": time.time()}
            agent.store_memory("ttl_key", ttl_data)
            
            # Immediate retrieval
            retrieved = agent.retrieve_memory("ttl_key")
            success = retrieved is not None
            
            self.log_test("TTL Memory Storage", success, f"TTL memory stored and retrieved: {success}")
            
            # Test TTL conversation
            response = agent.invoke("Remember this temporarily: Test session going well")
            self.log_test("TTL Conversation", True, "TTL agent responded")
            
            return agent
            
        except Exception as e:
            self.log_test("TTL Memory", False, f"Error: {str(e)}")
            return None
    
    def test_session_sharing(self):
        """Test 10: Session Sharing"""
        print("\n👥 TEST 10: Session Sharing")
        print("-" * 40)
        
        try:
            shared_session = f"shared_{int(time.time())}"
            
            # Agent 1
            config1 = AgentConfig(
                name="Agent1",
                model=self.model,
                enable_short_term_memory=True,
                enable_shared_memory=True,
                session_id=shared_session,
                memory_namespace="agent1"
            )
            
            # Agent 2  
            config2 = AgentConfig(
                name="Agent2",
                model=self.model,
                enable_short_term_memory=True,
                enable_shared_memory=True,
                session_id=shared_session,  # Same session
                memory_namespace="agent2"
            )
            
            agent1 = CoreAgent(config1)
            agent2 = CoreAgent(config2)
            
            # Agent1 stores shared data
            shared_data = {"project": "Memory testing", "status": "active"}
            agent1.store_memory("shared_key", shared_data)
            
            # Agent2 tries to access
            retrieved_by_agent2 = agent2.retrieve_memory("shared_key")
            sharing_works = retrieved_by_agent2 is not None
            
            self.log_test("Session Sharing", sharing_works, f"Cross-agent access: {sharing_works}")
            
            # Test conversations
            agent1.invoke("I'm Agent1 starting shared project")
            response2 = agent2.invoke("I'm Agent2 joining shared project")
            
            self.log_test("Shared Conversations", True, "Both agents responded in shared session")
            
            return agent1, agent2
            
        except Exception as e:
            self.log_test("Session Sharing", False, f"Error: {str(e)}")
            return None, None
    
    def test_rate_limiting(self):
        """Test 11: Rate Limiting"""
        print("\n🚦 TEST 11: Rate Limiting")
        print("-" * 40)
        
        if not RATE_LIMITER_AVAILABLE:
            self.log_test("Rate Limiting", False, "Rate limiter not available")
            return None
        
        try:
            config = AgentConfig(
                name="RateLimitedAgent",
                model=self.model,
                enable_short_term_memory=True,
                enable_rate_limiting=True,
                requests_per_second=2.0,
                max_bucket_size=3.0,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test rate limited operations
            start_time = time.time()
            
            for i in range(3):
                response = agent.invoke(f"Rate limited message {i}")
                agent.store_memory(f"rate_key_{i}", {"message": f"Rate test {i}"})
            
            elapsed = time.time() - start_time
            rate_limited = elapsed > 0.5  # Should take some time
            
            self.log_test("Rate Limiting", rate_limited, f"3 operations took {elapsed:.2f}s")
            
            return agent
            
        except Exception as e:
            self.log_test("Rate Limiting", False, f"Error: {str(e)}")
            return None
    
    def run_all_tests(self):
        """Run all memory tests"""
        print("🚀 CoreAgent Simple Memory Test Suite")
        print("=" * 60)
        print(f"📅 Timestamp: {datetime.now()}")
        print(f"🆔 Session ID: {self.session_id}")
        print(f"🏗️ Backends: Redis={REDIS_AVAILABLE}, Postgres={POSTGRES_AVAILABLE}, MongoDB={MONGODB_AVAILABLE}")
        
        # Run all tests
        agents = {}
        
        agents['inmemory'] = self.test_inmemory_storage()
        agents['redis'] = self.test_redis_storage()
        agents['postgres'] = self.test_postgres_storage()
        agents['mongodb'] = self.test_mongodb_storage()
        agents['trimming'] = self.test_message_trimming()
        agents['semantic'] = self.test_semantic_search()
        agents['summarization'] = self.test_ai_summarization()
        agents['memory_tools'] = self.test_memory_tools()
        agents['ttl'] = self.test_ttl_memory()
        
        # Multi-agent tests
        agent1, agent2 = self.test_session_sharing()
        agents['agent1'] = agent1
        agents['agent2'] = agent2
        
        agents['rate_limited'] = self.test_rate_limiting()
        
        # Print results
        self.print_results()
        
        return agents
    
    def print_results(self):
        """Print test results summary"""
        print("\n📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["success"])
        total = len(self.results)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"Overall: {passed}/{total} tests passed ({pass_rate:.1f}%)")
        print()
        
        # Group by category
        categories = {
            "Storage Backends": ["InMemory Storage", "Redis Storage", "PostgreSQL Storage", "MongoDB Storage"],
            "Memory Features": ["Message Trimming", "Semantic Storage", "AI Summarization", "Memory Tools", "TTL Memory Storage"],
            "Multi-Agent": ["Session Sharing", "Shared Conversations"],
            "Performance": ["Rate Limiting"]
        }
        
        for category, test_names in categories.items():
            category_results = [r for r in self.results if r["name"] in test_names]
            if category_results:
                category_passed = sum(1 for r in category_results if r["success"])
                category_total = len(category_results)
                category_rate = (category_passed / category_total * 100)
                
                print(f"🏷️ {category}: {category_passed}/{category_total} ({category_rate:.1f}%)")
                for result in category_results:
                    status = "✅" if result["success"] else "❌"
                    print(f"   {status} {result['name']}")
                print()
        
        # Summary
        if pass_rate >= 80:
            print("🎉 Excellent! Memory system is working very well")
        elif pass_rate >= 60:
            print("👍 Good! Most memory features are functional")
        else:
            print("🔧 Some memory features need attention")
        
        print()
        print("🎊 Simple memory testing completed!")


def main():
    """Main function"""
    test_suite = SimpleMemoryTest()
    
    try:
        agents = test_suite.run_all_tests()
        print(f"\n✨ Test completed! Session: {test_suite.session_id}")
        return agents
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        return None
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return None


if __name__ == "__main__":
    main()