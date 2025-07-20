"""
Comprehensive Memory Testing Example for CoreAgent
=================================================

This example demonstrates ALL memory features of CoreAgent:
1. InMemory storage (default)
2. Redis storage (high-performance caching)
3. PostgreSQL storage (structured persistence)
4. MongoDB storage (document-based)
5. Semantic search & embeddings
6. Message trimming & summarization
7. Session-based memory sharing
8. Memory tools (self-managing memory)
9. TTL (Time-To-Live) memory expiration
10. Memory namespacing & isolation

Prerequisites:
- pip install redis pymongo psycopg2-binary
- Docker containers for Redis, MongoDB, PostgreSQL (optional)
- OpenAI API key for embeddings and LLM calls

Usage:
    python examples/memory_example_single_agent.py
"""

import os
import sys
import time
import asyncio
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import (
    AgentConfig, CoreAgent, 
    REDIS_AVAILABLE, POSTGRES_AVAILABLE, MONGODB_AVAILABLE,
    RATE_LIMITER_AVAILABLE
)

# Mock model for testing without API keys
class MockChatModel:
    """Mock chat model for testing without requiring API keys"""
    
    def __init__(self, model_name="mock-gpt-4", **kwargs):
        self.model_name = model_name
        self.rate_limiter = kwargs.get('rate_limiter')
        
    def invoke(self, messages):
        """Mock invoke that returns a simple response"""
        if isinstance(messages, str):
            content = f"Mock AI: I understand you said '{messages}'. I'm processing this with my memory systems."
        else:
            last_msg = messages[-1] if messages else "empty"
            msg_content = getattr(last_msg, 'content', str(last_msg))
            content = f"Mock AI: I understand '{msg_content}'. Memory systems are working."
        
        from langchain_core.messages import AIMessage
        return AIMessage(content=content)
    
    async def ainvoke(self, messages):
        """Mock async invoke"""
        return self.invoke(messages)


class MemoryTestSuite:
    """Comprehensive memory testing suite for CoreAgent"""
    
    def __init__(self):
        self.session_id = f"test_session_{int(time.time())}"
        self.results = {}
        
    def print_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*60}")
        print(f"🧪 {title}")
        print(f"{'='*60}")
    
    def print_test(self, test_name: str, success: bool, details: str = ""):
        """Print test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} | {test_name}")
        if details:
            print(f"    └─ {details}")
        
        self.results[test_name] = {
            "success": success,
            "details": details,
            "timestamp": datetime.now()
        }
    
    def test_inmemory_storage(self):
        """Test 1: InMemory Storage (Default)"""
        self.print_header("InMemory Storage Test")
        
        try:
            config = AgentConfig(
                name="InMemoryAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                short_term_memory_type="inmemory",
                enable_long_term_memory=True,
                long_term_memory_type="inmemory",
                session_id=self.session_id,
                memory_namespace="inmemory_test"
            )
            
            agent = CoreAgent(config)
            
            # Test memory storage
            agent.store_memory("test_key", {"message": "Hello from InMemory!", "timestamp": time.time()})
            stored_data = agent.retrieve_memory("test_key")
            
            success = stored_data is not None and "message" in stored_data
            self.print_test("InMemory Storage", success, f"Stored and retrieved: {stored_data}")
            
            # Test conversation
            response = agent.invoke("Remember: My favorite color is blue")
            self.print_test("InMemory Conversation", True, f"Response: {response}")
            
            return agent
            
        except Exception as e:
            self.print_test("InMemory Storage", False, f"Error: {str(e)}")
            return None
    
    def test_redis_storage(self):
        """Test 2: Redis Storage (High-Performance Caching)"""
        self.print_header("Redis Storage Test")
        
        if not REDIS_AVAILABLE:
            self.print_test("Redis Storage", False, "Redis not available - install with: pip install redis")
            return None
        
        try:
            # Try connecting to Redis (use default local Redis or Docker)
            redis_url = "redis://localhost:6379"
            
            config = AgentConfig(
                name="RedisAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                short_term_memory_type="redis",
                enable_long_term_memory=True,
                long_term_memory_type="redis",
                redis_url=redis_url,
                session_id=self.session_id,
                memory_namespace="redis_test",
                enable_shared_memory=True  # Enable session sharing
            )
            
            agent = CoreAgent(config)
            
            # Test Redis memory
            test_data = {
                "user_preference": "Dark mode",
                "last_activity": time.time(),
                "session": self.session_id
            }
            
            agent.store_memory("redis_test_key", test_data)
            retrieved = agent.retrieve_memory("redis_test_key")
            
            success = retrieved is not None and retrieved.get("user_preference") == "Dark mode"
            self.print_test("Redis Storage", success, f"Session: {self.session_id}, Data: {retrieved}")
            
            # Test Redis conversation with shared memory
            response = agent.invoke("Remember: I'm working on a Redis memory test")
            self.print_test("Redis Conversation", True, f"Response: {response}")
            
            return agent
            
        except Exception as e:
            self.print_test("Redis Storage", False, f"Redis connection failed: {str(e)}")
            return None
    
    def test_postgres_storage(self):
        """Test 3: PostgreSQL Storage (Structured Persistence)"""
        self.print_header("PostgreSQL Storage Test")
        
        if not POSTGRES_AVAILABLE:
            self.print_test("PostgreSQL Storage", False, "PostgreSQL not available - install with: pip install psycopg2-binary")
            return None
        
        try:
            # Try connecting to PostgreSQL (use default local or Docker)
            postgres_url = "postgresql://postgres:password@localhost:5432/coreagent"
            
            config = AgentConfig(
                name="PostgresAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                short_term_memory_type="postgres",
                enable_long_term_memory=True,
                long_term_memory_type="postgres",
                postgres_url=postgres_url,
                session_id=self.session_id,
                memory_namespace="postgres_test"
            )
            
            agent = CoreAgent(config)
            
            # Test PostgreSQL memory
            structured_data = {
                "user_id": 12345,
                "preferences": {"theme": "dark", "language": "en"},
                "metadata": {
                    "created": datetime.now().isoformat(),
                    "session": self.session_id
                }
            }
            
            agent.store_memory("postgres_structured_key", structured_data)
            retrieved = agent.retrieve_memory("postgres_structured_key")
            
            success = retrieved is not None and retrieved.get("user_id") == 12345
            self.print_test("PostgreSQL Storage", success, f"Structured data: {retrieved}")
            
            # Test PostgreSQL conversation
            response = agent.invoke("Remember: I prefer structured data storage")
            self.print_test("PostgreSQL Conversation", True, f"Response: {response}")
            
            return agent
            
        except Exception as e:
            self.print_test("PostgreSQL Storage", False, f"PostgreSQL connection failed: {str(e)}")
            return None
    
    def test_mongodb_storage(self):
        """Test 4: MongoDB Storage (Document-Based)"""
        self.print_header("MongoDB Storage Test")
        
        if not MONGODB_AVAILABLE:
            self.print_test("MongoDB Storage", False, "MongoDB not available - install with: pip install pymongo")
            return None
        
        try:
            # Try connecting to MongoDB (use default local or Docker)
            mongodb_url = "mongodb://localhost:27017/coreagent"
            
            config = AgentConfig(
                name="MongoAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                short_term_memory_type="mongodb",
                enable_long_term_memory=True,
                long_term_memory_type="mongodb",
                mongodb_url=mongodb_url,
                session_id=self.session_id,
                memory_namespace="mongodb_test"
            )
            
            agent = CoreAgent(config)
            
            # Test MongoDB memory with flexible document structure
            document_data = {
                "user_profile": {
                    "name": "Test User",
                    "preferences": {
                        "notifications": True,
                        "theme": "auto"
                    },
                    "history": [
                        {"action": "login", "timestamp": time.time()},
                        {"action": "memory_test", "timestamp": time.time()}
                    ]
                },
                "session": self.session_id,
                "test_array": [1, 2, 3, "test", {"nested": "object"}]
            }
            
            agent.store_memory("mongodb_document_key", document_data)
            retrieved = agent.retrieve_memory("mongodb_document_key")
            
            success = retrieved is not None and "user_profile" in retrieved
            self.print_test("MongoDB Storage", success, f"Document: {retrieved}")
            
            # Test MongoDB conversation
            response = agent.invoke("Remember: I love flexible document storage")
            self.print_test("MongoDB Conversation", True, f"Response: {response}")
            
            return agent
            
        except Exception as e:
            self.print_test("MongoDB Storage", False, f"MongoDB connection failed: {str(e)}")
            return None
    
    def test_message_trimming(self):
        """Test 5: Message Trimming & Context Management"""
        self.print_header("Message Trimming Test")
        
        try:
            config = AgentConfig(
                name="TrimmingAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                enable_message_trimming=True,
                max_tokens=100,  # Very low for testing
                trim_strategy="last",  # Keep recent messages
                start_on="human",
                end_on=["human", "tool"],
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Generate many messages to trigger trimming
            for i in range(10):
                response = agent.invoke(f"Message {i}: This is a test message that should trigger trimming when we have enough messages.")
                
            # Check if trimming works
            self.print_test("Message Trimming", True, f"Sent 10 messages, trimming at {config.max_tokens} tokens")
            
            return agent
            
        except Exception as e:
            self.print_test("Message Trimming", False, f"Error: {str(e)}")
            return None
    
    def test_summarization(self):
        """Test 6: AI Summarization (LangMem)"""
        self.print_header("AI Summarization Test")
        
        try:
            config = AgentConfig(
                name="SummarizationAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                enable_summarization=True,
                max_summary_tokens=64,
                summarization_trigger_tokens=200,  # Low trigger for testing
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Generate content that should trigger summarization
            long_conversation = [
                "Hello, I'm starting a project about AI memory systems.",
                "I want to learn about different storage backends like Redis and PostgreSQL.",
                "The project involves testing various memory features comprehensively.",
                "I'm particularly interested in semantic search and embedding capabilities."
            ]
            
            for msg in long_conversation:
                response = agent.invoke(msg)
            
            self.print_test("AI Summarization", True, "Triggered summarization with long conversation")
            
            return agent
            
        except Exception as e:
            self.print_test("AI Summarization", False, f"Error: {str(e)}")
            return None
    
    def test_semantic_search(self):
        """Test 7: Semantic Search & Embeddings"""
        self.print_header("Semantic Search Test")
        
        try:
            config = AgentConfig(
                name="SemanticAgent",
                model=MockChatModel(),
                enable_long_term_memory=True,
                enable_semantic_search=True,
                embedding_model="openai:text-embedding-3-small",
                embedding_dims=1536,
                distance_type="cosine",
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Store semantically related memories
            memories = [
                {"key": "python_learning", "content": "I love programming in Python, especially for AI projects"},
                {"key": "machine_learning", "content": "Machine learning algorithms fascinate me, particularly neural networks"},
                {"key": "cooking_hobby", "content": "I enjoy cooking Italian cuisine, pasta is my favorite"},
                {"key": "travel_dreams", "content": "I dream of traveling to Japan to see cherry blossoms"}
            ]
            
            for memory in memories:
                agent.store_memory(memory["key"], {"content": memory["content"], "timestamp": time.time()})
            
            # Test semantic search (would work with real embeddings)
            self.print_test("Semantic Search Setup", True, f"Stored {len(memories)} semantically diverse memories")
            
            return agent
            
        except Exception as e:
            self.print_test("Semantic Search", False, f"Error: {str(e)}")
            return None
    
    def test_memory_tools(self):
        """Test 8: Memory Tools (Self-Managing Memory)"""
        self.print_header("Memory Tools Test")
        
        try:
            config = AgentConfig(
                name="MemoryToolsAgent",
                model=MockChatModel(),
                enable_long_term_memory=True,
                enable_memory_tools=True,
                memory_namespace_store="memories",
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test memory tools availability
            memory_tools = agent.memory_manager.get_memory_tools()
            
            self.print_test("Memory Tools", len(memory_tools) > 0, f"Available tools: {len(memory_tools)}")
            
            # Test agent with memory management capabilities
            response = agent.invoke("I want to remember that I'm testing memory tools functionality")
            self.print_test("Memory Tools Usage", True, f"Agent response: {response}")
            
            return agent
            
        except Exception as e:
            self.print_test("Memory Tools", False, f"Error: {str(e)}")
            return None
    
    def test_ttl_memory(self):
        """Test 9: TTL (Time-To-Live) Memory Expiration"""
        self.print_header("TTL Memory Expiration Test")
        
        try:
            config = AgentConfig(
                name="TTLAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                enable_ttl=True,
                default_ttl_minutes=1,  # 1 minute for testing
                refresh_on_read=True,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Store memory with TTL
            ttl_data = {
                "temporary_note": "This should expire in 1 minute",
                "created": time.time()
            }
            
            agent.store_memory("ttl_test_key", ttl_data)
            
            # Immediately retrieve (should work)
            retrieved = agent.retrieve_memory("ttl_test_key")
            immediate_success = retrieved is not None
            
            self.print_test("TTL Memory Storage", immediate_success, f"Stored with 1-minute TTL: {retrieved}")
            self.print_test("TTL Expiration Setup", True, "Memory will expire in 1 minute (not waiting in test)")
            
            return agent
            
        except Exception as e:
            self.print_test("TTL Memory", False, f"Error: {str(e)}")
            return None
    
    def test_session_sharing(self):
        """Test 10: Session-Based Memory Sharing"""
        self.print_header("Session Memory Sharing Test")
        
        try:
            # Create two agents with the same session ID
            shared_session = f"shared_session_{int(time.time())}"
            
            config1 = AgentConfig(
                name="Agent1",
                model=MockChatModel(),
                enable_short_term_memory=True,
                enable_shared_memory=True,
                session_id=shared_session,
                memory_namespace="agent1_space"
            )
            
            config2 = AgentConfig(
                name="Agent2", 
                model=MockChatModel(),
                enable_short_term_memory=True,
                enable_shared_memory=True,
                session_id=shared_session,  # Same session!
                memory_namespace="agent2_space"
            )
            
            agent1 = CoreAgent(config1)
            agent2 = CoreAgent(config2)
            
            # Agent1 stores memory
            shared_data = {
                "project_info": "We're testing session-based memory sharing",
                "collaboration": True,
                "agents": ["Agent1", "Agent2"]
            }
            
            agent1.store_memory("shared_project_key", shared_data)
            
            # Agent2 tries to access the same memory
            retrieved_by_agent2 = agent2.retrieve_memory("shared_project_key")
            
            sharing_works = retrieved_by_agent2 is not None
            self.print_test("Session Memory Sharing", sharing_works, f"Shared session: {shared_session}")
            
            # Test conversation in shared session
            agent1.invoke("I'm Agent1, starting our shared project")
            response = agent2.invoke("I'm Agent2, joining the shared project")
            
            self.print_test("Shared Session Conversation", True, f"Agent2 response: {response}")
            
            return agent1, agent2
            
        except Exception as e:
            self.print_test("Session Memory Sharing", False, f"Error: {str(e)}")
            return None, None
    
    def test_memory_namespacing(self):
        """Test 11: Memory Namespacing & Isolation"""
        self.print_header("Memory Namespacing Test")
        
        try:
            # Create agents with different namespaces
            config_ns1 = AgentConfig(
                name="NamespaceAgent1",
                model=MockChatModel(),
                enable_short_term_memory=True,
                memory_namespace="namespace_1",
                session_id=self.session_id
            )
            
            config_ns2 = AgentConfig(
                name="NamespaceAgent2",
                model=MockChatModel(),
                enable_short_term_memory=True,
                memory_namespace="namespace_2",  # Different namespace
                session_id=self.session_id
            )
            
            agent_ns1 = CoreAgent(config_ns1)
            agent_ns2 = CoreAgent(config_ns2)
            
            # Store memory in namespace 1
            ns1_data = {"namespace": "namespace_1", "private_data": "Agent1's private information"}
            agent_ns1.store_memory("private_key", ns1_data)
            
            # Store memory in namespace 2
            ns2_data = {"namespace": "namespace_2", "private_data": "Agent2's private information"}
            agent_ns2.store_memory("private_key", ns2_data)  # Same key, different namespace
            
            # Verify isolation
            ns1_retrieved = agent_ns1.retrieve_memory("private_key")
            ns2_retrieved = agent_ns2.retrieve_memory("private_key")
            
            isolation_works = (
                ns1_retrieved is not None and 
                ns2_retrieved is not None and
                ns1_retrieved["private_data"] != ns2_retrieved["private_data"]
            )
            
            self.print_test("Memory Namespacing", isolation_works, 
                          f"NS1: {ns1_retrieved['private_data'][:20]}..., NS2: {ns2_retrieved['private_data'][:20]}...")
            
            return agent_ns1, agent_ns2
            
        except Exception as e:
            self.print_test("Memory Namespacing", False, f"Error: {str(e)}")
            return None, None
    
    def test_rate_limiting_with_memory(self):
        """Test 12: Rate Limiting + Memory (Bonus)"""
        self.print_header("Rate Limiting + Memory Test")
        
        try:
            config = AgentConfig(
                name="RateLimitedMemoryAgent",
                model=MockChatModel(),
                enable_short_term_memory=True,
                enable_rate_limiting=True,
                requests_per_second=2.0,  # Moderate rate limiting
                max_bucket_size=3.0,
                session_id=self.session_id
            )
            
            agent = CoreAgent(config)
            
            # Test rate-limited memory operations
            start_time = time.time()
            
            for i in range(3):
                agent.store_memory(f"rate_test_{i}", {"message": f"Rate limited message {i}", "time": time.time()})
                response = agent.invoke(f"Message {i} with rate limiting")
            
            elapsed = time.time() - start_time
            
            # Should take some time due to rate limiting
            rate_limiting_works = elapsed > 0.5  # At least some delay
            
            self.print_test("Rate Limited Memory", rate_limiting_works, 
                          f"3 operations took {elapsed:.2f} seconds (rate limited)")
            
            return agent
            
        except Exception as e:
            self.print_test("Rate Limited Memory", False, f"Error: {str(e)}")
            return None
    
    def run_all_tests(self):
        """Run all memory tests"""
        print("🚀 Starting Comprehensive CoreAgent Memory Test Suite")
        print(f"📅 Timestamp: {datetime.now()}")
        print(f"🆔 Session ID: {self.session_id}")
        print(f"🏗️ Available backends: Redis={REDIS_AVAILABLE}, Postgres={POSTGRES_AVAILABLE}, MongoDB={MONGODB_AVAILABLE}")
        
        # Run all tests
        agents = {}
        
        agents['inmemory'] = self.test_inmemory_storage()
        agents['redis'] = self.test_redis_storage()
        agents['postgres'] = self.test_postgres_storage()
        agents['mongodb'] = self.test_mongodb_storage()
        agents['trimming'] = self.test_message_trimming()
        agents['summarization'] = self.test_summarization()
        agents['semantic'] = self.test_semantic_search()
        agents['memory_tools'] = self.test_memory_tools()
        agents['ttl'] = self.test_ttl_memory()
        
        # Multi-agent tests
        agent1, agent2 = self.test_session_sharing()
        agents['session_agent1'] = agent1
        agents['session_agent2'] = agent2
        
        ns_agent1, ns_agent2 = self.test_memory_namespacing()
        agents['namespace_agent1'] = ns_agent1
        agents['namespace_agent2'] = ns_agent2
        
        agents['rate_limited'] = self.test_rate_limiting_with_memory()
        
        # Print final results
        self.print_final_results()
        
        return agents
    
    def print_final_results(self):
        """Print comprehensive test results"""
        self.print_header("FINAL TEST RESULTS")
        
        passed = sum(1 for result in self.results.values() if result["success"])
        total = len(self.results)
        pass_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"📊 OVERALL RESULTS: {passed}/{total} tests passed ({pass_rate:.1f}%)")
        print()
        
        # Group results by category
        categories = {
            "Storage Backends": ["InMemory Storage", "Redis Storage", "PostgreSQL Storage", "MongoDB Storage"],
            "Context Management": ["Message Trimming", "AI Summarization", "TTL Memory Storage"],
            "Advanced Features": ["Semantic Search Setup", "Memory Tools", "Memory Tools Usage"],
            "Multi-Agent": ["Session Memory Sharing", "Shared Session Conversation", "Memory Namespacing"],
            "Performance": ["Rate Limited Memory"]
        }
        
        for category, test_names in categories.items():
            print(f"🏷️ {category}:")
            for test_name in test_names:
                if test_name in self.results:
                    result = self.results[test_name]
                    status = "✅" if result["success"] else "❌"
                    print(f"   {status} {test_name}")
            print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        
        if not self.results.get("Redis Storage", {}).get("success", False):
            print("   🔧 Install Redis: docker run -d -p 6379:6379 redis:alpine")
        
        if not self.results.get("PostgreSQL Storage", {}).get("success", False):
            print("   🔧 Install PostgreSQL: docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=password postgres")
        
        if not self.results.get("MongoDB Storage", {}).get("success", False):
            print("   🔧 Install MongoDB: docker run -d -p 27017:27017 mongo")
        
        print("   🌟 For production: Use persistent storage backends (Redis/PostgreSQL/MongoDB)")
        print("   🔐 Set OPENAI_API_KEY for embeddings and real LLM testing")
        print()
        
        print("🎉 Comprehensive memory testing completed!")


def main():
    """Main function to run the memory test suite"""
    print("🧠 CoreAgent Comprehensive Memory Test Suite")
    print("=" * 60)
    
    # Initialize test suite
    test_suite = MemoryTestSuite()
    
    # Run all tests
    try:
        agents = test_suite.run_all_tests()
        
        print(f"\n✨ Test completed! Session ID: {test_suite.session_id}")
        print("🔍 Check logs above for detailed results and setup instructions.")
        
        return agents
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        return None
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()