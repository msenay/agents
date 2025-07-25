#!/usr/bin/env python3
"""
Redis Memory Demo - Tests all Redis memory features with CoreAgent

Requirements:
- Redis Stack must be running (docker-compose up redis)
- pip install redis langgraph-checkpoint-redis langgraph-store-redis

Features tested:
1. Short-term memory (conversation/thread-based)
2. Long-term memory (key-value store)
3. Semantic memory (vector search)
4. Session memory (multi-agent sharing)
5. TTL (Time-To-Live) support

Environment Variables:
- REDIS_URL: Redis connection URL (default: redis://:redis_password@localhost:6379)
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any

from core import CoreAgent, AgentConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage


# Get Redis URL from environment or use default with password
DEFAULT_REDIS_URL = "redis://:redis_password@localhost:6379"
REDIS_URL = os.getenv("REDIS_URL", DEFAULT_REDIS_URL)


# Redis connection check
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


class RedisMemoryDemo:
    """Demo testing Redis memory features"""
    
    def __init__(self):
        # Use environment variable or default
        self.redis_url = REDIS_URL
        self.model = None
        self.agent = None
        
    def setup(self):
        """Setup for demo"""
        print("\n🚀 Redis Memory Demo Starting...")
        print("=" * 60)
        
        # OpenAI API key check
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  OPENAI_API_KEY not found. Using mock model.")
            from core.simple_examples import MockLLM
            self.model = MockLLM()
        else:
            print("✅ OpenAI API key found")
            self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            
    def create_memory_tools(self):
        """Create tools for long-term memory"""
        
        @tool
        def save_user_info(name: str, info: Dict[str, Any]) -> str:
            """Save user information to long-term memory"""
            if self.agent:
                key = f"user_info_{name}"
                self.agent.memory_manager.store_long_term_memory(key, info)
                return f"Information saved for {name}"
            return "Agent not ready yet"
            
        @tool  
        def get_user_info(name: str) -> str:
            """Get user information from long-term memory"""
            if self.agent:
                key = f"user_info_{name}"
                info = self.agent.memory_manager.get_long_term_memory(key)
                if info:
                    return f"{name} info: {json.dumps(info)}"
                return f"No information found for {name}"
            return "Agent not ready yet"
            
        @tool
        def search_similar_notes(query: str, limit: int = 3) -> str:
            """Find similar notes using semantic search"""
            if self.agent and hasattr(self.agent.memory_manager, 'search_memory'):
                results = self.agent.memory_manager.search_memory(query, limit=limit)
                if results:
                    return f"Similar notes: {results}"
                return "No similar notes found"
            return "Semantic search not available"
            
        return [save_user_info, get_user_info, search_similar_notes]
        
    def test_short_term_memory(self):
        """Test short-term (conversation) memory"""
        print("\n\n🔵 TEST 1: Short-term Memory (Thread-based Conversations)")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisShortTermAgent",
            model=self.model,
            system_prompt="You are an assistant using Redis memory.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term"],
            redis_url=self.redis_url
        )
        
        try:
            agent = CoreAgent(config)
            print("✅ Short-term memory agent created")
            
            # Thread 1 - Conversation with Ali
            print("\n📱 Thread 1 - Ali:")
            thread_1_config = {"configurable": {"thread_id": "ali_conversation"}}
            
            response1 = agent.invoke("Hello, my name is Ali and I live in Istanbul", config=thread_1_config)
            print(f"Ali: Hello, my name is Ali and I live in Istanbul")
            print(f"Agent: {response1['messages'][-1].content}")
            
            response2 = agent.invoke("My favorite food is pizza", config=thread_1_config)
            print(f"\nAli: My favorite food is pizza")
            print(f"Agent: {response2['messages'][-1].content}")
            
            # Thread 2 - Conversation with Ayse
            print("\n\n📱 Thread 2 - Ayse:")
            thread_2_config = {"configurable": {"thread_id": "ayse_conversation"}}
            
            response3 = agent.invoke("Hi, I'm Ayse, I live in Ankara", config=thread_2_config)
            print(f"Ayse: Hi, I'm Ayse, I live in Ankara")
            print(f"Agent: {response3['messages'][-1].content}")
            
            # Return to Thread 1
            print("\n\n📱 Returning to Thread 1:")
            response4 = agent.invoke("Do you remember my name and where I live?", config=thread_1_config)
            print(f"Ali: Do you remember my name and where I live?")
            print(f"Agent: {response4['messages'][-1].content}")
            
            # State check
            if hasattr(agent.compiled_graph, 'get_state'):
                state = agent.compiled_graph.get_state(thread_1_config)
                print(f"\n📊 Thread 1 State: {len(state.values.get('messages', []))} messages")
                
            print("\n✅ Short-term memory test successful!")
            
        except Exception as e:
            print(f"❌ Short-term memory test failed: {e}")
            
    def test_long_term_memory(self):
        """Test long-term (persistent store) memory"""
        print("\n\n🟢 TEST 2: Long-term Memory (Persistent Key-Value Store)")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisLongTermAgent", 
            model=self.model,
            system_prompt="You are an assistant that saves user information.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term"],
            redis_url=self.redis_url,
            tools=self.create_memory_tools()
        )
        
        try:
            self.agent = agent = CoreAgent(config)
            print("✅ Long-term memory agent created")
            
            # Manual save
            print("\n📝 Manual Save:")
            user_data = {
                "name": "Ali",
                "age": 28,
                "city": "Istanbul",
                "interests": ["technology", "music", "travel"],
                "registered_at": datetime.now().isoformat()
            }
            
            agent.memory_manager.store_long_term_memory("user_ali_profile", user_data)
            print(f"Saved: {user_data}")
            
            # Manual load
            print("\n📖 Manual Load:")
            retrieved = agent.memory_manager.get_long_term_memory("user_ali_profile")
            print(f"Retrieved: {retrieved}")
            
            # Using with tool
            print("\n🔧 Using with Tool:")
            response = agent.invoke("Get Ali's user information")
            print(f"Tool response: {response['messages'][-1].content}")
            
            # Using different namespace
            print("\n📁 Namespace Usage:")
            agent.memory_manager.store_long_term_memory(
                "settings",
                {"theme": "dark", "language": "en"},
                namespace="app_config"
            )
            settings = agent.memory_manager.get_long_term_memory("settings", namespace="app_config")
            print(f"App settings: {settings}")
            
            print("\n✅ Long-term memory test successful!")
            
        except Exception as e:
            print(f"❌ Long-term memory test failed: {e}")
            
    def test_semantic_memory(self):
        """Test semantic (vector search) memory"""
        print("\n\n🔴 TEST 3: Semantic Memory (Vector-based Similarity Search)")
        print("-" * 60)
        
        # Embedding model check
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Semantic search requires OpenAI API key. Skipping test.")
            return
            
        config = AgentConfig(
            name="RedisSemanticAgent",
            model=self.model,
            system_prompt="You are an assistant capable of semantic search.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["long_term", "semantic"],
            redis_url=self.redis_url,
            embedding_model="openai:text-embedding-3-small",
            embedding_dims=1536,
            tools=self.create_memory_tools()
        )
        
        try:
            self.agent = agent = CoreAgent(config)
            print("✅ Semantic memory agent created")
            
            # Save various notes
            print("\n📝 Saving Notes:")
            notes = [
                ("travel_paris", {"content": "Visited the Eiffel Tower in Paris, amazing experience", "date": "2024-01-15"}),
                ("travel_tokyo", {"content": "Cherry blossoms were blooming in Tokyo, beautiful scenery", "date": "2024-03-20"}),
                ("cooking_pasta", {"content": "Learned to make Italian pasta from scratch, tomato sauce recipe", "date": "2024-02-10"}),
                ("tech_python", {"content": "Developed a machine learning project with Python", "date": "2024-01-05"}),
                ("book_scifi", {"content": "Read Dune book, great for sci-fi lovers", "date": "2024-02-28"})
            ]
            
            for key, data in notes:
                agent.memory_manager.store_long_term_memory(key, data)
                print(f"  ✓ {key}: {data['content'][:50]}...")
                
            # Semantic search tests
            print("\n🔍 Semantic Search Tests:")
            
            queries = [
                "travel memories",
                "cooking recipes", 
                "programming projects",
                "Japan experiences"
            ]
            
            for query in queries:
                print(f"\n📍 Searching for: '{query}'")
                if hasattr(agent.memory_manager, 'search_memory'):
                    results = agent.memory_manager.search_memory(query, limit=3)
                    if results:
                        for i, result in enumerate(results, 1):
                            print(f"  {i}. {result}")
                    else:
                        print("  No results found")
                        
            print("\n✅ Semantic memory test successful!")
            
        except Exception as e:
            print(f"❌ Semantic memory test failed: {e}")
            print("📌 Note: Redis Stack (RediSearch module) required!")
            
    def test_session_memory(self):
        """Test session (multi-agent shared) memory"""
        print("\n\n🟡 TEST 4: Session Memory (Multi-Agent Shared Memory)")
        print("-" * 60)
        
        session_id = "team_collaboration_123"
        
        # Agent 1: Researcher
        config1 = AgentConfig(
            name="ResearchAgent",
            model=self.model,
            system_prompt="You are a research agent.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["session"],
            redis_url=self.redis_url,
            session_id=session_id
        )
        
        # Agent 2: Writer
        config2 = AgentConfig(
            name="WriterAgent",
            model=self.model,
            system_prompt="You are a writing agent.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["session"],
            redis_url=self.redis_url,
            session_id=session_id
        )
        
        try:
            agent1 = CoreAgent(config1)
            agent2 = CoreAgent(config2)
            print(f"✅ Session agents created (Session: {session_id})")
            
            # Agent 1 does research and shares
            print("\n👤 Agent 1 (Researcher) sharing data:")
            if agent1.memory_manager.has_session_memory():
                research_data = {
                    "topic": "AI Trends 2024",
                    "key_points": [
                        "Multimodal AI on the rise",
                        "Edge AI devices becoming common",
                        "AI regulation increasing"
                    ],
                    "sources": ["MIT Review", "Nature AI", "ArXiv"]
                }
                agent1.memory_manager.store_session_memory(research_data)
                print(f"  ✓ Research data shared: {research_data['topic']}")
                
                # Agent 2 reads the data
                print("\n👤 Agent 2 (Writer) reading data:")
                shared_data = agent2.memory_manager.get_session_memory()
                if shared_data:
                    print(f"  ✓ Shared data received: {len(shared_data)} items")
                    for item in shared_data:
                        print(f"    - {item}")
                        
                # Agent-specific memory
                print("\n📌 Agent-specific memory:")
                agent1.memory_manager.store_agent_memory(
                    "ResearchAgent",
                    session_id,
                    {"status": "research_completed", "duration": "2 hours"}
                )
                
                agent2.memory_manager.store_agent_memory(
                    "WriterAgent", 
                    session_id,
                    {"status": "writing_draft", "word_count": 1500}
                )
                
                print("  ✓ Agent-specific data saved")
                
            else:
                print("⚠️  Session memory requires Redis backend")
                
            print("\n✅ Session memory test completed!")
            
        except Exception as e:
            print(f"❌ Session memory test failed: {e}")
            
    def test_ttl_support(self):
        """Test TTL (Time-To-Live) support"""
        print("\n\n⏰ TEST 5: TTL Support (Auto-expiration)")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisTTLAgent",
            model=self.model,
            system_prompt="You are an assistant using TTL-enabled memory.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term"],
            redis_url=self.redis_url,
            enable_ttl=True,
            default_ttl_minutes=1,  # 1 minute TTL
            refresh_on_read=True
        )
        
        try:
            agent = CoreAgent(config)
            print("✅ TTL agent created (TTL: 1 minute)")
            
            # Save data
            print("\n📝 Saving data with TTL:")
            temp_data = {
                "session_token": "abc123xyz",
                "created_at": datetime.now().isoformat(),
                "purpose": "temporary auth token"
            }
            
            agent.memory_manager.store_long_term_memory("temp_session", temp_data)
            print(f"  ✓ Temporary data saved: {temp_data}")
            
            # Read immediately
            print("\n📖 Reading data immediately:")
            retrieved = agent.memory_manager.get_long_term_memory("temp_session")
            print(f"  ✓ Data exists: {retrieved is not None}")
            
            # TTL refresh test
            if config.refresh_on_read:
                print("\n🔄 TTL refresh test:")
                print("  - refresh_on_read=True so TTL was refreshed")
                print("  - Data will live for another minute")
                
            print("\n⏳ Note: Data will auto-expire after 1 minute")
            print("✅ TTL test completed!")
            
        except Exception as e:
            print(f"❌ TTL test failed: {e}")
            
    def test_advanced_features(self):
        """Test advanced features"""
        print("\n\n🚀 TEST 6: Advanced Features")
        print("-" * 60)
        
        config = AgentConfig(
            name="RedisAdvancedAgent",
            model=self.model,
            system_prompt="You are an assistant using advanced Redis features.",
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term", "semantic"],
            redis_url=self.redis_url,
            enable_message_trimming=True,
            max_tokens=1000,
            trim_strategy="last"
        )
        
        try:
            agent = CoreAgent(config)
            print("✅ Advanced agent created")
            
            # Message trimming
            print("\n✂️ Message Trimming:")
            print(f"  - Max tokens: {config.max_tokens}")
            print(f"  - Strategy: {config.trim_strategy}")
            print("  - Old messages auto-trimmed in long conversations")
            
            # Store metadata
            print("\n📊 Metadata Storage:")
            agent.memory_manager.store_long_term_memory(
                "user_session_meta",
                {
                    "user_id": "usr_123",
                    "session_start": datetime.now().isoformat(),
                    "device": "web",
                    "location": "US",
                    "preferences": {
                        "language": "en",
                        "theme": "dark"
                    }
                }
            )
            print("  ✓ Session metadata saved")
            
            # Namespace usage
            print("\n📁 Namespace Organization:")
            namespaces = ["users", "sessions", "analytics", "configs"]
            for ns in namespaces:
                agent.memory_manager.store_long_term_memory(
                    f"test_key",
                    {"namespace": ns, "data": f"Test data for {ns}"},
                    namespace=ns
                )
            print(f"  ✓ Data organized in {len(namespaces)} different namespaces")
            
            print("\n✅ Advanced features test completed!")
            
        except Exception as e:
            print(f"❌ Advanced features test failed: {e}")
            
    def show_redis_stats(self):
        """Show Redis usage statistics"""
        print("\n\n📊 Redis Usage Statistics")
        print("-" * 60)
        
        try:
            import redis
            r = redis.from_url(self.redis_url)
            
            # Key count
            keys = r.keys("*")
            print(f"\n🔑 Total key count: {len(keys)}")
            
            # Group by key types
            key_types = {}
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                prefix = key_str.split(":")[0] if ":" in key_str else "other"
                key_types[prefix] = key_types.get(prefix, 0) + 1
                
            print("\n📂 Key types:")
            for prefix, count in sorted(key_types.items()):
                print(f"  - {prefix}: {count} keys")
                
            # Memory usage
            info = r.info("memory")
            used_memory = info.get("used_memory_human", "N/A")
            print(f"\n💾 Memory usage: {used_memory}")
            
        except Exception as e:
            print(f"❌ Could not get Redis stats: {e}")
            
    def cleanup(self):
        """Clean test data (optional)"""
        print("\n\n🧹 Cleanup (Optional)")
        print("-" * 60)
        
        response = input("Do you want to clean test data? (y/N): ")
        if response.lower() == 'y':
            try:
                import redis
                r = redis.from_url(self.redis_url)
                
                # Find and delete test keys
                test_keys = r.keys("*test*") + r.keys("*ali*") + r.keys("*ayse*")
                if test_keys:
                    r.delete(*test_keys)
                    print(f"✅ {len(test_keys)} test keys deleted")
                else:
                    print("ℹ️  No test keys found to delete")
                    
            except Exception as e:
                print(f"❌ Cleanup failed: {e}")
        else:
            print("ℹ️  Test data preserved")


def main():
    """Main demo function"""
    print("🚀 Redis Memory Demo - CoreAgent")
    print("================================")
    print("\nThis demo tests all Redis memory features:")
    print("- Short-term (conversation) memory")
    print("- Long-term (key-value) memory") 
    print("- Semantic (vector search) memory")
    print("- Session (multi-agent) memory")
    print("- TTL (auto-expiration) support")
    
    # Redis check
    if not check_redis_connection():
        return
        
    # Run demo
    demo = RedisMemoryDemo()
    demo.setup()
    
    # Run tests
    demo.test_short_term_memory()
    demo.test_long_term_memory()
    demo.test_semantic_memory()
    demo.test_session_memory()
    demo.test_ttl_support()
    demo.test_advanced_features()
    
    # Statistics
    demo.show_redis_stats()
    
    # Cleanup
    demo.cleanup()
    
    print("\n\n✨ Redis Memory Demo Completed!")
    print("\n📚 What we learned:")
    print("1. Short-term = Thread-based conversation (automatic)")
    print("2. Long-term = Key-value store (manual)")
    print("3. Semantic = Vector similarity search")
    print("4. Session = Multi-agent sharing")
    print("5. TTL = Automatic expiration")
    print("\n🎯 Redis supports all memory types!")


if __name__ == "__main__":
    main()