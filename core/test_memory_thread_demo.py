#!/usr/bin/env python3
"""
Core Agent Memory Demo - Thread-based conversation memory like LangGraph tutorial
Demonstrates how different thread_ids maintain separate conversation contexts
"""

from core import CoreAgent, AgentConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from typing import List, Optional, Any, Dict
import uuid


class MockChatModel(BaseChatModel):
    """Mock chat model for demonstration purposes"""
    
    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        # Simple mock responses based on conversation context
        last_message = messages[-1].content if messages else ""
        
        # Check conversation history for context
        conversation_history = [m.content for m in messages]
        conversation_text = " ".join(conversation_history)
        
        # Generate contextual responses
        if "my name is alice" in last_message.lower():
            response = "Hello Alice! It's nice to meet you. How can I help you today?"
        elif "what's my name" in last_message.lower() or "what is my name" in last_message.lower():
            # Look for names in conversation history
            if "alice" in conversation_text.lower():
                response = "Your name is Alice. You introduced yourself earlier in our conversation."
            elif "bob" in conversation_text.lower():
                response = "Your name is Bob. You mentioned that when we started talking."
            else:
                response = "I don't believe you've told me your name yet. Would you like to introduce yourself?"
        elif "remember my name" in last_message.lower():
            if "bob" in conversation_text.lower():
                response = "Yes, I remember! Your name is Bob, and you mentioned you love programming."
            elif "alice" in conversation_text.lower():
                response = "Of course! Your name is Alice."
            else:
                response = "I'm sorry, but I don't recall you mentioning your name in our conversation."
        elif "who i am" in last_message.lower() or "remember who i am" in last_message.lower():
            if "alice" in conversation_text.lower():
                response = "Yes, absolutely! You're Alice. We've been having a nice conversation."
            elif "bob" in conversation_text.lower():
                response = "You're Bob, the programming enthusiast!"
            else:
                response = "I'm not sure we've been properly introduced. Could you remind me who you are?"
        elif "i'm bob" in last_message.lower() or "i am bob" in last_message.lower():
            response = "Hello Bob! Nice to meet you. I hear you love programming - that's fantastic!"
        elif "testing" in last_message.lower() and "backend" in last_message.lower():
            response = f"I'll remember that you're testing the memory backend. This is stored in our conversation."
        else:
            response = f"I understand. Thank you for sharing that with me."
            
        return AIMessage(content=response)
    
    async def _agenerate(self, messages: List[BaseMessage], **kwargs) -> Any:
        return self._generate(messages, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def invoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> BaseMessage:
        if isinstance(input, list):
            return self._generate(input, **kwargs)
        elif isinstance(input, str):
            from langchain_core.messages import HumanMessage
            return self._generate([HumanMessage(content=input)], **kwargs)
        return self._generate([], **kwargs)
    
    async def ainvoke(self, input: Any, config: Optional[Dict] = None, **kwargs) -> BaseMessage:
        return self.invoke(input, config, **kwargs)


def demo_thread_memory():
    """Demonstrate thread-based memory (short-term memory with checkpointer)"""
    print("🧠 Core Agent Memory Demo - Thread-based Conversations")
    print("=" * 60)
    
    # Initialize mock model
    model = MockChatModel()
    
    # Configure agent with memory enabled
    config = AgentConfig(
        name="MemoryDemoAgent",
        model=model,
        system_prompt="You are a helpful assistant who remembers conversations.",
        enable_memory=True,
        memory_types=["short_term"],  # Thread-level persistence
        memory_backend="inmemory"  # Using InMemorySaver
    )
    
    # Create agent
    agent = CoreAgent(config)
    
    print("\n📝 Configuration:")
    print(f"  - Memory enabled: {config.enable_memory}")
    print(f"  - Memory types: {config.memory_types}")
    print(f"  - Memory backend: {config.memory_backend}")
    print(f"  - Checkpointer type: {type(agent.memory_manager.checkpointer).__name__}")
    
    # First conversation thread
    print("\n\n🔵 Thread 1 - Introducing ourselves")
    print("-" * 40)
    
    # First message in thread 1
    response1 = agent.invoke(
        "Hi there! My name is Alice.",
        config={"configurable": {"thread_id": "thread_1"}}
    )
    print(f"Human: Hi there! My name is Alice.")
    print(f"Agent: {response1['messages'][-1].content}")
    
    # Follow-up in thread 1
    response2 = agent.invoke(
        "What's my name?",
        config={"configurable": {"thread_id": "thread_1"}}
    )
    print(f"\nHuman: What's my name?")
    print(f"Agent: {response2['messages'][-1].content}")
    
    # Second conversation thread
    print("\n\n🟢 Thread 2 - Different conversation")
    print("-" * 40)
    
    # First message in thread 2
    response3 = agent.invoke(
        "Hi! I'm Bob and I love programming.",
        config={"configurable": {"thread_id": "thread_2"}}
    )
    print(f"Human: Hi! I'm Bob and I love programming.")
    print(f"Agent: {response3['messages'][-1].content}")
    
    # Try asking for name in thread 2
    response4 = agent.invoke(
        "Do you remember my name?",
        config={"configurable": {"thread_id": "thread_2"}}
    )
    print(f"\nHuman: Do you remember my name?")
    print(f"Agent: {response4['messages'][-1].content}")
    
    # Go back to thread 1
    print("\n\n🔵 Back to Thread 1")
    print("-" * 40)
    
    response5 = agent.invoke(
        "Can you still remember who I am?",
        config={"configurable": {"thread_id": "thread_1"}}
    )
    print(f"Human: Can you still remember who I am?")
    print(f"Agent: {response5['messages'][-1].content}")
    
    # Show state inspection
    print("\n\n📊 State Inspection")
    print("-" * 40)
    
    # Get state for thread 1
    if hasattr(agent.compiled_graph, 'get_state'):
        state1 = agent.compiled_graph.get_state({"configurable": {"thread_id": "thread_1"}})
        print(f"\nThread 1 message count: {len(state1.values.get('messages', []))}")
        print("Thread 1 messages:")
        for i, msg in enumerate(state1.values.get('messages', [])):
            role = msg.__class__.__name__.replace('Message', '')
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  {i+1}. {role}: {content_preview}")
        
        # Get state for thread 2
        state2 = agent.compiled_graph.get_state({"configurable": {"thread_id": "thread_2"}})
        print(f"\nThread 2 message count: {len(state2.values.get('messages', []))}")
        print("Thread 2 messages:")
        for i, msg in enumerate(state2.values.get('messages', [])):
            role = msg.__class__.__name__.replace('Message', '')
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            print(f"  {i+1}. {role}: {content_preview}")


def demo_memory_backends():
    """Demonstrate different memory backends"""
    print("\n\n🗄️ Memory Backend Comparison")
    print("=" * 60)
    
    model = MockChatModel()
    
    backends_config = [
        {
            "name": "InMemory",
            "config": {
                "memory_backend": "inmemory",
                "memory_types": ["short_term", "long_term"]
            }
        },
        {
            "name": "Redis (if available)",
            "config": {
                "memory_backend": "redis",
                "memory_types": ["short_term", "long_term", "semantic"],
                "redis_url": "redis://localhost:6379",
                "enable_ttl": True,
                "default_ttl_minutes": 60
            }
        }
    ]
    
    for backend_info in backends_config:
        print(f"\n\n🔧 Testing {backend_info['name']} Backend")
        print("-" * 40)
        
        try:
            config = AgentConfig(
                name=f"{backend_info['name']}Agent",
                model=model,
                enable_memory=True,
                **backend_info['config']
            )
            
            agent = CoreAgent(config)
            
            # Show what's available
            mm = agent.memory_manager
            print(f"✅ Successfully initialized {backend_info['name']}")
            print(f"  - Has short-term memory: {mm.has_short_term_memory()}")
            print(f"  - Has long-term memory: {mm.has_long_term_memory()}")
            print(f"  - Checkpointer type: {type(mm.checkpointer).__name__ if mm.checkpointer else 'None'}")
            print(f"  - Store type: {type(mm.store).__name__ if mm.store else 'None'}")
            
            # Test short-term (thread-based) memory
            if mm.has_short_term_memory():
                response = agent.invoke(
                    f"Remember this: I'm testing the {backend_info['name']} backend",
                    config={"configurable": {"thread_id": f"{backend_info['name'].lower()}_test"}}
                )
                print(f"\n  Thread memory test: ✅ Stored message")
            
            # Test long-term memory
            if mm.has_long_term_memory():
                mm.store_long_term_memory(
                    key="backend_test",
                    data={"backend": backend_info['name'], "status": "tested"}
                )
                retrieved = mm.get_long_term_memory("backend_test")
                if retrieved:
                    print(f"  Long-term memory test: ✅ Stored and retrieved: {retrieved}")
                    
        except Exception as e:
            print(f"❌ Failed to initialize {backend_info['name']}: {str(e)}")


def demo_memory_features():
    """Demonstrate advanced memory features"""
    print("\n\n🚀 Advanced Memory Features")
    print("=" * 60)
    
    model = MockChatModel()
    
    # Test semantic memory with embeddings
    print("\n📍 Semantic Memory (Vector Search)")
    print("-" * 40)
    
    try:
        config = AgentConfig(
            name="SemanticMemoryAgent",
            model=model,
            enable_memory=True,
            memory_types=["long_term", "semantic"],
            memory_backend="inmemory",
            embedding_model="openai:text-embedding-3-small",
            embedding_dims=1536
        )
        
        agent = CoreAgent(config)
        mm = agent.memory_manager
        
        # Store some memories with semantic content
        memories = [
            ("paris_trip", {"content": "I visited Paris last summer. The Eiffel Tower was amazing!"}),
            ("tokyo_trip", {"content": "Tokyo was incredible in spring. Cherry blossoms everywhere!"}),
            ("cooking_pasta", {"content": "I learned to make authentic Italian pasta from scratch."}),
            ("python_project", {"content": "Built a web scraper using Python and BeautifulSoup."})
        ]
        
        print("Storing memories...")
        for key, data in memories:
            mm.store_long_term_memory(key, data)
            print(f"  ✅ Stored: {key}")
        
        # Search memories
        if hasattr(mm, 'search_memory'):
            print("\nSearching for 'travel experiences'...")
            results = mm.search_memory("travel experiences", limit=3)
            for i, result in enumerate(results):
                print(f"  {i+1}. {result}")
                
    except Exception as e:
        print(f"❌ Semantic memory demo failed: {str(e)}")
    
    # Test session memory (shared between agents)
    print("\n\n🤝 Session Memory (Multi-agent sharing)")
    print("-" * 40)
    
    try:
        session_id = "collab_session_123"
        
        # Agent 1
        config1 = AgentConfig(
            name="ResearchAgent",
            model=model,
            enable_memory=True,
            memory_types=["session"],
            memory_backend="inmemory",
            session_id=session_id
        )
        
        # Agent 2  
        config2 = AgentConfig(
            name="WriterAgent",
            model=model,
            enable_memory=True,
            memory_types=["session"],
            memory_backend="inmemory",
            session_id=session_id
        )
        
        agent1 = CoreAgent(config1)
        agent2 = CoreAgent(config2)
        
        print(f"Created two agents sharing session: {session_id}")
        print(f"  - Agent 1: {agent1.config.name}")
        print(f"  - Agent 2: {agent2.config.name}")
        
        # Agent 1 stores information
        if agent1.memory_manager.has_session_memory():
            agent1.memory_manager.store_session_memory({
                "research_topic": "AI advancements in 2024",
                "key_findings": ["LLMs getting smaller", "Multi-modal improvements"]
            })
            print("\n✅ Agent 1 stored research findings in session memory")
            
            # Agent 2 retrieves information
            shared_data = agent2.memory_manager.get_session_memory()
            if shared_data:
                print(f"✅ Agent 2 retrieved from session: {shared_data}")
                
    except Exception as e:
        print(f"❌ Session memory demo failed: {str(e)}")


def explain_memory_system():
    """Explain how the memory system works"""
    print("\n\n📚 Memory System Explanation")
    print("=" * 60)
    
    print("\n🔍 How Memory Works in Core Agent:")
    print("\n1. **Thread-based Memory (Short-term)**")
    print("   - Uses LangGraph checkpointers (InMemorySaver, RedisSaver, PostgresSaver)")
    print("   - Each thread_id maintains separate conversation state")
    print("   - Messages are automatically saved after each interaction")
    print("   - Perfect for multi-turn conversations")
    
    print("\n2. **Long-term Memory (Persistent Storage)**")
    print("   - Uses LangGraph stores (InMemoryStore, RedisStore, PostgresStore)")
    print("   - Data persists across different threads and sessions")
    print("   - Can store arbitrary key-value pairs")
    print("   - Supports semantic search with embeddings")
    
    print("\n3. **Memory Backends:**")
    print("   - **inmemory**: Fast, but data lost on restart")
    print("   - **redis**: Persistent, supports TTL, requires Redis Stack for vectors")
    print("   - **postgres**: Persistent, supports vectors with pgvector extension")
    print("   - **mongodb**: Persistent, supports TTL (store not yet available)")
    
    print("\n4. **Special Features:**")
    print("   - **TTL (Time-To-Live)**: Auto-expire old data (Redis/MongoDB)")
    print("   - **Semantic Search**: Find similar content using embeddings")
    print("   - **Session Memory**: Share data between multiple agents")
    print("   - **Message Trimming**: Keep conversations within token limits")


if __name__ == "__main__":
    # Run all demos
    demo_thread_memory()
    demo_memory_backends()
    demo_memory_features()
    explain_memory_system()
    
    print("\n\n✨ Memory Demo Complete!")
    print("\nKey Takeaways:")
    print("1. thread_id creates separate conversation contexts (like chat sessions)")
    print("2. Different memory backends (inmemory, redis, postgres) offer different features")
    print("3. Memory types include: short_term (thread), long_term (persistent), session (shared), semantic (vector)")
    print("4. Redis/MongoDB support TTL for automatic expiration")
    print("5. Semantic memory enables similarity search with embeddings")