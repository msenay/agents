#!/usr/bin/env python3
"""
CoreAgent Framework - LangGraph Memory Examples

This file demonstrates how to use ALL memory options from the LangGraph documentation
with our enhanced CoreAgent framework.
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from core_agent import (
    # Memory-optimized factory functions
    create_memory_agent,
    create_short_term_memory_agent,
    create_long_term_memory_agent,
    create_message_management_agent,
    create_semantic_search_agent,
    create_ttl_memory_agent,
    # Session-based functions
    create_session_agent,
    create_collaborative_agents
)

def create_mock_model():
    """Create a mock model for examples"""
    return FakeListChatModel(responses=[
        "I can help you with memory management.",
        "I remember our previous conversation.",
        "Let me search my memory for relevant information.",
        "I've stored this information for future reference."
    ])

@tool
def example_tool() -> str:
    """Example tool for demonstration"""
    return "Tool executed successfully"

# =============================================================================
# 1. SHORT-TERM MEMORY EXAMPLES (Thread-level persistence)
# =============================================================================

def example_short_term_memory():
    """Example: Short-term memory for conversations"""
    print("🧠 SHORT-TERM MEMORY EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    
    # InMemorySaver - Development
    agent = create_short_term_memory_agent(
        model=model,
        memory_backend="inmemory",
        enable_trimming=True,
        max_tokens=4000
    )
    
    # Use with thread ID for conversation persistence
    response = agent.invoke(
        {"messages": [HumanMessage(content="Remember I like coffee")]},
        config={"configurable": {"thread_id": "user_123"}}
    )
    print(f"Agent response: {response['messages'][-1].content}")
    
    # Continue conversation with same thread
    response2 = agent.invoke(
        {"messages": [HumanMessage(content="What do I like to drink?")]},
        config={"configurable": {"thread_id": "user_123"}}
    )
    print(f"Follow-up response: {response2['messages'][-1].content}")


# =============================================================================
# 2. LONG-TERM MEMORY EXAMPLES (Cross-session persistence)
# =============================================================================

def example_long_term_memory():
    """Example: Long-term memory for persistent data"""
    print("\n🗄️ LONG-TERM MEMORY EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    
    agent = create_long_term_memory_agent(
        model=model,
        memory_backend="inmemory",
        enable_semantic_search=True,
        enable_memory_tools=True
    )
    
    # Store user preferences
    agent.memory_manager.store_long_term_memory(
        "user_preferences",
        {
            "language": "python",
            "framework": "langgraph",
            "memory_type": "comprehensive"
        }
    )
    
    # Retrieve across sessions
    preferences = agent.memory_manager.get_long_term_memory("user_preferences")
    print(f"Stored preferences: {preferences}")


# =============================================================================
# 3. MESSAGE MANAGEMENT EXAMPLES (trimming, summarization, deletion)
# =============================================================================

def example_message_management():
    """Example: Message management for long conversations"""
    print("\n✂️ MESSAGE MANAGEMENT EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    
    # Message trimming agent
    agent = create_message_management_agent(
        model=model,
        management_strategy="trim",
        max_tokens=500,  # Small limit to trigger trimming
        trim_strategy="last"
    )
    
    # Create long conversation
    long_conversation = [
        HumanMessage(content="Tell me about AI " * 50),
        HumanMessage(content="Explain machine learning " * 50),
        HumanMessage(content="What is deep learning? " * 50),
    ]
    
    # Test trimming hook
    pre_hook = agent.memory_manager.get_pre_model_hook()
    if pre_hook:
        trimmed = pre_hook({"messages": long_conversation})
        print(f"Original messages: {len(long_conversation)}")
        print(f"Trimmed messages: {len(trimmed.get('llm_input_messages', []))}")


# =============================================================================
# 4. SEMANTIC SEARCH EXAMPLES (with embeddings)
# =============================================================================

def example_semantic_search():
    """Example: Semantic search with embeddings"""
    print("\n🔍 SEMANTIC SEARCH EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    
    agent = create_semantic_search_agent(
        model=model,
        memory_backend="inmemory",
        embedding_model="openai:text-embedding-3-small",
        enable_memory_tools=True
    )
    
    # Store some documents
    documents = [
        {"topic": "python", "content": "Python is a programming language"},
        {"topic": "ai", "content": "Artificial intelligence enables automation"},
        {"topic": "memory", "content": "Memory systems store and retrieve information"}
    ]
    
    for i, doc in enumerate(documents):
        agent.memory_manager.store_long_term_memory(f"doc_{i}", doc)
    
    # Search semantically (would work with proper embeddings)
    results = agent.memory_manager.search_memory("programming languages", limit=3)
    print(f"Search results: {len(results)} items found")


# =============================================================================
# 5. SESSION-BASED MEMORY EXAMPLES (Agent collaboration)
# =============================================================================

def example_session_memory():
    """Example: Session-based memory for agent collaboration"""
    print("\n🤝 SESSION-BASED MEMORY EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    session_id = "coding_session_123"
    
    # Create collaborative agents
    coder_agent = create_session_agent(
        model=model,
        session_id=session_id,
        name="CoderAgent",
        redis_url="redis://localhost:6379"
    )
    
    reviewer_agent = create_session_agent(
        model=model,
        session_id=session_id,
        name="ReviewerAgent", 
        redis_url="redis://localhost:6379"
    )
    
    # Coder stores code
    if coder_agent.memory_manager.has_session_memory():
        coder_agent.memory_manager.store_session_memory({
            "code": "def hello_world(): return 'Hello, World!'",
            "author": "CoderAgent",
            "status": "needs_review"
        })
        
        # Reviewer accesses the code
        shared_data = reviewer_agent.memory_manager.get_session_memory()
        print(f"Reviewer sees: {len(shared_data)} shared items")
        if shared_data:
            print(f"Code to review: {shared_data[0].get('code', 'No code')}")
    else:
        print("Session memory requires Redis configuration")


# =============================================================================
# 6. TTL MEMORY EXAMPLES (Time-To-Live with automatic cleanup)
# =============================================================================

def example_ttl_memory():
    """Example: TTL memory with automatic expiration"""
    print("\n⏰ TTL MEMORY EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    
    agent = create_ttl_memory_agent(
        model=model,
        memory_backend="redis",  # TTL works best with Redis
        ttl_minutes=60,  # 1 hour expiration
        refresh_on_read=True,
        redis_url="redis://localhost:6379"
    )
    
    # Store temporary data
    if agent.memory_manager.has_long_term_memory():
        agent.memory_manager.store_long_term_memory(
            "temporary_session",
            {
                "user_id": "temp_123", 
                "session_data": "This will expire in 1 hour",
                "created_at": "2024-01-01T10:00:00"
            }
        )
        print("Stored temporary data with TTL - will auto-expire")
    else:
        print("TTL memory requires Redis configuration")


# =============================================================================
# 7. COMPREHENSIVE MEMORY EXAMPLES (All features combined)
# =============================================================================

def example_comprehensive_memory():
    """Example: Comprehensive memory with all features"""
    print("\n🚀 COMPREHENSIVE MEMORY EXAMPLE")
    print("-" * 50)
    
    model = create_mock_model()
    
    # Create agent with all memory features
    agent = create_memory_agent(
        model=model,
        name="ComprehensiveAgent",
        tools=[example_tool],
        short_term_memory="inmemory",
        long_term_memory="inmemory", 
        enable_semantic_search=True,
        enable_memory_tools=True,
        enable_message_trimming=True,
        enable_summarization=False,  # Requires LangMem
        redis_url="redis://localhost:6379"
    )
    
    print(f"✅ Created comprehensive agent with:")
    print(f"   - Short-term memory: {agent.memory_manager.has_short_term_memory()}")
    print(f"   - Long-term memory: {agent.memory_manager.has_long_term_memory()}")
    print(f"   - Memory tools: {len(agent.memory_manager.get_memory_tools())} tools")
    print(f"   - Message trimming: {agent.config.enable_message_trimming}")
    print(f"   - Semantic search: {agent.config.enable_semantic_search}")
    
    # Test conversation with memory
    response = agent.invoke(
        {"messages": [HumanMessage(content="Store my preference for Python programming")]},
        config={"configurable": {"thread_id": "comprehensive_test"}}
    )
    print(f"Agent response: {response['messages'][-1].content}")


# =============================================================================
# MAIN EXAMPLES EXECUTION
# =============================================================================

def main():
    """Run all memory examples"""
    print("🎯 COREAGENT MEMORY EXAMPLES")
    print("Demonstrating all LangGraph memory patterns")
    print("=" * 80)
    
    try:
        example_short_term_memory()
        example_long_term_memory()
        example_message_management()
        example_semantic_search()
        example_session_memory()
        example_ttl_memory()
        example_comprehensive_memory()
        
        print("\n" + "=" * 80)
        print("🎉 ALL MEMORY EXAMPLES COMPLETED!")
        print("   The CoreAgent framework supports all LangGraph memory patterns!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("   Check dependencies and configuration")


if __name__ == "__main__":
    main()