#!/usr/bin/env python3
"""
MongoDB Usage Examples for CoreAgent Framework

This file demonstrates various ways to use MongoDB with the CoreAgent framework:
1. Basic MongoDB memory setup
2. Short-term memory with MongoDB
3. Long-term memory with MongoDB
4. Comprehensive memory (both short and long term)
5. TTL memory for automatic data expiration
6. Session-based memory for multi-agent collaboration
7. Semantic search with MongoDB and embeddings

Requirements:
- MongoDB running (local or Atlas)
- pip install pymongo langgraph-checkpoint-mongodb langgraph-store-mongodb
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/coreagent")

# Example tools
class CalculatorInput(BaseModel):
    a: float = Field(description="First number")
    b: float = Field(description="Second number")
    operation: str = Field(description="Operation: add, subtract, multiply, divide")

class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform basic mathematical operations"
    args_schema = CalculatorInput
    
    def _run(self, a: float, b: float, operation: str) -> str:
        if operation == "add":
            return f"Result: {a} + {b} = {a + b}"
        elif operation == "subtract":
            return f"Result: {a} - {b} = {a - b}"
        elif operation == "multiply":
            return f"Result: {a} × {b} = {a * b}"
        elif operation == "divide":
            return f"Result: {a} ÷ {b} = {a / b if b != 0 else 'undefined'}"
        else:
            return "Unsupported operation"

def example_1_basic_mongodb_memory():
    """Example 1: Basic MongoDB memory setup"""
    print("🚀 Example 1: Basic MongoDB Memory Setup")
    print("-" * 50)
    
    from core_agent import create_memory_agent
    
    # Create a basic agent with MongoDB memory
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    tools = [CalculatorTool()]
    
    agent = create_memory_agent(
        model=model,
        name="MongoBasicAgent",
        short_term_memory="mongodb",
        long_term_memory="mongodb",
        mongodb_url=MONGODB_URL,
        tools=tools,
        system_prompt="You are a helpful assistant with MongoDB memory. You can remember our conversation and perform calculations."
    )
    
    # Test the agent
    config = {"configurable": {"thread_id": "basic_mongo_example"}}
    
    print("💬 User: Remember that my favorite programming language is Python")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Remember that my favorite programming language is Python"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n💬 User: What's my favorite programming language?")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What's my favorite programming language?"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n✅ Basic MongoDB memory working!\n")

def example_2_short_term_mongodb():
    """Example 2: Short-term memory only with MongoDB"""
    print("🚀 Example 2: Short-term MongoDB Memory Only")
    print("-" * 50)
    
    from core_agent import create_short_term_memory_agent
    
    # Create agent with only short-term MongoDB memory
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    agent = create_short_term_memory_agent(
        model=model,
        name="MongoShortTermAgent",
        memory_backend="mongodb",
        mongodb_url=MONGODB_URL,
        system_prompt="You are an assistant with short-term MongoDB memory for this conversation."
    )
    
    # Test conversation continuity
    config = {"configurable": {"thread_id": "short_term_mongo_example"}}
    
    print("💬 User: My name is Alice and I work as a data scientist")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "My name is Alice and I work as a data scientist"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n💬 User: What do you know about me?")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What do you know about me?"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n✅ Short-term MongoDB memory working!\n")

def example_3_long_term_mongodb():
    """Example 3: Long-term memory only with MongoDB"""
    print("🚀 Example 3: Long-term MongoDB Memory Only")
    print("-" * 50)
    
    from core_agent import create_long_term_memory_agent
    
    # Create agent with only long-term MongoDB memory
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    agent = create_long_term_memory_agent(
        model=model,
        name="MongoLongTermAgent",
        memory_backend="mongodb",
        mongodb_url=MONGODB_URL,
        enable_semantic_search=True,
        system_prompt="You are an assistant with long-term MongoDB memory that persists across sessions."
    )
    
    # Test persistent memory
    config = {"configurable": {"thread_id": "long_term_mongo_example"}}
    
    print("💬 User: Store this important information: The quarterly sales target is $500K")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Store this important information: The quarterly sales target is $500K"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Use a different thread to simulate a new session
    new_config = {"configurable": {"thread_id": "long_term_mongo_example_2"}}
    
    print("\n💬 User (new session): What was the quarterly sales target we discussed?")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What was the quarterly sales target we discussed?"}]
    }, new_config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n✅ Long-term MongoDB memory working!\n")

def example_4_comprehensive_mongodb():
    """Example 4: Comprehensive MongoDB memory (short + long term)"""
    print("🚀 Example 4: Comprehensive MongoDB Memory")
    print("-" * 50)
    
    from core_agent import create_memory_agent
    
    # Create agent with both short and long term MongoDB memory
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    tools = [CalculatorTool()]
    
    agent = create_memory_agent(
        model=model,
        name="MongoComprehensiveAgent",
        short_term_memory="mongodb",
        long_term_memory="mongodb",
        mongodb_url=MONGODB_URL,
        tools=tools,
        enable_semantic_search=True,
        enable_memory_tools=True,
        system_prompt="You are an advanced assistant with comprehensive MongoDB memory capabilities."
    )
    
    # Test comprehensive memory
    config = {"configurable": {"thread_id": "comprehensive_mongo_example"}}
    
    print("💬 User: Calculate 245 * 8 and remember this calculation for future reference")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Calculate 245 * 8 and remember this calculation for future reference"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n💬 User: What calculation did we just perform?")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What calculation did we just perform?"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n✅ Comprehensive MongoDB memory working!\n")

def example_5_ttl_mongodb():
    """Example 5: TTL memory with MongoDB for auto-expiration"""
    print("🚀 Example 5: TTL MongoDB Memory (Auto-expiration)")
    print("-" * 50)
    
    from core_agent import create_ttl_memory_agent
    
    # Create agent with TTL MongoDB memory
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    agent = create_ttl_memory_agent(
        model=model,
        name="MongoTTLAgent",
        memory_backend="mongodb",
        ttl_minutes=60,  # Data expires after 1 hour
        mongodb_url=MONGODB_URL,
        system_prompt="You are an assistant with time-limited MongoDB memory that automatically expires."
    )
    
    # Test TTL memory
    config = {"configurable": {"thread_id": "ttl_mongo_example"}}
    
    print("💬 User: Remember this temporary meeting: Team standup at 2 PM today")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Remember this temporary meeting: Team standup at 2 PM today"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n💬 User: What meeting do we have today?")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "What meeting do we have today?"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n✅ TTL MongoDB memory working (will auto-expire in 60 minutes)!\n")

def example_6_session_mongodb():
    """Example 6: Session-based MongoDB memory for multi-agent collaboration"""
    print("🚀 Example 6: Session-based MongoDB Memory (Multi-agent)")
    print("-" * 50)
    
    from core_agent import create_session_agent
    import time
    
    # Shared session ID for collaboration
    session_id = f"mongo_collab_{int(time.time())}"
    
    # Create multiple agents with shared session memory
    model1 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    model2 = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    
    # Agent 1: Developer
    developer_agent = create_session_agent(
        model=model1,
        session_id=session_id,
        name="MongoDevAgent",
        memory_namespace="developer",
        mongodb_url=MONGODB_URL,
        tools=[CalculatorTool()],
        system_prompt="You are a developer agent. You write code and share it with the team via session memory."
    )
    
    # Agent 2: Code Reviewer
    reviewer_agent = create_session_agent(
        model=model2,
        session_id=session_id,
        name="MongoReviewAgent",
        memory_namespace="reviewer",
        mongodb_url=MONGODB_URL,
        system_prompt="You are a code reviewer agent. You review code shared by developers via session memory."
    )
    
    # Test collaboration
    dev_config = {"configurable": {"thread_id": "dev_thread"}}
    review_config = {"configurable": {"thread_id": "review_thread"}}
    
    print("👨‍💻 Developer: Write a function to calculate compound interest")
    dev_response = developer_agent.invoke({
        "messages": [{"role": "user", "content": "Write a Python function to calculate compound interest and store it in our shared session"}]
    }, dev_config)
    print(f"🤖 Developer Agent: {dev_response['messages'][-1].content}")
    
    print("\n🔍 Reviewer: Review the compound interest function")
    review_response = reviewer_agent.invoke({
        "messages": [{"role": "user", "content": "Review the compound interest function that was just written by the developer"}]
    }, review_config)
    print(f"🤖 Reviewer Agent: {review_response['messages'][-1].content}")
    
    print(f"\n✅ Session-based MongoDB memory working! (Session ID: {session_id})\n")

def example_7_semantic_search_mongodb():
    """Example 7: Semantic search with MongoDB and embeddings"""
    print("🚀 Example 7: Semantic Search with MongoDB")
    print("-" * 50)
    
    from core_agent import create_semantic_search_agent
    
    # Create agent with semantic search capabilities
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)
    
    agent = create_semantic_search_agent(
        model=model,
        name="MongoSemanticAgent",
        memory_backend="mongodb",
        mongodb_url=MONGODB_URL,
        embedding_model="openai:text-embedding-3-small",
        enable_memory_tools=True,
        system_prompt="You are an assistant with semantic search capabilities using MongoDB and embeddings."
    )
    
    # Test semantic search
    config = {"configurable": {"thread_id": "semantic_mongo_example"}}
    
    # Store diverse information
    print("💬 User: Store information about machine learning algorithms")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Store this: Random Forest is an ensemble learning method that uses multiple decision trees"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n💬 User: Store information about neural networks")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Store this: Neural networks are computing systems inspired by biological neural networks"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    # Test semantic search
    print("\n💬 User: Find information about tree-based algorithms")
    response = agent.invoke({
        "messages": [{"role": "user", "content": "Find information related to tree-based algorithms"}]
    }, config)
    print(f"🤖 Agent: {response['messages'][-1].content}")
    
    print("\n✅ Semantic search with MongoDB working!\n")

def main():
    """Run all MongoDB usage examples"""
    print("🎯 MongoDB Usage Examples for CoreAgent Framework")
    print("=" * 60)
    print("Note: Make sure MongoDB is running and OPENAI_API_KEY is set")
    print("=" * 60)
    
    try:
        # Check if MongoDB is available
        import pymongo
        client = pymongo.MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print("✅ MongoDB connection successful\n")
        
        # Run examples
        example_1_basic_mongodb_memory()
        example_2_short_term_mongodb()
        example_3_long_term_mongodb()
        example_4_comprehensive_mongodb()
        example_5_ttl_mongodb()
        example_6_session_mongodb()
        example_7_semantic_search_mongodb()
        
        print("🎉 All MongoDB examples completed successfully!")
        print("🔍 Check your MongoDB database to see the stored data")
        
    except ImportError:
        print("❌ MongoDB dependencies not installed")
        print("Run: pip install pymongo langgraph-checkpoint-mongodb langgraph-store-mongodb")
        
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("💡 Make sure MongoDB is running and connection string is correct")

if __name__ == "__main__":
    main()