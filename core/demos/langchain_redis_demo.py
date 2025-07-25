#!/usr/bin/env python3
"""
Demo using langchain-redis (modern Redis implementation)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["AZURE_OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
os.environ["OPENAI_API_KEY"] = os.environ["AZURE_OPENAI_API_KEY"]

from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
import redis

REDIS_URL = "redis://:redis_password@localhost:6379"


def test_langchain_redis():
    """Test langchain-redis for conversation memory"""
    print("\n" + "="*60)
    print("🚀 LANGCHAIN-REDIS DEMO")
    print("="*60)
    
    # Check Redis
    try:
        r = redis.from_url(REDIS_URL)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis error: {e}")
        return
    
    # Create message history
    history = RedisChatMessageHistory(
        session_id="user_123",
        redis_url=REDIS_URL
    )
    
    # Clear previous messages
    history.clear()
    
    # Add messages
    print("\n📝 Adding messages to history...")
    history.add_message(HumanMessage(content="Hi! My name is Alice."))
    history.add_message(AIMessage(content="Hello Alice! Nice to meet you."))
    history.add_message(HumanMessage(content="I work as a data scientist."))
    history.add_message(AIMessage(content="That's great! Data science is a fascinating field."))
    
    # Retrieve messages
    print("\n📖 Retrieving conversation history:")
    messages = history.messages
    for i, msg in enumerate(messages):
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"{i+1}. {role}: {msg.content}")
    
    # Test with model
    print("\n🤖 Testing with Azure OpenAI...")
    model = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version="2023-12-01-preview",
        azure_deployment="gpt4o",
        temperature=0.7
    )
    
    # Ask a question based on history
    messages.append(HumanMessage(content="What's my name and profession?"))
    response = model.invoke(messages)
    print(f"\nUser: What's my name and profession?")
    print(f"AI: {response.content}")
    
    # Test different session
    print("\n🔄 Testing different session...")
    history2 = RedisChatMessageHistory(
        session_id="user_456",
        redis_url=REDIS_URL
    )
    
    if not history2.messages:
        print("✅ Different session has no messages (correct isolation)")
    
    print("\n✅ DEMO COMPLETED!")
    print("\n📚 Summary:")
    print("- langchain-redis provides simple chat message history")
    print("- Works with Python 3.12+")
    print("- No complex index requirements")
    print("- Good for conversation memory")


if __name__ == "__main__":
    test_langchain_redis()