#!/usr/bin/env python3
"""
Simple Core Agent Examples
Demonstrates basic agent creation patterns
"""

from core import CoreAgent, AgentConfig
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import tool
from typing import List, Any


# Mock model for testing (gerçek kullanımda ChatOpenAI kullan)
class MockLLM(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], **kwargs) -> Any:
        user_msg = messages[-1].content if messages else ""
        return AIMessage(content=f"Merhaba! Mesajını aldım: '{user_msg}'")
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def invoke(self, input: Any, config=None, **kwargs) -> BaseMessage:
        if isinstance(input, list):
            return self._generate(input)
        return AIMessage(content=f"Yanıt: {input}")


# ============================================================
# EXAMPLE 1: Minimal Agent
# ============================================================
def example_1_minimal_agent():
    """The most minimal agent example"""
    print("=" * 50)
    print("EXAMPLE 1: Minimal Agent")
    print("=" * 50)
    
    # Config - sadece zorunlu parametreler
    config = AgentConfig(
        name="MinimalAgent",
        model=MockLLM()
    )
    
    # Agent oluştur
    agent = CoreAgent(config)
    
    # Kullan
    response = agent.invoke("Merhaba dünya!")
    print(f"Yanıt: {response['messages'][-1].content}")


# ============================================================
# EXAMPLE 2: Agent with System Prompt
# ============================================================
def example_2_with_prompt():
    """Agent with custom system prompt"""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: Agent with System Prompt")
    print("=" * 50)
    
    config = AgentConfig(
        name="AssistantAgent",
        model=MockLLM(),
        system_prompt="You are a helpful assistant. Always be kind and professional."
    )
    
    agent = CoreAgent(config)
    response = agent.invoke("How is the weather today?")
    print(f"Response: {response['messages'][-1].content}")


# ============================================================
# EXAMPLE 3: Agent with Tools
# ============================================================
def example_3_with_tools():
    """Agent using tools"""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: Agent with Tools")
    print("=" * 50)
    
    # Define a simple tool
    @tool
    def calculator(operation: str) -> str:
        """Performs simple math operations. Example: '2 + 2' or '10 * 5'"""
        try:
            result = eval(operation)
            return f"Result: {result}"
        except:
            return "Error: Invalid operation"
    
    @tool
    def datetime_now() -> str:
        """Returns the current date and time"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Config
    config = AgentConfig(
        name="ToolAgent",
        model=MockLLM(),
        system_prompt="You are an assistant that helps with math and time-related questions.",
        tools=[calculator, datetime_now]
    )
    
    agent = CoreAgent(config)
    print(f"Available tools: {[t.name for t in config.tools]}")


# ============================================================
# EXAMPLE 4: Agent with Memory
# ============================================================
def example_4_with_memory():
    """Agent using memory"""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: Agent with Memory")
    print("=" * 50)
    
    config = AgentConfig(
        name="MemoryAgent",
        model=MockLLM(),
        system_prompt="You are an assistant with memory.",
        
        # Memory settings
        enable_memory=True,
        memory_backend="inmemory",
        memory_types=["short_term"]  # Thread-based conversation memory
    )
    
    agent = CoreAgent(config)
    
    # Conversation in Thread 1
    print("\nThread 1:")
    agent.invoke("My name is Ahmet", config={"configurable": {"thread_id": "user_1"}})
    agent.invoke("Do you remember my name?", config={"configurable": {"thread_id": "user_1"}})
    
    # Conversation in Thread 2
    print("\nThread 2:")
    agent.invoke("I am Mehmet", config={"configurable": {"thread_id": "user_2"}})
    agent.invoke("Do you know who I am?", config={"configurable": {"thread_id": "user_2"}})


# ============================================================
# EXAMPLE 5: Streaming Agent
# ============================================================
def example_5_streaming():
    """Streaming agent"""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: Streaming Agent")
    print("=" * 50)
    
    config = AgentConfig(
        name="StreamingAgent",
        model=MockLLM(),
        enable_streaming=True
    )
    
    agent = CoreAgent(config)
    
    # Stream usage
    print("Streaming response:")
    for chunk in agent.stream("Tell me a long story"):
        # In real usage, chunks come piece by piece
        print(".", end="", flush=True)
    print("\nStreaming complete!")


# ============================================================
# EXAMPLE 6: Rate Limited Agent
# ============================================================
def example_6_rate_limited():
    """Rate limiting agent"""
    print("\n" + "=" * 50)
    print("EXAMPLE 6: Rate Limited Agent")
    print("=" * 50)
    
    config = AgentConfig(
        name="RateLimitedAgent",
        model=MockLLM(),
        
        # Rate limiting
        enable_rate_limiting=True,
        requests_per_second=2.0,  # Max 2 requests per second
        max_bucket_size=5.0
    )
    
    agent = CoreAgent(config)
    print(f"Rate limit: {config.requests_per_second} requests/second")


# ============================================================
# FACTORY PATTERN EXAMPLE
# ============================================================
class AgentFactory:
    """Factory pattern for creating agents"""
    
    @staticmethod
    def create_chatbot(name: str = "Chatbot") -> CoreAgent:
        """Create a simple chatbot"""
        return CoreAgent(AgentConfig(
            name=name,
            model=MockLLM(),
            system_prompt="You are a friendly and helpful chatbot.",
            enable_memory=True,
            memory_backend="inmemory"
        ))
    
    @staticmethod
    def create_coder(name: str = "Coder") -> CoreAgent:
        """Create a coding agent"""
        
        @tool
        def python_runner(code: str) -> str:
            """Runs Python code"""
            return "Code executed (simulation)"
        
        return CoreAgent(AgentConfig(
            name=name,
            model=MockLLM(),
            system_prompt="You are an expert Python developer.",
            tools=[python_runner],
            enable_memory=True
        ))
    
    @staticmethod
    def create_researcher(name: str = "Researcher") -> CoreAgent:
        """Create a research agent"""
        
        @tool
        def web_search(query: str) -> str:
            """Searches the web"""
            return f"Search results for '{query}' (simulation)"
        
        return CoreAgent(AgentConfig(
            name=name,
            model=MockLLM(),
            system_prompt="You are an assistant that conducts detailed research.",
            tools=[web_search],
            enable_memory=True,
            memory_types=["short_term", "long_term"]
        ))


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    print("🚀 Core Agent Simple Examples\n")
    
    # Run all examples
    example_1_minimal_agent()
    example_2_with_prompt()
    example_3_with_tools()
    example_4_with_memory()
    example_5_streaming()
    example_6_rate_limited()
    
    # Factory pattern example
    print("\n" + "=" * 50)
    print("FACTORY PATTERN EXAMPLE")
    print("=" * 50)
    
    factory = AgentFactory()
    
    # Create ready agents
    chatbot = factory.create_chatbot("ChatBot")
    coder = factory.create_coder("CodeWriter")
    researcher = factory.create_researcher("Researcher")
    
    print(f"✅ {chatbot.config.name} created")
    print(f"✅ {coder.config.name} created")
    print(f"✅ {researcher.config.name} created")
    
    print("\n✨ All examples completed!")
    print("\n📌 For real usage:")
    print("   - Use ChatOpenAI instead of MockLLM")
    print("   - Add real tools")
    print("   - Use Redis/PostgreSQL backend for production")