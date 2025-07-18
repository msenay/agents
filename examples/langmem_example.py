"""
LangMem Memory Management Example

This file demonstrates how to use LangMem for advanced memory management
in CoreAgent framework, including short-term, long-term, and combined memory.
"""

import sys
import os
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import CoreAgent, AgentConfig, create_langmem_agent, LANGMEM_AVAILABLE
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool


# Mock model for demonstration
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        content = f"Mock response to: {messages[-1].content if messages else 'empty'}"
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "mock"


@tool
def sample_tool(query: str) -> str:
    """A sample tool for testing memory with tools"""
    return f"Tool processed: {query}"


def demo_langmem_availability():
    """Check LangMem availability and features"""
    print("\n📦 LANGMEM AVAILABILITY")
    print("=" * 40)
    
    print(f"🔍 LangMem Available: {'✅ Yes' if LANGMEM_AVAILABLE else '❌ No'}")
    
    if LANGMEM_AVAILABLE:
        try:
            from langmem import ShortTermMemory, LongTermMemory
            from langmem.short_term import SummarizationNode, RunningSummary
            print("✅ ShortTermMemory: Available")
            print("✅ LongTermMemory: Available") 
            print("✅ SummarizationNode: Available")
            print("✅ RunningSummary: Available")
        except ImportError as e:
            print(f"⚠️ Partial import error: {e}")
    else:
        print("📥 To enable LangMem functionality:")
        print("   pip install -U langmem")


def demo_memory_types():
    """Demonstrate different LangMem memory types"""
    print("\n🧠 LANGMEM MEMORY TYPES")
    print("=" * 40)
    
    memory_types = {
        "langmem_short": {
            "description": "Short-term memory with summarization",
            "use_case": "Multi-turn conversations, context retention",
            "features": ["Message summarization", "Token limit management", "Context preservation"]
        },
        "langmem_long": {
            "description": "Long-term memory for persistent storage",
            "use_case": "Cross-session information storage",
            "features": ["Persistent storage", "User profiles", "Historical data"]
        },
        "langmem_combined": {
            "description": "Both short-term and long-term memory",
            "use_case": "Complete memory management solution",
            "features": ["All short-term features", "All long-term features", "Unified interface"]
        }
    }
    
    for memory_type, info in memory_types.items():
        print(f"\n🔹 {memory_type.upper().replace('_', ' ')}:")
        print(f"   Description: {info['description']}")
        print(f"   Use case: {info['use_case']}")
        print(f"   Features: {', '.join(info['features'])}")


def demo_langmem_agent_creation():
    """Demonstrate creating agents with different LangMem configurations"""
    print("\n🤖 LANGMEM AGENT CREATION")
    print("=" * 40)
    
    model = MockChatModel()
    tools = [sample_tool]
    
    # Test different memory types
    memory_configs = [
        {
            "memory_type": "langmem_short",
            "max_tokens": 256,
            "max_summary_tokens": 64,
            "name": "ShortMemAgent"
        },
        {
            "memory_type": "langmem_long", 
            "max_tokens": 512,
            "max_summary_tokens": 128,
            "name": "LongMemAgent"
        },
        {
            "memory_type": "langmem_combined",
            "max_tokens": 384,
            "max_summary_tokens": 96,
            "name": "CombinedMemAgent"
        }
    ]
    
    agents = []
    for config in memory_configs:
        print(f"\n📝 Creating {config['name']}:")
        
        # Create agent with manual configuration
        agent_config = AgentConfig(
            name=config["name"],
            model=model,
            tools=tools,
            system_prompt=f"You are an assistant with {config['memory_type']} memory.",
            enable_memory=True,
            memory_type=config["memory_type"],
            langmem_max_tokens=config["max_tokens"],
            langmem_max_summary_tokens=config["max_summary_tokens"],
            langmem_enable_summarization=True
        )
        
        agent = CoreAgent(agent_config)
        agents.append(agent)
        
        print(f"✅ Agent created: {agent.config.name}")
        print(f"   Memory type: {agent.config.memory_type}")
        print(f"   Max tokens: {agent.config.langmem_max_tokens}")
        print(f"   LangMem support: {agent.has_langmem_support()}")
        
        # Display memory summary
        memory_summary = agent.get_memory_summary()
        print(f"   Memory summary: {memory_summary}")
    
    return agents


def demo_langmem_factory_function():
    """Demonstrate LangMem factory function"""
    print("\n🏭 LANGMEM FACTORY FUNCTION")
    print("=" * 40)
    
    model = MockChatModel()
    tools = [sample_tool]
    
    # Create agent using factory function
    agent = create_langmem_agent(
        model=model,
        tools=tools,
        memory_type="langmem_combined",
        max_tokens=512,
        max_summary_tokens=128,
        enable_summarization=True,
        prompt="You are an advanced assistant with comprehensive memory capabilities."
    )
    
    print(f"✅ LangMem agent created via factory: {agent.config.name}")
    print(f"📊 Configuration:")
    print(f"   Memory type: {agent.config.memory_type}")
    print(f"   Max tokens: {agent.config.langmem_max_tokens}")
    print(f"   Max summary tokens: {agent.config.langmem_max_summary_tokens}")
    print(f"   Summarization enabled: {agent.config.langmem_enable_summarization}")
    
    return agent


def demo_memory_configuration():
    """Demonstrate advanced memory configuration options"""
    print("\n⚙️ MEMORY CONFIGURATION OPTIONS")
    print("=" * 40)
    
    model = MockChatModel()
    
    # Advanced configuration example
    config = AgentConfig(
        name="AdvancedMemoryAgent",
        model=model,
        system_prompt="You are an assistant with advanced memory configuration.",
        enable_memory=True,
        memory_type="langmem_combined",
        
        # LangMem specific settings
        langmem_max_tokens=1024,        # Higher token limit
        langmem_max_summary_tokens=256,  # More detailed summaries
        langmem_enable_summarization=True,
        
        # Other settings
        enable_streaming=True,
        enable_human_feedback=False
    )
    
    agent = CoreAgent(config)
    
    print("📝 Advanced Memory Configuration:")
    memory_summary = agent.get_memory_summary()
    for key, value in memory_summary.items():
        print(f"   {key}: {value}")
    
    return agent


def demo_summarization_features():
    """Demonstrate message summarization features"""
    print("\n📄 SUMMARIZATION FEATURES")
    print("=" * 40)
    
    if not LANGMEM_AVAILABLE:
        print("⚠️ LangMem not available - showing conceptual example")
        return
    
    model = MockChatModel()
    
    # Create agent with summarization enabled
    agent = create_langmem_agent(
        model=model,
        memory_type="langmem_short",
        max_tokens=384,
        max_summary_tokens=128,
        enable_summarization=True
    )
    
    print("✅ Summarization Features:")
    print("   🔸 Automatic message summarization when token limit reached")
    print("   🔸 Configurable token limits for context and summary")
    print("   🔸 Preservation of important conversation context")
    print("   🔸 Integration with LangGraph prebuilt agents")
    
    print(f"\n📊 Current Configuration:")
    print(f"   Max tokens: {agent.config.langmem_max_tokens}")
    print(f"   Max summary tokens: {agent.config.langmem_max_summary_tokens}")
    print(f"   Summarization enabled: {agent.config.langmem_enable_summarization}")


def demo_memory_integration():
    """Demonstrate memory integration with agent workflows"""
    print("\n🔄 MEMORY INTEGRATION")
    print("=" * 40)
    
    model = MockChatModel()
    
    agent = create_langmem_agent(
        model=model,
        tools=[sample_tool],
        memory_type="langmem_combined"
    )
    
    print("✅ Memory Integration Features:")
    print("   🔸 Automatic memory initialization")
    print("   🔸 Integration with LangGraph checkpointers")
    print("   🔸 Support for both short and long-term storage")
    print("   🔸 Tool state persistence")
    print("   🔸 Cross-session data continuity")
    
    # Show status with memory information
    status = agent.get_status()
    print(f"\n📈 Agent Status:")
    print(f"   Memory enabled: {status['features']['memory']}")
    print(f"   Memory type: {status['memory_type']}")
    print(f"   LangMem support: {status['langmem_support']}")


def demo_best_practices():
    """Demonstrate LangMem best practices"""
    print("\n🎯 LANGMEM BEST PRACTICES")
    print("=" * 40)
    
    practices = [
        "🔸 Use langmem_short for conversational agents with context retention",
        "🔸 Use langmem_long for user profile and preference storage",
        "🔸 Use langmem_combined for comprehensive memory management",
        "🔸 Configure appropriate token limits based on your LLM's context window",
        "🔸 Enable summarization for long conversations to preserve context",
        "🔸 Set summary token limits to balance detail and efficiency",
        "🔸 Test memory behavior with your specific use cases",
        "🔸 Monitor memory usage and performance in production",
        "🔸 Consider memory cleanup strategies for long-running applications",
        "🔸 Use thread_id for session-based memory separation"
    ]
    
    for practice in practices:
        print(f"   {practice}")


def demo_usage_examples():
    """Show practical usage examples"""
    print("\n💡 USAGE EXAMPLES")
    print("=" * 40)
    
    examples = {
        "Conversational Assistant": {
            "memory_type": "langmem_short",
            "max_tokens": 512,
            "use_case": "Multi-turn conversations with context retention"
        },
        "Customer Support": {
            "memory_type": "langmem_combined",
            "max_tokens": 1024,
            "use_case": "Customer history + conversation context"
        },
        "Personal Assistant": {
            "memory_type": "langmem_long",
            "max_tokens": 256,
            "use_case": "User preferences and long-term planning"
        },
        "Educational Tutor": {
            "memory_type": "langmem_combined", 
            "max_tokens": 768,
            "use_case": "Student progress tracking + lesson context"
        }
    }
    
    for name, config in examples.items():
        print(f"\n🔹 {name}:")
        print(f"   Memory type: {config['memory_type']}")
        print(f"   Max tokens: {config['max_tokens']}")
        print(f"   Use case: {config['use_case']}")


async def run_all_demos():
    """Run all LangMem demonstrations"""
    print("🧠 CoreAgent LangMem Memory Management Demo")
    print("=" * 60)
    
    # Run demos
    demo_langmem_availability()
    demo_memory_types()
    demo_langmem_agent_creation()
    demo_langmem_factory_function()
    demo_memory_configuration()
    demo_summarization_features()
    demo_memory_integration()
    demo_best_practices()
    demo_usage_examples()
    
    print("\n" + "=" * 60)
    print("🎉 LangMem demo completed!")
    
    if not LANGMEM_AVAILABLE:
        print("\n📥 To enable full LangMem functionality:")
        print("pip install -U langmem")
        print("\nLangMem provides:")
        print("• Advanced message summarization")
        print("• Token-aware memory management")
        print("• Short-term and long-term memory")
        print("• Integration with LangGraph agents")


if __name__ == "__main__":
    asyncio.run(run_all_demos())