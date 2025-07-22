#!/usr/bin/env python3
"""
Coder Agent Demo
===============

Demonstrates the capabilities of CoderAgent with various examples.
"""

import sys
sys.path.insert(0, '/workspace')

from agent.coder.coder import CoderAgent

def demo_simple_agent():
    """Demo: Generate a simple standalone agent"""
    print("\n" + "="*60)
    print("📝 DEMO 1: Simple Standalone Agent")
    print("="*60)
    
    coder = CoderAgent()
    
    result = coder.generate_agent(
        template_type="simple",
        agent_name="DataProcessor",
        purpose="Process and analyze CSV data files",
        use_our_core=False  # Standalone LangGraph
    )
    
    if result["success"]:
        print(f"✅ Generated: {result['agent_name']}")
        print(f"📊 Type: {result['template_type']} (Standalone)")
        print(f"\n📄 Code preview (first 500 chars):")
        print("-" * 40)
        print(result["code"][:500] + "...")
    else:
        print(f"❌ Error: {result['error']}")


def demo_core_agent_with_tools():
    """Demo: Generate a Core Agent with tools"""
    print("\n" + "="*60)
    print("🛠️ DEMO 2: Core Agent with Tools")
    print("="*60)
    
    coder = CoderAgent()
    
    result = coder.generate_agent(
        template_type="with_tools",
        agent_name="WebResearcher",
        purpose="Research topics using web search and summarization",
        tools_needed=["web_search", "text_summarizer", "pdf_reader"],
        use_our_core=True  # Use Core Agent infrastructure
    )
    
    if result["success"]:
        print(f"✅ Generated: {result['agent_name']}")
        print(f"📊 Type: {result['template_type']} (Core Agent)")
        print(f"🔧 Tools: {result['tools']}")
        print(f"\n📄 Code preview (first 500 chars):")
        print("-" * 40)
        print(result["code"][:500] + "...")
    else:
        print(f"❌ Error: {result['error']}")


def demo_multi_agent_system():
    """Demo: Generate a multi-agent system"""
    print("\n" + "="*60)
    print("👥 DEMO 3: Multi-Agent System")
    print("="*60)
    
    coder = CoderAgent()
    
    result = coder.generate_agent(
        template_type="multi_agent",
        agent_name="DevelopmentTeam",
        purpose="Coordinate code writing, testing, and review workflow",
        tools_needed=["coder", "tester", "reviewer"],
        use_our_core=False  # Supervisor pattern
    )
    
    if result["success"]:
        print(f"✅ Generated: {result['agent_name']}")
        print(f"📊 Type: {result['template_type']}")
        print(f"👥 Sub-agents: {result['tools']}")
        print(f"\n📄 Code preview (first 500 chars):")
        print("-" * 40)
        print(result["code"][:500] + "...")
    else:
        print(f"❌ Error: {result['error']}")


def demo_chat_interface():
    """Demo: Interactive chat for custom requests"""
    print("\n" + "="*60)
    print("💬 DEMO 4: Interactive Chat Interface")
    print("="*60)
    
    coder = CoderAgent()
    
    # Example 1: Natural language request
    print("\n🗨️ Request: Create an email processor agent")
    response = coder.chat(
        "Create a simple agent called EmailProcessor that can "
        "read emails, classify them by importance, and route them "
        "to appropriate handlers. Make it standalone."
    )
    print(f"\n🤖 Response preview (first 500 chars):")
    print("-" * 40)
    print(response[:500] + "...")
    
    # Example 2: Optimization request
    print("\n🗨️ Request: Optimize existing code")
    code_snippet = '''
def process_data(data):
    result = []
    for item in data:
        if item > 0:
            result.append(item * 2)
    return result
'''
    
    response = coder.chat(
        f"Optimize this code for better performance:\n```python\n{code_snippet}\n```"
    )
    print(f"\n🤖 Response preview (first 500 chars):")
    print("-" * 40)
    print(response[:500] + "...")


def main():
    """Run all demos"""
    print("🚀 CODER AGENT DEMO")
    print("=" * 80)
    print("Demonstrating CoderAgent capabilities with 3 essential tools:")
    print("- agent_generator: Creates agent code")
    print("- optimize_agent: Optimizes code quality")
    print("- format_code: Ensures clean formatting")
    
    try:
        # Run all demos
        demo_simple_agent()
        demo_core_agent_with_tools()
        demo_multi_agent_system()
        demo_chat_interface()
        
        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("\n💡 Tips:")
        print("- Use template_type='simple' for basic agents")
        print("- Use template_type='with_tools' for tool-enabled agents")
        print("- Use template_type='multi_agent' for supervisor systems")
        print("- Set use_our_core=True to leverage Core Agent infrastructure")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()