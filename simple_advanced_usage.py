#!/usr/bin/env python3
"""
🚀 Simple Advanced Usage Examples
=================================

Kolay kullanım örnekleri Advanced Ultimate Coder Agent için.

Features:
- ⚡ Single Agent Creation (optimized tools)
- 🎼 Multi-Agent System (supervisor pattern)
- 🛠️ Tool Selection Strategies
- 📊 Quality Assessment

Usage:
    python simple_advanced_usage.py
"""

from advanced_ultimate_coder import AdvancedUltimateCoderAgent, AdvancedAgentRequest


def create_single_agent_example():
    """Basit tek agent yaratma örneği"""
    print("🎯 EXAMPLE 1: Single Agent with Smart Tool Selection")
    print("=" * 60)
    
    # Initialize the advanced coder
    coder = AdvancedUltimateCoderAgent(
        api_key="your-openai-api-key-here"  # Real key için değiştir
    )
    
    # Create request
    request = AdvancedAgentRequest(
        task="Create a PDF document processor that extracts text and images",
        api_key="your-openai-api-key-here",
        
        # Single agent options
        multiple_agent=False,
        use_existing_tools="select_intelligently",  # LLM tool seçimi
        complexity="intermediate",
        requirements=[
            "Handle multiple PDF formats",
            "Extract text and images separately", 
            "Export to JSON format",
            "Progress tracking"
        ]
    )
    
    # Create the agent
    result = coder.create_agent(request)
    
    # Show results
    if result.success:
        print(f"✅ PDF Processor Agent created successfully!")
        print(f"   🛠️ Tools selected: {len(result.tools_used)} tools")
        print(f"   📊 Quality Score: {result.quality_score:.2f}")
        print(f"   ⚙️ Complexity: {result.complexity_score:.2f}")
        print(f"   ⏱️ Creation Time: {result.creation_time:.2f}s")
        print(f"   💾 Files: {len(result.file_paths or [])}")
        
        if result.tools_used:
            print(f"\n   🔧 Selected Tools:")
            for tool in result.tools_used[:8]:  # Show first 8
                print(f"      - {tool}")
            if len(result.tools_used) > 8:
                print(f"      ... and {len(result.tools_used) - 8} more")
    else:
        print(f"❌ Failed: {result.errors}")


def create_multi_agent_example():
    """Multi-agent sistem örneği"""
    print("\n🎼 EXAMPLE 2: Multi-Agent System with Supervisor")
    print("=" * 60)
    
    coder = AdvancedUltimateCoderAgent(
        api_key="your-openai-api-key-here"
    )
    
    request = AdvancedAgentRequest(
        task="Build a comprehensive data science workflow system",
        api_key="your-openai-api-key-here",
        
        # Multi-agent options
        multiple_agent=True,
        use_existing_tools="use_all",  # Tüm tool'ları kullan
        supervisor_pattern=True,
        max_agents=3,
        complexity="advanced",
        requirements=[
            "Data ingestion and cleaning",
            "Statistical analysis and modeling",
            "Visualization and reporting",
            "REST API for results"
        ]
    )
    
    result = coder.create_agent(request)
    
    if result.success:
        print(f"✅ Multi-Agent Data Science System created!")
        print(f"   🤖 Agents: {len(result.agents or [])} specialized agents")
        print(f"   🎯 Supervisor: {'✅' if result.supervisor_code else '❌'}")
        print(f"   🤝 Coordination: {'✅' if result.coordination_code else '❌'}")
        print(f"   📊 Average Quality: {result.quality_score:.2f}")
        print(f"   💾 Files Generated: {len(result.file_paths or [])}")
        
        if result.agents:
            print(f"\n   📋 Agent Breakdown:")
            for i, agent in enumerate(result.agents, 1):
                print(f"      {i}. {agent['name']}")
                print(f"         Type: {agent['type']}")
                print(f"         Task: {agent['task'][:50]}...")
                print(f"         Quality: {agent['quality']:.2f}")
                print(f"         Tools: {len(agent['tools'])} tools")
    else:
        print(f"❌ Failed: {result.errors}")


def create_custom_tools_example():
    """Custom tool selection örneği"""
    print("\n🛠️ EXAMPLE 3: Custom Tool Selection")
    print("=" * 60)
    
    coder = AdvancedUltimateCoderAgent(
        api_key="your-openai-api-key-here"
    )
    
    # Specify exact tools you want
    custom_tools = [
        "fastapi",      # Web framework
        "sqlalchemy",   # Database ORM
        "redis-py",     # Caching
        "pydantic",     # Data validation
        "pytest",       # Testing
        "logging"       # Logging
    ]
    
    request = AdvancedAgentRequest(
        task="Create a high-performance user authentication API",
        api_key="your-openai-api-key-here",
        
        # Custom tool options
        multiple_agent=False,
        use_existing_tools="none",  # Hiç otomatik seçim yapma
        custom_tools=custom_tools,  # Sadece bunları kullan
        complexity="advanced",
        requirements=[
            "JWT token authentication",
            "Redis session storage",
            "Rate limiting",
            "Input validation",
            "Comprehensive testing"
        ]
    )
    
    result = coder.create_agent(request)
    
    if result.success:
        print(f"✅ Authentication API Agent created!")
        print(f"   🎯 Used ONLY specified tools: {len(result.tools_used)}")
        print(f"   📊 Quality: {result.quality_score:.2f}")
        print(f"   ⚙️ Complexity: {result.complexity_score:.2f}")
        
        print(f"\n   🔧 Tools Used (exactly as specified):")
        for tool in result.tools_used:
            print(f"      ✓ {tool}")
    else:
        print(f"❌ Failed: {result.errors}")


def comparison_example():
    """Tool selection stratejilerini karşılaştır"""
    print("\n📊 EXAMPLE 4: Tool Selection Strategy Comparison")
    print("=" * 60)
    
    coder = AdvancedUltimateCoderAgent(
        api_key="your-openai-api-key-here"
    )
    
    task = "Create a web scraper with data analysis capabilities"
    
    strategies = [
        ("select_intelligently", "LLM selects best tools"),
        ("use_all", "Use all available tools"),
        ("none", "No automatic selection (custom only)")
    ]
    
    results = []
    
    for strategy, description in strategies:
        print(f"\n🔄 Testing strategy: {strategy}")
        print(f"   Description: {description}")
        
        custom_tools = ["requests", "pandas", "matplotlib"] if strategy == "none" else None
        
        request = AdvancedAgentRequest(
            task=task,
            api_key="your-openai-api-key-here",
            multiple_agent=False,
            use_existing_tools=strategy,
            custom_tools=custom_tools,
            complexity="intermediate"
        )
        
        result = coder.create_agent(request)
        
        if result.success:
            results.append({
                "strategy": strategy,
                "tools_count": len(result.tools_used or []),
                "quality": result.quality_score,
                "creation_time": result.creation_time,
                "tools_selected_by": result.tools_selected_by
            })
            
            print(f"   ✅ Success: {len(result.tools_used or [])} tools, Quality: {result.quality_score:.2f}")
        else:
            print(f"   ❌ Failed")
    
    # Comparison summary
    if results:
        print(f"\n📈 STRATEGY COMPARISON SUMMARY:")
        print(f"{'Strategy':<20} {'Tools':<8} {'Quality':<8} {'Time':<8} {'Selection'}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['strategy']:<20} {r['tools_count']:<8} {r['quality']:<8.2f} {r['creation_time']:<8.2f} {r['tools_selected_by']}")


def interactive_example():
    """Interactive agent yaratma"""
    print("\n🤖 EXAMPLE 5: Interactive Agent Creation")
    print("=" * 60)
    
    coder = AdvancedUltimateCoderAgent(
        api_key="your-openai-api-key-here"
    )
    
    print("Let's create a custom agent together!")
    
    # Get user input
    task = input("\n📝 What should your agent do? ").strip()
    if not task:
        task = "Create a file organizer that sorts files by type and date"
    
    # Multi-agent choice
    multi_choice = input("🎼 Multi-agent system? (y/N): ").strip().lower()
    multiple_agent = multi_choice in ['y', 'yes']
    
    # Tool selection strategy
    print("\n🛠️ Tool selection strategy:")
    print("  1. select_intelligently - Let AI choose the best tools")
    print("  2. use_all - Use all available tools")
    print("  3. custom - Specify your own tools")
    
    tool_choice = input("Choose (1/2/3): ").strip()
    
    if tool_choice == "1":
        use_existing_tools = "select_intelligently"
        custom_tools = None
    elif tool_choice == "2":
        use_existing_tools = "use_all"
        custom_tools = None
    else:  # custom
        use_existing_tools = "none"
        tools_input = input("Enter tools (comma-separated): ").strip()
        custom_tools = [t.strip() for t in tools_input.split(",") if t.strip()] if tools_input else ["logging", "pathlib"]
    
    # Complexity
    complexity_input = input("\n⚙️ Complexity (basic/intermediate/advanced): ").strip().lower()
    complexity = complexity_input if complexity_input in ['basic', 'intermediate', 'advanced'] else 'intermediate'
    
    print(f"\n🚀 Creating your custom agent...")
    print(f"   Task: {task}")
    print(f"   Multi-agent: {multiple_agent}")
    print(f"   Tool strategy: {use_existing_tools}")
    print(f"   Complexity: {complexity}")
    
    request = AdvancedAgentRequest(
        task=task,
        api_key="your-openai-api-key-here",
        multiple_agent=multiple_agent,
        use_existing_tools=use_existing_tools,
        custom_tools=custom_tools,
        complexity=complexity,
        supervisor_pattern=True if multiple_agent else False
    )
    
    result = coder.create_agent(request)
    
    if result.success:
        print(f"\n🎉 Your agent was created successfully!")
        
        if multiple_agent and result.agents:
            print(f"   🤖 Multi-Agent System:")
            print(f"      Agents: {len(result.agents)}")
            print(f"      Supervisor: {'✅' if result.supervisor_code else '❌'}")
            print(f"      Average Quality: {result.quality_score:.2f}")
            
            for agent in result.agents:
                print(f"         - {agent['name']}: {agent['task'][:40]}...")
        else:
            print(f"   📊 Single Agent:")
            print(f"      Quality: {result.quality_score:.2f}")
            print(f"      Tools: {len(result.tools_used or [])} selected")
            print(f"      Code: {result.code_length if hasattr(result, 'code_length') else len(result.agent_code)} characters")
        
        print(f"   💾 Files: {len(result.file_paths or [])} generated")
        print(f"   ⏱️ Time: {result.creation_time:.2f} seconds")
        
        if result.file_paths:
            print(f"\n   📁 Generated Files:")
            for file_path in result.file_paths:
                print(f"      📄 {file_path}")
    else:
        print(f"❌ Agent creation failed: {result.errors}")


def main():
    """Ana demo fonksiyonu"""
    print("🚀 ADVANCED ULTIMATE CODER - SIMPLE USAGE EXAMPLES")
    print("=" * 70)
    print("Gelişmiş seçeneklerle agent yaratma örnekleri")
    print()
    
    examples = [
        ("1", "Single Agent (Smart Tools)", create_single_agent_example),
        ("2", "Multi-Agent System", create_multi_agent_example),
        ("3", "Custom Tool Selection", create_custom_tools_example),
        ("4", "Strategy Comparison", comparison_example),
        ("5", "Interactive Creation", interactive_example),
        ("A", "Run All Examples", None)
    ]
    
    print("Available examples:")
    for code, name, _ in examples:
        print(f"  {code}: {name}")
    
    choice = input(f"\nWhich example to run? (1-5, A, or Enter=1): ").strip().upper()
    
    if not choice:
        choice = "1"
    
    if choice == "A":
        # Run all examples except interactive
        for code, name, func in examples:
            if func and code != "5":  # Skip interactive
                print(f"\n{'='*70}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Example {name} failed: {e}")
                print("\n⏱️ Press Enter to continue...")
                input()
    else:
        # Run selected example
        for code, name, func in examples:
            if code == choice and func:
                print(f"\n{'='*70}")
                try:
                    func()
                except Exception as e:
                    print(f"❌ Example {name} failed: {e}")
                break
        else:
            print("❌ Invalid selection")
    
    print(f"\n🎊 Examples completed!")
    print(f"\n💡 Key Features Available:")
    print(f"  ✅ multiple_agent=False/True - Single vs Multi-agent")
    print(f"  ✅ use_existing_tools='select_intelligently' - LLM tool selection")
    print(f"  ✅ use_existing_tools='use_all' - Use all 60+ tools")
    print(f"  ✅ use_existing_tools='none' + custom_tools - Exact tool control")
    print(f"  ✅ supervisor_pattern=True - Multi-agent coordination")
    print(f"  ✅ Quality assessment and file generation")
    print(f"\n🚀 Your advanced agent creation system is ready!")


if __name__ == "__main__":
    main()