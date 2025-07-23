#!/usr/bin/env python3
"""
Coder Agent Demo - Specification-Based Agent Generation
======================================================

Demonstrates how CoderAgent generates agents from specifications.
"""
import os
import sys
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists"""
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        print("📋 Loading environment variables from .env file...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    os.environ[key] = value
        print("✅ Environment variables loaded!")
    else:
        print("⚠️  No .env file found. Using system environment variables.")

# Load environment variables
load_env()

from agent.coder.coder import CoderAgent


def demo_simple_agent_from_spec():
    """Demo: Generate a simple agent from specifications"""
    print("\n" + "="*60)
    print("📝 DEMO 1: Simple Agent from Specifications")
    print("="*60)
    
    coder = CoderAgent()
    
    # Agent specification (like a recipe)
    spec = """
    Agent Name: DataAnalyzer
    
    Purpose: Analyze CSV data files and generate insights
    
    Requirements:
    - Load CSV files from a given path
    - Calculate basic statistics (mean, median, mode)
    - Identify data quality issues (missing values, outliers)
    - Generate a summary report
    - Handle errors gracefully
    
    Workflow:
    1. Validate input file exists
    2. Load CSV data
    3. Analyze data quality
    4. Calculate statistics
    5. Generate report
    
    Output: JSON report with statistics and insights
    """
    
    result = coder.generate_from_spec(
        spec=spec,
        agent_type="simple",  # Default
        use_our_core=False    # Default: standalone
    )
    
    if result["success"]:
        print("✅ Agent generated successfully!")
        print(f"📊 Type: {result['agent_type']} (Standalone)")
        print(f"\n📄 Code preview (first 500 chars):")
        print("-" * 40)
        print(result["code"])
        print("-" * 40)
    else:
        print(f"❌ Error: {result['error']}")


def demo_core_agent_with_tools_from_spec():
    """Demo: Generate a Core Agent with tools from specifications"""
    print("\n" + "="*60)
    print("🛠️ DEMO 2: Core Agent with Tools from Specifications")
    print("="*60)
    
    coder = CoderAgent()
    
    # Detailed specification with tools
    spec = """
    Agent Name: ResearchAssistant
    
    Purpose: Help researchers find and summarize information from various sources
    
    Required Tools:
    - web_search: Search the internet for information
    - pdf_reader: Extract text from PDF documents
    - summarizer: Generate concise summaries
    
    Capabilities:
    - Search for academic papers and articles
    - Extract key information from PDFs
    - Summarize findings in a structured format
    - Maintain conversation context
    - Provide citations and sources
    
    Workflow:
    1. Understand research query
    2. Search for relevant sources
    3. Extract information from documents
    4. Summarize findings
    5. Format response with citations
    
    Special Requirements:
    - Use memory to track research context
    - Enable rate limiting for web searches
    - Provide confidence scores for findings
    """
    
    result = coder.generate_from_spec(
        spec=spec,
        agent_type="with_tools",
        use_our_core=True  # Use Core Agent infrastructure
    )
    
    if result["success"]:
        print("✅ Agent generated successfully!")
        print(f"📊 Type: {result['agent_type']} (Core Agent)")
        print(f"\n📄 Code preview (first 500 chars):")
        print("-" * 40)
        print(result["code"])
        print("-" * 40)
    else:
        print(f"❌ Error: {result['error']}")


def demo_multi_agent_system_from_spec():
    """Demo: Generate a multi-agent system from specifications"""
    print("\n" + "="*60)
    print("👥 DEMO 3: Multi-Agent System from Specifications")
    print("="*60)
    
    coder = CoderAgent()
    
    # Multi-agent system specification
    spec = """
    System Name: SoftwareDevelopmentTeam
    
    Purpose: Automate software development workflow with specialized agents
    
    Agents Required:
    
    1. RequirementsAnalyst:
       - Analyze user requirements
       - Create technical specifications
       - Identify edge cases
    
    2. Developer:
       - Write code based on specifications
       - Implement best practices
       - Add error handling
    
    3. Tester:
       - Write unit tests
       - Perform integration testing
       - Report bugs
    
    4. Reviewer:
       - Review code quality
       - Check for security issues
       - Suggest improvements
    
    Supervisor Logic:
    - Route requirements to analyst first
    - Pass specs from analyst to developer
    - Send code from developer to tester
    - Forward tested code to reviewer
    - Aggregate feedback and iterate if needed
    
    Communication:
    - Agents share state through supervisor
    - Each agent can request clarification
    - Final output includes all artifacts
    """
    
    result = coder.generate_from_spec(
        spec=spec,
        agent_type="multi_agent",
        use_our_core=False  # Standalone supervisor pattern
    )
    
    if result["success"]:
        print("✅ Multi-agent system generated successfully!")
        print(f"📊 Type: {result['agent_type']}")
        print(f"\n📄 Code preview (first 500 chars):")
        print("-" * 40)
        print(result["code"])
        print("-" * 40)
    else:
        print(f"❌ Error: {result['error']}")


def demo_chat_interface():
    """Demo: Interactive chat for custom specifications"""
    print("\n" + "="*60)
    print("💬 DEMO 4: Interactive Chat Interface")
    print("="*60)
    
    coder = CoderAgent()
    
    # Example: Natural language specification
    print("\n🗨️ Request: Create an email automation agent")
    response = coder.chat("""
    I need an agent that can:
    - Read emails from Gmail
    - Classify them by importance and category
    - Auto-respond to common queries
    - Forward urgent emails to specific people
    - Generate daily summary reports
    
    Please make it a simple standalone agent with clear error handling.
    """)
    
    print(f"\n🤖 Response preview (first 800 chars):")
    print("-" * 40)
    print(response)
    print("-" * 40)
    print("-" * 40)


def main():
    """Run all demos"""
    print("🚀 CODER AGENT DEMO - Specification-Based Generation")
    print("=" * 80)
    print("CoderAgent generates complete agent code from specifications (like recipes)")
    print("\nCapabilities:")
    print("- Simple agents (default)")
    print("- Agents with tools")
    print("- Multi-agent systems")
    print("- Standalone (default) or Core Agent based")
    
    try:
        # Run all demos
        print("demo_simple_agent_from_spec")
        demo_simple_agent_from_spec()
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("demo_core_agent_with_tools_from_spec")
        demo_core_agent_with_tools_from_spec()
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("demo_multi_agent_system_from_spec")
        demo_multi_agent_system_from_spec()
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("demo_chat_interface")
        demo_chat_interface()
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        print("-" * 40)
        
        print("\n" + "=" * 80)
        print("✅ All demos completed successfully!")
        print("\n💡 Key Points:")
        print("- Provide detailed specifications like a recipe")
        print("- CoderAgent extracts requirements and generates code")
        print("- Default: simple, standalone agents")
        print("- Optional: with_tools, multi_agent, use_our_core")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()