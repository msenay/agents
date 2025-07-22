#!/usr/bin/env python3
"""
Coder Agent - Advanced AI Agent Generator with Comprehensive Toolkit
===================================================================

An intelligent agent that generates, analyzes, optimizes, tests, and deploys LangGraph agents.

Features:
- 12 specialized tools for complete agent development lifecycle
- Supports both standalone LangGraph and Core Agent based implementations  
- Intelligent tool selection and chaining for complex workflows
- Memory integration for learning from past generations
- Production-ready code generation with validation and optimization

Agent Types:
- Simple: Basic agents with state management
- With Tools: Agents with custom tool integration
- Multi-Agent: Supervisor-based multi-agent systems

Available Tools:
- Generation: agent_generator, generate_rag_agent
- Analysis: analyze_agent_code, validate_agent, optimize_agent
- Documentation: generate_agent_docs
- Testing: generate_unit_tests
- Deployment: dockerize_agent, convert_to_api
- Enhancement: add_monitoring, format_code
- Templates: save_agent_template
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

# Core Agent Infrastructure
from core.core_agent import CoreAgent
from core.config import AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

# Import all tools
from core_agents.tools import (
    create_all_coder_tools,
    get_tools_by_category,
    suggest_tools_for_task,
    create_agent_generator_tool
)


class CoderConfig:
    """Coder Agent Configuration"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = "https://oai-202-fbeta-dev.openai.azure.com/"
    OPENAI_API_KEY = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    
    # Model Parameters
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000


# AgentGeneratorInput and create_agent_generator_tool are now imported from tools.py


class CoderAgent(CoreAgent):
    """
    Advanced AI Agent Generator with Comprehensive Development Toolkit
    
    A powerful agent that handles the complete agent development lifecycle:
    - Generates standalone LangGraph or Core Agent based implementations
    - Analyzes and optimizes existing agent code
    - Creates tests, documentation, and deployment configurations
    - Intelligently chains tools for complex workflows
    
    Equipped with 12 specialized tools that the LLM can intelligently
    select and combine based on the task requirements.
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize Coder Agent with all development tools
        
        Args:
            session_id: Unique session identifier (auto-generated if not provided)
        """
        if session_id is None:
            session_id = f"coder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create Azure OpenAI model
        model = AzureChatOpenAI(
            azure_endpoint=CoderConfig.AZURE_OPENAI_ENDPOINT,
            api_key=CoderConfig.OPENAI_API_KEY,
            api_version=CoderConfig.OPENAI_API_VERSION,
            model=CoderConfig.GPT4_MODEL_NAME,
            deployment_name=CoderConfig.GPT4_DEPLOYMENT_NAME,
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Create ALL tools for maximum capability
        # The LLM will decide which tools to use based on the task
        tools = create_all_coder_tools(model)
        
        # Create configuration
        config = AgentConfig(
            name="CoderAgent",
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            enable_memory=True,  # Enable memory
            memory_backend="inmemory",  # Use InMemory backend
            memory_types=["short_term", "long_term"],  # Enable both memory types
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Initialize parent CoreAgent
        super().__init__(config)
        
        print(f"✅ Coder Agent initialized with {len(tools)} tools")
        print(f"🔧 Available tools:")
        for tool in self.config.tools:
            print(f"   - {tool.name}: {tool.description}")
    
    def _get_system_prompt(self) -> str:
        """
        Generate comprehensive system prompt with tool awareness
        
        Returns:
            System prompt that guides the LLM in tool selection and usage
        """
        return """You are an expert Coder Agent with a comprehensive toolkit for creating, analyzing, and deploying LangGraph agents.

Your expertise includes:
- Creating standalone LangGraph agents with proper state management
- Building agents with custom tools and tool integration
- Designing multi-agent systems with supervisor patterns
- Using Core Agent infrastructure when requested
- Writing clean, maintainable, production-ready code

Your available tools include:
1. GENERATION: agent_generator, generate_rag_agent - Create various types of agents
2. ANALYSIS: analyze_agent_code, validate_agent, optimize_agent - Improve existing code
3. DOCUMENTATION: generate_agent_docs - Create comprehensive documentation
4. TESTING: generate_unit_tests - Create test suites
5. DEPLOYMENT: dockerize_agent, convert_to_api - Deploy agents
6. ENHANCEMENT: add_monitoring, format_code - Enhance code quality

When given a task:
- Use the appropriate tools to complete the full workflow
- For example: generate → validate → optimize → test → document → deploy
- Chain tools together for comprehensive solutions
- Always validate generated code before returning
- Add tests and documentation when appropriate

Best practices you follow:
- Always include ALL necessary imports
- Add proper error handling and logging
- Create working demo/example code
- Use TypedDict for state definitions
- Add clear documentation and comments
- Make code immediately executable
- Follow LangGraph best practices

You intelligently select and use the right tools to deliver complete, production-ready solutions."""
    
    def generate_agent(self, template_type: str, agent_name: str, purpose: str, 
                      tools_needed: List[str] = None, use_our_core: bool = False) -> Dict[str, Any]:
        """Generate an agent based on requirements
        
        Args:
            template_type: Type of agent (simple, with_tools, multi_agent)
            agent_name: Name for the generated agent
            purpose: What the agent should do
            tools_needed: List of tools if needed
            use_our_core: Whether to use Core Agent infrastructure (default: False for standalone)
        
        Returns:
            Dict with success status, generated code, and metadata
        """
        
        if tools_needed is None:
            tools_needed = []
        
        try:
            # Get the agent generator tool
            generator_tool = None
            for tool in self.config.tools:
                if tool.name == "agent_generator":
                    generator_tool = tool
                    break
            
            if not generator_tool:
                return {
                    "success": False,
                    "error": "Agent generator tool not found",
                    "code": ""
                }
            
            # Generate the agent code
            print(f"🎯 Generating {template_type} agent: {agent_name} (Core Agent: {use_our_core})")
            agent_code = generator_tool._run(template_type, agent_name, purpose, tools_needed, use_our_core)
            
            return {
                "success": True,
                "agent_name": agent_name,
                "template_type": template_type,
                "purpose": purpose,
                "tools": tools_needed,
                "code": agent_code
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": ""
            }
    
    def chat(self, message: str) -> str:
        """Chat with the Coder Agent using Core Agent's invoke method"""
        try:
            # Use Core Agent's invoke method
            result = self.invoke({"messages": [HumanMessage(content=message)]})
            
            # Extract the response from messages
            if result and "messages" in result and len(result["messages"]) > 0:
                # Get the last AI message
                for msg in reversed(result["messages"]):
                    if hasattr(msg, 'content') and not isinstance(msg, HumanMessage):
                        return msg.content
            
            return "No response generated"
            
        except Exception as e:
            return f"Error in chat: {str(e)}"


def demo_coder_agent():
    """
    Comprehensive demonstration of Coder Agent capabilities
    
    Showcases:
    - Simple standalone agent generation
    - Core Agent based implementation with tools
    - Multi-agent system creation
    - Interactive chat functionality
    - Custom agent generation from natural language
    
    All 12 tools are available for the LLM to use intelligently.
    """
    print("🚀 CODER AGENT DEMO - The Ultimate Agent Generator")
    print("=" * 80)
    
    try:
        # Create coder agent
        agent = CoderAgent()
        
        # Example 1: Generate a simple standalone agent
        print("\n📝 Example 1: Generating a simple STANDALONE agent")
        result = agent.generate_agent(
            template_type="simple",
            agent_name="DataAnalyzer",
            purpose="Analyze and summarize data from various sources",
            use_our_core=False  # Standalone LangGraph agent
        )
        
        if result["success"]:
            print(f"✅ Generated: {result['agent_name']} (Standalone)")
            print(f"📄 Code preview: {result['code'][:200]}...")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Example 2: Generate agent with tools using Core Agent
        print("\n📝 Example 2: Generating agent with tools using CORE AGENT")
        result = agent.generate_agent(
            template_type="with_tools",
            agent_name="WebSearcher", 
            purpose="Search and extract information from the web",
            tools_needed=["web_search", "web_scraper", "summarizer"],
            use_our_core=True  # Use Core Agent infrastructure
        )
        
        if result["success"]:
            print(f"✅ Generated: {result['agent_name']} with tools: {result['tools']} (Core Agent)")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Example 3: Multi-agent system
        print("\n📝 Example 3: Generating a MULTI-AGENT system")
        result = agent.generate_agent(
            template_type="multi_agent",
            agent_name="ResearchTeam",
            purpose="Coordinate multiple agents for comprehensive research tasks",
            tools_needed=["researcher", "analyst", "writer"],
            use_our_core=False  # Standalone for flexibility
        )
        
        if result["success"]:
            print(f"✅ Generated: {result['agent_name']} multi-agent system")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Example 4: Test chat functionality
        print("\n💬 Example 4: Chat with Coder Agent")
        response = agent.chat("What are the key differences between standalone and Core Agent based implementations?")
        print(f"Response: {response[:400]}...")
        
        # Example 5: Generate with specific requirements
        print("\n📝 Example 5: Custom agent with specific requirements")
        response = agent.chat(
            "Create a simple agent called 'EmailProcessor' that can read, classify, and respond to emails. "
            "Make it standalone with proper error handling."
        )
        print(f"Generated code preview: {response[:300]}...")
        
        print("\n" + "=" * 80)
        print("✅ Coder Agent demo completed successfully!")
        print("\n💡 Tips:")
        print("- Use 'use_our_core=True' to leverage Core Agent infrastructure")
        print("- Use 'use_our_core=False' (default) for standalone LangGraph agents")
        print("- The agent can generate simple, with_tools, and multi_agent systems")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_coder_agent()