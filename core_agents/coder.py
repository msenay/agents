#!/usr/bin/env python3
"""
Coder Agent - Specialized agent for generating other agents
==========================================================

An AI agent that specializes in creating other agents using the Core Agent infrastructure.
Supports simple agents, agents with tools, and multi-agent systems.
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


class AgentGeneratorInput(BaseModel):
    """Input schema for agent generation"""
    template_type: str = Field(description="Type: simple, with_tools, multi_agent")
    agent_name: str = Field(description="Name for the agent")
    purpose: str = Field(description="What the agent should do")
    tools_needed: List[str] = Field(default=[], description="List of tools if needed")
    use_our_core: bool = Field(default=False, description="Whether to use Core Agent infrastructure")


def create_agent_generator_tool(model):
    """Factory function to create AgentGeneratorTool with model"""
    
    class AgentGeneratorTool(BaseTool):
        """Tool for generating agent code"""
        name: str = "agent_generator"
        description: str = "Generate LangGraph agent code based on specifications"
        args_schema: type[BaseModel] = AgentGeneratorInput
        
        def _run(self, template_type: str, agent_name: str, purpose: str, 
                 tools_needed: List[str] = None, use_our_core: bool = False) -> str:
            """Generate agent code based on template type"""
            
            if tools_needed is None:
                tools_needed = []
            
            # Different prompts based on whether to use Core Agent
            if use_our_core:
                prompt = f"""Generate a complete LangGraph agent that uses our Core Agent infrastructure:
            
Type: {template_type}
Name: {agent_name}
Purpose: {purpose}
Tools: {tools_needed if tools_needed else 'None'}

Requirements for Core Agent based implementation:
1. Import and inherit from CoreAgent at /workspace/core/core_agent.py
2. Use AgentConfig from /workspace/core/config.py
3. Leverage Core Agent's built-in features (memory, tools, etc.)
4. Create a proper __init__ method that calls super().__init__(config)
5. Include all necessary imports
6. Add a demo function
7. Use proper error handling and logging

Generate the complete Python code:"""
            else:
                prompt = f"""Generate a complete standalone LangGraph agent:
            
Type: {template_type}
Name: {agent_name}  
Purpose: {purpose}
Tools: {tools_needed if tools_needed else 'None'}

Requirements for standalone implementation:
1. Use standard LangGraph patterns (StateGraph, add_node, add_edge)
2. Define proper agent state with TypedDict
3. Create clean node functions for each step
4. Add proper tool integration if needed
5. Include all necessary imports (langchain, langgraph, etc.)
6. Create a working example/demo at the end
7. Make it self-contained and ready to run

For {template_type} type:
{"- Simple: Basic agent with state management and clear workflow" if template_type == "simple" else ""}
{"- With Tools: Include tool node, proper tool calling and result handling" if template_type == "with_tools" else ""}
{"- Multi-agent: Create supervisor pattern with multiple sub-agents coordinating" if template_type == "multi_agent" else ""}

Generate the complete Python code:"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return AgentGeneratorTool()


class CoderAgent(CoreAgent):
    """Coder Agent - Specializes in creating other agents"""
    
    def __init__(self, session_id: str = None):
        """Initialize Coder Agent with Core Agent infrastructure"""
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
        
        # Create tools
        tools = [create_agent_generator_tool(model)]
        
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
        
        print(f"✅ Coder Agent initialized")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")
    
    def _get_system_prompt(self) -> str:
        """System prompt for the Coder Agent"""
        return """You are an expert Coder Agent specialized in creating LangGraph agents.

Your expertise includes:
- Creating standalone LangGraph agents with proper state management
- Building agents with custom tools and tool integration
- Designing multi-agent systems with supervisor patterns
- Using Core Agent infrastructure when requested
- Writing clean, maintainable, production-ready code

Key capabilities:
1. STANDALONE AGENTS (default): Create pure LangGraph agents with StateGraph, nodes, edges
2. CORE AGENT BASED: Use our Core Agent framework when use_our_core=True
3. SIMPLE AGENTS: Basic workflow with state management
4. WITH TOOLS: Proper tool integration, tool node, and result handling
5. MULTI-AGENT: Supervisor pattern with agent coordination

Best practices you follow:
- Always include ALL necessary imports
- Add proper error handling and logging
- Create working demo/example code
- Use TypedDict for state definitions
- Add clear documentation and comments
- Make code immediately executable
- Follow LangGraph best practices

You generate complete, working, production-ready code."""
    
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
    """Demonstrate the Coder Agent functionality"""
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