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


def create_agent_generator_tool(model):
    """Factory function to create AgentGeneratorTool with model"""
    
    class AgentGeneratorTool(BaseTool):
        """Tool for generating agent code"""
        name: str = "agent_generator"
        description: str = "Generate LangGraph agent code based on specifications"
        args_schema: type[BaseModel] = AgentGeneratorInput
        
        def _run(self, template_type: str, agent_name: str, purpose: str, 
                 tools_needed: List[str] = None) -> str:
            """Generate agent code based on template type"""
            
            if tools_needed is None:
                tools_needed = []
            
            prompt = f"""Generate a complete LangGraph agent with the following specifications:
            
Type: {template_type}
Name: {agent_name}
Purpose: {purpose}
Tools: {tools_needed if tools_needed else 'None'}

Requirements:
1. Use the Core Agent infrastructure from /workspace/core/core_agent.py
2. Include all necessary imports
3. Create a working agent class that inherits from CoreAgent
4. Add proper configuration using AgentConfig
5. Include a demo function
6. Add error handling and logging
7. Make it production-ready

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
        return """You are a Coder Agent specialized in creating LangGraph agents.

Your expertise includes:
- Creating simple agents for basic tasks
- Building agents with custom tools
- Designing multi-agent systems with coordination
- Using Core Agent infrastructure effectively
- Writing clean, maintainable, production-ready code

When creating agents:
1. Always use the Core Agent framework as foundation
2. Include proper error handling and logging
3. Add comprehensive documentation
4. Create demo functions for testing
5. Follow Python best practices

You generate complete, working code that can be executed immediately."""
    
    def generate_agent(self, template_type: str, agent_name: str, purpose: str, 
                      tools_needed: List[str] = None) -> Dict[str, Any]:
        """Generate an agent based on requirements"""
        
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
            print(f"🎯 Generating {template_type} agent: {agent_name}")
            agent_code = generator_tool._run(template_type, agent_name, purpose, tools_needed)
            
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
    print("🚀 CODER AGENT DEMO")
    print("=" * 80)
    
    try:
        # Create coder agent
        agent = CoderAgent()
        
        # Example 1: Generate a simple agent
        print("\n📝 Example 1: Generating a simple agent")
        result = agent.generate_agent(
            template_type="simple",
            agent_name="DataAnalyzer",
            purpose="Analyze and summarize data from various sources"
        )
        
        if result["success"]:
            print(f"✅ Generated: {result['agent_name']}")
            print(f"📄 Code preview: {result['code'][:200]}...")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Example 2: Generate agent with tools
        print("\n📝 Example 2: Generating agent with tools")
        result = agent.generate_agent(
            template_type="with_tools",
            agent_name="WebSearcher", 
            purpose="Search and extract information from the web",
            tools_needed=["web_search", "web_scraper", "summarizer"]
        )
        
        if result["success"]:
            print(f"✅ Generated: {result['agent_name']} with tools: {result['tools']}")
        else:
            print(f"❌ Error: {result['error']}")
        
        # Example 3: Test chat
        print("\n💬 Example 3: Chat with Coder Agent")
        response = agent.chat("What are the best practices for creating a LangGraph agent?")
        print(f"Response: {response[:300]}...")
        
        print("\n" + "=" * 80)
        print("✅ Coder Agent demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_coder_agent()