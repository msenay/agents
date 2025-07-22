#!/usr/bin/env python3
"""
Coder Agent - Agent Code Generator from Specifications
======================================================

CoderAgent receives agent specifications (like a recipe) and generates complete agent code.

Key Features:
- Accepts detailed agent specifications/requirements
- Generates standalone agents by default
- Optional: Generate agents using Core Agent infrastructure (use_our_core=True)
- Supports: simple agents, agents with tools, multi-agent systems

The agent uses its LLM to intelligently convert specifications into working code.
"""
from typing import Dict, Any
from datetime import datetime

from agent.coder.models import CoderConfig
# Core Agent Infrastructure
from core.core_agent import CoreAgent
from core.config import AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# Import prompts
from agent.coder.prompts import SYSTEM_PROMPT

# Import tools from tools.py
from agent.coder.tools import get_coder_tools


class CoderAgent(CoreAgent):
    """
    Specialized agent for generating agent code from specifications
    
    The CoderAgent receives detailed specifications (like a recipe) and generates
    complete, working agent code. It can create:
    - Simple agents (default)
    - Agents with tools
    - Multi-agent systems
    
    By default, it generates standalone LangGraph agents, but can also generate
    agents using the Core Agent infrastructure when requested.
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize Coder Agent
        
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
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Create tools using the dispatcher from tools.py
        tools = get_coder_tools(model)  # Gets all 3 tools: agent_generator, optimize_agent, format_code
        
        # Create configuration
        config = AgentConfig(
            name="CoderAgent",
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            enable_memory=True,  # Remember successful patterns
            memory_backend="inmemory",
            memory_types=["short_term", "long_term"],
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Initialize parent CoreAgent
        super().__init__(config)
        
        print(f"✅ Coder Agent initialized - Ready to generate agents from specifications!")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return SYSTEM_PROMPT
    
    def generate_from_spec(self, spec: str, agent_type: str = "simple", use_our_core: bool = False) -> Dict[str, Any]:
        """
        Generate agent code from specifications
        
        Args:
            spec: Detailed agent specifications (like a recipe)
            agent_type: Type of agent to generate (simple, with_tools, multi_agent)
            use_our_core: Whether to use Core Agent infrastructure (default: False)
        
        Returns:
            Dict with success status, generated code, and metadata
        """
        
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
            
            # Generate the agent code from specifications
            print(f"🎯 Generating {agent_type} agent from specifications...")
            print(f"📋 Core Agent: {use_our_core}")
            agent_code = generator_tool._run(spec, agent_type, use_our_core)
            
            # Optimize the generated code
            print("🔧 Optimizing generated code...")
            optimize_tool = next((t for t in self.config.tools if t.name == "optimize_agent"), None)
            if optimize_tool:
                agent_code = optimize_tool._run(agent_code)
            
            # Format the code
            print("✨ Formatting code...")
            format_tool = next((t for t in self.config.tools if t.name == "format_code"), None)
            if format_tool:
                agent_code = format_tool._run(agent_code)
            
            print("✅ Agent generation complete!")
            
            return {
                "success": True,
                "agent_type": agent_type,
                "use_our_core": use_our_core,
                "code": agent_code,
                "spec": spec
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "code": ""
            }
    
    def chat(self, message: str) -> str:
        """
        Chat interface for interactive agent generation
        
        The agent will analyze the message and:
        - Extract specifications if provided
        - Determine the appropriate agent type
        - Generate optimized code
        
        Args:
            message: User's request or specifications
            
        Returns:
            Response from the agent
        """
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