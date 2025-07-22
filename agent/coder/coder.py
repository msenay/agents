#!/usr/bin/env python3
"""
Coder Agent - Specialized Agent for Code Generation
==================================================

A focused agent that generates high-quality LangGraph agents using only essential tools:
- agent_generator: Creates agent code (simple, with_tools, multi_agent)
- optimize_agent: Optimizes generated code
- format_code: Ensures clean formatting

Supports both standalone LangGraph and Core Agent based implementations.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

from agent.coder.prompts import CORE_AGENT_SIMPLE_PROMPT, CORE_AGENT_WITH_TOOLS_PROMPT, MULTI_AGENT_PROMPT, SIMPLE_AGENT_PROMPT, WITH_TOOLS_AGENT_PROMPT, SYSTEM_PROMPT

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

# Core Agent Infrastructure
from core.core_agent import CoreAgent
from core.config import AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage

# We need to create our own simplified tools since we're removing the complex tools.py
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List


# Tool Input Schemas
class AgentGeneratorInput(BaseModel):
    """Input schema for agent generation"""
    template_type: str = Field(description="Type: simple, with_tools, multi_agent")
    agent_name: str = Field(description="Name for the agent")
    purpose: str = Field(description="What the agent should do")
    tools_needed: List[str] = Field(default=[], description="List of tools if needed")
    use_our_core: bool = Field(default=False, description="Whether to use Core Agent infrastructure")


class CodeInput(BaseModel):
    """Input schema for code operations"""
    code: str = Field(description="Python code to process")


def create_agent_generator_tool(model):
    """Create tool for generating agent code"""
    
    class AgentGeneratorTool(BaseTool):
        name: str = "agent_generator"
        description: str = "Generate LangGraph agent code based on specifications"
        args_schema: type[BaseModel] = AgentGeneratorInput
        
        def _run(self, template_type: str, agent_name: str, purpose: str, 
                 tools_needed: List[str] = None, use_our_core: bool = False) -> str:
            if tools_needed is None:
                tools_needed = []
            
            # Select appropriate prompt
            if use_our_core:
                if template_type == "simple":
                    prompt = CORE_AGENT_SIMPLE_PROMPT
                elif template_type == "with_tools":
                    prompt = CORE_AGENT_WITH_TOOLS_PROMPT
                else:
                    prompt = MULTI_AGENT_PROMPT
            else:
                if template_type == "simple":
                    prompt = SIMPLE_AGENT_PROMPT
                elif template_type == "with_tools":
                    prompt = WITH_TOOLS_AGENT_PROMPT
                else:
                    prompt = MULTI_AGENT_PROMPT
            
            # Format prompt
            formatted_prompt = prompt.format(
                agent_name=agent_name,
                purpose=purpose,
                tools_needed=tools_needed if tools_needed else "None"
            )
            
            response = model.invoke([HumanMessage(content=formatted_prompt)])
            return response.content
    
    return AgentGeneratorTool()


def create_optimize_agent_tool(model):
    """Create tool for optimizing agent code"""
    
    class OptimizeAgentTool(BaseTool):
        name: str = "optimize_agent"
        description: str = "Optimize agent code for performance and best practices"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = f"""Optimize this agent code:

```python
{code}
```

Optimization areas:
1. Performance improvements
2. Memory efficiency
3. Better error handling
4. Code simplification
5. Design pattern improvements
6. Add missing docstrings
7. Improve type hints

Return the optimized code. Keep all functionality intact."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return OptimizeAgentTool()


def create_format_code_tool(model):
    """Create tool for formatting code"""
    
    class FormatCodeTool(BaseTool):
        name: str = "format_code"
        description: str = "Format code with Black/isort standards"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = f"""Format and clean up this code:

```python
{code}
```

Apply:
1. Black formatting standards
2. isort import ordering
3. Remove unused imports
4. Fix line lengths
5. Consistent naming conventions
6. Proper spacing

Return the formatted code."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return FormatCodeTool()


class CoderConfig:
    """Coder Agent Configuration"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = "https://oai-202-fbeta-dev.openai.azure.com/"
    OPENAI_API_KEY = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    
    # Model Parameters
    TEMPERATURE = 0.1  # Low temperature for consistent code generation
    MAX_TOKENS = 4000


class CoderAgent(CoreAgent):
    """
    Specialized agent for generating high-quality LangGraph agents
    
    Focuses on code generation with three essential tools:
    - agent_generator: Creates agents (simple, with_tools, multi_agent)
    - optimize_agent: Improves code quality and performance
    - format_code: Ensures clean, readable formatting
    """
    
    def __init__(self, session_id: str = None):
        """
        Initialize Coder Agent with essential code generation tools
        
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
        
        # Create only essential tools for code generation
        tools = [
            create_agent_generator_tool(model),  # Main code generation
            create_optimize_agent_tool(model),   # Code optimization
            create_format_code_tool(model)       # Code formatting
        ]
        
        # Create configuration
        config = AgentConfig(
            name="CoderAgent",
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            enable_memory=True,  # Remember successful patterns
            memory_backend="inmemory",
            memory_types=["short_term", "long_term"],
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Initialize parent CoreAgent
        super().__init__(config)
        
        print(f"✅ Coder Agent initialized with {len(tools)} essential tools")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")

    
    def generate_agent(self, template_type: str, agent_name: str, purpose: str, 
                      tools_needed: List[str] = None, use_our_core: bool = False) -> Dict[str, Any]:
        """
        Generate an agent based on requirements
        
        Args:
            template_type: Type of agent (simple, with_tools, multi_agent)
            agent_name: Name for the generated agent
            purpose: What the agent should do
            tools_needed: List of tools if needed
            use_our_core: Whether to use Core Agent infrastructure (default: False)
        
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
                "agent_name": agent_name,
                "template_type": template_type,
                "purpose": purpose,
                "tools": tools_needed,
                "use_our_core": use_our_core,
                "code": agent_code
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
        
        The agent will use its tools intelligently based on the request:
        - Generates code if asked to create an agent
        - Optimizes code if provided with existing code
        - Formats code when needed
        
        Args:
            message: User's request
            
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