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

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

# Core Agent Infrastructure
from core.core_agent import CoreAgent
from core.config import AgentConfig
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

# Import prompts
from agent.coder.prompts import (
    SIMPLE_AGENT_PROMPT,
    WITH_TOOLS_AGENT_PROMPT,
    MULTI_AGENT_PROMPT,
    CORE_AGENT_SIMPLE_PROMPT,
    CORE_AGENT_WITH_TOOLS_PROMPT,
    SYSTEM_PROMPT
)


# Tool Input Schemas
class AgentSpecInput(BaseModel):
    """Input schema for agent generation from specifications"""
    agent_spec: str = Field(description="Detailed agent specifications/requirements (like a recipe)")
    agent_type: str = Field(default="simple", description="Type: simple, with_tools, multi_agent")
    use_our_core: bool = Field(default=False, description="Use Core Agent infrastructure (default: False)")


class CodeInput(BaseModel):
    """Input schema for code operations"""
    code: str = Field(description="Python code to process")


class CoderConfig:
    """Coder Agent Configuration"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-202-fbeta-dev.openai.azure.com/")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4")
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    
    # Model Parameters
    TEMPERATURE = 0.1  # Low temperature for consistent code generation
    MAX_TOKENS = 4000


def create_agent_generator_tool(model):
    """Create tool for generating agent code from specifications"""
    
    class AgentGeneratorTool(BaseTool):
        name: str = "agent_generator"
        description: str = "Generate agent code from detailed specifications"
        args_schema: type[BaseModel] = AgentSpecInput
        
        def _run(self, agent_spec: str, agent_type: str = "simple", use_our_core: bool = False) -> str:
            """Generate agent code based on specifications"""
            
            # Select appropriate prompt based on type and core usage
            if use_our_core:
                if agent_type == "simple":
                    base_prompt = CORE_AGENT_SIMPLE_PROMPT
                elif agent_type == "with_tools":
                    base_prompt = CORE_AGENT_WITH_TOOLS_PROMPT
                else:  # multi_agent
                    base_prompt = MULTI_AGENT_PROMPT
            else:
                if agent_type == "simple":
                    base_prompt = SIMPLE_AGENT_PROMPT
                elif agent_type == "with_tools":
                    base_prompt = WITH_TOOLS_AGENT_PROMPT
                else:  # multi_agent
                    base_prompt = MULTI_AGENT_PROMPT
            
            # Create the full prompt with specifications
            full_prompt = f"""{base_prompt}

SPECIFICATIONS:
==============
{agent_spec}

Based on these specifications, generate complete, working agent code.
Make sure to:
1. Extract the agent name, purpose, and requirements from the specifications
2. Implement all requested functionality
3. Include proper error handling
4. Add comprehensive documentation
5. Provide usage examples

Generate ONLY the Python code, no explanations."""
            
            response = model.invoke([HumanMessage(content=full_prompt)])
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
            deployment_name=CoderConfig.GPT4_DEPLOYMENT_NAME,
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Create tools
        tools = [
            create_agent_generator_tool(model),  # Main code generation from specs
            create_optimize_agent_tool(model),   # Code optimization
            create_format_code_tool(model)       # Code formatting
        ]
        
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