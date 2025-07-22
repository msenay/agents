from typing import List
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from agent.coder.prompts import CORE_AGENT_SIMPLE_PROMPT, CORE_AGENT_WITH_TOOLS_PROMPT, MULTI_AGENT_PROMPT, SIMPLE_AGENT_PROMPT, WITH_TOOLS_AGENT_PROMPT


# Tool Input Schemas
class AgentSpecInput(BaseModel):
    """Input schema for agent generation from specifications"""
    agent_spec: str = Field(description="Detailed agent specifications (like a recipe)")
    agent_type: str = Field(default="simple", description="Type: simple, with_tools, multi_agent")
    use_our_core: bool = Field(default=False, description="Whether to use Core Agent infrastructure")


class CodeInput(BaseModel):
    """Input schema for code operations"""
    code: str = Field(description="Python code to process")


def create_agent_generator_tool(model):
    """Factory function to create AgentGeneratorTool with model"""

    class AgentGeneratorTool(BaseTool):
        """Tool for generating agent code from specifications"""
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
    """Factory function to create OptimizeAgentTool with model"""
    
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
    """Factory function to create FormatCodeTool with model"""
    
    class FormatCodeTool(BaseTool):
        name: str = "format_code"
        description: str = "Format Python code according to PEP8 standards"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = f"""Format this Python code according to PEP8 standards:

```python
{code}
```

Apply:
1. Proper indentation (4 spaces)
2. Line length limits (79 chars)
3. Proper spacing around operators
4. Consistent quotes
5. Import organization
6. Docstring formatting

Return ONLY the formatted code."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return FormatCodeTool()


# ============================================================================
# TOOL DISPATCHER / HELPER FUNCTIONS
# ============================================================================

def get_coder_tools(model, tool_names: List[str] = None):
    """
    Get specific tools for CoderAgent or all tools if none specified
    
    Args:
        model: The LLM model to pass to tools
        tool_names: List of tool names to create. If None, returns all tools.
    
    Returns:
        List of tool instances
    
    Example:
        # Get all tools
        tools = get_coder_tools(model)
        
        # Get specific tools
        tools = get_coder_tools(model, ["agent_generator", "optimize_agent"])
    """
    # Available tool factories
    available_tools = {
        "agent_generator": create_agent_generator_tool,
        "optimize_agent": create_optimize_agent_tool,
        "format_code": create_format_code_tool
    }
    
    # If no specific tools requested, return all
    if tool_names is None:
        return [factory(model) for factory in available_tools.values()]
    
    # Return only requested tools
    tools = []
    for name in tool_names:
        if name in available_tools:
            tools.append(available_tools[name](model))
        else:
            print(f"⚠️ Warning: Tool '{name}' not found. Available tools: {list(available_tools.keys())}")
    
    return tools


def get_all_tool_names():
    """Get list of all available tool names"""
    return ["agent_generator", "optimize_agent", "format_code"]


def create_tool_by_name(model, tool_name: str):
    """
    Create a single tool by name
    
    Args:
        model: The LLM model
        tool_name: Name of the tool to create
        
    Returns:
        Tool instance or None if not found
    """
    tools = get_coder_tools(model, [tool_name])
    return tools[0] if tools else None
