import os
from pydantic import BaseModel, Field

# Tool Input Schemas
class AgentSpecInput(BaseModel):
    """Input schema for agent generation from specifications"""
    agent_spec: str = Field(description="Detailed agent specifications/requirements (like a recipe)")
    agent_type: str = Field(default="simple", description="Type: simple, with_tools, multi_agent")
    use_our_core: bool = Field(default=False, description="Use Core Agent infrastructure (default: False)")


class CodeInput(BaseModel):
    """Input schema for code operations"""
    code: str = Field(description="Python code to process")
