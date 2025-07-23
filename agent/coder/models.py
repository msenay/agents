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


class CoderConfig:
    """Coder Agent Configuration"""

    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-202-fbeta-dev.openai.azure.com/")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4")
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4o"
    GPT4_DEPLOYMENT_NAME = "gpt4o"

    # Model Parameters
    TEMPERATURE = 0.1  # Low temperature for consistent code generation
    MAX_TOKENS = 4000
