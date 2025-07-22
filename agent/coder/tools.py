from typing import List
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel
from agent.coder.coder import AgentGeneratorInput
from agent.coder.prompts import CORE_AGENT_SIMPLE_PROMPT, CORE_AGENT_WITH_TOOLS_PROMPT, MULTI_AGENT_PROMPT, SIMPLE_AGENT_PROMPT, WITH_TOOLS_AGENT_PROMPT


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

            # Select appropriate prompt based on type and core usage
            if use_our_core:
                if template_type == "simple":
                    prompt = CORE_AGENT_SIMPLE_PROMPT
                elif template_type == "with_tools":
                    prompt = CORE_AGENT_WITH_TOOLS_PROMPT
                else:  # multi_agent
                    prompt = MULTI_AGENT_PROMPT  # Multi-agent always uses supervisor pattern
            else:
                if template_type == "simple":
                    prompt = SIMPLE_AGENT_PROMPT
                elif template_type == "with_tools":
                    prompt = WITH_TOOLS_AGENT_PROMPT
                else:  # multi_agent
                    prompt = MULTI_AGENT_PROMPT

            # Format the prompt with actual values
            formatted_prompt = prompt.format(
                agent_name=agent_name,
                purpose=purpose,
                tools_needed=tools_needed if tools_needed else "None"
            )

            response = model.invoke([HumanMessage(content=formatted_prompt)])
            return response.content

    return AgentGeneratorTool()
