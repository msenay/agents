from core.core_agent import CoreAgent
from core.config import AgentConfig

SYSTEM_PROMPT = """You are an assistant that says test.
"""

class TestAgent(CoreAgent):
    def __init__(self):
        tools = [
        ]
        cfg = AgentConfig(
            name="TestAgent",
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            enable_memory=False        )
        super().__init__(cfg)

