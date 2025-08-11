from core import CoreAgent
from core import AgentConfig

SYSTEM_PROMPT = """You are an assistant that says hello.
"""

class HelloAgent(CoreAgent):
    def __init__(self):
        tools = [
        ]
        cfg = AgentConfig(
            name="HelloAgent",
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            enable_memory=False        )
        super().__init__(cfg)

