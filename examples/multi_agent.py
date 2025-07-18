"""
Multi-Agent Examples - Supervisor, Swarm, and Handoff Patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import (
    CoreAgent, AgentConfig, 
    create_supervisor_agent, create_swarm_agent, create_handoff_agent
)
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage


class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        message = AIMessage(content="Mock response")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "mock"


@tool
def book_flight(from_airport: str, to_airport: str) -> str:
    """Book a flight"""
    return f"Flight booked from {from_airport} to {to_airport}"


@tool
def book_hotel(hotel_name: str) -> str:
    """Book a hotel"""
    return f"Hotel {hotel_name} booked"


def demo_supervisor():
    """Demonstrate supervisor pattern"""
    print("\n🏢 Supervisor Pattern Demo")
    print("-" * 30)
    
    model = MockChatModel()
    
    # Create specialized agents
    flight_agent = CoreAgent(AgentConfig(
        name="flight_assistant",
        model=model,
        tools=[book_flight],
        system_prompt="Flight booking specialist"
    ))
    
    hotel_agent = CoreAgent(AgentConfig(
        name="hotel_assistant", 
        model=model,
        tools=[book_hotel],
        system_prompt="Hotel booking specialist"
    ))
    
    # Create supervisor
    agents = {"flight": flight_agent, "hotel": hotel_agent}
    supervisor = create_supervisor_agent(model, agents)
    
    print(f"✅ Supervisor created with {len(agents)} agents")
    print(f"Status: {supervisor.get_status()}")


def demo_swarm():
    """Demonstrate swarm pattern"""
    print("\n🐝 Swarm Pattern Demo")
    print("-" * 30)
    
    model = MockChatModel()
    
    agents = {
        "flight": CoreAgent(AgentConfig(name="flight", model=model, tools=[book_flight])),
        "hotel": CoreAgent(AgentConfig(name="hotel", model=model, tools=[book_hotel]))
    }
    
    swarm = create_swarm_agent(model, agents, "flight")
    
    print(f"✅ Swarm created with {len(agents)} agents")
    print(f"Default agent: flight")


def demo_handoff():
    """Demonstrate handoff pattern"""
    print("\n🤝 Handoff Pattern Demo")
    print("-" * 30)
    
    model = MockChatModel()
    
    agents = {
        "flight": CoreAgent(AgentConfig(name="flight", model=model, tools=[book_flight])),
        "hotel": CoreAgent(AgentConfig(name="hotel", model=model, tools=[book_hotel]))
    }
    
    handoff = create_handoff_agent(model, agents, "flight")
    
    print(f"✅ Handoff system created with {len(agents)} agents")
    print(f"Available transfers: {list(agents.keys())}")


if __name__ == "__main__":
    print("🤖 Multi-Agent Patterns Demo")
    print("=" * 40)
    
    demo_supervisor()
    demo_swarm()
    demo_handoff()
    
    print("\n✅ All demos completed!")