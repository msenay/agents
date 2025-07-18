"""
Optimized CoreAgent Usage Examples

This demonstrates the core philosophy:
- Minimal by default, powerful when needed
- Each feature is optional and independently configurable
- Perfect for creating specialized agents or orchestrators
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import (
    CoreAgent, AgentConfig,
    create_simple_agent, create_advanced_agent,
    create_supervisor_agent, create_swarm_agent, create_handoff_agent,
    create_memory_agent, create_evaluated_agent, create_human_interactive_agent
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional


# Mock model for examples
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        content = f"Mock response to: {messages[-1].content if messages else 'empty'}"
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "mock"
        
    def bind_tools(self, tools, **kwargs):
        """Mock implementation of bind_tools"""
        return self


# Sample tools for different agent types
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression"""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except:
        return "Invalid expression"


@tool 
def search_web(query: str) -> str:
    """Search the web for information"""
    return f"Web search results for: {query}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email"""
    return f"Email sent to {to} with subject: {subject}"


@tool
def analyze_data(data: str) -> str:
    """Analyze data and provide insights"""
    return f"Analysis of data: {data} - showing trends and patterns"


# Custom response format example
class TaskResponse(BaseModel):
    """Structured response for task completion"""
    task_completed: bool = Field(description="Whether the task was completed")
    result: str = Field(description="The result or explanation")
    next_steps: Optional[str] = Field(description="Suggested next steps")


def demo_simple_specialized_agents():
    """Demonstrate creating simple, specialized agents"""
    print("\n🎯 SIMPLE SPECIALIZED AGENTS")
    print("=" * 50)
    
    model = MockChatModel()
    
    # 1. Calculator Agent - Minimal configuration
    calc_agent = create_simple_agent(
        model=model,
        name="CalculatorAgent",
        tools=[calculate],
        system_prompt="You are a math calculator. Just do calculations.",
        enable_memory=False  # No memory needed for calculations
    )
    
    print(f"✅ Calculator Agent: {calc_agent.config.name}")
    print(f"   Features: Memory={calc_agent.config.enable_memory}, "
          f"Tools={len(calc_agent.config.tools)}")
    
    # 2. Research Assistant - With memory for context
    research_agent = create_simple_agent(
        model=model,
        name="ResearchAgent",
        tools=[search_web],
        system_prompt="You are a research assistant. Help users find information.",
        enable_memory=True,  # Need memory for research context
        memory_type="memory"
    )
    
    print(f"✅ Research Agent: {research_agent.config.name}")
    print(f"   Features: Memory={research_agent.config.enable_memory}, "
          f"Tools={len(research_agent.config.tools)}")
    
    # 3. Email Assistant - No memory (stateless email tasks)
    email_agent = create_simple_agent(
        model=model,
        name="EmailAgent", 
        tools=[send_email],
        system_prompt="You help compose and send emails.",
        enable_memory=False  # Stateless email operations
    )
    
    print(f"✅ Email Agent: {email_agent.config.name}")
    print(f"   Features: Memory={email_agent.config.enable_memory}, "
          f"Tools={len(email_agent.config.tools)}")


def demo_advanced_feature_agents():
    """Demonstrate agents with specific advanced features"""
    print("\n🚀 ADVANCED FEATURE AGENTS")
    print("=" * 50)
    
    model = MockChatModel()
    
    # 1. Memory-Intensive Agent (Customer Support)
    support_agent = create_memory_agent(
        model=model,
        name="CustomerSupportAgent",
        tools=[search_web, send_email],
        memory_type="langmem_combined",  # Advanced memory for customer history
        system_prompt="You are a customer support agent. Remember customer interactions."
    )
    
    print(f"✅ Customer Support Agent: {support_agent.config.name}")
    print(f"   Memory: {support_agent.config.memory_type}")
    print(f"   Advanced Memory: {support_agent.has_langmem_support()}")
    
    # 2. Quality-Controlled Agent (Content Creator)
    content_agent = create_evaluated_agent(
        model=model,
        name="ContentCreatorAgent",
        tools=[search_web],
        evaluation_metrics=["creativity", "accuracy", "engagement"],
        system_prompt="You create high-quality content with performance monitoring."
    )
    
    print(f"✅ Content Creator Agent: {content_agent.config.name}")
    print(f"   Evaluation: {content_agent.config.enable_evaluation}")
    print(f"   Metrics: {content_agent.config.evaluation_metrics}")
    
    # 3. Human-Collaborative Agent (Data Analyst)
    analyst_agent = create_human_interactive_agent(
        model=model,
        name="DataAnalystAgent",
        tools=[analyze_data],
        system_prompt="You analyze data with human oversight and approval.",
        interrupt_before=["tools"],  # Get approval before using tools
        interrupt_after=["human"]  # Review results before sending
    )
    
    print(f"✅ Data Analyst Agent: {analyst_agent.config.name}")
    print(f"   Human Interaction: {analyst_agent.config.enable_human_feedback}")
    print(f"   Interrupts: {analyst_agent.config.interrupt_before}")


def demo_orchestrator_agents():
    """Demonstrate multi-agent orchestrators"""
    print("\n🎼 ORCHESTRATOR AGENTS")
    print("=" * 50)
    
    model = MockChatModel()
    
    # Create specialized workers first
    calc_agent = create_simple_agent(model, "CalcWorker", [calculate], enable_memory=False)
    research_agent = create_simple_agent(model, "ResearchWorker", [search_web], enable_memory=True)
    email_agent = create_simple_agent(model, "EmailWorker", [send_email], enable_memory=False)
    
    workers = {
        "calculator": calc_agent,
        "researcher": research_agent, 
        "emailer": email_agent
    }
    
    # 1. Supervisor - Central coordinator
    supervisor = create_supervisor_agent(
        model=model,
        name="TaskSupervisor",
        agents=workers,
        system_prompt="You coordinate tasks by delegating to specialized workers.",
        enable_memory=True,  # Remember delegation decisions
        enable_evaluation=True  # Monitor worker performance
    )
    
    print(f"✅ Supervisor Agent: {supervisor.config.name}")
    print(f"   Workers: {list(workers.keys())}")
    print(f"   Supervision: {supervisor.config.enable_supervisor}")
    
    # 2. Swarm - Dynamic coordination
    swarm = create_swarm_agent(
        model=model,
        name="TaskSwarm",
        agents=workers,
        default_active_agent="researcher",  # Start with research
        system_prompt="You dynamically switch between workers based on task needs.",
        enable_memory=True
    )
    
    print(f"✅ Swarm Agent: {swarm.config.name}")
    print(f"   Workers: {list(workers.keys())}")
    print(f"   Default: {swarm.config.default_active_agent}")
    
    # 3. Handoff - Manual transfers
    handoff = create_handoff_agent(
        model=model,
        name="TaskHandoff",
        agents=workers,
        system_prompt="You handle tasks and transfer to specialists when needed.",
        enable_memory=True
    )
    
    print(f"✅ Handoff Agent: {handoff.config.name}")
    print(f"   Workers: {list(workers.keys())}")
    print(f"   Handoff: {handoff.config.enable_handoff}")


def demo_custom_configurations():
    """Demonstrate completely custom agent configurations"""
    print("\n⚙️ CUSTOM CONFIGURATIONS")
    print("=" * 50)
    
    model = MockChatModel()
    
    # 1. Ultra-minimal agent (just LLM, no extras)
    minimal_config = AgentConfig(
        name="MinimalAgent",
        model=model,
        system_prompt="You are a simple chatbot."
        # Everything else defaults to False/disabled
    )
    minimal_agent = CoreAgent(minimal_config)
    
    print(f"✅ Minimal Agent: {minimal_agent.config.name}")
    print(f"   Memory: {minimal_agent.config.enable_memory}")
    print(f"   Evaluation: {minimal_agent.config.enable_evaluation}")
    print(f"   Human Feedback: {minimal_agent.config.enable_human_feedback}")
    
    # 2. Full-featured agent (everything enabled)
    maximal_config = AgentConfig(
        name="MaximalAgent",
        model=model,
        system_prompt="You are a comprehensive AI assistant with all features.",
        tools=[calculate, search_web, send_email, analyze_data],
        
        # Enable all features
        enable_memory=True,
        memory_type="langmem_combined",
        enable_evaluation=True,
        evaluation_metrics=["accuracy", "helpfulness", "efficiency"],
        enable_human_feedback=True,
        interrupt_before=["analyze_data"],
        enable_mcp=False,  # Would need MCP servers
        
        # Custom response structure
        response_format=TaskResponse,
        enable_streaming=True
    )
    maximal_agent = CoreAgent(maximal_config)
    
    print(f"✅ Maximal Agent: {maximal_agent.config.name}")
    print(f"   Memory: {maximal_agent.config.memory_type}")
    print(f"   Tools: {len(maximal_agent.config.tools)}")
    print(f"   Response Format: {maximal_agent.config.response_format.__name__}")
    
    # 3. Domain-specific agent (E-commerce Assistant)
    ecommerce_config = AgentConfig(
        name="EcommerceAssistant",
        model=model,
        system_prompt="You help customers with online shopping and order management.",
        tools=[search_web, send_email],  # Product search, order emails
        
        # Selective features for e-commerce
        enable_memory=True,
        memory_type="langmem_long",  # Remember customer preferences
        enable_evaluation=True,
        evaluation_metrics=["customer_satisfaction", "task_completion"],
        enable_human_feedback=False,  # Autonomous customer service
        enable_streaming=True  # Real-time chat experience
    )
    ecommerce_agent = CoreAgent(ecommerce_config)
    
    print(f"✅ E-commerce Agent: {ecommerce_agent.config.name}")
    print(f"   Specialization: Customer service + product assistance")
    print(f"   Memory: Long-term customer preferences")
    print(f"   Evaluation: Customer-focused metrics")


def demo_practical_use_cases():
    """Show practical real-world use cases"""
    print("\n💼 PRACTICAL USE CASES")
    print("=" * 50)
    
    use_cases = {
        "Simple Task Automation": {
            "agent": "create_simple_agent(model, tools=[task_tool], enable_memory=False)",
            "use_case": "One-time tasks, calculations, data transformation"
        },
        "Customer Support": {
            "agent": "create_memory_agent(model, memory_type='langmem_combined')",
            "use_case": "Remember customer history, provide personalized support"
        },
        "Content Creation": {
            "agent": "create_evaluated_agent(model, evaluation_metrics=['quality'])",
            "use_case": "Monitor output quality, ensure brand standards"
        },
        "Data Analysis": {
            "agent": "create_human_interactive_agent(model, interrupt_before=['analyze'])",
            "use_case": "Get human approval for sensitive analysis"
        },
        "Multi-step Workflows": {
            "agent": "create_supervisor_agent(model, agents={...})",
            "use_case": "Complex processes requiring different specialists"
        },
        "Dynamic Routing": {
            "agent": "create_swarm_agent(model, agents={...})",
            "use_case": "Route tasks based on content and complexity"
        }
    }
    
    for use_case, details in use_cases.items():
        print(f"\n🔹 {use_case}:")
        print(f"   Agent: {details['agent']}")
        print(f"   Use case: {details['use_case']}")


def main():
    """Run all demonstrations"""
    print("🎯 CoreAgent Optimized Usage Examples")
    print("=" * 60)
    print("Philosophy: Minimal by default, powerful when needed")
    print("Each feature is optional and independently configurable")
    
    demo_simple_specialized_agents()
    demo_advanced_feature_agents()
    demo_orchestrator_agents()
    demo_custom_configurations()
    demo_practical_use_cases()
    
    print("\n" + "=" * 60)
    print("🎉 CoreAgent examples completed!")
    print("\n💡 Key Takeaways:")
    print("• Start simple - only enable features you need")
    print("• Memory is optional - disable for stateless tasks")
    print("• Human interaction only when required")
    print("• Evaluation for quality-critical applications")
    print("• Orchestrators for complex multi-agent workflows")
    print("• Complete flexibility with custom AgentConfig")


if __name__ == "__main__":
    main()