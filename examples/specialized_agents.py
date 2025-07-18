"""
Specialized Agent Examples

This file demonstrates how to create specialized agents by extending the CoreAgent
with different prompts, tools, and configurations for specific use cases.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import CoreAgent, AgentConfig
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from typing import List, Dict, Any


# Mock model for demonstration
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        return "Mock response"
    
    def _llm_type(self):
        return "mock"


# Response formats for different agent types
class CodeAnalysisResult(BaseModel):
    """Response format for code analysis agent"""
    issues: List[str]
    suggestions: List[str]
    complexity_score: float
    summary: str


class ResearchResult(BaseModel):
    """Response format for research agent"""
    findings: List[str]
    sources: List[str]
    confidence: float
    summary: str


class CustomerServiceResult(BaseModel):
    """Response format for customer service agent"""
    response: str
    sentiment: str
    category: str
    escalation_needed: bool


# Specialized Tools

@tool
def analyze_code(code: str) -> str:
    """Analyze code for issues and improvements."""
    return f"Code analysis complete. Found 3 issues in {len(code)} characters of code."


@tool
def search_documentation(query: str) -> str:
    """Search technical documentation."""
    return f"Documentation search results for: {query}"


@tool
def run_tests(test_suite: str) -> str:
    """Run automated tests."""
    return f"Test suite '{test_suite}' executed. 8/10 tests passed."


@tool
def search_academic_papers(query: str) -> str:
    """Search academic papers and journals."""
    return f"Found 15 academic papers related to: {query}"


@tool
def fact_check(claim: str) -> str:
    """Verify facts and claims."""
    return f"Fact-checking: {claim}. Status: Verified"


@tool
def web_search(query: str) -> str:
    """Perform web search."""
    return f"Web search results for: {query}"


@tool
def access_customer_database(customer_id: str) -> str:
    """Access customer information."""
    return f"Customer {customer_id}: Premium member since 2020"


@tool
def create_ticket(issue: str) -> str:
    """Create support ticket."""
    return f"Ticket #12345 created for: {issue}"


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate to human agent."""
    return f"Escalated to human agent. Reason: {reason}"


# Specialized Agent Classes

class CodeReviewAgent(CoreAgent):
    """Specialized agent for code review and analysis"""
    
    def __init__(self, model: BaseChatModel):
        config = AgentConfig(
            name="CodeReviewAgent",
            description="Specialized agent for code review, analysis, and technical assistance",
            model=model,
            system_prompt="""You are an expert software engineer specializing in code review and analysis.
            Your role is to:
            - Review code for bugs, performance issues, and best practices
            - Suggest improvements and optimizations
            - Provide technical guidance and documentation
            - Help with debugging and troubleshooting
            
            Always provide constructive feedback and explain your reasoning.""",
            tools=[analyze_code, search_documentation, run_tests],
            response_format=CodeAnalysisResult,
            enable_memory=True,
            memory_type="memory",
            enable_streaming=True,
            evaluation_metrics=["technical_accuracy", "helpfulness", "clarity"]
        )
        super().__init__(config)
        
    def review_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """Specialized method for code review"""
        prompt = f"""Please review this {language} code:

```{language}
{code}
```

Analyze for:
1. Bugs and potential issues
2. Performance optimizations
3. Code style and best practices
4. Security vulnerabilities
5. Maintainability improvements
"""
        return self.invoke(prompt)


class ResearchAgent(CoreAgent):
    """Specialized agent for research and information gathering"""
    
    def __init__(self, model: BaseChatModel):
        config = AgentConfig(
            name="ResearchAgent",
            description="Specialized agent for academic research and fact-finding",
            model=model,
            system_prompt="""You are a research specialist with expertise in academic research, 
            fact-checking, and information synthesis. Your role is to:
            - Conduct thorough research on various topics
            - Verify facts and sources
            - Synthesize information from multiple sources
            - Provide well-documented findings with proper citations
            
            Always prioritize accuracy and provide credible sources.""",
            tools=[search_academic_papers, fact_check, web_search],
            response_format=ResearchResult,
            enable_memory=True,
            memory_type="memory",
            enable_evaluation=True,
            evaluation_metrics=["accuracy", "completeness", "source_quality"]
        )
        super().__init__(config)
        
    def research_topic(self, topic: str, depth: str = "comprehensive") -> Dict[str, Any]:
        """Specialized method for topic research"""
        prompt = f"""Please conduct {depth} research on: {topic}

Requirements:
1. Find credible academic sources
2. Verify key facts and claims
3. Provide a balanced perspective
4. Include recent developments
5. Summarize findings with proper citations
"""
        return self.invoke(prompt)


class CustomerServiceAgent(CoreAgent):
    """Specialized agent for customer service and support"""
    
    def __init__(self, model: BaseChatModel):
        config = AgentConfig(
            name="CustomerServiceAgent",
            description="Specialized agent for customer service and support",
            model=model,
            system_prompt="""You are a customer service specialist focused on providing 
            excellent customer support. Your role is to:
            - Help customers with their questions and issues
            - Provide accurate product and service information
            - Resolve problems efficiently and professionally
            - Escalate complex issues when necessary
            - Maintain a positive and helpful attitude
            
            Always be polite, patient, and solution-focused.""",
            tools=[access_customer_database, create_ticket, escalate_to_human],
            response_format=CustomerServiceResult,
            enable_memory=True,
            memory_type="memory",
            enable_human_feedback=True,
            interrupt_before=["execute_tools"],  # Fixed: use node names that actually exist
            evaluation_metrics=["customer_satisfaction", "resolution_rate", "professionalism"]
        )
        super().__init__(config)
        
    def handle_customer_inquiry(self, customer_id: str, inquiry: str) -> Dict[str, Any]:
        """Specialized method for handling customer inquiries"""
        prompt = f"""Customer ID: {customer_id}
Customer Inquiry: {inquiry}

Please:
1. Look up customer information
2. Address their inquiry professionally
3. Provide helpful solutions
4. Determine if escalation is needed
5. Create a ticket if necessary
"""
        return self.invoke(prompt)


class DataAnalysisAgent(CoreAgent):
    """Specialized agent for data analysis and insights"""
    
    def __init__(self, model: BaseChatModel):
        # Data analysis tools
        @tool
        def analyze_dataset(data_description: str) -> str:
            """Analyze a dataset and provide insights."""
            return f"Analysis complete for: {data_description}. Key insights: trend analysis, outliers detected."
        
        @tool
        def generate_visualization(data_type: str) -> str:
            """Generate data visualizations."""
            return f"Created {data_type} visualization with key metrics highlighted."
        
        @tool
        def statistical_analysis(analysis_type: str) -> str:
            """Perform statistical analysis."""
            return f"Statistical analysis ({analysis_type}) complete. P-value: 0.01, significance confirmed."
        
        config = AgentConfig(
            name="DataAnalysisAgent",
            description="Specialized agent for data analysis and business intelligence",
            model=model,
            system_prompt="""You are a data scientist specializing in data analysis and insights.
            Your role is to:
            - Analyze datasets and identify patterns
            - Generate meaningful visualizations
            - Perform statistical analysis
            - Provide actionable business insights
            - Ensure data quality and accuracy
            
            Always explain your methodology and reasoning clearly.""",
            tools=[analyze_dataset, generate_visualization, statistical_analysis],
            enable_memory=True,
            memory_type="memory",
            enable_streaming=True,
            evaluation_metrics=["analytical_accuracy", "insight_quality", "clarity"]
        )
        super().__init__(config)


class CreativeWritingAgent(CoreAgent):
    """Specialized agent for creative writing and content creation"""
    
    def __init__(self, model: BaseChatModel):
        # Creative writing tools
        @tool
        def check_grammar(text: str) -> str:
            """Check grammar and style."""
            return f"Grammar check complete. Found 2 suggestions for improvement in {len(text)} characters."
        
        @tool
        def generate_ideas(topic: str) -> str:
            """Generate creative ideas for a topic."""
            return f"Generated 5 creative ideas for: {topic}"
        
        @tool
        def research_style(style: str) -> str:
            """Research writing styles and techniques."""
            return f"Style guide for {style}: key characteristics and examples found."
        
        config = AgentConfig(
            name="CreativeWritingAgent",
            description="Specialized agent for creative writing and content creation",
            model=model,
            system_prompt="""You are a creative writing specialist with expertise in various 
            writing styles and formats. Your role is to:
            - Create engaging and original content
            - Adapt writing style to different audiences
            - Provide creative ideas and inspiration
            - Ensure proper grammar and style
            - Help with editing and refinement
            
            Always maintain creativity while ensuring quality and clarity.""",
            tools=[check_grammar, generate_ideas, research_style],
            enable_memory=True,
            memory_type="memory",
            enable_streaming=True,
            evaluation_metrics=["creativity", "engagement", "quality"]
        )
        super().__init__(config)


# Multi-Agent Supervisor Example
class MultiAgentSupervisor(CoreAgent):
    """Supervisor agent that coordinates multiple specialized agents"""
    
    def __init__(self, model: BaseChatModel, specialized_agents: Dict[str, CoreAgent]):
        config = AgentConfig(
            name="MultiAgentSupervisor",
            description="Supervisor agent coordinating multiple specialized agents",
            model=model,
            system_prompt="""You are a supervisor coordinating multiple specialized agents.
            Your role is to:
            - Route tasks to the most appropriate agent
            - Coordinate complex multi-step tasks
            - Ensure quality and consistency across agents
            - Handle escalations and complex decisions
            - Provide overall project management
            
            Available agents: Code Review, Research, Customer Service, Data Analysis, Creative Writing""",
            enable_supervisor=True,
            enable_memory=True,
            memory_type="memory",
            enable_evaluation=True,
            evaluation_metrics=["coordination_quality", "task_completion", "efficiency"]
        )
        super().__init__(config)
        
        # Add specialized agents to supervision
        for name, agent in specialized_agents.items():
            self.add_supervised_agent(name, agent)
    
    def route_task(self, task: str, task_type: str) -> Dict[str, Any]:
        """Route a task to the appropriate specialized agent"""
        routing_map = {
            "code": "CodeReviewAgent",
            "research": "ResearchAgent", 
            "support": "CustomerServiceAgent",
            "data": "DataAnalysisAgent",
            "writing": "CreativeWritingAgent"
        }
        
        target_agent = routing_map.get(task_type, "default")
        return self.coordinate_task(f"Route '{task}' to {target_agent}")


# Factory function for creating agent teams
def create_agent_team(model: BaseChatModel) -> Dict[str, CoreAgent]:
    """Create a team of specialized agents"""
    return {
        "code_review": CodeReviewAgent(model),
        "research": ResearchAgent(model),
        "customer_service": CustomerServiceAgent(model),
        "data_analysis": DataAnalysisAgent(model),
        "creative_writing": CreativeWritingAgent(model)
    }


# Example usage
def demo_specialized_agents():
    """Demonstrate specialized agents in action"""
    print("Specialized Agent Framework Demo")
    print("=" * 40)
    
    model = MockChatModel()
    
    # Create specialized agents
    code_agent = CodeReviewAgent(model)
    research_agent = ResearchAgent(model)
    cs_agent = CustomerServiceAgent(model)
    
    # Demo code review agent
    print("\n1. Code Review Agent:")
    print(f"Name: {code_agent.config.name}")
    print(f"Tools: {[tool.name for tool in code_agent.config.tools]}")
    
    # Demo research agent  
    print("\n2. Research Agent:")
    print(f"Name: {research_agent.config.name}")
    print(f"Response Format: {research_agent.config.response_format.__name__}")
    
    # Demo customer service agent
    print("\n3. Customer Service Agent:")
    print(f"Name: {cs_agent.config.name}")
    print(f"Human Feedback: {cs_agent.config.enable_human_feedback}")
    
    # Create agent team
    team = create_agent_team(model)
    supervisor = MultiAgentSupervisor(model, team)
    
    print(f"\n4. Multi-Agent Team:")
    print(f"Supervisor: {supervisor.config.name}")
    print(f"Team Members: {list(team.keys())}")
    print(f"Supervised Agents: {len(supervisor.supervisor_manager.agents)}")


if __name__ == "__main__":
    demo_specialized_agents()