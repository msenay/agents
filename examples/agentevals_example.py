"""
AgentEvals Evaluation Example

This file demonstrates how to use AgentEvals for evaluating agent performance
in CoreAgent framework, including trajectory evaluation and LLM-as-a-judge.
"""

import sys
import os
import json
import asyncio
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_agent import CoreAgent, AgentConfig, create_evaluated_agent, AGENTEVALS_AVAILABLE
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool


# Mock model for demonstration
class MockChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        content = f"Mock response to: {messages[-1].content if messages else 'empty'}"
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "mock"


@tool
def get_weather(city: str) -> str:
    """Get weather for a city"""
    return f"Weather in {city}: Sunny, 22°C"


@tool  
def get_directions(destination: str) -> str:
    """Get directions to destination"""
    return f"Directions to {destination}: Head north for 2 miles"


def demo_agentevals_availability():
    """Check AgentEvals availability and features"""
    print("\n📊 AGENTEVALS AVAILABILITY")
    print("=" * 40)
    
    print(f"🔍 AgentEvals Available: {'✅ Yes' if AGENTEVALS_AVAILABLE else '❌ No'}")
    
    if AGENTEVALS_AVAILABLE:
        try:
            from agentevals import AgentEvaluator
            from agentevals.trajectory.match import create_trajectory_match_evaluator
            from agentevals.trajectory.llm import create_trajectory_llm_as_judge
            print("✅ AgentEvaluator: Available")
            print("✅ Trajectory Match Evaluator: Available")
            print("✅ LLM-as-a-Judge Evaluator: Available")
        except ImportError as e:
            print(f"⚠️ Partial import error: {e}")
    else:
        print("📥 To enable AgentEvals functionality:")
        print("   pip install -U agentevals")


def demo_evaluation_types():
    """Demonstrate different evaluation types"""
    print("\n📋 AGENTEVALS EVALUATION TYPES")
    print("=" * 40)
    
    evaluation_types = {
        "Basic Evaluation": {
            "description": "Basic response quality evaluation",
            "use_case": "General quality assessment",
            "metrics": ["accuracy", "relevance", "helpfulness"]
        },
        "Trajectory Evaluation": {
            "description": "Tool usage sequence evaluation",
            "use_case": "Agent workflow validation",
            "features": ["Tool call ordering", "Argument validation", "Sequence matching"]
        },
        "LLM-as-a-Judge": {
            "description": "LLM-powered evaluation",
            "use_case": "Sophisticated quality assessment",
            "features": ["Natural language evaluation", "Context-aware scoring", "Reference comparison"]
        }
    }
    
    for eval_type, info in evaluation_types.items():
        print(f"\n🔹 {eval_type}:")
        print(f"   Description: {info['description']}")
        print(f"   Use case: {info['use_case']}")
        if 'metrics' in info:
            print(f"   Metrics: {', '.join(info['metrics'])}")
        if 'features' in info:
            print(f"   Features: {', '.join(info['features'])}")


def demo_evaluated_agent_creation():
    """Demonstrate creating agents with evaluation capabilities"""
    print("\n🤖 EVALUATED AGENT CREATION")
    print("=" * 40)
    
    model = MockChatModel()
    tools = [get_weather, get_directions]
    
    # Test different evaluation configurations
    evaluation_configs = [
        {
            "name": "BasicEvalAgent",
            "metrics": ["accuracy", "relevance"],
            "description": "Basic evaluation metrics"
        },
        {
            "name": "ComprehensiveEvalAgent", 
            "metrics": ["accuracy", "relevance", "helpfulness", "efficiency"],
            "description": "Comprehensive evaluation"
        },
        {
            "name": "CustomEvalAgent",
            "metrics": ["task_completion", "tool_usage", "response_quality"],
            "description": "Custom evaluation metrics"
        }
    ]
    
    agents = []
    for config in evaluation_configs:
        print(f"\n📝 Creating {config['name']}:")
        
        # Create agent with manual configuration
        agent_config = AgentConfig(
            name=config["name"],
            model=model,
            tools=tools,
            system_prompt=f"You are an assistant for {config['description']}.",
            enable_evaluation=True,
            evaluation_metrics=config["metrics"],
            enable_memory=True
        )
        
        agent = CoreAgent(agent_config)
        agents.append(agent)
        
        print(f"✅ Agent created: {agent.config.name}")
        print(f"   Evaluation enabled: {agent.config.enable_evaluation}")
        print(f"   Metrics: {agent.config.evaluation_metrics}")
        
        # Display evaluator status
        evaluator_status = agent.get_evaluator_status()
        print(f"   Evaluator status: {evaluator_status}")
    
    return agents


def demo_agentevals_factory_function():
    """Demonstrate AgentEvals factory function"""
    print("\n🏭 AGENTEVALS FACTORY FUNCTION")
    print("=" * 40)
    
    model = MockChatModel()
    tools = [get_weather, get_directions]
    
    # Create agent using factory function
    agent = create_evaluated_agent(
        model=model,
        tools=tools,
        evaluation_metrics=["accuracy", "relevance", "tool_usage", "efficiency"],
        prompt="You are an assistant with comprehensive evaluation capabilities."
    )
    
    print(f"✅ Evaluated agent created via factory: {agent.config.name}")
    print(f"📊 Configuration:")
    print(f"   Evaluation enabled: {agent.config.enable_evaluation}")
    print(f"   Metrics: {agent.config.evaluation_metrics}")
    
    evaluator_status = agent.get_evaluator_status()
    print(f"   Available evaluators:")
    for evaluator, available in evaluator_status.items():
        status = "✅" if available else "❌"
        print(f"     {status} {evaluator}")
    
    return agent


def demo_trajectory_evaluation():
    """Demonstrate trajectory evaluation"""
    print("\n🛤️ TRAJECTORY EVALUATION")
    print("=" * 40)
    
    model = MockChatModel()
    agent = create_evaluated_agent(model, tools=[get_weather, get_directions])
    
    # Sample trajectory data
    outputs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "san francisco"}),
                    }
                },
                {
                    "function": {
                        "name": "get_directions", 
                        "arguments": json.dumps({"destination": "presidio"}),
                    }
                }
            ],
        }
    ]
    
    reference_outputs = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_weather",
                        "arguments": json.dumps({"city": "san francisco"}),
                    }
                },
            ],
        }
    ]
    
    print("📝 Sample Trajectory Evaluation:")
    print("Agent outputs:")
    for output in outputs:
        if "tool_calls" in output:
            for tool_call in output["tool_calls"]:
                func = tool_call["function"]
                print(f"  - {func['name']}({func['arguments']})")
    
    print("Reference outputs:")
    for output in reference_outputs:
        if "tool_calls" in output:
            for tool_call in output["tool_calls"]:
                func = tool_call["function"]
                print(f"  - {func['name']}({func['arguments']})")
    
    # Evaluate trajectory
    result = agent.evaluate_trajectory(outputs, reference_outputs)
    print(f"\n📊 Trajectory evaluation result:")
    if "error" in result:
        print(f"   ⚠️ {result['error']}")
    else:
        print(f"   ✅ Evaluation completed: {result}")


def demo_llm_judge_evaluation():
    """Demonstrate LLM-as-a-judge evaluation"""
    print("\n👨‍⚖️ LLM-AS-A-JUDGE EVALUATION")
    print("=" * 40)
    
    model = MockChatModel()
    agent = create_evaluated_agent(model, tools=[get_weather])
    
    # Sample outputs for LLM judge
    outputs = [
        {
            "role": "assistant",
            "content": "The weather in San Francisco is sunny with a temperature of 22°C."
        }
    ]
    
    reference_outputs = [
        {
            "role": "assistant", 
            "content": "San Francisco weather: Sunny, 22 degrees Celsius."
        }
    ]
    
    print("📝 Sample LLM Judge Evaluation:")
    print(f"Agent output: {outputs[0]['content']}")
    print(f"Reference: {reference_outputs[0]['content']}")
    
    # Evaluate with LLM judge
    result = agent.evaluate_with_llm_judge(outputs, reference_outputs)
    print(f"\n📊 LLM judge evaluation result:")
    if "error" in result:
        print(f"   ⚠️ {result['error']}")
    else:
        print(f"   ✅ Evaluation completed: {result}")


def demo_evaluation_dataset():
    """Demonstrate evaluation dataset creation"""
    print("\n📁 EVALUATION DATASET CREATION")
    print("=" * 40)
    
    model = MockChatModel()
    agent = create_evaluated_agent(model)
    
    # Sample conversation data
    conversations = [
        {
            "input_messages": [
                {"role": "user", "content": "What's the weather in New York?"}
            ],
            "expected_output_messages": [
                {"role": "assistant", "content": "Let me check the weather for you."},
                {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "new york"}'}}]}
            ]
        },
        {
            "input_messages": [
                {"role": "user", "content": "How do I get to the airport?"}
            ],
            "expected_output_messages": [
                {"role": "assistant", "content": "I'll help you with directions."},
                {"role": "assistant", "tool_calls": [{"function": {"name": "get_directions", "arguments": '{"destination": "airport"}'}}]}
            ]
        }
    ]
    
    # Create evaluation dataset
    dataset = agent.create_evaluation_dataset(conversations)
    
    print(f"✅ Created evaluation dataset with {len(dataset)} examples")
    print("📋 Dataset format:")
    for i, example in enumerate(dataset):
        print(f"   Example {i+1}:")
        print(f"     Input: {len(example['input']['messages'])} messages")
        print(f"     Output: {len(example['output']['messages'])} messages")
    
    return dataset


def demo_evaluation_metrics():
    """Demonstrate evaluation metrics and configuration"""
    print("\n📊 EVALUATION METRICS")
    print("=" * 40)
    
    metrics_categories = {
        "Quality Metrics": [
            "accuracy",
            "relevance", 
            "helpfulness",
            "coherence"
        ],
        "Performance Metrics": [
            "efficiency",
            "response_time",
            "tool_usage",
            "task_completion"
        ],
        "User Experience": [
            "clarity",
            "friendliness",
            "informativeness",
            "appropriateness"
        ]
    }
    
    for category, metrics in metrics_categories.items():
        print(f"\n🔹 {category}:")
        for metric in metrics:
            print(f"   • {metric}")
    
    # Show custom metrics configuration
    print(f"\n⚙️ Custom Metrics Configuration:")
    model = MockChatModel()
    custom_agent = create_evaluated_agent(
        model,
        evaluation_metrics=["task_completion", "tool_efficiency", "response_quality"]
    )
    print(f"   Configured metrics: {custom_agent.config.evaluation_metrics}")


def demo_evaluation_best_practices():
    """Demonstrate evaluation best practices"""
    print("\n🎯 EVALUATION BEST PRACTICES")
    print("=" * 40)
    
    practices = [
        "🔸 Define clear evaluation metrics aligned with your use case",
        "🔸 Use trajectory evaluation for workflow-dependent agents",
        "🔸 Implement LLM-as-a-judge for nuanced quality assessment",
        "🔸 Create comprehensive reference datasets",
        "🔸 Test evaluation setup before production deployment",
        "🔸 Monitor evaluation results over time for performance trends",
        "🔸 Use multiple evaluation methods for comprehensive assessment",
        "🔸 Regularly update evaluation criteria based on user feedback",
        "🔸 Consider domain-specific evaluation metrics",
        "🔸 Document evaluation methodology for reproducibility"
    ]
    
    for practice in practices:
        print(f"   {practice}")


def demo_integration_examples():
    """Show practical integration examples"""
    print("\n💡 INTEGRATION EXAMPLES")
    print("=" * 40)
    
    examples = {
        "Customer Support Agent": {
            "metrics": ["helpfulness", "accuracy", "response_time"],
            "evaluation_type": "LLM-as-a-judge",
            "use_case": "Customer satisfaction assessment"
        },
        "Research Assistant": {
            "metrics": ["accuracy", "comprehensiveness", "source_quality"],
            "evaluation_type": "Trajectory + Quality",
            "use_case": "Information gathering validation"
        },
        "Task Automation Agent": {
            "metrics": ["task_completion", "efficiency", "error_rate"],
            "evaluation_type": "Trajectory matching",
            "use_case": "Workflow execution validation"
        },
        "Educational Tutor": {
            "metrics": ["clarity", "pedagogical_value", "engagement"],
            "evaluation_type": "LLM-as-a-judge",
            "use_case": "Teaching effectiveness assessment"
        }
    }
    
    for name, config in examples.items():
        print(f"\n🔹 {name}:")
        print(f"   Metrics: {', '.join(config['metrics'])}")
        print(f"   Evaluation: {config['evaluation_type']}")
        print(f"   Use case: {config['use_case']}")


async def run_all_demos():
    """Run all AgentEvals demonstrations"""
    print("📊 CoreAgent AgentEvals Evaluation Demo")
    print("=" * 60)
    
    # Run demos
    demo_agentevals_availability()
    demo_evaluation_types()
    demo_evaluated_agent_creation()
    demo_agentevals_factory_function()
    demo_trajectory_evaluation()
    demo_llm_judge_evaluation()
    demo_evaluation_dataset()
    demo_evaluation_metrics()
    demo_evaluation_best_practices()
    demo_integration_examples()
    
    print("\n" + "=" * 60)
    print("🎉 AgentEvals demo completed!")
    
    if not AGENTEVALS_AVAILABLE:
        print("\n📥 To enable full AgentEvals functionality:")
        print("pip install -U agentevals")
        print("\nAgentEvals provides:")
        print("• Trajectory evaluation for workflow validation")
        print("• LLM-as-a-judge for quality assessment")
        print("• Performance metrics and benchmarking")
        print("• Integration with LangSmith evaluations")


if __name__ == "__main__":
    asyncio.run(run_all_demos())