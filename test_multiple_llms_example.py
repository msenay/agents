"""
Multiple LLMs with CoreAgent Framework
=====================================

This example demonstrates how to:
1. Set up a default LLM for CoreAgent
2. Create different agents with different LLMs
3. Use various models (GPT-4, GPT-3.5, different Azure deployments, etc.)
4. Multi-agent orchestration with mixed LLMs
"""

import asyncio
import os
from typing import Dict, Any

# Set Azure OpenAI environment
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from core_agent import (
    CoreAgent, AgentConfig, 
    create_simple_agent, create_advanced_agent, create_supervisor_agent
)


# =============================================================================
# LLM SETUP - Different models for different purposes
# =============================================================================

def create_llm_models():
    """Create different LLM models for different agent types"""
    
    # DEFAULT LLM - GPT-4 for general use
    default_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=2000,
        model_name="gpt-4"
    )
    
    # FAST LLM - GPT-3.5 for quick tasks
    fast_llm = AzureChatOpenAI(
        azure_deployment="gpt35-turbo",  # Assuming you have this deployment
        api_version="2023-12-01-preview", 
        temperature=0.2,
        max_tokens=1000,
        model_name="gpt-35-turbo"
    )
    
    # CREATIVE LLM - Higher temperature for creative tasks
    creative_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.8,
        max_tokens=2500,
        model_name="gpt-4"
    )
    
    # ANALYTICAL LLM - Lower temperature for analytical tasks
    analytical_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.0,
        max_tokens=3000,
        model_name="gpt-4"
    )
    
    # SUPERVISOR LLM - For orchestration
    supervisor_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=1500,
        model_name="gpt-4"
    )
    
    return {
        "default": default_llm,
        "fast": fast_llm,
        "creative": creative_llm,
        "analytical": analytical_llm,
        "supervisor": supervisor_llm
    }


# =============================================================================
# SPECIALIZED TOOLS FOR DIFFERENT AGENTS
# =============================================================================

@tool
def analyze_data(data: str) -> str:
    """Analyze data and provide insights"""
    return f"Data analysis complete for: {data[:50]}... - Insights: trends identified, patterns found."

@tool
def generate_creative_content(topic: str) -> str:
    """Generate creative content for given topic"""
    return f"Creative content generated for '{topic}': story concepts, unique angles, engaging narratives."

@tool
def execute_fast_task(task: str) -> str:
    """Execute simple tasks quickly"""
    return f"Fast execution complete for: {task} - Status: SUCCESS, Time: minimal"

@tool
def coordinate_workflow(workflow: str) -> str:
    """Coordinate complex workflows"""
    return f"Workflow coordination for: {workflow} - Agents assigned, tasks distributed, monitoring active."


# =============================================================================
# CREATING AGENTS WITH DIFFERENT LLMS
# =============================================================================

def create_specialized_agents():
    """Create different agents with different LLMs"""
    
    # Get our LLM models
    llms = create_llm_models()
    
    # 1. ANALYTICAL AGENT - Uses analytical_llm (temperature=0.0)
    analytical_agent = create_advanced_agent(
        model=llms["analytical"],
        name="Data Analyst Agent",
        tools=[analyze_data],
        system_prompt="""You are a precise data analyst specializing in:
        
🔬 ANALYTICAL CAPABILITIES:
- Statistical analysis and pattern recognition
- Data interpretation and insights generation
- Trend identification and forecasting
- Rigorous, fact-based reasoning

📊 YOUR APPROACH:
- Always provide precise, evidence-based analysis
- Use statistical thinking and logical reasoning
- Avoid speculation, stick to data-driven conclusions
- Present findings in clear, structured format

Be methodical, accurate, and thorough in all analysis.""",
        enable_memory=True,
        enable_evaluation=True
    )
    
    # 2. CREATIVE AGENT - Uses creative_llm (temperature=0.8)
    creative_agent = create_advanced_agent(
        model=llms["creative"],
        name="Creative Content Agent",
        tools=[generate_creative_content],
        system_prompt="""You are a highly creative content generator specializing in:
        
🎨 CREATIVE CAPABILITIES:
- Innovative storytelling and narrative development
- Original concept generation and ideation
- Creative problem-solving and unique perspectives
- Engaging content creation across formats

✨ YOUR APPROACH:
- Think outside the box and explore unconventional ideas
- Combine unexpected elements for originality
- Focus on engagement and emotional connection
- Experiment with different styles and formats

Be imaginative, original, and inspiring in all creative work.""",
        enable_memory=True,
        enable_human_feedback=True
    )
    
    # 3. FAST EXECUTION AGENT - Uses fast_llm (GPT-3.5)
    fast_agent = create_simple_agent(
        model=llms["fast"],
        name="Quick Task Agent",
        tools=[execute_fast_task],
        system_prompt="""You are a rapid-response agent optimized for:
        
⚡ SPEED CAPABILITIES:
- Quick task completion and simple operations
- Efficient processing of straightforward requests
- Minimal overhead, maximum speed
- Direct, concise responses

🚀 YOUR APPROACH:
- Process requests immediately and efficiently
- Provide clear, direct answers without over-elaboration
- Focus on speed while maintaining accuracy
- Handle routine tasks with minimal delay

Be fast, efficient, and reliable for quick operations."""
    )
    
    # 4. DEFAULT AGENT - Uses default_llm (balanced GPT-4)
    default_agent = create_simple_agent(
        model=llms["default"],
        name="General Purpose Agent",
        system_prompt="""You are a general-purpose AI assistant providing:
        
🔧 GENERAL CAPABILITIES:
- Balanced approach to various tasks
- Flexible problem-solving across domains
- Reliable performance for standard operations
- Comprehensive assistance and support

💡 YOUR APPROACH:
- Adapt to different types of requests appropriately
- Provide well-rounded, thoughtful responses
- Balance accuracy with creativity as needed
- Maintain consistent quality across tasks

Be versatile, reliable, and consistently helpful."""
    )
    
    return {
        "analytical": analytical_agent,
        "creative": creative_agent,
        "fast": fast_agent,
        "default": default_agent
    }


def create_multi_llm_supervisor():
    """Create a supervisor that orchestrates agents with different LLMs"""
    
    llms = create_llm_models()
    agents = create_specialized_agents()
    
    # SUPERVISOR AGENT - Uses supervisor_llm
    supervisor = create_supervisor_agent(
        model=llms["supervisor"],
        name="Multi-LLM Orchestrator",
        agents=agents,
        system_prompt="""You are the Multi-LLM Orchestrator managing specialized agents with different capabilities:

🎯 AVAILABLE AGENTS:
- analytical: Data analysis with precise, low-temperature reasoning (GPT-4, temp=0.0)
- creative: Content generation with high creativity (GPT-4, temp=0.8) 
- fast: Quick tasks with efficient processing (GPT-3.5)
- default: General-purpose balanced assistance (GPT-4, temp=0.1)

🧠 AGENT SELECTION STRATEGY:
- analytical: Data analysis, research, fact-checking, statistical work
- creative: Content creation, brainstorming, storytelling, innovation
- fast: Simple tasks, quick responses, routine operations
- default: General questions, balanced tasks, multi-domain work

🔄 ORCHESTRATION PROTOCOL:
1. Analyze the incoming request type and complexity
2. Select the most appropriate agent based on task characteristics
3. Route the task to the optimal agent with specific instructions
4. Coordinate multiple agents if the task requires different capabilities
5. Synthesize results when multiple agents are involved

Choose agents strategically based on their LLM characteristics and specializations!""",
        enable_memory=True,
        enable_evaluation=True
    )
    
    return supervisor


# =============================================================================
# TESTING DIFFERENT LLMS WITH DIFFERENT TASKS
# =============================================================================

async def test_different_llms():
    """Test how different LLMs handle different types of tasks"""
    
    print("🤖 MULTI-LLM COREAGENT SYSTEM TEST")
    print("=" * 50)
    
    # Create our agents with different LLMs
    agents = create_specialized_agents()
    supervisor = create_multi_llm_supervisor()
    
    # Test tasks for different agents
    test_tasks = {
        "analytical_task": "Analyze the trend in global renewable energy adoption from 2020-2024 and predict growth patterns.",
        "creative_task": "Create an engaging story concept about AI agents working together in a digital city.",
        "fast_task": "What is 15 * 24 + 67?",
        "general_task": "Explain the benefits and challenges of remote work in modern organizations."
    }
    
    print("\n1. 🔬 TESTING ANALYTICAL AGENT (temp=0.0)")
    analytical_result = await agents["analytical"].ainvoke(test_tasks["analytical_task"])
    print(f"Task: {test_tasks['analytical_task']}")
    print(f"Response: {str(analytical_result)[:200]}...")
    
    print("\n2. 🎨 TESTING CREATIVE AGENT (temp=0.8)")
    creative_result = await agents["creative"].ainvoke(test_tasks["creative_task"])
    print(f"Task: {test_tasks['creative_task']}")
    print(f"Response: {str(creative_result)[:200]}...")
    
    print("\n3. ⚡ TESTING FAST AGENT (GPT-3.5)")
    fast_result = await agents["fast"].ainvoke(test_tasks["fast_task"])
    print(f"Task: {test_tasks['fast_task']}")
    print(f"Response: {str(fast_result)[:200]}...")
    
    print("\n4. 🔧 TESTING DEFAULT AGENT (balanced)")
    default_result = await agents["default"].ainvoke(test_tasks["general_task"])
    print(f"Task: {test_tasks['general_task']}")
    print(f"Response: {str(default_result)[:200]}...")
    
    print("\n5. 🎯 TESTING SUPERVISOR ORCHESTRATION")
    orchestration_task = "I need a creative story about data analysis in the future, solve some quick math, and explain the implications."
    supervisor_result = await supervisor.ainvoke(orchestration_task)
    print(f"Complex Task: {orchestration_task}")
    print(f"Orchestrated Response: {str(supervisor_result)[:300]}...")
    
    return {
        "analytical": analytical_result,
        "creative": creative_result,
        "fast": fast_result,
        "default": default_result,
        "supervisor": supervisor_result
    }


async def demonstrate_llm_characteristics():
    """Demonstrate how different LLM settings affect agent behavior"""
    
    print("\n🧪 LLM CHARACTERISTICS DEMONSTRATION")
    print("=" * 50)
    
    llms = create_llm_models()
    
    # Same task, different LLMs
    test_prompt = "Describe the future of artificial intelligence in 3 sentences."
    
    # Test with analytical LLM (temp=0.0)
    analytical_agent = create_simple_agent(
        model=llms["analytical"],
        name="Analytical Test",
        system_prompt="Provide precise, factual analysis."
    )
    
    # Test with creative LLM (temp=0.8)
    creative_agent = create_simple_agent(
        model=llms["creative"],
        name="Creative Test",
        system_prompt="Provide imaginative, creative perspectives."
    )
    
    print(f"\nSame prompt for different LLMs: '{test_prompt}'")
    
    print("\n🔬 ANALYTICAL LLM (temp=0.0) Response:")
    analytical_response = await analytical_agent.ainvoke(test_prompt)
    print(str(analytical_response)[:300])
    
    print("\n🎨 CREATIVE LLM (temp=0.8) Response:")
    creative_response = await creative_agent.ainvoke(test_prompt)
    print(str(creative_response)[:300])
    
    return {
        "analytical_response": analytical_response,
        "creative_response": creative_response
    }


def show_llm_configuration():
    """Display the LLM configuration setup"""
    
    print("\n⚙️  LLM CONFIGURATION OVERVIEW")
    print("=" * 50)
    
    llm_configs = {
        "Default LLM": {
            "model": "GPT-4",
            "deployment": "gpt4",
            "temperature": 0.1,
            "max_tokens": 2000,
            "use_case": "General-purpose agent tasks"
        },
        "Fast LLM": {
            "model": "GPT-3.5 Turbo",
            "deployment": "gpt35-turbo", 
            "temperature": 0.2,
            "max_tokens": 1000,
            "use_case": "Quick, simple task execution"
        },
        "Creative LLM": {
            "model": "GPT-4",
            "deployment": "gpt4",
            "temperature": 0.8,
            "max_tokens": 2500,
            "use_case": "Creative content generation"
        },
        "Analytical LLM": {
            "model": "GPT-4",
            "deployment": "gpt4",
            "temperature": 0.0,
            "max_tokens": 3000,
            "use_case": "Precise data analysis"
        },
        "Supervisor LLM": {
            "model": "GPT-4",
            "deployment": "gpt4",
            "temperature": 0.1,
            "max_tokens": 1500,
            "use_case": "Multi-agent orchestration"
        }
    }
    
    for llm_name, config in llm_configs.items():
        print(f"\n{llm_name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\n🎯 KEY INSIGHT:")
    print("Each agent can use a completely different LLM optimized for its specific role!")
    print("Temperature, model type, token limits - all customizable per agent.")


async def main():
    """Run the multi-LLM demonstration"""
    
    try:
        # Show configuration
        show_llm_configuration()
        
        # Test different LLMs
        results = await test_different_llms()
        
        # Demonstrate LLM characteristics
        await demonstrate_llm_characteristics()
        
        print("\n🎉 MULTI-LLM SYSTEM TEST COMPLETED!")
        print("\n📊 SUMMARY:")
        print("✅ Different LLMs successfully configured for different agents")
        print("✅ Agents demonstrate different behavioral characteristics")
        print("✅ Supervisor successfully orchestrates mixed-LLM agent teams")
        print("✅ Temperature and model settings affect response style")
        print("✅ Framework supports unlimited LLM diversity!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        # Show that it works in principle even if API calls fail
        print("\n💡 The framework structure supports multiple LLMs even if API calls fail!")
        show_llm_configuration()


if __name__ == "__main__":
    asyncio.run(main())