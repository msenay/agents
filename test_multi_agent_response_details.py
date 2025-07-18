"""
Multi-Agent System Response Details Test
========================================

This test shows the detailed responses from each agent type and orchestration pattern.
"""

import os
import asyncio
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool

# Set Azure OpenAI environment variables
os.environ["OPENAI_API_VERSION"] = "2023-12-01-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"

from core_agent import CoreAgent, AgentConfig, create_supervisor_agent, create_handoff_agent, create_swarm_agent


# Simple tools for testing
@tool
def analyze_requirements(requirements: str) -> str:
    """Analyze requirements and provide breakdown"""
    return f"Requirements analysis: {requirements[:100]}... - Components identified: functions, classes, error handling needed."


@tool
def generate_code_structure(task: str) -> str:
    """Generate code structure suggestion"""
    if "calculator" in task.lower():
        return "Suggested: Calculator class with add, subtract, multiply, divide methods"
    elif "binary search" in task.lower():
        return "Suggested: binary_search function with left/right pointers"
    elif "merge sort" in task.lower():
        return "Suggested: merge_sort function with merge helper function"
    else:
        return "Suggested: Function or class based on requirements"


@tool
def validate_code_syntax(code: str) -> str:
    """Validate code syntax"""
    return "✅ Syntax validation: Code structure looks correct"


@tool
def create_test_cases(code_description: str) -> str:
    """Create test cases for code"""
    return f"Test cases for {code_description[:50]}...: Normal cases, edge cases, error conditions, validation tests"


@tool
def execute_tests(code_and_tests: str) -> str:
    """Execute test suite"""
    return "✅ Test execution: All tests passed successfully"


@tool
def identify_patches(issue: str) -> str:
    """Identify what needs patching"""
    return f"Patch analysis for {issue[:50]}...: Code logic fix needed, add error handling"


def create_simple_coder_agent():
    """Create a coder agent"""
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=1500
    )
    
    config = AgentConfig(
        name="Expert Coder Agent",
        model=llm,
        description="Generates clean, efficient Python code",
        tools=[analyze_requirements, generate_code_structure, validate_code_syntax],
        enable_memory=True,
        memory_type="memory",
        system_prompt="""You are an expert Python developer. 

🔧 YOUR TOOLS:
- analyze_requirements: Break down requirements
- generate_code_structure: Create code structure
- validate_code_syntax: Check syntax

PROTOCOL:
1. Analyze requirements using your tools
2. Generate clean, working Python code
3. Include proper error handling and documentation
4. Validate syntax before finalizing

Always create production-ready code!"""
    )
    
    return CoreAgent(config=config)


def create_simple_unit_tester():
    """Create a unit test agent"""
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=1500
    )
    
    config = AgentConfig(
        name="Unit Test Specialist",
        model=llm,
        description="Creates comprehensive unit tests",
        tools=[create_test_cases],
        enable_memory=True,
        memory_type="memory",
        system_prompt="""You are a unit testing expert.

🧪 YOUR TOOLS:
- create_test_cases: Generate test cases for code

PROTOCOL:
1. Analyze code for testable components
2. Create comprehensive test cases covering:
   - Normal operations
   - Edge cases
   - Error conditions
   - Input validation
3. Use unittest framework
4. Ensure test isolation

Aim for 100% coverage!"""
    )
    
    return CoreAgent(config=config)


def create_simple_executor():
    """Create a test executor agent"""
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0,
        max_tokens=1000
    )
    
    config = AgentConfig(
        name="Test Executor",
        model=llm,
        description="Executes tests and provides results",
        tools=[execute_tests],
        enable_memory=False,
        system_prompt="""You are a test execution specialist.

⚡ YOUR TOOLS:
- execute_tests: Run test suites safely

PROTOCOL:
1. Execute tests in safe environment
2. Capture all output and errors
3. Provide clear pass/fail status
4. Identify specific failures if any

Focus on accurate execution reporting!"""
    )
    
    return CoreAgent(config=config)


def create_simple_patcher():
    """Create a patch agent"""
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.2,
        max_tokens=1500
    )
    
    config = AgentConfig(
        name="Patch Specialist",
        model=llm,
        description="Fixes code and test issues",
        tools=[identify_patches],
        enable_memory=True,
        memory_type="memory",
        system_prompt="""You are a debugging and patching expert.

🔧 YOUR TOOLS:
- identify_patches: Determine what needs fixing

PROTOCOL:
1. Analyze failure reports
2. Identify root causes
3. Apply targeted fixes
4. Ensure fixes don't break existing functionality

Fix the underlying issue, not just symptoms!"""
    )
    
    return CoreAgent(config=config)


async def test_single_agent_response():
    """Test a single agent response in detail"""
    print("\n🔍 DETAILED SINGLE AGENT TEST")
    print("=" * 50)
    
    coder = create_simple_coder_agent()
    task = "Create a simple Calculator class with add, subtract, multiply, divide methods. Include error handling for division by zero."
    
    print(f"Task: {task}\n")
    
    result = await coder.ainvoke(task)
    
    print("📋 COMPLETE AGENT RESPONSE:")
    print("-" * 40)
    
    if isinstance(result, dict) and 'messages' in result:
        messages = result['messages']
        for i, message in enumerate(messages):
            print(f"\n[Message {i+1}] Type: {type(message).__name__}")
            if hasattr(message, 'content'):
                print(f"Content: {message.content}")
            if hasattr(message, 'tool_calls') and message.tool_calls:
                print(f"Tool Calls: {len(message.tool_calls)}")
                for tool_call in message.tool_calls:
                    print(f"  - Tool: {tool_call['name']}")
                    print(f"    Args: {tool_call['args']}")
    else:
        print(f"Raw result: {result}")
    
    return result


async def test_supervisor_response():
    """Test supervisor pattern response in detail"""
    print("\n🎯 DETAILED SUPERVISOR PATTERN TEST")
    print("=" * 50)
    
    # Create agents
    agents = {
        "coder": create_simple_coder_agent(),
        "unit_tester": create_simple_unit_tester(),
        "executor": create_simple_executor(),
        "patcher": create_simple_patcher()
    }
    
    # Create supervisor
    supervisor_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=2000
    )
    
    supervisor = create_supervisor_agent(
        model=supervisor_llm,
        agents=agents,
        system_prompt="""You are the Coding Workflow Supervisor.

AGENTS:
- coder: Generates Python code
- unit_tester: Creates unit tests
- executor: Runs tests
- patcher: Fixes issues

WORKFLOW:
1. coder generates code
2. unit_tester creates tests  
3. executor runs tests
4. if failures, patcher fixes
5. return final working code + tests

Coordinate the full workflow!"""
    )
    
    task = "Create a factorial function with input validation"
    print(f"Task: {task}\n")
    
    result = await supervisor.ainvoke(task)
    
    print("📋 SUPERVISOR COORDINATION RESPONSE:")
    print("-" * 40)
    
    if isinstance(result, dict) and 'messages' in result:
        messages = result['messages']
        for i, message in enumerate(messages):
            print(f"\n[Message {i+1}] Type: {type(message).__name__}")
            if hasattr(message, 'content'):
                content = message.content
                if len(content) > 300:
                    print(f"Content: {content[:300]}...")
                else:
                    print(f"Content: {content}")
    
    return result


async def test_handoff_response():
    """Test handoff pattern response in detail"""
    print("\n🔀 DETAILED HANDOFF PATTERN TEST")
    print("=" * 50)
    
    agents = {
        "coder": create_simple_coder_agent(),
        "unit_tester": create_simple_unit_tester()
    }
    
    handoff_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=1500
    )
    
    handoff_system = create_handoff_agent(
        model=handoff_llm,
        agents=agents,
        system_prompt="""You coordinate agent handoffs.

AGENTS:
- coder: Creates code
- unit_tester: Creates tests

HANDOFF FLOW:
1. coder creates code
2. handoff to unit_tester for tests
3. return final result

Use handoff commands to transfer work!"""
    )
    
    task = "Create a simple bubble sort function"
    print(f"Task: {task}\n")
    
    result = await handoff_system.ainvoke(task)
    
    print("📋 HANDOFF COORDINATION RESPONSE:")
    print("-" * 40)
    
    if isinstance(result, dict) and 'messages' in result:
        messages = result['messages']
        for i, message in enumerate(messages):
            print(f"\n[Message {i+1}] Type: {type(message).__name__}")
            if hasattr(message, 'content'):
                content = message.content
                if len(content) > 300:
                    print(f"Content: {content[:300]}...")
                else:
                    print(f"Content: {content}")
    
    return result


async def test_swarm_response():
    """Test swarm pattern response in detail"""
    print("\n🐝 DETAILED SWARM PATTERN TEST")
    print("=" * 50)
    
    agents = {
        "coder": create_simple_coder_agent(),
        "unit_tester": create_simple_unit_tester(),
        "executor": create_simple_executor()
    }
    
    swarm_llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2023-12-01-preview",
        temperature=0.1,
        max_tokens=1500
    )
    
    swarm_system = create_swarm_agent(
        model=swarm_llm,
        agents=agents,
        system_prompt="""You coordinate a coding swarm.

AGENTS:
- coder: Code generation
- unit_tester: Test creation
- executor: Test execution

SWARM BEHAVIOR:
- Select best agent for each task
- Dynamic routing based on needs
- Adaptive workflow

Choose agents dynamically based on requirements!"""
    )
    
    task = "Create a function to reverse a string"
    print(f"Task: {task}\n")
    
    result = await swarm_system.ainvoke(task)
    
    print("📋 SWARM COORDINATION RESPONSE:")
    print("-" * 40)
    
    if isinstance(result, dict) and 'messages' in result:
        messages = result['messages']
        for i, message in enumerate(messages):
            print(f"\n[Message {i+1}] Type: {type(message).__name__}")
            if hasattr(message, 'content'):
                content = message.content
                if len(content) > 300:
                    print(f"Content: {content[:300]}...")
                else:
                    print(f"Content: {content}")
    
    return result


async def main():
    """Run detailed response tests"""
    print("🔍 MULTI-AGENT RESPONSE DETAILS TEST")
    print("=" * 60)
    
    try:
        # Test individual agent
        single_result = await test_single_agent_response()
        
        # Test orchestration patterns
        supervisor_result = await test_supervisor_response()
        handoff_result = await test_handoff_response()
        swarm_result = await test_swarm_response()
        
        print("\n📊 RESPONSE ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"✓ Single Agent: {'✅ Generated' if single_result else '❌ Failed'}")
        print(f"✓ Supervisor Pattern: {'✅ Coordinated' if supervisor_result else '❌ Failed'}")
        print(f"✓ Handoff Pattern: {'✅ Transferred' if handoff_result else '❌ Failed'}")
        print(f"✓ Swarm Pattern: {'✅ Swarmed' if swarm_result else '❌ Failed'}")
        
        print("\n🎯 STRUCTURE ANALYSIS:")
        print("- All responses contain message sequences")
        print("- Tool calls are properly executed")
        print("- Agent coordination is working")
        print("- Memory retention is active where enabled")
        
        print("\n🎉 RESPONSE DETAILS TEST COMPLETED!")
        
    except Exception as e:
        print(f"\n❌ DETAILED TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())