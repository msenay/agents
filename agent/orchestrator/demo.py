"""
Orchestrator Agent Demo

Demonstrates the OrchestratorAgent's ability to coordinate CoderAgent, TesterAgent, 
and ExecutorAgent in harmony for complete development workflows.
"""

import os
from typing import Dict, Any

# Check environment variables before importing
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Warning: OPENAI_API_KEY not set in environment")
    print("The demo may not work without proper API credentials")
    print("Please set: export OPENAI_API_KEY='your-api-key'")

from agent.orchestrator import OrchestratorAgent


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🎭 {title}")
    print(f"{'='*60}\n")


def demo_full_development_workflow():
    """Demo: Complete development workflow from idea to tested code"""
    print_section("Full Development Workflow")
    
    orchestrator = OrchestratorAgent(
        coordination_pattern="supervisor",
        enable_monitoring=True
    )
    
    # Request a complete development task
    request = """
    Create a Python class for managing a todo list with the following features:
    1. Add tasks with priority levels (high, medium, low)
    2. Mark tasks as complete
    3. Get tasks by priority
    4. Save/load tasks to/from JSON file
    
    Ensure the code is well-tested and validated.
    """
    
    print("📋 Request:", request)
    print("\n🚀 Starting orchestrated workflow...\n")
    
    # Orchestrate the workflow
    result = orchestrator.orchestrate(request, workflow_type="full_development")
    
    # Print results
    if result["success"]:
        print("\n✅ Workflow completed successfully!")
        print("\n📊 Report:")
        print(result["report"])
    else:
        print(f"\n❌ Workflow failed: {result['error']}")
    
    return orchestrator


def demo_code_review_workflow():
    """Demo: Code review and improvement workflow"""
    print_section("Code Review Workflow")
    
    orchestrator = OrchestratorAgent(
        coordination_pattern="supervisor",
        enable_monitoring=True
    )
    
    # Existing code to review
    existing_code = '''
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def find_prime_numbers(limit):
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes
'''
    
    request = f"""
    Review and improve this code:
    
    ```python
    {existing_code}
    ```
    
    Focus on:
    1. Performance optimization
    2. Code quality
    3. Test coverage
    4. Best practices
    """
    
    print("📋 Request: Review and improve existing code")
    print(f"\n📝 Code to review:\n{existing_code}")
    print("\n🚀 Starting code review workflow...\n")
    
    # Orchestrate the review
    result = orchestrator.orchestrate(request, workflow_type="code_review")
    
    # Print results
    if result["success"]:
        print("\n✅ Review completed successfully!")
        print("\n📊 Review Report:")
        print(result["report"])
    else:
        print(f"\n❌ Review failed: {result['error']}")
    
    return orchestrator


def demo_bug_fix_workflow():
    """Demo: Bug fix workflow with testing"""
    print_section("Bug Fix Workflow")
    
    orchestrator = OrchestratorAgent(
        coordination_pattern="pipeline",  # Use pipeline for strict sequence
        enable_monitoring=True
    )
    
    # Buggy code
    buggy_code = '''
def divide_numbers(a, b):
    return a / b

def calculate_average(numbers):
    total = sum(numbers)
    return divide_numbers(total, len(numbers))
'''
    
    request = f"""
    Fix the bugs in this code and ensure it handles edge cases properly:
    
    ```python
    {buggy_code}
    ```
    
    The code fails when:
    1. Dividing by zero
    2. Empty list in calculate_average
    3. Non-numeric values
    
    Fix these issues and add comprehensive tests.
    """
    
    print("📋 Request: Fix bugs and add error handling")
    print(f"\n🐛 Buggy code:\n{buggy_code}")
    print("\n🚀 Starting bug fix workflow...\n")
    
    # Orchestrate the fix
    result = orchestrator.orchestrate(request, workflow_type="bug_fix")
    
    # Print results
    if result["success"]:
        print("\n✅ Bug fix completed successfully!")
        print("\n📊 Fix Report:")
        print(result["report"])
    else:
        print(f"\n❌ Bug fix failed: {result['error']}")
    
    return orchestrator


def demo_parallel_tasks():
    """Demo: Parallel task execution using swarm pattern"""
    print_section("Parallel Task Execution (Swarm Pattern)")
    
    orchestrator = OrchestratorAgent(
        coordination_pattern="swarm",  # Use swarm for parallel execution
        enable_monitoring=True
    )
    
    request = """
    Create three independent utility modules in parallel:
    
    1. String utilities: Functions for string manipulation (capitalize_words, reverse_string, count_vowels)
    2. Math utilities: Functions for basic math operations (factorial, is_prime, gcd)
    3. Date utilities: Functions for date operations (days_between, is_weekend, next_business_day)
    
    Each module should have comprehensive tests.
    """
    
    print("📋 Request:", request)
    print("\n🚀 Starting parallel workflow with swarm pattern...\n")
    
    # Orchestrate parallel tasks
    result = orchestrator.orchestrate(request)
    
    # Print results
    if result["success"]:
        print("\n✅ Parallel tasks completed successfully!")
        print("\n📊 Execution Report:")
        print(result["report"])
    else:
        print(f"\n❌ Parallel execution failed: {result['error']}")
    
    return orchestrator


def demo_adaptive_pattern():
    """Demo: Adaptive pattern selection based on task"""
    print_section("Adaptive Pattern Selection")
    
    orchestrator = OrchestratorAgent(
        coordination_pattern="adaptive",  # Let orchestrator choose pattern
        enable_monitoring=True
    )
    
    request = """
    Build a REST API for a simple blog system with:
    1. CRUD operations for posts
    2. User authentication
    3. Comment system
    4. Search functionality
    
    Some tasks can be done in parallel (like creating different endpoints),
    while others need sequential execution (like authentication before testing).
    Choose the best approach for efficiency.
    """
    
    print("📋 Request:", request)
    print("\n🤖 Using adaptive pattern - orchestrator will choose best approach")
    print("\n🚀 Starting adaptive workflow...\n")
    
    # Let orchestrator adapt
    result = orchestrator.orchestrate(request)
    
    # Print results
    if result["success"]:
        print("\n✅ Adaptive workflow completed successfully!")
        print("\n📊 Execution Report:")
        print(result["report"])
        print(f"\n🎯 Pattern chosen: Check report for details")
    else:
        print(f"\n❌ Adaptive workflow failed: {result['error']}")
    
    return orchestrator


def demo_interactive_orchestration():
    """Demo: Interactive orchestration with user input"""
    print_section("Interactive Orchestration")
    
    orchestrator = OrchestratorAgent(
        coordination_pattern="supervisor",
        use_all_tools=True,  # Enable all tools
        enable_monitoring=True
    )
    
    print("💬 Interactive Orchestrator ready!")
    print("You can ask for any development task and see the orchestration in action.")
    print("Type 'exit' to quit.\n")
    
    while True:
        request = input("\n🎭 Enter your request: ")
        
        if request.lower() == 'exit':
            break
        
        if not request.strip():
            print("Please enter a valid request.")
            continue
        
        print("\n🚀 Orchestrating your request...\n")
        
        # Use chat for interactive requests
        response = orchestrator.chat(request)
        
        print("\n📊 Response:")
        print(response)
        
        # Option to see detailed status
        show_status = input("\n📈 Show agent status? (y/n): ")
        if show_status.lower() == 'y':
            status = orchestrator.get_agent_status()
            print("\n📊 Agent Status:")
            for agent, info in status["agents"].items():
                print(f"  - {agent}: {info}")
    
    print("\n👋 Thank you for using the Orchestrator!")
    return orchestrator


def demo_direct_agent_access():
    """Demo: Direct access to individual agents through orchestrator"""
    print_section("Direct Agent Access")
    
    orchestrator = OrchestratorAgent()
    
    print("🎯 Demonstrating direct access to individual agents:\n")
    
    # Direct CoderAgent access
    print("1️⃣ Direct CoderAgent access:")
    code_result = orchestrator.coder("Generate a simple function to check if a number is even")
    print(f"CoderAgent result:\n{code_result[:200]}...\n")
    
    # Direct TesterAgent access
    print("2️⃣ Direct TesterAgent access:")
    test_result = orchestrator.tester("Generate tests for an 'is_even' function")
    print(f"TesterAgent result:\n{test_result[:200]}...\n")
    
    # Direct ExecutorAgent access
    print("3️⃣ Direct ExecutorAgent access:")
    exec_result = orchestrator.executor("Execute: print('Hello from ExecutorAgent!')")
    print(f"ExecutorAgent result:\n{exec_result[:200]}...\n")
    
    return orchestrator


def main():
    """Run all orchestrator demos"""
    print("\n" + "="*60)
    print("🎭 ORCHESTRATOR AGENT DEMO SUITE")
    print("="*60)
    print("\nThis demo showcases the OrchestratorAgent's ability to coordinate")
    print("multiple agents (Coder, Tester, Executor) in harmony.\n")
    
    # Check dependencies
    try:
        from agent.orchestrator import OrchestratorAgent
        from agent.coder import CoderAgent
        from agent.tester import TesterAgent
        from agent.executor import ExecutorAgent
        print("✅ All agent modules loaded successfully")
    except ImportError as e:
        print(f"❌ Error: Missing dependencies - {e}")
        print("\nPlease ensure all requirements are installed:")
        print("pip install -r requirements.txt")
        return
    
    demos = [
        ("Full Development Workflow", demo_full_development_workflow),
        ("Code Review Workflow", demo_code_review_workflow),
        ("Bug Fix Workflow", demo_bug_fix_workflow),
        ("Parallel Tasks (Swarm)", demo_parallel_tasks),
        ("Adaptive Pattern", demo_adaptive_pattern),
        ("Direct Agent Access", demo_direct_agent_access),
        ("Interactive Mode", demo_interactive_orchestration),
    ]
    
    print("Available demos:")
    for i, (name, _) in enumerate(demos):
        print(f"{i+1}. {name}")
    
    while True:
        choice = input("\nSelect a demo (1-7) or 'all' to run all, 'exit' to quit: ")
        
        if choice.lower() == 'exit':
            break
        
        if choice.lower() == 'all':
            for name, demo_func in demos[:-1]:  # Skip interactive in 'all'
                try:
                    demo_func()
                    input("\nPress Enter to continue to next demo...")
                except Exception as e:
                    print(f"\n❌ Error in {name}: {e}")
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(demos):
                    demos[idx][1]()
                else:
                    print("Invalid choice. Please select 1-7.")
            except ValueError:
                print("Invalid input. Please enter a number or 'all'.")
    
    print("\n🎭 Thank you for exploring the OrchestratorAgent!")
    print("The future of coordinated AI development is here! 🚀\n")


if __name__ == "__main__":
    main()