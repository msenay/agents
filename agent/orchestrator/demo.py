"""
Orchestrator Agent Demo

Demonstrates the OrchestratorAgent's ability to coordinate CoderAgent, TesterAgent, 
and ExecutorAgent in harmony for complete development workflows.
"""

import os
import sys
from typing import Dict, Any

# Try to import the OrchestratorAgent

from orchestrator import OrchestratorAgent



def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🎭 {title}")
    print(f"{'='*60}\n")


def check_environment():
    """Check environment and dependencies"""
    print("🔍 Checking environment...")
    
    # Check Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"✅ Python {python_version}")
    
    # Check API keys
    api_key_set = bool(os.getenv("OPENAI_API_KEY"))
    if api_key_set:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY not set (required for actual execution)")



def create_orchestrator(pattern="supervisor"):
    """Create an orchestrator instance if possible"""
    return OrchestratorAgent(coordination_pattern=pattern, enable_monitoring=True)


def demo_full_development_workflow():
    """Demo: Complete development workflow from idea to tested code"""
    print_section("Full Development Workflow")
    
    request = """
    Create a Python class for managing a todo list with the following features:
    1. Add tasks with priority levels (high, medium, low)
    2. Mark tasks as complete
    3. Get tasks by priority
    4. Save/load tasks to/from JSON file
    
    Ensure the code is well-tested and validated.
    """
    
    print("📋 Request:", request.strip())
    print("\n🎯 Pattern: Supervisor (Sequential with quality control)")
    
    orchestrator = create_orchestrator("supervisor")
    
    if orchestrator:
        try:
            print("\n🚀 Starting orchestrated workflow...\n")
            result = orchestrator.orchestrate(request, workflow_type="full_development")
            
            if result["success"]:
                print("\n✅ Workflow completed successfully!")
                print("\n📊 Report:")
                print(result["report"])
            else:
                print(f"\n❌ Workflow failed: {result['error']}")
            return
        except Exception as e:
            print(f"\n⚠️  Execution error: {e}")
            print("Showing demonstration instead...\n")
    
    # Demonstration when can't execute
    print("\n📊 Workflow Steps (What would happen):")
    print("1. 📝 Plan workflow based on requirements")
    print("2. 🚀 CoderAgent generates TodoList class:")
    print("   - __init__ method with task storage")
    print("   - add_task(task, priority) method")
    print("   - mark_complete(task_id) method")
    print("   - get_by_priority(priority) method")
    print("   - save_to_json() and load_from_json() methods")
    print("3. ✅ Quality check on generated code")
    print("4. 🧪 TesterAgent creates comprehensive tests:")
    print("   - Test task creation and storage")
    print("   - Test priority filtering")
    print("   - Test file save/load functionality")
    print("   - Test edge cases and error handling")
    print("5. ✅ Quality check on test coverage (target: 95%)")
    print("6. ⚙️  ExecutorAgent runs all tests")
    print("7. 📊 Generate final report with code and test results")


def demo_code_review_workflow():
    """Demo: Code review and improvement workflow"""
    print_section("Code Review Workflow")
    
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
    
    print("📝 Code to review:")
    print("```python")
    print(existing_code.strip())
    print("```")
    
    print("\n🎯 Pattern: Supervisor (Sequential analysis)")
    
    orchestrator = create_orchestrator("supervisor")
    
    if orchestrator:
        try:
            request = f"Review and improve this code:\n```python\n{existing_code}\n```"
            print("\n🚀 Starting code review workflow...\n")
            result = orchestrator.orchestrate(request, workflow_type="code_review")
            
            if result["success"]:
                print("\n✅ Review completed!")
                print(result["report"])
            return
        except Exception as e:
            print(f"\n⚠️  Execution error: {e}")
            print("Showing demonstration instead...\n")
    
    # Demonstration
    print("\n📊 Review Process (What would happen):")
    print("1. 🔍 CoderAgent analyzes the code:")
    print("   - Fibonacci: Inefficient recursive implementation")
    print("   - Prime finder: O(n²) complexity, can be optimized")
    print("2. 🧪 TesterAgent checks test coverage:")
    print("   - No existing tests found")
    print("   - Recommends unit tests for both functions")
    print("3. ⚙️  ExecutorAgent profiles performance:")
    print("   - Fibonacci(30) takes ~0.3s (exponential time)")
    print("   - find_prime_numbers(1000) takes ~0.1s")
    print("4. 💡 Improvements suggested:")
    print("   - Use memoization or iterative approach for Fibonacci")
    print("   - Implement Sieve of Eratosthenes for primes")
    print("   - Add comprehensive test suite")
    print("   - Add type hints and docstrings")


def demo_bug_fix_workflow():
    """Demo: Bug fix workflow with testing"""
    print_section("Bug Fix Workflow")
    
    buggy_code = '''
def divide_numbers(a, b):
    return a / b

def calculate_average(numbers):
    total = sum(numbers)
    return divide_numbers(total, len(numbers))
'''
    
    print("🐛 Buggy code:")
    print("```python")
    print(buggy_code.strip())
    print("```")
    
    print("\n⚠️  Known issues:")
    print("- Division by zero when b=0 or empty list")
    print("- No handling for non-numeric values")
    
    print("\n🎯 Pattern: Pipeline (Strict sequential fix)")
    
    orchestrator = create_orchestrator("pipeline")
    
    if orchestrator:
        try:
            request = f"Fix the bugs in this code:\n```python\n{buggy_code}\n```"
            print("\n🚀 Starting bug fix workflow...\n")
            result = orchestrator.orchestrate(request, workflow_type="bug_fix")
            
            if result["success"]:
                print("\n✅ Bugs fixed!")
                print(result["report"])
            return
        except Exception as e:
            print(f"\n⚠️  Execution error: {e}")
            print("Showing demonstration instead...\n")
    
    # Demonstration
    print("\n📊 Bug Fix Process (What would happen):")
    print("1. ⚙️  ExecutorAgent reproduces the bugs:")
    print("   - ZeroDivisionError with divide_numbers(1, 0)")
    print("   - ZeroDivisionError with calculate_average([])")
    print("2. 🚀 CoderAgent fixes the issues:")
    print("   - Add zero check in divide_numbers")
    print("   - Add empty list check in calculate_average")
    print("   - Add type validation")
    print("3. 🧪 TesterAgent creates regression tests:")
    print("   - Test normal cases")
    print("   - Test edge cases (zero, empty, None)")
    print("   - Test error messages")
    print("4. ⚙️  ExecutorAgent validates all tests pass")


def demo_parallel_tasks():
    """Demo: Parallel task execution using swarm pattern"""
    print_section("Parallel Task Execution (Swarm Pattern)")
    
    request = """
    Create three independent utility modules in parallel:
    
    1. String utilities: capitalize_words, reverse_string, count_vowels
    2. Math utilities: factorial, is_prime, gcd
    3. Date utilities: days_between, is_weekend, next_business_day
    
    Each module should have comprehensive tests.
    """
    
    print("📋 Request:", request.strip())
    print("\n🎯 Pattern: Swarm (Parallel execution)")
    
    orchestrator = create_orchestrator("swarm")
    
    if orchestrator:
        try:
            print("\n🚀 Starting parallel workflow...\n")
            result = orchestrator.orchestrate(request)
            
            if result["success"]:
                print("\n✅ Parallel tasks completed!")
                print(result["report"])
            return
        except Exception as e:
            print(f"\n⚠️  Execution error: {e}")
            print("Showing demonstration instead...\n")
    
    # Demonstration
    print("\n📊 Parallel Execution (What would happen):")
    print("🔄 All three modules developed simultaneously:")
    print("\nThread 1 - String Utilities:")
    print("  → CoderAgent creates string_utils.py")
    print("  → TesterAgent creates test_string_utils.py")
    print("  → ExecutorAgent validates (3/3 tests pass)")
    print("\nThread 2 - Math Utilities:")
    print("  → CoderAgent creates math_utils.py")
    print("  → TesterAgent creates test_math_utils.py")
    print("  → ExecutorAgent validates (3/3 tests pass)")
    print("\nThread 3 - Date Utilities:")
    print("  → CoderAgent creates date_utils.py")
    print("  → TesterAgent creates test_date_utils.py")
    print("  → ExecutorAgent validates (3/3 tests pass)")
    print("\n⚡ Total time: ~2 minutes (vs ~6 minutes sequential)")


def demo_adaptive_pattern():
    """Demo: Adaptive pattern selection based on task"""
    print_section("Adaptive Pattern Selection")
    
    request = """
    Build a REST API for a blog with:
    1. CRUD operations for posts
    2. User authentication
    3. Comment system
    4. Search functionality
    
    Some tasks can be parallel, others need sequential execution.
    """
    
    print("📋 Request:", request.strip())
    print("\n🎯 Pattern: Adaptive (Auto-selection)")
    
    orchestrator = create_orchestrator("adaptive")
    
    if orchestrator:
        try:
            print("\n🚀 Starting adaptive workflow...\n")
            result = orchestrator.orchestrate(request)
            
            if result["success"]:
                print("\n✅ Adaptive workflow completed!")
                print(result["report"])
            return
        except Exception as e:
            print(f"\n⚠️  Execution error: {e}")
            print("Showing demonstration instead...\n")
    
    # Demonstration
    print("\n📊 Adaptive Pattern Analysis (What would happen):")
    print("1. 🤔 Analyzing task complexity...")
    print("2. 📊 Pattern selection:")
    print("   - Authentication: Pipeline (security critical)")
    print("   - CRUD operations: Swarm (independent endpoints)")
    print("   - Integration: Supervisor (quality checks)")
    print("\n🔄 Execution plan:")
    print("Phase 1 (Pipeline): Authentication system")
    print("Phase 2 (Swarm): Parallel development of:")
    print("  - POST /posts endpoints")
    print("  - Comment system endpoints")
    print("  - Search functionality")
    print("Phase 3 (Supervisor): Integration and testing")


def demo_direct_agent_access():
    """Demo: Direct access to individual agents through orchestrator"""
    print_section("Direct Agent Access")
    
    print("🎯 Sometimes you need direct access to specific agents:")
    
    orchestrator = create_orchestrator()
    
    if orchestrator:
        try:
            print("\n1️⃣ Direct CoderAgent access:")
            code = orchestrator.coder("Generate a simple is_even function")
            print(f"Generated: {code[:100]}...")
            
            print("\n2️⃣ Direct TesterAgent access:")
            tests = orchestrator.tester("Create one test for is_even function")
            print(f"Generated: {tests[:100]}...")
            
            print("\n3️⃣ Direct ExecutorAgent access:")
            result = orchestrator.executor("print('Hello from ExecutorAgent!')")
            print(f"Executed: {result[:100]}...")
            return
        except Exception as e:
            print(f"\n⚠️  Execution error: {e}")
            print("Showing demonstration instead...\n")
    
    # Demonstration
    print("\n📊 Direct Access Examples (What would happen):")
    print("\n1️⃣ orchestrator.coder('Generate is_even function'):")
    print("```python")
    print("def is_even(number: int) -> bool:")
    print('    """Check if a number is even."""')
    print("    return number % 2 == 0")
    print("```")
    
    print("\n2️⃣ orchestrator.tester('Create test for is_even'):")
    print("```python")
    print("def test_is_even():")
    print("    assert is_even(2) == True")
    print("    assert is_even(3) == False")
    print("    assert is_even(0) == True")
    print("```")
    
    print("\n3️⃣ orchestrator.executor('Run the test'):")
    print("```")
    print("Running tests...")
    print("✅ test_is_even PASSED")
    print("1 passed in 0.01s")
    print("```")


def demo_interactive_orchestration():
    """Demo: Interactive orchestration with user input"""
    print_section("Interactive Orchestration")
    orchestrator = create_orchestrator()
    if not orchestrator:
        return
    
    print("💬 Interactive Orchestrator ready!")
    print("You can ask for any development task.")
    print("Type 'exit' to quit.\n")
    
    while True:
        try:
            request = input("\n🎭 Enter your request: ")
            
            if request.lower() == 'exit':
                break
            
            if not request.strip():
                print("Please enter a valid request.")
                continue
            
            print("\n🚀 Orchestrating your request...\n")
            try:
                response = orchestrator.chat(request)
                print("\n📊 Response:")
                print(response)
            except AttributeError as e:
                if "'chat'" in str(e):
                    # Fallback to invoke if chat is not available
                    response = orchestrator.invoke(request)
                    print("\n📊 Response:")
                    print(response)
                else:
                    raise
            
        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again.")
    
    print("\n👋 Thank you for using the Interactive Orchestrator!")


def show_patterns_overview():
    """Show overview of all coordination patterns"""
    print_section("Coordination Patterns Overview")
    
    patterns = {
        "Supervisor": {
            "desc": "Sequential execution with quality control at each step",
            "flow": "Plan → Execute → Check → Next Step",
            "best_for": "Complex workflows, Production code, Quality critical"
        },
        "Swarm": {
            "desc": "Parallel execution for independent tasks",
            "flow": "Plan → [Task1 || Task2 || Task3] → Aggregate",
            "best_for": "Independent modules, Time-critical tasks, Microservices"
        },
        "Pipeline": {
            "desc": "Strict sequential processing without interruption",
            "flow": "Step1 → Step2 → Step3 → Done",
            "best_for": "Simple workflows, Predictable tasks, Minimal overhead"
        },
        "Adaptive": {
            "desc": "Dynamic pattern selection based on task analysis",
            "flow": "Analyze → Choose Pattern → Execute",
            "best_for": "Mixed complexity, Flexible requirements, Unknown tasks"
        }
    }
    
    for name, info in patterns.items():
        print(f"\n🎯 {name} Pattern")
        print(f"   {info['desc']}")
        print(f"   Flow: {info['flow']}")
        print(f"   Best for: {info['best_for']}")


def main():
    """Run all orchestrator demos"""
    print("\n" + "="*60)
    print("🎭 ORCHESTRATOR AGENT DEMO SUITE")
    print("="*60)
    print("\nThe OrchestratorAgent coordinates multiple specialized agents to handle")
    print("complex software development tasks efficiently. It manages:")
    print("  • CoderAgent - Generates and analyzes code")
    print("  • TesterAgent - Creates comprehensive test suites")
    print("  • ExecutorAgent - Runs and validates code execution")
    print("\nUsing intelligent coordination patterns for optimal workflow management.\n")
    
    # Check environment
    ready = check_environment()
    
    if not ready:
        print("\n⚠️  Demo will run in demonstration mode.")
        print("   To see actual execution:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Set API key: export OPENAI_API_KEY='your-key'")
    
    # Show patterns overview
    show_patterns_overview()
    
    demos = [
        ("Full Development Workflow", demo_full_development_workflow),
        ("Code Review Workflow", demo_code_review_workflow),
        ("Bug Fix Workflow", demo_bug_fix_workflow),
        ("Parallel Tasks (Swarm)", demo_parallel_tasks),
        ("Adaptive Pattern", demo_adaptive_pattern),
        ("Direct Agent Access", demo_direct_agent_access),
        ("Interactive Mode", demo_interactive_orchestration),
    ]
    
    print("\n" + "="*60)
    print("📋 Available Demos:")
    print("="*60)
    
    demo_descriptions = [
        ("Full Development Workflow", 
         "Complete software development cycle: Code → Test → Execute\n"
         "   Pattern: Supervisor (sequential with quality control)\n"
         "   Best for: Building new features from scratch"),
        
        ("Code Review Workflow",
         "Analyze and improve existing code quality\n"
         "   Pattern: Pipeline (efficient sequential processing)\n"
         "   Best for: Refactoring, optimization, code improvements"),
        
        ("Bug Fix Workflow",
         "Identify and fix reported bugs systematically\n"
         "   Pattern: Pipeline (quick sequential fixes)\n"
         "   Best for: Debugging, error resolution, hotfixes"),
        
        ("Parallel Tasks (Swarm)",
         "Execute multiple independent tasks simultaneously\n"
         "   Pattern: Swarm (parallel execution)\n"
         "   Best for: Microservices, independent modules, speed-critical tasks"),
        
        ("Adaptive Pattern",
         "Automatically selects the best coordination pattern\n"
         "   Pattern: Dynamic (analyzes and chooses)\n"
         "   Best for: Mixed complexity tasks, flexible requirements"),
        
        ("Direct Agent Access",
         "Use individual agents without orchestration\n"
         "   Pattern: None (direct agent usage)\n"
         "   Best for: Single-purpose tasks, quick operations"),
        
        ("Interactive Mode",
         "Step-by-step execution with user guidance\n"
         "   Pattern: User-driven coordination\n"
         "   Best for: Exploratory development, learning, custom workflows")
    ]
    
    for i, (name, desc) in enumerate(demo_descriptions):
        print(f"\n{i+1}. {name}")
        print(f"   {desc}")
    
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