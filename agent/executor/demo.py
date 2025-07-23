#!/usr/bin/env python3
"""
Executor Agent Demo
==================

Demonstrates ExecutorAgent's capabilities for safe code execution.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, '/workspace')

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file if it exists"""
    env_path = Path(__file__).parent.parent.parent / '.env'
    if env_path.exists():
        print("📋 Loading environment variables from .env file...")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"').strip("'")
                    os.environ[key] = value
        print("✅ Environment variables loaded!")
    else:
        print("⚠️  No .env file found. Using system environment variables.")

# Load environment variables
load_env()

from agent.executor.executor import ExecutorAgent


def demo_simple_code_execution():
    """Demo: Execute simple Python code"""
    print("\n" + "="*60)
    print("📝 DEMO 1: Simple Code Execution")
    print("="*60)
    
    code = """
# Simple calculation demo
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"Fibonacci({i}) = {calculate_fibonacci(i)}")

# Some basic operations
numbers = [1, 2, 3, 4, 5]
squared = [n**2 for n in numbers]
print(f"\\nOriginal: {numbers}")
print(f"Squared: {squared}")
print(f"Sum: {sum(squared)}")
"""
    
    executor = ExecutorAgent()
    result = executor.execute_code(code)
    
    if result["success"]:
        print("✅ Code executed successfully!")
        print("\n📄 Execution Output:")
        print(result["output"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_test_execution():
    """Demo: Run unit tests"""
    print("\n" + "="*60)
    print("🧪 DEMO 2: Unit Test Execution")
    print("="*60)
    
    test_code = """
import pytest

def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

# Unit tests
def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    assert add(0, 0) == 0

def test_multiply():
    assert multiply(3, 4) == 12
    assert multiply(-2, 3) == -6
    assert multiply(0, 5) == 0

def test_divide():
    assert divide(10, 2) == 5
    assert divide(7, 2) == 3.5
    
    with pytest.raises(ValueError):
        divide(5, 0)

if __name__ == "__main__":
    # Run tests if executed directly
    test_add()
    test_multiply()
    test_divide()
    print("All tests passed!")
"""
    
    executor = ExecutorAgent()
    result = executor.run_tests(test_code)
    
    if result["success"]:
        print(f"✅ Tests executed with {result['framework']}!")
        print("\n📊 Test Results:")
        print(result["results"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_create_and_run_demo():
    """Demo: Create and run demo from specifications"""
    print("\n" + "="*60)
    print("🚀 DEMO 3: Create and Run Demo from Spec")
    print("="*60)
    
    code = """
class TodoList:
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        self.tasks.append({"task": task, "completed": False})
        return len(self.tasks) - 1
    
    def complete_task(self, index):
        if 0 <= index < len(self.tasks):
            self.tasks[index]["completed"] = True
            return True
        return False
    
    def get_pending_tasks(self):
        return [t for t in self.tasks if not t["completed"]]
    
    def get_completed_tasks(self):
        return [t for t in self.tasks if t["completed"]]
"""
    
    spec = """
Create a demo that:
1. Creates a TodoList instance
2. Adds several tasks (at least 5)
3. Marks some tasks as completed
4. Shows pending and completed tasks
5. Demonstrates error handling for invalid indices
6. Includes clear output messages showing the state changes
"""
    
    executor = ExecutorAgent()
    result = executor.create_and_run_demo(code, spec)
    
    if result["success"]:
        print("✅ Demo created and executed!")
        print("\n📝 Demo Output:")
        print(result["demo_output"])
    else:
        print(f"❌ Error: {result['error']}")


def demo_code_validation():
    """Demo: Validate code safety"""
    print("\n" + "="*60)
    print("🔍 DEMO 4: Code Safety Validation")
    print("="*60)
    
    # Safe code example
    safe_code = """
def calculate_statistics(numbers):
    if not numbers:
        return {"mean": 0, "min": 0, "max": 0}
    
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }

# Test the function
data = [10, 20, 30, 40, 50]
stats = calculate_statistics(data)
print(f"Statistics: {stats}")
"""
    
    # Potentially unsafe code
    unsafe_code = """
import os
import subprocess

def run_system_command(command):
    # This could be dangerous!
    result = subprocess.run(command, shell=True, capture_output=True)
    return result.stdout.decode()

# Example usage
output = run_system_command("ls -la")
print(output)
"""
    
    executor = ExecutorAgent(use_all_tools=True)
    
    print("\n🟢 Validating safe code...")
    result1 = executor.validate_code(safe_code)
    if result1["success"]:
        print(result1["validation"][:500] + "..." if len(result1["validation"]) > 500 else result1["validation"])
    
    print("\n\n🔴 Validating potentially unsafe code...")
    result2 = executor.validate_code(unsafe_code)
    if result2["success"]:
        print(result2["validation"][:500] + "..." if len(result2["validation"]) > 500 else result2["validation"])


def demo_error_handling():
    """Demo: Execute code with errors"""
    print("\n" + "="*60)
    print("❗ DEMO 5: Error Handling")
    print("="*60)
    
    error_code = """
# This code has intentional errors
def process_data(data):
    # TypeError: unsupported operand
    result = data + "suffix"
    return result

# This will cause an error
numbers = [1, 2, 3]
processed = process_data(numbers)
print(processed)
"""
    
    executor = ExecutorAgent(use_all_tools=True)
    
    print("🚨 Executing code with errors...")
    result = executor.execute_code(error_code)
    
    if result["success"]:
        print("✅ Execution output:")
        print(result["output"])
    else:
        print("❌ Execution failed (as expected):")
        print(result["output"] if "output" in result else result.get("error", "Unknown error"))
        
        # Use error handler tool
        print("\n🔧 Analyzing error...")
        analysis = executor.stream(f"Analyze this execution error and suggest fixes:\n```python\n{error_code}\n```")
        
        full_response = ""
        for chunk in analysis:
            if isinstance(chunk, dict):
                if "agent" in chunk and "messages" in chunk["agent"]:
                    for msg in chunk["agent"]["messages"]:
                        if hasattr(msg, "content"):
                            full_response += msg.content
                elif "messages" in chunk:
                    for msg in chunk["messages"]:
                        if hasattr(msg, "content"):
                            full_response += msg.content
            else:
                full_response += str(chunk)
        
        print("\n📋 Error Analysis:")
        print(full_response[:800] + "..." if len(full_response) > 800 else full_response)


def demo_performance_test():
    """Demo: Execute code with performance monitoring"""
    print("\n" + "="*60)
    print("⚡ DEMO 6: Performance Monitoring")
    print("="*60)
    
    perf_code = """
import time

def slow_function(n):
    '''Simulate a slow function'''
    time.sleep(0.1)
    return n * 2

def fast_function(n):
    '''A fast function'''
    return n * 2

# Performance comparison
import timeit

slow_time = timeit.timeit('slow_function(5)', globals=globals(), number=5)
fast_time = timeit.timeit('fast_function(5)', globals=globals(), number=1000)

print(f"Slow function (5 calls): {slow_time:.4f} seconds")
print(f"Fast function (1000 calls): {fast_time:.6f} seconds")
print(f"Speed difference: {slow_time / (fast_time / 200):.1f}x")
"""
    
    executor = ExecutorAgent()
    result = executor.execute_code(perf_code, timeout=10)
    
    if result["success"]:
        print("✅ Performance test completed!")
        print("\n📊 Results:")
        print(result["output"])
    else:
        print(f"❌ Error: {result['error']}")


if __name__ == "__main__":
    print("🚀 EXECUTOR AGENT DEMO - Safe Code Execution")
    print("="*80)
    print("ExecutorAgent provides:")
    print("- Safe code execution with validation")
    print("- Unit test execution")
    print("- Demo generation from specifications")
    print("- Error analysis and debugging")
    print("- Performance monitoring")
    
    try:
        demo_simple_code_execution()
        demo_test_execution()
        demo_create_and_run_demo()
        demo_code_validation()
        demo_error_handling()
        demo_performance_test()
        
        print("\n" + "="*80)
        print("✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo error: {str(e)}")
        import traceback
        traceback.print_exc()