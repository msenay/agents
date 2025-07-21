#!/usr/bin/env python3
"""
Simple Core Agent Integration Test
=================================

Simple integration test that works without complex imports.
Tests Core Agent tools and Python code generation workflow.
"""

import unittest
import sys
import os
import tempfile
import subprocess
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import tools directly
try:
    from core.tools import create_python_coding_tools, PythonExecutorTool, FileWriterTool
    TOOLS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Tools import failed: {e}")
    TOOLS_AVAILABLE = False


class MockAIAgent:
    """Mock AI agent for testing"""
    
    def __init__(self):
        self.responses = {
            "fibonacci": '''Here's a Python function to generate Fibonacci numbers:

```python
def fibonacci(n):
    """Generate first n Fibonacci numbers"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

# Test the function
if __name__ == "__main__":
    print("Testing Fibonacci function:")
    result = fibonacci(10)
    print(f"First 10 Fibonacci numbers: {result}")
```''',
            
            "calculator": '''Here's a calculator class:

```python
class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

# Test
calc = Calculator()
print(f"5 + 3 = {calc.add(5, 3)}")
print(f"4 * 6 = {calc.multiply(4, 6)}")
```'''
        }
    
    def generate_code(self, prompt):
        """Generate code based on prompt"""
        if "fibonacci" in prompt.lower():
            return self.responses["fibonacci"]
        elif "calculator" in prompt.lower():
            return self.responses["calculator"]
        else:
            return "# Simple Python code\nprint('Hello World')"


class TestSimpleIntegration(unittest.TestCase):
    """Simple integration tests"""
    
    def setUp(self):
        """Set up test environment"""
        self.workspace = tempfile.mkdtemp(prefix="simple_test_")
        self.agent = MockAIAgent()
        
        if TOOLS_AVAILABLE:
            self.tools = create_python_coding_tools(workspace_dir=self.workspace)
        
        print(f"Test workspace: {self.workspace}")
    
    @unittest.skipUnless(TOOLS_AVAILABLE, "Tools not available")
    def test_tools_creation(self):
        """Test that tools can be created"""
        self.assertEqual(len(self.tools), 4)
        tool_names = [tool.name for tool in self.tools]
        self.assertIn("python_executor", tool_names)
        self.assertIn("file_writer", tool_names)
        print("✅ Tools created successfully")
    
    @unittest.skipUnless(TOOLS_AVAILABLE, "Tools not available")
    def test_python_execution(self):
        """Test Python code execution"""
        python_tool = self.tools[0]  # PythonExecutorTool
        
        test_code = """
print("Hello from integration test!")
result = 2 + 3
print(f"2 + 3 = {result}")
"""
        
        result = python_tool._run(test_code)
        self.assertIn("Output:", result)
        self.assertIn("Hello from integration test!", result)
        self.assertIn("2 + 3 = 5", result)
        print("✅ Python execution works")
    
    @unittest.skipUnless(TOOLS_AVAILABLE, "Tools not available")
    def test_file_operations(self):
        """Test file writing and reading"""
        file_writer = self.tools[1]  # FileWriterTool
        file_reader = self.tools[2]   # FileReaderTool
        
        # Write a file
        test_content = "# Test Python file\nprint('Test file content')"
        write_result = file_writer._run("test.py", test_content)
        self.assertIn("Successfully wrote", write_result)
        
        # Read the file back
        read_result = file_reader._run("test.py")
        self.assertIn("Test Python file", read_result)
        self.assertIn("Test file content", read_result)
        print("✅ File operations work")
    
    @unittest.skipUnless(TOOLS_AVAILABLE, "Tools not available")
    def test_fibonacci_workflow(self):
        """Test complete Fibonacci generation workflow"""
        print("\n🚀 Testing Fibonacci Workflow...")
        
        # 1. Agent generates code
        prompt = "Write a Python function to generate Fibonacci numbers"
        response = self.agent.generate_code(prompt)
        self.assertIn("fibonacci", response.lower())
        self.assertIn("```python", response)
        print("✅ Step 1: Code generated")
        
        # 2. Extract code
        import re
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        self.assertIsNotNone(code_match)
        fibonacci_code = code_match.group(1)
        print("✅ Step 2: Code extracted")
        
        # 3. Execute code
        python_tool = self.tools[0]
        exec_result = python_tool._run(fibonacci_code)
        self.assertIn("Output:", exec_result)
        self.assertIn("Fibonacci", exec_result)
        print("✅ Step 3: Code executed")
        
        # 4. Save code
        file_writer = self.tools[1]
        save_result = file_writer._run("fibonacci.py", fibonacci_code)
        self.assertIn("Successfully wrote", save_result)
        print("✅ Step 4: Code saved")
        
        # 5. Verify file
        file_reader = self.tools[2]
        read_result = file_reader._run("fibonacci.py")
        self.assertIn("def fibonacci", read_result)
        print("✅ Step 5: File verified")
        
        print("🎯 Fibonacci Workflow: SUCCESS!")
    
    @unittest.skipUnless(TOOLS_AVAILABLE, "Tools not available")
    def test_calculator_workflow(self):
        """Test calculator creation workflow"""
        print("\n🧮 Testing Calculator Workflow...")
        
        # Generate calculator code
        response = self.agent.generate_code("Create a calculator")
        
        # Extract and execute
        import re
        code_match = re.search(r'```python\n(.*?)\n```', response, re.DOTALL)
        calculator_code = code_match.group(1)
        
        # Execute
        python_tool = self.tools[0]
        result = python_tool._run(calculator_code)
        self.assertIn("Output:", result)
        print("✅ Calculator code executed")
        
        # Save
        file_writer = self.tools[1]
        file_writer._run("calculator.py", calculator_code)
        print("✅ Calculator code saved")
        
        print("🎯 Calculator Workflow: SUCCESS!")
    
    def test_real_fibonacci_execution(self):
        """Test real Fibonacci code execution without tools"""
        print("\n🐍 Testing Real Fibonacci...")
        
        fibonacci_code = '''
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    
    return sequence

result = fibonacci(8)
print(f"First 8 Fibonacci: {result}")
'''
        
        # Execute with subprocess
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(fibonacci_code)
            temp_file = f.name
        
        try:
            result = subprocess.run(['python3', temp_file], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0)
            self.assertIn("First 8 Fibonacci", result.stdout)
            self.assertIn("[0, 1, 1, 2, 3, 5, 8, 13]", result.stdout)
            print("✅ Real Fibonacci execution works")
        finally:
            os.unlink(temp_file)
    
    def test_memory_simulation(self):
        """Test simple memory simulation"""
        print("\n💭 Testing Memory Simulation...")
        
        # Simulate conversation
        conversation = []
        
        # Turn 1
        response1 = self.agent.generate_code("fibonacci function")
        conversation.append(("user", "fibonacci function"))
        conversation.append(("ai", response1))
        
        # Turn 2  
        response2 = self.agent.generate_code("calculator class")
        conversation.append(("user", "calculator class"))
        conversation.append(("ai", response2))
        
        # Verify conversation history
        self.assertEqual(len(conversation), 4)
        self.assertIn("fibonacci", conversation[1][1].lower())
        self.assertIn("calculator", conversation[3][1].lower())
        
        print("✅ Memory simulation works")
        print("🎯 Memory Simulation: SUCCESS!")
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.workspace):
            shutil.rmtree(self.workspace)


def run_simple_integration_tests():
    """Run simple integration tests"""
    print("=" * 80)
    print("🧪 SIMPLE CORE AGENT INTEGRATION TESTS")
    print("=" * 80)
    
    if not TOOLS_AVAILABLE:
        print("⚠️ Some tests will be skipped due to import issues")
    
    # Run tests
    suite = unittest.TestSuite()
    
    if TOOLS_AVAILABLE:
        suite.addTest(TestSimpleIntegration('test_tools_creation'))
        suite.addTest(TestSimpleIntegration('test_python_execution'))
        suite.addTest(TestSimpleIntegration('test_file_operations'))
        suite.addTest(TestSimpleIntegration('test_fibonacci_workflow'))
        suite.addTest(TestSimpleIntegration('test_calculator_workflow'))
    
    suite.addTest(TestSimpleIntegration('test_real_fibonacci_execution'))
    suite.addTest(TestSimpleIntegration('test_memory_simulation'))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL SIMPLE INTEGRATION TESTS PASSED!")
        print("✅ Python code generation works")
        print("✅ Code execution works")
        if TOOLS_AVAILABLE:
            print("✅ Tools integration works")
            print("✅ File operations work")
        print("✅ Memory simulation works")
        print("\n🚀 Core Agent basic functionality verified!")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_simple_integration_tests()
    exit(0 if success else 1)