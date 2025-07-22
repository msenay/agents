"""
Execution Tools for ExecutorAgent
=================================

Specialized tools for safely executing code and running tests.
"""

import subprocess
import tempfile
import os
import sys
import re
import time
import traceback
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from agent.executor.prompts import (
    CREATE_DEMO_PROMPT,
    ANALYZE_CODE_SAFETY_PROMPT,
    PREPARE_TEST_EXECUTION_PROMPT,
    FORMAT_EXECUTION_RESULTS_PROMPT,
    CREATE_TEST_REPORT_PROMPT,
    HANDLE_EXECUTION_ERROR_PROMPT,
    DANGEROUS_PATTERNS,
    SAFE_IMPORTS
)


# Tool Input Schemas
class CodeExecutionInput(BaseModel):
    """Input schema for code execution"""
    code: str = Field(description="Python code to execute")
    timeout: int = Field(default=30, description="Execution timeout in seconds")
    capture_output: bool = Field(default=True, description="Whether to capture output")


class DemoCreationInput(BaseModel):
    """Input schema for demo creation"""
    code: str = Field(description="Code to create demo for")
    spec: str = Field(description="Specification for the demo")


class TestExecutionInput(BaseModel):
    """Input schema for test execution"""
    test_code: str = Field(description="Test code to execute")
    framework: str = Field(default="pytest", description="Testing framework (pytest, unittest)")
    timeout: int = Field(default=60, description="Execution timeout in seconds")


class CodeSafetyInput(BaseModel):
    """Input schema for code safety analysis"""
    code: str = Field(description="Code to analyze for safety")
    strict_mode: bool = Field(default=True, description="Use strict safety checks")


# Utility Functions
def check_code_safety(code: str, strict_mode: bool = True) -> Tuple[bool, str, str]:
    """
    Check if code is safe to execute
    
    Returns:
        Tuple of (is_safe, risk_level, concerns)
    """
    concerns = []
    risk_level = "SAFE"
    
    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, code, re.IGNORECASE):
            concerns.append(f"Potentially dangerous pattern found: {pattern}")
            risk_level = "HIGH_RISK"
    
    # Check imports
    import_matches = re.findall(r"import\s+(\w+)", code)
    import_matches.extend(re.findall(r"from\s+(\w+)", code))
    
    for imp in import_matches:
        if imp not in SAFE_IMPORTS and strict_mode:
            concerns.append(f"Import '{imp}' not in safe imports whitelist")
            risk_level = "MEDIUM_RISK" if risk_level == "SAFE" else risk_level
    
    # Check for infinite loops (basic check)
    if re.search(r"while\s+True\s*:", code) and "break" not in code:
        concerns.append("Potential infinite loop detected")
        risk_level = "HIGH_RISK"
    
    is_safe = risk_level in ["SAFE", "LOW_RISK"] or (risk_level == "MEDIUM_RISK" and not strict_mode)
    
    return is_safe, risk_level, "\n".join(concerns) if concerns else "No security concerns found"


def execute_python_code(code: str, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute Python code in a subprocess with timeout
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(code)
        temp_file = f.name
    
    try:
        start_time = time.time()
        process = subprocess.Popen(
            [sys.executable, temp_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        execution_time = time.time() - start_time
        
        return {
            "success": process.returncode == 0,
            "exit_code": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "execution_time": execution_time,
            "error": None
        }
        
    except subprocess.TimeoutExpired:
        process.kill()
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": f"Execution timeout after {timeout} seconds",
            "execution_time": timeout,
            "error": "TimeoutError"
        }
    except Exception as e:
        return {
            "success": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": 0,
            "error": type(e).__name__
        }
    finally:
        try:
            os.unlink(temp_file)
        except:
            pass


# Tool Factory Functions
def create_code_executor_tool(model):
    """Create tool for executing Python code"""
    
    class CodeExecutorTool(BaseTool):
        name: str = "execute_code"
        description: str = "Safely execute Python code and return results"
        args_schema: type[BaseModel] = CodeExecutionInput
        
        def _run(self, code: str, timeout: int = 30, capture_output: bool = True) -> str:
            # First check safety
            is_safe, risk_level, concerns = check_code_safety(code)
            
            if not is_safe:
                return f"Code execution blocked for safety reasons:\nRisk Level: {risk_level}\nConcerns:\n{concerns}"
            
            # Execute the code
            result = execute_python_code(code, timeout)
            
            # Format the results
            if result["success"]:
                output = f"✅ Execution successful!\n"
                output += f"Exit code: {result['exit_code']}\n"
                output += f"Execution time: {result['execution_time']:.2f}s\n"
                if result["stdout"]:
                    output += f"\nOutput:\n{result['stdout']}"
                if result["stderr"]:
                    output += f"\nWarnings/Errors:\n{result['stderr']}"
            else:
                output = f"❌ Execution failed!\n"
                output += f"Exit code: {result['exit_code']}\n"
                if result["error"]:
                    output += f"Error type: {result['error']}\n"
                if result["stderr"]:
                    output += f"Error details:\n{result['stderr']}"
                if result["stdout"]:
                    output += f"\nPartial output:\n{result['stdout']}"
            
            return output
    
    return CodeExecutorTool()


def create_demo_creator_tool(model):
    """Create tool for generating and executing demos"""
    
    class DemoCreatorTool(BaseTool):
        name: str = "create_demo"
        description: str = "Create and execute a demo based on code and specifications"
        args_schema: type[BaseModel] = DemoCreationInput
        
        def _run(self, code: str, spec: str) -> str:
            # Generate demo using LLM
            prompt = CREATE_DEMO_PROMPT.format(code=code, spec=spec)
            response = model.invoke([HumanMessage(content=prompt)])
            demo_code = response.content
            
            # Extract code from response if wrapped in markdown
            code_match = re.search(r"```python\n(.*?)\n```", demo_code, re.DOTALL)
            if code_match:
                demo_code = code_match.group(1)
            
            # Execute the demo
            is_safe, risk_level, concerns = check_code_safety(demo_code)
            
            if not is_safe:
                return f"Demo execution blocked for safety:\n{concerns}\n\nGenerated demo:\n{demo_code}"
            
            result = execute_python_code(demo_code, timeout=30)
            
            output = "📝 Generated Demo:\n"
            output += f"```python\n{demo_code}\n```\n\n"
            output += "🚀 Execution Results:\n"
            
            if result["success"]:
                output += f"✅ Demo ran successfully!\n"
                if result["stdout"]:
                    output += f"\nOutput:\n{result['stdout']}"
            else:
                output += f"❌ Demo failed!\n"
                if result["stderr"]:
                    output += f"Error:\n{result['stderr']}"
            
            return output
    
    return DemoCreatorTool()


def create_test_runner_tool(model):
    """Create tool for running unit tests"""
    
    class TestRunnerTool(BaseTool):
        name: str = "run_tests"
        description: str = "Run unit tests and return detailed results"
        args_schema: type[BaseModel] = TestExecutionInput
        
        def _run(self, test_code: str, framework: str = "pytest", timeout: int = 60) -> str:
            with tempfile.NamedTemporaryFile(mode='w', suffix='_test.py', delete=False) as f:
                f.write(test_code)
                test_file = f.name
            
            try:
                # Determine test command based on framework
                if framework == "pytest":
                    cmd = [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short"]
                elif framework == "unittest":
                    cmd = [sys.executable, "-m", "unittest", test_file, "-v"]
                else:
                    return f"Unsupported test framework: {framework}"
                
                # Run tests
                start_time = time.time()
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                stdout, stderr = process.communicate(timeout=timeout)
                execution_time = time.time() - start_time
                
                # Parse results
                output = "🧪 Test Execution Results:\n"
                output += f"Framework: {framework}\n"
                output += f"Execution time: {execution_time:.2f}s\n\n"
                
                if process.returncode == 0:
                    output += "✅ All tests passed!\n\n"
                else:
                    output += "❌ Some tests failed!\n\n"
                
                output += "Test Output:\n"
                output += stdout if stdout else stderr
                
                return output
                
            except subprocess.TimeoutExpired:
                process.kill()
                return f"❌ Test execution timeout after {timeout} seconds"
            except Exception as e:
                return f"❌ Test execution error: {str(e)}"
            finally:
                try:
                    os.unlink(test_file)
                except:
                    pass
    
    return TestRunnerTool()


def create_code_validator_tool(model):
    """Create tool for validating code safety"""
    
    class CodeValidatorTool(BaseTool):
        name: str = "validate_code"
        description: str = "Validate code for safety and security issues"
        args_schema: type[BaseModel] = CodeSafetyInput
        
        def _run(self, code: str, strict_mode: bool = True) -> str:
            # Basic safety check
            is_safe, risk_level, concerns = check_code_safety(code, strict_mode)
            
            # Get detailed analysis from LLM
            prompt = ANALYZE_CODE_SAFETY_PROMPT.format(code=code)
            response = model.invoke([HumanMessage(content=prompt)])
            llm_analysis = response.content
            
            output = "🔍 Code Safety Analysis:\n"
            output += f"Risk Level: {risk_level}\n"
            output += f"Safe to execute: {'Yes' if is_safe else 'No'}\n\n"
            
            if concerns:
                output += "Security Concerns:\n"
                output += concerns + "\n\n"
            
            output += "Detailed Analysis:\n"
            output += llm_analysis
            
            return output
    
    return CodeValidatorTool()


def create_error_handler_tool(model):
    """Create tool for analyzing and handling execution errors"""
    
    class ErrorHandlerTool(BaseTool):
        name: str = "handle_error"
        description: str = "Analyze execution errors and provide solutions"
        args_schema: type[BaseModel] = CodeExecutionInput
        
        def _run(self, code: str, timeout: int = 30, capture_output: bool = True) -> str:
            # First try to execute the code
            result = execute_python_code(code, timeout)
            
            if result["success"]:
                return "✅ Code executed successfully without errors!"
            
            # If failed, analyze the error
            prompt = HANDLE_EXECUTION_ERROR_PROMPT.format(
                code=code,
                error=result.get("error", "Unknown error"),
                traceback=result.get("stderr", "No traceback available")
            )
            
            response = model.invoke([HumanMessage(content=prompt)])
            
            output = "❌ Execution Error Analysis:\n\n"
            output += response.content
            
            return output
    
    return ErrorHandlerTool()


# Dispatcher Functions
def get_executor_tools(model, tool_names: List[str] = None):
    """
    Get specific tools for ExecutorAgent or all tools if none specified
    
    Args:
        model: The LLM model to pass to tools
        tool_names: List of tool names to create. If None, returns all tools.
    
    Returns:
        List of tool instances
    """
    available_tools = {
        "execute_code": create_code_executor_tool,
        "create_demo": create_demo_creator_tool,
        "run_tests": create_test_runner_tool,
        "validate_code": create_code_validator_tool,
        "handle_error": create_error_handler_tool,
    }
    
    if tool_names is None:
        return [factory(model) for factory in available_tools.values()]
    
    tools = []
    for name in tool_names:
        if name in available_tools:
            tools.append(available_tools[name](model))
        else:
            print(f"⚠️ Warning: Tool '{name}' not found. Available tools: {list(available_tools.keys())}")
    
    return tools


def get_core_executor_tools(model):
    """Get only the core/essential executor tools"""
    core_tools = [
        "execute_code",
        "run_tests",
        "create_demo"
    ]
    return get_executor_tools(model, core_tools)


def get_all_tool_names():
    """Get list of all available tool names"""
    return [
        "execute_code",
        "create_demo",
        "run_tests",
        "validate_code",
        "handle_error"
    ]