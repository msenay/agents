"""
Testing Tools for TesterAgent
=============================

Specialized tools for generating and analyzing unit tests.
"""

from typing import List, Optional
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from agent.tester.prompts import (
    GENERATE_UNIT_TESTS_PROMPT,
    GENERATE_TESTS_WITH_FRAMEWORK_PROMPT,
    FRAMEWORK_REQUIREMENTS,
    GENERATE_INTEGRATION_TESTS_PROMPT,
    GENERATE_FIXTURES_PROMPT,
    ANALYZE_COVERAGE_PROMPT,
    GENERATE_PARAMETERIZED_TESTS_PROMPT,
    GENERATE_PERFORMANCE_TESTS_PROMPT
)


# Tool Input Schemas
class CodeInput(BaseModel):
    """Input schema for code to test"""
    code: str = Field(description="Python code to generate tests for")


class FrameworkTestInput(BaseModel):
    """Input schema for framework-specific test generation"""
    code: str = Field(description="Python code to generate tests for")
    framework: str = Field(default="pytest", description="Testing framework to use (pytest, unittest, nose2)")


class CoverageAnalysisInput(BaseModel):
    """Input schema for test coverage analysis"""
    code: str = Field(description="Original code")
    tests: str = Field(description="Existing tests")


class TestTypeInput(BaseModel):
    """Input schema for specific test type generation"""
    code: str = Field(description="Python code to test")
    test_type: str = Field(description="Type of test: unit, integration, performance, parameterized")


# Tool Factory Functions
def create_unit_test_generator_tool(model):
    """Create tool for generating unit tests"""
    
    class UnitTestGeneratorTool(BaseTool):
        name: str = "generate_unit_tests"
        description: str = "Generate comprehensive unit tests for given Python code"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = GENERATE_UNIT_TESTS_PROMPT.format(code=code)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return UnitTestGeneratorTool()


def create_framework_test_generator_tool(model):
    """Create tool for generating tests with specific framework"""
    
    class FrameworkTestGeneratorTool(BaseTool):
        name: str = "generate_framework_tests"
        description: str = "Generate tests using a specific testing framework"
        args_schema: type[BaseModel] = FrameworkTestInput
        
        def _run(self, code: str, framework: str = "pytest") -> str:
            framework_reqs = FRAMEWORK_REQUIREMENTS.get(
                framework, 
                FRAMEWORK_REQUIREMENTS["pytest"]
            )
            
            prompt = GENERATE_TESTS_WITH_FRAMEWORK_PROMPT.format(
                code=code,
                framework=framework,
                framework_requirements=framework_reqs
            )
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return FrameworkTestGeneratorTool()


def create_test_fixture_generator_tool(model):
    """Create tool for generating test fixtures and mocks"""
    
    class TestFixtureGeneratorTool(BaseTool):
        name: str = "generate_test_fixtures"
        description: str = "Generate test fixtures, mocks, and test data for given code"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = GENERATE_FIXTURES_PROMPT.format(code=code)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return TestFixtureGeneratorTool()


def create_coverage_analyzer_tool(model):
    """Create tool for analyzing test coverage"""
    
    class CoverageAnalyzerTool(BaseTool):
        name: str = "analyze_test_coverage"
        description: str = "Analyze existing tests and identify coverage gaps"
        args_schema: type[BaseModel] = CoverageAnalysisInput
        
        def _run(self, code: str, tests: str) -> str:
            prompt = ANALYZE_COVERAGE_PROMPT.format(code=code, tests=tests)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return CoverageAnalyzerTool()


def create_parameterized_test_generator_tool(model):
    """Create tool for generating parameterized tests"""
    
    class ParameterizedTestGeneratorTool(BaseTool):
        name: str = "generate_parameterized_tests"
        description: str = "Generate data-driven parameterized tests"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = GENERATE_PARAMETERIZED_TESTS_PROMPT.format(code=code)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return ParameterizedTestGeneratorTool()


def create_integration_test_generator_tool(model):
    """Create tool for generating integration tests"""
    
    class IntegrationTestGeneratorTool(BaseTool):
        name: str = "generate_integration_tests"
        description: str = "Generate integration tests for component interactions"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = GENERATE_INTEGRATION_TESTS_PROMPT.format(code=code)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return IntegrationTestGeneratorTool()


def create_performance_test_generator_tool(model):
    """Create tool for generating performance tests"""
    
    class PerformanceTestGeneratorTool(BaseTool):
        name: str = "generate_performance_tests"
        description: str = "Generate performance and benchmark tests"
        args_schema: type[BaseModel] = CodeInput
        
        def _run(self, code: str) -> str:
            prompt = GENERATE_PERFORMANCE_TESTS_PROMPT.format(code=code)
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return PerformanceTestGeneratorTool()


# Dispatcher Functions
def get_tester_tools(model, tool_names: List[str] = None):
    """
    Get specific tools for TesterAgent or all tools if none specified
    
    Args:
        model: The LLM model to pass to tools
        tool_names: List of tool names to create. If None, returns all tools.
    
    Returns:
        List of tool instances
    """
    available_tools = {
        "generate_unit_tests": create_unit_test_generator_tool,
        "generate_framework_tests": create_framework_test_generator_tool,
        "generate_test_fixtures": create_test_fixture_generator_tool,
        "analyze_test_coverage": create_coverage_analyzer_tool,
        "generate_parameterized_tests": create_parameterized_test_generator_tool,
        "generate_integration_tests": create_integration_test_generator_tool,
        "generate_performance_tests": create_performance_test_generator_tool,
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


def get_core_tester_tools(model):
    """Get only the core/essential tester tools"""
    core_tools = [
        "generate_unit_tests",
        "generate_test_fixtures",
        "analyze_test_coverage"
    ]
    return get_tester_tools(model, core_tools)


def get_all_tool_names():
    """Get list of all available tool names"""
    return [
        "generate_unit_tests",
        "generate_framework_tests", 
        "generate_test_fixtures",
        "analyze_test_coverage",
        "generate_parameterized_tests",
        "generate_integration_tests",
        "generate_performance_tests"
    ]