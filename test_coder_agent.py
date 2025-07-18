"""
Coder Agent Test - Testing agent with coding tools and structured outputs
"""

import os
import asyncio
import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

# Set Azure OpenAI environment variables
os.environ["OPENAI_API_VERSION"] = "2024-08-01-preview"  # Updated for structured outputs
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://oai-202-fbeta-dev.openai.azure.com/"
os.environ["OPENAI_API_KEY"] = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"

from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
from core_agent import CoreAgent, AgentConfig


# Structured output models
class CodeAnalysis(BaseModel):
    """Code analysis result structure"""
    language: str = Field(description="Programming language detected")
    complexity_score: int = Field(description="Code complexity score (1-10)")
    issues_found: List[str] = Field(description="List of issues or improvements")
    suggestions: List[str] = Field(description="Improvement suggestions")
    summary: str = Field(description="Overall analysis summary")


class CodeGenerationResult(BaseModel):
    """Code generation result structure"""
    requested_functionality: str = Field(description="What was requested")
    generated_code: str = Field(description="The generated code")
    language: str = Field(description="Programming language used")
    dependencies: List[str] = Field(description="Required dependencies/imports")
    usage_example: str = Field(description="Example of how to use the code")
    explanation: str = Field(description="How the code works")


# Define coding tools
@tool
def analyze_code(code: str) -> str:
    """
    Analyze code for complexity, issues, and suggestions.
    
    Args:
        code: The source code to analyze
    
    Returns:
        Analysis results as formatted string
    """
    # Simple analysis logic
    lines = code.split('\n')
    line_count = len([l for l in lines if l.strip()])
    
    issues = []
    if line_count > 50:
        issues.append("Function might be too long")
    if 'TODO' in code:
        issues.append("Contains TODO comments")
    if 'print(' in code and 'debug' in code.lower():
        issues.append("Contains debug print statements")
    
    language = "unknown"
    if 'def ' in code or 'import ' in code:
        language = "python"
    elif 'function ' in code or 'const ' in code:
        language = "javascript"
    elif 'public class' in code:
        language = "java"
    
    complexity = min(10, max(1, line_count // 10))
    
    return f"""
    Code Analysis Results:
    - Language: {language}
    - Lines of code: {line_count}
    - Complexity score: {complexity}/10
    - Issues found: {', '.join(issues) if issues else 'None'}
    """


@tool
def format_code(code: str, language: str = "python") -> str:
    """
    Format and clean up code.
    
    Args:
        code: The code to format
        language: Programming language (python, javascript, java, etc.)
    
    Returns:
        Formatted code
    """
    # Simple formatting
    lines = code.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        if line.strip():
            formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)


@tool
def generate_documentation(code: str) -> str:
    """
    Generate documentation for the provided code.
    
    Args:
        code: The code to document
    
    Returns:
        Generated documentation
    """
    lines = code.split('\n')
    functions = [l for l in lines if l.strip().startswith('def ')]
    classes = [l for l in lines if l.strip().startswith('class ')]
    
    doc = "# Code Documentation\n\n"
    
    if classes:
        doc += "## Classes\n"
        for cls in classes:
            class_name = cls.split('class ')[1].split('(')[0].split(':')[0].strip()
            doc += f"- **{class_name}**: [Description needed]\n"
    
    if functions:
        doc += "\n## Functions\n"
        for func in functions:
            func_name = func.split('def ')[1].split('(')[0].strip()
            doc += f"- **{func_name}()**: [Description needed]\n"
    
    return doc


@tool  
def run_code_tests(code: str) -> str:
    """
    Simulate running tests on the provided code.
    
    Args:
        code: The code to test
    
    Returns:
        Test results
    """
    # Simple test simulation
    issues = []
    
    if 'def ' not in code:
        issues.append("No functions found to test")
    
    if 'import ' not in code and 'from ' not in code:
        issues.append("No imports found - might be missing dependencies")
    
    if not issues:
        return "✅ All tests passed! Code looks good."
    else:
        return f"⚠️  Test issues found:\n" + '\n'.join(f"- {issue}" for issue in issues)


# Create the coder agent
def create_coder_agent():
    """Create a specialized coder agent with tools and structured outputs"""
    
    # Initialize the LLM
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2024-08-01-preview",
        temperature=0.1,
        max_tokens=2000
    )
    
    # Define tools for coding tasks
    tools = [analyze_code, format_code, generate_documentation, run_code_tests]
    
    # Create agent config
    config = AgentConfig(
        name="Expert Coder Agent",
        model=llm,  # Set the LLM model in config
        description="Professional software developer specializing in code analysis, generation, and optimization",
        tools=tools,
        enable_memory=False,  # Keep it simple for this test
        enable_human_feedback=False,
        response_format=None,  # We'll test structured outputs separately
        system_prompt="""You are an expert software developer and code analyst. You have access to powerful tools for:

🔧 CODING TOOLS:
- analyze_code: Analyze code for complexity, issues, and improvement suggestions
- format_code: Clean and format code according to best practices  
- generate_documentation: Create comprehensive documentation for code
- run_code_tests: Simulate testing and validation of code

💡 YOUR EXPERTISE:
- Write clean, efficient, and maintainable code
- Provide detailed code analysis and optimization suggestions
- Follow best practices and design patterns
- Generate comprehensive documentation
- Identify potential issues and security vulnerabilities

📋 RESPONSE STYLE:
- Always use your tools to analyze code before providing feedback
- Provide specific, actionable suggestions
- Explain your reasoning clearly
- Include code examples when helpful
- Structure your responses professionally

When given code, ALWAYS:
1. First analyze it using the analyze_code tool
2. Check for formatting issues and suggest improvements
3. Generate documentation if requested
4. Test the code logic if applicable

Be thorough, professional, and helpful in all your responses!"""
    )
    
    # Create the agent
    agent = CoreAgent(config=config)
    return agent


async def test_basic_functionality():
    """Test basic coder agent functionality"""
    print("🚀 Creating Coder Agent...")
    agent = create_coder_agent()
    
    print("✅ Coder Agent created successfully!")
    print(f"Agent Name: {agent.config.name}")
    print(f"Available Tools: {[tool.name for tool in agent.config.tools]}")
    
    # Test code to analyze
    test_code = """
def calculate_fibonacci(n):
    # TODO: Add input validation
    if n <= 1:
        return n
    print(f"Debug: calculating fib({n})")  # debug print
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

class MathUtils:
    def __init__(self):
        pass
    
    def factorial(self, n):
        if n <= 1:
            return 1
        result = 1
        for i in range(2, n + 1):
            result *= i
        return result
"""
    
    print("\n📝 Testing code analysis...")
    print("Code to analyze:")
    print("=" * 50)
    print(test_code)
    print("=" * 50)
    
    # Test the agent  
    response = await agent.ainvoke(
        f"Please analyze this Python code thoroughly:\n\n{test_code}\n\nUse all your tools to provide a comprehensive analysis including code quality, formatting suggestions, documentation, and testing insights."
    )
    
    print("\n🤖 Agent Response:")
    print("=" * 80)
    print(response)
    print("=" * 80)
    
    return response


async def test_structured_output():
    """Test structured output with Pydantic models"""
    print("\n🏗️  Testing Structured Output...")
    
    # Create agent with structured output
    llm = AzureChatOpenAI(
        azure_deployment="gpt4", 
        api_version="2024-08-01-preview",
        temperature=0.1
    )
    
    # Test with CodeAnalysis structure
    config = AgentConfig(
        name="Structured Coder Agent",
        model=llm,
        description="Coder agent that returns structured analysis",
        tools=[analyze_code],
        enable_memory=False,
        enable_human_feedback=False,
        response_format=CodeAnalysis,
        system_prompt="You are a code analyst. Analyze the given code and return a structured analysis using the provided format."
    )
    
    structured_agent = CoreAgent(config=config)
    
    simple_code = """
def hello_world():
    print("Hello, World!")
    return "success"
"""
    
    print("Testing structured code analysis...")
    structured_response = await structured_agent.ainvoke(
        f"Analyze this code and provide structured output:\n\n{simple_code}"
    )
    
    print("\n📊 Structured Response:")
    print("=" * 60)
    print(f"Type: {type(structured_response)}")
    if isinstance(structured_response, CodeAnalysis):
        print("✅ Successfully returned CodeAnalysis object!")
        print(f"Language: {structured_response.language}")
        print(f"Complexity: {structured_response.complexity_score}")
        print(f"Issues: {structured_response.issues_found}")
        print(f"Suggestions: {structured_response.suggestions}")
        print(f"Summary: {structured_response.summary}")
    else:
        print(f"📄 Raw response: {structured_response}")
    print("=" * 60)
    
    return structured_response


async def test_code_generation():
    """Test code generation with structured output"""
    print("\n⚡ Testing Code Generation...")
    
    llm = AzureChatOpenAI(
        azure_deployment="gpt4",
        api_version="2024-08-01-preview", 
        temperature=0.3
    )
    
    config = AgentConfig(
        name="Code Generator Agent",
        model=llm,
        description="Agent that generates code with structured output",
        tools=[format_code, generate_documentation],
        enable_memory=False,
        enable_human_feedback=False,
        response_format=CodeGenerationResult,
        system_prompt="You are a code generator. Generate clean, well-documented code based on user requests and return structured results."
    )
    
    generator_agent = CoreAgent(config=config)
    
    response = await generator_agent.ainvoke(
        "Generate a Python function that calculates the area of different geometric shapes (circle, rectangle, triangle). Include proper error handling and documentation."
    )
    
    print("\n🔧 Code Generation Result:")
    print("=" * 70)
    if isinstance(response, CodeGenerationResult):
        print("✅ Successfully returned CodeGenerationResult object!")
        print(f"\n📋 Requested: {response.requested_functionality}")
        print(f"\n🐍 Language: {response.language}")
        print(f"\n📦 Dependencies: {response.dependencies}")
        print(f"\n💻 Generated Code:\n{response.generated_code}")
        print(f"\n📚 Usage Example:\n{response.usage_example}")
        print(f"\n💡 Explanation: {response.explanation}")
    else:
        print(f"📄 Raw response: {response}")
    print("=" * 70)
    
    return response


async def main():
    """Run all tests"""
    print("🧪 CODER AGENT COMPREHENSIVE TEST")
    print("=" * 50)
    
    try:
        # Test 1: Basic functionality
        basic_result = await test_basic_functionality()
        
        # Test 2: Structured output
        structured_result = await test_structured_output()
        
        # Test 3: Code generation  
        generation_result = await test_code_generation()
        
        print("\n✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("\n📊 SUMMARY:")
        print(f"✓ Basic functionality: {'✅ PASS' if basic_result else '❌ FAIL'}")
        print(f"✓ Structured output: {'✅ PASS' if structured_result else '❌ FAIL'}")
        print(f"✓ Code generation: {'✅ PASS' if generation_result else '❌ FAIL'}")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())