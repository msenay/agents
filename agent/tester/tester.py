#!/usr/bin/env python3
"""
Tester Agent - Comprehensive Unit Test Generator
===============================================

TesterAgent specializes in generating high-quality unit tests for Python code.

Key Features:
- Generates comprehensive unit tests with high coverage
- Supports multiple testing frameworks (pytest, unittest, nose2)
- Creates test fixtures and mocks automatically
- Analyzes test coverage and identifies gaps
- Generates parameterized, integration, and performance tests

The agent uses Core Agent infrastructure for memory and tool management.
"""

import sys
from typing import Dict, Any
from datetime import datetime

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

# Core Agent Infrastructure
from ai_factory.agents.core import CoreAgent
from ai_factory.agents.core import AgentConfig
from ai_factory.agents.core import get_tester_llm

# Import prompts and tools
from agent.tester.prompts import SYSTEM_PROMPT
from agent.tester.tools import get_tester_tools, get_core_tester_tools


# TesterConfig removed - now using LLM Factory


class TesterAgent(CoreAgent):
    """
    Specialized agent for generating comprehensive unit tests
    
    This agent analyzes code and generates high-quality tests with:
    - High code coverage
    - Edge case handling
    - Proper mocking and fixtures
    - Multiple framework support
    """
    
    def __init__(self, session_id: str = None, use_all_tools: bool = False):
        """
        Initialize Tester Agent
        
        Args:
            session_id: Unique session identifier (auto-generated if not provided)
            use_all_tools: If True, loads all tools. If False, only core tools.
        """
        if session_id is None:
            session_id = f"tester_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create model using LLM Factory
        model = get_tester_llm()
        
        # Create tools - use core tools by default for efficiency
        if use_all_tools:
            tools = get_tester_tools(model)
        else:
            tools = get_core_tester_tools(model)
        
        # Create configuration
        config = AgentConfig(
            name="TesterAgent",
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            enable_memory=True,  # Remember test patterns and strategies
            memory_backend="inmemory",
            memory_types=["short_term", "long_term"],
            max_tokens=3000  # Default max tokens for tester
        )
        
        # Initialize parent CoreAgent
        super().__init__(config)
        
        print(f"✅ Tester Agent initialized - Ready to generate comprehensive tests!")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return SYSTEM_PROMPT
    
    def chat(self, message: str) -> str:
        """Chat interface - delegates to invoke"""
        return self.invoke(message)
    
    def generate_tests(self, code: str, framework: str = "pytest", test_type: str = "unit") -> Dict[str, Any]:
        """
        Generate tests for given code
        
        Args:
            code: Python code to generate tests for
            framework: Testing framework to use (pytest, unittest, nose2)
            test_type: Type of tests to generate (unit, integration, performance)
        
        Returns:
            Dict with success status, generated tests, and metadata
        """
        
        try:
            # Build the request based on test type
            if test_type == "unit":
                request = f"Generate comprehensive unit tests for this code:\n\n```python\n{code}\n```"
            elif test_type == "integration":
                request = f"Generate integration tests for this code:\n\n```python\n{code}\n```"
            elif test_type == "performance":
                request = f"Generate performance tests for this code:\n\n```python\n{code}\n```"
            elif test_type == "parameterized":
                request = f"Generate parameterized tests for this code:\n\n```python\n{code}\n```"
            else:
                request = f"Generate {test_type} tests for this code:\n\n```python\n{code}\n```"
            
            # Add framework preference if not pytest
            if framework != "pytest":
                request += f"\n\nUse {framework} as the testing framework."
            
            print(f"🧪 Generating {test_type} tests using {framework}...")
            
            # Use the stream method to generate tests
            full_response = ""
            for chunk in self.stream(request):
                if isinstance(chunk, dict):
                    # Extract the actual content from the chunk
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
            
            return {
                "success": True,
                "tests": full_response,
                "framework": framework,
                "test_type": test_type,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tests": "",
                "framework": framework,
                "test_type": test_type
            }
    
    def analyze_coverage(self, code: str, existing_tests: str) -> Dict[str, Any]:
        """
        Analyze test coverage and identify gaps
        
        Args:
            code: Original code
            existing_tests: Current test code
        
        Returns:
            Dict with coverage analysis and recommendations
        """
        
        try:
            request = f"""Analyze the test coverage for this code:

Original Code:
```python
{code}
```

Existing Tests:
```python
{existing_tests}
```

Identify coverage gaps and suggest additional tests."""
            
            print("📊 Analyzing test coverage...")
            
            full_response = ""
            for chunk in self.stream(request):
                if isinstance(chunk, dict):
                    # Extract the actual content from the chunk
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
            
            return {
                "success": True,
                "analysis": full_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis": ""
            }
    
    def generate_fixtures(self, code: str) -> Dict[str, Any]:
        """
        Generate test fixtures and mocks for given code
        
        Args:
            code: Python code to generate fixtures for
        
        Returns:
            Dict with generated fixtures and mocks
        """
        
        try:
            request = f"Generate test fixtures and mocks for this code:\n\n```python\n{code}\n```"
            
            print("🏗️ Generating test fixtures and mocks...")
            
            full_response = ""
            for chunk in self.stream(request):
                if isinstance(chunk, dict):
                    # Extract the actual content from the chunk
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
            
            return {
                "success": True,
                "fixtures": full_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fixtures": ""
            }