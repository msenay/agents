#!/usr/bin/env python3
"""
Executor Agent - Safe Code and Test Execution
============================================

ExecutorAgent specializes in safely executing Python code and running tests.

Key Features:
- Executes Python code with safety validation
- Runs unit tests with multiple framework support
- Creates and executes demo scripts from specifications
- Handles errors gracefully with detailed reporting
- Enforces security through code validation

The agent uses Core Agent infrastructure for memory and tool management.
"""

import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

# Core Agent Infrastructure
from core.core_agent import CoreAgent
from core.config import AgentConfig
from langchain_openai import AzureChatOpenAI

# Import prompts and tools
from agent.executor.prompts import SYSTEM_PROMPT
from agent.executor.tools import get_executor_tools, get_core_executor_tools


class ExecutorConfig:
    """Executor Agent Configuration"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://oai-202-fbeta-dev.openai.azure.com/")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4")
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    
    # Model Parameters
    TEMPERATURE = 0.1  # Low temperature for consistent execution
    MAX_TOKENS = 4000
    
    # Execution Parameters
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_TIMEOUT = 300  # 5 minutes max


class ExecutorAgent(CoreAgent):
    """
    Specialized agent for safely executing code and running tests
    
    This agent provides:
    - Safe code execution with validation
    - Unit test execution with detailed reports
    - Demo generation and execution
    - Error analysis and debugging help
    """
    
    def __init__(self, session_id: str = None, use_all_tools: bool = False):
        """
        Initialize Executor Agent
        
        Args:
            session_id: Unique session identifier (auto-generated if not provided)
            use_all_tools: If True, loads all tools. If False, only core tools.
        """
        if session_id is None:
            session_id = f"executor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create Azure OpenAI model
        model = AzureChatOpenAI(
            azure_endpoint=ExecutorConfig.AZURE_OPENAI_ENDPOINT,
            api_key=ExecutorConfig.OPENAI_API_KEY,
            api_version=ExecutorConfig.OPENAI_API_VERSION,
            model=ExecutorConfig.GPT4_MODEL_NAME,
            deployment_name=ExecutorConfig.GPT4_DEPLOYMENT_NAME,
            temperature=ExecutorConfig.TEMPERATURE,
            max_tokens=ExecutorConfig.MAX_TOKENS
        )
        
        # Create tools - use core tools by default for efficiency
        if use_all_tools:
            tools = get_executor_tools(model)
        else:
            tools = get_core_executor_tools(model)
        
        # Create configuration
        config = AgentConfig(
            name="ExecutorAgent",
            model=model,
            tools=tools,
            system_prompt=self._get_system_prompt(),
            enable_memory=True,  # Remember execution patterns and errors
            memory_backend="inmemory",
            memory_types=["short_term", "long_term"],
            max_tokens=ExecutorConfig.MAX_TOKENS
        )
        
        # Initialize parent CoreAgent
        super().__init__(config)
        
        print(f"✅ Executor Agent initialized - Ready to execute code safely!")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return SYSTEM_PROMPT
    
    def execute_code(self, code: str, timeout: int = None) -> Dict[str, Any]:
        """
        Execute Python code safely
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
        
        Returns:
            Dict with execution results
        """
        if timeout is None:
            timeout = ExecutorConfig.DEFAULT_TIMEOUT
        
        if timeout > ExecutorConfig.MAX_TIMEOUT:
            timeout = ExecutorConfig.MAX_TIMEOUT
        
        try:
            request = f"Execute this Python code with a timeout of {timeout} seconds:\n\n```python\n{code}\n```"
            
            print(f"🚀 Executing code...")
            
            # Use the stream method to execute
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
                "output": full_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
    
    def run_tests(self, test_code: str, framework: str = "pytest") -> Dict[str, Any]:
        """
        Run unit tests
        
        Args:
            test_code: Test code to execute
            framework: Testing framework (pytest, unittest)
        
        Returns:
            Dict with test results
        """
        
        try:
            request = f"Run these unit tests using {framework}:\n\n```python\n{test_code}\n```"
            
            print(f"🧪 Running tests with {framework}...")
            
            # Use the stream method
            full_response = ""
            for chunk in self.stream(request):
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
            
            return {
                "success": True,
                "results": full_response,
                "framework": framework,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "results": "",
                "framework": framework
            }
    
    def create_and_run_demo(self, code: str, spec: str) -> Dict[str, Any]:
        """
        Create and run a demo based on code and specifications
        
        Args:
            code: Code to create demo for
            spec: Specifications for the demo
        
        Returns:
            Dict with demo code and execution results
        """
        
        try:
            request = f"""Create and execute a demo for this code based on the specifications:

Code:
```python
{code}
```

Specifications:
{spec}

Create a working demo that showcases the functionality."""
            
            print("📝 Creating and running demo...")
            
            # Use the stream method
            full_response = ""
            for chunk in self.stream(request):
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
            
            return {
                "success": True,
                "demo_output": full_response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "demo_output": ""
            }
    
    def validate_code(self, code: str, strict: bool = True) -> Dict[str, Any]:
        """
        Validate code for safety before execution
        
        Args:
            code: Code to validate
            strict: Whether to use strict validation
        
        Returns:
            Dict with validation results
        """
        
        try:
            request = f"""Validate this code for safety and security issues:

```python
{code}
```

Use {'strict' if strict else 'relaxed'} validation mode."""
            
            print("🔍 Validating code safety...")
            
            # Use the stream method
            full_response = ""
            for chunk in self.stream(request):
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
            
            return {
                "success": True,
                "validation": full_response,
                "strict_mode": strict,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validation": "",
                "strict_mode": strict
            }