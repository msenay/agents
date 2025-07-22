#!/usr/bin/env python3
"""
Coder Agent Tools
================

A comprehensive collection of tools for the CoderAgent to generate, analyze,
optimize, test, and deploy agents.
"""

from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import json
import os


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class AgentGeneratorInput(BaseModel):
    """Input schema for agent generation"""
    template_type: str = Field(description="Type: simple, with_tools, multi_agent")
    agent_name: str = Field(description="Name for the agent")
    purpose: str = Field(description="What the agent should do")
    tools_needed: List[str] = Field(default=[], description="List of tools if needed")
    use_our_core: bool = Field(default=False, description="Whether to use Core Agent infrastructure")


class CodeAnalysisInput(BaseModel):
    """Input for code analysis"""
    code: str = Field(description="Agent code to analyze")
    focus_areas: List[str] = Field(default=[], description="Specific areas to focus on")


class TemplateInput(BaseModel):
    """Input for template operations"""
    template_name: str = Field(description="Name of the template")
    code: Optional[str] = Field(default=None, description="Code for the template")
    customizations: Optional[Dict[str, Any]] = Field(default=None, description="Customization parameters")


class DocumentationInput(BaseModel):
    """Input for documentation generation"""
    agent_name: str = Field(description="Name of the agent")
    code: str = Field(description="Agent code")
    doc_type: str = Field(default="markdown", description="Type of documentation")


class TestingInput(BaseModel):
    """Input for test generation"""
    agent_name: str = Field(description="Name of the agent")
    code: str = Field(description="Agent code")
    test_type: str = Field(default="unit", description="Type of tests: unit, integration, benchmark")


class DeploymentInput(BaseModel):
    """Input for deployment tools"""
    agent_name: str = Field(description="Name of the agent")
    code: str = Field(description="Agent code")
    deployment_type: str = Field(description="docker, k8s, ci_cd")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Deployment configuration")


class ConversionInput(BaseModel):
    """Input for conversion tools"""
    agent_name: str = Field(description="Name of the agent")
    code: str = Field(description="Agent code")
    target_format: str = Field(description="api, cli, sdk")
    framework: Optional[str] = Field(default=None, description="Target framework")


class EnhancementInput(BaseModel):
    """Input for enhancement tools"""
    code: str = Field(description="Agent code to enhance")
    enhancement_type: str = Field(description="monitoring, logging, tracing")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Enhancement configuration")


# ============================================================================
# TOOL FACTORY FUNCTIONS
# ============================================================================

def create_agent_generator_tool(model):
    """Factory function to create AgentGeneratorTool with model"""
    
    class AgentGeneratorTool(BaseTool):
        """Tool for generating agent code"""
        name: str = "agent_generator"
        description: str = "Generate LangGraph agent code based on specifications"
        args_schema: type[BaseModel] = AgentGeneratorInput
        
        def _run(self, template_type: str, agent_name: str, purpose: str, 
                 tools_needed: List[str] = None, use_our_core: bool = False) -> str:
            """Generate agent code based on template type"""
            
            if tools_needed is None:
                tools_needed = []
            
            # Different prompts based on whether to use Core Agent
            if use_our_core:
                prompt = f"""Generate a complete LangGraph agent that uses our Core Agent infrastructure:
            
Type: {template_type}
Name: {agent_name}
Purpose: {purpose}
Tools: {tools_needed if tools_needed else 'None'}

Requirements for Core Agent based implementation:
1. Import and inherit from CoreAgent at /workspace/core/core_agent.py
2. Use AgentConfig from /workspace/core/config.py
3. Leverage Core Agent's built-in features (memory, tools, etc.)
4. Create a proper __init__ method that calls super().__init__(config)
5. Include all necessary imports
6. Add a demo function
7. Use proper error handling and logging

Generate the complete Python code:"""
            else:
                prompt = f"""Generate a complete standalone LangGraph agent:
            
Type: {template_type}
Name: {agent_name}  
Purpose: {purpose}
Tools: {tools_needed if tools_needed else 'None'}

Requirements for standalone implementation:
1. Use standard LangGraph patterns (StateGraph, add_node, add_edge)
2. Define proper agent state with TypedDict
3. Create clean node functions for each step
4. Add proper tool integration if needed
5. Include all necessary imports (langchain, langgraph, etc.)
6. Create a working example/demo at the end
7. Make it self-contained and ready to run

For {template_type} type:
{"- Simple: Basic agent with state management and clear workflow" if template_type == "simple" else ""}
{"- With Tools: Include tool node, proper tool calling and result handling" if template_type == "with_tools" else ""}
{"- Multi-agent: Create supervisor pattern with multiple sub-agents coordinating" if template_type == "multi_agent" else ""}

Generate the complete Python code:"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return AgentGeneratorTool()


def create_analyze_agent_code_tool(model):
    """Analyze existing agent code and provide improvement suggestions"""
    
    class AnalyzeAgentCodeTool(BaseTool):
        name: str = "analyze_agent_code"
        description: str = "Analyze agent code and provide improvement suggestions"
        args_schema: type[BaseModel] = CodeAnalysisInput
        
        def _run(self, code: str, focus_areas: List[str] = None) -> str:
            prompt = f"""Analyze this agent code and provide detailed improvement suggestions:

Code:
```python
{code}
```

Focus areas: {focus_areas if focus_areas else "General analysis"}

Provide:
1. Code quality assessment
2. Performance optimization opportunities
3. Error handling improvements
4. Best practice violations
5. Security concerns
6. Suggested refactoring
7. Missing features

Format as actionable recommendations."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return AnalyzeAgentCodeTool()


def create_validate_agent_tool(model):
    """Validate if generated code will work"""
    
    class ValidateAgentTool(BaseTool):
        name: str = "validate_agent"
        description: str = "Validate if the agent code will run without errors"
        args_schema: type[BaseModel] = CodeAnalysisInput
        
        def _run(self, code: str, focus_areas: List[str] = None) -> str:
            prompt = f"""Validate this agent code for runtime errors:

```python
{code}
```

Check for:
1. Import errors
2. Syntax errors
3. Missing dependencies
4. Undefined variables
5. Type mismatches
6. Logic errors
7. Configuration issues

Return validation report with:
- Status: PASS/FAIL
- Issues found
- How to fix each issue"""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return ValidateAgentTool()


def create_optimize_agent_tool(model):
    """Optimize agent code for performance and best practices"""
    
    class OptimizeAgentTool(BaseTool):
        name: str = "optimize_agent"
        description: str = "Optimize agent code for performance and best practices"
        args_schema: type[BaseModel] = CodeAnalysisInput
        
        def _run(self, code: str, focus_areas: List[str] = None) -> str:
            prompt = f"""Optimize this agent code:

```python
{code}
```

Optimization areas:
1. Performance improvements
2. Memory efficiency
3. Async/await usage
4. Caching opportunities
5. Better error handling
6. Code simplification
7. Design pattern improvements

Return the optimized code with comments explaining each optimization."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return OptimizeAgentTool()


def create_save_template_tool(model):
    """Save successful agents as reusable templates"""
    
    class SaveTemplateTool(BaseTool):
        name: str = "save_agent_template"
        description: str = "Save agent code as a reusable template"
        args_schema: type[BaseModel] = TemplateInput
        
        def _run(self, template_name: str, code: str = None, customizations: Dict[str, Any] = None) -> str:
            # In real implementation, this would save to a database or file system
            prompt = f"""Create a reusable template from this agent code:

Template Name: {template_name}
Code:
```python
{code}
```

Create a template that:
1. Identifies customizable parameters
2. Adds template variables
3. Creates configuration schema
4. Includes usage instructions
5. Provides example customizations

Return the template structure."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return SaveTemplateTool()


def create_generate_docs_tool(model):
    """Generate comprehensive documentation for agents"""
    
    class GenerateDocsTool(BaseTool):
        name: str = "generate_agent_docs"
        description: str = "Generate documentation for an agent"
        args_schema: type[BaseModel] = DocumentationInput
        
        def _run(self, agent_name: str, code: str, doc_type: str = "markdown") -> str:
            prompt = f"""Generate comprehensive documentation for this agent:

Agent Name: {agent_name}
Code:
```python
{code}
```

Documentation type: {doc_type}

Include:
1. Overview and purpose
2. Installation instructions
3. Configuration options
4. API reference
5. Usage examples
6. Architecture diagram (as ASCII art)
7. Troubleshooting guide
8. Contributing guidelines

Format as professional documentation."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return GenerateDocsTool()


def create_generate_tests_tool(model):
    """Generate test suites for agents"""
    
    class GenerateTestsTool(BaseTool):
        name: str = "generate_unit_tests"
        description: str = "Generate tests for an agent"
        args_schema: type[BaseModel] = TestingInput
        
        def _run(self, agent_name: str, code: str, test_type: str = "unit") -> str:
            prompt = f"""Generate {test_type} tests for this agent:

Agent Name: {agent_name}
Code:
```python
{code}
```

Create comprehensive tests that:
1. Test all public methods
2. Cover edge cases
3. Mock external dependencies
4. Test error scenarios
5. Validate outputs
6. Include fixtures and setup
7. Use pytest best practices

Return complete test file."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return GenerateTestsTool()


def create_dockerize_tool(model):
    """Create Docker configuration for agents"""
    
    class DockerizeTool(BaseTool):
        name: str = "dockerize_agent"
        description: str = "Create Docker configuration for an agent"
        args_schema: type[BaseModel] = DeploymentInput
        
        def _run(self, agent_name: str, code: str, deployment_type: str = "docker", config: Dict[str, Any] = None) -> str:
            prompt = f"""Create Docker configuration for this agent:

Agent Name: {agent_name}
Code:
```python
{code}
```

Generate:
1. Optimized Dockerfile
2. docker-compose.yml
3. .dockerignore
4. Environment configuration
5. Health checks
6. Multi-stage build if needed
7. Security best practices

Include comments explaining each decision."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return DockerizeTool()


def create_convert_to_api_tool(model):
    """Convert agents to REST APIs"""
    
    class ConvertToAPITool(BaseTool):
        name: str = "convert_to_api"
        description: str = "Convert agent to FastAPI/Flask API"
        args_schema: type[BaseModel] = ConversionInput
        
        def _run(self, agent_name: str, code: str, target_format: str = "api", framework: str = "fastapi") -> str:
            prompt = f"""Convert this agent to a {framework} API:

Agent Name: {agent_name}
Code:
```python
{code}
```

Create a complete API with:
1. RESTful endpoints
2. Request/response models
3. Authentication
4. Rate limiting
5. OpenAPI documentation
6. Error handling
7. Async support (if FastAPI)
8. Deployment ready

Return complete API code."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return ConvertToAPITool()


def create_add_monitoring_tool(model):
    """Add monitoring capabilities to agents"""
    
    class AddMonitoringTool(BaseTool):
        name: str = "add_monitoring"
        description: str = "Add monitoring and observability to agent"
        args_schema: type[BaseModel] = EnhancementInput
        
        def _run(self, code: str, enhancement_type: str = "monitoring", config: Dict[str, Any] = None) -> str:
            prompt = f"""Add monitoring to this agent code:

```python
{code}
```

Add:
1. Prometheus metrics (counters, gauges, histograms)
2. Health check endpoints
3. Performance tracking
4. Resource usage monitoring
5. Custom business metrics
6. Grafana dashboard config
7. Alerting rules

Return enhanced code with monitoring."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return AddMonitoringTool()


def create_generate_rag_agent_tool(model):
    """Generate RAG (Retrieval Augmented Generation) agents"""
    
    class GenerateRAGAgentTool(BaseTool):
        name: str = "generate_rag_agent"
        description: str = "Generate a RAG agent with vector store and retrieval"
        args_schema: type[BaseModel] = AgentGeneratorInput
        
        def _run(self, template_type: str, agent_name: str, purpose: str, 
                 tools_needed: List[str] = None, use_our_core: bool = False) -> str:
            prompt = f"""Generate a RAG (Retrieval Augmented Generation) agent:

Name: {agent_name}
Purpose: {purpose}
Additional Tools: {tools_needed if tools_needed else 'None'}
Use Core Agent: {use_our_core}

Create a complete RAG agent with:
1. Vector store setup (FAISS/Chroma/Pinecone)
2. Document loaders and splitters
3. Embedding generation
4. Retrieval chain
5. Query optimization
6. Source citation
7. Memory integration
8. Fallback strategies

Return production-ready RAG agent code."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return GenerateRAGAgentTool()


def create_format_code_tool(model):
    """Format and clean up agent code"""
    
    class FormatCodeTool(BaseTool):
        name: str = "format_code"
        description: str = "Format code with Black/isort standards"
        args_schema: type[BaseModel] = CodeAnalysisInput
        
        def _run(self, code: str, focus_areas: List[str] = None) -> str:
            prompt = f"""Format and clean up this code:

```python
{code}
```

Apply:
1. Black formatting standards
2. isort import ordering
3. Remove unused imports
4. Fix line lengths
5. Consistent naming conventions
6. Proper docstrings
7. Type hints where missing

Return the formatted code."""
            
            response = model.invoke([HumanMessage(content=prompt)])
            return response.content
    
    return FormatCodeTool()


# ============================================================================
# TOOL COLLECTION FACTORY
# ============================================================================

def create_all_coder_tools(model) -> List[BaseTool]:
    """Create all available tools for CoderAgent"""
    
    return [
        # Core generation
        create_agent_generator_tool(model),
        
        # Analysis tools
        create_analyze_agent_code_tool(model),
        create_validate_agent_tool(model),
        create_optimize_agent_tool(model),
        
        # Template tools
        create_save_template_tool(model),
        
        # Documentation tools
        create_generate_docs_tool(model),
        
        # Testing tools
        create_generate_tests_tool(model),
        
        # Deployment tools
        create_dockerize_tool(model),
        
        # Conversion tools
        create_convert_to_api_tool(model),
        
        # Enhancement tools
        create_add_monitoring_tool(model),
        
        # Advanced agent tools
        create_generate_rag_agent_tool(model),
        
        # Code quality tools
        create_format_code_tool(model)
    ]


def get_tools_by_category(model, categories: List[str]) -> List[BaseTool]:
    """Get tools filtered by category"""
    
    tool_categories = {
        "generation": [create_agent_generator_tool, create_generate_rag_agent_tool],
        "analysis": [create_analyze_agent_code_tool, create_validate_agent_tool, create_optimize_agent_tool],
        "documentation": [create_generate_docs_tool],
        "testing": [create_generate_tests_tool],
        "deployment": [create_dockerize_tool],
        "conversion": [create_convert_to_api_tool],
        "enhancement": [create_add_monitoring_tool],
        "quality": [create_format_code_tool],
        "template": [create_save_template_tool]
    }
    
    tools = []
    for category in categories:
        if category in tool_categories:
            for tool_factory in tool_categories[category]:
                tools.append(tool_factory(model))
    
    return tools


# ============================================================================
# SMART TOOL SELECTION
# ============================================================================

def suggest_tools_for_task(task_description: str) -> List[str]:
    """Suggest relevant tools based on task description"""
    
    tool_keywords = {
        "agent_generator": ["create", "generate", "build", "new agent"],
        "analyze_agent_code": ["analyze", "review", "check", "inspect"],
        "validate_agent": ["validate", "test", "verify", "working"],
        "optimize_agent": ["optimize", "improve", "performance", "faster"],
        "generate_agent_docs": ["document", "docs", "readme", "documentation"],
        "generate_unit_tests": ["test", "unit test", "testing", "pytest"],
        "dockerize_agent": ["docker", "container", "deploy", "deployment"],
        "convert_to_api": ["api", "rest", "fastapi", "flask", "endpoint"],
        "add_monitoring": ["monitor", "metrics", "prometheus", "observability"],
        "generate_rag_agent": ["rag", "retrieval", "vector", "knowledge base"],
        "format_code": ["format", "clean", "style", "black", "lint"]
    }
    
    suggested_tools = []
    task_lower = task_description.lower()
    
    for tool, keywords in tool_keywords.items():
        if any(keyword in task_lower for keyword in keywords):
            suggested_tools.append(tool)
    
    return suggested_tools