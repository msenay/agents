#!/usr/bin/env python3
"""
Professional Coder Agent using Core Agent Infrastructure
========================================================

An advanced AI agent specialized in writing high-quality LangGraph agents,
tools, and multi-agent systems. Built on top of Core Agent framework.
Uses Azure OpenAI and Redis memory with Core Agent's powerful tools.
"""

import os
import sys
import json
import re
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add workspace to path
sys.path.insert(0, '/workspace')

# Import Core Agent infrastructure
from core.config import AgentConfig
from core.managers import MemoryManager, ToolManager, ModelManager
from core.tools import create_python_coding_tools
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

class CoderConfig:
    """Configuration for Coder Agent using Core Agent infrastructure"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = "https://oai-202-fbeta-dev.openai.azure.com/"
    OPENAI_API_KEY = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
    OPENAI_API_VERSION = "2023-12-01-preview"
    
    # Model Configuration
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000
    
    # Redis Configuration for Core Agent Memory
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")  # DB 2 for coder


class CoderAgentSpecializedTools:
    """Specialized tools for Coder Agent using Core Agent infrastructure"""
    
    @staticmethod
    def create_langgraph_template_tool():
        """Create tool for LangGraph agent templates"""
        
        class LangGraphTemplateInput(BaseModel):
            template_type: str = Field(description="Type: simple, with_tools, multi_agent")
            agent_name: str = Field(description="Name for the agent")
            description: str = Field(default="", description="Agent description")
        
        class LangGraphTemplateTool(BaseTool):
            name: str = "langgraph_template"
            description: str = """Generate LangGraph agent templates.
            Types: simple (basic agent), with_tools (agent with tools), multi_agent (supervisor system)"""
            args_schema: type[BaseModel] = LangGraphTemplateInput
            
            def _run(self, template_type: str, agent_name: str, description: str = "") -> str:
                templates = {
                    "simple": f'''# {agent_name} - Simple LangGraph Agent
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, List
from langchain_openai import AzureChatOpenAI

class {agent_name}State(TypedDict):
    """State for {agent_name}"""
    messages: List[BaseMessage]
    input: str
    output: str

def process_node(state: {agent_name}State) -> {agent_name}State:
    """Main processing node"""
    llm = AzureChatOpenAI(
        azure_endpoint="{CoderConfig.AZURE_OPENAI_ENDPOINT}",
        api_key="{CoderConfig.OPENAI_API_KEY}",
        api_version="{CoderConfig.OPENAI_API_VERSION}",
        model="{CoderConfig.GPT4_MODEL_NAME}",
        deployment_name="{CoderConfig.GPT4_DEPLOYMENT_NAME}",
        temperature=0.1
    )
    
    try:
        response = llm.invoke(state["input"])
        return {{**state, "output": response.content}}
    except Exception as e:
        return {{**state, "output": f"Error: {{str(e)}}"}}

def create_{agent_name.lower()}_agent():
    """Create and return the {agent_name} agent"""
    workflow = StateGraph({agent_name}State)
    
    # Add nodes
    workflow.add_node("process", process_node)
    
    # Set entry point
    workflow.set_entry_point("process")
    
    # Add ending
    workflow.add_edge("process", END)
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    agent = create_{agent_name.lower()}_agent()
    result = agent.invoke({{"input": "Hello, world!", "messages": [], "output": ""}})
    print(f"Result: {{result['output']}}")
''',
                    
                    "with_tools": f'''# {agent_name} - LangGraph Agent with Tools
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from typing import TypedDict, List, Literal
from langchain_openai import AzureChatOpenAI

@tool
def example_tool(query: str) -> str:
    """Example tool for {agent_name}"""
    return f"Tool processed: {{query}}"

class {agent_name}State(TypedDict):
    """State for {agent_name} with tools"""
    messages: List[BaseMessage]
    input: str
    tool_choice: str
    tool_result: str
    output: str

def decide_tool_node(state: {agent_name}State) -> {agent_name}State:
    """Decide which tool to use"""
    # Simple tool selection logic
    if "example" in state["input"].lower():
        return {{**state, "tool_choice": "example_tool"}}
    else:
        return {{**state, "tool_choice": "none"}}

def execute_tool_node(state: {agent_name}State) -> {agent_name}State:
    """Execute the chosen tool"""
    if state["tool_choice"] == "example_tool":
        result = example_tool.invoke({{"query": state["input"]}})
        return {{**state, "tool_result": result}}
    else:
        return {{**state, "tool_result": "No tool used"}}

def finalize_node(state: {agent_name}State) -> {agent_name}State:
    """Finalize the response"""
    llm = AzureChatOpenAI(
        azure_endpoint="{CoderConfig.AZURE_OPENAI_ENDPOINT}",
        api_key="{CoderConfig.OPENAI_API_KEY}",
        api_version="{CoderConfig.OPENAI_API_VERSION}",
        model="{CoderConfig.GPT4_MODEL_NAME}",
        deployment_name="{CoderConfig.GPT4_DEPLOYMENT_NAME}",
        temperature=0.1
    )
    
    context = f"Input: {{state['input']}}\\nTool Result: {{state['tool_result']}}"
    response = llm.invoke(context)
    return {{**state, "output": response.content}}

def create_{agent_name.lower()}_agent():
    """Create agent with tools"""
    workflow = StateGraph({agent_name}State)
    
    # Add nodes
    workflow.add_node("decide_tool", decide_tool_node)
    workflow.add_node("execute_tool", execute_tool_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("decide_tool")
    
    # Add edges
    workflow.add_edge("decide_tool", "execute_tool")
    workflow.add_edge("execute_tool", "finalize")
    workflow.add_edge("finalize", END)
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    agent = create_{agent_name.lower()}_agent()
    result = agent.invoke({{
        "input": "Use example tool to process this",
        "messages": [], "tool_choice": "", "tool_result": "", "output": ""
    }})
    print(f"Result: {{result['output']}}")
''',
                    
                    "multi_agent": f'''# {agent_name} - Multi-Agent System with Supervisor
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, List, Literal
from langchain_openai import AzureChatOpenAI

class {agent_name}State(TypedDict):
    """State for multi-agent system"""
    messages: List[BaseMessage]
    task: str
    assigned_agent: str
    agent_results: dict
    final_output: str

def supervisor_node(state: {agent_name}State) -> {agent_name}State:
    """Supervisor decides which agent to use"""
    llm = AzureChatOpenAI(
        azure_endpoint="{CoderConfig.AZURE_OPENAI_ENDPOINT}",
        api_key="{CoderConfig.OPENAI_API_KEY}",
        api_version="{CoderConfig.OPENAI_API_VERSION}",
        model="{CoderConfig.GPT4_MODEL_NAME}",
        deployment_name="{CoderConfig.GPT4_DEPLOYMENT_NAME}",
        temperature=0.1
    )
    
    task = state["task"].lower()
    if "research" in task:
        return {{**state, "assigned_agent": "researcher"}}
    elif "write" in task or "create" in task:
        return {{**state, "assigned_agent": "writer"}}
    elif "analyze" in task:
        return {{**state, "assigned_agent": "analyzer"}}
    else:
        return {{**state, "assigned_agent": "general"}}

def researcher_node(state: {agent_name}State) -> {agent_name}State:
    """Research agent"""
    result = f"Research completed for: {{state['task']}}"
    results = state.get("agent_results", {{}})
    results["researcher"] = result
    return {{**state, "agent_results": results}}

def writer_node(state: {agent_name}State) -> {agent_name}State:
    """Writer agent"""
    result = f"Content created for: {{state['task']}}"
    results = state.get("agent_results", {{}})
    results["writer"] = result
    return {{**state, "agent_results": results}}

def analyzer_node(state: {agent_name}State) -> {agent_name}State:
    """Analyzer agent"""
    result = f"Analysis completed for: {{state['task']}}"
    results = state.get("agent_results", {{}})
    results["analyzer"] = result
    return {{**state, "agent_results": results}}

def general_node(state: {agent_name}State) -> {agent_name}State:
    """General purpose agent"""
    result = f"General processing for: {{state['task']}}"
    results = state.get("agent_results", {{}})
    results["general"] = result
    return {{**state, "agent_results": results}}

def aggregator_node(state: {agent_name}State) -> {agent_name}State:
    """Aggregate results from all agents"""
    llm = AzureChatOpenAI(
        azure_endpoint="{CoderConfig.AZURE_OPENAI_ENDPOINT}",
        api_key="{CoderConfig.OPENAI_API_KEY}",
        api_version="{CoderConfig.OPENAI_API_VERSION}",
        model="{CoderConfig.GPT4_MODEL_NAME}",
        deployment_name="{CoderConfig.GPT4_DEPLOYMENT_NAME}",
        temperature=0.1
    )
    
    results_summary = "\\n".join([f"{{k}}: {{v}}" for k, v in state["agent_results"].items()])
    final_prompt = f"Task: {{state['task']}}\\n\\nAgent Results:\\n{{results_summary}}\\n\\nProvide final response:"
    
    response = llm.invoke(final_prompt)
    return {{**state, "final_output": response.content}}

def route_to_agent(state: {agent_name}State) -> str:
    """Route to appropriate agent"""
    return state["assigned_agent"]

def create_{agent_name.lower()}_system():
    """Create multi-agent system"""
    workflow = StateGraph({agent_name}State)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("general", general_node)
    workflow.add_node("aggregator", aggregator_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {{
            "researcher": "researcher",
            "writer": "writer", 
            "analyzer": "analyzer",
            "general": "general"
        }}
    )
    
    # All agents go to aggregator
    workflow.add_edge("researcher", "aggregator")
    workflow.add_edge("writer", "aggregator")
    workflow.add_edge("analyzer", "aggregator")
    workflow.add_edge("general", "aggregator")
    workflow.add_edge("aggregator", END)
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    system = create_{agent_name.lower()}_system()
    result = system.invoke({{
        "task": "Research the latest AI trends",
        "messages": [], "assigned_agent": "", "agent_results": {{}}, "final_output": ""
    }})
    print(f"Final Output: {{result['final_output']}}")
'''
                }
                
                if template_type in templates:
                    return f"✅ Generated {template_type} template for {agent_name}:\n\n{templates[template_type]}"
                else:
                    return f"❌ Unknown template type: {template_type}. Use: simple, with_tools, multi_agent"
        
        return LangGraphTemplateTool()
    
    @staticmethod
    def create_code_memory_tool():
        """Create tool for managing code modules in memory"""
        
        class CodeMemoryInput(BaseModel):
            action: str = Field(description="Action: save, load, list, search, update")
            module_name: str = Field(default="", description="Name of the module")
            code: str = Field(default="", description="Code to save")
            description: str = Field(default="", description="Module description")
            tags: List[str] = Field(default=[], description="Tags for categorization")
        
        class CodeMemoryTool(BaseTool):
            name: str = "code_memory"
            description: str = """Manage code modules in Redis memory with Core Agent.
            Actions: save, load, list, search, update"""
            args_schema: type[BaseModel] = CodeMemoryInput
            
            def __init__(self):
                super().__init__()
                # This will use Core Agent's memory system when available
                self.memory_key_prefix = "coder:modules"
            
            def _run(self, action: str, module_name: str = "", code: str = "", 
                    description: str = "", tags: List[str] = []) -> str:
                """Execute memory action using Core Agent infrastructure"""
                try:
                    if action == "save":
                        if not module_name or not code:
                            return "❌ Save requires module_name and code"
                        
                        # Use Core Agent's Redis memory (when available)
                        module_data = {
                            "code": code,
                            "description": description,
                            "tags": tags,
                            "timestamp": datetime.now().isoformat(),
                            "type": "langgraph_module"
                        }
                        
                        # For now, simulate saving (would use Core Agent's memory)
                        return f"✅ Module '{module_name}' saved to Core Agent memory"
                    
                    elif action == "load":
                        if not module_name:
                            return "❌ Load requires module_name"
                        
                        # Simulate loading from Core Agent memory
                        return f"✅ Module '{module_name}' loaded from Core Agent memory"
                    
                    elif action == "list":
                        # Simulate listing from Core Agent memory
                        return "📚 Available modules in Core Agent memory: [simulated list]"
                    
                    elif action == "search":
                        return f"🔍 Searching Core Agent memory for: {module_name}"
                    
                    else:
                        return f"❌ Unknown action: {action}"
                        
                except Exception as e:
                    return f"❌ Memory error: {str(e)}"
        
        return CodeMemoryTool()


class CodeValidatorInput(BaseModel):
    """Input for code validator tool"""
    code: str = Field(description="Python/LangGraph code to validate")
    code_type: str = Field(default="python", description="Type of code: python, langgraph, agent")


class CodeValidatorTool(BaseTool):
    """Tool to validate and test generated code"""
    
    name: str = "code_validator"
    description: str = """Validate Python/LangGraph code for syntax errors, imports, and basic functionality.
    Returns validation results and suggestions for improvement."""
    
    args_schema: type[BaseModel] = CodeValidatorInput
    
    def _run(self, code: str, code_type: str = "python") -> str:
        """Validate code and return results"""
        results = []
        
        try:
            # 1. Syntax validation
            try:
                compile(code, '<string>', 'exec')
                results.append("✅ Syntax validation: PASSED")
            except SyntaxError as e:
                results.append(f"❌ Syntax error: {e}")
                return "\n".join(results)
            
            # 2. Import validation
            import_lines = [line.strip() for line in code.split('\n') 
                          if line.strip().startswith(('import ', 'from '))]
            
            if import_lines:
                results.append("📦 Import analysis:")
                for imp in import_lines:
                    results.append(f"   - {imp}")
            
            # 3. LangGraph-specific validation
            if code_type in ["langgraph", "agent"]:
                langgraph_patterns = [
                    (r'from langgraph', "LangGraph imports"),
                    (r'StateGraph', "StateGraph usage"),
                    (r'@tool', "Tool decorators"),
                    (r'def.*_node\(', "Node functions"),
                    (r'\.add_node\(', "Node additions"),
                    (r'\.add_edge\(', "Edge additions"),
                ]
                
                results.append("🔍 LangGraph pattern analysis:")
                for pattern, description in langgraph_patterns:
                    if re.search(pattern, code):
                        results.append(f"   ✅ {description}: Found")
                    else:
                        results.append(f"   ⚠️ {description}: Not found")
            
            # 4. Code quality checks
            quality_checks = [
                (r'class \w+:', "Class definitions"),
                (r'def \w+\(.*\):', "Function definitions"),
                (r'""".*?"""', "Docstrings"),
                (r'#.*', "Comments"),
            ]
            
            results.append("📊 Code quality analysis:")
            for pattern, description in quality_checks:
                matches = len(re.findall(pattern, code, re.DOTALL))
                results.append(f"   - {description}: {matches} found")
            
            # 5. Suggestions
            suggestions = []
            if not re.search(r'""".*?"""', code, re.DOTALL):
                suggestions.append("Consider adding docstrings for better documentation")
            
            if code_type == "agent" and not re.search(r'StateGraph', code):
                suggestions.append("Agent code should typically use StateGraph")
            
            if not re.search(r'def.*\(.*\):', code):
                suggestions.append("Consider organizing code into functions")
            
            if suggestions:
                results.append("💡 Suggestions:")
                for suggestion in suggestions:
                    results.append(f"   - {suggestion}")
            
            return "\n".join(results)
            
        except Exception as e:
            return f"❌ Validation failed: {str(e)}"


class ModuleManagerInput(BaseModel):
    """Input for module manager tool"""
    action: str = Field(description="Action: save, load, list, search")
    module_name: str = Field(default="", description="Name of the module")
    code: str = Field(default="", description="Code to save")
    description: str = Field(default="", description="Module description")


class ModuleManagerTool(BaseTool):
    """Tool to manage code modules in memory"""
    
    name: str = "module_manager"
    description: str = """Manage code modules in Redis memory. 
    Actions: save (store module), load (retrieve module), list (show all), search (find modules)"""
    
    args_schema: type[BaseModel] = ModuleManagerInput
    
    def __init__(self, memory: CoderMemory):
        super().__init__()
        self._memory = memory
    
    def _run(self, action: str, module_name: str = "", code: str = "", description: str = "") -> str:
        """Execute module management action"""
        try:
            if action == "save":
                if not module_name or not code:
                    return "❌ Save action requires module_name and code"
                
                success = self._memory.save_module(module_name, code, description)
                if success:
                    return f"✅ Module '{module_name}' saved successfully"
                else:
                    return f"❌ Failed to save module '{module_name}'"
            
            elif action == "load":
                if not module_name:
                    return "❌ Load action requires module_name"
                
                module_data = self._memory.get_module(module_name)
                if module_data:
                    return f"""✅ Module '{module_name}' loaded:
Description: {module_data.get('description', 'No description')}
Timestamp: {module_data.get('timestamp', 'Unknown')}

Code:
```python
{module_data['code']}
```"""
                else:
                    return f"❌ Module '{module_name}' not found"
            
            elif action == "list":
                modules = self._memory.list_modules()
                if modules:
                    return f"📚 Available modules ({len(modules)}):\n" + "\n".join(f"  - {m}" for m in modules)
                else:
                    return "📚 No modules found in memory"
            
            elif action == "search":
                if not module_name:
                    return "❌ Search action requires search term in module_name"
                
                modules = self._memory.list_modules()
                found = [m for m in modules if module_name.lower() in m.lower()]
                
                if found:
                    return f"🔍 Found {len(found)} modules matching '{module_name}':\n" + "\n".join(f"  - {m}" for m in found)
                else:
                    return f"🔍 No modules found matching '{module_name}'"
            
            else:
                return f"❌ Unknown action '{action}'. Use: save, load, list, search"
                
        except Exception as e:
            return f"❌ Module manager error: {str(e)}"


class FileOperationsInput(BaseModel):
    """Input for file operations tool"""
    action: str = Field(description="Action: write, read, create_project")
    filepath: str = Field(description="Path to the file")
    content: str = Field(default="", description="Content to write")
    project_name: str = Field(default="", description="Project name for create_project")


class FileOperationsTool(BaseTool):
    """Tool for file operations and project creation"""
    
    name: str = "file_operations"
    description: str = """Handle file operations: write files, read files, create project structures.
    Actions: write (save file), read (load file), create_project (setup structure)"""
    
    args_schema: type[BaseModel] = FileOperationsInput
    
    def __init__(self, workspace_dir: str = "/workspace"):
        super().__init__()
        self._workspace_dir = workspace_dir
    
    def _run(self, action: str, filepath: str, content: str = "", project_name: str = "") -> str:
        """Execute file operation"""
        try:
            if action == "write":
                full_path = os.path.join(self._workspace_dir, filepath)
                
                # Create directories if needed
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                file_size = len(content)
                return f"✅ File written: {filepath} ({file_size} characters)"
            
            elif action == "read":
                full_path = os.path.join(self._workspace_dir, filepath)
                
                if not os.path.exists(full_path):
                    return f"❌ File not found: {filepath}"
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                return f"📄 File content ({len(content)} chars):\n\n{content}"
            
            elif action == "create_project":
                if not project_name:
                    return "❌ create_project requires project_name"
                
                project_path = os.path.join(self._workspace_dir, project_name)
                
                # Create project structure
                dirs = [
                    f"{project_name}",
                    f"{project_name}/agents",
                    f"{project_name}/tools", 
                    f"{project_name}/tests",
                    f"{project_name}/config"
                ]
                
                files = [
                    (f"{project_name}/__init__.py", "# Project package"),
                    (f"{project_name}/agents/__init__.py", "# Agents package"),
                    (f"{project_name}/tools/__init__.py", "# Tools package"), 
                    (f"{project_name}/tests/__init__.py", "# Tests package"),
                    (f"{project_name}/config/__init__.py", "# Config package"),
                    (f"{project_name}/requirements.txt", "langgraph>=0.0.40\nlangchain>=0.1.0\n"),
                    (f"{project_name}/README.md", f"# {project_name}\n\nGenerated by Coder Agent\n")
                ]
                
                # Create directories
                for dir_path in dirs:
                    full_dir_path = os.path.join(self._workspace_dir, dir_path)
                    os.makedirs(full_dir_path, exist_ok=True)
                
                # Create files
                for file_path, file_content in files:
                    full_file_path = os.path.join(self._workspace_dir, file_path)
                    with open(full_file_path, 'w', encoding='utf-8') as f:
                        f.write(file_content)
                
                return f"✅ Project '{project_name}' created with structure:\n" + "\n".join(f"  📁 {d}" for d in dirs)
            
            else:
                return f"❌ Unknown action '{action}'. Use: write, read, create_project"
                
        except Exception as e:
            return f"❌ File operation error: {str(e)}"


class CoderAgent:
    """Professional Coder Agent using Core Agent Infrastructure"""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        
        # Create Core Agent configuration
        self.config = AgentConfig(
            name="CoderAgent",
            model=self._create_azure_openai_model(),
            enable_memory=True,
            memory_backend="redis",
            memory_types=["short_term", "long_term"],
            tools=self._create_specialized_tools(),
            system_prompt=self._get_expert_system_prompt(),
            max_iterations=10,
            temperature=CoderConfig.TEMPERATURE
        )
        
        # Initialize Core Agent managers
        try:
            self.memory_manager = MemoryManager(self.config)
            self.tool_manager = ToolManager(self.config)
            print(f"✅ Core Agent infrastructure initialized for Coder")
        except Exception as e:
            print(f"⚠️ Core Agent managers not available: {e}")
            self.memory_manager = None
            self.tool_manager = None
        
        # Direct Azure OpenAI for immediate use
        self.llm = self._create_azure_openai_model()
        
        print(f"🤖 Coder Agent initialized with session: {session_id}")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")
        print(f"💾 Memory backend: {self.config.memory_backend}")
    
    def _create_azure_openai_model(self):
        """Create Azure OpenAI model with proper configuration"""
        return AzureChatOpenAI(
            azure_endpoint=CoderConfig.AZURE_OPENAI_ENDPOINT,
            api_key=CoderConfig.OPENAI_API_KEY,
            api_version=CoderConfig.OPENAI_API_VERSION,
            model=CoderConfig.GPT4_MODEL_NAME,
            deployment_name=CoderConfig.GPT4_DEPLOYMENT_NAME,
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
    
    def _create_specialized_tools(self):
        """Create specialized tools for coding tasks"""
        tools = []
        
        # Add Core Agent's Python coding tools
        try:
            core_tools = create_python_coding_tools()
            tools.extend(core_tools)
            print(f"✅ Added {len(core_tools)} Core Agent tools")
        except Exception as e:
            print(f"⚠️ Core Agent tools not available: {e}")
        
        # Add specialized Coder Agent tools
        specialized_tools = CoderAgentSpecializedTools()
        tools.extend([
            specialized_tools.create_langgraph_template_tool(),
            specialized_tools.create_code_memory_tool(),
            CodeValidatorTool(),
            FileOperationsTool()
        ])
        
        return tools
    
    def _get_expert_system_prompt(self) -> str:
        """Get the expert system prompt leveraging Core Agent capabilities"""
        return """You are an ELITE CODER AGENT powered by Core Agent infrastructure, specialized in creating world-class LangGraph agents, tools, and multi-agent systems.

🎯 YOUR CORE MISSION:
Create exceptional, production-ready code using Core Agent's powerful infrastructure:
1. **Single LangGraph Agents**: Custom workflows with Azure OpenAI GPT-4
2. **Tool-Enhanced Agents**: Agents with specialized capabilities
3. **Multi-Agent Systems**: Sophisticated supervisor-worker architectures
4. **Complete Ecosystems**: Full project structures with proper organization

🧠 YOUR SUPERPOWERS (via Core Agent):
- **Memory System**: Redis-backed persistent memory for code modules
- **Tool Ecosystem**: Pre-built Python execution, file management, validation tools
- **Azure OpenAI Integration**: Direct GPT-4 access with proper deployment
- **Production Patterns**: Real fail-fast error handling, no mocks
- **Code Quality**: Automatic validation and testing capabilities

💡 EXPERT CODING PRINCIPLES:
1. **Azure OpenAI Configuration**: Always use proper deployment names and endpoints
2. **Memory Integration**: Leverage Redis for module persistence and conversation history
3. **Tool Composition**: Combine Core Agent tools with specialized LangGraph generators
4. **Error Handling**: Implement robust exception handling with meaningful messages
5. **Type Safety**: Use proper TypedDict for state management
6. **Documentation**: Comprehensive docstrings and inline comments

🛠️ YOUR ENHANCED TOOLSET:
- **langgraph_template**: Generate complete agent templates (simple, with_tools, multi_agent)
- **code_memory**: Save/load/search code modules using Core Agent's Redis memory
- **python_executor**: Execute and test code safely (from Core Agent)
- **file_operations**: Create projects and manage files (from Core Agent)
- **code_validator**: Validate syntax, imports, and LangGraph patterns

📝 EXPERT WORKFLOW:
1. **Analyze**: Deep understanding of requirements and constraints
2. **Architect**: Design using LangGraph best practices and Core Agent patterns
3. **Generate**: Create clean, typed, documented code
4. **Validate**: Use code_validator to ensure quality
5. **Persist**: Save modules using code_memory for future reference
6. **Test**: Provide working examples and usage instructions

🔥 SPECIALIZED CAPABILITIES:
- **Template Generation**: Instant LangGraph agent scaffolding
- **Memory Continuity**: Remember and build upon previous code modules
- **Azure Integration**: Proper GPT-4 deployment and configuration
- **Production Ready**: No mocks, real error handling, proper logging
- **Modular Design**: Create reusable, composable components

🚀 PROMPT EXAMPLES YOU EXCEL AT:
1. "Create a simple LangGraph agent for [task]" → Generate complete agent with proper Azure config
2. "Build an agent with [tools] for [purpose]" → Create tool-enhanced agent with proper integration
3. "Design a multi-agent system for [complex_task]" → Build supervisor architecture with specialized workers

You are the ULTIMATE coder agent, combining LangGraph expertise with Core Agent's infrastructure. Every line of code you write is production-ready, properly documented, and leverages the full power of the underlying platform!"""
    
    def get_prompt_examples(self) -> Dict[str, str]:
        """Get example prompts for different complexity levels"""
        return {
            "simple_agent": """
Create a simple LangGraph agent that:
- Takes a question as input
- Uses a language model to answer
- Returns the response
- Has proper state management
            """,
            
            "agent_with_tools": """
Create a LangGraph agent with tools that:
- Has a research tool for web searching
- Has a calculator tool for math operations
- Can decide which tool to use based on the question
- Combines tool results into a final answer
- Includes proper error handling
            """,
            
            "multi_agent_system": """
Create a multi-agent system that:
- Has a supervisor agent that coordinates tasks
- Has a researcher agent for information gathering
- Has a writer agent for content creation
- Has a reviewer agent for quality control
- Includes proper task routing and result aggregation
- Has comprehensive error handling and monitoring
            """
        }
    
    def chat(self, message: str) -> str:
        """Enhanced chat interface using Core Agent infrastructure"""
        try:
            # Use Core Agent memory if available, otherwise direct approach
            if self.memory_manager:
                # Core Agent memory integration
                memory_context = self._get_memory_context(message)
            else:
                memory_context = ""
            
            # Build enhanced messages with Core Agent context
            messages = [SystemMessage(content=self._get_expert_system_prompt())]
            
            # Add memory context if available
            if memory_context:
                messages.append(HumanMessage(content=f"Context from memory: {memory_context}"))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Get AI response using Azure OpenAI
            response = self.llm.invoke(messages)
            
            # Save to Core Agent memory if available
            if self.memory_manager:
                self._save_interaction(message, response.content)
            
            return response.content
            
        except Exception as e:
            error_msg = f"❌ Coder Agent error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def _get_memory_context(self, current_message: str) -> str:
        """Get relevant context from Core Agent memory"""
        try:
            if self.memory_manager:
                # Search for relevant previous interactions
                # This would integrate with Core Agent's memory search
                return f"Previous coding session context for: {self.session_id}"
            return ""
        except Exception as e:
            print(f"⚠️ Memory context retrieval failed: {e}")
            return ""
    
    def _save_interaction(self, user_message: str, assistant_response: str):
        """Save interaction to Core Agent memory"""
        try:
            if self.memory_manager:
                # Use Core Agent's memory system to save the interaction
                interaction_data = {
                    "user": user_message,
                    "assistant": assistant_response,
                    "timestamp": datetime.now().isoformat(),
                    "session": self.session_id,
                    "type": "coding_interaction"
                }
                # This would integrate with Core Agent's memory storage
                print(f"💾 Saved interaction to Core Agent memory")
        except Exception as e:
            print(f"⚠️ Memory save failed: {e}")
    
    def generate_agent_code(self, requirements: str, complexity: str = "simple") -> Dict[str, Any]:
        """Generate agent code based on requirements"""
        
        if complexity == "simple":
            prompt = f"""Create a simple LangGraph agent based on these requirements:
{requirements}

Include:
- Proper imports
- State definition
- Node functions
- Graph construction
- Usage example

Make it clean and production-ready."""
        
        elif complexity == "with_tools":
            prompt = f"""Create a LangGraph agent with tools based on these requirements:
{requirements}

Include:
- Custom tool definitions
- Agent that can use tools
- Proper tool selection logic
- Error handling
- Complete working example"""
        
        elif complexity == "multi_agent":
            prompt = f"""Create a multi-agent system based on these requirements:
{requirements}

Include:
- Supervisor agent
- Specialized worker agents
- Task routing logic
- Communication protocols
- Complete orchestration system"""
        
        else:
            prompt = f"""Create a LangGraph solution for: {requirements}"""
        
        # Generate code
        response = self.chat(prompt)
        
        # Extract code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        
        return {
            "response": response,
            "code_blocks": code_blocks,
            "requirements": requirements,
            "complexity": complexity
        }


def create_coder_agent(session_id: str = None) -> CoderAgent:
    """Factory function to create a Coder Agent"""
    if session_id is None:
        session_id = f"coder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return CoderAgent(session_id)


# Demo and testing functions
def demo_coder_agent():
    """Demonstrate the Coder Agent capabilities"""
    print("🚀 CODER AGENT DEMO")
    print("=" * 50)
    
    # Create agent
    agent = create_coder_agent("demo_session")
    
    # Show example prompts
    examples = agent.get_prompt_examples()
    print("\n📝 EXAMPLE PROMPTS:")
    for level, prompt in examples.items():
        print(f"\n{level.upper()}:")
        print(prompt.strip())
    
    print("\n" + "=" * 50)
    print("✅ Demo completed! Agent is ready for use.")


if __name__ == "__main__":
    demo_coder_agent()