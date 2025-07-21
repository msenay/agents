#!/usr/bin/env python3
"""
Professional Coder Agent
========================

An advanced AI agent specialized in writing high-quality LangGraph agents,
tools, and multi-agent systems. Uses Azure OpenAI and Redis for memory.
"""

import os
import sys
import json
import re
import tempfile
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import redis
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# Add workspace to path
sys.path.insert(0, '/workspace')

class CoderConfig:
    """Configuration for Coder Agent"""
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_ENDPOINT = "https://oai-202-fbeta-dev.openai.azure.com/"
    OPENAI_API_KEY = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
    OPENAI_API_VERSION = "2023-12-01-preview"
    
    # Redis Configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/2")  # DB 2 for coder
    
    # Model Configuration
    MODEL_NAME = "gpt-4"
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000


class CoderMemory:
    """Redis-based memory system for Coder Agent"""
    
    def __init__(self, redis_url: str = CoderConfig.REDIS_URL):
        try:
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            print(f"✅ Connected to Redis: {redis_url}")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to Redis: {e}")
    
    def save_module(self, module_name: str, code: str, description: str = "") -> bool:
        """Save a code module to memory"""
        try:
            module_data = {
                "code": code,
                "description": description,
                "timestamp": datetime.now().isoformat(),
                "type": "module"
            }
            
            key = f"coder:module:{module_name}"
            self.redis_client.setex(key, 86400 * 7, json.dumps(module_data))  # 7 days TTL
            
            # Also add to module list
            self.redis_client.sadd("coder:modules", module_name)
            return True
        except Exception as e:
            print(f"❌ Failed to save module {module_name}: {e}")
            return False
    
    def get_module(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a code module from memory"""
        try:
            key = f"coder:module:{module_name}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            print(f"❌ Failed to get module {module_name}: {e}")
            return None
    
    def list_modules(self) -> List[str]:
        """List all saved modules"""
        try:
            modules = self.redis_client.smembers("coder:modules")
            return [m.decode() for m in modules] if modules else []
        except Exception as e:
            print(f"❌ Failed to list modules: {e}")
            return []
    
    def save_conversation(self, session_id: str, message: str, role: str) -> bool:
        """Save conversation message"""
        try:
            key = f"coder:conversation:{session_id}"
            conversation_entry = {
                "message": message,
                "role": role,
                "timestamp": datetime.now().isoformat()
            }
            
            self.redis_client.lpush(key, json.dumps(conversation_entry))
            self.redis_client.expire(key, 86400)  # 24 hours TTL
            return True
        except Exception as e:
            print(f"❌ Failed to save conversation: {e}")
            return False
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        try:
            key = f"coder:conversation:{session_id}"
            messages = self.redis_client.lrange(key, 0, limit - 1)
            return [json.loads(msg) for msg in messages] if messages else []
        except Exception as e:
            print(f"❌ Failed to get conversation history: {e}")
            return []


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
    """Professional Coder Agent specialized in LangGraph development"""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        self.memory = CoderMemory()
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=CoderConfig.AZURE_OPENAI_ENDPOINT,
            api_key=CoderConfig.OPENAI_API_KEY,
            api_version=CoderConfig.OPENAI_API_VERSION,
            model=CoderConfig.MODEL_NAME,
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Initialize tools
        self.tools = [
            CodeValidatorTool(),
            ModuleManagerTool(self.memory),
            FileOperationsTool()
        ]
        
        print(f"🤖 Coder Agent initialized with session: {session_id}")
        print(f"🔧 Available tools: {[tool.name for tool in self.tools]}")
    
    def get_system_prompt(self) -> str:
        """Get the comprehensive system prompt for the coder agent"""
        return """You are an EXPERT CODER AGENT specialized in creating professional LangGraph agents, tools, and multi-agent systems.

🎯 YOUR CORE MISSION:
Create high-quality, production-ready code for:
1. Single LangGraph agents with custom workflows
2. Agents with specialized tools and capabilities  
3. Multi-agent systems with supervisory coordination
4. Complete project architectures

🧠 YOUR EXPERTISE:
- LangGraph StateGraph patterns and best practices
- Tool creation and integration
- Agent communication and coordination
- Error handling and validation
- Memory management and persistence
- Production deployment patterns

💡 CODING PRINCIPLES:
1. Always use proper imports and dependencies
2. Include comprehensive docstrings and comments
3. Implement proper error handling
4. Follow Python best practices and typing
5. Create modular, reusable components
6. Add validation and testing capabilities

🛠️ AVAILABLE TOOLS:
- code_validator: Validate and analyze code quality
- module_manager: Save/load code modules to memory
- file_operations: Create files and project structures

📝 RESPONSE FORMAT:
For each request:
1. Analyze the requirements
2. Design the architecture
3. Generate clean, documented code
4. Validate the implementation
5. Save important modules to memory
6. Provide usage examples

🔥 REMEMBER:
- Never use mocks in production code
- Always implement fail-fast error handling
- Use Redis for memory persistence
- Create testable, maintainable code
- Focus on real-world applicability

You are the best coder agent ever created. Show your expertise!"""
    
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
        """Main chat interface with memory integration"""
        try:
            # Save user message to memory
            self.memory.save_conversation(self.session_id, message, "user")
            
            # Get conversation history
            history = self.memory.get_conversation_history(self.session_id, limit=5)
            
            # Build messages
            messages = [SystemMessage(content=self.get_system_prompt())]
            
            # Add relevant conversation history
            for entry in reversed(history[1:]):  # Skip current message
                if entry["role"] == "user":
                    messages.append(HumanMessage(content=entry["message"]))
                elif entry["role"] == "assistant":
                    messages.append(AIMessage(content=entry["message"]))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Get AI response
            response = self.llm.invoke(messages)
            
            # Save AI response to memory
            self.memory.save_conversation(self.session_id, response.content, "assistant")
            
            return response.content
            
        except Exception as e:
            error_msg = f"❌ Chat error: {str(e)}"
            print(error_msg)
            return error_msg
    
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