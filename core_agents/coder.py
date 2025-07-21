#!/usr/bin/env python3
"""
Elite Coder Agent - Using Core Agent Infrastructure
==================================================

Professional AI agent specialized in creating world-class LangGraph agents,
tools, and multi-agent systems. Built on Core Agent's powerful infrastructure
including MemoryManager, ToolManager, and AgentConfig.
"""

import os
import sys
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add workspace to path for imports
sys.path.insert(0, '/workspace')

# Core Agent Infrastructure
from core.config import AgentConfig
from core.managers import MemoryManager
from core.tools import create_python_coding_tools
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field


class CoderConfig:
    """Elite Coder Agent Configuration - Using Real Azure OpenAI"""
    
    # Azure OpenAI Configuration (exactly as specified by user)
    AZURE_OPENAI_ENDPOINT = "https://oai-202-fbeta-dev.openai.azure.com/"
    OPENAI_API_KEY = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    
    # Elite Coder Parameters
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000


class AgentGeneratorInput(BaseModel):
    """Input schema for agent generation"""
    template_type: str = Field(description="Type: simple, with_tools, multi_agent")
    agent_name: str = Field(description="Name for the agent")
    purpose: str = Field(description="What the agent should do")
    tools_needed: List[str] = Field(default=[], description="List of tools if needed")


class CleanAgentGeneratorTool(BaseTool):
    """Clean tool that uses prompts instead of string templates"""
    
    name: str = "clean_agent_generator"
    description: str = """Generate LangGraph agents using intelligent prompts.
    Much cleaner than string templates - lets the LLM generate proper code."""
    args_schema: type[BaseModel] = AgentGeneratorInput
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _run(self, template_type: str, agent_name: str, purpose: str, tools_needed: List[str] = []) -> str:
        """Generate elite LangGraph template based on type"""
        
        if template_type == "simple":
            return self._generate_simple_agent(agent_name, purpose)
        elif template_type == "with_tools":
            return self._generate_agent_with_tools(agent_name, purpose, tools_needed)
        elif template_type == "multi_agent":
            return self._generate_multi_agent_system(agent_name, purpose, tools_needed)
        else:
            return f"❌ Unknown template type: {template_type}. Use: simple, with_tools, multi_agent"
    
    def _generate_simple_agent(self, agent_name: str, purpose: str) -> str:
        """Generate agent using LLM - no templates!"""
        
        prompt = f"""Create a complete, production-ready LangGraph agent from scratch.

Requirements:
- Agent Name: {agent_name}
- Purpose: {purpose}
- Use Core Agent infrastructure (AgentConfig, MemoryManager)
- Include Azure OpenAI configuration
- Use LangGraph StateGraph
- Add proper error handling and logging
- Make it executable with examples

Generate ONLY the Python code. No explanations."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_agent_with_tools(self, agent_name: str, purpose: str, tools_needed: List[str]) -> str:
        """Generate agent with tools using LLM - no templates!"""
        
        tools_str = ", ".join(tools_needed)
        
        prompt = f"""Create a complete, production-ready LangGraph agent with tools from scratch.

Requirements:
- Agent Name: {agent_name}
- Purpose: {purpose}
- Tools Needed: {tools_str}
- Use Core Agent infrastructure
- Include tool selection logic
- Add tool execution with error handling
- Include response synthesis
- Use LangGraph StateGraph with proper workflow
- Add Azure OpenAI integration
- Make it executable with examples

Generate ONLY the Python code. No explanations."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_multi_agent_system(self, agent_name: str, purpose: str, agent_types: List[str]) -> str:
        """Generate multi-agent system using LLM - no templates!"""
        
        prompt = f"""Create a complete, production-ready multi-agent supervisor system from scratch.

Requirements:
- System Name: {agent_name}
- Purpose: {purpose}
- Include supervisor agent for routing
- Include specialized worker agents (researcher, analyzer, creator, reviewer)
- Use Core Agent infrastructure
- Use LangGraph StateGraph with conditional routing
- Add Azure OpenAI integration
- Include proper state management
- Add error handling and recovery
- Make it executable with examples

Generate ONLY the Python code. No explanations."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class CodeMemoryInput(BaseModel):
    """Input for elite code memory management"""
    action: str = Field(description="Action: save, load, list, search, tag")
    module_name: str = Field(default="", description="Name of the module")
    code: str = Field(default="", description="Code content to save")
    description: str = Field(default="", description="Module description")
    tags: List[str] = Field(default=[], description="Tags for categorization")
    search_query: str = Field(default="", description="Search query for finding modules")


class CodeMemoryTool(BaseTool):
    """Elite tool for Redis-based code memory management using Core Agent infrastructure"""
    
    name: str = "code_memory"
    description: str = """Elite Redis memory management for code modules.
    Actions: save (store with tags), load (retrieve), list (show all), search (find by name/tags), tag (update tags)
    Leverages Core Agent's Redis infrastructure for persistence."""
    args_schema: type[BaseModel] = CodeMemoryInput
    
    def __init__(self, memory_manager: MemoryManager):
        super().__init__()
        self._memory_manager = memory_manager
    
    def _run(self, action: str, module_name: str = "", code: str = "", 
            description: str = "", tags: List[str] = [], search_query: str = "") -> str:
        """Execute elite memory management with Core Agent Redis"""
        try:
            if action == "save":
                if not module_name or not code:
                    return "❌ Save requires module_name and code parameters"
                
                # Use Core Agent's memory infrastructure
                memory_key = f"elite_coder:module:{module_name}"
                module_data = {
                    "code": code,
                    "description": description,
                    "tags": tags,
                    "timestamp": datetime.now().isoformat(),
                    "type": "langgraph_agent"
                }
                
                # Save to Core Agent's Redis memory
                success = self._memory_manager.save_memory(memory_key, json.dumps(module_data))
                
                if success:
                    # Add to module index
                    self._memory_manager.save_memory("elite_coder:modules", module_name, append=True)
                    
                    # Add tag indices
                    for tag in tags:
                        self._memory_manager.save_memory(f"elite_coder:tag:{tag}", module_name, append=True)
                    
                    return f"✅ Elite module '{module_name}' saved with tags: {tags}"
                else:
                    return f"❌ Failed to save module '{module_name}' to Core Agent memory"
            
            elif action == "load":
                if not module_name:
                    return "❌ Load requires module_name parameter"
                
                memory_key = f"elite_coder:module:{module_name}"
                module_json = self._memory_manager.get_memory(memory_key)
                
                if module_json:
                    module_data = json.loads(module_json)
                    return f"""✅ Elite module '{module_name}' loaded from Core Agent Redis:

📝 Description: {module_data.get('description', 'No description')}
🏷️ Tags: {', '.join(module_data.get('tags', []))}
📅 Created: {module_data.get('timestamp', 'Unknown')}
🎯 Type: {module_data.get('type', 'Unknown')}

💻 Code:
```python
{module_data['code']}
```"""
                else:
                    return f"❌ Module '{module_name}' not found in Core Agent memory"
            
            elif action == "list":
                modules_data = self._memory_manager.get_memory("elite_coder:modules")
                if modules_data:
                    if isinstance(modules_data, str):
                        modules = [modules_data] if modules_data else []
                    else:
                        modules = modules_data
                    
                    return f"📚 Elite modules in Core Agent Redis ({len(modules)}):\\n" + "\\n".join(f"  🔹 {m}" for m in modules)
                else:
                    return "📚 No elite modules found in Core Agent memory"
            
            elif action == "search":
                query = search_query or module_name
                if not query:
                    return "❌ Search requires search_query or module_name"
                
                # Search implementation using Core Agent memory
                found_modules = []
                
                # Get all modules and search in names
                modules_data = self._memory_manager.get_memory("elite_coder:modules")
                if modules_data:
                    modules = [modules_data] if isinstance(modules_data, str) else modules_data
                    found_modules.extend([m for m in modules if query.lower() in m.lower()])
                
                # Search by tags
                tag_modules = self._memory_manager.get_memory(f"elite_coder:tag:{query.lower()}")
                if tag_modules:
                    if isinstance(tag_modules, str):
                        found_modules.append(tag_modules)
                    else:
                        found_modules.extend(tag_modules)
                
                unique_modules = list(set(found_modules))
                
                if unique_modules:
                    return f"🔍 Found {len(unique_modules)} elite modules matching '{query}':\\n" + "\\n".join(f"  🎯 {m}" for m in unique_modules)
                else:
                    return f"🔍 No elite modules found matching '{query}'"
            
            else:
                return f"❌ Unknown action '{action}'. Use: save, load, list, search"
                
        except Exception as e:
            return f"❌ Core Agent memory operation error: {str(e)}"


class EliteCoderAgent:
    """Elite Coder Agent leveraging Core Agent Infrastructure"""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        
        print(f"🚀 Initializing Elite Coder Agent with Core Agent Infrastructure")
        
        # Create Core Agent configuration for Coder Agent
        self.config = AgentConfig(
            name="EliteCoderAgent",
            model=self._create_azure_openai_model(),
            enable_memory=True,
            memory_backend="redis",
            redis_url="redis://localhost:6379/3",  # DB 3 for elite coder
            memory_types=["short_term", "long_term"],
            tools=self._create_elite_tools(),
            system_prompt=self._get_elite_system_prompt(),
            max_tokens=CoderConfig.MAX_TOKENS
        )
        
        # Initialize Core Agent managers
        try:
            self.memory_manager = MemoryManager(self.config)
            print(f"✅ Core Agent infrastructure initialized successfully")
        except Exception as e:
            print(f"⚠️ Core Agent managers initialization warning: {e}")
            self.memory_manager = None
        
        # Direct Azure OpenAI for immediate use
        self.llm = self._create_azure_openai_model()
        
        print(f"🤖 Elite Coder Agent ready with session: {session_id}")
        print(f"🔧 Available tools: {[tool.name for tool in self.config.tools]}")
        print(f"💾 Memory backend: {self.config.memory_backend}")
        print(f"🧠 Model: Azure OpenAI {CoderConfig.GPT4_MODEL_NAME}")
    
    def _create_azure_openai_model(self):
        """Create Azure OpenAI model with exact user specifications"""
        return AzureChatOpenAI(
            azure_endpoint=CoderConfig.AZURE_OPENAI_ENDPOINT,
            api_key=CoderConfig.OPENAI_API_KEY,
            api_version=CoderConfig.OPENAI_API_VERSION,
            model=CoderConfig.GPT4_MODEL_NAME,
            deployment_name=CoderConfig.GPT4_DEPLOYMENT_NAME,
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
    
    def _create_elite_tools(self):
        """Create elite tools combining Core Agent tools with specialized ones"""
        tools = []
        
        # Get Core Agent's Python coding tools
        try:
            core_tools = create_python_coding_tools()
            tools.extend(core_tools)
            print(f"✅ Added {len(core_tools)} Core Agent tools")
        except Exception as e:
            print(f"⚠️ Core Agent tools not available: {e}")
        
        # Add specialized Elite Coder tools
        elite_tools = [
            CleanAgentGeneratorTool(self.llm)
        ]
        
        # Add memory tool if memory manager is available
        if hasattr(self, 'memory_manager') and self.memory_manager:
            elite_tools.append(CodeMemoryTool(self.memory_manager))
        
        tools.extend(elite_tools)
        
        return tools
    
    def _get_elite_system_prompt(self) -> str:
        """Clean system prompt for intelligent code generation"""
        return """You are an ELITE CODER AGENT specializing in LangGraph agent development.

🎯 YOUR MISSION:
Create exceptional, production-ready LangGraph agents using intelligent code generation instead of templates.

🏗️ APPROACH:
- **Intelligent Generation**: Use your expertise to create code from scratch
- **No Templates**: Never use pre-written code templates - generate everything fresh
- **Core Agent Integration**: Leverage Core Agent's infrastructure properly
- **Quality Focus**: Production-ready, maintainable, well-documented code

🧠 SPECIALIZATIONS:
1. **Simple Agents**: Clean, focused single-purpose LangGraph agents
2. **Tool-Enhanced Agents**: Agents with custom tool integration
3. **Multi-Agent Systems**: Supervisor-worker architectures

🛠️ YOUR TOOLS:
- **langgraph_generator**: Generate agents using intelligent prompts (no templates!)
- **code_memory**: Save/load code modules with intelligent tagging
- Core Agent tools for execution and testing

📝 PRINCIPLES:
- **Fresh Code**: Generate everything from scratch based on requirements
- **Core Agent Integration**: Use AgentConfig, MemoryManager, tools properly
- **Quality**: Clean, readable, maintainable code
- **No Templates**: Let your intelligence create the best solution

🚀 WORKFLOW:
1. Understand requirements completely
2. Design architecture using Core Agent patterns
3. Generate clean, fresh code (no templates!)
4. Include proper Azure OpenAI integration
5. Add error handling and logging
6. Save to memory with intelligent tags

You create intelligent, original code that developers actually want to use!"""
    
    def get_example_prompts(self) -> Dict[str, str]:
        """Get example prompts showcasing elite capabilities"""
        return {
            "simple_agent": """Create a simple LangGraph agent for Python code analysis and optimization. 
The agent should use Core Agent's infrastructure, Azure OpenAI GPT-4, and save results to Redis memory.""",
            
            "agent_with_tools": """Build a LangGraph agent with tools for automated testing. 
Include tools for: test generation, code execution, and result analysis. 
Use Core Agent's tool ecosystem and memory system for persistence.""",
            
            "multi_agent_system": """Design a multi-agent system for full-stack development with:
- A backend specialist agent for API development
- A frontend specialist agent for UI creation  
- A testing specialist agent for QA automation
- A DevOps specialist agent for deployment
- A supervisor that coordinates the entire development workflow
Include Core Agent memory integration and proper task routing."""
        }
    
    def chat(self, message: str) -> str:
        """Elite chat interface with Core Agent memory integration"""
        try:
            # Get conversation context from Core Agent memory
            memory_context = ""
            if self.memory_manager:
                try:
                    memory_key = f"elite_coder:conversation:{self.session_id}"
                    context_data = self.memory_manager.get_memory(memory_key)
                    if context_data:
                        memory_context = f"Previous conversation context: {context_data[:500]}..."
                except Exception as e:
                    print(f"⚠️ Memory context retrieval warning: {e}")
            
            # Build elite conversation
            messages = [SystemMessage(content=self._get_elite_system_prompt())]
            
            # Add memory context if available
            if memory_context:
                messages.append(HumanMessage(content=memory_context))
            
            # Add current message
            messages.append(HumanMessage(content=message))
            
            # Get elite AI response
            response = self.llm.invoke(messages)
            
            # Save interaction to Core Agent memory
            if self.memory_manager:
                try:
                    conversation_entry = {
                        "user": message,
                        "agent": response.content,
                        "timestamp": datetime.now().isoformat()
                    }
                    memory_key = f"elite_coder:conversation:{self.session_id}"
                    self.memory_manager.save_memory(memory_key, json.dumps(conversation_entry))
                except Exception as e:
                    print(f"⚠️ Memory save warning: {e}")
            
            return response.content
            
        except Exception as e:
            error_msg = f"❌ Elite Coder Agent error: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_complete_agent(self, template_type: str, agent_name: str, purpose: str, 
                              tools_needed: List[str] = []) -> Dict[str, Any]:
        """Generate complete agent with Core Agent integration"""
        try:
            print(f"🎯 Generating elite {template_type} agent: {agent_name}")
            
            # Generate the agent code
            template_tool = CleanAgentGeneratorTool(self.llm)
            agent_code = template_tool._run(template_type, agent_name, purpose, tools_needed)
            
            # Save to Core Agent memory with intelligent tags
            tags = [template_type, "langgraph", "elite_agent", "core_agent_integrated"]
            if tools_needed:
                tags.extend(["with_tools"] + tools_needed)
            
            save_result = "Memory not available"
            if self.memory_manager:
                try:
                    memory_tool = CodeMemoryTool(self.memory_manager)
                    save_result = memory_tool._run("save", f"{agent_name.lower()}_agent", 
                                                 agent_code, purpose, tags)
                except Exception as e:
                    save_result = f"Memory save warning: {str(e)}"
            
            # Test the code using Core Agent tools
            test_result = "Testing not available"
            try:
                # Find Python executor tool from Core Agent tools
                python_tool = None
                for tool in self.config.tools:
                    if hasattr(tool, 'name') and 'python' in tool.name.lower():
                        python_tool = tool
                        break
                
                if python_tool:
                    test_result = python_tool._run(agent_code, timeout=10)
                else:
                    test_result = "✅ Code generated (Python executor not available for testing)"
            except Exception as e:
                test_result = f"Testing warning: {str(e)}"
            
            return {
                "agent_code": agent_code,
                "save_result": save_result,
                "test_result": test_result,
                "template_type": template_type,
                "agent_name": agent_name,
                "purpose": purpose,
                "tools": tools_needed,
                "tags": tags,
                "core_agent_integrated": True
            }
            
        except Exception as e:
            return {
                "error": f"Elite agent generation failed: {str(e)}",
                "agent_code": "",
                "save_result": "",
                "test_result": "",
                "core_agent_integrated": False
            }


def create_elite_coder_agent(session_id: str = None) -> EliteCoderAgent:
    """Factory function to create Elite Coder Agent with Core Agent infrastructure"""
    if session_id is None:
        session_id = f"elite_coder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return EliteCoderAgent(session_id)


def demo_elite_coder():
    """Demonstrate Elite Coder Agent with Core Agent integration"""
    print("🚀 ELITE CODER AGENT - CORE AGENT INTEGRATION DEMO")
    print("=" * 100)
    
    try:
        # Create elite agent
        agent = create_elite_coder_agent("demo_session")
        
        # Show example prompts
        examples = agent.get_example_prompts()
        print("\\n📝 ELITE EXAMPLE PROMPTS:")
        for level, prompt in examples.items():
            print(f"\\n🎯 {level.upper().replace('_', ' ')}:")
            print(f"   {prompt}")
        
        # Test basic functionality
        print("\\n🧪 TESTING BASIC FUNCTIONALITY:")
        test_response = agent.chat("Create a simple LangGraph agent for data processing")
        print(f"✅ Chat Response: {test_response[:200]}...")
        
        # Test agent generation
        print("\\n🎯 TESTING AGENT GENERATION:")
        result = agent.generate_complete_agent(
            template_type="simple",
            agent_name="DataProcessor", 
            purpose="Process and analyze data efficiently",
            tools_needed=[]
        )
        
        print(f"✅ Agent Generated: {result['agent_name']}")
        print(f"💾 Memory Save: {result['save_result']}")
        print(f"🧪 Test Result: {result['test_result'][:100]}...")
        
        print("\\n" + "=" * 100)
        print("✅ Elite Coder Agent with Core Agent integration ready!")
        print("💎 Powered by: Azure OpenAI GPT-4 + Core Agent Infrastructure + Redis Memory")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    demo_elite_coder()