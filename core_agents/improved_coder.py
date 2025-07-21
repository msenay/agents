#!/usr/bin/env python3
"""
Improved Elite Coder Agent - Clean Prompt-Based Approach
========================================================

Professional AI agent that uses intelligent prompts instead of long string templates.
Built on Core Agent's infrastructure with clean, maintainable code generation.
"""

import os
import json
import tempfile
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.config import AgentConfig
from core.managers import MemoryManager
from core.tools import create_python_coding_tools
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field


class CoderConfig:
    """Clean configuration"""
    AZURE_OPENAI_ENDPOINT = "https://oai-202-fbeta-dev.openai.azure.com/"
    OPENAI_API_KEY = "BDfLqbP0vVCTuRkXtE4Zy9mK7neLrJlHXlISgqJxVNTg2ca71EI5JQQJ99BDACfhMk5XJ3w3AAABACOGgIx4"
    OPENAI_API_VERSION = "2023-12-01-preview"
    GPT4_MODEL_NAME = "gpt-4"
    GPT4_DEPLOYMENT_NAME = "gpt4"
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000


class AgentGeneratorInput(BaseModel):
    """Input for agent generation"""
    agent_type: str = Field(description="Type: simple, with_tools, multi_agent")
    agent_name: str = Field(description="Name for the agent")
    purpose: str = Field(description="What the agent should do")
    requirements: str = Field(default="", description="Specific requirements")
    tools_needed: List[str] = Field(default=[], description="Tools if needed")


class CleanAgentGeneratorTool(BaseTool):
    """Clean tool that uses prompts instead of string templates"""
    
    name: str = "clean_agent_generator"
    description: str = """Generate LangGraph agents using intelligent prompts.
    Much cleaner than string templates - lets the LLM generate proper code."""
    args_schema: type[BaseModel] = AgentGeneratorInput
    
    def __init__(self, llm):
        super().__init__()
        self.llm = llm
    
    def _run(self, agent_type: str, agent_name: str, purpose: str, 
             requirements: str = "", tools_needed: List[str] = []) -> str:
        """Generate agent using intelligent prompts"""
        
        if agent_type == "simple":
            return self._generate_simple_agent(agent_name, purpose, requirements)
        elif agent_type == "with_tools":
            return self._generate_agent_with_tools(agent_name, purpose, requirements, tools_needed)
        elif agent_type == "multi_agent":
            return self._generate_multi_agent_system(agent_name, purpose, requirements)
        else:
            return f"❌ Unknown agent type: {agent_type}"
    
    def _generate_simple_agent(self, agent_name: str, purpose: str, requirements: str) -> str:
        """Generate simple agent with clean prompt"""
        
        prompt = f"""Create a production-ready LangGraph agent with these specifications:

**Agent Details:**
- Name: {agent_name}
- Purpose: {purpose}
- Requirements: {requirements}

**Technical Requirements:**
1. Use LangGraph StateGraph for workflow
2. Include Azure OpenAI configuration exactly like this:
   ```python
   AzureChatOpenAI(
       azure_endpoint="{CoderConfig.AZURE_OPENAI_ENDPOINT}",
       api_key="{CoderConfig.OPENAI_API_KEY}",
       api_version="{CoderConfig.OPENAI_API_VERSION}",
       model="{CoderConfig.GPT4_MODEL_NAME}",
       deployment_name="{CoderConfig.GPT4_DEPLOYMENT_NAME}",
       temperature={CoderConfig.TEMPERATURE},
       max_tokens={CoderConfig.MAX_TOKENS}
   )
   ```
3. Create a TypedDict state class
4. Include comprehensive error handling
5. Add logging and proper docstrings
6. Create a main execution example
7. Use professional Python code style

**Code Structure:**
- Clean imports
- State definition
- Processing functions
- Graph creation
- Main execution example

Generate ONLY the Python code, no explanations. Make it production-ready."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_agent_with_tools(self, agent_name: str, purpose: str, 
                                 requirements: str, tools_needed: List[str]) -> str:
        """Generate agent with tools using clean prompt"""
        
        tools_str = ", ".join(tools_needed)
        
        prompt = f"""Create a production-ready LangGraph agent with tools:

**Agent Details:**
- Name: {agent_name}
- Purpose: {purpose}
- Requirements: {requirements}
- Tools Needed: {tools_str}

**Technical Requirements:**
1. Use LangGraph StateGraph with tool integration
2. Include Azure OpenAI configuration (use the config from CoderConfig class)
3. Create custom tools for: {tools_str}
4. Implement tool selection logic
5. Add tool execution with error handling
6. Include response synthesis
7. Add comprehensive logging
8. Create proper state management

**Architecture:**
- Tool selection node
- Tool execution node  
- Response synthesis node
- Proper error handling throughout

**Code Quality:**
- Clean, modular functions
- Type hints and docstrings
- Professional error handling
- Production-ready code

Generate ONLY the Python code. Make it enterprise-grade."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    
    def _generate_multi_agent_system(self, agent_name: str, purpose: str, requirements: str) -> str:
        """Generate multi-agent system with clean prompt"""
        
        prompt = f"""Create a production-ready multi-agent supervisor system:

**System Details:**
- Name: {agent_name}
- Purpose: {purpose}
- Requirements: {requirements}

**Architecture Requirements:**
1. Supervisor agent that routes tasks
2. Specialized worker agents: researcher, analyzer, creator, reviewer
3. Final aggregation/synthesis
4. Use LangGraph conditional routing
5. Azure OpenAI integration for all agents
6. Comprehensive state management
7. Error handling and recovery

**Technical Implementation:**
- StateGraph with conditional edges
- Supervisor routing logic
- Worker agent implementations
- Result aggregation
- Professional logging
- Type hints and documentation

**Code Quality:**
- Clean, maintainable code
- Modular agent functions
- Proper error handling
- Production-ready implementation

Generate ONLY the Python code. Make it enterprise-quality."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content


class ImprovedCoderAgent:
    """Improved Coder Agent using clean prompt-based approach"""
    
    def __init__(self, session_id: str = "default"):
        self.session_id = session_id
        
        print(f"🚀 Initializing Improved Coder Agent (Clean Approach)")
        
        # Create model
        self.llm = self._create_azure_openai_model()
        
        # Create tools
        self.agent_generator = CleanAgentGeneratorTool(self.llm)
        
        # Core Agent integration
        try:
            self.config = AgentConfig(
                name="ImprovedCoderAgent",
                model=self.llm,
                enable_memory=True,
                memory_backend="inmemory",  # Simplified for demo
                tools=[self.agent_generator],
                system_prompt=self._get_clean_system_prompt()
            )
            self.memory_manager = MemoryManager(self.config)
            print(f"✅ Core Agent integration successful")
        except Exception as e:
            print(f"⚠️ Core Agent integration warning: {e}")
            self.memory_manager = None
        
        print(f"🎯 Ready with clean prompt-based generation!")
    
    def _create_azure_openai_model(self):
        """Create Azure OpenAI model"""
        return AzureChatOpenAI(
            azure_endpoint=CoderConfig.AZURE_OPENAI_ENDPOINT,
            api_key=CoderConfig.OPENAI_API_KEY,
            api_version=CoderConfig.OPENAI_API_VERSION,
            model=CoderConfig.GPT4_MODEL_NAME,
            deployment_name=CoderConfig.GPT4_DEPLOYMENT_NAME,
            temperature=CoderConfig.TEMPERATURE,
            max_tokens=CoderConfig.MAX_TOKENS
        )
    
    def _get_clean_system_prompt(self) -> str:
        """Clean, focused system prompt"""
        return """You are an ELITE CODER AGENT specializing in LangGraph agent development.

🎯 CORE MISSION:
Create exceptional, production-ready LangGraph agents using clean, intelligent code generation.

🏗️ APPROACH:
- Use intelligent prompts instead of string templates
- Let the LLM generate clean, proper code
- Focus on code quality and maintainability
- Leverage Core Agent infrastructure

💡 SPECIALIZATIONS:
1. **Simple Agents**: Clean, focused single-purpose agents
2. **Tool-Enhanced Agents**: Agents with custom tool integration
3. **Multi-Agent Systems**: Supervisor-worker architectures

🛠️ TOOLS:
- **clean_agent_generator**: Generate agents using intelligent prompts
- Core Agent tools for execution and testing

📝 PRINCIPLES:
- Quality over complexity
- Clean, readable code
- Proper error handling
- Production-ready implementations
- No unnecessary string templates

You create clean, maintainable code that developers actually want to use!"""
    
    def chat(self, message: str) -> str:
        """Clean chat interface"""
        try:
            messages = [
                SystemMessage(content=self._get_clean_system_prompt()),
                HumanMessage(content=message)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    def generate_agent(self, agent_type: str, agent_name: str, purpose: str, 
                      requirements: str = "", tools_needed: List[str] = []) -> Dict[str, Any]:
        """Generate agent using clean approach"""
        
        try:
            print(f"🎯 Generating {agent_type} agent: {agent_name}")
            
            # Generate using clean prompt approach
            agent_code = self.agent_generator._run(
                agent_type=agent_type,
                agent_name=agent_name, 
                purpose=purpose,
                requirements=requirements,
                tools_needed=tools_needed
            )
            
            # Save to memory if available
            save_result = "Memory not available"
            if self.memory_manager:
                try:
                    memory_key = f"clean_agent:{agent_name.lower()}"
                    self.memory_manager.store_memory(memory_key, agent_code)
                    save_result = f"✅ Saved to memory: {memory_key}"
                except Exception as e:
                    save_result = f"⚠️ Memory save warning: {e}"
            
            return {
                "agent_code": agent_code,
                "save_result": save_result,
                "agent_type": agent_type,
                "agent_name": agent_name,
                "purpose": purpose,
                "approach": "clean_prompt_based",
                "success": True
            }
            
        except Exception as e:
            return {
                "error": f"Generation failed: {str(e)}",
                "success": False
            }


def demo_improved_coder():
    """Demo the improved approach"""
    print("🚀 IMPROVED CODER AGENT - CLEAN APPROACH DEMO")
    print("=" * 80)
    
    try:
        # Create improved agent
        agent = ImprovedCoderAgent("demo_clean")
        
        print("\n🎯 Testing Simple Agent Generation:")
        result = agent.generate_agent(
            agent_type="simple",
            agent_name="DataProcessor",
            purpose="Process and analyze CSV data efficiently",
            requirements="Include error handling for malformed data, support large files"
        )
        
        if result["success"]:
            print(f"✅ Generated: {result['agent_name']}")
            print(f"💾 Save: {result['save_result']}")
            print(f"📝 Code length: {len(result['agent_code'])} characters")
            print(f"🎯 Approach: {result['approach']}")
            
            # Show first few lines
            code_lines = result['agent_code'].split('\n')[:10]
            print("\n📄 Generated code preview:")
            for i, line in enumerate(code_lines, 1):
                print(f"  {i:2d}: {line}")
            print("  ...")
        else:
            print(f"❌ Generation failed: {result.get('error')}")
        
        print("\n" + "=" * 80)
        print("✅ Improved approach: Clean prompts instead of string templates!")
        print("💡 Benefits: Readable, maintainable, lets LLM generate proper code")
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")


if __name__ == "__main__":
    demo_improved_coder()