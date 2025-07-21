#!/usr/bin/env python3
"""
Elite Coder Agent Tests
======================

Comprehensive tests for Elite Coder Agent leveraging Core Agent infrastructure.
Tests include memory management, tool integration, and agent generation.
"""

import sys
import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add workspace to path
sys.path.insert(0, '/workspace')

try:
    from core_agents.coder import (
        EliteCoderAgent, 
        CoderConfig, 
        LangGraphTemplateTool,
        CodeMemoryTool,
        create_elite_coder_agent
    )
    from core.config import AgentConfig
    from core.managers import MemoryManager
    CODER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Coder Agent imports not available: {e}")
    CODER_AVAILABLE = False


class MockMemoryManager:
    """Mock MemoryManager for testing"""
    
    def __init__(self):
        self.memory_store = {}
        self.lists = {}
    
    def save_memory(self, key: str, value: str, append: bool = False) -> bool:
        """Mock save memory implementation"""
        try:
            if append:
                if key not in self.lists:
                    self.lists[key] = []
                if isinstance(self.lists[key], list):
                    self.lists[key].append(value)
                else:
                    self.lists[key] = [self.lists[key], value]
                return True
            else:
                self.memory_store[key] = value
                return True
        except Exception:
            return False
    
    def get_memory(self, key: str) -> str:
        """Mock get memory implementation"""
        if key in self.lists:
            return self.lists[key]
        return self.memory_store.get(key, None)


class MockAzureChatOpenAI:
    """Mock Azure OpenAI for testing"""
    
    def __init__(self, **kwargs):
        self.config = kwargs
        self.call_count = 0
    
    def invoke(self, messages):
        """Mock invoke method"""
        self.call_count += 1
        
        # Handle different message formats
        if isinstance(messages, list):
            last_message = messages[-1] if messages else {}
            if hasattr(last_message, 'content'):
                user_input = last_message.content
            elif isinstance(last_message, dict):
                user_input = last_message.get('content', str(messages))
            else:
                user_input = str(messages)
        else:
            user_input = str(messages)
        
        # Smart mock responses based on input
        if "simple" in user_input.lower() and "agent" in user_input.lower():
            return Mock(content=f"""# Elite Simple Agent
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SimpleAgentState(TypedDict):
    input: str
    output: str

def process_node(state):
    return {"input": state["input"], "output": f"Processed: {state['input']}"}

def create_simple_agent():
    workflow = StateGraph(SimpleAgentState)
    workflow.add_node("process", process_node)
    workflow.set_entry_point("process")
    workflow.add_edge("process", END)
    return workflow.compile()
""")
        
        elif "tool" in user_input.lower() and "select" in user_input.lower():
            return Mock(content="calculator_tool")
        
        elif "supervisor" in user_input.lower():
            return Mock(content="creator")
        
        elif "research" in user_input.lower():
            return Mock(content="Elite research completed successfully")
        
        elif "analyze" in user_input.lower():
            return Mock(content="Elite analysis completed successfully")
        
        elif "create" in user_input.lower():
            return Mock(content="Elite creation completed successfully")
        
        elif "review" in user_input.lower():
            return Mock(content="Elite review completed successfully")
        
        elif "aggregate" in user_input.lower():
            return Mock(content="Elite synthesis of all agent contributions completed")
        
        else:
            return Mock(content=f"Elite response for: {user_input[:50]}...")


@unittest.skipUnless(CODER_AVAILABLE, "Coder Agent not available")
class TestEliteCoderAgent(unittest.TestCase):
    """Test Elite Coder Agent functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_memory = MockMemoryManager()
        
        # Mock Azure OpenAI
        self.mock_llm_patcher = patch('core_agents.coder.AzureChatOpenAI')
        self.mock_llm_class = self.mock_llm_patcher.start()
        self.mock_llm_class.return_value = MockAzureChatOpenAI()
        
        # Mock Core Agent managers
        self.mock_memory_manager_patcher = patch('core_agents.coder.MemoryManager')
        self.mock_memory_manager_class = self.mock_memory_manager_patcher.start()
        self.mock_memory_manager_class.return_value = self.mock_memory
        
        # ToolManager removed from Core Agent
        
        # Mock Core Agent tools
        self.mock_tools_patcher = patch('core_agents.coder.create_python_coding_tools')
        self.mock_tools_func = self.mock_tools_patcher.start()
        self.mock_tools_func.return_value = [
            Mock(name="python_executor", _run=lambda code, timeout=30: "✅ Code executed successfully"),
            Mock(name="file_manager", _run=lambda action, filepath, content="": f"✅ {action} operation completed")
        ]
    
    def tearDown(self):
        """Clean up after tests"""
        self.mock_llm_patcher.stop()
        self.mock_memory_manager_patcher.stop()
        self.mock_tools_patcher.stop()
    
    def test_coder_config(self):
        """Test Coder configuration values"""
        self.assertEqual(CoderConfig.AZURE_OPENAI_ENDPOINT, "https://oai-202-fbeta-dev.openai.azure.com/")
        self.assertEqual(CoderConfig.GPT4_MODEL_NAME, "gpt-4")
        self.assertEqual(CoderConfig.GPT4_DEPLOYMENT_NAME, "gpt4")
        self.assertEqual(CoderConfig.OPENAI_API_VERSION, "2023-12-01-preview")
        self.assertIsNotNone(CoderConfig.OPENAI_API_KEY)
        self.assertEqual(CoderConfig.TEMPERATURE, 0.1)
        self.assertEqual(CoderConfig.MAX_TOKENS, 4000)
    
    def test_elite_coder_agent_initialization(self):
        """Test Elite Coder Agent initialization"""
        agent = EliteCoderAgent("test_session")
        
        # Test basic attributes
        self.assertEqual(agent.session_id, "test_session")
        self.assertIsNotNone(agent.config)
        self.assertIsNotNone(agent.llm)
        
        # Test Core Agent integration
        self.assertEqual(agent.config.name, "EliteCoderAgent")
        self.assertTrue(agent.config.enable_memory)
        self.assertEqual(agent.config.memory_backend, "redis")
        self.assertIn("short_term", agent.config.memory_types)
        self.assertIn("long_term", agent.config.memory_types)
    
    def test_langgraph_template_tool(self):
        """Test LangGraph template generation tool"""
        tool = LangGraphTemplateTool()
        
        # Test tool properties
        self.assertEqual(tool.name, "langgraph_generator")
        self.assertIn("production-ready", tool.description.lower())
        
        # Test simple agent generation
        simple_code = tool._run("simple", "TestAgent", "Testing purposes")
        self.assertIn("class TestAgentState", simple_code)
        self.assertIn("def process_node", simple_code)
        self.assertIn("StateGraph", simple_code)
        self.assertIn("Elite", simple_code)
        self.assertIn(CoderConfig.AZURE_OPENAI_ENDPOINT, simple_code)
        
        # Test agent with tools generation
        tools_code = tool._run("with_tools", "ToolAgent", "Using tools", ["calculator", "search"])
        self.assertIn("calculator_tool", tools_code)
        self.assertIn("search_tool", tools_code)
        self.assertIn("tool_selection_node", tools_code)
        self.assertIn("tool_execution_node", tools_code)
        
        # Test multi-agent system generation
        multi_code = tool._run("multi_agent", "MultiSystem", "Complex tasks")
        self.assertIn("elite_supervisor_node", multi_code)
        self.assertIn("elite_researcher_node", multi_code)
        self.assertIn("elite_analyzer_node", multi_code)
        self.assertIn("elite_creator_node", multi_code)
        self.assertIn("elite_reviewer_node", multi_code)
        
        # Test invalid template type
        invalid_code = tool._run("invalid", "BadAgent", "Should fail")
        self.assertIn("Unknown template type", invalid_code)
    
    def test_code_memory_tool(self):
        """Test Code Memory Tool with Core Agent memory"""
        memory_tool = CodeMemoryTool(self.mock_memory)
        
        # Test tool properties
        self.assertEqual(memory_tool.name, "code_memory")
        self.assertIn("Redis", memory_tool.description)
        
        # Test save operation
        save_result = memory_tool._run(
            action="save",
            module_name="test_agent",
            code="print('Hello World')",
            description="Test module",
            tags=["test", "simple"]
        )
        self.assertIn("✅", save_result)
        self.assertIn("test_agent", save_result)
        
        # Test load operation
        load_result = memory_tool._run(action="load", module_name="test_agent")
        self.assertIn("✅", load_result)
        self.assertIn("test_agent", load_result)
        self.assertIn("Hello World", load_result)
        
        # Test list operation
        list_result = memory_tool._run(action="list")
        self.assertIn("Elite modules", list_result)
        
        # Test search operation
        search_result = memory_tool._run(action="search", search_query="test")
        self.assertIn("Found", search_result)
        
        # Test invalid action
        invalid_result = memory_tool._run(action="invalid")
        self.assertIn("Unknown action", invalid_result)
    
    def test_elite_coder_agent_chat(self):
        """Test Elite Coder Agent chat functionality"""
        agent = EliteCoderAgent("chat_test")
        
        # Test basic chat
        response = agent.chat("Create a simple agent")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)
        
        # Test memory integration
        response2 = agent.chat("Remember our previous conversation")
        self.assertIsNotNone(response2)
    
    def test_generate_complete_agent(self):
        """Test complete agent generation"""
        agent = EliteCoderAgent("gen_test")
        
        # Test simple agent generation
        result = agent.generate_complete_agent(
            template_type="simple",
            agent_name="TestAgent",
            purpose="Testing purposes"
        )
        
        self.assertIn("agent_code", result)
        self.assertIn("save_result", result)
        self.assertIn("test_result", result)
        self.assertTrue(result["core_agent_integrated"])
        self.assertEqual(result["agent_name"], "TestAgent")
        self.assertEqual(result["purpose"], "Testing purposes")
        
        # Verify code quality
        agent_code = result["agent_code"]
        self.assertIn("class TestAgentState", agent_code)
        self.assertIn("Elite", agent_code)
    
    def test_example_prompts(self):
        """Test example prompts for different agent types"""
        agent = EliteCoderAgent("examples_test")
        examples = agent.get_example_prompts()
        
        self.assertIn("simple_agent", examples)
        self.assertIn("agent_with_tools", examples)
        self.assertIn("multi_agent_system", examples)
        
        # Check prompt quality
        for prompt_type, prompt in examples.items():
            self.assertIsInstance(prompt, str)
            self.assertGreater(len(prompt), 50)  # Should be descriptive
            if prompt_type == "simple_agent":
                self.assertIn("Python code analysis", prompt)
            elif prompt_type == "agent_with_tools":
                self.assertIn("automated testing", prompt)
            elif prompt_type == "multi_agent_system":
                self.assertIn("full-stack development", prompt)
    
    def test_factory_function(self):
        """Test factory function for creating agents"""
        # Test with session ID
        agent1 = create_elite_coder_agent("factory_test")
        self.assertEqual(agent1.session_id, "factory_test")
        
        # Test without session ID (should generate one)
        agent2 = create_elite_coder_agent()
        self.assertIsNotNone(agent2.session_id)
        self.assertIn("elite_coder_", agent2.session_id)


@unittest.skipUnless(CODER_AVAILABLE, "Coder Agent not available")
class TestEliteCoderIntegration(unittest.TestCase):
    """Integration tests for Elite Coder Agent"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Mock only the external services, keep Core Agent logic
        self.mock_llm_patcher = patch('core_agents.coder.AzureChatOpenAI')
        self.mock_llm_class = self.mock_llm_patcher.start()
        self.mock_llm_class.return_value = MockAzureChatOpenAI()
    
    def tearDown(self):
        """Clean up integration tests"""
        self.mock_llm_patcher.stop()
    
    def test_full_agent_workflow(self):
        """Test complete agent creation workflow"""
        # Create agent
        agent = create_elite_coder_agent("workflow_test")
        
        # Test agent creation for different types
        test_cases = [
            ("simple", "DataProcessor", "Process data efficiently"),
            ("with_tools", "TestBot", "Automated testing", ["executor", "validator"]),
            ("multi_agent", "DevTeam", "Full development cycle")
        ]
        
        for case in test_cases:
            template_type, agent_name, purpose = case[:3]
            tools_needed = case[3] if len(case) > 3 else []
            
            with self.subTest(template_type=template_type):
                result = agent.generate_complete_agent(
                    template_type=template_type,
                    agent_name=agent_name,
                    purpose=purpose,
                    tools_needed=tools_needed
                )
                
                # Verify result structure
                self.assertIn("agent_code", result)
                self.assertIn("save_result", result)
                self.assertIn("test_result", result)
                self.assertEqual(result["template_type"], template_type)
                self.assertEqual(result["agent_name"], agent_name)
                self.assertEqual(result["purpose"], purpose)
                
                # Verify code contains expected elements
                code = result["agent_code"]
                self.assertIn(agent_name, code)
                self.assertIn("Elite", code)
                self.assertIn("StateGraph", code)
                self.assertIn(CoderConfig.AZURE_OPENAI_ENDPOINT, code)
    
    def test_memory_persistence(self):
        """Test memory persistence across agent sessions"""
        # Create first agent session
        agent1 = create_elite_coder_agent("memory_test_1")
        
        # Chat with first agent
        response1 = agent1.chat("Create a calculator agent")
        self.assertIsNotNone(response1)
        
        # Create second agent session with same ID
        agent2 = create_elite_coder_agent("memory_test_1")
        
        # Reference previous conversation
        response2 = agent2.chat("Enhance the calculator agent we discussed")
        self.assertIsNotNone(response2)
    
    def test_tool_integration(self):
        """Test integration with Core Agent tools"""
        agent = create_elite_coder_agent("tools_test")
        
        # Verify tools are available
        self.assertIsNotNone(agent.config.tools)
        self.assertGreater(len(agent.config.tools), 0)
        
        # Test tool execution through chat
        response = agent.chat("Execute some Python code to test our tools")
        self.assertIsNotNone(response)
    
    def test_error_handling(self):
        """Test error handling in various scenarios"""
        agent = create_elite_coder_agent("error_test")
        
        # Test with invalid template type
        result = agent.generate_complete_agent(
            template_type="invalid_type",
            agent_name="ErrorAgent",
            purpose="Should handle errors gracefully"
        )
        
        # Should still return a result structure
        self.assertIn("agent_code", result)
        
        # Test chat with potential errors
        response = agent.chat("This is a test of error handling")
        self.assertIsNotNone(response)
        self.assertIsInstance(response, str)


if __name__ == "__main__":
    print("🧪 ELITE CODER AGENT TESTS")
    print("=" * 80)
    
    # Run tests
    unittest.main(verbosity=2)