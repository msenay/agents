#!/usr/bin/env python3
"""
Coder Agent Tests
================

Comprehensive tests for the Professional Coder Agent.
Tests Azure OpenAI integration, Redis memory, tools, and code generation.
"""

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

# Add workspace to path
sys.path.insert(0, '/workspace')

try:
    from core_agents.coder import (
        CoderAgent, CoderMemory, CoderConfig,
        CodeValidatorTool, ModuleManagerTool, FileOperationsTool,
        create_coder_agent
    )
    CODER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Coder Agent import failed: {e}")
    CODER_AVAILABLE = False


class MockRedisClient:
    """Mock Redis client for testing"""
    
    def __init__(self):
        self.data = {}
        self.sets = {}
        self.lists = {}
    
    def ping(self):
        return True
    
    def setex(self, key, ttl, value):
        self.data[key] = value
        return True
    
    def get(self, key):
        return self.data.get(key)
    
    def sadd(self, key, value):
        if key not in self.sets:
            self.sets[key] = set()
        self.sets[key].add(value)
        return True
    
    def smembers(self, key):
        return [m.encode() for m in self.sets.get(key, set())]
    
    def lpush(self, key, value):
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].insert(0, value)
        return True
    
    def lrange(self, key, start, end):
        if key not in self.lists:
            return []
        return self.lists[key][start:end+1] if end >= 0 else self.lists[key][start:]
    
    def expire(self, key, ttl):
        return True


class MockAzureChatOpenAI:
    """Mock Azure OpenAI for testing"""
    
    def __init__(self, *args, **kwargs):
        self.call_count = 0
        self.responses = [
            # Simple agent response
            '''I'll create a simple LangGraph agent for you:

```python
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
from typing import TypedDict, List
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    """State for the simple agent"""
    messages: List[BaseMessage]
    question: str
    answer: str

def answer_node(state: AgentState) -> AgentState:
    """Node that answers questions using LLM"""
    llm = ChatOpenAI(temperature=0)
    
    try:
        response = llm.invoke(state["question"])
        return {
            **state,
            "answer": response.content
        }
    except Exception as e:
        return {
            **state,
            "answer": f"Error: {str(e)}"
        }

def create_simple_agent():
    """Create and return the simple agent graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("answer", answer_node)
    
    # Set entry point
    workflow.set_entry_point("answer")
    
    # Add ending
    workflow.add_edge("answer", END)
    
    # Compile graph
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    agent = create_simple_agent()
    
    result = agent.invoke({
        "question": "What is the capital of France?",
        "messages": [],
        "answer": ""
    })
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
```

This agent provides a clean, simple implementation that follows LangGraph best practices.''',

            # Agent with tools response
            '''Here's a LangGraph agent with tools:

```python
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, ToolMessage
from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely"""
    try:
        # Simple safe evaluation
        allowed_chars = set('0123456789+-*/.() ')
        if all(c in allowed_chars for c in expression):
            result = eval(expression)
            return f"Result: {result}"
        else:
            return "Error: Invalid characters in expression"
    except Exception as e:
        return f"Error: {str(e)}"

@tool  
def web_search(query: str) -> str:
    """Simulate web search (mock implementation)"""
    return f"Search results for '{query}': [Mock search results would appear here]"

class AgentState(TypedDict):
    """State for tool-using agent"""
    messages: List[BaseMessage] 
    question: str
    tool_choice: Literal["calculator", "web_search", "direct_answer"]
    answer: str

def decide_tool_node(state: AgentState) -> AgentState:
    """Decide which tool to use based on question"""
    question = state["question"].lower()
    
    if any(word in question for word in ["calculate", "math", "+", "-", "*", "/"]):
        tool_choice = "calculator"
    elif any(word in question for word in ["search", "find", "lookup", "what is"]):
        tool_choice = "web_search" 
    else:
        tool_choice = "direct_answer"
    
    return {**state, "tool_choice": tool_choice}

def execute_tool_node(state: AgentState) -> AgentState:
    """Execute the chosen tool"""
    question = state["question"]
    tool_choice = state["tool_choice"]
    
    if tool_choice == "calculator":
        # Extract math expression from question
        import re
        numbers = re.findall(r'\\d+(?:\\.\\d+)?', question)
        operators = re.findall(r'[+\\-*/]', question)
        
        if numbers and operators:
            expression = numbers[0]
            for i, op in enumerate(operators):
                if i + 1 < len(numbers):
                    expression += op + numbers[i + 1]
            result = calculator.invoke({"expression": expression})
        else:
            result = "Could not extract mathematical expression"
            
    elif tool_choice == "web_search":
        result = web_search.invoke({"query": question})
    else:
        # Direct LLM answer
        llm = ChatOpenAI(temperature=0)
        response = llm.invoke(question)
        result = response.content
    
    return {**state, "answer": result}

def create_tool_agent():
    """Create agent with tool capabilities"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("decide_tool", decide_tool_node)
    workflow.add_node("execute_tool", execute_tool_node)
    
    # Set entry point
    workflow.set_entry_point("decide_tool")
    
    # Add edges
    workflow.add_edge("decide_tool", "execute_tool")
    workflow.add_edge("execute_tool", END)
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    agent = create_tool_agent()
    
    test_questions = [
        "What is 15 + 27?",
        "Search for information about Python",
        "How are you today?"
    ]
    
    for question in test_questions:
        result = agent.invoke({
            "question": question,
            "messages": [],
            "tool_choice": "",
            "answer": ""
        })
        print(f"Q: {question}")
        print(f"Tool: {result['tool_choice']}")
        print(f"A: {result['answer']}")
        print("-" * 50)
```

This agent intelligently chooses tools based on the question type.''',

            # Memory response
            '''I remember our conversation! We've discussed creating LangGraph agents with different capabilities:

1. **Simple Agent**: Basic question-answering with LLM
2. **Tool Agent**: Agent that can use calculator and web search tools
3. **Code Generation**: Creating production-ready LangGraph implementations

The code modules we've created follow best practices with proper state management, error handling, and modular design.'''
        ]
    
    def invoke(self, messages):
        self.call_count += 1
        response_idx = min(self.call_count - 1, len(self.responses) - 1)
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(self.responses[response_idx])


@unittest.skipUnless(CODER_AVAILABLE, "Coder Agent not available")
class TestCoderMemory(unittest.TestCase):
    """Test Coder Memory functionality"""
    
    def setUp(self):
        """Set up test with mock Redis"""
        self.mock_redis = MockRedisClient()
        
        # Patch redis.from_url to return our mock
        patcher = patch('core_agents.coder.redis.from_url')
        self.mock_redis_factory = patcher.start()
        self.mock_redis_factory.return_value = self.mock_redis
        self.addCleanup(patcher.stop)
        
        self.memory = CoderMemory()
    
    def test_save_and_get_module(self):
        """Test saving and retrieving modules"""
        module_name = "test_agent"
        code = "def test(): pass"
        description = "Test module"
        
        # Save module
        success = self.memory.save_module(module_name, code, description)
        self.assertTrue(success)
        
        # Retrieve module
        module_data = self.memory.get_module(module_name)
        self.assertIsNotNone(module_data)
        self.assertEqual(module_data["code"], code)
        self.assertEqual(module_data["description"], description)
        self.assertEqual(module_data["type"], "module")
    
    def test_list_modules(self):
        """Test listing modules"""
        # Save multiple modules
        modules = [
            ("agent1", "code1", "desc1"),
            ("agent2", "code2", "desc2"),
            ("tool1", "code3", "desc3")
        ]
        
        for name, code, desc in modules:
            self.memory.save_module(name, code, desc)
        
        # List modules
        module_list = self.memory.list_modules()
        self.assertEqual(len(module_list), 3)
        for name, _, _ in modules:
            self.assertIn(name, module_list)
    
    def test_conversation_history(self):
        """Test conversation saving and retrieval"""
        session_id = "test_session"
        
        # Save conversation
        self.memory.save_conversation(session_id, "Hello", "user")
        self.memory.save_conversation(session_id, "Hi there!", "assistant")
        self.memory.save_conversation(session_id, "Create an agent", "user")
        
        # Get history
        history = self.memory.get_conversation_history(session_id, limit=3)
        self.assertEqual(len(history), 3)
        
        # Check order (most recent first)
        self.assertEqual(history[0]["message"], "Create an agent")
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["message"], "Hi there!")
        self.assertEqual(history[1]["role"], "assistant")


@unittest.skipUnless(CODER_AVAILABLE, "Coder Agent not available")
class TestCoderTools(unittest.TestCase):
    """Test Coder Agent tools"""
    
    def setUp(self):
        """Set up test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock memory
        self.mock_redis = MockRedisClient()
        patcher = patch('core_agents.coder.redis.from_url')
        mock_redis_factory = patcher.start()
        mock_redis_factory.return_value = self.mock_redis
        self.addCleanup(patcher.stop)
        
        self.memory = CoderMemory()
    
    def tearDown(self):
        """Clean up"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_code_validator_tool(self):
        """Test code validation tool"""
        tool = CodeValidatorTool()
        
        # Test valid Python code
        valid_code = """
def hello_world():
    \"\"\"Say hello\"\"\"
    print("Hello, World!")
    return "success"
"""
        result = tool._run(valid_code, "python")
        self.assertIn("✅ Syntax validation: PASSED", result)
        self.assertIn("Function definitions: 1 found", result)
        self.assertIn("Docstrings: 1 found", result)
        
        # Test invalid code
        invalid_code = "def broken_function(\n    print('error')"
        result = tool._run(invalid_code, "python")
        self.assertIn("❌ Syntax error", result)
        
        # Test LangGraph-specific validation
        langgraph_code = """
from langgraph.graph import StateGraph, END

@tool
def my_tool():
    pass

def my_node(state):
    return state

workflow = StateGraph()
workflow.add_node("test", my_node)
workflow.add_edge("test", END)
"""
        result = tool._run(langgraph_code, "langgraph")
        self.assertIn("LangGraph imports: Found", result)
        self.assertIn("Tool decorators: Found", result)
        self.assertIn("Node additions: Found", result)
    
    def test_module_manager_tool(self):
        """Test module manager tool"""
        tool = ModuleManagerTool(self.memory)
        
        # Test save
        result = tool._run("save", "test_module", "def test(): pass", "Test description")
        self.assertIn("✅ Module 'test_module' saved successfully", result)
        
        # Test load
        result = tool._run("load", "test_module")
        self.assertIn("✅ Module 'test_module' loaded", result)
        self.assertIn("def test(): pass", result)
        
        # Test list
        result = tool._run("list")
        self.assertIn("test_module", result)
        
        # Test search
        result = tool._run("search", "test")
        self.assertIn("test_module", result)
        
        # Test invalid actions
        result = tool._run("invalid_action")
        self.assertIn("❌ Unknown action", result)
    
    def test_file_operations_tool(self):
        """Test file operations tool"""
        tool = FileOperationsTool(self.temp_dir)
        
        # Test write
        test_content = "print('Hello from file')"
        result = tool._run("write", "test_script.py", test_content)
        self.assertIn("✅ File written", result)
        
        # Verify file exists
        file_path = os.path.join(self.temp_dir, "test_script.py")
        self.assertTrue(os.path.exists(file_path))
        
        # Test read
        result = tool._run("read", "test_script.py")
        self.assertIn("📄 File content", result)
        self.assertIn(test_content, result)
        
        # Test create project
        result = tool._run("create_project", "", "", "my_agent_project")
        self.assertIn("✅ Project 'my_agent_project' created", result)
        
        # Verify project structure
        project_path = os.path.join(self.temp_dir, "my_agent_project")
        self.assertTrue(os.path.exists(project_path))
        self.assertTrue(os.path.exists(os.path.join(project_path, "agents")))
        self.assertTrue(os.path.exists(os.path.join(project_path, "tools")))
        self.assertTrue(os.path.exists(os.path.join(project_path, "README.md")))


@unittest.skipUnless(CODER_AVAILABLE, "Coder Agent not available")
class TestCoderAgent(unittest.TestCase):
    """Test main Coder Agent functionality"""
    
    def setUp(self):
        """Set up test with mocks"""
        # Mock Redis
        self.mock_redis = MockRedisClient()
        redis_patcher = patch('core_agents.coder.redis.from_url')
        mock_redis_factory = redis_patcher.start()
        mock_redis_factory.return_value = self.mock_redis
        self.addCleanup(redis_patcher.stop)
        
        # Mock Azure OpenAI
        openai_patcher = patch('core_agents.coder.AzureChatOpenAI')
        mock_openai = openai_patcher.start()
        mock_openai.return_value = MockAzureChatOpenAI()
        self.addCleanup(openai_patcher.stop)
        
        self.agent = CoderAgent("test_session")
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        self.assertEqual(self.agent.session_id, "test_session")
        self.assertIsNotNone(self.agent.memory)
        self.assertIsNotNone(self.agent.llm)
        self.assertEqual(len(self.agent.tools), 3)
    
    def test_system_prompt(self):
        """Test system prompt content"""
        prompt = self.agent.get_system_prompt()
        self.assertIn("EXPERT CODER AGENT", prompt)
        self.assertIn("LangGraph", prompt)
        self.assertIn("multi-agent systems", prompt)
        self.assertIn("code_validator", prompt)
        self.assertIn("module_manager", prompt)
        self.assertIn("file_operations", prompt)
    
    def test_prompt_examples(self):
        """Test example prompts"""
        examples = self.agent.get_prompt_examples()
        self.assertIn("simple_agent", examples)
        self.assertIn("agent_with_tools", examples)
        self.assertIn("multi_agent_system", examples)
        
        # Check content
        self.assertIn("LangGraph agent", examples["simple_agent"])
        self.assertIn("tools", examples["agent_with_tools"])
        self.assertIn("supervisor", examples["multi_agent_system"])
    
    def test_chat_functionality(self):
        """Test chat with memory integration"""
        # Test simple question
        response = self.agent.chat("Create a simple LangGraph agent")
        self.assertIn("LangGraph agent", response)
        self.assertIn("```python", response)
        
        # Test memory recall
        response = self.agent.chat("What have we discussed?")
        self.assertIn("remember", response.lower())
    
    def test_generate_agent_code(self):
        """Test code generation methods"""
        # Test simple agent
        result = self.agent.generate_agent_code("Question answering agent", "simple")
        self.assertIn("response", result)
        self.assertIn("code_blocks", result)
        self.assertEqual(result["complexity"], "simple")
        self.assertTrue(len(result["code_blocks"]) > 0)
        
        # Test with tools
        result = self.agent.generate_agent_code("Agent with calculator", "with_tools")
        self.assertEqual(result["complexity"], "with_tools")
        
        # Test multi-agent
        result = self.agent.generate_agent_code("Task coordination system", "multi_agent")
        self.assertEqual(result["complexity"], "multi_agent")


class TestCoderAgentIntegration(unittest.TestCase):
    """Integration tests that work without external dependencies"""
    
    def test_coder_agent_factory(self):
        """Test factory function"""
        with patch('core_agents.coder.redis.from_url'), \
             patch('core_agents.coder.AzureChatOpenAI'):
            
            # Test with default session
            agent1 = create_coder_agent()
            self.assertTrue(agent1.session_id.startswith("coder_"))
            
            # Test with custom session
            agent2 = create_coder_agent("custom_session")
            self.assertEqual(agent2.session_id, "custom_session")
    
    def test_config_values(self):
        """Test configuration values"""
        self.assertIsNotNone(CoderConfig.AZURE_OPENAI_ENDPOINT)
        self.assertIsNotNone(CoderConfig.OPENAI_API_KEY)
        self.assertIsNotNone(CoderConfig.OPENAI_API_VERSION)
        self.assertEqual(CoderConfig.MODEL_NAME, "gpt-4")
        self.assertEqual(CoderConfig.TEMPERATURE, 0.1)
    
    def test_tools_integration(self):
        """Test that tools work together"""
        with patch('core_agents.coder.redis.from_url') as mock_redis:
            mock_redis.return_value = MockRedisClient()
            
            memory = CoderMemory()
            validator = CodeValidatorTool()
            module_manager = ModuleManagerTool(memory)
            
            # Test workflow: validate -> save -> load
            test_code = "def hello(): return 'world'"
            
            # 1. Validate
            validation_result = validator._run(test_code, "python")
            self.assertIn("✅ Syntax validation: PASSED", validation_result)
            
            # 2. Save to memory
            save_result = module_manager._run("save", "hello_module", test_code, "Test module")
            self.assertIn("✅ Module 'hello_module' saved", save_result)
            
            # 3. Load from memory
            load_result = module_manager._run("load", "hello_module")
            self.assertIn("def hello(): return 'world'", load_result)


def run_coder_tests():
    """Run all coder agent tests"""
    print("=" * 80)
    print("🧪 CODER AGENT TESTS")
    print("=" * 80)
    
    if not CODER_AVAILABLE:
        print("⚠️ Coder Agent not available - skipping tests")
        return False
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(TestCoderMemory('test_save_and_get_module'))
    suite.addTest(TestCoderMemory('test_list_modules'))
    suite.addTest(TestCoderMemory('test_conversation_history'))
    
    suite.addTest(TestCoderTools('test_code_validator_tool'))
    suite.addTest(TestCoderTools('test_module_manager_tool'))
    suite.addTest(TestCoderTools('test_file_operations_tool'))
    
    suite.addTest(TestCoderAgent('test_agent_initialization'))
    suite.addTest(TestCoderAgent('test_system_prompt'))
    suite.addTest(TestCoderAgent('test_prompt_examples'))
    suite.addTest(TestCoderAgent('test_chat_functionality'))
    suite.addTest(TestCoderAgent('test_generate_agent_code'))
    
    suite.addTest(TestCoderAgentIntegration('test_coder_agent_factory'))
    suite.addTest(TestCoderAgentIntegration('test_config_values'))
    suite.addTest(TestCoderAgentIntegration('test_tools_integration'))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    if result.wasSuccessful():
        print("🎉 ALL CODER AGENT TESTS PASSED!")
        print("✅ Memory system works")
        print("✅ Tools function correctly")
        print("✅ Agent integration works")
        print("✅ Code generation works")
        print("✅ Azure OpenAI integration ready")
        print("✅ Redis memory integration ready")
    else:
        print("❌ SOME CODER TESTS FAILED!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    print("=" * 80)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_coder_tests()
    exit(0 if success else 1)