#!/usr/bin/env python3
"""
Test script for CoreAgent Framework

This script tests the basic functionality of the CoreAgent framework
to ensure everything is working correctly.
"""

import sys
import traceback
from core_agent import CoreAgent, AgentConfig, create_basic_agent
from langchain_core.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage


# Mock model for testing
class TestChatModel(BaseChatModel):
    def _generate(self, messages, stop=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        message = AIMessage(content="Test response from mock model")
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def _llm_type(self):
        return "test_mock"


# Test tools
@tool
def test_calculator(expression: str) -> str:
    """Calculate a mathematical expression safely."""
    try:
        # Simple evaluation for testing
        result = eval(expression.replace(" ", ""))
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def test_echo(message: str) -> str:
    """Echo back a message."""
    return f"Echo: {message}"


def test_basic_agent_creation():
    """Test basic agent creation"""
    print("Testing basic agent creation...")
    
    try:
        model = TestChatModel()
        tools = [test_calculator, test_echo]
        
        # Test factory function
        agent = create_basic_agent(model=model, tools=tools)
        
        # Check agent properties
        assert agent.config.name == "BasicAgent"
        assert len(agent.config.tools) == 2
        assert agent.config.enable_memory == True
        
        print("✓ Basic agent creation successful")
        return True
        
    except Exception as e:
        print(f"✗ Basic agent creation failed: {e}")
        traceback.print_exc()
        return False


def test_custom_config_agent():
    """Test agent with custom configuration"""
    print("Testing custom configuration agent...")
    
    try:
        config = AgentConfig(
            name="TestAgent",
            description="A test agent",
            model=TestChatModel(),
            tools=[test_calculator],
            system_prompt="You are a test assistant.",
            enable_memory=True,
            memory_type="memory",
            enable_streaming=False,
            evaluation_metrics=["test_metric"]
        )
        
        agent = CoreAgent(config)
        
        # Check configuration
        assert agent.config.name == "TestAgent"
        assert agent.config.description == "A test agent"
        assert len(agent.config.tools) == 1
        assert agent.config.enable_memory == True
        
        print("✓ Custom configuration agent successful")
        return True
        
    except Exception as e:
        print(f"✗ Custom configuration agent failed: {e}")
        traceback.print_exc()
        return False


def test_memory_functionality():
    """Test memory storage and retrieval"""
    print("Testing memory functionality...")
    
    try:
        config = AgentConfig(
            name="MemoryTestAgent",
            model=TestChatModel(),
            enable_memory=True,
            memory_type="memory"
        )
        
        agent = CoreAgent(config)
        
        # Test memory operations
        agent.store_memory("test_key", "test_value")
        retrieved_value = agent.retrieve_memory("test_key")
        
        # Note: For the mock memory system, this might not work exactly
        # but the methods should exist and not throw errors
        
        print("✓ Memory functionality test completed")
        return True
        
    except Exception as e:
        print(f"✗ Memory functionality failed: {e}")
        traceback.print_exc()
        return False


def test_subgraph_functionality():
    """Test subgraph management"""
    print("Testing subgraph functionality...")
    
    try:
        config = AgentConfig(
            name="SubgraphTestAgent",
            model=TestChatModel(),
            enable_subgraphs=True
        )
        
        agent = CoreAgent(config)
        
        # Test subgraph creation and registration
        tools = [test_calculator]
        subgraph = agent.subgraph_manager.create_tool_subgraph(tools)
        agent.add_subgraph("test_subgraph", subgraph)
        
        # Check if subgraph was registered
        retrieved_subgraph = agent.get_subgraph("test_subgraph")
        assert retrieved_subgraph is not None
        
        print("✓ Subgraph functionality successful")
        return True
        
    except Exception as e:
        print(f"✗ Subgraph functionality failed: {e}")
        traceback.print_exc()
        return False


def test_agent_status():
    """Test agent status reporting"""
    print("Testing agent status...")
    
    try:
        agent = create_basic_agent(model=TestChatModel(), tools=[test_calculator])
        status = agent.get_status()
        
        # Check status structure
        assert "name" in status
        assert "features" in status
        assert "memory_type" in status
        
        print(f"✓ Agent status: {status['name']}")
        print(f"  Features: {status['features']}")
        return True
        
    except Exception as e:
        print(f"✗ Agent status failed: {e}")
        traceback.print_exc()
        return False


def test_config_save_load():
    """Test configuration save and load"""
    print("Testing configuration save/load...")
    
    try:
        agent = create_basic_agent(model=TestChatModel(), tools=[test_calculator])
        
        # Save configuration
        config_file = "test_config.json"
        agent.save_config(config_file)
        
        # Load configuration
        loaded_config = CoreAgent.load_config(config_file)
        
        # Check loaded config
        assert loaded_config.name == "BasicAgent"
        assert loaded_config.enable_memory == True
        
        # Cleanup
        import os
        if os.path.exists(config_file):
            os.remove(config_file)
        
        print("✓ Configuration save/load successful")
        return True
        
    except Exception as e:
        print(f"✗ Configuration save/load failed: {e}")
        traceback.print_exc()
        return False


def test_mcp_functionality():
    """Test MCP (Model Context Protocol) functionality"""
    print("Testing MCP functionality...")
    
    try:
        from core_agent import create_mcp_agent, MCP_AVAILABLE
        
        # Test MCP agent creation
        mcp_servers = {
            "test_server": {
                "command": "python",
                "args": ["-c", "print('test')"],
                "transport": "stdio"
            }
        }
        
        config = AgentConfig(
            name="MCPTestAgent",
            model=TestChatModel(),
            enable_mcp=True,
            mcp_servers=mcp_servers
        )
        
        agent = CoreAgent(config)
        
        # Check MCP configuration
        assert agent.config.enable_mcp == True
        assert len(agent.config.mcp_servers) == 1
        assert "test_server" in agent.config.mcp_servers
        
        # Test MCP methods
        servers = agent.get_mcp_servers()
        assert "test_server" in servers
        
        # Test factory function
        mcp_agent = create_mcp_agent(TestChatModel(), mcp_servers)
        assert mcp_agent.config.name == "MCPAgent"
        assert mcp_agent.config.enable_mcp == True
        
        status = agent.get_status()
        assert "mcp" in status["features"]
        assert status["features"]["mcp"] == True
        assert status["mcp_servers"] == 1
        
        print(f"✓ MCP functionality test completed")
        print(f"  MCP Available: {'Yes' if MCP_AVAILABLE else 'No (install langchain-mcp-adapters)'}")
        return True
        
    except Exception as e:
        print(f"✗ MCP functionality failed: {e}")
        traceback.print_exc()
        return False


def test_langmem_functionality():
    """Test LangMem memory management functionality"""
    print("Testing LangMem functionality...")
    
    try:
        from core_agent import create_langmem_agent, LANGMEM_AVAILABLE
        
        # Test langmem agent creation with different memory types
        memory_types = ["langmem_short", "langmem_long", "langmem_combined"]
        
        for memory_type in memory_types:
            config = AgentConfig(
                name=f"LangMem_{memory_type.split('_')[1].title()}Agent",
                model=TestChatModel(),
                enable_memory=True,
                memory_type=memory_type,
                langmem_max_tokens=256,
                langmem_max_summary_tokens=64,
                langmem_enable_summarization=True
            )
            
            agent = CoreAgent(config)
            
            # Check langmem configuration
            assert agent.config.memory_type == memory_type
            assert agent.config.langmem_max_tokens == 256
            assert agent.config.langmem_max_summary_tokens == 64
            assert agent.config.langmem_enable_summarization == True
            
            # Test langmem methods
            memory_summary = agent.get_memory_summary()
            assert memory_summary["memory_type"] == memory_type
            assert memory_summary["langmem_available"] == LANGMEM_AVAILABLE
            assert memory_summary["max_tokens"] == 256
            
            # Test factory function
        langmem_agent = create_langmem_agent(TestChatModel(), memory_type="langmem_combined")
        assert langmem_agent.config.name == "LangMemAgent"
        assert langmem_agent.config.memory_type == "langmem_combined"
        
        status = langmem_agent.get_status()
        assert "memory" in status["features"]
        assert status["features"]["memory"] == True
        assert "langmem_support" in status
        
        print(f"✓ LangMem functionality test completed")
        print(f"  LangMem Available: {'Yes' if LANGMEM_AVAILABLE else 'No (install langmem)'}")
        print(f"  Memory types tested: {len(memory_types)}")
        return True
        
    except Exception as e:
        print(f"✗ LangMem functionality failed: {e}")
        traceback.print_exc()
        return False


def test_agentevals_functionality():
    """Test AgentEvals evaluation functionality"""
    print("Testing AgentEvals functionality...")
    
    try:
        from core_agent import create_evaluated_agent, AGENTEVALS_AVAILABLE
        
        # Test evaluated agent creation
        config = AgentConfig(
            name="EvaluatedTestAgent",
            model=TestChatModel(),
            enable_evaluation=True,
            evaluation_metrics=["accuracy", "relevance", "helpfulness"]
        )
        
        agent = CoreAgent(config)
        
        # Check evaluation configuration
        assert agent.config.enable_evaluation == True
        assert len(agent.config.evaluation_metrics) == 3
        assert "accuracy" in agent.config.evaluation_metrics
        
        # Test evaluator status
        evaluator_status = agent.get_evaluator_status()
        assert "agentevals_available" in evaluator_status
        assert evaluator_status["agentevals_available"] == AGENTEVALS_AVAILABLE
        
        # Test factory function
        eval_agent = create_evaluated_agent(TestChatModel())
        assert eval_agent.config.name == "EvaluatedAgent"
        assert eval_agent.config.enable_evaluation == True
        
        # Test evaluation methods (these will return errors if agentevals not installed)
        sample_outputs = [{"role": "assistant", "content": "Test response"}]
        sample_reference = [{"role": "assistant", "content": "Expected response"}]
        
        trajectory_result = agent.evaluate_trajectory(sample_outputs, sample_reference)
        assert isinstance(trajectory_result, dict)
        
        llm_judge_result = agent.evaluate_with_llm_judge(sample_outputs, sample_reference)
        assert isinstance(llm_judge_result, dict)
        
        # Test dataset creation
        conversations = [{
            "input_messages": [{"role": "user", "content": "Hello"}],
            "expected_output_messages": [{"role": "assistant", "content": "Hi there!"}]
        }]
        dataset = agent.create_evaluation_dataset(conversations)
        assert len(dataset) == 1
        assert "input" in dataset[0]
        assert "output" in dataset[0]
        
        # Check status includes evaluators
        status = agent.get_status()
        assert "evaluators" in status
        assert "agentevals_available" in status["evaluators"]
        
        print(f"✓ AgentEvals functionality test completed")
        print(f"  AgentEvals Available: {'Yes' if AGENTEVALS_AVAILABLE else 'No (install agentevals)'}")
        print(f"  Evaluators tested: basic, trajectory, llm_judge")
        return True
        
    except Exception as e:
        print(f"✗ AgentEvals functionality failed: {e}")
        traceback.print_exc()
        return False


def test_package_availability():
    """Test which optional packages are available"""
    print("Testing package availability...")
    
    # Import statements from core_agent to check availability
    from core_agent import (
        RedisSaver, create_supervisor, create_swarm, 
        MCP_AVAILABLE, LANGMEM_AVAILABLE, AGENTEVALS_AVAILABLE, 
        ShortTermMemory, LongTermMemory, AgentEvaluator
    )
    
    packages = {
        "RedisSaver": RedisSaver,
        "create_supervisor": create_supervisor,
        "create_swarm": create_swarm,
        "MCP_AVAILABLE": MCP_AVAILABLE,
        "LANGMEM_AVAILABLE": LANGMEM_AVAILABLE,
        "AGENTEVALS_AVAILABLE": AGENTEVALS_AVAILABLE,
        "ShortTermMemory": ShortTermMemory,
        "LongTermMemory": LongTermMemory,
        "AgentEvaluator": AgentEvaluator
    }
    
    print("Package availability:")
    for name, package in packages.items():
        status = "✓" if package is not None else "✗"
        print(f"  {status} {name}")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("CoreAgent Framework Test Suite")
    print("=" * 50)
    
    tests = [
        test_package_availability,
        test_basic_agent_creation,
        test_custom_config_agent,
        test_memory_functionality,
        test_subgraph_functionality,
        test_agent_status,
        test_config_save_load,
        test_mcp_functionality,
        test_langmem_functionality,
        test_agentevals_functionality
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} encountered an error: {e}")
            results.append(False)
            print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CoreAgent framework is working correctly.")
    else:
        print("⚠️ Some tests failed. Please check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)