"""
Comprehensive Unit Tests for CoreAgent Framework
===============================================

This test suite covers every aspect of the CoreAgent framework:
- AgentConfig dataclass and all its parameters
- CoreAgentState model and validation
- All manager classes (SubgraphManager, MemoryManager, etc.)
- CoreAgent class and all its methods
- Factory functions and orchestration patterns
- Error handling and edge cases
- Memory management and persistence
- Tool integration and execution
- Optional feature dependencies
- Multi-agent coordination
- Streaming capabilities
- Configuration persistence
"""

import unittest
import asyncio
import tempfile
import json
import os
import sys
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field, ValidationError

# Mock dependencies before importing core_agent
sys.modules['langgraph.checkpoint.redis'] = MagicMock()
sys.modules['langgraph_supervisor'] = MagicMock()
sys.modules['langgraph_swarm'] = MagicMock()
sys.modules['langchain_mcp_adapters.client'] = MagicMock()
sys.modules['langmem'] = MagicMock()
sys.modules['agentevals'] = MagicMock()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models import BaseChatModel
from langgraph.graph import StateGraph

# Import core_agent after mocking dependencies
from core_agent import (
    AgentConfig, CoreAgentState, CoreAgent,
    SubgraphManager, MemoryManager, SupervisorManager, 
    MCPManager, EvaluationManager,
    create_simple_agent, create_advanced_agent,
    create_supervisor_agent, create_swarm_agent, create_handoff_agent,
    create_memory_agent, create_evaluated_agent, create_human_interactive_agent,
    create_mcp_agent, create_langmem_agent
)


class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig dataclass and all its parameters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_tool = Mock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
    
    def test_agent_config_minimal_creation(self):
        """Test creating AgentConfig with minimal parameters"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model
        )
        
        self.assertEqual(config.name, "TestAgent")
        self.assertEqual(config.model, self.mock_model)
        self.assertEqual(config.system_prompt, "You are a helpful AI assistant.")
        self.assertEqual(config.tools, [])
        self.assertFalse(config.enable_memory)
        self.assertFalse(config.enable_human_feedback)
        self.assertFalse(config.enable_evaluation)
        self.assertFalse(config.enable_mcp)
    
    def test_agent_config_full_creation(self):
        """Test creating AgentConfig with all parameters"""
        config = AgentConfig(
            name="FullAgent",
            model=self.mock_model,
            system_prompt="Custom prompt",
            tools=[self.mock_tool],
            description="Test description",
            enable_short_term_memory=True, enable_long_term_memory=True,
            short_term_memory_type="redis",
            redis_url="redis://localhost:6379",
            langmem_max_tokens=512,
            langmem_max_summary_tokens=256,
            langmem_enable_summarization=False,
            enable_human_feedback=True,
            interrupt_before=["tool_node"],
            interrupt_after=["agent_node"],
            enable_evaluation=True,
            evaluation_metrics=["accuracy", "speed"],
            enable_mcp=True,
            mcp_servers={"test": {"type": "stdio"}},
            enable_supervisor=True,
            agents={"agent1": Mock()},
            enable_swarm=True,
            default_active_agent="agent1",
            enable_handoff=True,
            handoff_agents=["agent1", "agent2"],
            enable_streaming=True,
            stream_mode="updates",
            response_format=None,
            custom_graph_builder=None,
            pre_model_hook=None,
            post_model_hook=None
        )
        
        self.assertEqual(config.name, "FullAgent")
        self.assertEqual(config.model, self.mock_model)
        self.assertEqual(config.system_prompt, "Custom prompt")
        self.assertEqual(config.tools, [self.mock_tool])
        self.assertEqual(config.description, "Test description")
        self.assertTrue(config.enable_memory)
        self.assertEqual(config.memory_type, "redis")
        self.assertEqual(config.redis_url, "redis://localhost:6379")
        self.assertEqual(config.langmem_max_tokens, 512)
        self.assertEqual(config.langmem_max_summary_tokens, 256)
        self.assertFalse(config.langmem_enable_summarization)
        self.assertTrue(config.enable_human_feedback)
        self.assertEqual(config.interrupt_before, ["tool_node"])
        self.assertEqual(config.interrupt_after, ["agent_node"])
        self.assertTrue(config.enable_evaluation)
        self.assertEqual(config.evaluation_metrics, ["accuracy", "speed"])
        self.assertTrue(config.enable_mcp)
        self.assertEqual(config.mcp_servers, {"test": {"type": "stdio"}})
        self.assertTrue(config.enable_supervisor)
        self.assertTrue(config.enable_swarm)
        self.assertEqual(config.default_active_agent, "agent1")
        self.assertTrue(config.enable_handoff)
        self.assertEqual(config.handoff_agents, ["agent1", "agent2"])
        self.assertTrue(config.enable_streaming)
        self.assertEqual(config.stream_mode, "updates")
    
    def test_agent_config_post_init(self):
        """Test AgentConfig __post_init__ validation"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            tools=[self.mock_tool]
        )
        
        # Should process tools and set tool_names
        self.assertIn("test_tool", [tool.name for tool in config.tools])
    
    def test_agent_config_invalid_memory_type(self):
        """Test AgentConfig with invalid memory type"""
        with self.assertRaises(ValueError):
            AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                enable_short_term_memory=True, enable_long_term_memory=True,
                short_term_memory_type="invalid_type"
            )
    
    def test_agent_config_memory_without_enable(self):
        """Test AgentConfig memory settings without enable_memory"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            short_term_memory_type="redis",  # Should be ignored
            redis_url="redis://localhost"  # Should be ignored
        )
        
        self.assertFalse(config.enable_memory)
        self.assertEqual(config.memory_type, "redis")  # Still set but not used


class TestCoreAgentState(unittest.TestCase):
    """Test CoreAgentState Pydantic model"""
    
    def test_core_agent_state_creation(self):
        """Test creating CoreAgentState with default values"""
        state = CoreAgentState()
        
        self.assertEqual(state.messages, [])
        self.assertEqual(state.next_agent, "")
        self.assertEqual(state.metadata, {})
        self.assertEqual(state.tool_results, [])
        self.assertEqual(state.human_feedback, "")
        self.assertEqual(state.evaluation_results, {})
    
    def test_core_agent_state_with_data(self):
        """Test creating CoreAgentState with data"""
        message = HumanMessage(content="Test message")
        state = CoreAgentState(
            messages=[message],
            next_agent="agent2",
            metadata={"key": "value"},
            tool_results=[{"tool": "test", "result": "success"}],
            human_feedback="Good job",
            evaluation_results={"score": 0.95}
        )
        
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.messages[0].content, "Test message")
        self.assertEqual(state.next_agent, "agent2")
        self.assertEqual(state.metadata["key"], "value")
        self.assertEqual(len(state.tool_results), 1)
        self.assertEqual(state.human_feedback, "Good job")
        self.assertEqual(state.evaluation_results["score"], 0.95)
    
    def test_core_agent_state_validation(self):
        """Test CoreAgentState validation"""
        # Test invalid message type
        with self.assertRaises(ValidationError):
            CoreAgentState(messages=["invalid_message"])
        
        # Test invalid metadata type
        with self.assertRaises(ValidationError):
            CoreAgentState(metadata="invalid_metadata")


class TestSubgraphManager(unittest.TestCase):
    """Test SubgraphManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manager = SubgraphManager()
        self.mock_graph = Mock(spec=StateGraph)
        self.mock_tool = Mock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
    
    def test_register_subgraph(self):
        """Test registering a subgraph"""
        self.manager.register_subgraph("test_graph", self.mock_graph)
        
        self.assertIn("test_graph", self.manager.subgraphs)
        self.assertEqual(self.manager.subgraphs["test_graph"], self.mock_graph)
    
    def test_get_subgraph_exists(self):
        """Test getting an existing subgraph"""
        self.manager.register_subgraph("test_graph", self.mock_graph)
        result = self.manager.get_subgraph("test_graph")
        
        self.assertEqual(result, self.mock_graph)
    
    def test_get_subgraph_not_exists(self):
        """Test getting a non-existing subgraph"""
        result = self.manager.get_subgraph("nonexistent")
        
        self.assertIsNone(result)
    
    def test_create_tool_subgraph(self):
        """Test creating a tool subgraph"""
        tools = [self.mock_tool]
        result = self.manager.create_tool_subgraph(tools)
        
        self.assertIsInstance(result, StateGraph)


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_short_term_memory=True, enable_long_term_memory=True,
            short_term_memory_type="inmemory"
        )
        self.manager = MemoryManager(self.config)
    
    def test_memory_manager_creation_disabled(self):
        """Test MemoryManager creation with memory disabled"""
        config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_short_term_memory=False, enable_long_term_memory=False
        )
        manager = MemoryManager(config)
        
        self.assertIsNone(manager.checkpointer)
        self.assertIsNone(manager.short_term_memory)
        self.assertIsNone(manager.long_term_memory)
    
    def test_memory_manager_creation_enabled(self):
        """Test MemoryManager creation with memory enabled"""
        self.assertIsNotNone(self.manager.checkpointer)
        self.assertEqual(self.manager.config.memory_type, "memory")
    
    @patch('core_agent.RedisSaver')
    def test_initialize_redis_memory(self, mock_redis_saver):
        """Test initializing Redis memory"""
        config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_short_term_memory=True, enable_long_term_memory=True,
            short_term_memory_type="redis",
            redis_url="redis://localhost:6379"
        )
        
        # Mock RedisSaver to be available
        with patch('core_agent.RedisSaver', mock_redis_saver):
            manager = MemoryManager(config)
            self.assertIsNotNone(manager.checkpointer)
    
    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving memory"""
        # Test storing memory
        self.manager.store_memory("test_key", "test_value")
        
        # Test retrieving memory
        result = self.manager.retrieve_memory("test_key")
        self.assertEqual(result, "test_value")
        
        # Test retrieving non-existent key
        result = self.manager.retrieve_memory("nonexistent")
        self.assertIsNone(result)
    
    def test_get_checkpointer(self):
        """Test getting checkpointer"""
        checkpointer = self.manager.get_checkpointer()
        self.assertIsNotNone(checkpointer)
    
    def test_has_langmem_support(self):
        """Test langmem support detection"""
        # Should return False since we mocked the import
        self.assertFalse(self.manager.has_langmem_support())


class TestSupervisorManager(unittest.TestCase):
    """Test SupervisorManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_supervisor=True,
            agents={"agent1": Mock(), "agent2": Mock()}
        )
        self.manager = SupervisorManager(self.config)
    
    def test_supervisor_manager_creation_disabled(self):
        """Test SupervisorManager creation with supervisor disabled"""
        config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_supervisor=False
        )
        manager = SupervisorManager(config)
        
        self.assertIsNone(manager.supervisor_graph)
        self.assertEqual(manager.agents, {})
    
    def test_supervisor_manager_creation_enabled(self):
        """Test SupervisorManager creation with supervisor enabled"""
        self.assertEqual(self.manager.agents, self.config.agents)
        self.assertTrue(self.manager.config.enable_supervisor)
    
    def test_add_agent(self):
        """Test adding an agent"""
        new_agent = Mock()
        self.manager.add_agent("agent3", new_agent)
        
        self.assertIn("agent3", self.manager.agents)
        self.assertEqual(self.manager.agents["agent3"], new_agent)
    
    def test_coordinate_agents(self):
        """Test coordinating agents"""
        task = "Test task"
        result = self.manager.coordinate_agents(task)
        
        self.assertIsInstance(result, dict)
        self.assertIn("task", result)
        self.assertEqual(result["task"], task)
    
    def test_get_available_transfers(self):
        """Test getting available transfers"""
        transfers = self.manager.get_available_transfers()
        
        self.assertIsInstance(transfers, list)
        self.assertIn("agent1", transfers)
        self.assertIn("agent2", transfers)


class TestMCPManager(unittest.TestCase):
    """Test MCPManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_mcp=True,
            mcp_servers={"test_server": {"type": "stdio", "command": "test"}}
        )
        self.manager = MCPManager(self.config)
    
    def test_mcp_manager_creation_disabled(self):
        """Test MCPManager creation with MCP disabled"""
        config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_mcp=False
        )
        manager = MCPManager(config)
        
        self.assertIsNone(manager.mcp_client)
        self.assertEqual(manager.servers, {})
    
    def test_mcp_manager_creation_enabled(self):
        """Test MCPManager creation with MCP enabled"""
        self.assertEqual(self.manager.servers, self.config.mcp_servers)
        self.assertTrue(self.manager.config.enable_mcp)
    
    @patch('core_agent.MCP_AVAILABLE', True)
    async def test_get_mcp_tools(self):
        """Test getting MCP tools"""
        # Mock the MCP client
        self.manager.mcp_client = AsyncMock()
        self.manager.mcp_client.get_tools.return_value = [{"name": "test_tool"}]
        
        tools = await self.manager.get_mcp_tools()
        self.assertIsInstance(tools, list)
    
    def test_get_server_names(self):
        """Test getting server names"""
        names = self.manager.get_server_names()
        
        self.assertIsInstance(names, list)
        self.assertIn("test_server", names)
    
    def test_add_server(self):
        """Test adding a server"""
        self.manager.add_server("new_server", {"type": "stdio", "command": "new"})
        
        self.assertIn("new_server", self.manager.servers)
        self.assertEqual(self.manager.servers["new_server"]["command"], "new")


class TestEvaluationManager(unittest.TestCase):
    """Test EvaluationManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_evaluation=True,
            evaluation_metrics=["accuracy", "relevance"]
        )
        self.manager = EvaluationManager(self.config)
    
    def test_evaluation_manager_creation_disabled(self):
        """Test EvaluationManager creation with evaluation disabled"""
        config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_evaluation=False
        )
        manager = EvaluationManager(config)
        
        self.assertIsNone(manager.evaluator)
        self.assertEqual(manager.metrics, [])
    
    def test_evaluation_manager_creation_enabled(self):
        """Test EvaluationManager creation with evaluation enabled"""
        self.assertEqual(self.manager.metrics, self.config.evaluation_metrics)
        self.assertTrue(self.manager.config.enable_evaluation)
    
    def test_evaluate_response(self):
        """Test evaluating a response"""
        result = self.manager.evaluate_response("input", "output")
        
        self.assertIsInstance(result, dict)
        self.assertIn("accuracy", result)
        self.assertIn("relevance", result)
    
    def test_evaluate_trajectory(self):
        """Test evaluating a trajectory"""
        outputs = [{"step": 1, "action": "test"}]
        reference = [{"step": 1, "action": "test"}]
        
        result = self.manager.evaluate_trajectory(outputs, reference)
        
        self.assertIsInstance(result, dict)
        self.assertIn("trajectory_score", result)
    
    def test_get_evaluator_status(self):
        """Test getting evaluator status"""
        status = self.manager.get_evaluator_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("basic_evaluator", status)


class TestCoreAgent(unittest.TestCase):
    """Test CoreAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_tool = Mock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
        
        self.config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            tools=[self.mock_tool]
        )
        
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_create_react.return_value = Mock()
            self.agent = CoreAgent(self.config)
    
    def test_core_agent_creation(self):
        """Test CoreAgent creation"""
        self.assertEqual(self.agent.config, self.config)
        self.assertIsInstance(self.agent.state, CoreAgentState)
        self.assertIsInstance(self.agent.subgraph_manager, SubgraphManager)
        self.assertIsInstance(self.agent.memory_manager, MemoryManager)
        self.assertIsInstance(self.agent.supervisor_manager, SupervisorManager)
        self.assertIsInstance(self.agent.mcp_manager, MCPManager)
        self.assertIsInstance(self.agent.evaluation_manager, EvaluationManager)
    
    def test_build_with_prebuilt(self):
        """Test building agent with prebuilt components"""
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_graph = Mock()
            mock_create_react.return_value = mock_graph
            
            agent = CoreAgent(self.config)
            
            mock_create_react.assert_called_once()
            self.assertIsNotNone(agent.compiled_graph)
    
    def test_build_custom_graph(self):
        """Test building custom graph"""
        config = AgentConfig(
            name="TestAgent",
            model=None,  # No model to trigger custom graph
            tools=[]
        )
        
        agent = CoreAgent(config)
        self.assertIsNotNone(agent.graph)
    
    @patch('core_agent.create_react_agent')
    async def test_invoke(self, mock_create_react):
        """Test invoke method"""
        mock_graph = AsyncMock()
        mock_graph.invoke.return_value = {"messages": []}
        mock_create_react.return_value = mock_graph
        
        agent = CoreAgent(self.config)
        agent.compiled_graph = mock_graph
        
        result = agent.invoke("test input")
        
        self.assertIsInstance(result, dict)
        mock_graph.invoke.assert_called_once()
    
    @patch('core_agent.create_react_agent')
    async def test_ainvoke(self, mock_create_react):
        """Test ainvoke method"""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"messages": []}
        mock_create_react.return_value = mock_graph
        
        agent = CoreAgent(self.config)
        agent.compiled_graph = mock_graph
        
        result = await agent.ainvoke("test input")
        
        self.assertIsInstance(result, dict)
        mock_graph.ainvoke.assert_called_once()
    
    def test_add_subgraph(self):
        """Test adding a subgraph"""
        mock_subgraph = Mock(spec=StateGraph)
        self.agent.add_subgraph("test_subgraph", mock_subgraph)
        
        result = self.agent.get_subgraph("test_subgraph")
        self.assertEqual(result, mock_subgraph)
    
    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving memory"""
        self.agent.store_memory("test_key", "test_value")
        result = self.agent.retrieve_memory("test_key")
        
        self.assertEqual(result, "test_value")
    
    def test_get_status(self):
        """Test getting agent status"""
        status = self.agent.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("name", status)
        self.assertIn("memory_enabled", status)
        self.assertIn("tools_count", status)
        self.assertEqual(status["name"], "TestAgent")
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save config
            self.agent.save_config(filepath)
            
            # Verify file exists and contains data
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r') as f:
                saved_data = json.load(f)
            
            self.assertIn("name", saved_data)
            self.assertEqual(saved_data["name"], "TestAgent")
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating agents"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_tool = Mock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
    
    @patch('core_agent.create_react_agent')
    def test_create_simple_agent(self, mock_create_react):
        """Test creating a simple agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_simple_agent(
            model=self.mock_model,
            name="SimpleAgent",
            tools=[self.mock_tool]
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertEqual(agent.config.name, "SimpleAgent")
        self.assertFalse(agent.config.enable_memory)
        self.assertFalse(agent.config.enable_evaluation)
    
    @patch('core_agent.create_react_agent')
    def test_create_advanced_agent(self, mock_create_react):
        """Test creating an advanced agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_advanced_agent(
            model=self.mock_model,
            name="AdvancedAgent",
            tools=[self.mock_tool],
            enable_short_term_memory=True, enable_long_term_memory=True,
            enable_evaluation=True,
            enable_human_feedback=True
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertEqual(agent.config.name, "AdvancedAgent")
        self.assertTrue(agent.config.enable_memory)
        self.assertTrue(agent.config.enable_evaluation)
        self.assertTrue(agent.config.enable_human_feedback)
    
    @patch('core_agent.create_react_agent')
    def test_create_memory_agent(self, mock_create_react):
        """Test creating a memory agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_memory_agent(
            model=self.mock_model,
            short_term_memory_type="redis",
            redis_url="redis://localhost:6379"
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_memory)
        self.assertEqual(agent.config.memory_type, "redis")
        self.assertEqual(agent.config.redis_url, "redis://localhost:6379")
    
    @patch('core_agent.create_react_agent')
    def test_create_evaluated_agent(self, mock_create_react):
        """Test creating an evaluated agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_evaluated_agent(
            model=self.mock_model,
            evaluation_metrics=["accuracy", "speed"]
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_evaluation)
        self.assertEqual(agent.config.evaluation_metrics, ["accuracy", "speed"])
    
    @patch('core_agent.create_react_agent')
    def test_create_human_interactive_agent(self, mock_create_react):
        """Test creating a human interactive agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_human_interactive_agent(
            model=self.mock_model,
            interrupt_before=["tool_node"],
            interrupt_after=["agent_node"]
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_human_feedback)
        self.assertEqual(agent.config.interrupt_before, ["tool_node"])
        self.assertEqual(agent.config.interrupt_after, ["agent_node"])
    
    @patch('core_agent.create_react_agent')
    def test_create_supervisor_agent(self, mock_create_react):
        """Test creating a supervisor agent"""
        mock_create_react.return_value = Mock()
        
        agents = {"agent1": Mock(), "agent2": Mock()}
        agent = create_supervisor_agent(
            model=self.mock_model,
            agents=agents
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_supervisor)
        self.assertEqual(agent.config.agents, agents)
    
    @patch('core_agent.create_react_agent')
    def test_create_swarm_agent(self, mock_create_react):
        """Test creating a swarm agent"""
        mock_create_react.return_value = Mock()
        
        agents = {"agent1": Mock(), "agent2": Mock()}
        agent = create_swarm_agent(
            model=self.mock_model,
            agents=agents,
            default_active_agent="agent1"
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_swarm)
        self.assertEqual(agent.config.agents, agents)
        self.assertEqual(agent.config.default_active_agent, "agent1")
    
    @patch('core_agent.create_react_agent')
    def test_create_handoff_agent(self, mock_create_react):
        """Test creating a handoff agent"""
        mock_create_react.return_value = Mock()
        
        agents = {"agent1": Mock(), "agent2": Mock()}
        agent = create_handoff_agent(
            model=self.mock_model,
            agents=agents,
            handoff_agents=["agent1", "agent2"]
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_handoff)
        self.assertEqual(agent.config.agents, agents)
        self.assertEqual(agent.config.handoff_agents, ["agent1", "agent2"])
    
    @patch('core_agent.create_react_agent')
    @patch('core_agent.MCP_AVAILABLE', True)
    def test_create_mcp_agent(self, mock_create_react):
        """Test creating an MCP agent"""
        mock_create_react.return_value = Mock()
        
        mcp_servers = {"server1": {"type": "stdio", "command": "test"}}
        agent = create_mcp_agent(
            model=self.mock_model,
            mcp_servers=mcp_servers
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_mcp)
        self.assertEqual(agent.config.mcp_servers, mcp_servers)
    
    @patch('core_agent.create_react_agent')
    @patch('core_agent.LANGMEM_AVAILABLE', True)
    def test_create_langmem_agent(self, mock_create_react):
        """Test creating a langmem agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_langmem_agent(
            model=self.mock_model,
            enable_summarization=True,
            max_tokens=512
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertTrue(agent.config.enable_memory)
        self.assertEqual(agent.config.memory_type, "langmem_short")
        self.assertEqual(agent.config.langmem_max_tokens, 512)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_agent_config_invalid_params(self):
        """Test AgentConfig with invalid parameters"""
        # Test missing required parameters
        with self.assertRaises(TypeError):
            AgentConfig()  # Missing name and model
        
        # Test invalid tool type
        with self.assertRaises(TypeError):
            AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                tools=["invalid_tool"]  # Should be BaseTool instances
            )
    
    def test_core_agent_state_invalid_data(self):
        """Test CoreAgentState with invalid data"""
        # Test invalid message type
        with self.assertRaises(ValidationError):
            CoreAgentState(messages=["not_a_message"])
        
        # Test invalid metadata type
        with self.assertRaises(ValidationError):
            CoreAgentState(metadata="not_a_dict")
    
    @patch('core_agent.create_react_agent')
    def test_core_agent_build_failure(self, mock_create_react):
        """Test CoreAgent build failure handling"""
        # Make create_react_agent raise an exception
        mock_create_react.side_effect = Exception("Build failed")
        
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model
        )
        
        # Agent should still be created but with fallback behavior
        agent = CoreAgent(config)
        self.assertIsNotNone(agent)
    
    def test_memory_manager_invalid_redis_url(self):
        """Test MemoryManager with invalid Redis URL"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            enable_short_term_memory=True, enable_long_term_memory=True,
            short_term_memory_type="redis",
            redis_url="invalid://url"
        )
        
        # Should fall back to memory checkpointer
        manager = MemoryManager(config)
        self.assertIsNotNone(manager.checkpointer)
    
    def test_subgraph_manager_duplicate_registration(self):
        """Test SubgraphManager duplicate subgraph registration"""
        manager = SubgraphManager()
        mock_graph1 = Mock(spec=StateGraph)
        mock_graph2 = Mock(spec=StateGraph)
        
        manager.register_subgraph("test", mock_graph1)
        manager.register_subgraph("test", mock_graph2)  # Should overwrite
        
        result = manager.get_subgraph("test")
        self.assertEqual(result, mock_graph2)
    
    def test_agent_missing_dependencies(self):
        """Test agent behavior with missing optional dependencies"""
        # All dependencies are mocked, so they should be handled gracefully
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            enable_mcp=True,
            enable_evaluation=True,
            enable_summarization=True
        )
        
        # Should create agent without errors
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_create_react.return_value = Mock()
            agent = CoreAgent(config)
            self.assertIsNotNone(agent)


class TestOptionalFeatures(unittest.TestCase):
    """Test optional feature availability and graceful degradation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_redis_availability(self):
        """Test Redis availability detection"""
        # Test when Redis is not available
        with patch('core_agent.RedisSaver', None):
            config = AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                enable_short_term_memory=True, enable_long_term_memory=True,
                short_term_memory_type="redis"
            )
            manager = MemoryManager(config)
            # Should fall back to memory checkpointer
            self.assertIsNotNone(manager.checkpointer)
    
    def test_langmem_availability(self):
        """Test LangMem availability detection"""
        # Test when LangMem is not available
        with patch('core_agent.LANGMEM_AVAILABLE', False):
            config = AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                enable_short_term_memory=True, enable_long_term_memory=True,
                enable_summarization=True
            )
            manager = MemoryManager(config)
            self.assertFalse(manager.has_langmem_support())
    
    def test_agentevals_availability(self):
        """Test AgentEvals availability detection"""
        # Test when AgentEvals is not available
        with patch('core_agent.AGENTEVALS_AVAILABLE', False):
            config = AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                enable_evaluation=True
            )
            manager = EvaluationManager(config)
            status = manager.get_evaluator_status()
            self.assertIn("basic_evaluator", status)
    
    def test_mcp_availability(self):
        """Test MCP availability detection"""
        # Test when MCP is not available
        with patch('core_agent.MCP_AVAILABLE', False):
            config = AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                enable_mcp=True
            )
            manager = MCPManager(config)
            self.assertIsNone(manager.mcp_client)


class TestAsyncOperations(unittest.TestCase):
    """Test asynchronous operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.config = AgentConfig(
            name="TestAgent",
            model=self.mock_model
        )
    
    @patch('core_agent.create_react_agent')
    async def test_async_invoke(self, mock_create_react):
        """Test asynchronous invoke"""
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"messages": []}
        mock_create_react.return_value = mock_graph
        
        agent = CoreAgent(self.config)
        agent.compiled_graph = mock_graph
        
        result = await agent.ainvoke("test input")
        
        self.assertIsInstance(result, dict)
        mock_graph.ainvoke.assert_called_once()
    
    @patch('core_agent.create_react_agent')
    async def test_async_stream(self, mock_create_react):
        """Test asynchronous streaming"""
        mock_graph = AsyncMock()
        
        async def mock_astream(*args, **kwargs):
            yield {"chunk": 1}
            yield {"chunk": 2}
        
        mock_graph.astream = mock_astream
        mock_create_react.return_value = mock_graph
        
        agent = CoreAgent(self.config)
        agent.compiled_graph = mock_graph
        
        chunks = []
        async for chunk in agent.astream("test input"):
            chunks.append(chunk)
        
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["chunk"], 1)
        self.assertEqual(chunks[1]["chunk"], 2)
    
    async def test_mcp_async_operations(self):
        """Test MCP async operations"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            enable_mcp=True
        )
        
        with patch('core_agent.MCP_AVAILABLE', True):
            manager = MCPManager(config)
            manager.mcp_client = AsyncMock()
            manager.mcp_client.get_tools.return_value = []
            
            tools = await manager.get_mcp_tools()
            self.assertIsInstance(tools, list)


class TestMultiAgentOperations(unittest.TestCase):
    """Test multi-agent coordination operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_agents = {
            "agent1": Mock(),
            "agent2": Mock()
        }
    
    def test_supervisor_coordination(self):
        """Test supervisor coordination"""
        config = AgentConfig(
            name="SupervisorAgent",
            model=self.mock_model,
            enable_supervisor=True,
            agents=self.mock_agents
        )
        
        manager = SupervisorManager(config)
        result = manager.coordinate_agents("test task")
        
        self.assertIsInstance(result, dict)
        self.assertIn("task", result)
    
    def test_swarm_agent_creation(self):
        """Test swarm agent creation and configuration"""
        config = AgentConfig(
            name="SwarmAgent",
            model=self.mock_model,
            enable_swarm=True,
            agents=self.mock_agents,
            default_active_agent="agent1"
        )
        
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_create_react.return_value = Mock()
            agent = CoreAgent(config)
            
            self.assertTrue(agent.config.enable_swarm)
            self.assertEqual(agent.config.default_active_agent, "agent1")
    
    def test_handoff_agent_creation(self):
        """Test handoff agent creation and configuration"""
        config = AgentConfig(
            name="HandoffAgent",
            model=self.mock_model,
            enable_handoff=True,
            agents=self.mock_agents,
            handoff_agents=["agent1", "agent2"]
        )
        
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_create_react.return_value = Mock()
            agent = CoreAgent(config)
            
            self.assertTrue(agent.config.enable_handoff)
            self.assertEqual(agent.config.handoff_agents, ["agent1", "agent2"])


class TestPerformanceAndMemory(unittest.TestCase):
    """Test performance and memory usage"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_memory_cleanup(self):
        """Test memory cleanup and garbage collection"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            enable_short_term_memory=True, enable_long_term_memory=True
        )
        
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_create_react.return_value = Mock()
            agent = CoreAgent(config)
            
            # Store some data
            agent.store_memory("key1", "value1")
            agent.store_memory("key2", "value2")
            
            # Verify data exists
            self.assertEqual(agent.retrieve_memory("key1"), "value1")
            self.assertEqual(agent.retrieve_memory("key2"), "value2")
            
            # Clean up agent
            del agent
            # Memory should be handled by garbage collection
    
    def test_large_message_handling(self):
        """Test handling of large messages"""
        large_content = "x" * 10000  # Large string
        
        state = CoreAgentState(
            messages=[HumanMessage(content=large_content)]
        )
        
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(len(state.messages[0].content), 10000)
    
    def test_many_tools_handling(self):
        """Test handling of many tools"""
        # Create many mock tools
        tools = []
        for i in range(100):
            tool = Mock(spec=BaseTool)
            tool.name = f"tool_{i}"
            tools.append(tool)
        
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            tools=tools
        )
        
        with patch('core_agent.create_react_agent') as mock_create_react:
            mock_create_react.return_value = Mock()
            agent = CoreAgent(config)
            
            self.assertEqual(len(agent.config.tools), 100)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestAgentConfig,
        TestCoreAgentState,
        TestSubgraphManager,
        TestMemoryManager,
        TestSupervisorManager,
        TestMCPManager,
        TestEvaluationManager,
        TestCoreAgent,
        TestFactoryFunctions,
        TestErrorHandling,
        TestOptionalFeatures,
        TestAsyncOperations,
        TestMultiAgentOperations,
        TestPerformanceAndMemory
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result


if __name__ == "__main__":
    # Run tests if script is executed directly
    result = run_tests()
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPREHENSIVE COREAGENT TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("🎉 ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED!")
    
    print(f"{'='*60}")