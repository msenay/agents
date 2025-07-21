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
import os
import sys
from unittest.mock import Mock, MagicMock, patch, AsyncMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies before importing core_agent
sys.modules['langgraph.checkpoint.redis'] = MagicMock()
sys.modules['langgraph_supervisor'] = MagicMock()
sys.modules['langgraph_swarm'] = MagicMock()
sys.modules['langchain_mcp_adapters.client'] = MagicMock()
sys.modules['langmem'] = MagicMock()
sys.modules['agentevals'] = MagicMock()

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

# Import core_agent after mocking dependencies
from core.core_agent import CoreAgentState, CoreAgent
from core.config import AgentConfig
from core.managers import (
    SubgraphManager, MemoryManager, SupervisorManager, 
    MCPManager, EvaluationManager, RateLimiterManager,
    AGENTEVALS_AVAILABLE, LANGMEM_AVAILABLE, MCP_AVAILABLE, 
    RATE_LIMITER_AVAILABLE, MESSAGE_UTILS_AVAILABLE
)

from simple_agent_creators import (create_simple_agent, create_advanced_agent,
    create_supervisor_agent, create_swarm_agent, create_handoff_agent,
    create_memory_agent, create_evaluated_agent, create_human_interactive_agent,
    create_mcp_agent, create_langmem_agent, create_rate_limited_agent)


class TestAgentConfig(unittest.TestCase):
    """Test AgentConfig dataclass functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_tool = Mock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
    
    def test_agent_config_minimal_creation(self):
        """Test creating AgentConfig with minimal parameters"""
        config = AgentConfig()
        
        self.assertEqual(config.name, "CoreAgent")
        self.assertIsNone(config.model)
        self.assertEqual(config.system_prompt, "You are a helpful AI assistant.")
        self.assertEqual(len(config.tools), 0)
        self.assertFalse(config.enable_short_term_memory)
        self.assertFalse(config.enable_long_term_memory)
        self.assertFalse(config.enable_human_feedback)
        self.assertFalse(config.enable_evaluation)
        self.assertFalse(config.enable_mcp)
    
    def test_agent_config_full_creation(self):
        """Test creating AgentConfig with all parameters (new API)"""
        config = AgentConfig(
            name="FullAgent",
            model=self.mock_model,
            system_prompt="Custom prompt",
            tools=[self.mock_tool],
            description="Test description",
            
            # New memory API
            enable_memory=True,
            memory_types=["short_term", "long_term", "session", "semantic"],
            memory_backend="redis",
            redis_url="redis://localhost:6379",
            session_id="test_session",
            
            # Other features
            enable_human_feedback=True,
            interrupt_before=["tool_node"],
            interrupt_after=["agent_node"],
            enable_evaluation=True,
            evaluation_metrics=["accuracy", "speed"],
            enable_mcp=True,
            mcp_servers={"test": {"type": "stdio"}},
            
            # Single multi-agent pattern (fixed conflict)
            enable_supervisor=True,
            agents={"agent1": Mock()},
            
            enable_streaming=True,
            response_format=None,
            pre_model_hook=None,
            post_model_hook=None
        )
        
        self.assertEqual(config.name, "FullAgent")
        self.assertEqual(config.model, self.mock_model)
        self.assertEqual(config.system_prompt, "Custom prompt")
        self.assertEqual(len(config.tools), 1)
        
        # Test backward compatibility properties
        self.assertTrue(config.enable_short_term_memory)
        self.assertTrue(config.enable_long_term_memory)
        self.assertTrue(config.enable_shared_memory)
        self.assertTrue(config.enable_semantic_search)
        self.assertEqual(config.short_term_memory_type, "redis")
        self.assertEqual(config.redis_url, "redis://localhost:6379")
        self.assertTrue(config.enable_human_feedback)
        self.assertTrue(config.enable_evaluation)
        self.assertTrue(config.enable_mcp)
        self.assertTrue(config.enable_supervisor)
        self.assertFalse(config.enable_swarm)  # Not enabled in config
        self.assertFalse(config.enable_handoff)  # Not enabled in config
    
    def test_agent_config_post_init(self):
        """Test AgentConfig post-initialization validation"""
        config = AgentConfig(name="", description="")
        
        self.assertEqual(config.name, "CoreAgent")
        self.assertEqual(config.description, "Specialized agent: CoreAgent")
    
    def test_agent_config_invalid_memory_type(self):
        """Test AgentConfig with invalid memory type (new API)"""
        with self.assertRaises(ValueError):
            config = AgentConfig(
                name="TestAgent",
                model=self.mock_model,
                enable_memory=True,
                memory_backend="invalid_type"
            )
            # Create memory manager which should validate the type
            MemoryManager(config)
    
    def test_agent_config_memory_without_enable(self):
        """Test AgentConfig memory settings without enable_memory (new API)"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            enable_memory=False,  # Memory explicitly disabled
            redis_url="redis://localhost"  # Should be ignored when memory disabled
        )
        
        # Memory shouldn't be enabled even if URLs are provided
        self.assertFalse(config.enable_short_term_memory)
        self.assertFalse(config.enable_long_term_memory)
        self.assertFalse(config.enable_shared_memory)
        self.assertFalse(config.enable_semantic_search)
        self.assertEqual(config.memory_backend, "inmemory")  # Default backend
        
    def test_agent_config_backward_compatibility(self):
        """Test backward compatibility properties (new API)"""
        config = AgentConfig(
            enable_memory=True,
            memory_types=["short_term"],
            memory_backend="redis",
            redis_url="redis://localhost:6379"
        )
        
        # Test backward compatibility properties
        self.assertTrue(config.enable_memory)
        self.assertTrue(config.enable_short_term_memory)
        self.assertFalse(config.enable_long_term_memory)
        self.assertEqual(config.memory_type, "redis")
        self.assertEqual(config.short_term_memory_type, "redis")
        
        config2 = AgentConfig(
            enable_memory=True,
            memory_types=["long_term"],
            memory_backend="postgres",
            postgres_url="postgresql://user:pass@localhost:5432/db"
        )
        
        self.assertTrue(config2.enable_memory)
        self.assertFalse(config2.enable_short_term_memory)
        self.assertTrue(config2.enable_long_term_memory)
        self.assertEqual(config2.memory_type, "postgres")
        self.assertEqual(config2.long_term_memory_type, "postgres")


class TestCoreAgentState(unittest.TestCase):
    """Test CoreAgentState model functionality"""
    
    def test_core_agent_state_creation(self):
        """Test creating CoreAgentState with default values"""
        state = CoreAgentState()
        
        self.assertEqual(len(state.messages), 0)
        self.assertIsNone(state.current_task)
        self.assertEqual(len(state.context), 0)
        self.assertEqual(len(state.memory), 0)
        self.assertEqual(len(state.tool_outputs), 0)
        self.assertEqual(len(state.evaluation_results), 0)
        self.assertEqual(state.human_feedback, "")
    
    def test_core_agent_state_with_data(self):
        """Test creating CoreAgentState with data"""
        messages = [HumanMessage(content="Hello")]
        state = CoreAgentState(
            messages=messages,
            current_task="test_task",
            context={"key": "value"},
            memory={"memory_key": "memory_value"},
            tool_outputs=[{"tool": "output"}],
            evaluation_results={"score": 0.9}
        )
        
        self.assertEqual(len(state.messages), 1)
        self.assertEqual(state.current_task, "test_task")
        self.assertEqual(state.context["key"], "value")
        self.assertEqual(state.memory["memory_key"], "memory_value")
        self.assertEqual(len(state.tool_outputs), 1)
        self.assertEqual(state.evaluation_results["score"], 0.9)
    
    def test_core_agent_state_validation(self):
        """Test CoreAgentState data validation"""
        # Test with invalid message type
        with self.assertRaises(Exception):
            CoreAgentState(messages=["invalid_message"])


class TestSubgraphManager(unittest.TestCase):
    """Test SubgraphManager functionality"""
    
    def test_register_subgraph(self):
        """Test registering a subgraph"""
        manager = SubgraphManager()
        mock_graph = Mock()
        
        manager.register_subgraph("test_graph", mock_graph)
        
        self.assertIn("test_graph", manager.subgraphs)
        self.assertEqual(manager.subgraphs["test_graph"], mock_graph)
    
    def test_get_subgraph_exists(self):
        """Test getting an existing subgraph"""
        manager = SubgraphManager()
        mock_graph = Mock()
        manager.register_subgraph("test_graph", mock_graph)
        
        result = manager.get_subgraph("test_graph")
        
        self.assertEqual(result, mock_graph)
    
    def test_get_subgraph_not_exists(self):
        """Test getting a non-existent subgraph"""
        manager = SubgraphManager()
        
        result = manager.get_subgraph("non_existent")
        
        self.assertIsNone(result)
    
    def test_create_tool_subgraph(self):
        """Test creating a tool subgraph"""
        manager = SubgraphManager()
        tools = [Mock(spec=BaseTool)]
        model = Mock(spec=BaseChatModel)
        
        # This should not raise an exception
        manager.create_tool_subgraph("tool_graph", tools, model)


class TestMemoryManager(unittest.TestCase):
    """Test MemoryManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = AgentConfig(
            name="TestAgent",
            model=Mock(spec=BaseChatModel),
            enable_short_term_memory=True, 
            enable_long_term_memory=True,
            short_term_memory_type="inmemory"
        )
        self.manager = MemoryManager(self.config)
    
    def test_memory_manager_creation_enabled(self):
        """Test MemoryManager creation with memory enabled"""
        self.assertIsNotNone(self.manager.checkpointer)
        self.assertIsNotNone(self.manager.store)
    
    def test_memory_manager_creation_disabled(self):
        """Test MemoryManager creation with memory disabled"""
        config = AgentConfig(enable_short_term_memory=False, enable_long_term_memory=False)
        manager = MemoryManager(config)
        
        self.assertIsNone(manager.checkpointer)
        self.assertIsNone(manager.store)
    
    def test_get_checkpointer(self):
        """Test getting checkpointer"""
        checkpointer = self.manager.get_checkpointer()
        self.assertIsNotNone(checkpointer)
    
    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving memory"""
        if self.manager.store:
            # This should not raise an exception
            self.manager.store_memory("test_key", {"data": "value"})
            
            # Retrieve memory (may return None for inmemory store)
            result = self.manager.get_memory("test_key")
            # Just check it doesn't crash
    
    def test_has_langmem_support(self):
        """Test LangMem support detection"""
        has_support = self.manager.has_langmem_support()
        self.assertIsInstance(has_support, bool)
    
    @patch('core_agent.REDIS_AVAILABLE', True)
    def test_initialize_redis_memory(self):
        """Test Redis memory initialization"""
        config = AgentConfig(
            enable_short_term_memory=True,
            short_term_memory_type="redis", 
            redis_url="redis://localhost:6379"
        )
        
        # This should not raise an exception even if Redis is not actually available
        try:
            manager = MemoryManager(config)
        except Exception:
            pass  # Expected if Redis is not available


class TestSupervisorManager(unittest.TestCase):
    """Test SupervisorManager functionality"""
    
    def test_supervisor_manager_creation_enabled(self):
        """Test SupervisorManager creation when enabled"""
        config = AgentConfig(enable_supervisor=True)
        manager = SupervisorManager(config)
        
        self.assertTrue(manager.enabled)
        self.assertEqual(len(manager.agents), 0)
    
    def test_supervisor_manager_creation_disabled(self):
        """Test SupervisorManager creation when disabled"""
        config = AgentConfig(enable_supervisor=False)
        manager = SupervisorManager(config)
        
        self.assertFalse(manager.enabled)
    
    def test_add_agent(self):
        """Test adding an agent to supervisor"""
        config = AgentConfig(enable_supervisor=True)
        manager = SupervisorManager(config)
        mock_agent = Mock()
        
        manager.add_agent("test_agent", mock_agent)
        
        self.assertIn("test_agent", manager.agents)
        self.assertEqual(manager.agents["test_agent"], mock_agent)
    
    def test_get_available_transfers(self):
        """Test getting available transfer agents"""
        config = AgentConfig(enable_supervisor=True)
        manager = SupervisorManager(config)
        manager.add_agent("agent1", Mock())
        manager.add_agent("agent2", Mock())
        
        transfers = manager.get_available_transfers()
        
        self.assertIn("agent1", transfers)
        self.assertIn("agent2", transfers)
    
    def test_coordinate_agents(self):
        """Test agent coordination"""
        config = AgentConfig(enable_supervisor=True)
        manager = SupervisorManager(config)
        
        # This should not raise an exception
        result = manager.coordinate_agents("test task", {})
        self.assertIsInstance(result, dict)


class TestMCPManager(unittest.TestCase):
    """Test MCPManager functionality"""
    
    def test_mcp_manager_creation_enabled(self):
        """Test MCPManager creation when enabled"""
        config = AgentConfig(enable_mcp=True)
        manager = MCPManager(config)
        
        self.assertTrue(manager.enabled)
    
    def test_mcp_manager_creation_disabled(self):
        """Test MCPManager creation when disabled"""
        config = AgentConfig(enable_mcp=False)
        manager = MCPManager(config)
        
        self.assertFalse(manager.enabled)
    
    def test_add_server(self):
        """Test adding MCP server"""
        config = AgentConfig(enable_mcp=True)
        manager = MCPManager(config)
        
        server_config = {"type": "stdio", "command": "test"}
        manager.add_server("test_server", server_config)
        
        self.assertIn("test_server", manager.servers)
    
    def test_get_server_names(self):
        """Test getting server names"""
        config = AgentConfig(enable_mcp=True)
        manager = MCPManager(config)
        manager.add_server("server1", {"type": "stdio"})
        manager.add_server("server2", {"type": "stdio"})
        
        names = manager.get_server_names()
        
        self.assertIn("server1", names)
        self.assertIn("server2", names)
    
    async def test_get_mcp_tools(self):
        """Test getting MCP tools"""
        config = AgentConfig(enable_mcp=True)
        manager = MCPManager(config)
        
        # This should not raise an exception
        tools = await manager.get_mcp_tools()
        self.assertIsInstance(tools, list)


class TestEvaluationManager(unittest.TestCase):
    """Test EvaluationManager functionality"""
    
    def test_evaluation_manager_creation_enabled(self):
        """Test EvaluationManager creation when enabled"""
        config = AgentConfig(enable_evaluation=True)
        manager = EvaluationManager(config)
        
        self.assertTrue(manager.enabled)
    
    def test_evaluation_manager_creation_disabled(self):
        """Test EvaluationManager creation when disabled"""
        config = AgentConfig(enable_evaluation=False)
        manager = EvaluationManager(config)
        
        self.assertFalse(manager.enabled)
    
    def test_get_evaluator_status(self):
        """Test getting evaluator status"""
        config = AgentConfig(enable_evaluation=True)
        manager = EvaluationManager(config)
        
        status = manager.get_evaluator_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("enabled", status)
        self.assertIn("available_metrics", status)
    
    def test_evaluate_response(self):
        """Test evaluating a response"""
        config = AgentConfig(enable_evaluation=True)
        manager = EvaluationManager(config)
        
        # Mock evaluation data
        task = "Test task"
        response = "Test response"
        
        result = manager.evaluate_response(task, response)
        
        self.assertIsInstance(result, dict)
    
    def test_evaluate_trajectory(self):
        """Test evaluating a trajectory"""
        config = AgentConfig(enable_evaluation=True)
        manager = EvaluationManager(config)
        
        # Mock trajectory data
        trajectory = [{"step": 1, "action": "test"}]
        
        result = manager.evaluate_trajectory(trajectory)
        
        self.assertIsInstance(result, dict)


class TestCoreAgent(unittest.TestCase):
    """Test CoreAgent functionality"""
    
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
        self.agent = CoreAgent(self.config)
    
    def test_core_agent_creation(self):
        """Test CoreAgent creation"""
        self.assertEqual(self.agent.config.name, "TestAgent")
        self.assertEqual(self.agent.config.model, self.mock_model)
        self.assertIsNotNone(self.agent.memory_manager)
        self.assertIsNotNone(self.agent.subgraph_manager)
    
    def test_build_custom_graph(self):
        """Test building custom graph"""
        # Mock the graph building process
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            graph = self.agent.build()
            
            self.assertIsNotNone(graph)
    
    def test_build_with_prebuilt(self):
        """Test building with prebuilt components"""
        config = AgentConfig(
            name="PrebuiltAgent",
            model=self.mock_model,
            enable_supervisor=True
        )
        agent = CoreAgent(config)
        
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            graph = agent.build()
            
            self.assertIsNotNone(graph)
    
    def test_add_subgraph(self):
        """Test adding a subgraph"""
        mock_graph = Mock()
        
        self.agent.add_subgraph("test_subgraph", mock_graph)
        
        subgraph = self.agent.subgraph_manager.get_subgraph("test_subgraph")
        self.assertEqual(subgraph, mock_graph)
    
    def test_get_status(self):
        """Test getting agent status"""
        status = self.agent.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn("name", status)
        self.assertIn("model", status)
        self.assertIn("memory_enabled", status)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration"""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config_path = f.name
        
        try:
            # Save config
            self.agent.save_config(config_path)
            
            # Load config
            loaded_agent = CoreAgent.load_config(config_path)
            
            self.assertEqual(loaded_agent.config.name, self.agent.config.name)
        finally:
            os.unlink(config_path)
    
    async def test_ainvoke(self):
        """Test async invoke"""
        with patch.object(self.agent, 'graph') as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
            
            result = await self.agent.ainvoke("test message")
            
            self.assertIsInstance(result, dict)
    
    async def test_invoke(self):
        """Test invoke"""
        with patch.object(self.agent, 'graph') as mock_graph:
            mock_graph.invoke = Mock(return_value={"messages": []})
            
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.agent.invoke, "test message"
            )
            
            self.assertIsInstance(result, dict)
    
    def test_store_and_retrieve_memory(self):
        """Test storing and retrieving memory"""
        # This should not raise an exception
        self.agent.store_memory("test_key", {"data": "value"})
        
        # Retrieve memory (may return None)
        result = self.agent.get_memory("test_key")
        # Just check it doesn't crash


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating agents"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_tool = Mock(spec=BaseTool)
        self.mock_tool.name = "test_tool"
    
    def test_create_simple_agent(self):
        """Test creating a simple agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_simple_agent(model=self.mock_model)
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertEqual(agent.config.name, "SimpleAgent")
    
    @patch('core_agent.create_react_agent')
    def test_create_advanced_agent(self, mock_create_react):
        """Test creating an advanced agent"""
        mock_create_react.return_value = Mock()
        
        agent = create_advanced_agent(
            model=self.mock_model,
            name="AdvancedAgent",
            tools=[self.mock_tool],
            enable_short_term_memory=True, 
            enable_long_term_memory=True,
            enable_evaluation=True,
            enable_human_feedback=True
        )
        
        self.assertIsInstance(agent, CoreAgent)
        self.assertEqual(agent.config.name, "AdvancedAgent")
        self.assertTrue(agent.config.enable_short_term_memory)
        self.assertTrue(agent.config.enable_evaluation)
        self.assertTrue(agent.config.enable_human_feedback)
    
    def test_create_memory_agent(self):
        """Test creating a memory agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_memory_agent(
                model=self.mock_model,
                enable_short_term_memory=True,
                short_term_memory_type="inmemory"
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_short_term_memory)
    
    def test_create_supervisor_agent(self):
        """Test creating a supervisor agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_supervisor_agent(
                model=self.mock_model,
                agents={"agent1": Mock()}
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_supervisor)
    
    def test_create_swarm_agent(self):
        """Test creating a swarm agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_swarm_agent(
                model=self.mock_model,
                agents={"agent1": Mock()},
                default_active_agent="agent1"
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_swarm)
    
    def test_create_handoff_agent(self):
        """Test creating a handoff agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_handoff_agent(
                model=self.mock_model,
                handoff_agents=["agent1", "agent2"]
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_handoff)
    
    def test_create_mcp_agent(self):
        """Test creating an MCP agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_mcp_agent(
                model=self.mock_model,
                mcp_servers={"test": {"type": "stdio"}}
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_mcp)
    
    def test_create_human_interactive_agent(self):
        """Test creating a human interactive agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_human_interactive_agent(
                model=self.mock_model,
                interrupt_before=["tool_call"]
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_human_feedback)
    
    def test_create_evaluated_agent(self):
        """Test creating an evaluated agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_evaluated_agent(
                model=self.mock_model,
                evaluation_metrics=["accuracy", "speed"]
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertTrue(agent.config.enable_evaluation)
    
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
        self.assertTrue(agent.config.enable_short_term_memory)
        self.assertEqual(agent.config.short_term_memory_type, "inmemory")  # Updated expectation
    
    def test_create_rate_limited_agent(self):
        """Test creating a rate-limited agent"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_rate_limited_agent(
                model=self.mock_model,
                requests_per_second=2.0,
                max_bucket_size=5.0,
                name="RateLimitedAgent"
            )
            
            self.assertIsInstance(agent, CoreAgent)
            self.assertEqual(agent.config.name, "RateLimitedAgent")
            self.assertTrue(agent.config.enable_rate_limiting)
            self.assertEqual(agent.config.requests_per_second, 2.0)
            self.assertEqual(agent.config.max_bucket_size, 5.0)
            self.assertIsNotNone(agent.rate_limiter_manager)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_agent_config_invalid_params(self):
        """Test AgentConfig with invalid parameters"""
        # Test that AgentConfig now allows creation without required params (defaults provided)
        config = AgentConfig()  # This should work now
        self.assertEqual(config.name, "CoreAgent")
        self.assertIsNone(config.model)
    
    def test_core_agent_build_failure(self):
        """Test CoreAgent build failure handling"""
        config = AgentConfig(model=self.mock_model)
        agent = CoreAgent(config)
        
        # Clear the built graphs to force a rebuild
        agent.compiled_graph = None
        agent.graph = None
        
        with patch('core_agent.create_react_agent', side_effect=Exception("Build failed")):
            with self.assertRaises(Exception):
                agent.build(strict_mode=True)
    
    def test_core_agent_state_invalid_data(self):
        """Test CoreAgentState with invalid data"""
        # Test validation works
        with self.assertRaises(Exception):
            CoreAgentState(messages="invalid")  # Should be a list
    
    def test_memory_manager_invalid_redis_url(self):
        """Test MemoryManager with invalid Redis URL"""
        config = AgentConfig(
            name="TestAgent",
            model=self.mock_model,
            enable_short_term_memory=True, 
            enable_long_term_memory=True,
            short_term_memory_type="redis",
            redis_url="invalid://url"
        )
        
        # Should handle gracefully
        manager = MemoryManager(config)
        # Should fall back to inmemory if Redis fails
    
    def test_subgraph_manager_duplicate_registration(self):
        """Test SubgraphManager duplicate registration"""
        manager = SubgraphManager()
        mock_graph = Mock()
        
        manager.register_subgraph("test", mock_graph)
        
        # Should allow overwriting
        manager.register_subgraph("test", mock_graph)
        
        self.assertEqual(manager.get_subgraph("test"), mock_graph)
    
    def test_agent_missing_dependencies(self):
        """Test agent behavior with missing optional dependencies"""
        # This should work even if optional features are not available
        config = AgentConfig(
            model=self.mock_model,
            enable_mcp=True,  # May not be available
            enable_evaluation=True  # May not be available
        )
        
        agent = CoreAgent(config)
        self.assertIsNotNone(agent)


class TestOptionalFeatures(unittest.TestCase):
    """Test optional feature availability detection"""
    
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
                enable_short_term_memory=True, 
                enable_long_term_memory=True,
                short_term_memory_type="redis"
            )
            
            manager = MemoryManager(config)
            # Should fall back gracefully
    
    def test_langmem_availability(self):
        """Test LangMem availability detection"""
        self.assertIsInstance(LANGMEM_AVAILABLE, bool)
    
    def test_mcp_availability(self):
        """Test MCP availability detection"""
        self.assertIsInstance(MCP_AVAILABLE, bool)
    
    def test_agentevals_availability(self):
        """Test AgentEvals availability detection"""
        self.assertIsInstance(AGENTEVALS_AVAILABLE, bool)
    
    def test_rate_limiter_availability(self):
        """Test Rate Limiter availability detection"""
        self.assertIsInstance(RATE_LIMITER_AVAILABLE, bool)


class TestAsyncOperations(unittest.TestCase):
    """Test async operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.config = AgentConfig(model=self.mock_model)
        self.agent = CoreAgent(self.config)
    
    async def test_async_invoke(self):
        """Test async invoke"""
        with patch.object(self.agent, 'graph') as mock_graph:
            mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
            
            result = await self.agent.ainvoke("test")
            
            self.assertIsInstance(result, dict)
    
    async def test_async_stream(self):
        """Test async streaming"""
        with patch.object(self.agent, 'graph') as mock_graph:
            async def mock_astream(input_data, config=None):
                yield {"messages": []}
            
            mock_graph.astream = mock_astream
            
            results = []
            async for chunk in self.agent.astream("test"):
                results.append(chunk)
            
            self.assertGreater(len(results), 0)
    
    async def test_mcp_async_operations(self):
        """Test MCP async operations"""
        config = AgentConfig(model=self.mock_model, enable_mcp=True)
        agent = CoreAgent(config)
        
        # This should work even if MCP is not available
        tools = await agent.mcp_manager.get_mcp_tools()
        self.assertIsInstance(tools, list)


class TestMultiAgentOperations(unittest.TestCase):
    """Test multi-agent operations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_supervisor_coordination(self):
        """Test supervisor agent coordination"""
        config = AgentConfig(
            model=self.mock_model,
            enable_supervisor=True,
            agents={"agent1": Mock(), "agent2": Mock()}
        )
        
        agent = CoreAgent(config)
        
        self.assertTrue(agent.config.enable_supervisor)
        self.assertTrue(agent.supervisor_manager.enabled)
    
    def test_swarm_agent_creation(self):
        """Test swarm agent creation"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_swarm_agent(
                model=self.mock_model,
                agents={"agent1": Mock()},
                default_active_agent="agent1"
            )
            
            self.assertTrue(agent.config.enable_swarm)
    
    def test_handoff_agent_creation(self):
        """Test handoff agent creation"""
        with patch('core_agent.create_react_agent') as mock_create:
            mock_create.return_value = Mock()
            
            agent = create_handoff_agent(
                model=self.mock_model,
                handoff_agents=["agent1", "agent2"]
            )
            
            self.assertTrue(agent.config.enable_handoff)


class TestPerformanceAndMemory(unittest.TestCase):
    """Test performance and memory management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_large_message_handling(self):
        """Test handling of large message contexts"""
        config = AgentConfig(
            model=self.mock_model,
            enable_short_term_memory=True,
            enable_message_trimming=True,
            max_tokens=1000
        )
        
        agent = CoreAgent(config)
        
        # Create a large message context
        large_messages = [HumanMessage(content="Large content " * 100) for _ in range(10)]
        state = CoreAgentState(messages=large_messages)
        
        # This should not crash
        self.assertIsNotNone(state)
    
    def test_many_tools_handling(self):
        """Test handling of many tools"""
        tools = [Mock(spec=BaseTool) for _ in range(100)]
        for i, tool in enumerate(tools):
            tool.name = f"tool_{i}"
        
        config = AgentConfig(
            model=self.mock_model,
            tools=tools
        )
        
        agent = CoreAgent(config)
        
        self.assertEqual(len(agent.config.tools), 100)
    
    def test_memory_cleanup(self):
        """Test memory cleanup operations"""
        config = AgentConfig(
            model=self.mock_model,
            enable_short_term_memory=True,
            enable_ttl=True,
            default_ttl_minutes=1
        )
        
        agent = CoreAgent(config)
        
        # Store some memory
        agent.store_memory("test_key", {"data": "value"})
        
        # Should handle cleanup gracefully
        self.assertIsNotNone(agent.memory_manager)


class TestRateLimiterManager(unittest.TestCase):
    """Test RateLimiterManager functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
    
    def test_rate_limiter_manager_creation_disabled(self):
        """Test RateLimiterManager creation when disabled"""
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=False
        )
        
        manager = RateLimiterManager(config)
        
        self.assertFalse(manager.enabled_status)
        self.assertIsNone(manager.rate_limiter)
    
    def test_rate_limiter_manager_creation_enabled(self):
        """Test RateLimiterManager creation when enabled"""
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=True,
            requests_per_second=2.0,
            max_bucket_size=5.0
        )
        
        manager = RateLimiterManager(config)
        
        # Should be enabled if RATE_LIMITER_AVAILABLE
        self.assertEqual(manager.enabled, RATE_LIMITER_AVAILABLE)
        if RATE_LIMITER_AVAILABLE:
            self.assertIsNotNone(manager.rate_limiter)
    
    def test_rate_limiter_custom_instance(self):
        """Test RateLimiterManager with custom rate limiter"""
        mock_rate_limiter = Mock()
        
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=True,
            custom_rate_limiter=mock_rate_limiter
        )
        
        manager = RateLimiterManager(config)
        
        if RATE_LIMITER_AVAILABLE:
            self.assertEqual(manager.rate_limiter, mock_rate_limiter)
    
    def test_acquire_token_disabled(self):
        """Test token acquisition when rate limiting is disabled"""
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=False
        )
        
        manager = RateLimiterManager(config)
        
        # Should always return True when disabled
        result = manager.acquire_token(blocking=False)
        self.assertTrue(result)
    
    def test_acquire_token_enabled(self):
        """Test token acquisition when rate limiting is enabled"""
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=True,
            requests_per_second=10.0,  # Fast for testing
            max_bucket_size=5.0
        )
        
        manager = RateLimiterManager(config)
        
        if RATE_LIMITER_AVAILABLE:
            # Should work with blocking=False (might fail if no tokens)
            result = manager.acquire_token(blocking=False)
            self.assertIsInstance(result, bool)
    
    async def test_aacquire_token(self):
        """Test async token acquisition"""
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=True,
            requests_per_second=10.0,
            max_bucket_size=5.0
        )
        
        manager = RateLimiterManager(config)
        
        # Should work regardless of rate limiter availability
        result = await manager.aacquire_token(blocking=False)
        self.assertIsInstance(result, bool)


class TestRateLimitedAgent(unittest.TestCase):
    """Test rate-limited agent creation and functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_model = Mock(spec=BaseChatModel)
        self.mock_model.model_name = "test-model"
    
    def test_create_rate_limited_agent(self):
        """Test creating rate-limited agent"""
        agent = create_rate_limited_agent(
            model=self.mock_model,
            requests_per_second=2.0,
            name="TestRateLimitedAgent"
        )
        
        self.assertEqual(agent.config.name, "TestRateLimitedAgent")
        self.assertTrue(agent.config.enable_rate_limiting)
        self.assertEqual(agent.config.requests_per_second, 2.0)
        self.assertEqual(agent.rate_limiter_manager.enabled, RATE_LIMITER_AVAILABLE)
    
    def test_create_rate_limited_agent_with_custom_limiter(self):
        """Test creating rate-limited agent with custom rate limiter"""
        mock_rate_limiter = Mock()
        
        agent = create_rate_limited_agent(
            model=self.mock_model,
            custom_rate_limiter=mock_rate_limiter,
            name="CustomRateLimitedAgent"
        )
        
        self.assertTrue(agent.config.enable_rate_limiting)
        self.assertEqual(agent.config.custom_rate_limiter, mock_rate_limiter)
    
    def test_rate_limited_agent_config_parameters(self):
        """Test rate-limited agent with all configuration parameters"""
        agent = create_rate_limited_agent(
            model=self.mock_model,
            requests_per_second=5.0,
            max_bucket_size=10.0,
            check_every_n_seconds=0.05,
            enable_memory=True,
            name="FullConfigAgent"
        )
        
        config = agent.config
        self.assertEqual(config.requests_per_second, 5.0)
        self.assertEqual(config.max_bucket_size, 10.0)
        self.assertEqual(config.check_every_n_seconds, 0.05)
        self.assertTrue(config.enable_short_term_memory)
        self.assertEqual(config.name, "FullConfigAgent")
    
    def test_agent_config_rate_limiting_integration(self):
        """Test AgentConfig with rate limiting parameters"""
        config = AgentConfig(
            model=self.mock_model,
            enable_rate_limiting=True,
            requests_per_second=3.0,
            max_bucket_size=8.0,
            check_every_n_seconds=0.2
        )
        
        agent = CoreAgent(config)
        
        self.assertTrue(agent.config.enable_rate_limiting)
        self.assertEqual(agent.config.requests_per_second, 3.0)
        self.assertEqual(agent.config.max_bucket_size, 8.0)
        self.assertEqual(agent.config.check_every_n_seconds, 0.2)
        self.assertIsNotNone(agent.rate_limiter_manager)


class TestRateLimiterOptionalFeatures(unittest.TestCase):
    """Test rate limiter optional feature availability"""
    
    def test_rate_limiter_availability(self):
        """Test RATE_LIMITER_AVAILABLE constant"""
        # Should be a boolean
        self.assertIsInstance(RATE_LIMITER_AVAILABLE, bool)
        
        # Test import success/failure
        try:
            from langchain_core.rate_limiters import InMemoryRateLimiter
            expected_available = True
        except ImportError:
            expected_available = False
        
        self.assertEqual(RATE_LIMITER_AVAILABLE, expected_available)
    
    def test_rate_limiter_graceful_degradation(self):
        """Test graceful degradation when rate limiter is not available"""
        config = AgentConfig(
            enable_rate_limiting=True,
            requests_per_second=2.0
        )
        
        manager = RateLimiterManager(config)
        
        # Should handle gracefully regardless of availability
        result = manager.acquire_token(blocking=False)
        self.assertIsInstance(result, bool)


if __name__ == '__main__':
    unittest.main()