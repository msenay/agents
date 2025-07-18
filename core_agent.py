"""
Core Agent Framework - Comprehensive LangGraph Agent with Optional Features

This core agent provides a foundation for creating specialized agents with:
1. Subgraph encapsulation for reusable components
2. Persistent memory with RedisSaver for multi-session long memory
3. SupervisorGraph for hierarchical multi-agent orchestration
4. All langgraph prebuilt components
5. Memory management (short-term and long-term)
6. Agent evaluation utilities
7. MCP server integration
8. Human-in-the-loop capabilities
9. Streaming support
"""

from typing import Optional, List, Dict, Any, Callable, Union, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import json
import logging
from pathlib import Path

# Core LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import Command

# LangGraph ecosystem imports
try:
    from langgraph.checkpoint.redis import RedisSaver
except ImportError:
    RedisSaver = None
    
try:
    from langgraph_supervisor import create_supervisor
except ImportError:
    create_supervisor = None

try:
    from langgraph_swarm import create_swarm, create_handoff_tool as swarm_handoff_tool
except ImportError:
    create_swarm = None
    swarm_handoff_tool = None

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_AVAILABLE = True
except ImportError:
    MultiServerMCPClient = None
    MCP_AVAILABLE = False

try:
    from langmem import ShortTermMemory, LongTermMemory
except ImportError:
    ShortTermMemory = None
    LongTermMemory = None

try:
    from agentevals import AgentEvaluator
except ImportError:
    AgentEvaluator = None

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool, tool, InjectedToolCallId
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing import Annotated


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration class for core agent settings"""
    
    # Core settings
    name: str = "CoreAgent"
    description: str = "A comprehensive core agent with optional features"
    
    # Model settings
    model: Optional[BaseChatModel] = None
    system_prompt: str = "You are a helpful AI assistant."
    
    # Tools and capabilities
    tools: List[BaseTool] = field(default_factory=list)
    tool_calling_enabled: bool = True
    
    # Memory settings
    enable_memory: bool = True
    memory_type: str = "memory"  # "memory", "redis", or "both"
    redis_url: Optional[str] = None
    
    # Multi-agent settings
    enable_supervisor: bool = False
    enable_swarm: bool = False
    enable_handoff: bool = False
    agents: Dict[str, Any] = field(default_factory=dict)  # For multi-agent systems
    default_active_agent: Optional[str] = None  # For swarm systems
    
    # Advanced features
    enable_mcp: bool = False
    mcp_servers: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # MCP server configurations
    enable_evaluation: bool = False
    
    # Hooks and customization
    pre_model_hook: Optional[Callable] = None
    post_model_hook: Optional[Callable] = None
    response_format: Optional[Type[BaseModel]] = None
    
    # Human-in-the-loop
    enable_human_feedback: bool = False
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)
    
    # Streaming
    enable_streaming: bool = True
    
    # Subgraph settings
    enable_subgraphs: bool = False
    subgraph_configs: Dict[str, Any] = field(default_factory=dict)
    
    # Evaluation settings
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "relevance", "helpfulness"])


class CoreAgentState(BaseModel):
    """State definition for the core agent"""
    
    messages: List[BaseMessage] = Field(default_factory=list)
    current_task: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    memory: Dict[str, Any] = Field(default_factory=dict)
    tool_outputs: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_results: Dict[str, Any] = Field(default_factory=dict)
    human_feedback: Optional[str] = None
    supervisor_decisions: List[Dict[str, Any]] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class SubgraphManager:
    """Manages reusable subgraph components"""
    
    def __init__(self):
        self.subgraphs: Dict[str, StateGraph] = {}
        
    def register_subgraph(self, name: str, subgraph: StateGraph):
        """Register a reusable subgraph component"""
        self.subgraphs[name] = subgraph
        logger.info(f"Registered subgraph: {name}")
        
    def get_subgraph(self, name: str) -> Optional[StateGraph]:
        """Get a registered subgraph by name"""
        return self.subgraphs.get(name)
        
    def create_tool_subgraph(self, tools: List[BaseTool]) -> StateGraph:
        """Create a subgraph for tool execution"""
        graph = StateGraph(CoreAgentState)
        
        def tool_executor(state: CoreAgentState):
            # Tool execution logic
            return {"tool_outputs": []}
            
        graph.add_node("execute_tools", tool_executor)
        graph.add_edge(START, "execute_tools")
        graph.add_edge("execute_tools", END)
        
        return graph.compile()


class MemoryManager:
    """Manages both short-term and long-term memory"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.short_term_memory = None
        self.long_term_memory = None
        self.redis_saver = None
        
        self._initialize_memory()
        
    def _initialize_memory(self):
        """Initialize memory components based on configuration"""
        if self.config.enable_memory:
            if self.config.memory_type in ["memory", "both"]:
                self.memory_saver = MemorySaver()
                
            if self.config.memory_type in ["redis", "both"] and RedisSaver:
                if self.config.redis_url:
                    self.redis_saver = RedisSaver.from_conn_string(self.config.redis_url)
                    logger.info("Redis memory initialized")
                    
            if ShortTermMemory:
                self.short_term_memory = ShortTermMemory()
                
            if LongTermMemory:
                self.long_term_memory = LongTermMemory()
                
    def get_checkpointer(self):
        """Get the appropriate checkpointer based on configuration"""
        if self.redis_saver:
            return self.redis_saver
        return getattr(self, 'memory_saver', MemorySaver())
        
    def store_memory(self, key: str, value: Any, memory_type: str = "short"):
        """Store information in memory"""
        if memory_type == "short" and self.short_term_memory:
            self.short_term_memory.store(key, value)
        elif memory_type == "long" and self.long_term_memory:
            self.long_term_memory.store(key, value)
            
    def retrieve_memory(self, key: str, memory_type: str = "short"):
        """Retrieve information from memory"""
        if memory_type == "short" and self.short_term_memory:
            return self.short_term_memory.retrieve(key)
        elif memory_type == "long" and self.long_term_memory:
            return self.long_term_memory.retrieve(key)
        return None


class SupervisorManager:
    """Manages hierarchical multi-agent orchestration with supervisor, swarm, and handoff patterns"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.supervisor_graph = None
        self.swarm_graph = None
        self.handoff_graph = None
        self.agents = {}
        self.handoff_tools = {}
        
        if config.enable_supervisor and create_supervisor:
            self._initialize_supervisor()
        elif config.enable_swarm and create_swarm:
            self._initialize_swarm()
        elif config.enable_handoff:
            self._initialize_handoff()
            
    def _initialize_supervisor(self):
        """Initialize the supervisor graph"""
        try:
            if self.config.agents:
                agents_list = list(self.config.agents.values())
                self.supervisor_graph = create_supervisor(
                    agents=agents_list,
                    model=self.config.model,
                    prompt=self.config.system_prompt
                ).compile()
                logger.info("Supervisor graph initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize supervisor: {e}")
            
    def _initialize_swarm(self):
        """Initialize the swarm graph"""
        try:
            if self.config.agents and create_swarm:
                agents_list = list(self.config.agents.values())
                self.swarm_graph = create_swarm(
                    agents=agents_list,
                    default_active_agent=self.config.default_active_agent or list(self.config.agents.keys())[0]
                ).compile()
                logger.info("Swarm graph initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize swarm: {e}")
            
    def _initialize_handoff(self):
        """Initialize the handoff graph"""
        try:
            if self.config.agents:
                self._create_handoff_tools()
                self._create_handoff_graph()
                logger.info("Handoff graph initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize handoff: {e}")
            
    def _create_handoff_tools(self):
        """Create handoff tools for agent-to-agent transfers"""
        def create_handoff_tool(*, agent_name: str, description: str = None):
            name = f"transfer_to_{agent_name}"
            description = description or f"Transfer to {agent_name}"

            @tool(name, description=description)
            def handoff_tool(
                state: Annotated[dict, InjectedState], 
                tool_call_id: Annotated[str, InjectedToolCallId],
            ) -> Command:
                tool_message = {
                    "role": "tool",
                    "content": f"Successfully transferred to {agent_name}",
                    "name": name,
                    "tool_call_id": tool_call_id,
                }
                return Command(  
                    goto=agent_name,  
                    update={"messages": state.get("messages", []) + [tool_message]},  
                    graph=Command.PARENT,  
                )
            return handoff_tool
            
        # Create handoff tools for each agent
        for agent_name in self.config.agents.keys():
            self.handoff_tools[agent_name] = create_handoff_tool(
                agent_name=agent_name,
                description=f"Transfer user to the {agent_name} assistant."
            )
            
    def _create_handoff_graph(self):
        """Create the handoff graph with all agents"""
        from langgraph.graph import MessagesState, START
        
        self.handoff_graph = StateGraph(MessagesState)
        
        # Add all agents as nodes
        for agent_name, agent in self.config.agents.items():
            self.handoff_graph.add_node(agent_name, agent)
            
        # Set starting agent
        if self.config.default_active_agent:
            self.handoff_graph.add_edge(START, self.config.default_active_agent)
        else:
            # Default to first agent
            first_agent = list(self.config.agents.keys())[0]
            self.handoff_graph.add_edge(START, first_agent)
            
        self.handoff_graph = self.handoff_graph.compile()
             
    def add_agent(self, name: str, agent: Any):
        """Add an agent to be managed by the supervisor"""
        self.agents[name] = agent
        
    def coordinate_agents(self, task: str) -> Dict[str, Any]:
        """Coordinate multiple agents for a complex task"""
        if self.supervisor_graph:
            return self._run_supervisor(task)
        elif self.swarm_graph:
            return self._run_swarm(task)
        elif self.handoff_graph:
            return self._run_handoff(task)
        else:
            return {"error": "No multi-agent system available"}
            
    def _run_supervisor(self, task: str) -> Dict[str, Any]:
        """Run task through supervisor pattern"""
        try:
            result = self.supervisor_graph.invoke({
                "messages": [{"role": "user", "content": task}]
            })
            return {"status": "supervised", "result": result}
        except Exception as e:
            return {"error": f"Supervisor execution failed: {e}"}
            
    def _run_swarm(self, task: str) -> Dict[str, Any]:
        """Run task through swarm pattern"""
        try:
            result = self.swarm_graph.invoke({
                "messages": [{"role": "user", "content": task}]
            })
            return {"status": "swarmed", "result": result}
        except Exception as e:
            return {"error": f"Swarm execution failed: {e}"}
            
    def _run_handoff(self, task: str) -> Dict[str, Any]:
        """Run task through handoff pattern"""
        try:
            result = self.handoff_graph.invoke({
                "messages": [{"role": "user", "content": task}]
            })
            return {"status": "handoff", "result": result}
        except Exception as e:
            return {"error": f"Handoff execution failed: {e}"}
            
    def get_available_transfers(self) -> List[str]:
        """Get list of available transfer targets"""
        return list(self.handoff_tools.keys()) if self.handoff_tools else []


class MCPManager:
    """Manages MCP (Model Context Protocol) server connections and tools"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = None
        self.mcp_tools = []
        
        if config.enable_mcp and MCP_AVAILABLE:
            self._initialize_mcp_client()
            
    def _initialize_mcp_client(self):
        """Initialize MCP client with configured servers"""
        try:
            if self.config.mcp_servers:
                self.client = MultiServerMCPClient(self.config.mcp_servers)
                logger.info("MCP client initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MCP client: {e}")
            
    async def get_mcp_tools(self) -> List[Any]:
        """Get tools from MCP servers"""
        if not self.client:
            return []
            
        try:
            tools = await self.client.get_tools()
            self.mcp_tools = tools
            logger.info(f"Retrieved {len(tools)} tools from MCP servers")
            return tools
        except Exception as e:
            logger.warning(f"Failed to get MCP tools: {e}")
            return []
            
    def get_server_names(self) -> List[str]:
        """Get list of configured MCP server names"""
        return list(self.config.mcp_servers.keys())
        
    def add_server(self, name: str, config: Dict[str, Any]):
        """Add a new MCP server configuration"""
        self.config.mcp_servers[name] = config
        if self.config.enable_mcp and MCP_AVAILABLE:
            self._initialize_mcp_client()


class EvaluationManager:
    """Manages agent performance evaluation"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.evaluator = None
        
        if config.enable_evaluation and AgentEvaluator:
            self.evaluator = AgentEvaluator(metrics=config.evaluation_metrics)
            
    def evaluate_response(self, input_text: str, output_text: str) -> Dict[str, float]:
        """Evaluate agent response quality"""
        if not self.evaluator:
            return {}
            
        try:
            return self.evaluator.evaluate(input_text, output_text)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            return {}


class CoreAgent:
    """
    Core Agent Framework - A comprehensive agent foundation
    
    This class provides a complete agent framework with optional features:
    - Subgraph encapsulation for reusable components
    - Persistent memory with Redis support
    - Supervisor graphs for multi-agent orchestration
    - Memory management (short-term and long-term)
    - Agent evaluation utilities
    - MCP server integration
    - Human-in-the-loop capabilities
    - Streaming support
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = CoreAgentState()
        
        # Initialize managers
        self.subgraph_manager = SubgraphManager()
        self.memory_manager = MemoryManager(config)
        self.supervisor_manager = SupervisorManager(config)
        self.mcp_manager = MCPManager(config)
        self.evaluation_manager = EvaluationManager(config)
        
        # Core graph
        self.graph = None
        self.compiled_graph = None
        
        # Initialize the agent
        self._build_agent()
        
    def _build_agent(self):
        """Build the core agent graph"""
        if self.config.model and hasattr(self.config, 'tools'):
            # Use prebuilt create_react_agent if model and tools are provided
            self._build_with_prebuilt()
        else:
            # Build custom graph
            self._build_custom_graph()
            
    def _build_with_prebuilt(self):
        """Build agent using LangGraph prebuilt components"""
        try:
            kwargs = {
                'model': self.config.model,
                'tools': self.config.tools,
            }
            
            if self.config.pre_model_hook:
                kwargs['pre_model_hook'] = self.config.pre_model_hook
                
            if self.config.post_model_hook:
                kwargs['post_model_hook'] = self.config.post_model_hook
                
            if self.config.response_format:
                kwargs['response_format'] = self.config.response_format
                
            self.compiled_graph = create_react_agent(**kwargs)
            logger.info("Agent built with prebuilt components")
            
        except Exception as e:
            logger.warning(f"Failed to use prebuilt agent: {e}")
            self._build_custom_graph()
            
    def _build_custom_graph(self):
        """Build custom agent graph"""
        self.graph = StateGraph(CoreAgentState)
        
        # Add core nodes
        self.graph.add_node("process_input", self._process_input)
        self.graph.add_node("generate_response", self._generate_response)
        self.graph.add_node("execute_tools", self._execute_tools)
        self.graph.add_node("human_feedback", self._handle_human_feedback)
        
        # Add edges
        self.graph.add_edge(START, "process_input")
        self.graph.add_conditional_edges(
            "process_input",
            self._should_use_tools,
            {
                "tools": "execute_tools",
                "response": "generate_response",
                "human": "human_feedback"
            }
        )
        self.graph.add_edge("execute_tools", "generate_response")
        self.graph.add_edge("human_feedback", "generate_response")
        self.graph.add_edge("generate_response", END)
        
        # Compile with memory
        checkpointer = self.memory_manager.get_checkpointer()
        
        compile_kwargs = {}
        if checkpointer:
            compile_kwargs['checkpointer'] = checkpointer
            
        if self.config.interrupt_before:
            compile_kwargs['interrupt_before'] = self.config.interrupt_before
            
        if self.config.interrupt_after:
            compile_kwargs['interrupt_after'] = self.config.interrupt_after
            
        self.compiled_graph = self.graph.compile(**compile_kwargs)
        logger.info("Custom agent graph built and compiled")
        
    def _process_input(self, state: CoreAgentState) -> Dict[str, Any]:
        """Process input and determine next action"""
        # Add memory retrieval
        relevant_memory = self.memory_manager.retrieve_memory("context")
        if relevant_memory:
            state.context.update(relevant_memory)
            
        return {"context": state.context}
        
    def _should_use_tools(self, state: CoreAgentState) -> str:
        """Determine whether to use tools, generate response, or get human feedback"""
        if self.config.enable_human_feedback and state.human_feedback is None:
            return "human"
        elif self.config.tools and self.config.tool_calling_enabled:
            return "tools"
        else:
            return "response"
            
    def _execute_tools(self, state: CoreAgentState) -> Dict[str, Any]:
        """Execute tools if available"""
        if not self.config.tools:
            return {"tool_outputs": []}
            
        # Tool execution logic would go here
        tool_outputs = []
        
        return {"tool_outputs": tool_outputs}
        
    def _generate_response(self, state: CoreAgentState) -> Dict[str, Any]:
        """Generate the final response"""
        if not self.config.model:
            response = AIMessage(content="No model configured")
        else:
            # Model inference logic would go here
            response = AIMessage(content="Generated response")
            
        # Store in memory
        self.memory_manager.store_memory("last_response", response.content)
        
        # Evaluate if enabled
        if self.config.enable_evaluation and state.messages:
            last_human_msg = next(
                (msg.content for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
                ""
            )
            evaluation = self.evaluation_manager.evaluate_response(last_human_msg, response.content)
            state.evaluation_results.update(evaluation)
            
        state.messages.append(response)
        return {"messages": state.messages, "evaluation_results": state.evaluation_results}
        
    def _handle_human_feedback(self, state: CoreAgentState) -> Dict[str, Any]:
        """Handle human-in-the-loop feedback"""
        # This would typically pause execution until human input is provided
        return {"human_feedback": "processed"}
        
    # Public interface methods
    
    def invoke(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Invoke the agent with input data"""
        if not self.compiled_graph:
            raise ValueError("Agent not properly initialized")
            
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        if "thread_id" not in config:
            config["configurable"] = {"thread_id": "default"}
            
        return self.compiled_graph.invoke(input_data, config=config)
        
    async def ainvoke(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Async invoke the agent"""
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        if "thread_id" not in config:
            config["configurable"] = {"thread_id": "default"}
            
        return await self.compiled_graph.ainvoke(input_data, config=config)
        
    def stream(self, input_data: Union[str, Dict[str, Any]], **kwargs):
        """Stream agent responses"""
        if not self.config.enable_streaming:
            yield self.invoke(input_data, **kwargs)
            return
            
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        if "thread_id" not in config:
            config["configurable"] = {"thread_id": "default"}
            
        for chunk in self.compiled_graph.stream(input_data, config=config):
            yield chunk
            
    async def astream(self, input_data: Union[str, Dict[str, Any]], **kwargs):
        """Async stream agent responses"""
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        if "thread_id" not in config:
            config["configurable"] = {"thread_id": "default"}
            
        async for chunk in self.compiled_graph.astream(input_data, config=config):
            yield chunk
            
    # Subgraph management
    
    def add_subgraph(self, name: str, subgraph: StateGraph):
        """Add a reusable subgraph component"""
        self.subgraph_manager.register_subgraph(name, subgraph)
        
    def get_subgraph(self, name: str) -> Optional[StateGraph]:
        """Get a registered subgraph"""
        return self.subgraph_manager.get_subgraph(name)
        
    # Agent management for supervisor
    
    def add_supervised_agent(self, name: str, agent: Any):
        """Add an agent to be supervised"""
        self.supervisor_manager.add_agent(name, agent)
        
    def coordinate_task(self, task: str) -> Dict[str, Any]:
        """Coordinate a task across multiple agents"""
        return self.supervisor_manager.coordinate_agents(task)
        
    # Memory management
    
    def store_memory(self, key: str, value: Any, memory_type: str = "short"):
        """Store information in memory"""
        self.memory_manager.store_memory(key, value, memory_type)
        
    def retrieve_memory(self, key: str, memory_type: str = "short"):
        """Retrieve information from memory"""
        return self.memory_manager.retrieve_memory(key, memory_type)
        
    # MCP (Model Context Protocol) management
    
    async def get_mcp_tools(self) -> List[Any]:
        """Get tools from configured MCP servers"""
        return await self.mcp_manager.get_mcp_tools()
        
    def add_mcp_server(self, name: str, server_config: Dict[str, Any]):
        """Add a new MCP server configuration"""
        self.mcp_manager.add_server(name, server_config)
        
    def get_mcp_servers(self) -> List[str]:
        """Get list of configured MCP server names"""
        return self.mcp_manager.get_server_names()
        
    async def load_mcp_tools_into_agent(self):
        """Load MCP tools into the agent's tool list"""
        if self.config.enable_mcp:
            mcp_tools = await self.get_mcp_tools()
            self.config.tools.extend(mcp_tools)
            logger.info(f"Added {len(mcp_tools)} MCP tools to agent")
            
    # Evaluation
    
    def evaluate_last_response(self) -> Dict[str, float]:
        """Evaluate the last response"""
        if not self.state.messages:
            return {}
            
        last_messages = self.state.messages[-2:]  # Human + AI message
        if len(last_messages) >= 2:
            human_msg = last_messages[0].content if isinstance(last_messages[0], HumanMessage) else ""
            ai_msg = last_messages[1].content if isinstance(last_messages[1], AIMessage) else ""
            return self.evaluation_manager.evaluate_response(human_msg, ai_msg)
        return {}
        
    # Utility methods
    
    def get_graph_visualization(self) -> bytes:
        """Get a visual representation of the agent graph"""
        if self.compiled_graph and hasattr(self.compiled_graph, 'get_graph'):
            try:
                return self.compiled_graph.get_graph().draw_mermaid_png()
            except Exception as e:
                logger.warning(f"Failed to generate graph visualization: {e}")
        return b""
        
    def save_config(self, filepath: str):
        """Save agent configuration to file"""
        config_dict = {
            "name": self.config.name,
            "description": self.config.description,
            "system_prompt": self.config.system_prompt,
            "enable_memory": self.config.enable_memory,
            "memory_type": self.config.memory_type,
            "tool_calling_enabled": self.config.tool_calling_enabled,
            "enable_supervisor": self.config.enable_supervisor,
            "enable_swarm": self.config.enable_swarm,
            "enable_evaluation": self.config.enable_evaluation,
            "enable_streaming": self.config.enable_streaming,
            "enable_human_feedback": self.config.enable_human_feedback,
            "evaluation_metrics": self.config.evaluation_metrics,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load_config(cls, filepath: str) -> AgentConfig:
        """Load agent configuration from file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        return AgentConfig(**config_dict)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and capabilities"""
        return {
            "name": self.config.name,
            "description": self.config.description,
            "features": {
                "memory": self.config.enable_memory,
                "tools": len(self.config.tools),
                "supervisor": self.config.enable_supervisor,
                "swarm": self.config.enable_swarm,
                "handoff": self.config.enable_handoff,
                "mcp": self.config.enable_mcp,
                "evaluation": self.config.enable_evaluation,
                "streaming": self.config.enable_streaming,
                "human_feedback": self.config.enable_human_feedback,
                "subgraphs": len(self.subgraph_manager.subgraphs),
            },
            "memory_type": self.config.memory_type,
            "supervised_agents": len(self.supervisor_manager.agents),
            "mcp_servers": len(self.config.mcp_servers),
            "mcp_tools": len(self.mcp_manager.mcp_tools),
        }


# Factory functions for common agent types

def create_basic_agent(model: BaseChatModel, tools: List[BaseTool] = None) -> CoreAgent:
    """Create a basic agent with minimal configuration"""
    config = AgentConfig(
        name="BasicAgent",
        model=model,
        tools=tools or [],
        enable_memory=True,
        memory_type="memory"
    )
    return CoreAgent(config)


def create_advanced_agent(
    model: BaseChatModel,
    tools: List[BaseTool] = None,
    enable_redis: bool = False,
    redis_url: str = None,
    enable_supervisor: bool = False,
    enable_evaluation: bool = False
) -> CoreAgent:
    """Create an advanced agent with enhanced capabilities"""
    config = AgentConfig(
        name="AdvancedAgent",
        model=model,
        tools=tools or [],
        enable_memory=True,
        memory_type="redis" if enable_redis else "memory",
        redis_url=redis_url,
        enable_supervisor=enable_supervisor,
        enable_evaluation=enable_evaluation,
        enable_streaming=True,
        enable_human_feedback=True
    )
    return CoreAgent(config)


def create_supervisor_agent(
    model: BaseChatModel,
    agents: Dict[str, Any] = None,
    prompt: str = "You manage multiple specialized agents. Assign work to them based on their capabilities."
) -> CoreAgent:
    """Create a supervisor agent for multi-agent orchestration"""
    config = AgentConfig(
        name="SupervisorAgent",
        model=model,
        system_prompt=prompt,
        enable_supervisor=True,
        agents=agents or {},
        enable_memory=True,
        memory_type="memory",
        enable_evaluation=True,
        enable_streaming=True
    )
    
    return CoreAgent(config)


def create_swarm_agent(
    model: BaseChatModel,
    agents: Dict[str, Any] = None,
    default_active_agent: str = None
) -> CoreAgent:
    """Create a swarm agent for dynamic multi-agent coordination"""
    config = AgentConfig(
        name="SwarmAgent",
        model=model,
        enable_swarm=True,
        agents=agents or {},
        default_active_agent=default_active_agent,
        enable_memory=True,
        memory_type="memory",
        enable_streaming=True
    )
    
    return CoreAgent(config)


def create_handoff_agent(
    model: BaseChatModel,
    agents: Dict[str, Any] = None,
    default_active_agent: str = None,
    prompt: str = "You can transfer conversations to specialized agents when needed."
) -> CoreAgent:
    """Create a handoff agent for manual agent transfers"""
    config = AgentConfig(
        name="HandoffAgent", 
        model=model,
        system_prompt=prompt,
        enable_handoff=True,
        agents=agents or {},
        default_active_agent=default_active_agent,
        enable_memory=True,
        memory_type="memory",
        enable_streaming=True
    )
    
    return CoreAgent(config)


def create_mcp_agent(
    model: BaseChatModel,
    mcp_servers: Dict[str, Dict[str, Any]] = None,
    tools: List[BaseTool] = None,
    prompt: str = "You are an assistant with access to MCP tools and services."
) -> CoreAgent:
    """Create an agent with MCP (Model Context Protocol) support"""
    config = AgentConfig(
        name="MCPAgent",
        model=model,
        system_prompt=prompt,
        tools=tools or [],
        enable_mcp=True,
        mcp_servers=mcp_servers or {},
        enable_memory=True,
        memory_type="memory",
        enable_streaming=True
    )
    
    return CoreAgent(config)


# Example usage and template
if __name__ == "__main__":
    # This is an example of how to use the CoreAgent
    
    # Basic usage example
    print("CoreAgent Framework initialized")
    print("Available packages:")
    print("- langgraph-prebuilt: ✓")
    print("- langgraph-supervisor:", "✓" if create_supervisor else "✗")
    print("- langgraph-swarm:", "✓" if create_swarm else "✗")
    print("- langchain-mcp-adapters:", "✓" if MCP_AVAILABLE else "✗")
    print("- langmem:", "✓" if ShortTermMemory else "✗")
    print("- agentevals:", "✓" if AgentEvaluator else "✗")
    print("- Redis support:", "✓" if RedisSaver else "✗")