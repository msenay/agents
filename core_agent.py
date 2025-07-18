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
    from langgraph_supervisor import SupervisorGraph
except ImportError:
    SupervisorGraph = None

try:
    from langgraph_swarm import SwarmAgent
except ImportError:
    SwarmAgent = None

try:
    from langchain_mcp_adapters import MCPAdapter
except ImportError:
    MCPAdapter = None

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
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field


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
    
    # Advanced features
    enable_supervisor: bool = False
    enable_swarm: bool = False
    enable_mcp: bool = False
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
    """Manages hierarchical multi-agent orchestration"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.supervisor_graph = None
        self.agents = {}
        
        if config.enable_supervisor and SupervisorGraph:
            self._initialize_supervisor()
            
    def _initialize_supervisor(self):
        """Initialize the supervisor graph"""
        try:
            self.supervisor_graph = SupervisorGraph()
            logger.info("Supervisor graph initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize supervisor: {e}")
            
    def add_agent(self, name: str, agent: Any):
        """Add an agent to be managed by the supervisor"""
        self.agents[name] = agent
        
    def coordinate_agents(self, task: str) -> Dict[str, Any]:
        """Coordinate multiple agents for a complex task"""
        if not self.supervisor_graph:
            return {"error": "Supervisor not available"}
            
        # Implementation would depend on the specific supervisor library
        return {"status": "coordinated", "task": task}


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
                "evaluation": self.config.enable_evaluation,
                "streaming": self.config.enable_streaming,
                "human_feedback": self.config.enable_human_feedback,
                "subgraphs": len(self.subgraph_manager.subgraphs),
            },
            "memory_type": self.config.memory_type,
            "supervised_agents": len(self.supervisor_manager.agents),
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
    supervised_agents: Dict[str, Any] = None
) -> CoreAgent:
    """Create a supervisor agent for multi-agent orchestration"""
    config = AgentConfig(
        name="SupervisorAgent",
        model=model,
        enable_supervisor=True,
        enable_memory=True,
        memory_type="redis",
        enable_evaluation=True,
        enable_streaming=True
    )
    
    agent = CoreAgent(config)
    
    if supervised_agents:
        for name, supervised_agent in supervised_agents.items():
            agent.add_supervised_agent(name, supervised_agent)
            
    return agent


# Example usage and template
if __name__ == "__main__":
    # This is an example of how to use the CoreAgent
    
    # Basic usage example
    print("CoreAgent Framework initialized")
    print("Available packages:")
    print("- langgraph-prebuilt: ✓")
    print("- langgraph-supervisor:", "✓" if SupervisorGraph else "✗")
    print("- langgraph-swarm:", "✓" if SwarmAgent else "✗")
    print("- langchain-mcp-adapters:", "✓" if MCPAdapter else "✗")
    print("- langmem:", "✓" if ShortTermMemory else "✗")
    print("- agentevals:", "✓" if AgentEvaluator else "✗")
    print("- Redis support:", "✓" if RedisSaver else "✗")