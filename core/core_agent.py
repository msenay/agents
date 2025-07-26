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

from typing import Optional, List, Dict, Any, Union
import json
import logging

# Core LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage

from core.config import AgentConfig
from core.managers import (
    SubgraphManager, MemoryManager, SupervisorManager, 
    MCPManager, EvaluationManager, RateLimiterManager
)
from core.model import CoreAgentState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        self.rate_limiter_manager = RateLimiterManager(config)
        
        # Core graph
        self.graph = None
        self.compiled_graph = None
        
        # Initialize the agent
        self._build_agent(strict_mode=False)
        
    def _build_agent(self, strict_mode: bool = False):
        """Build the core agent graph"""
        if self.config.model and hasattr(self.config, 'tools'):
            # Use prebuilt create_react_agent if model and tools are provided
            self._build_with_prebuilt(strict_mode=strict_mode)
        else:
            # Build custom graph
            self._build_custom_graph()
            
    def _prepare_model_with_rate_limiter(self):
        """Prepare model with rate limiter if enabled"""
        if not self.config.model:
            return None
            
        model = self.config.model
        
        # Apply rate limiter if enabled
        if self.rate_limiter_manager.enabled_status:
            rate_limiter = self.rate_limiter_manager.get_rate_limiter()
            
            # Check if model supports rate_limiter parameter
            if hasattr(model, 'rate_limiter') or hasattr(model.__class__, 'rate_limiter'):
                try:
                    # Create a new model instance with rate limiter
                    model_kwargs = {}
                    
                    # Copy existing model parameters
                    if hasattr(model, 'model_name'):
                        model_kwargs['model_name'] = model.model_name
                    if hasattr(model, 'model'):
                        model_kwargs['model'] = model.model
                    if hasattr(model, 'temperature'):
                        model_kwargs['temperature'] = model.temperature
                    if hasattr(model, 'max_tokens'):
                        model_kwargs['max_tokens'] = model.max_tokens
                    if hasattr(model, 'api_key'):
                        model_kwargs['api_key'] = model.api_key
                    
                    # Add rate limiter
                    model_kwargs['rate_limiter'] = rate_limiter
                    
                    # Create new model instance with rate limiter
                    model = model.__class__(**model_kwargs)
                    logger.info(f"Model configured with rate limiter: {self.config.requests_per_second} req/sec")
                    
                except Exception as e:
                    logger.warning(f"Failed to configure model with rate limiter: {e}")
                    logger.info("Continuing with original model without rate limiter")
            else:
                logger.warning("Model does not support rate_limiter parameter. Rate limiting disabled for this model.")
        
        return model

    def _build_with_prebuilt(self, strict_mode: bool = False):
        """Build agent using LangGraph prebuilt components"""
        try:
            # Combine configuration tools with memory tools
            all_tools = list(self.config.tools)
            
            # Add memory management tools if enabled
            if self.config.enable_memory_tools:
                memory_tools = self.memory_manager.get_memory_tools()
                all_tools.extend(memory_tools)
                logger.info(f"Added {len(memory_tools)} memory tools to agent")
            
            # Prepare model with rate limiter
            model_with_rate_limiter = self._prepare_model_with_rate_limiter()
            
            kwargs = {
                'model': model_with_rate_limiter,
                'tools': all_tools,
            }
            
            # Set up pre-model hook for memory management
            pre_model_hook = self.config.pre_model_hook
            memory_hook = self.memory_manager.get_pre_model_hook()
            
            if memory_hook:
                if pre_model_hook:
                    # Chain hooks if both exist
                    def combined_hook(state):
                        state = pre_model_hook(state)
                        return memory_hook(state)
                    kwargs['pre_model_hook'] = combined_hook
                else:
                    kwargs['pre_model_hook'] = memory_hook
                    
                if self.config.enable_message_trimming:
                    logger.info("Using message trimming hook")
                elif self.config.enable_summarization:
                    logger.info("Using LangMem summarization hook")
            elif pre_model_hook:
                kwargs['pre_model_hook'] = pre_model_hook
                
            if self.config.post_model_hook:
                kwargs['post_model_hook'] = self.config.post_model_hook
                
            if self.config.response_format:
                kwargs['response_format'] = self.config.response_format
                
            # Add checkpointer for short-term memory
            if self.config.enable_short_term_memory:
                kwargs['checkpointer'] = self.memory_manager.get_checkpointer()
                
            # Add store for long-term memory
            if self.config.enable_long_term_memory:
                kwargs['store'] = self.memory_manager.get_store()
                
            self.compiled_graph = create_react_agent(**kwargs)
            logger.info("Agent built with prebuilt components and comprehensive memory support")
            
        except Exception as e:
            if strict_mode:
                # Re-raise exception in strict mode
                raise e
            else:
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
        # Add memory retrieval from session or long-term memory
        if self.memory_manager.has_session_memory():
            relevant_memory = self.memory_manager.get_session_memory()
            if relevant_memory:
                state.context.update({"session_memory": relevant_memory})
        elif self.memory_manager.has_long_term_memory():
            # Could retrieve relevant context based on current input
            pass
            
        return {"context": state.context}
        
    def _should_use_tools(self, state: CoreAgentState) -> str:
        """Determine whether to use tools, generate response, or get human feedback"""
        if self.config.enable_human_feedback and state.human_feedback is None:
            return "human"
        elif self.config.tools:
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
            
        # Store in memory if available
        if self.memory_manager.has_long_term_memory():
            self.memory_manager.store_long_term_memory("last_response", response.content)
        
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
        # Ensure configurable exists
        if "configurable" not in config:
            config["configurable"] = {}
        # Only set default thread_id if not provided
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default"
            
        return self.compiled_graph.invoke(input_data, config=config)
        
    async def ainvoke(self, input_data: Union[str, Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Async invoke the agent"""
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        # Ensure configurable exists
        if "configurable" not in config:
            config["configurable"] = {}
        # Only set default thread_id if not provided
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default"
            
        return await self.compiled_graph.ainvoke(input_data, config=config)
        
    def stream(self, input_data: Union[str, Dict[str, Any]], **kwargs):
        """Stream agent responses"""
        if not self.config.enable_streaming:
            yield self.invoke(input_data, **kwargs)
            return
            
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        # Ensure configurable exists
        if "configurable" not in config:
            config["configurable"] = {}
        # Only set default thread_id if not provided
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default"
            
        for chunk in self.compiled_graph.stream(input_data, config=config):
            yield chunk
            
    async def astream(self, input_data: Union[str, Dict[str, Any]], **kwargs):
        """Async stream agent responses"""
        if isinstance(input_data, str):
            input_data = {"messages": [HumanMessage(content=input_data)]}
            
        config = kwargs.get("config", {})
        # Ensure configurable exists
        if "configurable" not in config:
            config["configurable"] = {}
        # Only set default thread_id if not provided
        if "thread_id" not in config["configurable"]:
            config["configurable"]["thread_id"] = "default"
            
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
    
    def store_memory(self, key: str, value: Any, namespace: Optional[str] = None):
        """Store information in long-term memory"""
        self.memory_manager.store_long_term_memory(key, value, namespace)
        
    def retrieve_memory(self, key: str, namespace: Optional[str] = None):
        """Retrieve information from long-term memory"""
        return self.memory_manager.get_long_term_memory(key, namespace)
        
    # LangMem specific methods
    
    def has_langmem_support(self) -> bool:
        """Check if LangMem is available and configured"""
        return self.memory_manager.has_langmem_support()
        
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory configuration summary"""
        return {
            "memory_type": self.config.memory_type,
            "langmem_configured": self.has_langmem_support(),
            "summarization_enabled": self.config.langmem_enable_summarization,
            "max_tokens": self.config.langmem_max_tokens,
            "max_summary_tokens": self.config.langmem_max_summary_tokens
        }
        
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
            
    # Evaluation with AgentEvals
    
    def evaluate_last_response(self) -> Dict[str, float]:
        """Evaluate the last response using basic evaluator"""
        if not self.state.messages:
            return {}
            
        last_messages = self.state.messages[-2:]  # Human + AI message
        if len(last_messages) >= 2:
            human_msg = last_messages[0].content if isinstance(last_messages[0], HumanMessage) else ""
            ai_msg = last_messages[1].content if isinstance(last_messages[1], AIMessage) else ""
            return self.evaluation_manager.evaluate_response(human_msg, ai_msg)
        return {}
        
    def evaluate_trajectory(self, outputs: List[Dict], reference_outputs: List[Dict]) -> Dict[str, Any]:
        """Evaluate agent trajectory against reference using AgentEvals"""
        return self.evaluation_manager.evaluate_trajectory(outputs, reference_outputs)
        
    def evaluate_with_llm_judge(self, outputs: List[Dict], reference_outputs: List[Dict]) -> Dict[str, Any]:
        """Evaluate using LLM-as-a-judge from AgentEvals"""
        return self.evaluation_manager.evaluate_with_llm_judge(outputs, reference_outputs)
        
    def get_evaluator_status(self) -> Dict[str, bool]:
        """Get status of available AgentEvals evaluators"""
        return self.evaluation_manager.get_evaluator_status()
        
    def create_evaluation_dataset(self, conversations: List[Dict]) -> List[Dict]:
        """Create evaluation dataset in AgentEvals format"""
        dataset = []
        for conv in conversations:
            dataset.append({
                "input": {"messages": conv.get("input_messages", [])},
                "output": {"messages": conv.get("expected_output_messages", [])}
            })
        return dataset
        
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
            "enable_short_term_memory": self.config.enable_short_term_memory,
            "short_term_memory_type": self.config.short_term_memory_type,
            "enable_long_term_memory": self.config.enable_long_term_memory,
            "long_term_memory_type": self.config.long_term_memory_type,
            "enable_supervisor": self.config.enable_supervisor,
            "enable_swarm": self.config.enable_swarm,
            "enable_evaluation": self.config.enable_evaluation,
            "enable_streaming": self.config.enable_streaming,
            "enable_human_feedback": self.config.enable_human_feedback,
            "evaluation_metrics": self.config.evaluation_metrics,
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def store_memory(self, key: str, value: str):
        """Store value in memory - backward compatibility method"""
        if self.memory_manager:
            self.memory_manager.store_memory(key, value)
    
    def retrieve_memory(self, key: str) -> Optional[str]:
        """Retrieve value from memory - backward compatibility method"""
        if self.memory_manager:
            return self.memory_manager.retrieve_memory(key)
        return None
            
    @classmethod
    def load_config(cls, filepath: str) -> 'CoreAgent':
        """Load agent configuration from file and create CoreAgent instance"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
            
        config = AgentConfig(**config_dict)
        return cls(config)
    
    def build(self, strict_mode: bool = False):
        """Build the agent graph and return it for testing purposes"""
        if hasattr(self, 'compiled_graph') and self.compiled_graph:
            return self.compiled_graph
        elif hasattr(self, 'graph') and self.graph:
            return self.graph
        else:
            # Try to build the agent if not already built
            self._build_agent(strict_mode=strict_mode)
            return self.compiled_graph if hasattr(self, 'compiled_graph') else self.graph
    

    def get_memory(self, key: str) -> Optional[str]:
        """Get memory value for a key - delegates to memory manager"""
        return self.memory_manager.get_memory(key)
        
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and capabilities"""
        return {
            "name": self.config.name,
            "model": str(self.config.model) if self.config.model else None,
            "description": self.config.description,
            "features": {
                "short_term_memory": self.config.enable_short_term_memory,
                "long_term_memory": self.config.enable_long_term_memory,
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
            "memory_enabled": self.config.enable_memory,  # Backward compatibility
            "tools_count": len(self.config.tools),  # For test compatibility
            "langmem_support": self.has_langmem_support(),
            "supervised_agents": len(self.supervisor_manager.agents),
            "mcp_servers": len(self.config.mcp_servers),
            "mcp_tools": len(self.mcp_manager.mcp_tools),
            "evaluators": self.get_evaluator_status(),
        }
