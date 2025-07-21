from typing import Optional, List, Dict, Any, Type

from pydantic import BaseModel

from core.config import AgentConfig
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.tools import BaseTool, tool
from langchain_core.language_models import BaseChatModel

from core.core_agent import CoreAgent


# ============================================================================
# SIMPLE AGENT CREATORS - For single specialized agents
# ============================================================================

def create_simple_agent(
        model: BaseChatModel,
        name: str = "SimpleAgent",
        tools: List[BaseTool] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        enable_memory: bool = False
) -> CoreAgent:
    """
    Create a simple, lightweight agent - minimal configuration
    Perfect for: Task-specific agents, testing, simple automation
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Memory configuration (optional)
        enable_short_term_memory=enable_memory,
        short_term_memory_type="inmemory" if enable_memory else "inmemory",
        enable_long_term_memory=False,

        enable_streaming=True
    )

    return CoreAgent(config)


def create_rate_limited_agent(
        model: BaseChatModel,
        requests_per_second: float = 1.0,
        name: str = "RateLimitedAgent",
        tools: List[BaseTool] = None,
        system_prompt: str = "You are a helpful AI assistant with rate limiting.",
        enable_memory: bool = False,
        max_bucket_size: float = 10.0,
        check_every_n_seconds: float = 0.1,
        custom_rate_limiter: Optional[BaseRateLimiter] = None
) -> CoreAgent:
    """
    Create an agent with built-in rate limiting to prevent API 429 errors

    Perfect for:
    - Production environments with API rate limits
    - Batch processing with many requests
    - Preventing rate limit errors during testing
    - Multi-agent systems with shared API quotas

    Args:
        model: The language model to use
        requests_per_second: Maximum requests per second (default: 1.0 - conservative)
        name: Agent name
        tools: List of tools for the agent
        system_prompt: System prompt for the agent
        enable_memory: Whether to enable memory
        max_bucket_size: Maximum burst size for token bucket
        check_every_n_seconds: How often to check token availability
        custom_rate_limiter: Custom rate limiter instance (overrides other rate limit settings)

    Returns:
        CoreAgent with rate limiting enabled
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Memory configuration (optional)
        enable_memory=enable_memory,
        memory_backend="inmemory" if enable_memory else "inmemory",

        # Rate limiting configuration
        enable_rate_limiting=True,
        requests_per_second=requests_per_second,
        max_bucket_size=max_bucket_size,
        check_every_n_seconds=check_every_n_seconds,
        custom_rate_limiter=custom_rate_limiter,

        enable_streaming=True
    )

    return CoreAgent(config)


def create_advanced_agent(
        model: BaseChatModel,
        name: str = "AdvancedAgent",
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an advanced AI assistant with enhanced capabilities.",
        enable_memory: bool = True,
        enable_short_term_memory: bool = None,
        enable_long_term_memory: bool = None,
        enable_evaluation: bool = False,
        enable_human_feedback: bool = False,
        response_format: Optional[Type[BaseModel]] = None,
        enable_mcp: bool = False,
        mcp_servers: Dict[str, Dict[str, Any]] = None
) -> CoreAgent:
    """
    Create an advanced agent with optional enhanced features
    Perfect for: Production agents, complex workflows, custom requirements
    """
    # Handle backward compatibility and new parameters
    if enable_short_term_memory is None:
        enable_short_term_memory = enable_memory
    if enable_long_term_memory is None:
        enable_long_term_memory = enable_memory

    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Memory configuration
        enable_short_term_memory=enable_short_term_memory,
        short_term_memory_type="inmemory",
        enable_long_term_memory=enable_long_term_memory,
        long_term_memory_type="inmemory",

        enable_evaluation=enable_evaluation,
        enable_human_feedback=enable_human_feedback,
        response_format=response_format,
        enable_mcp=enable_mcp,
        mcp_servers=mcp_servers or {},
        enable_streaming=True
    )

    return CoreAgent(config)


# ============================================================================
# ORCHESTRATOR CREATORS - For multi-agent systems
# ============================================================================

def create_supervisor_agent(
        model: BaseChatModel,
        name: str = "SupervisorAgent",
        agents: Dict[str, Any] = None,
        system_prompt: str = "You manage multiple specialized agents. Assign work to them based on their capabilities.",
        enable_memory: bool = True,
        enable_evaluation: bool = False
) -> CoreAgent:
    """
    Create a supervisor agent for hierarchical multi-agent orchestration
    Perfect for: Central coordination, task delegation, workflow management
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        enable_supervisor=True,
        agents=agents or {},

        # Memory configuration
        enable_short_term_memory=enable_memory,
        short_term_memory_type="inmemory",
        enable_long_term_memory=enable_memory,
        long_term_memory_type="inmemory",

        enable_evaluation=enable_evaluation,
        enable_streaming=True
    )

    return CoreAgent(config)


def create_swarm_agent(
        model: BaseChatModel,
        name: str = "SwarmAgent",
        agents: Dict[str, Any] = None,
        default_active_agent: str = None,
        system_prompt: str = "You coordinate with other agents dynamically based on expertise needed.",
        enable_memory: bool = True
) -> CoreAgent:
    """
    Create a swarm agent for dynamic agent coordination
    Perfect for: Flexible workflows, expertise-based routing, collaborative problem solving
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        enable_swarm=True,
        agents=agents or {},
        default_active_agent=default_active_agent,

        # Memory configuration
        enable_short_term_memory=enable_memory,
        short_term_memory_type="inmemory",
        enable_long_term_memory=enable_memory,
        long_term_memory_type="inmemory",

        enable_streaming=True
    )

    return CoreAgent(config)


def create_handoff_agent(
        model: BaseChatModel,
        name: str = "HandoffAgent",
        agents: Dict[str, Any] = None,
        handoff_agents: List[str] = None,
        default_active_agent: str = None,
        system_prompt: str = "You can transfer conversations to specialized agents when needed.",
        enable_memory: bool = True
) -> CoreAgent:
    """
    Create a handoff agent for manual agent transfers
    Perfect for: User-controlled routing, step-by-step workflows, escalation patterns
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        enable_handoff=True,
        agents=agents or {},
        handoff_agents=handoff_agents or [],
        default_active_agent=default_active_agent,

        # Memory configuration
        enable_short_term_memory=enable_memory,
        short_term_memory_type="inmemory",
        enable_long_term_memory=enable_memory,
        long_term_memory_type="inmemory",

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

        # Memory configuration
        enable_short_term_memory=True,
        short_term_memory_type="inmemory",
        enable_long_term_memory=True,
        long_term_memory_type="inmemory",

        enable_streaming=True
    )

    return CoreAgent(config)


def create_langmem_agent(
        model: BaseChatModel,
        tools: List[BaseTool] = None,
        max_tokens: int = 384,
        max_summary_tokens: int = 128,
        enable_summarization: bool = True,
        prompt: str = "You are an assistant with advanced memory management capabilities."
) -> CoreAgent:
    """Create an agent with LangMem memory management"""
    config = AgentConfig(
        name="LangMemAgent",
        model=model,
        system_prompt=prompt,
        tools=tools or [],

        # LangMem memory configuration
        enable_short_term_memory=True,
        short_term_memory_type="inmemory",
        enable_long_term_memory=True,
        long_term_memory_type="inmemory",
        enable_summarization=enable_summarization,
        max_summary_tokens=max_summary_tokens,
        summarization_trigger_tokens=max_tokens,

        enable_streaming=True
    )

    return CoreAgent(config)


# ============================================================================
# SPECIALIZED FEATURE AGENTS - For specific use cases
# ============================================================================

def create_memory_agent(
        model: BaseChatModel,
        name: str = "MemoryAgent",
        tools: List[BaseTool] = None,
        enable_short_term_memory: bool = True,
        enable_long_term_memory: bool = True,
        short_term_memory_type: str = "inmemory",  # "inmemory", "redis", "postgres", "mongodb"
        long_term_memory_type: str = "inmemory",  # "inmemory", "redis", "postgres", "mongodb"
        enable_semantic_search: bool = True,
        enable_memory_tools: bool = True,
        enable_message_trimming: bool = True,
        enable_summarization: bool = False,
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        system_prompt: str = "You are an assistant with advanced memory capabilities for persistent conversations. You can store and search information across sessions."
) -> CoreAgent:
    """
    Create an agent with comprehensive memory capabilities following LangGraph patterns:
    - Short-term memory (thread-level persistence)
    - Long-term memory (cross-session persistence)
    - Semantic search with embeddings
    - Memory management tools
    - Message trimming and summarization

    Perfect for: Long conversations, user profiling, context retention, knowledge management
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Memory configuration
        enable_short_term_memory=enable_short_term_memory,
        short_term_memory_type=short_term_memory_type,
        enable_long_term_memory=enable_long_term_memory,
        long_term_memory_type=long_term_memory_type,

        # Advanced memory features
        enable_semantic_search=enable_semantic_search,
        enable_memory_tools=enable_memory_tools,
        enable_message_trimming=enable_message_trimming,
        enable_summarization=enable_summarization,

        # Database connections
        redis_url=redis_url,
        postgres_url=postgres_url,
        mongodb_url=mongodb_url,

        # Performance optimizations
        max_tokens=4000,
        trim_strategy="last",
        summarization_trigger_tokens=2000,
        max_summary_tokens=128,

        enable_streaming=True
    )

    return CoreAgent(config)


def create_short_term_memory_agent(
        model: BaseChatModel,
        name: str = "ShortTermAgent",
        memory_backend: str = "inmemory",  # "inmemory", "redis", "postgres", "mongodb"
        enable_trimming: bool = True,
        max_tokens: int = 4000,
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an assistant with short-term memory for multi-turn conversations."
) -> CoreAgent:
    """
    Create an agent with only short-term memory (thread-level persistence)
    Following LangGraph pattern: checkpointer for conversation history

    Perfect for: Chat sessions, conversation continuity, context tracking
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Only short-term memory
        enable_short_term_memory=True,
        short_term_memory_type=memory_backend,
        enable_long_term_memory=False,

        # Message management
        enable_message_trimming=enable_trimming,
        max_tokens=max_tokens,
        trim_strategy="last",

        # Database connections
        redis_url=redis_url,
        postgres_url=postgres_url,
        mongodb_url=mongodb_url,
    )

    return CoreAgent(config)


def create_long_term_memory_agent(
        model: BaseChatModel,
        name: str = "LongTermAgent",
        memory_backend: str = "inmemory",  # "inmemory", "redis", "postgres", "mongodb"
        enable_semantic_search: bool = True,
        enable_memory_tools: bool = True,
        embedding_model: str = "openai:text-embedding-3-small",
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an assistant with long-term memory for persistent information storage and retrieval across sessions."
) -> CoreAgent:
    """
    Create an agent with only long-term memory (cross-session persistence)
    Following LangGraph pattern: store for persistent data with semantic search

    Perfect for: Knowledge bases, user profiling, persistent data storage
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Only long-term memory
        enable_short_term_memory=False,
        enable_long_term_memory=True,
        long_term_memory_type=memory_backend,

        # Advanced features
        enable_semantic_search=enable_semantic_search,
        enable_memory_tools=enable_memory_tools,
        embedding_model=embedding_model,

        # Database connections
        redis_url=redis_url,
        postgres_url=postgres_url,
        mongodb_url=mongodb_url,
    )

    return CoreAgent(config)


def create_message_management_agent(
        model: BaseChatModel,
        name: str = "MessageManagerAgent",
        management_strategy: str = "trim",  # "trim", "summarize", "delete"
        max_tokens: int = 4000,
        trim_strategy: str = "last",  # "first", "last"
        enable_summarization: bool = False,
        max_summary_tokens: int = 128,
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an assistant with advanced message management capabilities for long conversations."
) -> CoreAgent:
    """
    Create an agent optimized for message management in long conversations
    Following LangGraph patterns: trim_messages, summarization, RemoveMessage

    Perfect for: Long conversations, context window management, memory optimization
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Short-term memory for conversation tracking
        enable_short_term_memory=True,
        short_term_memory_type="inmemory",

        # Message management strategies
        enable_message_trimming=(management_strategy in ["trim", "both"]),
        enable_summarization=(management_strategy in ["summarize", "both"] or enable_summarization),

        # Configuration
        max_tokens=max_tokens,
        trim_strategy=trim_strategy,
        max_summary_tokens=max_summary_tokens,
        summarization_trigger_tokens=max_tokens - 500,  # Trigger before hitting limit
    )

    return CoreAgent(config)


def create_semantic_search_agent(
        model: BaseChatModel,
        name: str = "SemanticSearchAgent",
        memory_backend: str = "inmemory",  # "inmemory", "redis", "postgres", "mongodb"
        embedding_model: str = "openai:text-embedding-3-small",
        embedding_dims: int = 1536,
        distance_type: str = "cosine",
        enable_memory_tools: bool = True,
        redis_url: Optional[str] = None,
        postgres_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an assistant with semantic search capabilities. You can store and find information using meaning-based similarity."
) -> CoreAgent:
    """
    Create an agent with semantic search capabilities using embeddings
    Following LangGraph pattern: store with vector search

    Perfect for: Knowledge retrieval, content discovery, similarity search
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Memory with semantic search
        enable_short_term_memory=True,
        enable_long_term_memory=True,
        long_term_memory_type=memory_backend,

        # Semantic search configuration
        enable_semantic_search=True,
        enable_memory_tools=enable_memory_tools,
        embedding_model=embedding_model,
        embedding_dims=embedding_dims,
        distance_type=distance_type,

        # Database connections
        redis_url=redis_url,
        postgres_url=postgres_url,
        mongodb_url=mongodb_url,
    )

    return CoreAgent(config)


def create_ttl_memory_agent(
        model: BaseChatModel,
        name: str = "TTLMemoryAgent",
        memory_backend: str = "redis",  # TTL works with Redis and MongoDB
        ttl_minutes: int = 1440,  # 24 hours default
        refresh_on_read: bool = True,
        redis_url: Optional[str] = "redis://localhost:6379",
        mongodb_url: Optional[str] = None,
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an assistant with time-limited memory that automatically expires after a set time."
) -> CoreAgent:
    """
    Create an agent with TTL (Time-To-Live) memory for automatic cleanup
    Following LangGraph pattern: TTL configuration with Redis

    Perfect for: Temporary data, privacy compliance, automatic cleanup
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Memory with TTL
        enable_short_term_memory=True,
        short_term_memory_type=memory_backend,
        enable_long_term_memory=True,
        long_term_memory_type=memory_backend,

        # TTL configuration
        enable_ttl=True,
        default_ttl_minutes=ttl_minutes,
        refresh_on_read=refresh_on_read,

        # Database connections
        redis_url=redis_url,
        mongodb_url=mongodb_url,
    )

    return CoreAgent(config)


def create_evaluated_agent(
        model: BaseChatModel,
        name: str = "EvaluatedAgent",
        tools: List[BaseTool] = None,
        evaluation_metrics: List[str] = None,
        system_prompt: str = "You are an assistant with performance evaluation capabilities."
) -> CoreAgent:
    """
    Create an agent with comprehensive evaluation capabilities
    Perfect for: Quality assurance, performance monitoring, testing
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
        enable_evaluation=True,
        evaluation_metrics=evaluation_metrics or ["accuracy", "relevance", "helpfulness"],

        # Memory configuration
        enable_short_term_memory=True,
        short_term_memory_type="inmemory",
        enable_long_term_memory=True,
        long_term_memory_type="inmemory",

        enable_streaming=True
    )

    return CoreAgent(config)


def create_human_interactive_agent(
        model: BaseChatModel,
        name: str = "InteractiveAgent",
        tools: List[BaseTool] = None,
        system_prompt: str = "You are an assistant that works collaboratively with humans.",
        interrupt_before: List[str] = None,
        interrupt_after: List[str] = None
) -> CoreAgent:
    """
    Create an agent with human-in-the-loop capabilities
    Perfect for: Collaborative workflows, approval processes, guided tasks
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],
        enable_human_feedback=True,
        interrupt_before=interrupt_before or ["execute_tools"],
        interrupt_after=interrupt_after or ["generate_response"],

        # Memory configuration
        enable_short_term_memory=True,
        short_term_memory_type="inmemory",
        enable_long_term_memory=True,
        long_term_memory_type="inmemory",

        enable_streaming=True
    )

    return CoreAgent(config)


# ============================================================================
# SESSION-BASED MEMORY CREATORS - For agent collaboration with shared memory
# ============================================================================

def create_session_agent(
        model: BaseChatModel,
        session_id: str,
        name: str = "SessionAgent",
        tools: List[BaseTool] = None,
        memory_namespace: str = "default",
        enable_shared_memory: bool = True,
        redis_url: Optional[str] = None,
        mongodb_url: Optional[str] = None,
        system_prompt: str = "You are an agent with session-based shared memory capabilities."
) -> CoreAgent:
    """
    Create an agent with session-based memory for collaboration
    Perfect for: Multi-agent workflows, code collaboration, shared context

    Args:
        model: The LLM model to use
        session_id: Unique session identifier for shared memory
        name: Agent name (used for memory namespace)
        tools: Tools available to the agent
        memory_namespace: Memory namespace within session (for agent isolation)
        enable_shared_memory: Enable session-based shared memory
        redis_url: Redis connection URL
        system_prompt: Agent's system prompt
    """
    config = AgentConfig(
        name=name,
        model=model,
        system_prompt=system_prompt,
        tools=tools or [],

        # Use new memory configuration
        enable_short_term_memory=True,
        short_term_memory_type="inmemory",  # Fallback for session memory
        enable_long_term_memory=False,  # Focus on session memory

        # Session-based memory
        session_id=session_id,
        enable_shared_memory=enable_shared_memory,
        memory_namespace=memory_namespace,

        # Database connections
        redis_url=redis_url,
        mongodb_url=mongodb_url,

        enable_streaming=True
    )

    return CoreAgent(config)


def create_collaborative_agents(
        models: Dict[str, BaseChatModel],
        session_id: str,
        agent_configs: Dict[str, Dict[str, Any]],
        redis_url: Optional[str] = None
) -> Dict[str, CoreAgent]:
    """
    Create multiple collaborative agents sharing the same session memory

    Args:
        models: Dictionary of {agent_name: model} for different agents
        session_id: Shared session identifier
        agent_configs: Dictionary of agent configurations
        redis_url: Redis connection URL

    Returns:
        Dictionary of {agent_name: CoreAgent} instances

    Example:
        models = {
            "coder": gpt4_model,
            "tester": gpt35_model,
            "reviewer": claude_model
        }

        configs = {
            "coder": {
                "tools": [write_code_tool],
                "system_prompt": "You write code and store it in session memory"
            },
            "tester": {
                "tools": [test_code_tool],
                "system_prompt": "You create tests for session code"
            }
        }

        agents = create_collaborative_agents(models, "session_123", configs)
    """
    agents = {}

    for agent_name, model in models.items():
        config = agent_configs.get(agent_name, {})

        agent = create_session_agent(
            model=model,
            session_id=session_id,
            name=agent_name,
            tools=config.get("tools", []),
            memory_namespace=agent_name,  # Each agent gets its own namespace
            enable_shared_memory=True,
            redis_url=redis_url,
            system_prompt=config.get("system_prompt", f"You are {agent_name} with session memory access.")
        )

        agents[agent_name] = agent

    return agents


def create_coding_session_agents(
        model: BaseChatModel,
        session_id: str,
        redis_url: Optional[str] = None
) -> Dict[str, CoreAgent]:
    """
    Create a predefined set of coding collaboration agents
    Perfect for: Code development workflows, pair programming, code review
    """

    # Define coding tools (placeholder for now)
    @tool
    def store_code(code: str) -> str:
        """Store code in session memory"""
        return f"Code stored in session: {code[:50]}..."

    @tool
    def get_session_code() -> str:
        """Retrieve code from session memory"""
        return "Retrieved code from session memory"

    @tool
    def store_test_results(results: str) -> str:
        """Store test results in session memory"""
        return f"Test results stored: {results[:50]}..."

    # Agent configurations
    agents_config = {
        "coder": {
            "tools": [store_code, get_session_code],
            "system_prompt": f"""You are a Coder Agent with session-based memory (Session: {session_id}).

🧑‍💻 CODING CAPABILITIES:
- Write Python code and store it in session memory
- Access previously written code from session history
- Collaborate with other agents in the same session
- Remember all code iterations and improvements

📝 SESSION MEMORY USAGE:
- Store all code you write in session memory for other agents
- Reference previous code from session when asked
- Build upon existing code in the session
- Maintain coding context across agent interactions

Session ID: {session_id}
Remember: Other agents in this session can see and improve your code!"""
        },

        "tester": {
            "tools": [get_session_code, store_test_results],
            "system_prompt": f"""You are a Unit Test Agent with session-based memory (Session: {session_id}).

🧪 TESTING CAPABILITIES:
- Access code written by other agents in this session
- Create comprehensive unit tests for session code
- Store test results in session memory
- Validate code quality and functionality

📝 SESSION MEMORY USAGE:
- Retrieve code from session shared memory
- Create tests based on session code history
- Store test results for other agents to see
- Collaborate on code quality assurance

Session ID: {session_id}
Remember: You can access code written by CoderAgent and other agents in this session!"""
        },

        "reviewer": {
            "tools": [get_session_code, store_code],
            "system_prompt": f"""You are a Code Reviewer Agent with session-based memory (Session: {session_id}).

🔍 REVIEW CAPABILITIES:
- Review code written by other agents in this session
- Suggest improvements and optimizations
- Add improved code versions to session memory
- Ensure code quality and best practices

📝 SESSION MEMORY USAGE:
- Access all code from session history
- Add improvements and suggestions to session
- Collaborate with CoderAgent and TestAgent
- Maintain review history in session

Session ID: {session_id}
Remember: You can see and improve upon all code in this session!"""
        },

        "executor": {
            "tools": [get_session_code, store_test_results],
            "system_prompt": f"""You are an Executor Agent with session-based memory (Session: {session_id}).

🚀 EXECUTION CAPABILITIES:
- Execute code written by other agents in this session
- Run tests created by TestAgent
- Report execution results to session memory
- Validate code functionality

📝 SESSION MEMORY USAGE:
- Access all code and tests from session
- Execute and report results to session
- Collaborate on code validation
- Store execution history in session

Session ID: {session_id}
Remember: You execute code created by other agents in this session!"""
        }
    }

    # Create models dict (all agents use the same model for simplicity)
    models = {name: model for name in agents_config.keys()}

    return create_collaborative_agents(models, session_id, agents_config, redis_url)
