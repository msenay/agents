from typing import Optional, List, Dict, Any, Callable, Type
from dataclasses import dataclass, field
import logging
from langchain_core.rate_limiters import BaseRateLimiter
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """
    Smart, organized configuration for CoreAgent with intelligent parameter validation.

    Core Philosophy:
    - Enable/disable major features with clear flags
    - Backend-specific configs only available when that backend is selected
    - Validation prevents impossible configurations
    - User-friendly with meaningful defaults
    """

    # ============================================================================
    # CORE AGENT IDENTITY & BEHAVIOR
    # ============================================================================

    name: str = "CoreAgent"  # Agent's unique identifier
    model: Optional[BaseChatModel] = None  # Language model instance
    system_prompt: str = "You are a helpful AI assistant."  # Agent's role definition
    tools: List[BaseTool] = field(default_factory=list)  # Available tools
    description: str = ""  # Human-readable description

    # ============================================================================
    # MEMORY SYSTEM - Smart, controlled memory configuration
    # ============================================================================

    # MASTER MEMORY CONTROL
    enable_memory: bool = False  # Master switch - MUST be True to use ANY memory features

    # MEMORY TYPE SELECTION (only when enable_memory=True)
    memory_types: List[str] = field(default_factory=lambda: ["short_term"])  # Choose: ["short_term", "long_term", "session", "semantic"]

    # BACKEND SELECTION (only when enable_memory=True)
    memory_backend: str = "inmemory"  # Choose: "inmemory", "redis", "postgres", "mongodb"

    # DATABASE CONNECTIONS (only for respective backends)
    redis_url: Optional[str] = None  # Required when memory_backend="redis"
    postgres_url: Optional[str] = None  # Required when memory_backend="postgres"
    mongodb_url: Optional[str] = None  # Required when memory_backend="mongodb"

    # MEMORY FEATURES (only when specific types enabled)
    # Session Memory (only when "session" in memory_types)
    session_id: Optional[str] = None  # Session identifier for shared memory
    memory_namespace: str = "default"  # Namespace for memory isolation

    # Semantic Search (only when "semantic" in memory_types)
    embedding_model: str = "openai:text-embedding-3-small"  # Embedding model
    embedding_dims: int = 1536  # Vector dimensions

    # TTL Support (only for redis/mongodb backends)
    enable_ttl: bool = False  # Auto-expiration (Redis/MongoDB only)
    default_ttl_minutes: int = 1440  # TTL in minutes
    refresh_on_read: bool = True  # Refresh TTL on access

    # ============================================================================
    # CONTEXT MANAGEMENT - Message handling
    # ============================================================================

    # Message trimming (independent of memory)
    enable_message_trimming: bool = False
    max_tokens: int = 4000
    trim_strategy: str = "last"  # "first" or "last"
    start_on: str = "human"  # Message type to start trimming from
    end_on: List[str] = field(default_factory=lambda: ["human", "tool"])  # Message types to end trimming on
    distance_type: str = "cosine"  # For semantic search: "cosine", "euclidean", "dot_product"

    # AI Summarization (requires enable_memory=True)
    enable_summarization: bool = False
    max_summary_tokens: int = 128
    summarization_trigger_tokens: int = 2000

    # Memory Tools (requires enable_memory=True and "long_term" in memory_types)
    enable_memory_tools: bool = False
    memory_namespace_store: str = "memories"  # Storage namespace for memory tools

    # ============================================================================
    # EXTERNAL INTEGRATIONS
    # ============================================================================

    # MCP Integration
    enable_mcp: bool = False
    mcp_servers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # ============================================================================
    # MULTI-AGENT PATTERNS
    # ============================================================================

    enable_supervisor: bool = False
    enable_swarm: bool = False
    enable_handoff: bool = False
    default_active_agent: Optional[str] = None
    handoff_agents: List[str] = field(default_factory=list)
    agents: Dict[str, Any] = field(default_factory=dict)

    # ============================================================================
    # PERFORMANCE & SAFETY
    # ============================================================================

    # Rate Limiting (independent feature)
    enable_rate_limiting: bool = False
    requests_per_second: float = 1.0  # Only when enable_rate_limiting=True
    check_every_n_seconds: float = 0.1  # Only when enable_rate_limiting=True
    max_bucket_size: float = 10.0  # Only when enable_rate_limiting=True
    custom_rate_limiter: Optional[BaseRateLimiter] = None

    # Human oversight
    enable_human_feedback: bool = False
    interrupt_before: List[str] = field(default_factory=list)
    interrupt_after: List[str] = field(default_factory=list)

    # Evaluation
    enable_evaluation: bool = False
    evaluation_metrics: List[str] = field(default_factory=lambda: ["accuracy", "relevance"])

    # ============================================================================
    # EXTENSIBILITY
    # ============================================================================

    response_format: Optional[Type[BaseModel]] = None
    enable_streaming: bool = True
    pre_model_hook: Optional[Callable] = None
    post_model_hook: Optional[Callable] = None
    enable_subgraphs: bool = False
    subgraph_configs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Smart validation with helpful error messages"""

        # Set defaults
        if not self.model:
            logger.warning("⚠️ No model specified - some features may not work")
        if not self.name:
            self.name = "CoreAgent"
        if not self.description:
            self.description = f"Specialized agent: {self.name}"

        # ====================================================================
        # MEMORY VALIDATION - Prevent impossible configurations
        # ====================================================================

        if not self.enable_memory:
            # If memory disabled, block ALL memory-related features
            if self.memory_types != ["short_term"]:  # Don't reset if default
                if any(t != "short_term" for t in self.memory_types):
                    raise ValueError(
                        "❌ MEMORY DISABLED: Cannot use memory_types when enable_memory=False. "
                        "Set enable_memory=True first!"
                    )

            if self.enable_summarization:
                raise ValueError(
                    "❌ MEMORY DISABLED: Cannot use summarization when enable_memory=False. "
                    "Set enable_memory=True first!"
                )

            if self.enable_memory_tools:
                raise ValueError(
                    "❌ MEMORY DISABLED: Cannot use memory_tools when enable_memory=False. "
                    "Set enable_memory=True first!"
                )

            if self.session_id:
                raise ValueError(
                    "❌ MEMORY DISABLED: Cannot use session_id when enable_memory=False. "
                    "Set enable_memory=True first!"
                )

        else:
            # Memory enabled - validate memory types and backend compatibility
            valid_memory_types = ["short_term", "long_term", "session", "semantic"]
            for memory_type in self.memory_types:
                if memory_type not in valid_memory_types:
                    raise ValueError(
                        f"❌ INVALID MEMORY TYPE: '{memory_type}' not in {valid_memory_types}"
                    )

            # Validate backend selection
            valid_backends = ["inmemory", "redis", "postgres", "mongodb"]
            if self.memory_backend not in valid_backends:
                raise ValueError(
                    f"❌ INVALID BACKEND: '{self.memory_backend}' not in {valid_backends}"
                )

            # Validate backend-specific requirements
            if self.memory_backend == "redis" and not self.redis_url:
                raise ValueError(
                    "❌ REDIS BACKEND: redis_url is required when memory_backend='redis'"
                )

            if self.memory_backend == "postgres" and not self.postgres_url:
                raise ValueError(
                    "❌ POSTGRES BACKEND: postgres_url is required when memory_backend='postgres'"
                )

            if self.memory_backend == "mongodb" and not self.mongodb_url:
                raise ValueError(
                    "❌ MONGODB BACKEND: mongodb_url is required when memory_backend='mongodb'"
                )

            # TTL validation - only for compatible backends
            if self.enable_ttl and self.memory_backend not in ["redis", "mongodb"]:
                raise ValueError(
                    f"❌ TTL NOT SUPPORTED: TTL only works with Redis/MongoDB backends, "
                    f"not '{self.memory_backend}'"
                )

            # Memory tools validation
            if self.enable_memory_tools and "long_term" not in self.memory_types:
                raise ValueError(
                    "❌ MEMORY TOOLS: Requires 'long_term' in memory_types"
                )

            # Session memory validation
            if "session" in self.memory_types and not self.session_id:
                logger.warning(
                    "⚠️ SESSION MEMORY: No session_id provided - will use random session"
                )
                import uuid
                self.session_id = f"session_{uuid.uuid4().hex[:8]}"

        # ====================================================================
        # RATE LIMITING VALIDATION
        # ====================================================================

        if self.enable_rate_limiting:
            if self.requests_per_second <= 0:
                raise ValueError("❌ RATE LIMITING: requests_per_second must be > 0")
            if self.max_bucket_size <= 0:
                raise ValueError("❌ RATE LIMITING: max_bucket_size must be > 0")

        # ====================================================================
        # MULTI-AGENT VALIDATION
        # ====================================================================

        multi_agent_count = sum([
            self.enable_supervisor,
            self.enable_swarm,
            self.enable_handoff
        ])

        if multi_agent_count > 1:
            raise ValueError(
                "❌ MULTI-AGENT CONFLICT: Can only enable ONE multi-agent pattern "
                "(supervisor OR swarm OR handoff)"
            )

        if self.enable_handoff and not self.handoff_agents:
            raise ValueError(
                "❌ HANDOFF PATTERN: handoff_agents list cannot be empty when enable_handoff=True"
            )

    # ========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # ========================================================================

    @property
    def enable_short_term_memory(self) -> bool:
        """Backward compatibility"""
        return self.enable_memory and "short_term" in self.memory_types

    @property
    def enable_long_term_memory(self) -> bool:
        """Backward compatibility"""
        return self.enable_memory and "long_term" in self.memory_types

    @property
    def enable_shared_memory(self) -> bool:
        """Backward compatibility"""
        return self.enable_memory and "session" in self.memory_types

    @property
    def enable_semantic_search(self) -> bool:
        """Backward compatibility"""
        return self.enable_memory and "semantic" in self.memory_types

    @property
    def short_term_memory_type(self) -> str:
        """Backward compatibility"""
        return self.memory_backend

    @property
    def long_term_memory_type(self) -> str:
        """Backward compatibility"""
        return self.memory_backend

    @property
    def memory_type(self) -> str:
        """Backward compatibility for old memory_type"""
        return self.memory_backend
