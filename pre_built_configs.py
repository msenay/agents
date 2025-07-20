#!/usr/bin/env python3
"""
🎯 Pre-Built Agent Configurations
==================================

This module provides optimized, ready-to-use AgentConfig presets for common scenarios.
Each configuration is carefully tuned for specific use cases and production environments.

Usage:
    from pre_built_configs import HIGH_PERFORMANCE_CONFIG, DEVELOPMENT_CONFIG
    from core_agent import CoreAgent
    
    # Use a pre-built config
    agent = CoreAgent(HIGH_PERFORMANCE_CONFIG)
    
    # Or customize a preset
    custom_config = DEVELOPMENT_CONFIG.copy()
    custom_config.name = "MyCustomAgent"
    agent = CoreAgent(custom_config)

Author: CoreAgent Team
Version: 1.0.0
"""

from core_agent import AgentConfig
from langchain_openai import ChatOpenAI
from typing import Optional, Dict, Any


# =============================================================================
# 🚀 HIGH PERFORMANCE PRODUCTION CONFIGS
# =============================================================================

def create_high_performance_config(
    redis_url: str = "redis://localhost:6379",
    postgres_url: str = "postgresql://user:pass@localhost:5432/coreagent"
) -> AgentConfig:
    """
    🚀 HIGH PERFORMANCE PRODUCTION CONFIGURATION
    
    Perfect for: Production environments, high-throughput applications, enterprise use
    
    Features:
    - ✅ Redis for ultra-fast short-term memory and session sharing
    - ✅ PostgreSQL for reliable long-term memory persistence  
    - ✅ Aggressive rate limiting to prevent API abuse
    - ✅ Message trimming for optimal token usage
    - ✅ AI summarization to preserve context efficiently
    - ✅ Semantic search with embeddings for intelligent retrieval
    - ✅ Memory tools for self-optimizing agents
    - ✅ TTL support for automatic memory cleanup
    
    Use Cases:
    - Customer service chatbots with high volume
    - Multi-tenant applications requiring isolation
    - Enterprise AI assistants with complex workflows
    - Production APIs serving thousands of users
    
    Performance Characteristics:
    - Memory: Hybrid (Redis + PostgreSQL) for speed + persistence
    - Rate Limiting: 5 req/sec with burst capacity
    - Context Management: Smart trimming + AI summarization
    - Scalability: Horizontal scaling ready with Redis clustering
    
    Args:
        redis_url: Redis connection string for fast memory operations
        postgres_url: PostgreSQL connection string for persistent storage
        
    Returns:
        AgentConfig: Optimized configuration for high-performance production
    """
    return AgentConfig(
        name="HighPerformanceAgent",
        
        # Memory: Hybrid approach for speed + persistence
        enable_memory=True,
        memory_types=["short_term", "long_term", "session", "semantic"],
        memory_backend="redis",  # Redis for speed
        redis_url=redis_url,
        postgres_url=postgres_url,  # Available for fallback
        
        # Session management for multi-user environments
        session_id=None,  # Will be set per request
        memory_namespace="production",
        
        # Context optimization for cost efficiency
        enable_message_trimming=True,
        max_tokens=8000,  # Large context window
        trim_strategy="last",
        
        # AI features for intelligent operation
        enable_summarization=True,
        max_summary_tokens=256,
        summarization_trigger_tokens=6000,
        
        # Memory tools for self-optimization
        enable_memory_tools=True,
        memory_namespace_store="prod_memories",
        
        # Semantic search for intelligent retrieval
        embedding_model="openai:text-embedding-3-small",
        embedding_dims=1536,
        distance_type="cosine",
        
        # TTL for automatic cleanup
        enable_ttl=True,
        default_ttl_minutes=2880,  # 48 hours
        refresh_on_read=True,
        
        # Rate limiting for API protection
        enable_rate_limiting=True,
        requests_per_second=5.0,  # Conservative for production
        max_bucket_size=15.0,  # Allow bursts
        check_every_n_seconds=0.1,
        
        # Production-ready streaming
        enable_streaming=True,
        
        # Evaluation for quality monitoring
        enable_evaluation=True,
        evaluation_metrics=["accuracy", "relevance", "helpfulness", "safety"]
    )


def create_distributed_config(
    redis_url: str = "redis://redis-cluster:6379",
    mongodb_url: str = "mongodb://mongo-cluster:27017/coreagent"
) -> AgentConfig:
    """
    🌐 DISTRIBUTED SYSTEMS CONFIGURATION
    
    Perfect for: Microservices, cloud-native apps, multi-region deployments
    
    Features:
    - ✅ Redis clustering for distributed short-term memory
    - ✅ MongoDB for flexible, schema-less document storage
    - ✅ Session-based memory for stateless microservices
    - ✅ TTL support for automatic data expiration
    - ✅ Supervisor pattern for service orchestration
    - ✅ Rate limiting with shared state
    
    Use Cases:
    - Microservices architectures
    - Multi-region deployments
    - Cloud-native applications
    - Container orchestration (Kubernetes)
    
    Architecture Benefits:
    - Horizontal scaling with Redis/MongoDB clustering
    - Stateless service design with session memory
    - Automatic failover and recovery
    - Cross-service communication patterns
    
    Args:
        redis_url: Redis cluster connection string
        mongodb_url: MongoDB cluster connection string
        
    Returns:
        AgentConfig: Configuration optimized for distributed systems
    """
    return AgentConfig(
        name="DistributedAgent",
        
        # Distributed memory architecture
        enable_memory=True,
        memory_types=["short_term", "long_term", "session"],
        memory_backend="redis",  # Redis for distributed caching
        redis_url=redis_url,
        mongodb_url=mongodb_url,
        
        # Session-based for stateless services
        memory_namespace="distributed",
        
        # TTL for data lifecycle management
        enable_ttl=True,
        default_ttl_minutes=1440,  # 24 hours
        refresh_on_read=True,
        
        # Context management for cost control
        enable_message_trimming=True,
        max_tokens=4000,
        trim_strategy="last",
        
        # Supervisor pattern for service coordination
        enable_supervisor=True,
        agents={},  # Will be populated with microservices
        
        # Rate limiting with distributed state
        enable_rate_limiting=True,
        requests_per_second=3.0,
        max_bucket_size=10.0,
        
        # Streaming for real-time responses
        enable_streaming=True
    )


# =============================================================================
# 🛠️ DEVELOPMENT & TESTING CONFIGS
# =============================================================================

def create_development_config() -> AgentConfig:
    """
    🛠️ DEVELOPMENT CONFIGURATION
    
    Perfect for: Local development, testing, prototyping, debugging
    
    Features:
    - ✅ InMemory backend for fast iteration
    - ✅ No external dependencies (Redis/PostgreSQL not required)
    - ✅ Mock implementations for all features
    - ✅ Minimal rate limiting for development speed
    - ✅ All memory features enabled for testing
    - ✅ Comprehensive logging and debugging
    
    Use Cases:
    - Local development environment
    - Unit testing and integration testing
    - Feature prototyping and experimentation
    - CI/CD pipeline testing
    - Learning and educational purposes
    
    Development Benefits:
    - Zero infrastructure requirements
    - Fast startup and teardown
    - Deterministic behavior for testing
    - Easy debugging and inspection
    
    Returns:
        AgentConfig: Optimized configuration for development
    """
    return AgentConfig(
        name="DevelopmentAgent",
        
        # InMemory for zero dependencies
        enable_memory=True,
        memory_types=["short_term", "long_term", "session", "semantic"],
        memory_backend="inmemory",
        
        # Session support for multi-agent testing
        session_id="dev_session_001",
        memory_namespace="development",
        
        # Context management for testing
        enable_message_trimming=True,
        max_tokens=2000,  # Smaller for development
        trim_strategy="last",
        
        # AI features with mocks
        enable_summarization=True,
        max_summary_tokens=128,
        summarization_trigger_tokens=1500,
        
        # Memory tools for feature testing
        enable_memory_tools=True,
        memory_namespace_store="dev_memories",
        
        # Minimal rate limiting for development speed
        enable_rate_limiting=True,
        requests_per_second=10.0,  # High for development
        max_bucket_size=50.0,
        
        # Streaming for UI development
        enable_streaming=True,
        
        # Evaluation for quality testing
        enable_evaluation=True,
        evaluation_metrics=["accuracy", "relevance"]
    )


def create_testing_config() -> AgentConfig:
    """
    🧪 TESTING CONFIGURATION
    
    Perfect for: Automated testing, CI/CD, quality assurance, benchmarking
    
    Features:
    - ✅ Deterministic behavior for consistent testing
    - ✅ InMemory backend for isolated test environments
    - ✅ All features enabled for comprehensive coverage
    - ✅ No rate limiting for fast test execution
    - ✅ Minimal logging to reduce test noise
    - ✅ Evaluation hooks for automated quality assessment
    
    Use Cases:
    - Unit testing and integration testing
    - Automated quality assurance
    - Performance benchmarking
    - Regression testing
    - Feature validation
    
    Testing Benefits:
    - Deterministic and repeatable results
    - Fast execution without external dependencies
    - Comprehensive feature coverage
    - Easy assertion and validation
    
    Returns:
        AgentConfig: Configuration optimized for automated testing
    """
    return AgentConfig(
        name="TestingAgent",
        
        # InMemory for isolated testing
        enable_memory=True,
        memory_types=["short_term", "long_term", "session", "semantic"],
        memory_backend="inmemory",
        
        # Fixed session for deterministic testing
        session_id="test_session_fixed",
        memory_namespace="testing",
        
        # Message management for test scenarios
        enable_message_trimming=True,
        max_tokens=1000,  # Small for fast testing
        trim_strategy="last",
        
        # AI features with mocks for consistent testing
        enable_summarization=True,
        max_summary_tokens=64,
        summarization_trigger_tokens=800,
        
        # Memory tools for feature testing
        enable_memory_tools=True,
        memory_namespace_store="test_memories",
        
        # No rate limiting for fast test execution
        enable_rate_limiting=False,
        
        # No streaming for deterministic testing
        enable_streaming=False,
        
        # Evaluation for automated quality assessment
        enable_evaluation=True,
        evaluation_metrics=["accuracy", "relevance", "consistency"]
    )


# =============================================================================
# 🎯 SPECIALIZED USE CASE CONFIGS
# =============================================================================

def create_customer_service_config(
    redis_url: str = "redis://localhost:6379"
) -> AgentConfig:
    """
    👥 CUSTOMER SERVICE CONFIGURATION
    
    Perfect for: Customer support, help desks, service chatbots, FAQ systems
    
    Features:
    - ✅ Session memory for conversation continuity
    - ✅ Semantic search for knowledge base integration
    - ✅ Message trimming to handle long conversations
    - ✅ AI summarization for conversation history
    - ✅ Memory tools for learning from interactions
    - ✅ Rate limiting to prevent abuse
    
    Use Cases:
    - Customer support chatbots
    - Help desk automation
    - FAQ and knowledge base systems
    - Multi-turn problem solving
    - Conversation hand-off to humans
    
    Customer Service Benefits:
    - Maintains conversation context across sessions
    - Learns from previous interactions
    - Handles escalations gracefully
    - Provides consistent service quality
    
    Args:
        redis_url: Redis connection for session persistence
        
    Returns:
        AgentConfig: Configuration optimized for customer service
    """
    return AgentConfig(
        name="CustomerServiceAgent",
        
        # Session-focused memory for customer continuity
        enable_memory=True,
        memory_types=["short_term", "long_term", "session", "semantic"],
        memory_backend="redis",
        redis_url=redis_url,
        
        # Customer session management
        memory_namespace="customer_service",
        
        # Long conversation support
        enable_message_trimming=True,
        max_tokens=6000,  # Handle long support conversations
        trim_strategy="last",
        
        # AI summarization for conversation history
        enable_summarization=True,
        max_summary_tokens=200,
        summarization_trigger_tokens=4000,
        
        # Semantic search for knowledge retrieval
        embedding_model="openai:text-embedding-3-small",
        embedding_dims=1536,
        distance_type="cosine",
        
        # Memory tools for continuous learning
        enable_memory_tools=True,
        memory_namespace_store="support_knowledge",
        
        # Human handoff support
        enable_human_feedback=True,
        interrupt_before=["escalation_tool"],
        
        # Moderate rate limiting for customer comfort
        enable_rate_limiting=True,
        requests_per_second=2.0,
        max_bucket_size=8.0,
        
        # Streaming for responsive interaction
        enable_streaming=True,
        
        # Quality evaluation for service improvement
        enable_evaluation=True,
        evaluation_metrics=["helpfulness", "accuracy", "empathy", "resolution_rate"]
    )


def create_research_assistant_config(
    postgres_url: str = "postgresql://user:pass@localhost:5432/research"
) -> AgentConfig:
    """
    🔬 RESEARCH ASSISTANT CONFIGURATION
    
    Perfect for: Academic research, data analysis, literature reviews, knowledge work
    
    Features:
    - ✅ PostgreSQL for reliable long-term knowledge storage
    - ✅ Semantic search for literature and document retrieval
    - ✅ Memory tools for building knowledge graphs
    - ✅ No rate limiting for intensive research sessions
    - ✅ Large context windows for document analysis
    - ✅ AI summarization for research synthesis
    
    Use Cases:
    - Academic research assistance
    - Literature reviews and meta-analysis
    - Document analysis and synthesis
    - Knowledge base construction
    - Scientific writing support
    
    Research Benefits:
    - Builds comprehensive knowledge over time
    - Connects disparate information sources
    - Maintains research methodology consistency
    - Supports collaborative research projects
    
    Args:
        postgres_url: PostgreSQL connection for reliable knowledge storage
        
    Returns:
        AgentConfig: Configuration optimized for research assistance
    """
    return AgentConfig(
        name="ResearchAssistantAgent",
        
        # Long-term knowledge storage
        enable_memory=True,
        memory_types=["long_term", "semantic"],
        memory_backend="postgres",
        postgres_url=postgres_url,
        
        # Research project organization
        memory_namespace="research",
        
        # Large context for document analysis
        enable_message_trimming=True,
        max_tokens=16000,  # Large context for research
        trim_strategy="first",  # Keep recent research context
        
        # AI summarization for synthesis
        enable_summarization=True,
        max_summary_tokens=512,  # Detailed summaries
        summarization_trigger_tokens=12000,
        
        # Advanced semantic search for literature
        embedding_model="openai:text-embedding-3-large",  # Higher quality
        embedding_dims=3072,
        distance_type="cosine",
        
        # Memory tools for knowledge graph building
        enable_memory_tools=True,
        memory_namespace_store="research_knowledge",
        
        # No rate limiting for intensive research
        enable_rate_limiting=False,
        
        # No streaming for careful analysis
        enable_streaming=False,
        
        # Evaluation for research quality
        enable_evaluation=True,
        evaluation_metrics=["accuracy", "completeness", "methodology", "citation_quality"]
    )


def create_creative_assistant_config() -> AgentConfig:
    """
    🎨 CREATIVE ASSISTANT CONFIGURATION
    
    Perfect for: Content creation, creative writing, brainstorming, artistic projects
    
    Features:
    - ✅ InMemory for fast iteration and experimentation
    - ✅ Session memory for creative project continuity
    - ✅ Minimal constraints for creative freedom
    - ✅ No rate limiting for rapid ideation
    - ✅ Memory tools for inspiration and reference
    - ✅ Flexible context management
    
    Use Cases:
    - Creative writing and storytelling
    - Content creation and copywriting
    - Brainstorming and ideation sessions
    - Artistic project development
    - Marketing and advertising creative
    
    Creative Benefits:
    - Encourages free-flowing creative processes
    - Maintains creative project context
    - Supports iterative refinement
    - Enables experimental approaches
    
    Returns:
        AgentConfig: Configuration optimized for creative work
    """
    return AgentConfig(
        name="CreativeAssistantAgent",
        
                 # Fast iteration memory
         enable_memory=True,
         memory_types=["short_term", "long_term", "session"],  # Added long_term for memory tools
         memory_backend="inmemory",
        
        # Creative project sessions
        memory_namespace="creative",
        
        # Flexible context management
        enable_message_trimming=True,
        max_tokens=8000,  # Large creative context
        trim_strategy="last",
        
        # Memory tools for inspiration
        enable_memory_tools=True,
        memory_namespace_store="creative_inspiration",
        
        # No rate limiting for creative flow
        enable_rate_limiting=False,
        
        # Streaming for real-time creative feedback
        enable_streaming=True,
        
        # Creative quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["creativity", "originality", "engagement", "style_consistency"]
    )


def create_enterprise_config(
    redis_url: str = "redis://enterprise-redis:6379",
    postgres_url: str = "postgresql://enterprise-user:pass@enterprise-db:5432/agents"
) -> AgentConfig:
    """
    🏢 ENTERPRISE CONFIGURATION
    
    Perfect for: Large organizations, compliance requirements, audit trails, governance
    
    Features:
    - ✅ Multi-backend architecture for redundancy
    - ✅ Comprehensive audit trails and logging
    - ✅ Strict rate limiting for resource control
    - ✅ Session isolation for security
    - ✅ Human oversight and approval workflows
    - ✅ Evaluation for compliance monitoring
    - ✅ TTL for data governance
    
    Use Cases:
    - Enterprise AI assistants
    - Compliance-sensitive applications
    - Multi-departmental AI systems
    - Audit-required environments
    - Regulated industry applications
    
    Enterprise Benefits:
    - Meets compliance and audit requirements
    - Provides comprehensive governance controls
    - Ensures data privacy and security
    - Supports organizational policies
    
    Args:
        redis_url: Enterprise Redis cluster connection
        postgres_url: Enterprise PostgreSQL connection
        
    Returns:
        AgentConfig: Configuration for enterprise environments
    """
    return AgentConfig(
        name="EnterpriseAgent",
        
        # Multi-backend for redundancy
        enable_memory=True,
        memory_types=["short_term", "long_term", "session", "semantic"],
        memory_backend="redis",
        redis_url=redis_url,
        postgres_url=postgres_url,
        
        # Enterprise session management
        memory_namespace="enterprise",
        
        # Conservative context management
        enable_message_trimming=True,
        max_tokens=4000,
        trim_strategy="last",
        
        # AI summarization with audit trails
        enable_summarization=True,
        max_summary_tokens=128,
        summarization_trigger_tokens=3000,
        
        # Memory tools for knowledge management
        enable_memory_tools=True,
        memory_namespace_store="enterprise_knowledge",
        
        # Data governance with TTL
        enable_ttl=True,
        default_ttl_minutes=10080,  # 7 days
        refresh_on_read=False,  # Strict expiration
        
        # Human oversight for compliance
        enable_human_feedback=True,
        interrupt_before=["sensitive_action", "data_access"],
        interrupt_after=["decision_making", "external_api"],
        
        # Strict rate limiting
        enable_rate_limiting=True,
        requests_per_second=1.0,
        max_bucket_size=3.0,
        
        # No streaming for audit trails
        enable_streaming=False,
        
        # Comprehensive evaluation
        enable_evaluation=True,
        evaluation_metrics=["accuracy", "compliance", "security", "audit_trail", "policy_adherence"]
    )


# =============================================================================
# 📋 PRE-BUILT CONFIG REGISTRY
# =============================================================================

# Create pre-built configurations that are ready to use
HIGH_PERFORMANCE_CONFIG = create_high_performance_config()
DISTRIBUTED_CONFIG = create_distributed_config()
DEVELOPMENT_CONFIG = create_development_config()
TESTING_CONFIG = create_testing_config()
CUSTOMER_SERVICE_CONFIG = create_customer_service_config()
RESEARCH_ASSISTANT_CONFIG = create_research_assistant_config()
CREATIVE_ASSISTANT_CONFIG = create_creative_assistant_config()
ENTERPRISE_CONFIG = create_enterprise_config()

# Registry for easy access
CONFIG_REGISTRY = {
    "high_performance": HIGH_PERFORMANCE_CONFIG,
    "distributed": DISTRIBUTED_CONFIG,
    "development": DEVELOPMENT_CONFIG,
    "testing": TESTING_CONFIG,
    "customer_service": CUSTOMER_SERVICE_CONFIG,
    "research_assistant": RESEARCH_ASSISTANT_CONFIG,
    "creative_assistant": CREATIVE_ASSISTANT_CONFIG,
    "enterprise": ENTERPRISE_CONFIG,
}


def get_config(config_name: str) -> AgentConfig:
    """
    Get a pre-built configuration by name.
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        AgentConfig: The requested pre-built configuration
        
    Raises:
        ValueError: If config_name is not found in registry
        
    Example:
        >>> config = get_config("development")
        >>> agent = CoreAgent(config)
    """
    if config_name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")
    
    return CONFIG_REGISTRY[config_name]


def list_configs() -> Dict[str, str]:
    """
    List all available pre-built configurations with descriptions.
    
    Returns:
        Dict[str, str]: Mapping of config names to descriptions
        
    Example:
        >>> configs = list_configs()
        >>> for name, desc in configs.items():
        ...     print(f"{name}: {desc}")
    """
    return {
        "high_performance": "🚀 Production-ready with Redis+PostgreSQL, rate limiting, and all features",
        "distributed": "🌐 Microservices architecture with Redis clustering and MongoDB",
        "development": "🛠️ Local development with InMemory backend and minimal dependencies",
        "testing": "🧪 Automated testing with deterministic behavior and fast execution",
        "customer_service": "👥 Customer support with session continuity and knowledge base",
        "research_assistant": "🔬 Academic research with PostgreSQL and semantic search",
        "creative_assistant": "🎨 Creative work with flexible constraints and rapid iteration",
        "enterprise": "🏢 Enterprise-grade with compliance, audit trails, and governance",
    }


def create_custom_config(**kwargs) -> AgentConfig:
    """
    Create a custom configuration based on development config with overrides.
    
    Args:
        **kwargs: Configuration parameters to override
        
    Returns:
        AgentConfig: Custom configuration with specified overrides
        
    Example:
        >>> config = create_custom_config(
        ...     name="MyAgent",
        ...     enable_rate_limiting=True,
        ...     requests_per_second=5.0
        ... )
    """
    # Start with development config as base
    base_config = DEVELOPMENT_CONFIG
    
    # Create new config with overrides
    config_dict = base_config.__dict__.copy()
    config_dict.update(kwargs)
    
    return AgentConfig(**config_dict)


# =============================================================================
# 📖 CONFIGURATION COMPARISON TABLE
# =============================================================================

def print_config_comparison():
    """Print a comparison table of all pre-built configurations."""
    
    print("🎯 PRE-BUILT AGENT CONFIGURATIONS COMPARISON")
    print("=" * 80)
    print()
    
    configs = [
        ("High Performance", HIGH_PERFORMANCE_CONFIG, "🚀 Production Redis+PG"),
        ("Distributed", DISTRIBUTED_CONFIG, "🌐 Microservices"),
        ("Development", DEVELOPMENT_CONFIG, "🛠️ Local InMemory"),
        ("Testing", TESTING_CONFIG, "🧪 Automated Testing"),
        ("Customer Service", CUSTOMER_SERVICE_CONFIG, "👥 Support Chat"),
        ("Research Assistant", RESEARCH_ASSISTANT_CONFIG, "🔬 Academic Research"),
        ("Creative Assistant", CREATIVE_ASSISTANT_CONFIG, "🎨 Creative Work"),
        ("Enterprise", ENTERPRISE_CONFIG, "🏢 Compliance & Audit"),
    ]
    
    # Header
    print(f"{'Configuration':<20} {'Backend':<12} {'Memory Types':<25} {'Rate Limit':<12} {'Features':<15}")
    print("-" * 80)
    
    # Rows
    for name, config, desc in configs:
        backend = config.memory_backend if config.enable_memory else "None"
        memory_types = ",".join(config.memory_types) if config.enable_memory else "None"
        rate_limit = f"{config.requests_per_second}/s" if config.enable_rate_limiting else "None"
        
        features = []
        if config.enable_memory_tools: features.append("Tools")
        if config.enable_summarization: features.append("AI-Sum")
        if config.enable_semantic_search: features.append("Semantic")
        if config.enable_ttl: features.append("TTL")
        if config.enable_human_feedback: features.append("Human")
        
        features_str = ",".join(features[:3]) + ("..." if len(features) > 3 else "")
        
        print(f"{name:<20} {backend:<12} {memory_types[:24]:<25} {rate_limit:<12} {features_str:<15}")
    
    print()
    print("💡 Usage:")
    print("   from pre_built_configs import HIGH_PERFORMANCE_CONFIG")
    print("   from core_agent import CoreAgent")
    print("   agent = CoreAgent(HIGH_PERFORMANCE_CONFIG)")


if __name__ == "__main__":
    print_config_comparison()