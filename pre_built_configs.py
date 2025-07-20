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
# 🧑‍💻 SOFTWARE DEVELOPMENT WORKFLOW CONFIGS
# =============================================================================

def create_coder_agent_config(
    postgres_url: str = "postgresql://user:pass@localhost:5432/codeagent"
) -> AgentConfig:
    """
    👨‍💻 CODER AGENT CONFIGURATION
    
    Perfect for: Code generation, refactoring, feature implementation, bug fixes
    
    Features:
    - ✅ PostgreSQL for persistent code knowledge and patterns
    - ✅ Large context window for full file analysis
    - ✅ Semantic search for code pattern retrieval
    - ✅ Memory tools for learning coding patterns
    - ✅ No rate limiting for intensive coding sessions
    - ✅ Code-focused evaluation metrics
    
    Use Cases:
    - Feature implementation
    - Code refactoring and optimization
    - Bug fixing and debugging
    - Code pattern recognition and reuse
    - API integration and development
    
    Coding Benefits:
    - Maintains context across large codebases
    - Learns from previous coding patterns
    - Provides consistent code style
    - Handles complex multi-file changes
    
    Args:
        postgres_url: PostgreSQL connection for code knowledge storage
        
    Returns:
        AgentConfig: Configuration optimized for code generation
    """
    return AgentConfig(
        name="CoderAgent",
        
        # Long-term code knowledge storage
        enable_memory=True,
        memory_types=["long_term", "semantic"],
        memory_backend="postgres",
        postgres_url=postgres_url,
        
        # Code project organization
        memory_namespace="code_knowledge",
        
        # Large context for full file analysis
        enable_message_trimming=True,
        max_tokens=32000,  # Large context for code
        trim_strategy="first",  # Keep recent code context
        
        # Semantic search for code patterns
        embedding_model="openai:text-embedding-3-large",
        embedding_dims=3072,
        distance_type="cosine",
        
        # Memory tools for pattern learning
        enable_memory_tools=True,
        memory_namespace_store="code_patterns",
        
        # No rate limiting for intensive coding
        enable_rate_limiting=False,
        
        # No streaming for careful code generation
        enable_streaming=False,
        
        # Code quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["code_quality", "functionality", "maintainability", "performance"]
    )


def create_orchestrator_agent_config(
    redis_url: str = "redis://localhost:6379"
) -> AgentConfig:
    """
    🎼 ORCHESTRATOR AGENT CONFIGURATION
    
    Perfect for: Workflow coordination, agent supervision, task distribution
    
    Features:
    - ✅ Redis for fast inter-agent communication
    - ✅ Supervisor pattern for agent coordination
    - ✅ Session memory for workflow state management
    - ✅ Message trimming for workflow efficiency
    - ✅ Rate limiting for controlled coordination
    - ✅ Streaming for real-time workflow updates
    
    Use Cases:
    - Software development workflow coordination
    - Multi-agent task distribution
    - Build pipeline orchestration
    - Quality assurance coordination
    - Release management supervision
    
    Orchestration Benefits:
    - Coordinates multiple specialized agents
    - Maintains workflow state and progress
    - Handles error recovery and retries
    - Provides real-time workflow monitoring
    
    Args:
        redis_url: Redis connection for agent communication
        
    Returns:
        AgentConfig: Configuration optimized for workflow orchestration
    """
    return AgentConfig(
        name="OrchestratorAgent",
        
        # Fast communication for coordination
        enable_memory=True,
        memory_types=["short_term", "session"],
        memory_backend="redis",
        redis_url=redis_url,
        
        # Workflow state management
        memory_namespace="workflow",
        
        # Efficient context for coordination
        enable_message_trimming=True,
        max_tokens=8000,
        trim_strategy="last",
        
        # Supervisor pattern for agent management
        enable_supervisor=True,
        agents={},  # Will be populated with workflow agents
        
        # Controlled coordination rate
        enable_rate_limiting=True,
        requests_per_second=10.0,  # High for coordination
        max_bucket_size=30.0,
        
        # Real-time workflow updates
        enable_streaming=True,
        
        # Workflow quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["workflow_efficiency", "coordination_quality", "error_handling"]
    )


def create_unit_tester_agent_config() -> AgentConfig:
    """
    🧪 UNIT TESTER AGENT CONFIGURATION
    
    Perfect for: Test generation, test coverage analysis, test optimization
    
    Features:
    - ✅ InMemory for fast test iteration
    - ✅ Large context for analyzing code and existing tests
    - ✅ No rate limiting for rapid test generation
    - ✅ Memory tools for test pattern recognition
    - ✅ Test-focused evaluation metrics
    
    Use Cases:
    - Unit test generation for new code
    - Test coverage gap analysis
    - Test case optimization and refactoring
    - Edge case identification
    - Test data generation
    
    Testing Benefits:
    - Generates comprehensive test suites
    - Identifies missing test scenarios
    - Maintains test quality standards
    - Optimizes test execution speed
    
    Returns:
        AgentConfig: Configuration optimized for unit test generation
    """
    return AgentConfig(
        name="UnitTesterAgent",
        
        # Fast test iteration
        enable_memory=True,
        memory_types=["short_term", "long_term"],
        memory_backend="inmemory",
        
        # Test project organization
        memory_namespace="test_generation",
        
        # Large context for code analysis
        enable_message_trimming=True,
        max_tokens=24000,  # Large for analyzing code + tests
        trim_strategy="first",
        
        # Memory tools for test patterns
        enable_memory_tools=True,
        memory_namespace_store="test_patterns",
        
        # No rate limiting for rapid testing
        enable_rate_limiting=False,
        
        # No streaming for thorough test generation
        enable_streaming=False,
        
        # Test quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["test_coverage", "test_quality", "edge_case_detection", "maintainability"]
    )


def create_executer_agent_config() -> AgentConfig:
    """
    ⚡ EXECUTER AGENT CONFIGURATION
    
    Perfect for: Running tests, executing builds, environment management
    
    Features:
    - ✅ InMemory for fast execution cycles
    - ✅ Session memory for execution state tracking
    - ✅ Streaming for real-time execution output
    - ✅ Message trimming for execution log management
    - ✅ High rate limiting for rapid execution commands
    
    Use Cases:
    - Running unit tests and test suites
    - Executing build scripts and pipelines
    - Environment setup and teardown
    - Performance benchmarking
    - Integration test execution
    
    Execution Benefits:
    - Handles concurrent test execution
    - Manages execution environments
    - Provides real-time execution feedback
    - Tracks execution history and results
    
    Returns:
        AgentConfig: Configuration optimized for code execution
    """
    return AgentConfig(
        name="ExecuterAgent",
        
        # Fast execution tracking
        enable_memory=True,
        memory_types=["short_term", "session"],
        memory_backend="inmemory",
        
        # Execution state tracking
        memory_namespace="execution",
        
        # Execution log management
        enable_message_trimming=True,
        max_tokens=8000,
        trim_strategy="last",  # Keep recent execution logs
        
        # High rate for rapid execution
        enable_rate_limiting=True,
        requests_per_second=20.0,  # High for execution commands
        max_bucket_size=100.0,
        
        # Real-time execution feedback
        enable_streaming=True,
        
        # Execution quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["execution_success", "performance", "reliability", "error_handling"]
    )


def create_code_reviewer_agent_config(
    postgres_url: str = "postgresql://user:pass@localhost:5432/codereview"
) -> AgentConfig:
    """
    🔍 CODE REVIEWER AGENT CONFIGURATION
    
    Perfect for: Code review, quality analysis, security auditing, best practices
    
    Features:
    - ✅ PostgreSQL for persistent review knowledge and patterns
    - ✅ Semantic search for similar code review cases
    - ✅ Large context for comprehensive code analysis
    - ✅ Memory tools for review pattern learning
    - ✅ Quality-focused evaluation metrics
    
    Use Cases:
    - Pull request review automation
    - Code quality assessment
    - Security vulnerability detection
    - Best practice enforcement
    - Architecture review and suggestions
    
    Review Benefits:
    - Provides consistent review quality
    - Identifies security vulnerabilities
    - Enforces coding standards
    - Learns from review feedback
    
    Args:
        postgres_url: PostgreSQL connection for review knowledge storage
        
    Returns:
        AgentConfig: Configuration optimized for code review
    """
    return AgentConfig(
        name="CodeReviewerAgent",
        
        # Persistent review knowledge
        enable_memory=True,
        memory_types=["long_term", "semantic"],
        memory_backend="postgres",
        postgres_url=postgres_url,
        
        # Code review organization
        memory_namespace="code_review",
        
        # Large context for comprehensive review
        enable_message_trimming=True,
        max_tokens=28000,  # Large for reviewing multiple files
        trim_strategy="first",
        
        # Semantic search for similar reviews
        embedding_model="openai:text-embedding-3-large",
        embedding_dims=3072,
        distance_type="cosine",
        
        # Memory tools for review patterns
        enable_memory_tools=True,
        memory_namespace_store="review_patterns",
        
        # Moderate rate limiting for thorough review
        enable_rate_limiting=True,
        requests_per_second=2.0,  # Slower for thorough analysis
        max_bucket_size=8.0,
        
        # No streaming for comprehensive review
        enable_streaming=False,
        
        # Review quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["review_quality", "security_detection", "best_practices", "constructiveness"]
    )


def create_build_agent_config(
    redis_url: str = "redis://localhost:6379"
) -> AgentConfig:
    """
    🏗️ BUILD AGENT CONFIGURATION
    
    Perfect for: Build automation, CI/CD pipeline, versioning, GitHub operations
    
    Features:
    - ✅ Redis for fast build state management
    - ✅ Session memory for build pipeline tracking
    - ✅ TTL for build artifact cleanup
    - ✅ Streaming for real-time build output
    - ✅ Build-focused evaluation metrics
    
    Use Cases:
    - Automated build pipeline execution
    - Version tagging and release management
    - GitHub repository operations
    - Docker image building and publishing
    - Deployment automation
    
    Build Benefits:
    - Manages complex build pipelines
    - Handles version control operations
    - Provides build artifact management
    - Enables continuous integration/deployment
    
    Args:
        redis_url: Redis connection for build state management
        
    Returns:
        AgentConfig: Configuration optimized for build operations
    """
    return AgentConfig(
        name="BuildAgent",
        
        # Build state management
        enable_memory=True,
        memory_types=["short_term", "session"],
        memory_backend="redis",
        redis_url=redis_url,
        
        # Build pipeline organization
        memory_namespace="build_pipeline",
        
        # Build log management
        enable_message_trimming=True,
        max_tokens=12000,
        trim_strategy="last",  # Keep recent build logs
        
        # TTL for build artifact cleanup
        enable_ttl=True,
        default_ttl_minutes=4320,  # 3 days
        refresh_on_read=False,
        
        # Build rate management
        enable_rate_limiting=True,
        requests_per_second=5.0,
        max_bucket_size=20.0,
        
        # Real-time build feedback
        enable_streaming=True,
        
        # Build quality evaluation
        enable_evaluation=True,
        evaluation_metrics=["build_success", "build_time", "artifact_quality", "deployment_readiness"]
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
# 📋 PRE-BUILT CONFIG REGISTRY
# =============================================================================

# Create pre-built configurations for software development workflow
CODER_AGENT_CONFIG = create_coder_agent_config()
UNIT_TESTER_AGENT_CONFIG = create_unit_tester_agent_config()
EXECUTER_AGENT_CONFIG = create_executer_agent_config()
CODE_REVIEWER_AGENT_CONFIG = create_code_reviewer_agent_config()
BUILD_AGENT_CONFIG = create_build_agent_config()
ORCHESTRATOR_AGENT_CONFIG = create_orchestrator_agent_config()
DEVELOPMENT_CONFIG = create_development_config()
TESTING_CONFIG = create_testing_config()

# Registry for easy access
CONFIG_REGISTRY = {
    "coder": CODER_AGENT_CONFIG,
    "unit_tester": UNIT_TESTER_AGENT_CONFIG,
    "executer": EXECUTER_AGENT_CONFIG,
    "code_reviewer": CODE_REVIEWER_AGENT_CONFIG,
    "build": BUILD_AGENT_CONFIG,
    "orchestrator": ORCHESTRATOR_AGENT_CONFIG,
    "development": DEVELOPMENT_CONFIG,
    "testing": TESTING_CONFIG,
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
        "coder": "👨‍💻 Code generation with PostgreSQL knowledge storage and large context",
        "unit_tester": "🧪 Test generation with fast iteration and pattern recognition",
        "executer": "⚡ Test execution with real-time feedback and high throughput",
        "code_reviewer": "🔍 Code review with quality analysis and security auditing",
        "build": "🏗️ Build automation with CI/CD pipeline and version management",
        "orchestrator": "🎼 Workflow coordination with multi-agent supervision",
        "development": "🛠️ Local development with InMemory backend and minimal dependencies",
        "testing": "🧪 Automated testing with deterministic behavior and fast execution",
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
    
    print("🧑‍💻 SOFTWARE DEVELOPMENT WORKFLOW CONFIGURATIONS")
    print("=" * 80)
    print()
    
    configs = [
        ("Coder Agent", CODER_AGENT_CONFIG, "👨‍💻 Code Generation"),
        ("Unit Tester", UNIT_TESTER_AGENT_CONFIG, "🧪 Test Generation"),
        ("Executer", EXECUTER_AGENT_CONFIG, "⚡ Test Execution"),
        ("Code Reviewer", CODE_REVIEWER_AGENT_CONFIG, "🔍 Code Review"),
        ("Build Agent", BUILD_AGENT_CONFIG, "🏗️ Build & Deploy"),
        ("Orchestrator", ORCHESTRATOR_AGENT_CONFIG, "🎼 Workflow Control"),
        ("Development", DEVELOPMENT_CONFIG, "🛠️ Local Development"),
        ("Testing", TESTING_CONFIG, "🧪 Automated Testing"),
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
    print("   from pre_built_configs import CODER_AGENT_CONFIG")
    print("   from core_agent import CoreAgent")
    print("   agent = CoreAgent(CODER_AGENT_CONFIG)")


if __name__ == "__main__":
    print_config_comparison()